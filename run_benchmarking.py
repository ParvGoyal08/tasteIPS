from __future__ import annotations

# Allow running as a script: `python benchmarking_scripts/run_benchmarking.py`
# by ensuring the repository root is on sys.path.
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from benchmarking_scripts.benchmark_common import LABEL_COLS, normalize_label, stratified_split, to_one_hot_labels
from benchmarking_scripts.feature_extraction import (
    compute_maccs_fingerprints,
    compute_morgan_fingerprints,
    compute_rdkit_descriptors,
    preprocess_continuous_features,
)


def macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # In our one-vs-rest setting, y_true/y_pred are binary vectors.
    # Force a 2x2 confusion matrix even if only one label is present
    # (e.g., all-positives subset with all-positives predictions).
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    n_classes = cm.shape[0]
    specificities: list[float] = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denom = tn + fp
        spec_i = float(tn / denom) if denom > 0 else 0.0
        specificities.append(spec_i)
    return float(np.mean(specificities))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Metrics are computed for binary one-vs-rest vectors.
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom) if denom > 0 else 0.0

    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    bacc = float(np.mean(recalls))
    sens = float(np.mean(recalls))
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "BACC": float(bacc),
        "PRE": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "SPEC": float(macro_specificity(y_true, y_pred)),
        "SENS": float(sens),
        "F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "MCC": float(mcc),
    }


def load_selected_feature_names(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "Selected_Feature_Names" not in df.columns:
        raise ValueError(f"Expected column 'Selected_Feature_Names' in {path}")
    raw = str(df.loc[0, "Selected_Feature_Names"])
    feats = [f.strip() for f in raw.split("|") if f.strip()]
    if not feats:
        raise ValueError("No selected features parsed")
    return feats


def build_canonical_df_fart(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles = df["Canonicalized SMILES"].astype(str)
    labels = df["Canonicalized Taste"].map(normalize_label)
    return pd.DataFrame({"smiles": smiles, "label": labels, "label_raw": df["Canonicalized Taste"]})


def build_canonical_df_virtuous(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles = df["Std_SMILES"].astype(str)
    labels = df["Taste"].map(normalize_label)
    return pd.DataFrame({"smiles": smiles, "label": labels, "label_raw": df["Taste"]})


def build_canonical_df_chemtaste(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="ChemTastesDB")
    smiles = df["canonical SMILES"].astype(str)
    # Use the more standardized category column
    labels = df["Class taste"].map(normalize_label)
    return pd.DataFrame({"smiles": smiles, "label": labels, "label_raw": df["Class taste"]})


def build_canonical_df_ump442(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Std_SMILES" in df.columns:
        smiles = df["Std_SMILES"].astype(str)
    elif "Parent_SMILES" in df.columns:
        smiles = df["Parent_SMILES"].astype(str)
    else:
        raise ValueError("UMP442: expected 'Std_SMILES' or 'Parent_SMILES' column")

    if "TASTE" not in df.columns:
        raise ValueError("UMP442: expected 'TASTE' column")
    labels = df["TASTE"].map(normalize_label)
    return pd.DataFrame({"smiles": smiles, "label": labels, "label_raw": df["TASTE"]})


def featurize(df: pd.DataFrame, selected_feature_names: list[str]) -> pd.DataFrame:
    smiles_list = df["smiles"].astype(str).tolist()

    print("    - RDKit descriptors...")
    rdkit_df = compute_rdkit_descriptors(smiles_list)
    rdkit_cols = [c for c in selected_feature_names if c in rdkit_df.columns]
    rdkit_df = rdkit_df[rdkit_cols]
    if rdkit_cols:
        rdkit_df = preprocess_continuous_features(rdkit_df, rdkit_cols)

    print("    - MACCS...")
    maccs_df = compute_maccs_fingerprints(smiles_list)
    maccs_cols = [c for c in selected_feature_names if c.startswith("maccs_")]
    maccs_df = maccs_df.reindex(columns=maccs_cols, fill_value=0).astype(int)

    print("    - Morgan FP...")
    fp_df = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048)
    fp_cols = [c for c in selected_feature_names if c.startswith("fp_bit_")]
    fp_df = fp_df.reindex(columns=fp_cols, fill_value=0).astype(int)

    X = pd.concat([maccs_df, rdkit_df, fp_df], axis=1)

    # Ensure full schema present and ordered
    X = X.reindex(columns=selected_feature_names, fill_value=0)
    # Fill any remaining NaNs with 0 (e.g., from failed RDKit descriptor calculations)
    X = X.fillna(0)
    return X


def predict_with_bundle(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    base_models = bundle["fitted_base_models"]
    meta_estimator = bundle["meta_estimator"]

    probs = []
    for name, model in base_models:
        p = model.predict_proba(X)
        probs.append(p)

    Z = np.hstack(probs)
    y_proba = meta_estimator.predict_proba(Z)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def one_vs_rest_metrics(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> pd.DataFrame:
    """Compute one-vs-rest metrics for classes present in test set (SupportPos > 0)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    rows = []
    for class_idx, class_name in enumerate(LABEL_COLS):
        y_true_bin = (y_true_idx == class_idx).astype(int)
        y_pred_bin = (y_pred_idx == class_idx).astype(int)
        
        support = int(y_true_bin.sum())
        # Only report metrics for classes present in test set
        if support == 0:
            continue
        
        m = compute_metrics(y_true_bin, y_pred_bin)
        rows.append({"Class": class_name, "SupportPos": support, **m})
    
    # Add overall multiclass metrics
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    macro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    
    rows.append({
        "Class": "OVERALL_MULTICLASS",
        "SupportPos": len(y_true_idx),
        "ACC": acc,
        "BACC": acc,
        "PRE": macro_prec,
        "SPEC": 0.0,  # Not applicable for multiclass
        "SENS": macro_recall,
        "F1": weighted_f1,
        "MCC": float(macro_f1),
    })
    
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chemtaste-test-frac", type=float, default=0.25)
    ap.add_argument("--out-dir", type=str, default="fs_results/benchmarking")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    test_dir = out_dir / "test_sets"
    feat_dir = out_dir / "features"
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    for d in [test_dir, feat_dir, pred_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    selected_feature_names = load_selected_feature_names(Path("fs_results/categorical_feature_selection_L1_352.csv"))
    bundle = joblib.load("fs_results/stacked_model random forest + xgboost + extra tress + SVM meta_LR.joblib")
    
    print(f"✓ Loaded {len(selected_feature_names)} selected features")
    print(f"✓ Loaded stacked model with {len(bundle['fitted_base_models'])} base models")

    datasets: list[tuple[str, pd.DataFrame]] = []
    datasets.append(("FART", build_canonical_df_fart(Path("benchmarking/fartdb_test.csv"))))
    datasets.append(("VirtuousMultiTaste", build_canonical_df_virtuous(Path("benchmarking/VirtuousMultiTaste.csv"))))
    datasets.append(("UMP442", build_canonical_df_ump442(Path("benchmarking/UMP442.csv"))))

    chemdf = build_canonical_df_chemtaste(Path("benchmarking/ChemTastesDB_database.xlsx"))
    _, chem_test = stratified_split(chemdf, "label", test_frac=args.chemtaste_test_frac, seed=args.seed)
    datasets.append(("ChemTastesDB", chem_test))

    summary_rows = []
    full_dataset_summary_rows = []

    for ds_name, df in datasets:
        print(f"\n==================== {ds_name} ====================")
        df = df.dropna(subset=["smiles"]).copy()
        df["label"] = df["label"].map(normalize_label)

        # === FULL DATASET EVALUATION (for proper multiclass metrics) ===
        print(f"Full dataset evaluation: {len(df)} total rows")
        X_full = featurize(df, selected_feature_names)
        
        # Diagnostic: Check feature quality
        nan_count = X_full.isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  WARNING: {nan_count} NaN values in features (filled with 0)")
        print(f"  ✓ Features shape: {X_full.shape}, dtypes: {X_full.dtypes.unique()}")
        
        y_onehot_full = to_one_hot_labels(df["label"].tolist())
        y_true_idx_full = y_onehot_full.to_numpy().argmax(axis=1)
        y_pred_idx_full, y_proba_full = predict_with_bundle(bundle, X_full)
        
        # Diagnostic: Check prediction quality
        unique_preds = len(np.unique(y_pred_idx_full))
        pred_conf = np.max(y_proba_full, axis=1).mean()
        print(f"  ✓ Predictions: {unique_preds} classes predicted, avg confidence: {pred_conf:.3f}")

        # Save full-dataset predictions
        pred_df_full = pd.DataFrame({
            "smiles": df["smiles"].tolist(),
            "true_label": df["label"].tolist(),
            "pred_label": [LABEL_COLS[i] for i in y_pred_idx_full],
        })
        for i, c in enumerate(LABEL_COLS):
            pred_df_full[f"proba_{c}"] = y_proba_full[:, i]
        pred_path_full = pred_dir / f"{ds_name}__full_dataset__predictions.csv"
        pred_df_full.to_csv(pred_path_full, index=False)

        # Full-dataset metrics (one-vs-rest per class)
        mdf_full = one_vs_rest_metrics(y_true_idx_full, y_pred_idx_full)
        mdf_full.insert(0, "Dataset", ds_name)
        mdf_full.insert(1, "TestClassSubset", "full_dataset")
        metrics_path_full = metrics_dir / f"{ds_name}__full_dataset__onevsrest_metrics.csv"
        mdf_full.to_csv(metrics_path_full, index=False)

        # Add full-dataset metrics to summary
        for _, row in mdf_full.iterrows():
            full_dataset_summary_rows.append(row.to_dict())

        # === PER-CLASS SUBSET EVALUATION (for reference) ===
        # Build per-class test sets
        for test_class in LABEL_COLS:
            cls_df = df[df["label"] == test_class].copy()
            if len(cls_df) == 0:
                continue

            test_path = test_dir / f"{ds_name}__{test_class}.csv"
            cls_df.to_csv(test_path, index=False)

            print(f"  -> {ds_name} / {test_class}: {len(cls_df)} rows")
            X = featurize(cls_df, selected_feature_names)

            # Save feature file with one-hot labels + features
            y_onehot = to_one_hot_labels(cls_df["label"].tolist())
            feat_out = pd.concat([y_onehot, X], axis=1)
            feat_path = feat_dir / f"{ds_name}__{test_class}__selected_features.csv"
            feat_out.to_csv(feat_path, index=False)

            y_true_idx = y_onehot.to_numpy().argmax(axis=1)
            y_pred_idx, y_proba = predict_with_bundle(bundle, X)

            # Save predictions
            pred_df = pd.DataFrame({
                "smiles": cls_df["smiles"].tolist(),
                "true_label": cls_df["label"].tolist(),
                "pred_label": [LABEL_COLS[i] for i in y_pred_idx],
            })
            for i, c in enumerate(LABEL_COLS):
                pred_df[f"proba_{c}"] = y_proba[:, i]

            pred_path = pred_dir / f"{ds_name}__{test_class}__predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            # Metrics
            mdf = one_vs_rest_metrics(y_true_idx, y_pred_idx)
            mdf.insert(0, "Dataset", ds_name)
            mdf.insert(1, "TestClassSubset", test_class)
            metrics_path = metrics_dir / f"{ds_name}__{test_class}__onevsrest_metrics.csv"
            mdf.to_csv(metrics_path, index=False)

            # summary for the subset's positive class
            pos_row = mdf[mdf["Class"] == test_class].iloc[0].to_dict()
            summary_rows.append(pos_row)

    # Write summaries
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "metrics_summary_posclass.csv", index=False)
    print(f"\n✅ Wrote per-class subset summary → {out_dir / 'metrics_summary_posclass.csv'}")

    summary_full = pd.DataFrame(full_dataset_summary_rows)
    summary_full.to_csv(out_dir / "metrics_summary_full_dataset.csv", index=False)
    print(f"✅ Wrote full-dataset summary → {out_dir / 'metrics_summary_full_dataset.csv'}")


if __name__ == "__main__":
    main()
