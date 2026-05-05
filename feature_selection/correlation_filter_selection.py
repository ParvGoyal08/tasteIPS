import os

import numpy as np
import pandas as pd

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_correlation(x_train, threshold=0.95, min_features=10):
    # Compute absolute feature-feature correlation using pandas (more stable)
    import pandas as pd

    X = pd.DataFrame(x_train).astype("float32")
    corr = X.corr().abs().fillna(0).values
    n = corr.shape[0]
    keep = np.ones(n, dtype=bool)

    # Greedy keep: prefer features with lower average correlation
    avg_corr = corr.mean(axis=0)
    order = np.argsort(avg_corr)  # ascending: lower avg corr first

    for idx in order:
        if not keep[idx]:
            continue
        # mark all features highly correlated with idx for removal
        correlated = np.where((corr[idx] >= threshold) & (np.arange(n) != idx))[0]
        keep[correlated] = False
        if keep.sum() < min_features:
            # fallback: ensure at least min_features are kept (choose lowest avg_corr)
            keep[:] = False
            top_idx = np.argsort(avg_corr)[:min_features]
            keep[top_idx] = True
            break

    if keep.sum() == 0:
        # final fallback: keep top-variance features
        var = X.var(axis=0).values
        top_idx = np.argsort(var)[-min_features:]
        keep[top_idx] = True

    return keep


def main():
    parser = parse_common_args("Correlation-filter selection + baseline model suite")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-features", type=int, default=10)
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(__file__),
        "correlation_filter_selection_results.csv",
    )

    all_rows = []
    for emb_name in EMBEDDINGS:
        x, y, feature_names = load_embedding(args.emb_dir, emb_name)
        mask = select_features_correlation(x, threshold=args.threshold, min_features=args.min_features)
        x_selected = x[:, mask]

        rows = run_model_suite(
            x_selected,
            y,
            embedding_name=emb_name,
            selected_feature_count=int(mask.sum()),
            seed=args.seed,
            test_size=args.test_size,
        )

        selected_names = [feature_names[i] for i, keep in enumerate(mask) if keep]
        for r in rows:
            r["Selector"] = "CorrelationFilter"
            r["Selected_Feature_Names"] = "|".join(selected_names)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
