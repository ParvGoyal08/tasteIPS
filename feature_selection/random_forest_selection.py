import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_rf(x_train, y_train, min_features=10, seed=42):
    model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)
    importances = model.feature_importances_
    threshold = float(np.mean(importances))
    mask = importances >= threshold

    if mask.sum() < min_features:
        top_idx = np.argsort(importances)[-min_features:]
        mask = np.zeros_like(importances, dtype=bool)
        mask[top_idx] = True

    return mask


def main():
    parser = parse_common_args("Random-Forest feature selection + baseline model suite")
    parser.add_argument("--min-features", type=int, default=10)
    args = parser.parse_args()

    out_path = args.output or os.path.join(os.path.dirname(__file__), "random_forest_selection_results.csv")

    all_rows = []
    for emb_name in EMBEDDINGS:
        x, y, feature_names = load_embedding(args.emb_dir, emb_name)
        mask = select_features_rf(x, y, min_features=args.min_features, seed=args.seed)
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
            r["Selector"] = "RandomForestImportance"
            r["Selected_Feature_Names"] = "|".join(selected_names)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
