import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_rfe(x_train, y_train, feature_fraction=0.4, min_features=10, seed=42):
    n_features = x_train.shape[1]
    n_select = max(min_features, int(n_features * feature_fraction))
    n_select = min(n_select, n_features)

    base = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
        n_jobs=-1,
    )
    selector = RFE(estimator=base, n_features_to_select=n_select, step=0.1)
    selector.fit(x_train, y_train)
    return selector.support_


def main():
    parser = parse_common_args("RFE feature selection + baseline model suite")
    parser.add_argument("--feature-fraction", type=float, default=0.4)
    parser.add_argument("--min-features", type=int, default=10)
    args = parser.parse_args()

    out_path = args.output or os.path.join(os.path.dirname(__file__), "rfe_selection_results.csv")

    all_rows = []
    for emb_name in EMBEDDINGS:
        x, y, feature_names = load_embedding(args.emb_dir, emb_name)
        mask = select_features_rfe(
            x,
            y,
            feature_fraction=args.feature_fraction,
            min_features=args.min_features,
            seed=args.seed,
        )
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
            r["Selector"] = "RFE"
            r["Selected_Feature_Names"] = "|".join(selected_names)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
