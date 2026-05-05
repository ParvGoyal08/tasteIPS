import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_boruta(x_train, y_train, seed=42, max_iter=75):
    try:
        from boruta import BorutaPy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Boruta is not installed. Install with: pip install boruta"
        ) from exc

    estimator = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    selector = BorutaPy(estimator=estimator, n_estimators="auto", random_state=seed, max_iter=max_iter)
    selector.fit(x_train, y_train)

    mask = selector.support_.copy()
    if mask.sum() == 0:
        mask = selector.support_weak_.copy()
    if mask.sum() == 0:
        importances = estimator.fit(x_train, y_train).feature_importances_
        top_idx = np.argsort(importances)[-10:]
        mask = np.zeros_like(importances, dtype=bool)
        mask[top_idx] = True
    return mask


def main():
    parser = parse_common_args("Boruta feature selection + baseline model suite")
    parser.add_argument("--max-iter", type=int, default=75)
    args = parser.parse_args()

    out_path = args.output or os.path.join(os.path.dirname(__file__), "boruta_selection_results.csv")

    all_rows = []
    for emb_name in EMBEDDINGS:
        x, y, feature_names = load_embedding(args.emb_dir, emb_name)
        mask = select_features_boruta(x, y, seed=args.seed, max_iter=args.max_iter)
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
            r["Selector"] = "Boruta"
            r["Selected_Feature_Names"] = "|".join(selected_names)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
