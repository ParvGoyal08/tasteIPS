import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_l1(x_train, y_train, c=0.1, min_features=10, seed=42):
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=c,
        max_iter=4000,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    coefs = np.abs(model.coef_)
    importance = np.max(coefs, axis=0)
    mask = importance > 0.0

    if mask.sum() < min_features:
        top_idx = np.argsort(importance)[-min_features:]
        mask = np.zeros_like(importance, dtype=bool)
        mask[top_idx] = True

    return mask


def main():
    parser = parse_common_args("L1 selection + baseline model suite")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--min-features", type=int, default=10)
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(__file__),
        "l1_selection_results.csv"
    )

    all_rows = []

    for emb_idx, emb_name in enumerate(EMBEDDINGS, 1):
        print(f"\n🚀 [{emb_idx}/{len(EMBEDDINGS)}] Processing embedding: {emb_name}", flush=True)

        x, y, feature_names = load_embedding(args.emb_dir, emb_name)

        print("   🔍 Running L1 feature selection...", flush=True)
        mask = select_features_l1(
            x, y,
            c=args.c,
            min_features=args.min_features,
            seed=args.seed
        )

        x_selected = x[:, mask]
        print(f"   ✅ Selected {int(mask.sum())} features", flush=True)

        print("   🤖 Running model suite...", flush=True)
        rows = run_model_suite(
            x_selected,
            y,
            embedding_name=emb_name,
            selected_feature_count=int(mask.sum()),
            seed=args.seed,
            test_size=args.test_size,
        )

        selected_names = [
            feature_names[i] for i, keep in enumerate(mask) if keep
        ]

        for r in rows:
            r["Selector"] = "L1"
            r["Selected_Feature_Names"] = "|".join(selected_names)

        all_rows.extend(rows)

        # ✅ SAVE AFTER EACH EMBEDDING
        pd.DataFrame(all_rows).to_csv(out_path, index=False)

        print(f"   💾 Saved progress after {emb_name} → {out_path}", flush=True)

    print(f"\n✅ All done. Final results saved to {out_path}")


if __name__ == "__main__":
    main()