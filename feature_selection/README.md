# Feature Selection Experiments

This folder contains separate scripts for each feature-selection method, using the same model and embedding combinations from the baseline notebook:

- Models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, SVM, BiGRU, BiLSTM
- Embeddings: RDKit, Morgan FP, MACCS, Mol2Vec, ChemBERTa

## Scripts

- `random_forest_selection.py`
- `boruta_selection.py`
- `rfe_selection.py`
- `correlation_filter_selection.py`
- `l1_selection.py`
- `mutual_information_selection.py`

## Common Run Pattern

Run from `Multitaste-Model/feature_selection`:

```bash
python random_forest_selection.py
python boruta_selection.py
python rfe_selection.py
python correlation_filter_selection.py
python l1_selection.py
python mutual_information_selection.py
```

Each script writes one CSV result file in this folder.

## Optional Arguments

All scripts support:

- `--emb-dir` path to embedding files (default: `../final_embeddings`)
- `--seed` random seed (default: `42`)
- `--test-size` test split fraction (default: `0.2`)
- `--output` output CSV path


## Boruta Dependency

Boruta script requires:

```bash
pip install boruta
```
