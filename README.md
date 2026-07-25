# TasteIPS

A cheminformatics ML project for predicting taste category (Sweet, Bitter, Umami, Sour, Undefined) from molecular SMILES strings. Built as a research workspace — notebooks record experiments, scripts reproduce the pipeline, CSVs preserve results.

## What it does

Five-class classification pipeline:
- Converts SMILES to RDKit 2D descriptors, MACCS keys (167 bits), and Morgan fingerprints (2048 bits, radius 2). Mol2Vec and ChemBERTa were explored in notebooks but aren't in the main pipeline.
- Runs six feature selection strategies (L1, RFE, mutual information, RF importance, correlation filter, Boruta) and compares ten classifiers including BiGRU and BiLSTM.
- Builds a stacked ensemble (RF + XGBoost + Extra Trees + SVM as base models, logistic regression as meta-model) on the L1-selected 352-feature set using out-of-fold probabilities to avoid leakage.
- Evaluates on four external taste databases (FartDB, VirtuousMultiTaste, UMP442, ChemTastesDB).

Best results: **0.8963 balanced accuracy, 0.8632 F1, 0.9790 AUC** on the held-out test split.

## Repo layout

```
baseline_model.ipynb              # representation + model comparison
feature_selection/                # one script per selector
stacked_l1_ensemble.ipynb         # ensemble training on L1 features
stacked_l1_combo1_onevsrest_metrics.ipynb
feature_extraction.py             # RDKit/MACCS/Morgan generation
run_benchmarking.py               # external dataset evaluation
fs_results/                       # selection CSVs + saved model
```

## Setup

```bash
conda create -n tasteips python=3.11 -y
conda activate tasteips
conda install -c conda-forge rdkit -y
pip install numpy pandas scikit-learn xgboost torch joblib openpyxl jupyter boruta
```

## Running

Run notebooks in order once embeddings are available:

```bash
jupyter lab
# 1. baseline_model.ipynb
# 2. feature_selection/<selector>.py --emb-dir ../combined_embeddings
# 3. stacked_l1_ensemble.ipynb
# 4. stacked_l1_combo1_onevsrest_metrics.ipynb
```

External benchmark:

```bash
python run_benchmarking.py --out-dir fs_results/benchmarking --seed 42
```

## Caveats

- Source datasets, embedding directories, and the serialized model aren't tracked — regenerate embeddings before running notebooks.
- Preprocessing (imputation/scaling) is fitted on evaluation data, not reused from training — not production-grade.
- These are experimental results, not a validated taste predictor.
