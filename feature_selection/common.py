import argparse
import copy
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


LABEL_COLS = ["Sweet", "Bitter", "Umami", "Sour", "Undefined"]
NUM_CLASSES = len(LABEL_COLS)

EMBEDDINGS = {
    "RDKit": "rdkit_descriptors.csv",
    "Morgan FP": "morgan_fps.csv",
    "MACCS": "maccs.csv",
    "Mol2Vec": "mol2vec.csv",
    "ChemBERTa": "chemberta.csv",
}


@dataclass
class DeepConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 40
    patience: int = 8
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def load_embedding(emb_dir: str, name: str):
    path = os.path.join(emb_dir, EMBEDDINGS[name])
    df = pd.read_csv(path)
    labels_df = df[LABEL_COLS]
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    x = df[feature_cols].values
    y = labels_df.values.argmax(axis=1)
    return x, y, feature_cols


def macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec_i)
    return float(np.mean(specificities))


def compute_metrics(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "SPEC": macro_specificity(y_true, y_pred),
        "SENS": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def get_sklearn_models(seed: int):
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed, n_jobs=-1),
            True,
        ),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed), False),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=seed,
            ),
            False,
        ),
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=seed),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                random_state=seed,
                n_jobs=-1,
                eval_metric="mlogloss",
                use_label_encoder=False,
            ),
            False,
        ),
        "SVM": (SVC(kernel="rbf", random_state=seed, decision_function_shape="ovr"), True),
    }


class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BiRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3, rnn_type="GRU"):
        super().__init__()
        rnn = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = rnn_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def train_deep_model(model, train_loader, val_x, val_y, criterion, optimizer, device, config: DeepConfig):
    best_f1, best_state, no_improve = 0.0, None, 0
    for _ in range(config.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_f1 = f1_score(val_y, val_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


def predict_in_batches(model, x_np, device, batch_size=256):
    ds = SeqDataset(x_np, np.zeros(len(x_np), dtype=int))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in dl:
            x_batch = x_batch.to(device)
            preds.append(torch.argmax(model(x_batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def run_model_suite(x, y, embedding_name, selected_feature_count, seed=42, test_size=0.2, deep_config: DeepConfig = None):
    if deep_config is None:
        deep_config = DeepConfig()

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    rows = []

    for model_name, (model, needs_scaling) in get_sklearn_models(seed).items():
        if needs_scaling:
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_train)
            x_te = scaler.transform(x_test)
        else:
            x_tr, x_te = x_train, x_test

        model.fit(x_tr, y_train)
        y_train_pred = model.predict(x_tr)
        y_test_pred = model.predict(x_te)

        row = {
            "Model": model_name,
            "Embedding": embedding_name,
            "Selected_Features": selected_feature_count,
        }
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        for k, v in train_metrics.items():
            row[f"Train_{k}"] = round(v, 4)
        for k, v in test_metrics.items():
            row[f"Test_{k}"] = round(v, 4)
        rows.append(row)

    deep_scaler = StandardScaler()
    x_tr_scaled = deep_scaler.fit_transform(x_train)
    x_te_scaled = deep_scaler.transform(x_test)
    x_tr2, x_val, y_tr2, y_val = train_test_split(
        x_tr_scaled, y_train, test_size=0.15, random_state=seed, stratify=y_train
    )
    train_ds = SeqDataset(x_tr2, y_tr2)
    train_loader = DataLoader(train_ds, batch_size=deep_config.batch_size, shuffle=True)
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)

    counts = np.bincount(y_tr2, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0.0] = 1.0
    class_weights = torch.tensor(
        np.sqrt(len(y_tr2) / (NUM_CLASSES * counts)), dtype=torch.float32
    ).to(device)

    for rnn_type in ["GRU", "LSTM"]:
        model_name = f"Bi{rnn_type}"
        model = BiRNNClassifier(
            input_dim=x.shape[1],
            hidden_dim=deep_config.hidden_dim,
            num_layers=deep_config.num_layers,
            num_classes=NUM_CLASSES,
            dropout=deep_config.dropout,
            rnn_type=rnn_type,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=deep_config.lr, weight_decay=1e-4)

        train_deep_model(
            model,
            train_loader,
            val_x_tensor,
            y_val,
            criterion,
            optimizer,
            device,
            deep_config,
        )

        y_train_pred = predict_in_batches(model, x_tr_scaled, device)
        y_test_pred = predict_in_batches(model, x_te_scaled, device)

        row = {
            "Model": model_name,
            "Embedding": embedding_name,
            "Selected_Features": selected_feature_count,
        }
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        for k, v in train_metrics.items():
            row[f"Train_{k}"] = round(v, 4)
        for k, v in test_metrics.items():
            row[f"Test_{k}"] = round(v, 4)
        rows.append(row)

        del model, optimizer, criterion
        clear_device_cache(device)

    return rows


def parse_common_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--emb-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "final_embeddings"),
        help="Directory containing embedding CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default is set per method script.",
    )
    return parser
