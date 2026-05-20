import pandas as pd
import numpy as np

LABEL_COLS = ["Sweet", "Bitter", "Umami", "Sour", "Undefined"]


def normalize_label(label: str) -> str:
    """Map raw labels to canonical 5-class scheme."""
    if pd.isna(label):
        return "Undefined"
    label_str = str(label).strip().lower()
    
    # Standard matches
    if label_str in ["sweet", "sweetness"]:
        return "Sweet"
    if label_str in ["bitter", "bitterness"]:
        return "Bitter"
    if label_str in ["umami", "umaminess"]:
        return "Umami"
    if label_str in ["sour", "sourness"]:
        return "Sour"
    
    # Virtuous: "Other" → Undefined
    if label_str == "other":
        return "Undefined"
    
    # ChemTastesDB: Non-sweet, Tasteless, Multitaste, etc. → Undefined
    if any(x in label_str for x in ["non-", "non ", "tasteless", "multitaste"]):
        return "Undefined"
    
    # Default: unmapped label → Undefined
    return "Undefined"


def to_one_hot_labels(labels: list[str]) -> pd.DataFrame:
    """Convert label strings to one-hot encoding."""
    data = {col: [1 if label == col else 0 for label in labels] for col in LABEL_COLS}
    return pd.DataFrame(data)


def stratified_split(
    df: pd.DataFrame, label_col: str, test_frac: float = 0.25, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by label."""
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[label_col],
        random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
