import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys


def compute_rdkit_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute RDKit 2D descriptors for a list of SMILES strings."""
    descs = Descriptors.descList
    data = {}
    for desc_name, desc_fn in descs:
        data[desc_name] = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            for desc_name in data.keys():
                data[desc_name].append(np.nan)
        else:
            for desc_name, desc_fn in descs:
                try:
                    val = desc_fn(mol)
                    data[desc_name].append(float(val))
                except:
                    data[desc_name].append(np.nan)
    
    return pd.DataFrame(data)


def compute_morgan_fingerprints(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> pd.DataFrame:
    """Compute Morgan fingerprints (bit vector) for a list of SMILES."""
    data = {}
    for i in range(n_bits):
        data[f"fp_bit_{i}"] = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            for i in range(n_bits):
                data[f"fp_bit_{i}"].append(0)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            for i in range(n_bits):
                data[f"fp_bit_{i}"].append(int(fp[i]))
    
    return pd.DataFrame(data)


def compute_maccs_fingerprints(smiles_list: list[str]) -> pd.DataFrame:
    """Compute MACCS keys (167 bits) for a list of SMILES."""
    data = {}
    for i in range(167):
        data[f"maccs_{i}"] = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            for i in range(167):
                data[f"maccs_{i}"].append(0)
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            for i in range(167):
                data[f"maccs_{i}"].append(int(fp[i]))
    
    return pd.DataFrame(data)


def preprocess_continuous_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Median imputation, clipping, and z-score normalization for continuous features."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        # Median imputation
        median = df[col].median()
        df[col] = df[col].fillna(median)
        # Clip outliers (±3 sigma)
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            lower = mean - 3 * std
            upper = mean + 3 * std
            df[col] = df[col].clip(lower, upper)
            # Z-score normalization
            df[col] = (df[col] - mean) / std
    return df