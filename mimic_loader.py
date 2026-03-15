"""
mimic_loader.py
Loads MIMIC-IV extract and converts to discrete BN-ready format.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def discretize_continuous(series: pd.Series, n_bins: int = 3, strategy: str = "quantile") -> pd.Series:
    """
    Discretize a continuous variable into n_bins categories.
    
    Parameters
    ----------
    series : continuous data
    n_bins : number of bins
    strategy : 'quantile' (equal frequency) or 'uniform' (equal width)
    
    Returns
    -------
    Discretized series as integers 0 to n_bins-1
    """
    if strategy == "quantile":
        try:
            bins = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            # If duplicates='drop' still fails, fall back to uniform
            bins = pd.cut(series, bins=n_bins, labels=False, duplicates="drop")
    else:
        bins = pd.cut(series, bins=n_bins, labels=False, duplicates="drop")
    
    # Handle missing: map NaN to 0
    bins = bins.fillna(0).astype(int)
    return bins


def consolidate_categorical(series: pd.Series, top_k: int = 5, other_label: Optional[int] = None) -> pd.Series:
    """
    Keep top K most frequent categories, group rest into 'Other'.
    
    Parameters
    ----------
    series : categorical data
    top_k : keep this many top categories
    other_label : value for 'Other' category (default: max+1)
    
    Returns
    -------
    Consolidated series as integers
    """
    # Handle missing first
    series = series.fillna(-999)  # temporary marker
    
    counts = series.value_counts()
    top_cats = counts.head(top_k).index.tolist()
    
    if other_label is None:
        if pd.api.types.is_numeric_dtype(series):
            other_label = int(series.max()) + 1 if series.max() >= 0 else top_k
        else:
            other_label = top_k
    
    series_out = series.copy()
    series_out.loc[~series_out.isin(top_cats)] = other_label
    
    # Map to 0..k-1 range
    unique_vals = sorted(series_out.unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    series_out = series_out.map(mapping)
    
    return series_out.astype(int)


def load_mimic_extract(
    csv_path: str,
    variable_config: Dict[str, Dict],
    fixed_dag_edges: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], Dict[str, int]]:
    """
    Load MIMIC extract CSV and discretize according to variable_config.
    
    Parameters
    ----------
    csv_path : path to your MIMIC extract CSV (one row per ICU stay or patient)
    variable_config : dict mapping column name → discretization config
        Example:
        {
            "age": {"type": "continuous", "bins": 4},
            "gender": {"type": "categorical", "top_k": 2},
            "lab_wbc": {"type": "continuous", "bins": 3},
        }
    fixed_dag_edges : list of (parent, child) tuples defining your BN structure
    
    Returns
    -------
    df_discrete : DataFrame with all columns as integers 0..k-1
    edges : same as fixed_dag_edges (pass-through)
    cardinalities : dict mapping node → cardinality (number of states)
    """
    print(f"[mimic_loader] Loading {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    print(f"[mimic_loader] Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # Select only columns mentioned in variable_config
    missing_cols = [c for c in variable_config.keys() if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")
    
    df = df_raw[list(variable_config.keys())].copy()
    
    # Discretize each column
    cardinalities = {}
    for col, cfg in variable_config.items():
        print(f"[mimic_loader] Processing {col} ({cfg['type']})...")
        
        if cfg["type"] == "continuous":
            df[col] = discretize_continuous(df[col], n_bins=cfg.get("bins", 3))
            cardinalities[col] = int(df[col].max() + 1)
        
        elif cfg["type"] == "categorical":
            df[col] = consolidate_categorical(df[col], top_k=cfg.get("top_k", 5))
            cardinalities[col] = int(df[col].max() + 1)
        
        else:
            raise ValueError(f"Unknown type '{cfg['type']}' for column '{col}'")
        
        print(f"[mimic_loader]   → {col}: cardinality = {cardinalities[col]}")
    
    # Ensure all are int and no NaNs remain
    df = df.astype(int)
    
    if df.isnull().any().any():
        raise ValueError("NaN values remain after discretization!")
    
    print(f"[mimic_loader] Final discrete dataset: {len(df)} rows, {len(df.columns)} variables")
    print(f"[mimic_loader] Cardinalities: {cardinalities}")
    
    return df, fixed_dag_edges, cardinalities


def get_example_mimic_config():
    """
    Example configuration for a simple MIMIC-IV ICU cohort.
    Adapt this to your actual variables and clinical DAG.
    """
    
    # Example DAG: simple sepsis prediction network
    dag_edges = [
        ("age", "sepsis"),
        ("gender", "sepsis"),
        ("admission_type", "sepsis"),
        ("heart_rate", "sepsis"),
        ("temperature", "sepsis"),
        ("wbc", "sepsis"),
        ("lactate", "sepsis"),
    ]
    
    # Example variable discretization
    variable_config = {
        "age": {"type": "continuous", "bins": 4},           # 0=young, 1=middle, 2=senior, 3=elderly
        "gender": {"type": "categorical", "top_k": 2},      # 0=M, 1=F (or vice versa)
        "admission_type": {"type": "categorical", "top_k": 3},  # Emergency, Elective, etc.
        "heart_rate": {"type": "continuous", "bins": 3},    # Low/Normal/High
        "temperature": {"type": "continuous", "bins": 3},   # Low/Normal/High
        "wbc": {"type": "continuous", "bins": 3},           # Low/Normal/High
        "lactate": {"type": "continuous", "bins": 3},       # Low/Normal/High
        "sepsis": {"type": "categorical", "top_k": 2},      # 0=No, 1=Yes
    }
    
    return dag_edges, variable_config


if __name__ == "__main__":
    # Quick test
    print("Example MIMIC config:")
    edges, cfg = get_example_mimic_config()
    print(f"DAG edges: {edges}")
    print(f"Variables: {list(cfg.keys())}")
