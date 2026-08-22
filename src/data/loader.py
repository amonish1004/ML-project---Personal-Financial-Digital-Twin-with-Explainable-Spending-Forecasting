import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.config import DATA_RAW_DIR, SCHEMAS


def load_raw_tsv(file_name: str, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads a headerless TSV table from the raw dataset directory.
    
    Parameters
    ----------
    file_name : str
        Name of the TSV file (e.g. 'fin_trans.tsv').
    raw_dir : Path, optional
        Path to raw data directory. Defaults to DATA_RAW_DIR.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with explicit column names.
    """
    if raw_dir is None:
        raw_dir = DATA_RAW_DIR
        
    file_path = raw_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
        
    if file_name not in SCHEMAS:
        raise ValueError(f"Schema not defined for file: {file_name}")
        
    cols = SCHEMAS[file_name]
    
    # Read headerless TSV safely without mutating raw source
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=cols,
        low_memory=False
    )
    
    return df


def load_all_raw_tables(raw_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all 8 headerless raw TSV tables into a dictionary.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping file names to loaded DataFrames.
    """
    tables = {}
    for file_name in SCHEMAS.keys():
        tables[file_name] = load_raw_tsv(file_name, raw_dir=raw_dir)
    return tables
