from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BANK_DATA_PATH = ROOT / "ufce" / "data" / "bank.csv"
BANK_FOLDS_PATH = ROOT / "ufce" / "data" / "folds" / "bank"


def load_bank_dataframe() -> pd.DataFrame:
    return pd.read_csv(BANK_DATA_PATH)


def bank_folds_dir() -> Path:
    return BANK_FOLDS_PATH
