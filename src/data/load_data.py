"""Utilities for loading car reliability datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "car_reliability_synthetic.csv"


def load_reliability_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the reliability dataset.

    Parameters
    ----------
    csv_path: str, optional
        Custom path to the CSV file. When omitted the default synthetic dataset is used.
    """
    path = Path(csv_path) if csv_path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)
