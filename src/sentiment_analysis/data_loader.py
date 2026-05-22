from pathlib import Path

import pandas as pd

from .config import DEFAULT_RAW_DATA_PATH
from .preprocessing import prepare_reviews


def load_raw_reviews(path: Path | str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_excel(path)


def load_prepared_reviews(path: Path | str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    raw_df = load_raw_reviews(path)
    return prepare_reviews(raw_df)

