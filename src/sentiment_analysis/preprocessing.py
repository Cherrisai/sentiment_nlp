import re
import string

import pandas as pd

from .config import LABEL_COLUMN, RATING_COLUMN, TEXT_COLUMNS

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_DIGIT_RE = re.compile(r"\b\d+\b")
_PUNCT_TRANSLATOR = str.maketrans({p: " " for p in string.punctuation})


def normalize_text(text: object) -> str:
    """Clean review text while keeping sentiment-bearing words intact."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.translate(_PUNCT_TRANSLATOR)
    text = _DIGIT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def rating_to_sentiment(rating: object) -> str:
    rating_value = float(rating)
    if rating_value <= 2:
        return "Negative"
    if rating_value == 3:
        return "Neutral"
    return "Positive"


def build_review_text(df: pd.DataFrame) -> pd.Series:
    missing = [column for column in TEXT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required text columns: {missing}")

    return (
        df[list(TEXT_COLUMNS)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize_text)
    )


def prepare_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean modeling dataframe with review_text and sentiment labels."""
    if RATING_COLUMN not in df.columns:
        raise ValueError(f"Missing required rating column: {RATING_COLUMN}")

    prepared = df.copy()
    prepared["review_text"] = build_review_text(prepared)
    prepared[LABEL_COLUMN] = prepared[RATING_COLUMN].map(rating_to_sentiment)
    prepared = prepared[prepared["review_text"].str.len() > 0].reset_index(drop=True)
    return prepared

