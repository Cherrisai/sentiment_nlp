"""Reusable NLP utilities for the P676 sentiment analysis project."""

from .config import (
    DATASET_FILE_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_DATA_PATH,
    PROJECT_NAME,
)
from .predict import load_model, predict_sentiment

__all__ = [
    "DATASET_FILE_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PROCESSED_PATH",
    "DEFAULT_RAW_DATA_PATH",
    "PROJECT_NAME",
    "load_model",
    "predict_sentiment",
]

