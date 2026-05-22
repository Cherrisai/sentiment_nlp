from pathlib import Path

PROJECT_NAME = "NLP Sentiment Analysis"
DATASET_FILE_NAME = "dataset -P676 (1).xlsx"
GROUP_LIST_FILE_NAME = "P676 NLP Group Lists.xlsx"
OBJECTIVE_FILE_NAME = "Project_Objective P676 (1).docx"

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

DEFAULT_RAW_DATA_PATH = RAW_DATA_DIR / DATASET_FILE_NAME
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_DIR / "reviews_with_sentiment.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
DEFAULT_METRICS_PATH = REPORT_DIR / "metrics.json"
DEFAULT_CLASSIFICATION_REPORT_PATH = REPORT_DIR / "classification_report.csv"
DEFAULT_CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_COLUMNS = ("title", "body")
RATING_COLUMN = "rating"
LABEL_COLUMN = "sentiment"
