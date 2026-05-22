from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split

from .config import LABEL_COLUMN, RANDOM_STATE, TEST_SIZE
from .data_loader import load_prepared_reviews
from .evaluation import evaluate_model
from .modeling import build_pipeline


def train_sentiment_model(dataset_path):
    df = load_prepared_reviews(dataset_path)
    X = df["review_text"]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, predictions)

    split_info = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
        "class_distribution": y.value_counts().to_dict(),
    }

    return pipeline, metrics, split_info, df


def save_model(model, model_path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

