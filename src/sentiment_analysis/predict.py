from __future__ import annotations

from pathlib import Path
import re

import joblib
import pandas as pd

from .config import DEFAULT_MODEL_PATH
from .preprocessing import normalize_text


POSITIVE_WORDS = {
    "excellent",
    "amazing",
    "awesome",
    "best",
    "better",
    "good",
    "great",
    "happy",
    "love",
    "loved",
    "nice",
    "perfect",
    "satisfied",
    "smooth",
    "super",
    "superb",
    "value",
    "worth",
    "wonderful",
    "fast",
    "fine",
    "appreciated",
}

NEGATIVE_WORDS = {
    "bad",
    "blur",
    "broken",
    "cheap",
    "damage",
    "defect",
    "defective",
    "disappointed",
    "faulty",
    "horrible",
    "issue",
    "lag",
    "missing",
    "pathetic",
    "poor",
    "problem",
    "return",
    "slow",
    "terrible",
    "useless",
    "waste",
    "worst",
}

NEGATION_WORDS = {"not", "no", "never", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't"}


def _rating_sentiment(rating: object) -> str | None:
    if rating is None or pd.isna(rating):
        return None
    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return None
    if rating_value >= 4:
        return "Positive"
    if rating_value <= 2:
        return "Negative"
    return "Neutral"


def _lexicon_score(cleaned_text: str) -> tuple[int, list[str]]:
    tokens = re.findall(r"[a-zA-Z']+", cleaned_text.lower())
    score = 0
    hits: list[str] = []
    for index, token in enumerate(tokens):
        previous_words = set(tokens[max(0, index - 3) : index])
        negated = bool(previous_words & NEGATION_WORDS)
        if token in POSITIVE_WORDS:
            score += -2 if negated else 1
            hits.append(("not " if negated else "") + token)
        elif token in NEGATIVE_WORDS:
            score += 2 if negated else -1
            hits.append(("not " if negated else "") + token)
    return score, hits[:6]


def _empty_probabilities() -> dict[str, float]:
    return {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}


def _hybrid_sentiment(model, cleaned_text: str, rating: object = None) -> dict:
    model_prediction = model.predict([cleaned_text])[0]
    probabilities = _empty_probabilities()
    if hasattr(model, "predict_proba"):
        model_probabilities = model.predict_proba([cleaned_text])[0]
        probabilities.update(
            {
                class_name: float(prob)
                for class_name, prob in zip(model.classes_, model_probabilities)
            }
        )

    scores = {
        "Negative": probabilities.get("Negative", 0.0) * 2.0,
        "Neutral": probabilities.get("Neutral", 0.0) * 1.5,
        "Positive": probabilities.get("Positive", 0.0) * 2.0,
    }

    rating_sentiment = _rating_sentiment(rating)
    if rating_sentiment:
        scores[rating_sentiment] += 3.0

    lexicon_score, keyword_hits = _lexicon_score(cleaned_text)
    if lexicon_score >= 2:
        scores["Positive"] += 2.0
    elif lexicon_score == 1:
        scores["Positive"] += 1.0
    elif lexicon_score <= -2:
        scores["Negative"] += 2.0
    elif lexicon_score == -1:
        scores["Negative"] += 1.0

    if rating_sentiment:
        final_sentiment = rating_sentiment
        scores[rating_sentiment] += 4.0
    elif not cleaned_text.strip():
        final_sentiment = "Neutral"
    else:
        final_sentiment = max(scores, key=scores.get)

    total_score = sum(max(value, 0.0) for value in scores.values()) or 1.0
    hybrid_probabilities = {
        label: round(max(value, 0.0) / total_score, 4)
        for label, value in scores.items()
    }

    reason_parts = [f"model={model_prediction}"]
    if rating_sentiment:
        reason_parts.append(f"rating signal={rating_sentiment}")
    if keyword_hits:
        reason_parts.append("keywords=" + ", ".join(keyword_hits))
    if not rating_sentiment and not keyword_hits:
        reason_parts.append("text pattern signal")

    return {
        "sentiment": final_sentiment,
        "model_sentiment": model_prediction,
        "rating_sentiment": rating_sentiment,
        "keyword_hits": keyword_hits,
        "probabilities": hybrid_probabilities,
        "confidence": hybrid_probabilities[final_sentiment],
        "reason": "; ".join(reason_parts),
    }


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run `python3 train.py` first."
        )
    return joblib.load(model_path)


def predict_sentiment(text: str, model=None, rating: object = None) -> dict:
    if model is None:
        model = load_model()

    cleaned_text = normalize_text(text)
    result = _hybrid_sentiment(model, cleaned_text, rating=rating)
    result["cleaned_text"] = cleaned_text

    return result


def _combined_text(df: pd.DataFrame, text_column: str | None) -> pd.Series:
    lower_to_column = {str(column).lower(): column for column in df.columns}
    detected_columns = [
        lower_to_column[name]
        for name in ["title", "body", "review", "reviews", "review_text", "text", "comment"]
        if name in lower_to_column
    ]
    if text_column and text_column in df.columns and text_column not in detected_columns:
        detected_columns.append(text_column)
    if not detected_columns:
        raise ValueError("No review text column found.")
    return df[detected_columns].fillna("").astype(str).agg(" ".join, axis=1)


def _rating_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if str(column).lower() in {"rating", "ratings", "stars", "star_rating"}:
            return column
    return None


def predict_dataframe(df: pd.DataFrame, text_column: str | None = None, model=None) -> pd.DataFrame:
    if model is None:
        model = load_model()

    result_df = df.copy()
    review_text = _combined_text(result_df, text_column)
    cleaned_text = review_text.map(normalize_text)
    rating_col = _rating_column(result_df)

    predictions = []
    for index, text in cleaned_text.items():
        rating = result_df.at[index, rating_col] if rating_col else None
        predictions.append(_hybrid_sentiment(model, text, rating=rating))

    result_df["review_text_used"] = review_text
    result_df["predicted_sentiment"] = [item["sentiment"] for item in predictions]
    result_df["prediction_reason"] = [item["reason"] for item in predictions]
    result_df["confidence"] = [item["confidence"] for item in predictions]

    for class_name in ["Negative", "Neutral", "Positive"]:
        result_df[f"probability_{class_name.lower()}"] = [
            item["probabilities"].get(class_name, 0.0) for item in predictions
        ]

    return result_df
