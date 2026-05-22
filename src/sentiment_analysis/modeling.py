from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import build_tfidf_vectorizer


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
