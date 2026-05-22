from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(y_true, y_pred) -> dict:
    labels = ["Negative", "Neutral", "Positive"]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def save_metrics(metrics: dict, metrics_path: Path, report_path: Path, matrix_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    report_df.to_csv(report_path)

    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    )
    matrix_df.to_csv(matrix_path)

