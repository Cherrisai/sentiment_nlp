from src.sentiment_analysis.config import (
    DEFAULT_CLASSIFICATION_REPORT_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_DATA_PATH,
    PROJECT_NAME,
)
from src.sentiment_analysis.evaluation import save_metrics
from src.sentiment_analysis.train_utils import save_model, train_sentiment_model


def main() -> None:
    print(f"Project: {PROJECT_NAME}")
    print(f"Dataset: {DEFAULT_RAW_DATA_PATH.name}")
    print("Training sentiment model...")

    model, metrics, split_info, prepared_df = train_sentiment_model(DEFAULT_RAW_DATA_PATH)

    save_model(model, DEFAULT_MODEL_PATH)
    save_metrics(
        metrics,
        DEFAULT_METRICS_PATH,
        DEFAULT_CLASSIFICATION_REPORT_PATH,
        DEFAULT_CONFUSION_MATRIX_PATH,
    )

    DEFAULT_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(DEFAULT_PROCESSED_PATH, index=False)

    print("\nTraining complete.")
    print(f"Rows: {split_info['total_rows']} total | {split_info['train_rows']} train | {split_info['test_rows']} test")
    print(f"Class distribution: {split_info['class_distribution']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")
    print(f"Metrics saved to: {DEFAULT_METRICS_PATH}")
    print(f"Processed data saved to: {DEFAULT_PROCESSED_PATH}")


if __name__ == "__main__":
    main()

