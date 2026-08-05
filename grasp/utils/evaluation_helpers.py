from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from grasp.detection.classification_storage import ClassificationStorage


def get_anomalies(cs: ClassificationStorage, corrected: bool = True) -> list[int]:
    if not corrected:
        y_hat: list[int] = cs.y_hat
    else:
        y_hat = cs.y_hat_cluster_corrected
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(cs.y, y_hat)) if true != pred]
    return misclassified_indices


def create_cmd_to_id_inverse(
    subject_cmd_to_id: dict[str, int],
) -> dict[int, str]:
    subject_cmd_to_id_inverse: dict[int, str] = {v: k for k, v in subject_cmd_to_id.items()}
    return subject_cmd_to_id_inverse


def get_labels_from_cmd_ids(
    cmd_ids: list[int],
    cmd_to_id_inverse: dict[int, str],
) -> list[str]:
    cmd_labels: list[str] = [cmd_to_id_inverse[cmd_id] for cmd_id in cmd_ids]
    return cmd_labels


def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
) -> dict[str, Any]:
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    report = classification_report(true_labels, pred_labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }
    return metrics


def compute_metrics_multiclass(
    true_labels: list[int],
    pred_labels: list[int],
) -> dict[str, Any]:
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    report = classification_report(true_labels, pred_labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_f1_score": macro_f1,
        "classification_report": report,
    }
    return metrics


def print_metrics(metrics: dict[str, float]) -> None:
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Classification Report:")
    print(metrics["classification_report"])
