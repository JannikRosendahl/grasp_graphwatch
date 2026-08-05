import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ClassificationStorage:
    """Class to store and manage classification results for detected code snippets."""

    def __init__(self):
        self.y: list[int] = []
        self.y_hat: list[int] = []
        self.y_hat_cluster_corrected: list[int] = []
        self.true_cmd_labels: list[str] = []
        self.node_uuids: list[str] = []
        self.node_ids: list[str] = []
        self.window_path: list[str] = []
        self.y_hat_proba: list[list[float]] = []  # Store probabilities for each class

    def save_to_file(self, filepath: str) -> None:
        data = {
            "y": self.y,
            "y_hat": self.y_hat,
            "y_hat_cluster_corrected": self.y_hat_cluster_corrected,
            "true_cmd_labels": self.true_cmd_labels,
            "node_uuids": self.node_uuids,
            "node_ids": self.node_ids,
            "window_path": self.window_path,
            "y_hat_proba": self.y_hat_proba,
        }
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, filepath)
        logger.info(f"Classification results saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        data = torch.load(filepath)
        self.y = data["y"]
        self.y_hat = data["y_hat"]
        self.y_hat_cluster_corrected = data["y_hat_cluster_corrected"]
        self.true_cmd_labels = data["true_cmd_labels"]
        self.node_uuids = data["node_uuids"]
        self.node_ids = data["node_ids"]
        self.window_path = data["window_path"]
        logger.info(f"Classification results loaded from {filepath}")
