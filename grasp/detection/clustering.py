import logging

import numpy as np
import torch

from grasp.detection.classification_storage import (
    ClassificationStorage,
)

logger = logging.getLogger(__name__)


class ClusterManager:
    def __init__(self, classification_storage: ClassificationStorage) -> None:
        self.classification_storage: ClassificationStorage = classification_storage
        self.misclassification_clusters: dict[int, set[int]] = {}

    def create_misclassification_clusters(self) -> None:
        """Populate misclassification clusters from stored predictions."""
        # init
        self.misclassification_clusters = {
            int(label): set() for label in np.unique(self.classification_storage.y)
        }
        for y, y_hat in zip(self.classification_storage.y, self.classification_storage.y_hat):
            self.misclassification_clusters[y].add(y_hat)
        logger.info(
            "Misclassification clusters created: "
            f" with {len(self.misclassification_clusters)} "
            f"clusters and {len(set(self.classification_storage.y_hat))} "  # type: ignore
            f"unique predicted labels."
        )

    def save_to_disk(self, filepath: str) -> None:
        torch.save(self.misclassification_clusters, filepath)
        logger.info(f"Misclassification clusters saved to {filepath}")

    def load_from_disk(self, filepath: str) -> None:
        self.misclassification_clusters = torch.load(filepath)
        logger.info(f"Misclassification clusters loaded from {filepath}")
