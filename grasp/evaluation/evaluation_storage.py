import logging
from pathlib import Path

import torch

from grasp.detection.classification_storage import ClassificationStorage
from grasp.evaluation.ground_truth_storage import GroundTruthStorage
from grasp.utils.evaluation_helpers import (
    compute_metrics,
    create_cmd_to_id_inverse,
    get_anomalies,
    get_labels_from_cmd_ids,
)

logger = logging.getLogger(__name__)


class EvaluationStorage:
    def __init__(
        self,
        cs: ClassificationStorage | None = None,
        train_subject_cmd_to_id: dict[str, int] | None = None,
    ) -> None:
        self.misclassified_uuids: list[str] = []
        self.misclassified_ids: list[str] = []
        self.true_cmd_labels: list[str] = []
        self.pred_cmd_labels: list[str] = []
        self.window_path: list[str] = []
        self.hits_per_gt_group: dict[str, list[str]] = {}
        self.y_per_gt_group: dict[str, list[int]] = {}
        self.y_hat_per_gt_group: dict[str, list[int]] = {}
        self.y_per_gt_group_unique: dict[str, list[int]] = {}
        self.y_hat_per_gt_group_unique: dict[str, list[int]] = {}
        self.unknown_uuids_per_gt_group: dict[str, list[str]] = {}
        self.hits_per_gt_file: dict[str, list[str]] = {}
        self.y_per_gt_file: dict[str, list[int]] = {}
        self.y_hat_per_gt_file: dict[str, list[int]] = {}
        self.y_per_gt_file_unique: dict[str, list[int]] = {}
        self.y_hat_per_gt_file_unique: dict[str, list[int]] = {}
        self.unknown_uuids_per_gt_file: dict[str, list[str]] = {}

        if cs is not None:
            if train_subject_cmd_to_id is None:
                raise ValueError(
                    "train_subject_cmd_to_id is required when "
                    "initializing with a ClassificationStorage."
                )

            train_subject_cmd_to_id_inverse: dict[int, str] = create_cmd_to_id_inverse(
                train_subject_cmd_to_id
            )
            misclassified_indices: list[int] = get_anomalies(cs)
            self.misclassified_uuids = [cs.node_uuids[i] for i in misclassified_indices]
            self.misclassified_ids = [cs.node_ids[i] for i in misclassified_indices]
            pred_cmd_labels: list[str] = get_labels_from_cmd_ids(
                cs.y_hat_cluster_corrected, train_subject_cmd_to_id_inverse
            )
            pred_cmd_labels_misclassified: list[str] = [
                pred_cmd_labels[i] for i in misclassified_indices
            ]
            true_cmd_labels_misclassified: list[str] = [
                cs.true_cmd_labels[i] for i in misclassified_indices
            ]
            window_path_misclassified: list[str] = [
                cs.window_path[i] for i in misclassified_indices
            ]
            self.true_cmd_labels = true_cmd_labels_misclassified
            self.pred_cmd_labels = pred_cmd_labels_misclassified
            self.window_path = window_path_misclassified

    def save_to_file(self, filepath: str) -> None:
        data: dict[str, list[str] | dict[str, list[str]] | dict[str, list[int]]] = {
            "misclassified_uuids": self.misclassified_uuids,
            "misclassified_ids": self.misclassified_ids,
            "true_cmd_labels": self.true_cmd_labels,
            "pred_cmd_labels": self.pred_cmd_labels,
            "window_path": self.window_path,
            "hits_per_gt_group": self.hits_per_gt_group,
            "y_per_gt_group": self.y_per_gt_group,
            "y_hat_per_gt_group": self.y_hat_per_gt_group,
            "y_per_gt_group_unique": self.y_per_gt_group_unique,
            "y_hat_per_gt_group_unique": self.y_hat_per_gt_group_unique,
            "unknown_uuids_per_gt_group": self.unknown_uuids_per_gt_group,
            "hits_per_gt_file": self.hits_per_gt_file,
            "y_per_gt_file": self.y_per_gt_file,
            "y_hat_per_gt_file": self.y_hat_per_gt_file,
            "y_per_gt_file_unique": self.y_per_gt_file_unique,
            "y_hat_per_gt_file_unique": self.y_hat_per_gt_file_unique,
            "unknown_uuids_per_gt_file": self.unknown_uuids_per_gt_file,
        }
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, filepath)
        logger.info(f"Classification results saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        data: dict[str, list[str] | dict[str, list[str]] | dict[str, list[int]]] = torch.load(
            filepath
        )
        self.misclassified_uuids = data["misclassified_uuids"]  # type: ignore
        self.misclassified_ids = data["misclassified_ids"]  # type: ignore
        self.true_cmd_labels = data["true_cmd_labels"]  # type: ignore
        self.pred_cmd_labels = data["pred_cmd_labels"]  # type: ignore
        self.window_path = data["window_path"]  # type: ignore
        self.hits_per_gt_group = data.get("hits_per_gt_group", {})  # type: ignore
        self.y_per_gt_group = data.get("y_per_gt_group", {})  # type: ignore
        self.y_hat_per_gt_group = data.get("y_hat_per_gt_group", {})  # type: ignore
        self.y_per_gt_group_unique = data.get("y_per_gt_group_unique", {})  # type: ignore
        self.y_hat_per_gt_group_unique = data.get("y_hat_per_gt_group_unique", {})  # type: ignore
        self.unknown_uuids_per_gt_group = data.get("unknown_uuids_per_gt_group", {})  # type: ignore
        self.hits_per_gt_file = data.get("hits_per_gt_file", {})  # type: ignore
        self.y_per_gt_file = data.get("y_per_gt_file", {})  # type: ignore
        self.y_hat_per_gt_file = data.get("y_hat_per_gt_file", {})  # type: ignore
        self.y_per_gt_file_unique = data.get("y_per_gt_file_unique", {})  # type: ignore
        self.y_hat_per_gt_file_unique = data.get("y_hat_per_gt_file_unique", {})  # type: ignore
        self.unknown_uuids_per_gt_file = data.get("unknown_uuids_per_gt_file", {})  # type: ignore
        logger.info(f"Classification results loaded from {filepath}")

    @staticmethod
    def _compute_labeling(
        uuid_groups: dict[str, set[str]],
        cs: ClassificationStorage,
        misclassified_uuids: list[str],
    ) -> tuple[
        dict[str, list[int]],
        dict[str, list[int]],
        dict[str, list[str]],
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, list[int]],
    ]:
        y_per: dict[str, list[int]] = {}
        y_hat_per: dict[str, list[int]] = {}
        hits_per: dict[str, list[str]] = {}
        unknown_per: dict[str, list[str]] = {}
        y_per_unique: dict[str, list[int]] = {}
        y_hat_per_unique: dict[str, list[int]] = {}

        unique_node_uuids = set(cs.node_uuids)

        for group_name, uuid_set in uuid_groups.items():
            y_per[group_name] = []
            y_hat_per[group_name] = []

            hits_per[group_name] = [uuid for uuid in misclassified_uuids if uuid in uuid_set]

            for uuid in cs.node_uuids:
                y_per[group_name].append(1 if uuid in uuid_set else 0)
                is_anomaly = 1 if uuid in misclassified_uuids else 0
                y_hat_per[group_name].append(is_anomaly)

            for uuid in unique_node_uuids:
                y_per_unique.setdefault(group_name, []).append(1 if uuid in uuid_set else 0)
                is_anomaly = 1 if uuid in misclassified_uuids else 0
                y_hat_per_unique.setdefault(group_name, []).append(is_anomaly)

            unknown_per[group_name] = [uuid for uuid in uuid_set if uuid not in unique_node_uuids]

        return (
            y_per,
            y_hat_per,
            hits_per,
            unknown_per,
            y_per_unique,
            y_hat_per_unique,
        )

    def get_y_and_y_hat_per_gt_group(
        self, gts: GroundTruthStorage, cs: ClassificationStorage
    ) -> None:
        grasp_gt_uuids = set(gts.combined(grasp=True)["uuid"].tolist())
        non_grasp_gt_uuids = set(gts.combined(grasp=False)["uuid"].tolist())

        uuid_groups: dict[str, set[str]] = {
            "grasp": grasp_gt_uuids,
            "non_grasp": non_grasp_gt_uuids,
        }
        (
            self.y_per_gt_group,
            self.y_hat_per_gt_group,
            self.hits_per_gt_group,
            self.unknown_uuids_per_gt_group,
            self.y_per_gt_group_unique,
            self.y_hat_per_gt_group_unique,
        ) = self._compute_labeling(uuid_groups, cs, self.misclassified_uuids)
        logger.info("Computed y and y_hat per combined ground truth (grasp/non_grasp).")
        logger.info(
            f"Grasp hits: {len(self.hits_per_gt_group.get('grasp', []))} of {len(grasp_gt_uuids)}"
        )
        logger.info(
            f"Non-grasp hits: {len(self.hits_per_gt_group.get('non_grasp', []))}"
            f" of {len(non_grasp_gt_uuids)}"
        )
        logger.info(f"Grasp unknown UUIDs: {len(self.unknown_uuids_per_gt_group.get('grasp', []))}")
        logger.info(
            f"Non-grasp unknown UUIDs: {len(self.unknown_uuids_per_gt_group.get('non_grasp', []))}"
        )

    def get_y_and_y_hat_per_gt_file(
        self, gts: GroundTruthStorage, cs: ClassificationStorage
    ) -> None:
        uuid_groups: dict[str, set[str]] = {}
        for gs_file in gts.keys():  # noqa: SIM118
            gs_file_df = gts.get(gs_file)
            uuid_groups[gs_file] = set(gs_file_df["uuid"].tolist())

        (
            self.y_per_gt_file,
            self.y_hat_per_gt_file,
            self.hits_per_gt_file,
            self.unknown_uuids_per_gt_file,
            self.y_per_gt_file_unique,
            self.y_hat_per_gt_file_unique,
        ) = self._compute_labeling(uuid_groups, cs, self.misclassified_uuids)
        logger.info("Computed y and y_hat per ground truth file.")
        logger.info(f"Total ground truth files: {len(self.hits_per_gt_file)}")

    def get_metrics_per_gt_group(
        self,
    ) -> dict[str, dict[str, float]]:
        metrics_per_group: dict[str, dict[str, float]] = {}
        for group_name in self.y_per_gt_group:
            y_true = self.y_per_gt_group_unique[group_name]
            y_pred = self.y_hat_per_gt_group_unique[group_name]
            metrics = compute_metrics(y_true, y_pred)
            metrics_per_group[group_name] = metrics
            logger.info(
                f"Metrics for group '{group_name}': "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )

        return metrics_per_group

    def get_metrics_per_gt_file(
        self,
    ) -> dict[str, dict[str, float]]:
        metrics_per_file: dict[str, dict[str, float]] = {}
        for file_name in self.y_per_gt_file:
            y_true: list[int] = self.y_per_gt_file_unique[file_name]
            y_pred: list[int] = self.y_hat_per_gt_file_unique[file_name]
            metrics = compute_metrics(y_true, y_pred)
            metrics_per_file[file_name] = metrics
            logger.info(
                f"Metrics for file '{file_name}': "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )

        return metrics_per_file
