import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grasp.detection.classification_storage import ClassificationStorage
from grasp.evaluation.evaluation_storage import EvaluationStorage
from grasp.evaluation.ground_truth_storage import GroundTruthStorage
from grasp.graph.graph_storage import GraphStorage


class Report:
    def __init__(
        self,
        classification_storage: ClassificationStorage | None = None,
        evaluation_storage: EvaluationStorage | None = None,
        graph_storage: GraphStorage | None = None,
        ground_truth_storage: GroundTruthStorage | None = None,
    ) -> None:
        self.classification_storage = classification_storage
        self.evaluation_storage = evaluation_storage
        self.graph_storage = graph_storage
        self.ground_truth_storage = ground_truth_storage

    def _classification_core(self) -> dict[str, Any]:
        if self.classification_storage is None:
            return {}

        cs = self.classification_storage
        total = len(cs.y)
        unique_nodes = len(set(cs.node_uuids)) if cs.node_uuids else 0
        window_counter = Counter(cs.window_path)

        # Missed attacks: true positive class but predicted as 0 or wrong class
        anomalies = [
            uuid
            for uuid, true, pred in zip(cs.node_uuids, cs.y, cs.y_hat_cluster_corrected)
            if true != 0 and pred != true
        ]

        corrected_diff = sum(1 for a, b in zip(cs.y_hat, cs.y_hat_cluster_corrected) if a != b)

        return {
            "total_nodes": total,
            "unique_nodes": unique_nodes,
            "time_windows": len(window_counter),
            "anomalies": len(anomalies),
            "unique_anomalous_nodes": len(set(anomalies)),
            "corrections_applied": corrected_diff,
        }

    def _classification_by_time_window(self) -> list[dict[str, Any]]:
        if self.classification_storage is None:
            return []

        cs = self.classification_storage
        buckets: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "events": 0,
                "anomalies": 0,
                "corrected_anomalies": 0,
            }
        )

        for true, pred, pred_corrected, window in zip(
            cs.y, cs.y_hat, cs.y_hat_cluster_corrected, cs.window_path
        ):
            buckets[window]["events"] += 1
            if true != pred:
                buckets[window]["anomalies"] += 1
            if true != pred_corrected:
                buckets[window]["corrected_anomalies"] += 1
        window_rows: list[dict[str, Any]] = []
        for window, stats in sorted(
            buckets.items(),
            key=lambda kv: kv[1]["corrected_anomalies"],
            reverse=True,
        ):
            window_rows.append({"time_window": window, **stats})
        return window_rows

    def _evaluation_hits_summary(self) -> dict[str, Any]:
        if self.evaluation_storage is None:
            return {}

        es = self.evaluation_storage

        group_rows: list[dict[str, Any]] = []
        for group_name, y_true in es.y_per_gt_group.items():
            y_pred = es.y_hat_per_gt_group.get(group_name, [])
            y_true_unique = es.y_per_gt_group_unique.get(group_name, [])
            y_pred_unique = es.y_hat_per_gt_group_unique.get(group_name, [])

            hits = len(es.hits_per_gt_group.get(group_name, []))

            gt_total = sum(1 for v in y_true if v)
            gt_total_unique = sum(1 for v in y_true_unique if v)
            predicted = sum(1 for v in y_pred if v)
            predicted_unique = sum(1 for v in y_pred_unique if v)

            misses = max(gt_total - hits, 0)
            misses_unique = max(gt_total_unique - hits, 0)
            false_alarms = max(predicted - hits, 0)
            false_alarms_unique = max(predicted_unique - hits, 0)
            unknown = len(es.unknown_uuids_per_gt_group.get(group_name, []))

            group_rows.append(
                {
                    "group": group_name,
                    "gt_attacks": gt_total,
                    "detected": hits,
                    "missed": misses,
                    "false_alarms": false_alarms,
                    "predicted_anomalies": predicted,
                    "gt_attacks_unique": gt_total_unique,
                    "missed_unique": misses_unique,
                    "false_alarms_unique": false_alarms_unique,
                    "predicted_anomalies_unique": predicted_unique,
                    "unknown_gt_uuids": unknown,
                }
            )

        file_rows: list[dict[str, Any]] = []
        for file_name, y_true in es.y_per_gt_file.items():
            y_pred = es.y_hat_per_gt_file.get(file_name, [])
            y_true_unique = es.y_per_gt_file_unique.get(file_name, [])
            y_pred_unique = es.y_hat_per_gt_file_unique.get(file_name, [])

            hits = len(es.hits_per_gt_file.get(file_name, []))
            unique_hits = len(set(es.hits_per_gt_file.get(file_name, [])))

            gt_total = sum(1 for v in y_true if v)
            gt_total_unique = sum(1 for v in y_true_unique if v)
            predicted = sum(1 for v in y_pred if v)
            predicted_unique = sum(1 for v in y_pred_unique if v)

            unknown = len(es.unknown_uuids_per_gt_file.get(file_name, []))

            file_rows.append(
                {
                    "file": file_name,
                    "gt_attacks": gt_total,
                    "detected": hits,
                    "gt_attacks_unique": gt_total_unique,
                    "detected_unique": unique_hits,
                    "unknown_gt_uuids": unknown,
                }
            )

        return {
            "misclassified_nodes": len(es.misclassified_uuids),
            "by_group": group_rows,
            "by_file": file_rows,
        }

    def _ground_truth_files(self) -> list[dict[str, Any]]:
        if self.ground_truth_storage is None:
            return []

        gt = self.ground_truth_storage
        rows: list[dict[str, Any]] = []
        for path in sorted(gt.keys()):
            df = gt.get(path)
            rows.append({"file": path, "rows": len(df)})
        return rows

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "detection_summary": self._classification_core(),
            "time_window_breakdown": self._classification_by_time_window(),
            "evaluation_hits": self._evaluation_hits_summary(),
            "ground_truth_files": self._ground_truth_files(),
        }

        # Graph-level context
        if self.graph_storage is not None:
            gs = self.graph_storage
            payload["graph"] = {
                "train_paths": len(gs.train_data_paths),
                "test_paths": len(gs.test_data_paths),
                "extended_train_paths": len(gs.extended_train_data_paths),
                "extended_test_paths": len(gs.extended_test_data_paths),
                "unique_train_locations": len(gs.unique_train_locations),
                "unique_test_locations": len(gs.unique_test_locations),
            }

        return payload

    def save(self, filepath: str) -> None:
        payload = self.to_dict()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> dict[str, Any]:
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
