import json
from pathlib import Path
from typing import Any

from grasp.detection.classification_storage import ClassificationStorage
from grasp.evaluation.evaluation_storage import EvaluationStorage
from grasp.evaluation.ground_truth_storage import GroundTruthStorage
from grasp.graph.graph_storage import GraphStorage


class DetailedReport:
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
        if self.evaluation_storage is None:
            return {}

        es = self.evaluation_storage
        anomalies: list[dict[str, Any]] = []
        for pred_label, true_label, time_window, id, uuid in zip(
            es.pred_cmd_labels,
            es.true_cmd_labels,
            es.window_path,
            es.misclassified_ids,
            es.misclassified_uuids,
        ):
            anomalies.append(
                {
                    "pred_label": pred_label,
                    "true_label": true_label,
                    "time_window": time_window,
                    "id": id,
                    "uuid": uuid,
                }
            )

        return {
            "anomalies": anomalies,
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
            "detailed_detection": self._classification_core(),
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
