import torch
import numpy as np
from torch_geometric.data import Data

from grasp.detection.classification_storage import ClassificationStorage
from grasp.detection.clustering import ClusterManager
import logging

logger = logging.getLogger(__name__)


def apply_max_node_constrain(data: Data, max_nodes: int) -> Data:
    num_nodes = data.x.shape[0]  # type: ignore
    if num_nodes > max_nodes:
        data = data.subgraph(torch.arange(min(max_nodes, num_nodes)))
        # logger.info(
        #     f"Applied max node constrain: {num_nodes} -> {data.x.shape[0]}"  # type: ignore
        # )
    return data


def zero_out_executable_node_features(
    data: Data,
) -> Data:
    data.x[: data.batch_size, -data.y.shape[-1] :] = 0  # type: ignore
    return data


def prepare_data(
    data: Data,
    max_nodes=900000,
) -> Data:
    data = apply_max_node_constrain(data, max_nodes)
    data = zero_out_executable_node_features(data)
    return data


def check_classification_with_clusters(
    cluster_manager: ClusterManager, cls_storage: ClassificationStorage
) -> None:
    true_labels: list[int] = cls_storage.y
    pred_labels: list[int] = cls_storage.y_hat
    allowed_clusters: dict[int, set[int]] = (
        cluster_manager.misclassification_clusters
    )

    for y, y_hat in zip(true_labels, pred_labels):
        allowed: set[int] = allowed_clusters.get(y, set[int]())

        if (
            y_hat == y or y_hat in allowed
        ):  # treat as correct by mapping prediction to truth
            cls_storage.y_hat_cluster_corrected.append(y)
        else:
            cls_storage.y_hat_cluster_corrected.append(y_hat)


def extract_node_cmd_labels_uuids_and_ids(
    cls_storage: ClassificationStorage,
    node_cmd_labels: torch.Tensor,
    node_uuids: torch.Tensor,
    node_ids: torch.Tensor,
    window_path: str,
    subject_mask: torch.Tensor,
) -> None:
    subject_node_cmd_labels = np.array(node_cmd_labels)[subject_mask.numpy()]
    cls_storage.true_cmd_labels.extend(subject_node_cmd_labels.tolist())
    subject_node_uuids = np.array(node_uuids)[subject_mask.numpy()]
    cls_storage.node_uuids.extend(subject_node_uuids.tolist())
    subject_node_ids = np.array(node_ids)[subject_mask.numpy()]
    cls_storage.node_ids.extend(subject_node_ids.tolist())
    cls_storage.window_path.extend([window_path] * subject_node_ids.shape[0])


def compute_class_weights(train_data: Data, num_classes: int) -> torch.Tensor:
    """Compute class weights using inverse frequency normalization."""
    y = train_data.y
    if y.dim() > 1:
        y = y.argmax(dim=1)

    # Count samples per class
    class_counts = torch.bincount(y, minlength=num_classes)
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1)
    # Weight = total_samples / (num_classes * class_count)
    weights = len(y) / (num_classes * class_counts.float())
    return weights
