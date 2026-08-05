import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

from grasp.utils.detection_helpers import (
    prepare_data,
)
from grasp.detection.classification_storage import (
    ClassificationStorage,
)

import logging

logger = logging.getLogger(__name__)


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        experiment_prefix: str = "default",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_nodes: int = 100000,
        use_class_weights: bool = False,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.in_channels: int = in_channels
        self.hidden_channels: int = hidden_channels
        self.out_channels: int = out_channels
        self.num_layers: int = num_layers
        self.heads: int = heads
        self.edge_dim: int = edge_dim
        self.dropout: float = dropout
        self.experiment_prefix: str = experiment_prefix
        self.device: torch.device = device
        self.max_nodes: int = max_nodes
        self.use_class_weights: bool = use_class_weights
        self.class_weights: torch.Tensor | None = class_weights

        self.convs: torch.nn.ModuleList = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                # First layer: input features directly to hidden_channels
                in_dim = in_channels
                out_dim = hidden_channels
            else:
                # Subsequent layers: concatenated heads from previous layer
                in_dim = hidden_channels * heads
                out_dim = hidden_channels

            conv = GATConv(
                in_dim,
                out_dim,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.convs.append(conv)

        mlp_input_size = hidden_channels * heads
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x)

        x = self.mlp(x)
        return x

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))


def _labels_to_indices(y: torch.Tensor) -> torch.Tensor:
    if y.dim() > 1:
        y = y.argmax(dim=1)
    return y.long()


def train(
    model: GAT,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    selected_input_node_ids: torch.Tensor | None = None,
) -> float:
    model = model.to(device)
    model.train()
    total_loss = 0.0
    # Use class weights if available
    if model.use_class_weights and model.class_weights is not None:
        weights = model.class_weights.to(device)

        def criterion(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(y_hat, y, weight=weights)
    else:

        def criterion(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(y_hat, y)

    selected_ids_on_device: torch.Tensor | None = None
    if selected_input_node_ids is not None:
        selected_ids_on_device = selected_input_node_ids.to(device=device, dtype=torch.long)
    optimized_steps = 0

    for data in train_loader:
        data: Data = prepare_data(data, max_nodes=model.max_nodes)
        data = data.to(device=device)
        optimizer.zero_grad()
        out = model(
            data.x,  # type: ignore
            data.edge_index,  # type: ignore
            data.edge_attr,  # type: ignore
        )
        # Mask only process nodes
        y_hat = out[: data.batch_size]  # type: ignore
        y = _labels_to_indices(data.y[: data.batch_size])  # type: ignore

        if selected_ids_on_device is not None:
            if not hasattr(data, "n_id"):
                continue
            input_node_ids = data.n_id[: data.batch_size]
            keep_mask = torch.isin(input_node_ids, selected_ids_on_device)
            if not keep_mask.any():
                continue
            y_hat = y_hat[keep_mask]
            y = y[keep_mask]

        loss: torch.Tensor = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        optimized_steps += 1

    if optimized_steps == 0:
        return 0.0
    return total_loss / optimized_steps


def collect_hard_example_node_ids(
    model: GAT,
    data_loader: NeighborLoader,
    device: torch.device,
) -> set[int]:
    model = model.to(device)
    model.eval()
    selected_node_ids: set[int] = set()

    with torch.no_grad():
        for data in data_loader:
            data = prepare_data(data, max_nodes=model.max_nodes)
            data = data.to(device=device)
            out = model(
                data.x,  # type: ignore
                data.edge_index,  # type: ignore
                data.edge_attr,  # type: ignore
            )
            y_hat = out.softmax(dim=-1).argmax(dim=-1)[: data.batch_size]  # type: ignore
            y = _labels_to_indices(data.y[: data.batch_size])  # type: ignore

            keep_mask = y_hat != y
            if not keep_mask.any() or not hasattr(data, "n_id"):
                continue

            input_node_ids = data.n_id[: data.batch_size]
            selected_node_ids.update(
                int(nid) for nid in input_node_ids[keep_mask].detach().cpu().tolist()
            )
    return selected_node_ids


def test(
    model: GAT,
    test_loader: NeighborLoader,
    device: torch.device,
    classification_storage: ClassificationStorage,
) -> float:
    model = model.to(device)
    model.eval()
    y_preds: list[torch.Tensor] = []
    y_trues: list[torch.Tensor] = []
    with torch.no_grad():
        for data in test_loader:
            data: Data = prepare_data(data=data, max_nodes=model.max_nodes)
            data = data.to(device)
            out = model(
                data.x,  # type: ignore
                data.edge_index,  # type: ignore
                data.edge_attr,  # type: ignore
            )

            y_hat = out.softmax(dim=-1).argmax(dim=-1)[: data.batch_size]  # type: ignore
            y = data.y[: data.batch_size]  # type: ignore

            # Handly special case: 0 labels within mask -> aka. unknown process_cmd
            zero_labels_within_mask = y.sum(dim=1) == 0
            if zero_labels_within_mask.any():
                logger.debug(
                    f"Found {zero_labels_within_mask.sum().item()} zero labels within mask"
                )
            y = y.argmax(dim=1) if y.dim() > 1 else y  # type: ignore
            y[zero_labels_within_mask] = -1  # Set to invalid class

            y_preds.append(y_hat)
            y_trues.append(y)
            y_preds_propabilities = out.softmax(dim=-1)[: data.batch_size]  # type: ignore
            # classification_storage.y_hat_proba.extend(y_preds_propabilities.detach().cpu().tolist())
            topk_probs, topk_indices = torch.topk(
                y_preds_propabilities.detach().cpu(),
                k=min(3, y_preds_propabilities.size(-1)),
                dim=-1,
            )
            classification_storage.y_hat_proba.extend(
                [
                    [(int(idx), float(prob)) for idx, prob in zip(idxs.tolist(), probs.tolist())]
                    for idxs, probs in zip(topk_indices, topk_probs)
                ]  # type: ignore
            )

        y_preds_tensor = torch.cat(y_preds, dim=0)
        y_trues_tensor = torch.cat(y_trues, dim=0)
        accuracy = (y_preds_tensor == y_trues_tensor).float().mean().item()

        classification_storage.y_hat.extend(y_preds_tensor.tolist())
        classification_storage.y.extend(y_trues_tensor.tolist())

    return accuracy
