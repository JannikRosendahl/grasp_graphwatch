import os.path as osp
from typing import Any, Callable
import torch
from torch_geometric.data import Data, InMemoryDataset
import logging
import numpy as np

from grasp.utils.graph_helpers import (
    generate_graph_extendedname,
    load_graph_data,
    extract_graph_basename,
)
from grasp.graph.location_transformer import (
    LocationEncoder,
    build_location_embeddings_from_model,
)


logger: logging.Logger = logging.getLogger(__name__)


class ExtendedGraspGraph(InMemoryDataset):
    def __init__(
        self,
        raw_grasp_graph_path: str,
        location_autoencoder: LocationEncoder,
        train_cmd_to_id: dict[str, int],
        root: str | None = None,
        experiment_prefix: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        force_reload: bool = False,
        autoencoder_device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        raw_grasp_graph_basename = extract_graph_basename(
            osp.basename(raw_grasp_graph_path)
        )
        self.raw_grasp_graph_path: str = raw_grasp_graph_path
        self.file_basename = generate_graph_extendedname(
            raw_grasp_graph_basename
        )
        self.file_name = self.file_basename + ".pt"

        self.location_autoencoder: LocationEncoder = (
            location_autoencoder
        )
        self.train_cmd_to_id: dict[str, int] = train_cmd_to_id

        self.autoencoder_device: torch.device = autoencoder_device

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.load(osp.join(self.processed_dir, self.file_name))
        logging.info(
            f"Extended graph loaded from "
            f"{osp.join(self.processed_dir, self.file_name)}."
        )

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.file_name}.csv"]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    @property
    def processed_file_names(self) -> list[str]:
        return [self.file_name]

    def download(self) -> None:
        pass

    def process(self) -> None:
        # read existing graph from file
        grasp_graph_data: Data = load_graph_data(self.raw_grasp_graph_path)

        node_locations: list[str] = [
            str(location) for location in grasp_graph_data.node_location
        ]

        location_embeddings: np.ndarray = build_location_embeddings_from_model(
            self.location_autoencoder,
            node_locations,
            device=self.autoencoder_device,
        )
        location_embeddings_tensors = torch.tensor(
            location_embeddings, dtype=torch.float32
        )

        grasp_graph_data.x = torch.cat(
            [grasp_graph_data.x, location_embeddings_tensors],  # type: ignore
            dim=1,
        )

        mask = grasp_graph_data.subject_mask
        num_nodes = len(mask)
        cmd_map = self.train_cmd_to_id
        # Build y as class indices; 0 for non-subject and unknown commands
        y_idx = torch.full((num_nodes, len(cmd_map)), 0, dtype=torch.float)
        for i in range(num_nodes):
            if bool(mask[i]):
                c = grasp_graph_data.node_cmd_labels[i]
                if c in cmd_map:
                    y_idx[i][cmd_map[c]] = 1.0
                else:
                    y_idx[i] = (
                        0.0  # Treat unknown as "nothing observed / same as file or ip"
                    )

        grasp_graph_data.y = y_idx

        # also add y_idx to x for masked learning
        grasp_graph_data.x = torch.cat(
            [grasp_graph_data.x, y_idx],  # type: ignore
            dim=1,
        )

        graph_data_extended = Data(
            x=grasp_graph_data.x,
            edge_index=grasp_graph_data.edge_index,
            edge_attr=grasp_graph_data.edge_attr,
            t=grasp_graph_data.t,
            y=grasp_graph_data.y,
            subject_mask=grasp_graph_data.subject_mask,
            node_cmd_labels=grasp_graph_data.node_cmd_labels,
            node_uuid=grasp_graph_data.node_uuid,
            node_index_id=grasp_graph_data.node_index_id,
            node_location=grasp_graph_data.node_location,
        )
        logging.info(
            f"Extended graph processed with {len(graph_data_extended.x)} "
            f"nodes, {graph_data_extended.edge_index.shape[1]} edges, "
            f"{graph_data_extended.subject_mask.sum().item()} subjects, "
            f" node feature dimension {graph_data_extended.x.shape[1]}, "
            f" edge feature dimension {graph_data_extended.edge_attr.shape[1]} ,"
            f" label dimension {graph_data_extended.y.shape[1]} "
            f" and will be saved under {osp.join(self.processed_dir, self.file_name)}"
        )

        torch.save(
            self.collate([graph_data_extended]),
            osp.join(self.processed_dir, self.file_name),
        )
