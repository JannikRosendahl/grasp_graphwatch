import logging
import os.path as osp
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from grasp.db import connector
from grasp.schema import EdgeColumns, NodeColumns, NodeType
from grasp.utils.graph_helpers import (
    generate_graph_basename,
    get_db_url,
    get_label_column_name_based_on_dataset,
    get_node_types,
    get_operations,
    record_collection_to_df,
)

logger: logging.Logger = logging.getLogger(__name__)


class GraspGraph(InMemoryDataset):
    def __init__(
        self,
        start_time: str,
        end_time: str,
        root: str | None = None,
        experiment_prefix: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        force_reload: bool = False,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.file_basename = generate_graph_basename(
            experiment_prefix or "default", start_time, end_time
        )
        self.file_name = self.file_basename + ".pt"

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        # Check if processed file exists before loading
        processed_path = osp.join(self.processed_dir, self.file_name)
        if osp.exists(processed_path):
            self.load(processed_path)
            logger.info(f"Loaded graph from {processed_path}.")
            logger.info(
                f"Graph has {self.data.num_nodes} nodes, "
                f"{self.data.num_edges} edges, and "
                f"{self.data.subject_mask.sum().item()} subjects."
            )
        else:
            logger.warning(
                f"Processed file not found: {processed_path}. "
                "Graph may be empty or processing failed."
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
        operations: dict[int, str] = get_operations()
        operations_tuple = tuple(operations.values())
        logger.debug(f"Operations tuple: {operations_tuple}")

        with connector.DBConnector(get_db_url()) as db_connector:
            edges, nodes = db_connector.fetch_graph_data(
                operations_tuple, int(self.start_time), int(self.end_time)
            )

        edges_df: pd.DataFrame = record_collection_to_df(edges)
        nodes_df: pd.DataFrame = record_collection_to_df(nodes)

        if edges_df.empty or nodes_df.empty:
            logger.warning(f"No data found for time range {self.start_time} to {self.end_time}")
            return

        index_id_to_idx = self._get_node_mapping(edges_df)

        # Ensure consistent types for mapping
        nodes_df[NodeColumns.INDEX_ID.value] = nodes_df[NodeColumns.INDEX_ID.value].astype(str)
        nodes_df["idx"] = nodes_df[NodeColumns.INDEX_ID.value].map(index_id_to_idx)
        nodes_df = nodes_df.sort_values("idx").reset_index(drop=True)

        src = edges_df[EdgeColumns.SRC_INDEX_ID.value].astype(str).map(index_id_to_idx)
        dst = edges_df[EdgeColumns.DST_INDEX_ID.value].astype(str).map(index_id_to_idx)

        edge_index: list[list[int]] = [src.tolist(), dst.tolist()]

        edge_types: list[int] = [
            list(operations.values()).index(op) for op in edges_df[EdgeColumns.OPERATION.value]
        ]

        node_types: dict[int, str] = get_node_types()
        node_type_indices = [
            list(node_types.values()).index(node_type)
            for node_type in nodes_df[NodeColumns.TYPE.value]
        ]

        edge_index_tensor, edge_attr_tensor, x_tensor = self._transform_to_tensors(
            edge_index,
            edge_types,
            operations_tuple,
            node_types,
            node_type_indices,
        )
        t_tensor: torch.Tensor = self._get_timestamps(edges_df)
        node_labels_placeholder: torch.Tensor = torch.zeros(nodes_df.shape[0], dtype=torch.long)
        node_uuid: list[str] = nodes_df[NodeColumns.UUID.value].tolist()
        node_index_id: list[int] = nodes_df[NodeColumns.INDEX_ID.value].tolist()

        node_location: list[str] = self._build_node_location(nodes_df)

        subject_mask, node_cmd_labels = self._build_subject_mask_and_node_cmd_labels(nodes_df)
        graph_data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            t=t_tensor,
            y=node_labels_placeholder,
            subject_mask=subject_mask,
            node_cmd_labels=node_cmd_labels,
            node_uuid=node_uuid,
            node_index_id=node_index_id,
            node_location=node_location,
        )
        graph_data_bidirectional: Data = self._make_graph_bidirectional(graph_data)
        logger.info(
            f"Graph processed with {graph_data_bidirectional.num_nodes} nodes, "
            f"{graph_data_bidirectional.num_edges} edges and "
            f"{graph_data_bidirectional.subject_mask.sum().item()} "
            f"subjects and will be saved under "
            f"{osp.join(self.processed_dir, self.file_name)}."
        )
        torch.save(
            self.collate([graph_data_bidirectional]),
            osp.join(self.processed_dir, self.file_name),
        )

    def _make_graph_bidirectional(self, data: Data) -> Data:
        assert data.edge_index is not None, "edge_index cannot be None"
        assert data.edge_attr is not None, "edge_attr cannot be None"
        edge_index: torch.Tensor = data.edge_index
        edge_attr: torch.Tensor = data.edge_attr

        # Create reverse edges
        reverse_edge_index = edge_index.flip(0)
        reverse_edge_attr = edge_attr.clone()

        # Combine original and reverse edges
        combined_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        combined_edge_attr = torch.cat([edge_attr, reverse_edge_attr], dim=0)

        data.edge_index = combined_edge_index
        data.edge_attr = combined_edge_attr
        data.t = data.t.repeat(2)

        return data

    def _get_timestamps(self, edges_df) -> torch.Tensor:
        return torch.tensor(edges_df["timestamp_rec"].tolist(), dtype=torch.float)

    def _build_node_location(self, nodes_df: pd.DataFrame) -> list[str]:
        # Create location for netflow nodes (dst_addr:dst_port)
        # netflow_location = (
        #     nodes_df[NodeColumns.DST_ADDR.value].fillna("").astype(str)
        #     + ":"
        #     + nodes_df[NodeColumns.DST_PORT.value].fillna("").astype(str)
        # )
        netflow_location = (
            nodes_df[NodeColumns.SRC_ADDR.value].fillna("").astype(str)
            + ":"
            + nodes_df[NodeColumns.DST_ADDR.value].fillna("").astype(str)
        )

        # Create location for file/subject nodes (path)
        # Handle case where label column is 'path' (no path available)
        if get_label_column_name_based_on_dataset() != NodeColumns.CMD.value:
            path_location = pd.Series([""] * len(nodes_df), dtype=str)
        else:
            path_location = nodes_df[NodeColumns.PATH.value].fillna("")
            path_location = path_location.astype(str)

        # Select based on node type using vectorized operation
        is_netflow = nodes_df[NodeColumns.TYPE.value] == NodeType.NETFLOW.value
        node_location = netflow_location.where(is_netflow, path_location)

        return node_location.tolist()

    def _build_subject_mask_and_node_cmd_labels(
        self, nodes_df: pd.DataFrame
    ) -> tuple[torch.Tensor, list[str]]:
        subject_mask: torch.Tensor = torch.tensor(
            nodes_df[NodeColumns.TYPE.value] == NodeType.SUBJECT.value,
            dtype=torch.bool,
        )

        label_column_name: str = get_label_column_name_based_on_dataset()
        node_cmd_labels: list[str] = nodes_df[label_column_name].tolist()
        node_cmd_labels = np.array(node_cmd_labels, dtype=object)  # type: ignore
        node_cmd_labels = np.where(pd.isna(node_cmd_labels), "<NONE>", node_cmd_labels).tolist()

        return subject_mask, node_cmd_labels

    def _transform_to_tensors(
        self, edge_index, edge_types, operations, node_types, node_type_indices
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_tensor = torch.nn.functional.one_hot(
            torch.tensor(edge_types, dtype=torch.long),
            num_classes=len(operations),
        ).float()
        x_tensor = torch.nn.functional.one_hot(
            torch.tensor(node_type_indices, dtype=torch.long),
            num_classes=len(node_types),
        ).float()
        return edge_index_tensor, edge_attr_tensor, x_tensor

    def _get_node_mapping(self, edges_df):
        src_col = (
            EdgeColumns.SRC_INDEX_ID.value
            if EdgeColumns.SRC_INDEX_ID.value in edges_df.columns
            else "source"
        )
        dst_col = (
            EdgeColumns.DST_INDEX_ID.value
            if EdgeColumns.DST_INDEX_ID.value in edges_df.columns
            else "target"
        )

        src_ids = edges_df[src_col].astype(str)
        dst_ids = edges_df[dst_col].astype(str)

        unique_index_ids = set(src_ids.unique()).union(dst_ids.unique())
        return {index_id: idx for idx, index_id in enumerate(sorted(unique_index_ids))}
