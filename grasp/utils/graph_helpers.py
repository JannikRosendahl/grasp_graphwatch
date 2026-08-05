import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from grasp import config
from grasp.graph import graph_storage
from grasp.schema import EdgeColumns, NodeColumns, NodeType

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    return config.DB_URL


def get_node_types() -> dict[int, str]:
    return {
        0: NodeType.SUBJECT.value,
        1: NodeType.FILE.value,
        2: NodeType.NETFLOW.value,
    }


def get_operations() -> dict[int, str]:
    dataset_name: str = config.DATASET_NAME
    experiment_name = config.EXPERIMENT_PREFIX
    if experiment_name.endswith("_modified"):
        return config.TC_OPERATIONS_MODIFIED
    if dataset_name.startswith("optc"):
        return config.OPTC_OPERATIONS
    if dataset_name.startswith("atlasv2_edr"):
        return config.ATLASV2_EDR_OPERATIONS
    if dataset_name.startswith("carbanakv2_edr"):
        return config.CARBANAKV2_EDR_OPERATIONS
    if dataset_name.startswith("spade_grasp"):
        return config.SPADE_GRASP_OPERATIONS
    if dataset_name.startswith("sysdig"):
        return config.SYSDIG_OPERATIONS
    return config.TC_OPERATIONS


def get_label_column_name_based_on_dataset() -> str:
    dataset_name: str = config.DATASET_NAME
    if dataset_name.startswith(("optc", "theia", "carbanak", "atlas")):
        return NodeColumns.PATH.value
    else:
        return NodeColumns.CMD.value


def clean_graph_attributes_for_neighborloader(
    graph: Data,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    extracted_attributes: dict[str, None | list[str]] = {
        "node_location": None,
        "node_uuid": None,
        "node_index_id": None,
        "node_cmd_labels": None,
    }

    for attr in extracted_attributes:
        if hasattr(graph, attr):
            extracted_attributes[attr] = getattr(graph, attr)
            delattr(graph, attr)
    return (
        extracted_attributes["node_uuid"],
        extracted_attributes["node_index_id"],
        extracted_attributes["node_cmd_labels"],
    )


def get_all_cmds_and_locations(
    path_storage: graph_storage.GraphStorage,
) -> None:
    for train_data_paths in [path_storage.train_data_paths]:
        for path in train_data_paths:
            data: Data = load_graph_data(path)
            subject_mask = data.subject_mask
            train_cmds = data.node_cmd_labels
            train_locations = data.node_location

            train_cmds = np.array(train_cmds, dtype=object)
            train_cmds = np.where(pd.isna(train_cmds), "<NONE>", train_cmds).tolist()

            path_storage.train_cmds.extend(train_cmds)
            path_storage.train_locations.extend(train_locations)
            subject_cmds = [cmd for cmd, is_subject in zip(train_cmds, subject_mask) if is_subject]
            subject_locations = [
                loc for loc, is_subject in zip(train_locations, subject_mask) if is_subject
            ]

            path_storage.train_subject_cmds.extend(subject_cmds)

            path_storage.train_subject_locations.extend(subject_locations)
    logger.info(f"Loaded {len(path_storage.train_cmds)} training commands.")
    logger.info(f"Loaded {len(path_storage.train_subject_locations)} training locations.")

    for test_data_paths in [path_storage.test_data_paths]:
        for path in test_data_paths:
            data: Data = load_graph_data(path)
            subject_mask = data.subject_mask
            test_cmds = data.node_cmd_labels
            test_locations = data.node_location

            test_cmds = np.array(path_storage.test_subject_cmds, dtype=object)
            test_cmds = np.where(pd.isna(test_cmds), "<NONE>", test_cmds).tolist()

            path_storage.test_cmds.extend(test_cmds)
            path_storage.test_locations.extend(test_locations)
            subject_cmds = [cmd for cmd, is_subject in zip(test_cmds, subject_mask) if is_subject]
            subject_locations = [
                loc for loc, is_subject in zip(test_locations, subject_mask) if is_subject
            ]

            path_storage.test_subject_cmds.extend(subject_cmds)
            path_storage.test_subject_locations.extend(subject_locations)

    logger.info(f"Loaded {len(path_storage.test_cmds)} testing commands.")
    logger.info(f"Loaded {len(path_storage.test_subject_locations)} testing locations.")


def create_train_cmd_mapping(path_storage: graph_storage.GraphStorage) -> None:
    unique_cmds = set(path_storage.train_subject_cmds)
    path_storage.train_subject_cmd_to_id = {
        cmd: idx for idx, cmd in enumerate(sorted(cmd for cmd in unique_cmds if cmd is not None))
    }
    logger.info(f"Unique training commands: {len(path_storage.train_subject_cmd_to_id)}")
    logger.debug(f"Training command to ID mapping:   {path_storage.train_subject_cmd_to_id}")


def create_extended_cmd_mapping(
    path_storage: graph_storage.GraphStorage,
) -> None:
    unique_cmds = set(path_storage.test_subject_cmds).union(set(path_storage.train_subject_cmds))
    path_storage.extended_subject_cmd_to_id = {
        cmd: idx
        for idx, cmd in enumerate(
            sorted(
                cmd
                for cmd in unique_cmds
                if cmd is not None and cmd not in path_storage.train_subject_cmd_to_id
            ),
            start=len(path_storage.train_subject_cmd_to_id),
        )
    }
    path_storage.combined_cmd_to_id = {
        **path_storage.train_subject_cmd_to_id,
        **path_storage.extended_subject_cmd_to_id,
    }


def create_combined_location_list(
    path_storage: graph_storage.GraphStorage,
) -> None:
    path_storage.combined_locations = list(
        set(path_storage.train_locations + path_storage.test_locations)
    )


def create_unique_locations_lists(
    path_storage: graph_storage.GraphStorage,
) -> None:
    path_storage.unique_train_locations = list(set(path_storage.train_locations))
    path_storage.unique_test_locations = list(set(path_storage.test_locations))
    logger.info(f"Unique training locations: {len(path_storage.unique_train_locations)}")
    logger.info(f"Unique testing locations: {len(path_storage.unique_test_locations)}")


def load_graph_data(path: str) -> Data:
    return torch.load(path, weights_only=False)[0]


def get_timestamp_column_name() -> str:
    return EdgeColumns.TIMESTAMP.value


def record_collection_to_df(records: Iterable) -> pd.DataFrame:
    return pd.DataFrame([row.as_dict() for row in records])


def generate_graph_basename(prefix: str, start_time: str, end_time: str) -> str:
    return f"{prefix}_graph_{start_time}_to_{end_time}"


def extract_graph_basename(
    graph_filename: str,
) -> str:
    return graph_filename.split(".")[0]


def generate_graph_extendedname(graph_basename: str, extension: str = "extended") -> str:
    return f"{graph_basename}_{extension}"
