import logging
from pathlib import Path

import torch

from grasp.graph.extended_graph import ExtendedGraspGraph
from grasp.graph.graph import GraspGraph
from grasp.graph.graph_storage import GraphStorage
from grasp.graph.location_transformer import (
    train_transformer_autoencoder_model,
    train_word2vec_location_model,
)
from grasp.graph.window_management import create_train_and_test_windows
from grasp.utils import graph_helpers
from grasp.utils.time_helpers import datetime_to_ns_time_US

logger = logging.getLogger(__name__)


class GraphManager:
    def __init__(
        self,
        dataset_name: str,
        train_start_times: list[str],
        train_end_times: list[str],
        test_start_times: list[str],
        test_end_times: list[str],
        context_size: int,
        step_size: int,
        experiment_prefix: str = "default",
        force_reload: bool = False,
        root_dir: str = "./data",
        autoencoder_embedding_dim: int = 8,
        autoencoder_hidden_dim: int = 16,
        autoencoder_num_epochs: int = 20,
        autoencoder_patience: int = 5,
        autoencoder_batch_size: int = 32,
        autoencoder_model_file_path: str = "./models/location_autoencoder.pt",
        location_embedding_model_type: str = "transformer",
        word2vec_window_size: int = 2,
        word2vec_min_count: int = 1,
        word2vec_negative_samples: int = 5,
        word2vec_learning_rate: float = 0.025,
        autoencoder_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.dataset_name: str = dataset_name
        self.train_start_times: list[str] = train_start_times
        self.train_end_times: list[str] = train_end_times
        self.test_start_times: list[str] = test_start_times
        self.test_end_times: list[str] = test_end_times
        self.context_size: int = context_size
        self.step_size: int = step_size

        self.train_windows: list[tuple[str, str]] = []
        self.test_windows: list[tuple[str, str]] = []

        # self.experiment_prefix_train: str = experiment_prefix + "_train"
        # self.experiment_prefix_test: str = experiment_prefix + "_test"

        self.experiment_prefix_train: str = dataset_name + "_train"
        self.experiment_prefix_test: str = dataset_name + "_test"

        self.force_reload: bool = force_reload
        self.graphs_root_dir: str = root_dir
        self.extended_graphs_root_dir: str = str(Path(root_dir) / "extended_graphs")

        self.graph_storage = GraphStorage()

        if self.dataset_name in ["carbanakv2_edr", "atlasv2_edr", "spade_grasp", "sysdig"]:
            self.utc: bool = True
        else:
            self.utc: bool = False

        self.embedding_dim = autoencoder_embedding_dim
        self.hidden_dim = autoencoder_hidden_dim
        self.num_epochs = autoencoder_num_epochs
        self.patience = autoencoder_patience
        self.batch_size = autoencoder_batch_size
        self.model_file_path = autoencoder_model_file_path
        self.location_embedding_model_type = location_embedding_model_type
        self.word2vec_window_size = word2vec_window_size
        self.word2vec_min_count = word2vec_min_count
        self.word2vec_negative_samples = word2vec_negative_samples
        self.word2vec_learning_rate = word2vec_learning_rate
        self.autoencoder_device: torch.device = torch.device(autoencoder_device)

    def _setup_windows(self) -> None:
        self.train_windows, self.test_windows = create_train_and_test_windows(
            train_start_times=self.train_start_times,
            train_end_times=self.train_end_times,
            test_start_times=self.test_start_times,
            test_end_times=self.test_end_times,
            context_size=self.context_size,
            step_size=self.step_size,
        )

    def run_full_workflow(self) -> None:
        self._setup_windows()
        logger.info("Graph windows have been set up successfully")
        logger.debug(f"Train windows: {self.train_windows}, Test windows: {self.test_windows}")

        for train_window in self.train_windows:
            logger.debug(
                "Processing train window: %s to %s",
                train_window[0],
                train_window[1],
            )
            logger.info(
                "Progress %d/%d",
                self.train_windows.index(train_window) + 1,
                len(self.train_windows),
            )
            gg = GraspGraph(
                root=self.graphs_root_dir,
                start_time=str(datetime_to_ns_time_US(train_window[0], self.utc)),
                end_time=str(datetime_to_ns_time_US(train_window[1], self.utc)),
                experiment_prefix=self.experiment_prefix_train,
                force_reload=self.force_reload,
            )
            graph_path = Path(gg.processed_dir) / gg.processed_file_names[0]
            if graph_path.exists():
                self.graph_storage.train_data_paths.append(str(graph_path))

        for test_window in self.test_windows:
            logger.debug(
                "Processing test window: %s to %s",
                test_window[0],
                test_window[1],
            )
            logger.info(
                "Progress %d/%d",
                self.test_windows.index(test_window) + 1,
                len(self.test_windows),
            )
            gg = GraspGraph(
                root=self.graphs_root_dir,
                start_time=str(datetime_to_ns_time_US(test_window[0], self.utc)),
                end_time=str(datetime_to_ns_time_US(test_window[1], self.utc)),
                experiment_prefix=self.experiment_prefix_test,
                force_reload=self.force_reload,
            )
            graph_path = Path(gg.processed_dir) / gg.processed_file_names[0]
            if graph_path.exists():
                self.graph_storage.test_data_paths.append(str(graph_path))

        graph_helpers.get_all_cmds_and_locations(
            self.graph_storage,
        )
        graph_helpers.create_unique_locations_lists(self.graph_storage)
        graph_helpers.create_train_cmd_mapping(self.graph_storage)

        if self.location_embedding_model_type.strip().lower() == "word2vec":
            autoencoder = train_word2vec_location_model(
                embedding_dim=self.embedding_dim,
                num_epochs=self.num_epochs,
                locations=self.graph_storage.unique_train_locations,
                model_file_path=Path(self.model_file_path),
                window_size=self.word2vec_window_size,
                min_count=self.word2vec_min_count,
                negative_samples=self.word2vec_negative_samples,
                learning_rate=self.word2vec_learning_rate,
            )
        else:
            autoencoder = train_transformer_autoencoder_model(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_epochs=self.num_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                locations=self.graph_storage.unique_train_locations,
                device=self.autoencoder_device,
                model_file_path=Path(self.model_file_path),
            )

        for path in self.graph_storage.train_data_paths:
            logger.info(f"Creating extended graph for training data: {path}")
            logger.info(
                f"Progress "
                f"{self.graph_storage.train_data_paths.index(path) + 1}"
                f"/{len(self.graph_storage.train_data_paths)}"
            )
            e_gg = ExtendedGraspGraph(
                raw_grasp_graph_path=path,
                location_autoencoder=autoencoder,
                root=self.extended_graphs_root_dir,
                train_cmd_to_id=self.graph_storage.train_subject_cmd_to_id,
                experiment_prefix=self.experiment_prefix_train,
                force_reload=self.force_reload,
                autoencoder_device=self.autoencoder_device,
            )
            e_graph_path = Path(e_gg.processed_dir) / e_gg.processed_file_names[0]
            self.graph_storage.extended_train_data_paths.append(str(e_graph_path))

        for path in self.graph_storage.test_data_paths:
            logger.info(f"Creating extended graph for testing data: {path}")
            logger.info(
                f"Progress "
                f"{self.graph_storage.test_data_paths.index(path) + 1}"
                f"/{len(self.graph_storage.test_data_paths)}"
            )
            e_gg = ExtendedGraspGraph(
                raw_grasp_graph_path=path,
                location_autoencoder=autoencoder,
                root=self.extended_graphs_root_dir,
                train_cmd_to_id=self.graph_storage.train_subject_cmd_to_id,
                experiment_prefix=self.experiment_prefix_test,
                force_reload=self.force_reload,
                autoencoder_device=self.autoencoder_device,
            )
            e_graph_path = Path(e_gg.processed_dir) / e_gg.processed_file_names[0]
            self.graph_storage.extended_test_data_paths.append(str(e_graph_path))
        logger.info("Extended graphs have been created successfully")

    def save_graph_storage(self, file_path: str = "./graph_storage.pt") -> None:
        self.graph_storage.save_graph_storage(file_path=file_path)
        logger.info(f"Graph storage saved to {file_path}")
