import random
import torch
import json
from pathlib import Path


class GraphStorage:
    def __init__(
        self,
        train_data_paths: list[str] | None = None,
        test_data_paths: list[str] | None = None,
    ) -> None:
        self.train_data_paths: list[str] = (
            train_data_paths if train_data_paths is not None else []
        )
        self.test_data_paths: list[str] = (
            test_data_paths if test_data_paths is not None else []
        )
        self.extended_train_data_paths: list[str] = []
        self.extended_test_data_paths: list[str] = []
        self.train_subject_cmds: list[str] = []
        self.train_subject_locations: list[str] = []
        self.train_cmds: list[str] = []
        self.train_locations: list[str] = []
        self.unique_train_locations: list[str] = []
        self.train_subject_cmd_to_id: dict[str, int] = {}

        self.test_subject_cmds: list[str] = []
        self.test_subject_locations: list[str] = []
        self.test_cmds: list[str] = []
        self.test_locations: list[str] = []
        self.unique_test_locations: list[str] = []
        self.extended_subject_cmd_to_id: dict[str, int] = {}
        self.extended_locations: list[str] = []

        self.combined_cmd_to_id: dict[str, int] = {}
        self.combined_locations: list[str] = []

    def save_graph_storage(
        self, file_path: str = "./graph_storage.pt"
    ) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, file_path)

    def generate_shuffled_train_paths(self) -> list[str]:
        shuffled_paths = self.extended_train_data_paths.copy()
        random.shuffle(shuffled_paths)
        return shuffled_paths


def load_graph_storage(
    file_path: str = "./graph_storage.pt",
) -> "GraphStorage":
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GraphStorage file not found: {file_path}")

    obj = torch.load(path, weights_only=False)
    if not isinstance(obj, GraphStorage):
        raise TypeError(
            f"Expected GraphStorage, but loaded object of type {type(obj)}"
        )

    return obj
