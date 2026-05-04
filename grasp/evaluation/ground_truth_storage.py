from pathlib import Path
import logging
import pandas as pd
import torch


logger = logging.getLogger(__name__)


class GroundTruthStorage:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.files: dict[str, pd.DataFrame] = {}
        self._load_files()

    def _load_files(self) -> None:
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Ground truth path does not exist: {self.base_path}"
            )
        if not self.base_path.is_dir():
            raise NotADirectoryError(
                f"Ground truth path is not a directory: {self.base_path}"
            )

        for csv_path in sorted(self.base_path.rglob("*.csv")):
            rel_path = str(csv_path.relative_to(self.base_path))
            try:
                rows = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        uuid, rest = line.split(",", 1)  # split at first comma
                        label, node_id = rest.rsplit(
                            ",", 1
                        )  # split at last comma
                        rows.append((uuid, label, node_id))

                df = pd.DataFrame(rows, columns=["uuid", "label", "node_id"])
                df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
            except Exception:
                logger.exception(
                    "Failed to load ground truth file %s", csv_path
                )
                raise

            self.files[rel_path] = df
            logger.info("Loaded ground truth file %s", rel_path)

    def get(self, relative_path: str) -> pd.DataFrame:
        return self.files[relative_path]

    def keys(self) -> list[str]:
        return list(self.files.keys())

    def combined(self, grasp: bool = True) -> pd.DataFrame:
        if not self.files:
            return pd.DataFrame(columns=["uuid", "label", "node_id", "source"])

        frames: list[pd.DataFrame] = []
        for rel_path, df in self.files.items():
            if grasp and "grasp" not in rel_path:
                continue
            if not grasp and (
                "grasp" in rel_path or "unknown_exec" in rel_path
            ):
                continue

            frame = df.copy()
            frame["source"] = rel_path
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["uuid", "label", "node_id", "source"])

        return pd.concat(frames, ignore_index=True)

    def save_to_file(self, path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.files, output_path)
        logger.info("Saved ground truth file %s", output_path)
