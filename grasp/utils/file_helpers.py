import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def clean_filename(filename: str) -> str:
    invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def generate_experiment_file_prefix(base_name: str, params: dict[str, str]) -> str:
    param_str = "_".join(f"{key}-{value}" for key, value in params.items())
    filename = f"{base_name}_{param_str}"
    logger.info(f"Generated experiment filename: {filename}")
    return clean_filename(filename)


def generate_storage_paths(
    experiment_prefix: str,
    base_dir: str | Path = Path("./data"),
    create_dirs: bool = True,
) -> dict[str, Path]:
    base = Path(base_dir)
    paths: dict[str, Path] = {
        "graph_storage": base / "graph_storage" / f"{experiment_prefix}_graph_storage.pt",
        "classification_train": base
        / "classification_storage"
        / f"{experiment_prefix}_train_cls_storage.pt",
        "classification_test": base
        / "classification_storage"
        / f"{experiment_prefix}_test_cls_storage.pt",
        "location_autoencoder": base / "models" / f"{experiment_prefix}_location_autoencoder.pt",
        "classification_model": base / "models" / f"{experiment_prefix}_classification_model.pt",
        "cluster_manager": base / "models" / f"{experiment_prefix}_cluster_manager.pt",
        "evaluation_storage": base
        / "evaluation_storage"
        / f"{experiment_prefix}_evaluation_storage.pt",
        "ground_truth_storage": base
        / "ground_truth_storage"
        / f"{experiment_prefix}_ground_truth_storage.pt",
        "report": base / "reports" / f"{experiment_prefix}_report.json",
        "detailed_report": base / "reports" / f"{experiment_prefix}_detailed_report.json",
        "visualizations_dir": base / "reports" / "visualizations" / experiment_prefix,
    }

    if create_dirs:
        for path in paths.values():
            target_dir = path if path.suffix == "" else path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Storage paths generated for %s: %s", experiment_prefix, paths)
    return paths
