import torch
import logging
from torch_geometric.data.data import Data
from torch_geometric.loader import NeighborLoader


from grasp.graph.graph_storage import GraphStorage
from grasp.detection.classification_storage import (
    ClassificationStorage,
)
from grasp.detection.classification import (
    GAT,
    train,
    collect_hard_example_node_ids,
    test,
)
from grasp.utils.graph_helpers import (
    load_graph_data,
    clean_graph_attributes_for_neighborloader,
)
from grasp.utils.detection_helpers import (
    extract_node_cmd_labels_uuids_and_ids,
)

logger: logging.Logger = logging.getLogger(__name__)


def _set_backbone_trainable(model: GAT, trainable: bool) -> None:
    for param in model.convs.parameters():
        param.requires_grad = trainable


def window_based_train(
    graph_storage: GraphStorage,
    batch_size: int,
    num_neighbors: list[int],
    epochs: int = 1,
    hidden_channels: int = 128,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    shuffle_train_paths: bool = False,
    shuffle_train_batches: bool = True,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    use_class_weights: bool = False,
    enable_subset_finetune: bool = False,
    subset_finetune_epochs: int = 1,
    subset_finetune_learning_rate: float | None = None,
    subset_finetune_lr_factor: float = 0.1,
    subset_finetune_freeze_backbone: bool = True,
) -> GAT:
    first_train_window = graph_storage.extended_train_data_paths[0]
    first_train_data: Data = load_graph_data(first_train_window)
    in_channels: int = first_train_data.x.shape[1]  # type: ignore
    num_classes: int = first_train_data.y.shape[-1]  # type: ignore
    edge_dim: int = first_train_data.edge_attr.shape[1]  # type: ignore

    # Compute class weights if enabled
    class_weights = None
    if use_class_weights:
        from grasp.utils.detection_helpers import compute_class_weights

        class_weights = compute_class_weights(first_train_data, num_classes)
        logger.info(f"Class weights: {class_weights}")

    model = GAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        heads=heads,
        edge_dim=edge_dim,
        dropout=dropout,
        device=device,
        use_class_weights=use_class_weights,
        class_weights=class_weights,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    if shuffle_train_paths:
        train_paths: list[str] = graph_storage.generate_shuffled_train_paths()
    else:
        train_paths = graph_storage.extended_train_data_paths

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_loss: float = 0.0
        for train_window in train_paths:
            logging.info(
                f"processing training window: {train_window}. "
                f"Which is {train_paths.index(train_window) + 1}/{len(train_paths)}"
            )
            train_data: Data = load_graph_data(train_window)
            train_node_uuids, train_node_index_ids, train_node_cmd_labels = (
                clean_graph_attributes_for_neighborloader(train_data)
            )

            train_loader = NeighborLoader(
                train_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_data.subject_mask,
                shuffle=shuffle_train_batches,
            )

            loss: float = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            epoch_loss += loss
            scheduler.step(loss)
            logger.info(
                (
                    f"Epoch {epoch + 1}, Train Window {train_window}, "
                    f"Loss: {loss:.4f}"
                )
            )
        avg_epoch_loss = epoch_loss / len(train_paths)
        logger.info(
            f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}"
        )
    logger.info("Base training completed.")

    if not enable_subset_finetune:
        logger.info("Training completed.")
        return model

    if subset_finetune_freeze_backbone:
        _set_backbone_trainable(model, trainable=False)
        logger.info(
            (
                "Subset fine-tuning: froze GAT backbone layers and "
                "training mostly head."
            )
        )

    finetune_lr = (
        subset_finetune_learning_rate
        if subset_finetune_learning_rate is not None
        else learning_rate * subset_finetune_lr_factor
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning(
            (
                "No trainable parameters available for subset "
                "fine-tuning; skipping."
            )
        )
        if subset_finetune_freeze_backbone:
            _set_backbone_trainable(model, trainable=True)
        logger.info("Training completed.")
        return model

    finetune_optimizer = torch.optim.Adam(
        trainable_params,
        lr=finetune_lr,
        weight_decay=weight_decay,
    )

    for finetune_epoch in range(subset_finetune_epochs):
        logger.info(
            "Starting subset fine-tuning epoch %d/%d with lr=%.6f",
            finetune_epoch + 1,
            subset_finetune_epochs,
            finetune_lr,
        )
        epoch_loss = 0.0
        contributing_windows = 0

        for train_window in train_paths:
            train_data = load_graph_data(train_window)
            clean_graph_attributes_for_neighborloader(train_data)

            selection_loader = NeighborLoader(
                train_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_data.subject_mask,
                shuffle=False,
            )
            selected_ids = collect_hard_example_node_ids(
                model=model,
                data_loader=selection_loader,
                device=device,
            )
            if not selected_ids:
                continue

            finetune_loader = NeighborLoader(
                train_data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_data.subject_mask,
                shuffle=shuffle_train_batches,
            )
            selected_ids_tensor = torch.tensor(
                sorted(selected_ids),
                dtype=torch.long,
                device=device,
            )
            loss = train(
                model=model,
                train_loader=finetune_loader,
                optimizer=finetune_optimizer,
                device=device,
                selected_input_node_ids=selected_ids_tensor,
            )
            if loss == 0.0:
                continue

            epoch_loss += loss
            contributing_windows += 1
            logger.info(
                "Subset fine-tune epoch %d, window %s, selected=%d, loss=%.4f",
                finetune_epoch + 1,
                train_window,
                len(selected_ids),
                loss,
            )

        if contributing_windows == 0:
            logger.warning(
                "Subset fine-tune epoch %d had no selected samples.",
                finetune_epoch + 1,
            )
        else:
            logger.info(
                "Subset fine-tune epoch %d completed. Average loss=%.4f",
                finetune_epoch + 1,
                epoch_loss / contributing_windows,
            )

    if subset_finetune_freeze_backbone:
        _set_backbone_trainable(model, trainable=True)

    logger.info("Training completed.")
    return model


def window_based_test(
    model: GAT,
    graph_storage: GraphStorage,
    batch_size: int,
    num_neighbors: list[int],
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    for_train_data: bool = False,
) -> ClassificationStorage:
    cls_storage = ClassificationStorage()

    if for_train_data:
        data_paths: list[str] = graph_storage.extended_train_data_paths
    else:
        data_paths: list[str] = graph_storage.extended_test_data_paths

    for test_window in data_paths:
        logging.info(
            f"processing test window: {test_window}. "
            f"Which is {data_paths.index(test_window) + 1}/{len(data_paths)}"
        )
        test_data: Data = load_graph_data(test_window)
        test_node_uuids, test_node_index_ids, test_node_cmd_labels = (
            clean_graph_attributes_for_neighborloader(test_data)
        )

        extract_node_cmd_labels_uuids_and_ids(
            cls_storage=cls_storage,
            node_cmd_labels=test_node_cmd_labels,  # type: ignore
            node_uuids=test_node_uuids,  # type: ignore
            node_ids=test_node_index_ids,  # type: ignore
            window_path=test_window,
            subject_mask=test_data.subject_mask,
        )
        test_loader = NeighborLoader(
            test_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=test_data.subject_mask,
            shuffle=False,
        )

        accuracy: float = test(
            model=model,
            test_loader=test_loader,
            device=device,
            classification_storage=cls_storage,
        )
        logger.info(f"Test Window {test_window}, Accuracy: {accuracy:.4f}")

    logger.info("Testing completed.")
    return cls_storage
