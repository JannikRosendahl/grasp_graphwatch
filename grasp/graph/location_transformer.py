import math
import re
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from grasp.config import LocationTransformerConfig

logger = logging.getLogger(__name__)


def _tokenize_location(location: str) -> list[str]:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", location.lower()) if t]
    return tokens


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = LocationTransformerConfig.POS_ENC_MAX_LEN,
    ):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (
                -torch.log(
                    torch.tensor(
                        LocationTransformerConfig.POS_ENC_DIV_TERM_HYPERPARAM
                    )
                )
                / d_model
            )
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as batch-first for simpler broadcasting
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        pe_buffer = getattr(self, "pe")
        return x + pe_buffer[:seq_len, :]


class LocationDataset(Dataset):
    def __init__(
        self,
        locations: list[str],
        max_length: int = LocationTransformerConfig.MAX_LOCATION_LENGTH,
        padding_idx: int = LocationTransformerConfig.PADDING_CHAR,
        chars=LocationTransformerConfig.CHARS,
        unique_locations: bool = True,
    ) -> None:
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        if unique_locations:
            self.locations: list[str] = list(set(locations))
        else:
            self.locations = locations

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        location = self.locations[idx][: self.max_length]
        idx_sequence = [self.char2idx.get(char, 0) for char in location]
        # Pad to max_length
        idx_sequence += [self.padding_idx] * (
            self.max_length - len(idx_sequence)
        )
        return torch.tensor(idx_sequence, dtype=torch.long)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embedding_dim, nhead, hidden_dim),
            num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.embedding_dim, nhead, hidden_dim),
            num_layers,
        )
        self.output_layer = nn.Linear(self.embedding_dim, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x) * (self.embedding_dim**0.5)
        encoded = self.pos_encoder(embed)
        memory = self.encoder(encoded)
        decoded = self.decoder(encoded, memory)
        return self.output_layer(decoded)

    def encode_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(sequences) * (self.embedding_dim**0.5)
        encoded = self.pos_encoder(embed)
        memory = self.encoder(encoded)
        return memory.mean(dim=1)

    def fit(
        self,
        dataloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        patience: int,
        lr: float = LocationTransformerConfig.LEARNING_RATE,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss, epochs_no_improve = float("inf"), 0

        logger.info(
            "Starting autoencoder fit for %d epochs (patience=%d) on %s",
            num_epochs,
            patience,
            device,
        )

        for epoch in range(num_epochs):
            self.train()
            epoch_loss, batch_count = 0.0, 0

            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = self.forward(batch)
                loss = criterion(
                    output.view(-1, self.input_dim), batch.view(-1)
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            logger.info(
                "Epoch %d/%d - avg loss %.4f",
                epoch + 1,
                num_epochs,
                avg_loss,
            )

            if avg_loss < best_loss:
                best_loss, epochs_no_improve = avg_loss, 0
            else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(
                        (
                            "Early stopping triggered after epoch %d "
                            "with best loss %.4f"
                        ),
                        epoch + 1,
                        best_loss,
                    )
                    break
        else:
            logger.info(
                "Completed all %d epochs with best loss %.4f",
                num_epochs,
                best_loss,
            )

    def save(self, path: Path) -> None:
        """Persist model weights to disk."""

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Path,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        nhead: int,
        device: torch.device,
    ) -> "TransformerAutoencoder":
        """Load model weights from disk."""

        resolved_device = device or torch.device(
            LocationTransformerConfig.DEVICE
        )
        model = cls(
            input_dim, embedding_dim, hidden_dim, num_layers, nhead
        ).to(resolved_device)
        state_dict = torch.load(path, map_location=resolved_device)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class Word2VecLocationEncoder:
    def __init__(
        self,
        embedding_dim: int,
        window_size: int,
        min_count: int,
        negative_samples: int,
        learning_rate: float,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.window_size = max(1, window_size)
        self.min_count = max(1, min_count)
        self.negative_samples = max(1, negative_samples)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.token_to_id: dict[str, int] = {"<unk>": 0}
        self.id_to_token: list[str] = ["<unk>"]
        self.input_embeddings: np.ndarray = np.zeros((1, embedding_dim))
        self.output_embeddings: np.ndarray = np.zeros((1, embedding_dim))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _build_vocab(self, locations: list[str]) -> list[list[int]]:
        tokenized = [_tokenize_location(loc) for loc in locations]
        token_counts: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        kept_tokens = sorted(
            [t for t, c in token_counts.items() if c >= self.min_count]
        )
        self.id_to_token = ["<unk>"] + kept_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        scale = 0.5 / max(1, self.embedding_dim)
        self.input_embeddings = self.rng.uniform(
            -scale, scale, size=(self.vocab_size, self.embedding_dim)
        )
        self.output_embeddings = np.zeros(
            (self.vocab_size, self.embedding_dim)
        )

        encoded_sequences: list[list[int]] = []
        for tokens in tokenized:
            ids = [self.token_to_id.get(t, 0) for t in tokens]
            if not ids:
                ids = [0]
            encoded_sequences.append(ids)
        return encoded_sequences

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _sgd_step(self, center_id: int, target_id: int, label: int) -> float:
        center_vec = self.input_embeddings[center_id]
        target_vec = self.output_embeddings[target_id]
        score = float(np.dot(center_vec, target_vec))
        pred = self._sigmoid(score)
        grad = pred - float(label)

        center_update = grad * target_vec
        target_update = grad * center_vec
        self.input_embeddings[center_id] -= self.learning_rate * center_update
        self.output_embeddings[target_id] -= self.learning_rate * target_update

        eps = 1e-8
        if label == 1:
            return -math.log(max(pred, eps))
        return -math.log(max(1.0 - pred, eps))

    def fit(self, locations: list[str], num_epochs: int) -> None:
        encoded_sequences = self._build_vocab(locations)
        logger.info(
            "Training Word2Vec location encoder (|paths|=%d, |vocab|=%d)",
            len(encoded_sequences),
            self.vocab_size,
        )

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pair_count = 0

            for sequence in encoded_sequences:
                seq_len = len(sequence)
                for i, center_id in enumerate(sequence):
                    left = max(0, i - self.window_size)
                    right = min(seq_len, i + self.window_size + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        context_id = sequence[j]
                        epoch_loss += self._sgd_step(center_id, context_id, 1)
                        pair_count += 1

                        for _ in range(self.negative_samples):
                            neg_id = int(self.rng.integers(1, self.vocab_size))
                            if neg_id == context_id:
                                continue
                            epoch_loss += self._sgd_step(center_id, neg_id, 0)

            avg_loss = epoch_loss / max(1, pair_count)
            logger.info(
                "Word2Vec epoch %d/%d - avg loss %.4f",
                epoch + 1,
                num_epochs,
                avg_loss,
            )

    def encode_locations(self, locations: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for location in locations:
            token_ids = [
                self.token_to_id.get(token, 0)
                for token in _tokenize_location(location)
            ]
            if not token_ids:
                token_ids = [0]

            token_vecs = self.input_embeddings[token_ids]
            vectors.append(token_vecs.mean(axis=0))
        return np.asarray(vectors, dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_dim": self.embedding_dim,
            "window_size": self.window_size,
            "min_count": self.min_count,
            "negative_samples": self.negative_samples,
            "learning_rate": self.learning_rate,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "input_embeddings": self.input_embeddings,
            "output_embeddings": self.output_embeddings,
        }
        torch.save(payload, path)


LocationEncoder = TransformerAutoencoder | Word2VecLocationEncoder


def _embed_locations(
    dataloader: DataLoader,
    model: TransformerAutoencoder,
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            # Use the model's embedding dimension for scaling
            embedded = model.embedding(batch) * (model.embedding_dim**0.5)
            encoded = model.pos_encoder(embedded)
            memory = model.encoder(encoded)
            all_embeddings.append(memory.mean(dim=1).cpu().numpy())

            if batch_idx % 50 == 0:  # Log progress every 50 batches
                logger.debug(
                    "Processed batch %d/%d for embedding extraction",
                    batch_idx + 1,
                    len(dataloader),
                )

    result = np.vstack(all_embeddings)
    logger.debug("Generated embeddings with shape: %s", result.shape)
    return result


def train_transformer_autoencoder_model(
    embedding_dim: int,
    hidden_dim: int,
    num_epochs: int,
    patience: int,
    batch_size: int,
    locations: list[str],
    model_file_path: Path,
    device: torch.device,
) -> TransformerAutoencoder:
    dataset = LocationDataset(locations)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim: int = len(LocationTransformerConfig.CHARS)

    logger.info(
        (
            "Training transformer autoencoder (|paths|=%d, "
            "embedding_dim=%d) on %s"
        ),
        len(dataset),
        embedding_dim,
        device,
    )

    model: TransformerAutoencoder = TransformerAutoencoder(
        input_dim, embedding_dim, hidden_dim
    ).to(device)
    model.fit(
        train_loader,
        device,
        num_epochs,
        patience,
    )

    model.save(model_file_path)
    logger.info("Saved transformer autoencoder weights to %s", model_file_path)
    return model


def train_word2vec_location_model(
    embedding_dim: int,
    num_epochs: int,
    locations: list[str],
    model_file_path: Path,
    window_size: int,
    min_count: int,
    negative_samples: int,
    learning_rate: float,
) -> Word2VecLocationEncoder:
    model = Word2VecLocationEncoder(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        negative_samples=negative_samples,
        learning_rate=learning_rate,
    )
    model.fit(locations=locations, num_epochs=num_epochs)
    model.save(model_file_path)
    logger.info("Saved Word2Vec location model to %s", model_file_path)
    return model


def train_location_encoder_model(
    model_type: str,
    embedding_dim: int,
    hidden_dim: int,
    num_epochs: int,
    patience: int,
    batch_size: int,
    locations: list[str],
    model_file_path: Path,
    device: torch.device,
    word2vec_window_size: int,
    word2vec_min_count: int,
    word2vec_negative_samples: int,
    word2vec_learning_rate: float,
) -> LocationEncoder:
    model_type_normalized = model_type.strip().lower()
    if model_type_normalized == "word2vec":
        return train_word2vec_location_model(
            embedding_dim=embedding_dim,
            num_epochs=num_epochs,
            locations=locations,
            model_file_path=model_file_path,
            window_size=word2vec_window_size,
            min_count=word2vec_min_count,
            negative_samples=word2vec_negative_samples,
            learning_rate=word2vec_learning_rate,
        )

    return train_transformer_autoencoder_model(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
        locations=locations,
        model_file_path=model_file_path,
        device=device,
    )


def build_location_embeddings_from_model(
    model: LocationEncoder,
    locations: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    if isinstance(model, Word2VecLocationEncoder):
        embeddings = model.encode_locations(locations)
        logger.info("Generated embeddings with shape: %s", embeddings.shape)
        return embeddings

    dataset = LocationDataset(locations, unique_locations=False)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings: np.ndarray = _embed_locations(dataloader, model, device)
    logger.info("Generated embeddings with shape: %s", embeddings.shape)  # type: ignore
    return embeddings


def build_location_embeddings_from_file(
    model_file_path: Path,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    nhead: int,
    paths: list[str],
    device: torch.device,
) -> np.ndarray:
    if not model_file_path.exists():
        raise FileNotFoundError(
            f"Model file {model_file_path} does not exist."
        )

    input_dim = len(LocationTransformerConfig.CHARS)
    model = TransformerAutoencoder.load(
        model_file_path,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        nhead,
        device=device,
    )
    logger.info(
        "Loaded transformer autoencoder weights from %s", model_file_path
    )
    return build_location_embeddings_from_model(
        model,
        paths,
        device,
    )


def train_and_save_embeddings(
    embedding_dim: int,
    hidden_dim: int,
    num_epochs: int,
    patience: int,
    batch_size: int,
    paths: list[str],
    model_file_path: Path,
    device: torch.device,
) -> TransformerAutoencoder:
    model: TransformerAutoencoder | None = None

    model = train_transformer_autoencoder_model(
        embedding_dim,
        hidden_dim,
        num_epochs,
        patience,
        batch_size,
        paths,
        model_file_path,
        device,
    )
    return model
