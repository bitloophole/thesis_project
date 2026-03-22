from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMN = "Label"


@dataclass(slots=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_csv_frame(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows)


def load_csv_frame_random_sample(
    path: Path,
    max_rows: int,
    seed: int = 42,
    chunksize: int = 50000,
) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    rng = np.random.default_rng(seed)
    sampled_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        sampled_chunks.append(chunk)
        merged = pd.concat(sampled_chunks, ignore_index=True)
        if len(merged) > max_rows:
            merged = merged.nsmallest(max_rows, "_sample_key")
        sampled_chunks = [merged]

    if not sampled_chunks:
        return pd.DataFrame()

    result = sampled_chunks[0].drop(columns=["_sample_key"]).reset_index(drop=True)
    return result


def split_features_labels(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> tuple[np.ndarray, np.ndarray]:
    feature_df = df.drop(columns=[label_column])
    labels = df[label_column].astype(int).to_numpy()
    features = feature_df.astype(np.float32).to_numpy()
    return features, labels


def train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return DatasetSplit(
        x_train=features[train_idx],
        y_train=labels[train_idx],
        x_val=features[val_idx],
        y_val=labels[val_idx],
        x_test=features[test_idx],
        y_test=labels[test_idx],
    )


def train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:]

    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


def stratified_train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        train_end = int(len(label_indices) * train_ratio)
        val_end = train_end + int(len(label_indices) * val_ratio)

        train_parts.append(label_indices[:train_end])
        val_parts.append(label_indices[train_end:val_end])
        test_parts.append(label_indices[val_end:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return DatasetSplit(
        x_train=features[train_idx],
        y_train=labels[train_idx],
        x_val=features[val_idx],
        y_val=labels[val_idx],
        x_test=features[test_idx],
        y_test=labels[test_idx],
    )


def stratified_train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        train_end = int(len(label_indices) * train_ratio)
        train_parts.append(label_indices[:train_end])
        val_parts.append(label_indices[train_end:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


def build_test_split_from_global(
    global_path: Path,
    max_rows: int | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    random_sample: bool = True,
    stratified: bool = True,
) -> DatasetSplit:
    if max_rows is not None and random_sample:
        df = load_csv_frame_random_sample(global_path, max_rows=max_rows, seed=seed)
    else:
        df = load_csv_frame(global_path, max_rows=max_rows)
    x, y = split_features_labels(df)
    if stratified:
        return stratified_train_val_test_split(x, y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return train_val_test_split(x, y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def load_client_frames(
    client_dir: Path,
    max_rows_per_client: int | None = None,
    random_sample: bool = True,
    seed: int = 42,
) -> list[tuple[str, pd.DataFrame]]:
    client_paths = sorted(client_dir.glob("client_*.csv"))
    if not client_paths:
        raise FileNotFoundError(f"No client files found in {client_dir}")

    clients: list[tuple[str, pd.DataFrame]] = []
    for client_index, path in enumerate(client_paths):
        if max_rows_per_client is not None and random_sample:
            frame = load_csv_frame_random_sample(path, max_rows=max_rows_per_client, seed=seed + client_index)
        else:
            frame = load_csv_frame(path, max_rows=max_rows_per_client)
        clients.append((path.stem, frame))
    return clients


def infer_num_classes(labels: Iterable[int]) -> int:
    label_array = np.asarray(list(labels), dtype=int)
    return int(label_array.max()) + 1


def label_distribution(labels: np.ndarray, num_classes: int) -> dict[str, int]:
    counts = np.bincount(labels, minlength=num_classes)
    return {str(class_idx): int(count) for class_idx, count in enumerate(counts)}


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    weights = len(labels) / (num_classes * counts)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def oversample_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    seed: int = 42,
    target_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=num_classes)
    max_count = int(counts.max())
    target_count = max(1, int(max_count * target_fraction))

    sampled_features: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) == 0:
            continue
        replace = len(class_indices) < target_count
        chosen_indices = rng.choice(class_indices, size=target_count, replace=replace)
        sampled_features.append(features[chosen_indices])
        sampled_labels.append(labels[chosen_indices])

    oversampled_x = np.concatenate(sampled_features, axis=0)
    oversampled_y = np.concatenate(sampled_labels, axis=0)
    permutation = rng.permutation(len(oversampled_y))
    return oversampled_x[permutation], oversampled_y[permutation]


@dataclass(slots=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def fit_standardizer(x_train: np.ndarray) -> Standardizer:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Standardizer(mean=mean, std=std)


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
