from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FEDERATED_DIR = DATA_DIR / "federated_clients"
RESULTS_DIR = ROOT_DIR / "results"


@dataclass(slots=True)
class TrainingConfig:
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 15
    weight_decay: float = 1e-4
    dropout: float = 0.0
    seed: int = 42


@dataclass(slots=True)
class FederatedConfig:
    rounds: int = 10
    local_epochs: int = 2
    client_fraction: float = 1.0
    seed: int = 42
