# ============================================================
# Federated IDS Binary Classifier with TabTransformer
# Non-IID + Clean Round Logs + Final Summary + Confusion Matrix
# ============================================================

import logging
import io
import importlib.util
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple 

import flwr as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from flwr.common import Context, Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# Reduce Flower log noise
# ------------------------------
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("flwr.server").setLevel(logging.WARNING)
logging.getLogger("flwr.client").setLevel(logging.WARNING)

# ------------------------------
# Global config
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5
QUANTIZATION_MODE = "dynamic_int8"

# Globals initialized in main
global_clients = None
global_input_dim = None
global_model_stats = None


def format_bytes(size_in_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_in_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_in_bytes:.2f} B"


# ============================================================
# 1. LOAD + PREPROCESS DATA
# ============================================================

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("data/iot_dataset_undersampled_mapped1.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    original_attack_categories = df["Attack_Category"].copy()

    df["Attack_Category"] = df["Attack_Category"].apply(
        lambda value: 0 if value == "BENIGN" else 1
    )

    df_benign = df[df["Attack_Category"] == 0]
    df_attack = df[df["Attack_Category"] == 1]

    df_attack_undersampled = resample(
        df_attack,
        replace=False,
        n_samples=len(df_benign),
        random_state=RANDOM_STATE,
    )

    df_balanced = pd.concat([df_attack_undersampled, df_benign])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE)

    attack_type_counts = (
        original_attack_categories.loc[df_attack_undersampled.index]
        .value_counts()
        .sort_index()
    )

    print("\n===== ATTACK TYPES IN CLASS 1 AFTER UNDERSAMPLING =====")
    print(attack_type_counts)
    print("=======================================================")

    print("\n================ DATA SUMMARY ================")
    print(df_balanced["Attack_Category"].value_counts())

    X = df_balanced.drop(["Label", "Attack_Category"], axis=1).values
    y = df_balanced["Attack_Category"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# ============================================================
# 2. NON-IID CLIENT SPLIT + LOCAL TEST
# ============================================================

def create_clients(
    X: np.ndarray, y: np.ndarray, num_clients: int = NUM_CLIENTS
) -> List[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    clients = []
    rng = np.random.default_rng(RANDOM_STATE)

    idx_benign = np.where(y == 0)[0]
    idx_attack = np.where(y == 1)[0]

    rng.shuffle(idx_benign)
    rng.shuffle(idx_attack)

    benign_splits = np.array_split(idx_benign, num_clients)
    attack_splits = np.array_split(idx_attack, num_clients)

    print("\n============== CLIENT DISTRIBUTION ==============")

    for client_idx in range(num_clients):
        if client_idx % 2 == 0:
            idx = np.concatenate(
                [
                    benign_splits[client_idx],
                    attack_splits[client_idx][: len(attack_splits[client_idx]) // 3],
                ]
            )
        else:
            idx = np.concatenate(
                [
                    benign_splits[client_idx][: len(benign_splits[client_idx]) // 3],
                    attack_splits[client_idx],
                ]
            )

        rng.shuffle(idx)

        X_client = X[idx]
        y_client = y[idx]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_client,
            y_client,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_client,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=y_train_full,
        )

        benign_count = int((y_client == 0).sum())
        attack_count = int((y_client == 1).sum())

        print(
            f"Client {client_idx}: total={len(y_client)} | "
            f"benign={benign_count} | attack={attack_count} | "
            f"train={len(y_train)} | val={len(y_val)} | test={len(y_test)}"
        )

        clients.append((X_train, X_val, X_test, y_train, y_val, y_test))

    print("=================================================\n")
    return clients


# ============================================================
# 3. MODEL
# ============================================================

class TabTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.feature_value_proj = nn.Linear(1, embed_dim)
        self.feature_position_embed = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.feature_value_proj(x)
        x = x + self.feature_position_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def get_model_stats(model: nn.Module, input_dim: int) -> Dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    embed_dim = 32
    num_heads = 4
    num_layers = 2

    value_proj_flops = input_dim * ((2 * 1 * embed_dim) + embed_dim)
    attention_qkv_flops = input_dim * 3 * (2 * embed_dim * embed_dim)
    attention_scores_flops = num_heads * (2 * input_dim * input_dim * (embed_dim // num_heads))
    attention_out_flops = input_dim * (2 * embed_dim * embed_dim)
    ff_flops = input_dim * ((2 * embed_dim * (embed_dim * 4)) + (2 * (embed_dim * 4) * embed_dim))
    norm_relu_flops = input_dim * embed_dim * 6
    transformer_flops = num_layers * (
        attention_qkv_flops
        + attention_scores_flops
        + attention_out_flops
        + ff_flops
        + norm_relu_flops
    )
    head_flops = (2 * embed_dim * 64) + 64 + (2 * 64 * 1) + 1

    forward_flops = value_proj_flops + transformer_flops + head_flops
    training_flops = forward_flops * 3

    return {
        "parameters": float(total_params),
        "model_size_bytes": float(total_bytes),
        "forward_flops_per_sample": float(forward_flops),
        "training_flops_per_sample": float(training_flops),
    }


def get_parameter_bytes(parameters: List[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in parameters))


def get_serialized_model_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def get_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_dynamic_quantized_tabtransformer(model: nn.Module) -> nn.Module:
    supported_engines = [str(engine) for engine in torch.backends.quantized.supported_engines]
    if not supported_engines:
        raise RuntimeError("No PyTorch quantization engine is available on this runtime.")

    previous_engine = str(torch.backends.quantized.engine)
    preferred_engine = "qnnpack" if "qnnpack" in supported_engines else supported_engines[0]

    try:
        torch.backends.quantized.engine = preferred_engine
        quantized_model = deepcopy(model).to("cpu").eval()
        quantized_model.feature_value_proj = torch.ao.quantization.quantize_dynamic(
            quantized_model.feature_value_proj,
            {nn.Linear},
            dtype=torch.qint8,
        )
        quantized_model.classifier = torch.ao.quantization.quantize_dynamic(
            quantized_model.classifier,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model
    finally:
        if previous_engine in supported_engines:
            torch.backends.quantized.engine = previous_engine


def get_quantization_label() -> str:
    if QUANTIZATION_MODE == "dynamic_int8":
        return "Dynamic INT8 (projection + classifier Linear layers)"
    return "Disabled"


# ============================================================
# 4. TRAIN + TEST
# ============================================================

def train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
) -> Dict[str, float]:
    model.train()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()
    train_losses: List[float] = []
    val_losses: List[float] = []

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    start_time = time.perf_counter()

    for _ in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_X.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_losses.append(running_loss / max(total_samples, 1))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        val_losses.append(float(val_loss))
        model.train()

    training_time = time.perf_counter() - start_time
    return {
        "train_time_sec": float(training_time),
        "train_loss": float(train_losses[-1]),
        "val_loss": float(val_losses[-1]),
    }


def test(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model_to_eval = model
    eval_device = DEVICE
    dynamic_quantized_model_size_bytes = 0.0
    dynamic_quantized_parameter_count = 0.0
    quantization_applied = 0.0

    if QUANTIZATION_MODE == "dynamic_int8":
        supported_engines = [str(engine) for engine in torch.backends.quantized.supported_engines]
        if supported_engines:
            try:
                model_to_eval = build_dynamic_quantized_tabtransformer(model)
                eval_device = torch.device("cpu")
                dynamic_quantized_model_size_bytes = float(
                    get_serialized_model_size_bytes(model_to_eval)
                )
                dynamic_quantized_parameter_count = float(get_parameter_count(model_to_eval))
                quantization_applied = 1.0
            except Exception:
                model_to_eval = deepcopy(model).to("cpu").eval()
                eval_device = torch.device("cpu")
        else:
            model_to_eval = deepcopy(model).to("cpu").eval()
            eval_device = torch.device("cpu")

    model_to_eval.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(eval_device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model_to_eval(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    inference_time = time.perf_counter() - start_time

    preds = (probs >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1_score": float(f1_score(y, preds, zero_division=0)),
        "inference_time_sec": float(inference_time),
        "dynamic_quantized_model_size_bytes": float(dynamic_quantized_model_size_bytes),
        "dynamic_quantized_parameter_count": float(dynamic_quantized_parameter_count),
        "quantization_applied": float(quantization_applied),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y, probs))
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


# ============================================================
# 5. METRIC AGGREGATION
# ============================================================

def weighted_average(metrics: List[Tuple[int, Metrics]], keys: List[str]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for key in keys:
        valid = []
        for num_examples, metric in metrics:
            value = metric.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)) and not np.isnan(value):
                valid.append((num_examples, float(value)))

        if valid:
            aggregated[key] = sum(n * v for n, v in valid) / sum(n for n, _ in valid)

    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    scalar_keys = [
        "train_time_sec",
        "model_size_bytes",
        "parameter_count",
        "train_loss",
        "val_loss",
    ]
    aggregated = weighted_average(metrics, scalar_keys)

    aggregated["total_train_examples"] = float(sum(num for num, _ in metrics))
    aggregated["communication_bytes"] = float(
        sum(metric.get("communication_bytes", 0.0) for _, metric in metrics)
    )
    aggregated["training_flops"] = float(
        sum(metric.get("training_flops", 0.0) for _, metric in metrics)
    )

    aggregated["parameter_count"] = float(metrics[0][1].get("parameter_count", 0.0))
    aggregated["model_size_bytes"] = float(metrics[0][1].get("model_size_bytes", 0.0))
    return aggregated


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    scalar_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "inference_time_sec",
        "dynamic_quantized_model_size_bytes",
        "dynamic_quantized_parameter_count",
        "quantization_applied",
    ]
    aggregated = weighted_average(metrics, scalar_keys)

    aggregated["communication_bytes"] = float(
        sum(metric.get("communication_bytes", 0.0) for _, metric in metrics)
    )
    aggregated["inference_flops"] = float(
        sum(metric.get("inference_flops", 0.0) for _, metric in metrics)
    )

    aggregated["tn"] = float(sum(metric.get("tn", 0.0) for _, metric in metrics))
    aggregated["fp"] = float(sum(metric.get("fp", 0.0) for _, metric in metrics))
    aggregated["fn"] = float(sum(metric.get("fn", 0.0) for _, metric in metrics))
    aggregated["tp"] = float(sum(metric.get("tp", 0.0) for _, metric in metrics))
    aggregated["model_size_bytes"] = float(metrics[0][1].get("model_size_bytes", 0.0))
    aggregated["quantization_backend"] = str(metrics[0][1].get("quantization_backend", "none"))
    aggregated["quantization_error"] = str(metrics[0][1].get("quantization_error", ""))

    return aggregated


# ============================================================
# 6. FLOWER CLIENT
# ============================================================

class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: int,
        model_stats: Dict[str, float],
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        self.model_stats = model_stats

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {key: torch.tensor(value, device=DEVICE) for key, value in state_dict.items()},
            strict=True,
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        parameter_bytes = get_parameter_bytes(parameters)
        train_stats = train(
            self.model,
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=LOCAL_EPOCHS,
        )

        metrics = {
            "train_time_sec": float(train_stats["train_time_sec"]),
            "train_loss": float(train_stats["train_loss"]),
            "val_loss": float(train_stats["val_loss"]),
            "communication_bytes": float(parameter_bytes * 2),
            "model_size_bytes": float(self.model_stats["model_size_bytes"]),
            "parameter_count": float(self.model_stats["parameters"]),
            "training_flops": float(
                len(self.X_train)
                * LOCAL_EPOCHS
                * self.model_stats["training_flops_per_sample"]
            ),
        }

        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        import torch as _torch
        from sklearn.metrics import (
            accuracy_score as _accuracy_score,
            confusion_matrix as _confusion_matrix,
            f1_score as _f1_score,
            precision_score as _precision_score,
            recall_score as _recall_score,
            roc_auc_score as _roc_auc_score,
        )

        self.set_parameters(parameters)

        model_to_eval = self.model
        eval_device = DEVICE
        dynamic_quantized_model_size_bytes = 0.0
        dynamic_quantized_parameter_count = 0.0
        quantization_applied = 0.0
        quantization_backend = "none"
        quantization_error = ""

        if self.model_stats.get("quantization_mode") == "dynamic_int8":
            supported_engines = [
                str(engine) for engine in _torch.backends.quantized.supported_engines
            ]
            if supported_engines:
                preferred_engine = (
                    "qnnpack" if "qnnpack" in supported_engines else supported_engines[0]
                )
                try:
                    previous_engine = str(_torch.backends.quantized.engine)
                    _torch.backends.quantized.engine = preferred_engine
                    model_to_eval = deepcopy(self.model).to("cpu").eval()
                    model_to_eval.feature_value_proj = _torch.ao.quantization.quantize_dynamic(
                        model_to_eval.feature_value_proj,
                        {_torch.nn.Linear},
                        dtype=_torch.qint8,
                    )
                    model_to_eval.classifier = _torch.ao.quantization.quantize_dynamic(
                        model_to_eval.classifier,
                        {_torch.nn.Linear},
                        dtype=_torch.qint8,
                    )
                    dynamic_quantized_model_size_bytes = float(
                        get_serialized_model_size_bytes(model_to_eval)
                    )
                    dynamic_quantized_parameter_count = float(
                        get_parameter_count(model_to_eval)
                    )
                    quantization_applied = 1.0
                    quantization_backend = preferred_engine
                    eval_device = _torch.device("cpu")
                except Exception as exc:
                    model_to_eval = deepcopy(self.model).to("cpu").eval()
                    eval_device = _torch.device("cpu")
                    quantization_backend = preferred_engine
                    quantization_error = str(exc)
                finally:
                    if previous_engine in supported_engines:
                        _torch.backends.quantized.engine = previous_engine
            else:
                model_to_eval = deepcopy(self.model).to("cpu").eval()
                eval_device = _torch.device("cpu")
                quantization_error = "No supported quantization backend"

        model_to_eval.eval()
        X_tensor = _torch.tensor(self.X_test, dtype=_torch.float32).to(eval_device)

        start_time = time.perf_counter()
        with _torch.no_grad():
            logits = model_to_eval(X_tensor)
            probs = _torch.sigmoid(logits).cpu().numpy().ravel()
        inference_time = time.perf_counter() - start_time

        preds = (probs >= DECISION_THRESHOLD).astype(int)
        tn, fp, fn, tp = _confusion_matrix(self.y_test, preds, labels=[0, 1]).ravel()

        eval_metrics = {
            "accuracy": float(_accuracy_score(self.y_test, preds)),
            "precision": float(_precision_score(self.y_test, preds, zero_division=0)),
            "recall": float(_recall_score(self.y_test, preds, zero_division=0)),
            "f1_score": float(_f1_score(self.y_test, preds, zero_division=0)),
            "inference_time_sec": float(inference_time),
            "model_size_bytes": float(self.model_stats["model_size_bytes"]),
            "dynamic_quantized_model_size_bytes": float(dynamic_quantized_model_size_bytes),
            "dynamic_quantized_parameter_count": float(dynamic_quantized_parameter_count),
            "quantization_applied": float(quantization_applied),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }

        try:
            eval_metrics["auc_roc"] = float(_roc_auc_score(self.y_test, probs))
        except ValueError:
            eval_metrics["auc_roc"] = float("nan")

        eval_metrics.update(
            {
                "communication_bytes": float(get_parameter_bytes(parameters)),
                "inference_flops": float(
                    len(self.X_test) * self.model_stats["forward_flops_per_sample"]
                ),
                "quantization_backend": quantization_backend,
                "quantization_error": quantization_error,
            }
        )

        loss = float(1.0 - eval_metrics["accuracy"])
        return loss, len(self.X_test), eval_metrics


# ============================================================
# 7. CUSTOM STRATEGY WITH CLEAN LOGS
# ============================================================

class MetricsFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_metrics_cache: Dict[int, Dict[str, float]] = {}
        self.round_logs: List[Dict[str, float]] = []
        self.final_metrics: Dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_metrics:
            self.fit_metrics_cache[server_round] = {
                "communication_bytes": float(
                    aggregated_metrics.get("communication_bytes", 0.0)
                ),
                "train_time_sec": float(aggregated_metrics.get("train_time_sec", 0.0)),
                "train_loss": float(aggregated_metrics.get("train_loss", 0.0)),
                "val_loss": float(aggregated_metrics.get("val_loss", 0.0)),
            }

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_metrics:
            self.log_round_metrics(server_round, aggregated_metrics)

        return aggregated_loss, aggregated_metrics

    def log_round_metrics(self, server_round: int, aggregated_metrics: Metrics) -> None:
        fit_metrics = self.fit_metrics_cache.get(server_round, {})
        total_comm = (
            aggregated_metrics.get("communication_bytes", 0.0)
            + fit_metrics.get("communication_bytes", 0.0)
        )

        row = {
            "round": float(server_round),
            "accuracy": float(aggregated_metrics.get("accuracy", 0.0)),
            "precision": float(aggregated_metrics.get("precision", 0.0)),
            "recall": float(aggregated_metrics.get("recall", 0.0)),
            "f1_score": float(aggregated_metrics.get("f1_score", 0.0)),
            "auc_roc": float(aggregated_metrics.get("auc_roc", 0.0)),
            "train_time_sec": float(fit_metrics.get("train_time_sec", 0.0)),
            "train_loss": float(fit_metrics.get("train_loss", 0.0)),
            "val_loss": float(fit_metrics.get("val_loss", 0.0)),
            "inference_time_sec": float(aggregated_metrics.get("inference_time_sec", 0.0)),
            "communication_bytes": float(total_comm),
            "model_size_bytes": float(aggregated_metrics.get("model_size_bytes", 0.0)),
            "dynamic_quantized_model_size_bytes": float(
                aggregated_metrics.get("dynamic_quantized_model_size_bytes", 0.0)
            ),
            "dynamic_quantized_parameter_count": float(
                aggregated_metrics.get("dynamic_quantized_parameter_count", 0.0)
            ),
            "quantization_applied": float(aggregated_metrics.get("quantization_applied", 0.0)),
            "tn": float(aggregated_metrics.get("tn", 0.0)),
            "fp": float(aggregated_metrics.get("fp", 0.0)),
            "fn": float(aggregated_metrics.get("fn", 0.0)),
            "tp": float(aggregated_metrics.get("tp", 0.0)),
        }

        self.round_logs.append(row)
        self.final_metrics = row

        print(
            f"Round {server_round:02d} | "
            f"Acc={row['accuracy']:.4f} | "
            f"Prec={row['precision']:.4f} | "
            f"Rec={row['recall']:.4f} | "
            f"F1={row['f1_score']:.4f} | "
            f"AUC={row['auc_roc']:.4f} | "
            f"TrainLoss={row['train_loss']:.4f} | "
            f"ValLoss={row['val_loss']:.4f} | "
            f"Comm={format_bytes(row['communication_bytes'])} | "
            f"Model={format_bytes(row['model_size_bytes'])} | "
            f"DynModel={format_bytes(row['dynamic_quantized_model_size_bytes'])} | "
            f"DynParams={int(row['dynamic_quantized_parameter_count'])} | "
            f"DynQ={'on' if row['quantization_applied'] >= 0.5 else 'fallback'}"
        )


# ============================================================
# 8. CLIENT FUNCTION
# ============================================================

def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    X_train, X_val, X_test, y_train, y_val, y_test = global_clients[cid]
    model = TabTransformer(input_dim=global_input_dim).to(DEVICE)

    client = FLClient(
        model=model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        client_id=cid,
        model_stats=global_model_stats,
    )
    return client.to_client()


def create_local_client(cid: int) -> FLClient:
    X_train, X_val, X_test, y_train, y_val, y_test = global_clients[cid]
    model = TabTransformer(input_dim=global_input_dim).to(DEVICE)

    return FLClient(
        model=model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        client_id=cid,
        model_stats=global_model_stats,
    )


def aggregate_parameters(
    client_parameters: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in client_parameters)
    return [
        sum(
            parameters[layer_idx] * (num_examples / total_examples)
            for parameters, num_examples in client_parameters
        )
        for layer_idx in range(len(client_parameters[0][0]))
    ]


def run_local_fedavg(strategy: MetricsFedAvg) -> None:
    print("Running local in-process FedAvg backend.\n")

    clients = [create_local_client(cid) for cid in range(NUM_CLIENTS)]
    global_parameters = clients[0].get_parameters({})

    for server_round in range(1, NUM_ROUNDS + 1):
        fit_results = [
            client.fit(global_parameters, {})
            for client in clients
        ]
        fit_metrics = fit_metrics_aggregation(
            [(num_examples, metrics) for _, num_examples, metrics in fit_results]
        )
        strategy.fit_metrics_cache[server_round] = {
            "communication_bytes": float(fit_metrics.get("communication_bytes", 0.0)),
            "train_time_sec": float(fit_metrics.get("train_time_sec", 0.0)),
            "train_loss": float(fit_metrics.get("train_loss", 0.0)),
            "val_loss": float(fit_metrics.get("val_loss", 0.0)),
        }

        global_parameters = aggregate_parameters(
            [(parameters, num_examples) for parameters, num_examples, _ in fit_results]
        )

        evaluate_results = [
            client.evaluate(global_parameters, {})
            for client in clients
        ]
        evaluate_metrics = evaluate_metrics_aggregation(
            [(num_examples, metrics) for _, num_examples, metrics in evaluate_results]
        )
        strategy.log_round_metrics(server_round, evaluate_metrics)


def plot_loss_curves(round_logs: List[Dict[str, float]]) -> None:
    if not round_logs:
        return

    rounds = [int(row["round"]) for row in round_logs]
    train_losses = [float(row["train_loss"]) for row in round_logs]
    val_losses = [float(row["val_loss"]) for row in round_logs]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_losses, marker="o", linewidth=2, label="Training Loss")
    plt.plot(rounds, val_losses, marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Round")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    total_start = time.perf_counter()

    X, y = load_data()
    global_clients = create_clients(X, y, num_clients=NUM_CLIENTS)
    global_input_dim = X.shape[1]
    global_model_stats = get_model_stats(TabTransformer(global_input_dim), global_input_dim)
    global_model_stats["dynamic_quantized_model_size_bytes"] = 0.0
    global_model_stats["quantization_mode"] = QUANTIZATION_MODE

    print("================ MODEL PROFILE ================")
    print(f"Parameters           : {int(global_model_stats['parameters'])}")
    print(f"Model Size           : {format_bytes(global_model_stats['model_size_bytes'])}")
    print(f"Quantization         : {get_quantization_label()}")
    print(f"Forward FLOPs/sample : {int(global_model_stats['forward_flops_per_sample'])}")
    print(f"Training FLOPs/sample: {int(global_model_stats['training_flops_per_sample'])}")
    print("================================================\n")

    strategy = MetricsFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    print("============== FEDERATED TRAINING ==============")
    print(
        f"Clients={NUM_CLIENTS} | Rounds={NUM_ROUNDS} | "
        f"Local Epochs={LOCAL_EPOCHS} | Batch Size={BATCH_SIZE}"
    )
    print("================================================\n")

    if importlib.util.find_spec("ray") is None:
        run_local_fedavg(strategy)
    else:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

    total_runtime = time.perf_counter() - total_start
    final = strategy.final_metrics
    total_training_time = sum(row.get("train_time_sec", 0.0) for row in strategy.round_logs)
    average_training_time = (
        total_training_time / len(strategy.round_logs) if strategy.round_logs else 0.0
    )

    print("\n================ FINAL RESULTS ================")
    print(f"Final Accuracy       : {final.get('accuracy', 0.0):.4f}")
    print(f"Final Precision      : {final.get('precision', 0.0):.4f}")
    print(f"Final Recall         : {final.get('recall', 0.0):.4f}")
    print(f"Final F1-score       : {final.get('f1_score', 0.0):.4f}")
    print(f"Final AUC-ROC        : {final.get('auc_roc', 0.0):.4f}")
    print(f"Final Train Loss     : {final.get('train_loss', 0.0):.4f}")
    print(f"Final Validation Loss: {final.get('val_loss', 0.0):.4f}")
    print(f"Training Time        : {final.get('train_time_sec', 0.0):.4f} s")
    print(f"Total Training Time  : {total_training_time:.4f} s")
    print(f"Avg Train Time/Round : {average_training_time:.4f} s")
    print(f"Inference Time       : {final.get('inference_time_sec', 0.0):.4f} s")
    print(f"Final Communication  : {format_bytes(final.get('communication_bytes', 0.0))}")
    print(f"Model Size           : {format_bytes(final.get('model_size_bytes', 0.0))}")
    print(
        f"Dynamic QModel Size  : "
        f"{format_bytes(final.get('dynamic_quantized_model_size_bytes', 0.0))}"
    )
    print(
        f"Dynamic Q Parameters : "
        f"{int(final.get('dynamic_quantized_parameter_count', 0.0))}"
    )
    print(f"Total Runtime        : {total_runtime:.2f} s")
    print("================================================")

    print("\n========== FINAL AGGREGATED CONFUSION MATRIX ==========")
    print(f"TN: {int(final.get('tn', 0))}    FP: {int(final.get('fp', 0))}")
    print(f"FN: {int(final.get('fn', 0))}    TP: {int(final.get('tp', 0))}")
    print("=======================================================\n")

    print("Round-wise Results (copy into thesis table):")
    print("Round\tAccuracy\tPrecision\tRecall\tF1\tAUC")
    for row in strategy.round_logs:
        print(
            f"{int(row['round'])}\t"
            f"{row['accuracy']:.4f}\t"
            f"{row['precision']:.4f}\t"
            f"{row['recall']:.4f}\t"
            f"{row['f1_score']:.4f}\t"
            f"{row['auc_roc']:.4f}"
        )

    plot_loss_curves(strategy.round_logs)
