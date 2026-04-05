# ============================================================
# Improved Federated IDS Binary Classifier with 1D CNN
# Non-IID + Clean Round Logs + Final Summary + Confusion Matrix
# ============================================================

import logging
import io
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import flwr as fl
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
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
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

class ImprovedCNN1D(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def get_model_stats(model: nn.Module, input_dim: int) -> Dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    length_1 = input_dim
    length_2 = max(length_1 // 2, 1)
    length_3 = max(length_2 // 2, 1)

    conv1_flops = length_1 * 32 * ((2 * 1 * 5) + 1)
    bn1_flops = length_1 * 32 * 2
    relu1_flops = length_1 * 32
    pool1_flops = length_2 * 32

    conv2_flops = length_2 * 64 * ((2 * 32 * 3) + 1)
    bn2_flops = length_2 * 64 * 2
    relu2_flops = length_2 * 64
    pool2_flops = length_3 * 64

    conv3_flops = length_3 * 128 * ((2 * 64 * 3) + 1)
    bn3_flops = length_3 * 128 * 2
    relu3_flops = length_3 * 128
    gap_flops = length_3 * 128

    linear1_flops = (2 * 128 * 64) + 64
    relu4_flops = 64
    linear2_flops = (2 * 64 * 1) + 1

    forward_flops = (
        conv1_flops
        + bn1_flops
        + relu1_flops
        + pool1_flops
        + conv2_flops
        + bn2_flops
        + relu2_flops
        + pool2_flops
        + conv3_flops
        + bn3_flops
        + relu3_flops
        + gap_flops
        + linear1_flops
        + relu4_flops
        + linear2_flops
    )
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


def quantize_model_dynamically(model: nn.Module) -> nn.Module:
    supported_engines = [str(engine) for engine in torch.backends.quantized.supported_engines]
    if not supported_engines:
        raise RuntimeError("No PyTorch quantization engine is available on this runtime.")

    previous_engine = str(torch.backends.quantized.engine)
    preferred_engine = "qnnpack" if "qnnpack" in supported_engines else supported_engines[0]

    try:
        torch.backends.quantized.engine = preferred_engine
        return torch.ao.quantization.quantize_dynamic(
            deepcopy(model).to("cpu").eval(),
            {nn.Linear},
            dtype=torch.qint8,
        )
    finally:
        if previous_engine in supported_engines:
            torch.backends.quantized.engine = previous_engine


def build_dynamic_quantized_model(model: nn.Module) -> nn.Module:
    return quantize_model_dynamically(model)


def get_dynamic_quantized_model_size_bytes(model: nn.Module) -> int:
    quantized_model = build_dynamic_quantized_model(model)
    return get_serialized_model_size_bytes(quantized_model)


def get_quantization_label() -> str:
    if QUANTIZATION_MODE == "dynamic_int8":
        return "Dynamic INT8 (Linear layers)"
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
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
        scheduler.step()

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
    quantization_applied = 0.0

    if QUANTIZATION_MODE == "dynamic_int8":
        supported_engines = [str(engine) for engine in torch.backends.quantized.supported_engines]
        if supported_engines:
            previous_engine = str(torch.backends.quantized.engine)
            preferred_engine = "qnnpack" if "qnnpack" in supported_engines else supported_engines[0]
            try:
                torch.backends.quantized.engine = preferred_engine
                model_to_eval = torch.ao.quantization.quantize_dynamic(
                    deepcopy(model).to("cpu").eval(),
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                eval_device = torch.device("cpu")
                dynamic_quantized_model_size_bytes = float(
                    get_serialized_model_size_bytes(model_to_eval)
                )
                quantization_applied = 1.0
            except Exception:
                model_to_eval = deepcopy(model).to("cpu").eval()
                eval_device = torch.device("cpu")
            finally:
                if previous_engine in supported_engines:
                    torch.backends.quantized.engine = previous_engine
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
        "dynamic_quantized_model_size_bytes",
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
    aggregated["dynamic_quantized_model_size_bytes"] = float(
        metrics[0][1].get("dynamic_quantized_model_size_bytes", 0.0)
    )
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
            "dynamic_quantized_model_size_bytes": float(
                self.model_stats["dynamic_quantized_model_size_bytes"]
            ),
            "parameter_count": float(self.model_stats["parameters"]),
            "training_flops": float(
                len(self.X_train)
                * LOCAL_EPOCHS
                * self.model_stats["training_flops_per_sample"]
            ),
        }

        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        import numpy as _np
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
        quantization_applied = 0.0
        quantization_backend = "none"
        quantization_error = ""

        if self.model_stats.get("quantization_mode") == "dynamic_int8":
            supported_engines = [
                str(engine) for engine in _torch.backends.quantized.supported_engines
            ]
            if supported_engines:
                previous_engine = str(_torch.backends.quantized.engine)
                preferred_engine = (
                    "qnnpack" if "qnnpack" in supported_engines else supported_engines[0]
                )

                try:
                    _torch.backends.quantized.engine = preferred_engine
                    model_to_eval = _torch.ao.quantization.quantize_dynamic(
                        deepcopy(self.model).to("cpu").eval(),
                        {_torch.nn.Linear},
                        dtype=_torch.qint8,
                    )
                    buffer = io.BytesIO()
                    _torch.save(model_to_eval.state_dict(), buffer)
                    dynamic_quantized_model_size_bytes = float(buffer.getbuffer().nbytes)
                    quantization_applied = 1.0
                    quantization_backend = preferred_engine
                    eval_device = _torch.device("cpu")
                except Exception as exc:
                    model_to_eval = deepcopy(self.model).to("cpu").eval()
                    eval_device = _torch.device("cpu")
                    quantization_error = str(exc)
                    quantization_backend = preferred_engine
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
            "dynamic_quantized_model_size_bytes": float(dynamic_quantized_model_size_bytes),
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
                "dynamic_quantized_model_size_bytes": float(
                    aggregated_metrics.get("dynamic_quantized_model_size_bytes", 0.0)
                ),
                "quantization_applied": float(
                    aggregated_metrics.get("quantization_applied", 0.0)
                ),
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
                f"DynQ={'on' if row['quantization_applied'] >= 0.5 else 'fallback'}"
            )

        return aggregated_loss, aggregated_metrics


# ============================================================
# 8. CLIENT FUNCTION
# ============================================================

def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    X_train, X_val, X_test, y_train, y_val, y_test = global_clients[cid]
    model = ImprovedCNN1D(input_dim=global_input_dim).to(DEVICE)

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
    base_model = ImprovedCNN1D(global_input_dim)
    global_model_stats = get_model_stats(base_model, global_input_dim)
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

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    total_runtime = time.perf_counter() - total_start
    final = strategy.final_metrics

    print("\n================ FINAL RESULTS ================")
    print(f"Final Accuracy       : {final.get('accuracy', 0.0):.4f}")
    print(f"Final Precision      : {final.get('precision', 0.0):.4f}")
    print(f"Final Recall         : {final.get('recall', 0.0):.4f}")
    print(f"Final F1-score       : {final.get('f1_score', 0.0):.4f}")
    print(f"Final AUC-ROC        : {final.get('auc_roc', 0.0):.4f}")
    print(f"Final Train Loss     : {final.get('train_loss', 0.0):.4f}")
    print(f"Final Validation Loss: {final.get('val_loss', 0.0):.4f}")
    print(f"Training Time        : {final.get('train_time_sec', 0.0):.4f} s")
    print(f"Inference Time       : {final.get('inference_time_sec', 0.0):.4f} s")
    print(f"Final Communication  : {format_bytes(final.get('communication_bytes', 0.0))}")
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
