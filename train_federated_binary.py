from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ciciot.config import DATA_DIR, FEDERATED_DIR, RESULTS_DIR
from ciciot.data import (
    build_test_split_from_global,
    compute_class_weights,
    ensure_results_dir,
    fit_standardizer,
    label_distribution,
    load_client_frames,
    split_features_labels,
    stratified_train_val_split,
    train_val_split,
)
from ciciot.metrics import evaluate_classification_detailed
from ciciot.models.mlp_numpy import NumpyMLP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated FedAvg for binary CICIoT2023 classification")
    parser.add_argument("--model", choices=["mlp", "tabtransformer"], default="mlp")
    parser.add_argument("--global-data-path", type=Path, default=DATA_DIR / "global.csv")
    parser.add_argument("--client-dir", type=Path, default=FEDERATED_DIR)
    parser.add_argument("--test-rows", type=int, default=100000)
    parser.add_argument("--client-max-rows", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--round-eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train-loss", action="store_true")
    parser.add_argument("--disable-class-weighted", action="store_true")
    parser.add_argument("--disable-standardize", action="store_true")
    parser.add_argument("--sequential-sample", action="store_true")
    parser.add_argument("--non-stratified", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--benign-label", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-local-val", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--fedavg-weighting", choices=["samples", "sqrt_samples", "uniform"], default="samples")
    parser.add_argument("--round-early-stop-patience", type=int, default=0)
    parser.add_argument("--round-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--early-stop-metric",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc_macro_ovr"],
        default="f1_macro",
    )
    return parser.parse_args(argv)


def resolve_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        from ciciot.models.tabtransformer_torch import TabTransformerClassifier, TabTransformerConfig
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Binary federated TabTransformer training requires PyTorch. Install torch and rerun with --model tabtransformer."
        ) from exc
    return torch, nn, DataLoader, TensorDataset, WeightedRandomSampler, TabTransformerClassifier, TabTransformerConfig


def choose_device(torch_module, requested: str) -> str:
    mps_available = bool(hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available())
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available.")
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def to_binary_labels(labels: np.ndarray, benign_label: int) -> np.ndarray:
    return np.where(labels == benign_label, 0, 1).astype(np.int64)


def summarize_binary(labels: np.ndarray) -> dict[str, object]:
    return {
        "distribution": label_distribution(labels.astype(np.int64), 2),
        "attack_rate": float(np.mean(labels == 1)) if len(labels) else 0.0,
    }


def binary_task_definition(args: argparse.Namespace) -> dict[str, object]:
    return {
        "name": "thesis_binary_benign_vs_attack",
        "problem_type": "binary_classification",
        "label_space": f"{args.benign_label}=benign, all other original labels=attack",
        "expected_num_classes": 2,
        "class_names": ("benign", "attack"),
        "notes": (
            "Derived from the prepared multiclass CICIoT2023 CSV files used elsewhere in this repository.",
            "By default this maps original label 0 to benign and labels 1..N to attack.",
        ),
    }


def prepare_global_split(args: argparse.Namespace):
    split = build_test_split_from_global(
        args.global_data_path,
        max_rows=args.test_rows,
        seed=args.seed,
        random_sample=not args.sequential_sample,
        stratified=not args.non_stratified,
    )
    split.y_train = to_binary_labels(split.y_train, args.benign_label)
    split.y_val = to_binary_labels(split.y_val, args.benign_label)
    split.y_test = to_binary_labels(split.y_test, args.benign_label)
    return split


def prepare_clients(args: argparse.Namespace) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    client_frames = load_client_frames(
        args.client_dir,
        max_rows_per_client=args.client_max_rows,
        random_sample=not args.sequential_sample,
        seed=args.seed,
    )
    prepared = []
    for client_name, frame in client_frames:
        x_client, y_client = split_features_labels(frame)
        y_client = to_binary_labels(y_client, args.benign_label)
        if args.non_stratified:
            x_train, y_train, x_val, y_val = train_val_split(x_client, y_client, train_ratio=0.85, seed=args.seed)
        else:
            x_train, y_train, x_val, y_val = stratified_train_val_split(
                x_client, y_client, train_ratio=0.85, seed=args.seed
            )
        prepared.append((client_name, x_train, y_train, x_val, y_val))
    return prepared


def count_numpy_parameters(params: list[np.ndarray]) -> dict[str, object]:
    parameter_tensors = []
    total_parameters = 0
    total_bytes = 0
    for index, param in enumerate(params):
        param_count = int(param.size)
        param_bytes = int(param.nbytes)
        total_parameters += param_count
        total_bytes += param_bytes
        parameter_tensors.append(
            {
                "name": f"param_{index}",
                "shape": list(param.shape),
                "count": param_count,
                "bytes": param_bytes,
                "dtype": str(param.dtype),
            }
        )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "parameter_tensors": parameter_tensors,
        "model_size_bytes": total_bytes,
        "model_size_mb": total_bytes / (1024**2),
    }


def count_torch_parameters(model) -> dict[str, object]:
    parameter_tensors = []
    total_parameters = 0
    trainable_parameters = 0
    total_bytes = 0
    for name, param in model.named_parameters():
        param_count = int(param.numel())
        param_bytes = int(param.numel() * param.element_size())
        total_parameters += param_count
        total_bytes += param_bytes
        if param.requires_grad:
            trainable_parameters += param_count
        parameter_tensors.append(
            {
                "name": name,
                "shape": list(param.shape),
                "count": param_count,
                "bytes": param_bytes,
                "dtype": str(param.dtype).replace("torch.", ""),
                "trainable": bool(param.requires_grad),
            }
        )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_tensors": parameter_tensors,
        "model_size_bytes": total_bytes,
        "model_size_mb": total_bytes / (1024**2),
    }


def estimate_round_communication_cost(
    *,
    client_names: list[str],
    train_sample_counts: dict[str, int],
    model_size_bytes: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    round_client_costs = []
    total_upload_bytes = 0
    total_download_bytes = 0
    for client_name in client_names:
        train_samples = int(train_sample_counts[client_name])
        download_bytes = int(model_size_bytes)
        upload_bytes = int(model_size_bytes)
        total_download_bytes += download_bytes
        total_upload_bytes += upload_bytes
        round_client_costs.append(
            {
                "client_name": client_name,
                "train_samples": train_samples,
                "server_to_client_bytes": download_bytes,
                "client_to_server_bytes": upload_bytes,
                "total_bytes": download_bytes + upload_bytes,
            }
        )
    total_bytes = total_download_bytes + total_upload_bytes
    summary = {
        "assumption": "Each round sends the full global model to every client and receives one full updated model back from every client.",
        "clients_per_round": len(client_names),
        "server_to_client_bytes": total_download_bytes,
        "client_to_server_bytes": total_upload_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
    }
    return round_client_costs, summary


def measure_numpy_inference(model: NumpyMLP, x_test: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    started_at = time.perf_counter()
    probabilities = model.predict_proba(x_test)
    elapsed_seconds = time.perf_counter() - started_at
    sample_count = max(len(x_test), 1)
    samples_per_second = len(x_test) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return probabilities, {
        "total_seconds": elapsed_seconds,
        "total_milliseconds": elapsed_seconds * 1000.0,
        "samples": int(len(x_test)),
        "milliseconds_per_sample": (elapsed_seconds * 1000.0) / sample_count,
        "samples_per_second": samples_per_second,
    }


def measure_torch_inference(torch_module, model, x_test: np.ndarray, y_test: np.ndarray, batch_size: int, device: str):
    started_at = time.perf_counter()
    _, probabilities = evaluate_torch_model(torch_module, model, x_test, y_test, batch_size, device)
    elapsed_seconds = time.perf_counter() - started_at
    sample_count = max(len(x_test), 1)
    samples_per_second = len(x_test) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return probabilities, {
        "total_seconds": elapsed_seconds,
        "total_milliseconds": elapsed_seconds * 1000.0,
        "samples": int(len(x_test)),
        "milliseconds_per_sample": (elapsed_seconds * 1000.0) / sample_count,
        "samples_per_second": samples_per_second,
    }


def build_binary_distribution_plot(
    client_summaries: list[dict[str, object]],
    output_path: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    client_names = [str(summary["client_name"]) for summary in client_summaries]
    benign_counts = []
    attack_counts = []
    for summary in client_summaries:
        train_distribution = summary["train_distribution"]["distribution"]
        val_distribution = summary["val_distribution"]["distribution"]
        benign_counts.append(int(train_distribution["0"]) + int(val_distribution["0"]))
        attack_counts.append(int(train_distribution["1"]) + int(val_distribution["1"]))

    positions = np.arange(len(client_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(positions, benign_counts, label="Benign", color="#4C956C")
    ax.bar(positions, attack_counts, bottom=benign_counts, label="Attack", color="#D1495B")
    ax.set_xticks(positions)
    ax.set_xticklabels(client_names, rotation=0)
    ax.set_ylabel("Samples")
    ax.set_title("Client Class Distribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def finalize_binary_metrics(metrics: dict[str, object], inference_time: dict[str, float]) -> dict[str, object]:
    finalized = dict(metrics)
    finalized["inference_time"] = inference_time
    finalized["test_confusion_matrix"] = finalized.get("confusion_matrix")
    return finalized


def average_mlp_parameters(weighted_params: list[tuple[int, list[np.ndarray]]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for num_examples, _ in weighted_params)
    averaged: list[np.ndarray] = []
    for param_idx in range(len(weighted_params[0][1])):
        weighted_sum = sum(num_examples * params[param_idx] for num_examples, params in weighted_params)
        averaged.append(weighted_sum / total_examples)
    return averaged


def average_state_dicts(torch_module, weighted_states: list[tuple[int, dict[str, object]]], weighting_mode: str) -> dict[str, object]:
    if weighting_mode == "samples":
        client_weights = [float(num_examples) for num_examples, _ in weighted_states]
    elif weighting_mode == "sqrt_samples":
        client_weights = [float(np.sqrt(num_examples)) for num_examples, _ in weighted_states]
    else:
        client_weights = [1.0 for _ in weighted_states]

    total_weight = float(sum(client_weights))
    averaged: dict[str, object] = {}
    first_state = weighted_states[0][1]
    for key in first_state:
        first_value = first_state[key]
        if torch_module.is_tensor(first_value):
            if not torch_module.is_floating_point(first_value):
                averaged[key] = first_value.detach().clone()
                continue
            acc = torch_module.zeros_like(first_value, dtype=torch_module.float32)
            for client_weight, (_, state) in zip(client_weights, weighted_states):
                acc += state[key].detach().to(dtype=torch_module.float32, device=acc.device) * (
                    client_weight / total_weight
                )
            averaged[key] = acc.to(dtype=first_value.dtype)
        else:
            averaged[key] = first_value
    return averaged


def run_federated_mlp(args: argparse.Namespace) -> dict[str, object]:
    test_split = prepare_global_split(args)
    standardization_enabled = not args.disable_standardize
    class_weighting_enabled = not args.disable_class_weighted

    if standardization_enabled:
        standardizer = fit_standardizer(test_split.x_train)
        test_split.x_train = standardizer.transform(test_split.x_train)
        test_split.x_val = standardizer.transform(test_split.x_val)
        test_split.x_test = standardizer.transform(test_split.x_test)

    prepared_clients = prepare_clients(args)
    train_sample_counts = {client_name: int(len(y_train)) for client_name, x_train, y_train, x_val, y_val in prepared_clients}
    global_model = NumpyMLP(
        input_dim=test_split.x_train.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        output_dim=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    model_stats = count_numpy_parameters(global_model.get_parameters())
    per_client_round_costs, per_round_communication = estimate_round_communication_cost(
        client_names=[client_name for client_name, *_ in prepared_clients],
        train_sample_counts=train_sample_counts,
        model_size_bytes=int(model_stats["model_size_bytes"]),
    )

    round_history: list[dict[str, object]] = []
    client_summaries: list[dict[str, object]] = []
    start_time = time.time()

    for round_idx in range(1, args.rounds + 1):
        client_updates: list[tuple[int, list[np.ndarray]]] = []
        for client_index, (client_name, x_train, y_train, x_val, y_val) in enumerate(prepared_clients):
            x_train_local = x_train
            x_val_local = x_val
            if standardization_enabled:
                client_standardizer = fit_standardizer(x_train_local)
                x_train_local = client_standardizer.transform(x_train_local)
                x_val_local = client_standardizer.transform(x_val_local)

            class_weights = compute_class_weights(y_train, 2) if class_weighting_enabled else None
            local_model = NumpyMLP(
                input_dim=test_split.x_train.shape[1],
                hidden_dims=tuple(args.hidden_dims),
                output_dim=2,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed + round_idx + client_index,
                class_weights=class_weights,
                focal_gamma=args.focal_gamma,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            local_model.set_parameters(global_model.get_parameters())
            local_model.fit(
                x_train_local,
                y_train,
                x_val_local,
                y_val,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                eval_every=args.eval_every,
                log_train_loss=not args.skip_train_loss,
            )
            client_updates.append((len(y_train), local_model.get_parameters()))

            if round_idx == 1:
                client_summaries.append(
                    {
                        "client_name": client_name,
                        "client_index": client_index + 1,
                        "train_samples": int(len(y_train)),
                        "val_samples": int(len(y_val)),
                        "train_distribution": summarize_binary(y_train),
                        "val_distribution": summarize_binary(y_val),
                    }
                )

        global_model.set_parameters(average_mlp_parameters(client_updates))
        round_probabilities, round_inference_time = measure_numpy_inference(global_model, test_split.x_test)
        test_metrics = finalize_binary_metrics(
            evaluate_classification_detailed(test_split.y_test, round_probabilities),
            round_inference_time,
        )
        test_metrics["round"] = float(round_idx)
        test_metrics["communication_cost"] = {
            "round_bytes": per_round_communication["total_bytes"],
            "round_mb": per_round_communication["total_mb"],
            "per_client": per_client_round_costs,
        }
        round_history.append(test_metrics)
        print(json.dumps({"round": round_idx, "metrics": test_metrics}))

    artifact_path = RESULTS_DIR / f"federated_binary_{args.model}_client_class_distribution.png"
    client_distribution_plot = build_binary_distribution_plot(client_summaries, artifact_path)
    final_probabilities, final_inference_time = measure_numpy_inference(global_model, test_split.x_test)
    final_metrics = finalize_binary_metrics(
        evaluate_classification_detailed(test_split.y_test, final_probabilities),
        final_inference_time,
    )

    return build_output(
        args=args,
        model_name="mlp",
        split_summary={
            "train_samples": int(len(test_split.y_train)),
            "val_samples": int(len(test_split.y_val)),
            "test_samples": int(len(test_split.y_test)),
            "train_distribution": summarize_binary(test_split.y_train),
            "val_distribution": summarize_binary(test_split.y_val),
            "test_distribution": summarize_binary(test_split.y_test),
        },
        client_summaries=client_summaries,
        training_time=time.time() - start_time,
        round_history=round_history,
        model_stats=model_stats,
        per_round_communication=per_round_communication,
        total_communication={
            "total_bytes": per_round_communication["total_bytes"] * len(round_history),
            "total_mb": per_round_communication["total_mb"] * len(round_history),
            "evaluated_rounds": len(round_history),
        },
        client_distribution_plot=client_distribution_plot,
        training_config={
            "input_dim": int(test_split.x_train.shape[1]),
            "output_dim": 2,
            "hidden_dims": tuple(args.hidden_dims),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "focal_gamma": args.focal_gamma,
            "gradient_clip_norm": args.gradient_clip_norm,
            "eval_every": args.eval_every,
            "local_epochs": args.local_epochs,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        preprocessing={
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "patience": args.patience,
        },
        final_metrics=final_metrics,
    )


def make_loader(torch_module, dataloader_cls, tensor_dataset_cls, weighted_sampler_cls, x, y, args, device):
    features = torch_module.from_numpy(x.astype(np.float32))
    labels = torch_module.from_numpy(y.astype(np.int64))
    dataset = tensor_dataset_cls(features, labels)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "pin_memory": device == "cuda",
        "num_workers": max(args.num_workers, 0),
    }
    if args.balanced_sampler:
        class_counts = np.bincount(y.astype(np.int64), minlength=2)
        class_counts = np.where(class_counts > 0, class_counts, 1)
        sample_weights = (1.0 / class_counts[y.astype(np.int64)]).astype(np.float64)
        loader_kwargs["sampler"] = weighted_sampler_cls(
            weights=torch_module.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True
        )
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return dataloader_cls(**loader_kwargs)


def evaluate_torch_model(torch_module, model, x, y, batch_size: int, device: str) -> tuple[float, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    criterion = torch_module.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch_module.no_grad():
        for start in range(0, len(y), batch_size):
            end = start + batch_size
            batch_x = torch_module.from_numpy(x[start:end].astype(np.float32)).to(device)
            batch_y = torch_module.from_numpy(y[start:end].astype(np.int64)).to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            total_samples += len(batch_y)
            all_probs.append(torch_module.softmax(logits, dim=1).cpu().numpy())
    return total_loss / max(total_samples, 1), np.concatenate(all_probs, axis=0)


def train_one_tabtransformer_client(
    torch_module,
    nn_module,
    dataloader_cls,
    tensor_dataset_cls,
    weighted_sampler_cls,
    model_cls,
    model_config,
    global_state,
    x_train,
    y_train,
    x_val,
    y_val,
    class_weights,
    args,
    device,
    seed: int,
):
    torch_module.manual_seed(seed)
    if device == "cuda":
        torch_module.cuda.manual_seed_all(seed)

    local_model = model_cls(model_config).to(device)
    local_model.load_state_dict(global_state)
    train_loader = make_loader(
        torch_module, dataloader_cls, tensor_dataset_cls, weighted_sampler_cls, x_train, y_train, args, device
    )
    weight_tensor = (
        torch_module.tensor(class_weights, dtype=torch_module.float32, device=device) if class_weights is not None else None
    )
    criterion = nn_module.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch_module.optim.AdamW(local_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
    patience_left = args.patience

    for epoch in range(1, args.local_epochs + 1):
        local_model.train()
        running_loss = 0.0
        seen_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=device != "cpu")
            batch_y = batch_y.to(device, non_blocking=device != "cpu")
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(batch_x)
            if args.focal_gamma > 0.0:
                ce_per_sample = nn_module.functional.cross_entropy(
                    logits, batch_y, weight=weight_tensor, reduction="none"
                )
                pt = torch_module.exp(-ce_per_sample)
                loss = ((1.0 - pt) ** args.focal_gamma * ce_per_sample).mean()
            else:
                loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
            seen_samples += len(batch_y)

        entry = {"epoch": float(epoch)}
        if not args.skip_train_loss:
            entry["train_loss"] = running_loss / max(seen_samples, 1)

        should_eval = (not args.disable_local_val) and ((epoch % args.eval_every == 0) or (epoch == args.local_epochs))
        if should_eval:
            val_loss, _ = evaluate_torch_model(torch_module, local_model, x_val, y_val, args.batch_size, device)
            entry["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    history.append(entry)
                    break
        history.append(entry)

    if args.disable_local_val:
        best_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
    return best_state, history


def run_federated_tabtransformer(args: argparse.Namespace) -> dict[str, object]:
    torch, nn, DataLoader, TensorDataset, WeightedRandomSampler, TabTransformerClassifier, TabTransformerConfig = (
        resolve_torch()
    )
    device = choose_device(torch, args.device)
    test_split = prepare_global_split(args)
    standardization_enabled = not args.disable_standardize
    class_weighting_enabled = not args.disable_class_weighted

    if standardization_enabled:
        standardizer = fit_standardizer(test_split.x_train)
        test_split.x_train = standardizer.transform(test_split.x_train)
        test_split.x_val = standardizer.transform(test_split.x_val)
        test_split.x_test = standardizer.transform(test_split.x_test)

    model_config = TabTransformerConfig(
        num_features=int(test_split.x_train.shape[1]),
        num_classes=2,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    global_model = TabTransformerClassifier(model_config).to(device)
    model_stats = count_torch_parameters(global_model)

    prepared_clients = []
    for client_index, (client_name, x_train, y_train, x_val, y_val) in enumerate(prepare_clients(args)):
        if standardization_enabled:
            client_standardizer = fit_standardizer(x_train)
            x_train = client_standardizer.transform(x_train)
            x_val = client_standardizer.transform(x_val)
        prepared_clients.append(
            {
                "client_name": client_name,
                "client_index": client_index + 1,
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
                "class_weights": compute_class_weights(y_train, 2) if class_weighting_enabled else None,
            }
        )
    per_client_round_costs, per_round_communication = estimate_round_communication_cost(
        client_names=[str(client_data["client_name"]) for client_data in prepared_clients],
        train_sample_counts={str(client_data["client_name"]): int(len(client_data["y_train"])) for client_data in prepared_clients},
        model_size_bytes=int(model_stats["model_size_bytes"]),
    )

    round_history: list[dict[str, object]] = []
    client_summaries: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_round = 0
    best_global_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
    no_improve_evals = 0
    early_stopped = False
    start_time = time.time()

    for round_idx in range(1, args.rounds + 1):
        global_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
        client_updates: list[tuple[int, dict[str, object]]] = []
        round_client_histories: list[dict[str, object]] = []

        for client_index, client_data in enumerate(prepared_clients):
            client_state, client_history = train_one_tabtransformer_client(
                torch,
                nn,
                DataLoader,
                TensorDataset,
                WeightedRandomSampler,
                TabTransformerClassifier,
                model_config,
                global_state,
                client_data["x_train"],
                client_data["y_train"],
                client_data["x_val"],
                client_data["y_val"],
                client_data["class_weights"],
                args,
                device,
                seed=args.seed + round_idx * 100 + client_index,
            )
            client_updates.append((len(client_data["y_train"]), client_state))
            round_client_histories.append({"client_name": client_data["client_name"], "history": client_history})
            if round_idx == 1:
                client_summaries.append(
                    {
                        "client_name": client_data["client_name"],
                        "client_index": client_data["client_index"],
                        "train_samples": int(len(client_data["y_train"])),
                        "val_samples": int(len(client_data["y_val"])),
                        "train_distribution": summarize_binary(client_data["y_train"]),
                        "val_distribution": summarize_binary(client_data["y_val"]),
                    }
                )

        global_model.load_state_dict(average_state_dicts(torch, client_updates, args.fedavg_weighting))
        should_eval = (round_idx % args.round_eval_every == 0) or (round_idx == args.rounds)
        if should_eval:
            test_probabilities, round_inference_time = measure_torch_inference(
                torch, global_model, test_split.x_test, test_split.y_test, args.batch_size, device
            )
            test_metrics = finalize_binary_metrics(
                evaluate_classification_detailed(test_split.y_test, test_probabilities),
                round_inference_time,
            )
            round_result: dict[str, object] = {"round": float(round_idx), **test_metrics}
            round_result["communication_cost"] = {
                "round_bytes": per_round_communication["total_bytes"],
                "round_mb": per_round_communication["total_mb"],
                "per_client": per_client_round_costs,
            }
            if not args.skip_train_loss:
                round_result["client_histories"] = round_client_histories
            round_history.append(round_result)
            print(json.dumps({"round": round_idx, "metrics": test_metrics}))

            metric_value = float(round_result.get(args.early_stop_metric, float("nan")))
            if np.isfinite(metric_value):
                if metric_value > best_metric + args.round_min_delta:
                    best_metric = metric_value
                    best_round = round_idx
                    best_global_state = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1
                    if args.round_early_stop_patience > 0 and no_improve_evals >= args.round_early_stop_patience:
                        early_stopped = True
                        break
        else:
            round_history.append({"round": float(round_idx), "skipped_test_eval": True})

    training_time = time.time() - start_time
    global_model.load_state_dict(best_global_state)
    final_probabilities, final_inference_time = measure_torch_inference(
        torch, global_model, test_split.x_test, test_split.y_test, args.batch_size, device
    )
    final_metrics = finalize_binary_metrics(
        evaluate_classification_detailed(test_split.y_test, final_probabilities),
        final_inference_time,
    )
    final_metrics["round"] = float(best_round)
    artifact_path = RESULTS_DIR / f"federated_binary_{args.model}_client_class_distribution.png"
    client_distribution_plot = build_binary_distribution_plot(client_summaries, artifact_path)

    return build_output(
        args=args,
        model_name="tabtransformer",
        split_summary={
            "train_samples": int(len(test_split.y_train)),
            "val_samples": int(len(test_split.y_val)),
            "test_samples": int(len(test_split.y_test)),
            "train_distribution": summarize_binary(test_split.y_train),
            "val_distribution": summarize_binary(test_split.y_val),
            "test_distribution": summarize_binary(test_split.y_test),
        },
        client_summaries=client_summaries,
        training_time=training_time,
        round_history=round_history,
        model_stats=model_stats,
        per_round_communication=per_round_communication,
        total_communication={
            "total_bytes": per_round_communication["total_bytes"] * len(round_history),
            "total_mb": per_round_communication["total_mb"] * len(round_history),
            "configured_rounds": args.rounds,
            "executed_rounds": len(round_history),
        },
        client_distribution_plot=client_distribution_plot,
        training_config={
            "input_dim": int(model_config.num_features),
            "output_dim": 2,
            "d_token": args.d_token,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ffn_dim": args.ffn_dim,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "round_eval_every": args.round_eval_every,
            "local_epochs": args.local_epochs,
            "rounds": args.rounds,
            "seed": args.seed,
            "device": device,
            "num_workers": args.num_workers,
            "fedavg_weighting": args.fedavg_weighting,
        },
        preprocessing={
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "balanced_sampler": args.balanced_sampler,
            "patience": args.patience,
        },
        final_metrics=final_metrics,
        early_stopping={
            "enabled": args.round_early_stop_patience > 0,
            "metric": args.early_stop_metric,
            "min_delta": args.round_min_delta,
            "patience": args.round_early_stop_patience,
            "best_round": int(best_round),
            "best_metric": float(best_metric) if np.isfinite(best_metric) else None,
            "early_stopped": early_stopped,
        },
    )


def build_output(
    *,
    args: argparse.Namespace,
    model_name: str,
    split_summary: dict[str, object],
    client_summaries: list[dict[str, object]],
    training_time: float,
    round_history: list[dict[str, object]],
    model_stats: dict[str, object],
    per_round_communication: dict[str, object],
    total_communication: dict[str, object],
    client_distribution_plot: str | None,
    training_config: dict[str, object],
    preprocessing: dict[str, object],
    final_metrics: dict[str, object] | None = None,
    early_stopping: dict[str, object] | None = None,
) -> dict[str, object]:
    output = {
        "mode": "federated_binary",
        "model": model_name,
        "task": binary_task_definition(args),
        "global_data_path": str(args.global_data_path),
        "client_dir": str(args.client_dir),
        "test_rows": args.test_rows,
        "client_max_rows": args.client_max_rows,
        "training_config": training_config,
        "preprocessing": preprocessing,
        "split_summary": split_summary,
        "client_summaries": client_summaries,
        "artifacts": {
            "client_class_distribution_plot": client_distribution_plot,
        },
        "model_details": {
            "parameter_summary": {
                "total_parameters": model_stats["total_parameters"],
                "trainable_parameters": model_stats["trainable_parameters"],
            },
            "model_size": {
                "bytes": model_stats["model_size_bytes"],
                "mb": model_stats["model_size_mb"],
            },
            "parameter_tensors": model_stats["parameter_tensors"],
        },
        "communication": {
            "per_round": per_round_communication,
            "total": total_communication,
        },
        "training_time_seconds": training_time,
        "round_history": round_history,
        "final_test_metrics": final_metrics if final_metrics is not None else (round_history[-1] if round_history else {}),
    }
    if early_stopping is not None:
        output["early_stopping"] = early_stopping
    return output


def main() -> None:
    args = parse_args()
    ensure_results_dir(RESULTS_DIR)
    output = run_federated_mlp(args) if args.model == "mlp" else run_federated_tabtransformer(args)
    output_path = RESULTS_DIR / f"federated_binary_{args.model}_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
