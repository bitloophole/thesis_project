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
    infer_num_classes,
    label_distribution,
    load_client_frames,
    split_features_labels,
    stratified_train_val_split,
    train_val_split,
)
from ciciot.metrics import evaluate_classification_detailed
from ciciot.tasks import THESIS_MULTICLASS_TASK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated FedAvg with TabTransformer on CICIoT2023")
    parser.add_argument("--global-data-path", type=Path, default=DATA_DIR / "global.csv")
    parser.add_argument("--client-dir", type=Path, default=FEDERATED_DIR)
    parser.add_argument("--test-rows", type=int, default=100000)
    parser.add_argument("--client-max-rows", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--round-eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-train-loss", action="store_true")
    parser.add_argument("--disable-local-val", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--fedavg-weighting",
        choices=["samples", "sqrt_samples", "uniform"],
        default="samples",
        help="Client weight mode during FedAvg aggregation.",
    )
    parser.add_argument(
        "--round-early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evaluated rounds without improvement on --early-stop-metric. 0 disables.",
    )
    parser.add_argument(
        "--round-min-delta",
        type=float,
        default=0.001,
        help="Minimum metric improvement required to reset round early-stopping patience.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc_macro_ovr"],
        default="f1_macro",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use inverse-frequency sampling for local client train loader instead of plain shuffle.",
    )
    parser.add_argument("--disable-class-weighted", action="store_true")
    parser.add_argument("--disable-standardize", action="store_true")
    parser.add_argument("--sequential-sample", action="store_true")
    parser.add_argument("--non-stratified", action="store_true")
    parser.add_argument("--allow-class-mismatch", action="store_true")
    return parser.parse_args(argv)


def resolve_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        from ciciot.models.tabtransformer_torch import TabTransformerClassifier, TabTransformerConfig
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Federated TabTransformer training requires PyTorch. Install torch and rerun train_federated_tabtransformer.py."
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


def make_loader(
    torch_module,
    dataloader_cls,
    tensor_dataset_cls,
    weighted_sampler_cls,
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: str,
):
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
        class_counts = np.bincount(y.astype(np.int64))
        class_counts = np.where(class_counts > 0, class_counts, 1)
        sample_weights = (1.0 / class_counts[y.astype(np.int64)]).astype(np.float64)
        sampler = weighted_sampler_cls(
            weights=torch_module.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return dataloader_cls(**loader_kwargs)


def evaluate_model(torch_module, model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> tuple[float, np.ndarray]:
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


def average_state_dicts(
    torch_module, weighted_states: list[tuple[int, dict[str, object]]], weighting_mode: str
) -> dict[str, object]:
    if weighting_mode == "samples":
        client_weights = [float(num_examples) for num_examples, _ in weighted_states]
    elif weighting_mode == "sqrt_samples":
        client_weights = [float(np.sqrt(num_examples)) for num_examples, _ in weighted_states]
    elif weighting_mode == "uniform":
        client_weights = [1.0 for _ in weighted_states]
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported FedAvg weighting mode: {weighting_mode}")

    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("Cannot average model states with zero examples.")

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


def train_one_client(
    torch_module,
    nn_module,
    dataloader_cls,
    tensor_dataset_cls,
    weighted_sampler_cls,
    model_cls,
    model_config,
    global_state: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray | None,
    args: argparse.Namespace,
    device: str,
    seed: int,
    run_local_validation: bool,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    torch_module.manual_seed(seed)
    if device == "cuda":
        torch_module.cuda.manual_seed_all(seed)

    local_model = model_cls(model_config).to(device)
    local_model.load_state_dict(global_state)

    train_loader = make_loader(
        torch_module, dataloader_cls, tensor_dataset_cls, weighted_sampler_cls, x_train, y_train, args, device
    )
    weight_tensor = (
        torch_module.tensor(class_weights, dtype=torch_module.float32, device=device)
        if class_weights is not None
        else None
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
                    logits,
                    batch_y,
                    weight=weight_tensor,
                    reduction="none",
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

        should_eval = run_local_validation and ((epoch % args.eval_every == 0) or (epoch == args.local_epochs))
        if should_eval:
            val_loss, _ = evaluate_model(torch_module, local_model, x_val, y_val, args.batch_size, device)
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

    if not run_local_validation:
        best_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
    return best_state, history


def run_federated_experiment(args: argparse.Namespace) -> dict[str, object]:
    torch, nn, DataLoader, TensorDataset, WeightedRandomSampler, TabTransformerClassifier, TabTransformerConfig = (
        resolve_torch()
    )
    device = choose_device(torch, args.device)

    test_split = build_test_split_from_global(
        args.global_data_path,
        max_rows=args.test_rows,
        seed=args.seed,
        random_sample=not args.sequential_sample,
        stratified=not args.non_stratified,
    )
    all_labels = list(test_split.y_train) + list(test_split.y_val) + list(test_split.y_test)
    num_classes = infer_num_classes(all_labels)
    if not args.allow_class_mismatch and num_classes != THESIS_MULTICLASS_TASK.expected_num_classes:
        raise ValueError(
            f"Expected {THESIS_MULTICLASS_TASK.expected_num_classes} classes for the thesis multiclass setup, found {num_classes}. "
            "Increase --test-rows or use a different sample."
        )

    class_weighting_enabled = not args.disable_class_weighted
    standardization_enabled = not args.disable_standardize
    if standardization_enabled:
        standardizer = fit_standardizer(test_split.x_train)
        test_split.x_train = standardizer.transform(test_split.x_train)
        test_split.x_val = standardizer.transform(test_split.x_val)
        test_split.x_test = standardizer.transform(test_split.x_test)

    client_frames = load_client_frames(
        args.client_dir,
        max_rows_per_client=args.client_max_rows,
        random_sample=not args.sequential_sample,
        seed=args.seed,
    )

    model_config = TabTransformerConfig(
        num_features=int(test_split.x_train.shape[1]),
        num_classes=int(num_classes),
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    global_model = TabTransformerClassifier(model_config).to(device)

    # Precompute per-client train/val splits and preprocessing once.
    prepared_clients: list[dict[str, object]] = []
    for client_index, (client_name, frame) in enumerate(client_frames):
        x_client, y_client = split_features_labels(frame)
        if args.non_stratified:
            x_train, y_train, x_val, y_val = train_val_split(x_client, y_client, train_ratio=0.85, seed=args.seed)
        else:
            x_train, y_train, x_val, y_val = stratified_train_val_split(x_client, y_client, train_ratio=0.85, seed=args.seed)

        if standardization_enabled:
            client_standardizer = fit_standardizer(x_train)
            x_train = client_standardizer.transform(x_train)
            x_val = client_standardizer.transform(x_val)

        class_weights = compute_class_weights(y_train, num_classes) if class_weighting_enabled else None
        prepared_clients.append(
            {
                "client_name": client_name,
                "client_index": client_index + 1,
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
                "class_weights": class_weights,
            }
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
            x_train = client_data["x_train"]
            y_train = client_data["y_train"]
            x_val = client_data["x_val"]
            y_val = client_data["y_val"]
            class_weights = client_data["class_weights"]
            client_state, client_history = train_one_client(
                torch,
                nn,
                DataLoader,
                TensorDataset,
                WeightedRandomSampler,
                TabTransformerClassifier,
                model_config,
                global_state,
                x_train,
                y_train,
                x_val,
                y_val,
                class_weights,
                args,
                device,
                seed=args.seed + round_idx * 100 + client_index,
                run_local_validation=not args.disable_local_val,
            )
            client_updates.append((len(y_train), client_state))
            round_client_histories.append({"client_name": client_data["client_name"], "history": client_history})

            if round_idx == 1:
                client_summaries.append(
                    {
                        "client_name": client_data["client_name"],
                        "client_index": client_data["client_index"],
                        "train_samples": int(len(y_train)),
                        "val_samples": int(len(y_val)),
                        "train_distribution": label_distribution(y_train, num_classes),
                        "val_distribution": label_distribution(y_val, num_classes),
                    }
                )

        averaged_state = average_state_dicts(torch, client_updates, args.fedavg_weighting)
        global_model.load_state_dict(averaged_state)

        should_eval_global = (round_idx % args.round_eval_every == 0) or (round_idx == args.rounds)
        if should_eval_global:
            _, test_probabilities = evaluate_model(
                torch,
                global_model,
                test_split.x_test,
                test_split.y_test,
                args.batch_size,
                device,
            )
            test_metrics = evaluate_classification_detailed(test_split.y_test, test_probabilities)
            round_result = {"round": float(round_idx), **test_metrics}
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
                        print(
                            json.dumps(
                                {
                                    "round": round_idx,
                                    "early_stopped": True,
                                    "metric": args.early_stop_metric,
                                    "best_round": best_round,
                                    "best_metric": best_metric,
                                }
                            )
                        )
                        break
        else:
            round_history.append({"round": float(round_idx), "skipped_test_eval": True})
            print(json.dumps({"round": round_idx, "skipped_test_eval": True}))

    training_time = time.time() - start_time
    global_model.load_state_dict(best_global_state)
    _, final_best_probabilities = evaluate_model(
        torch,
        global_model,
        test_split.x_test,
        test_split.y_test,
        args.batch_size,
        device,
    )
    final_best_metrics = evaluate_classification_detailed(test_split.y_test, final_best_probabilities)
    final_best_metrics["round"] = float(best_round)

    split_summary = {
        "train_samples": int(len(test_split.y_train)),
        "val_samples": int(len(test_split.y_val)),
        "test_samples": int(len(test_split.y_test)),
        "train_distribution": label_distribution(test_split.y_train, num_classes),
        "val_distribution": label_distribution(test_split.y_val, num_classes),
        "test_distribution": label_distribution(test_split.y_test, num_classes),
    }
    output = {
        "mode": "federated_tabtransformer",
        "task": THESIS_MULTICLASS_TASK.to_dict(),
        "global_data_path": str(args.global_data_path),
        "client_dir": str(args.client_dir),
        "test_rows": args.test_rows,
        "client_max_rows": args.client_max_rows,
        "training_config": {
            "input_dim": int(model_config.num_features),
            "output_dim": int(model_config.num_classes),
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
            "skip_train_loss": args.skip_train_loss,
            "disable_local_val": args.disable_local_val,
            "local_epochs": args.local_epochs,
            "rounds": args.rounds,
            "focal_gamma": args.focal_gamma,
            "fedavg_weighting": args.fedavg_weighting,
            "round_early_stop_patience": args.round_early_stop_patience,
            "round_min_delta": args.round_min_delta,
            "early_stop_metric": args.early_stop_metric,
            "seed": args.seed,
            "device": device,
            "num_workers": args.num_workers,
        },
        "preprocessing": {
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "balanced_sampler": args.balanced_sampler,
            "patience": args.patience,
        },
        "split_summary": split_summary,
        "client_summaries": client_summaries,
        "training_time_seconds": training_time,
        "early_stopping": {
            "enabled": args.round_early_stop_patience > 0,
            "metric": args.early_stop_metric,
            "min_delta": args.round_min_delta,
            "patience": args.round_early_stop_patience,
            "best_round": int(best_round),
            "best_metric": float(best_metric) if np.isfinite(best_metric) else None,
            "early_stopped": early_stopped,
        },
        "round_history": round_history,
        "final_test_metrics": final_best_metrics,
    }
    return output


def main() -> None:
    args = parse_args()
    ensure_results_dir(RESULTS_DIR)
    output = run_federated_experiment(args)

    output_path = RESULTS_DIR / "federated_tabtransformer_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
