from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ciciot.config import DATA_DIR, RESULTS_DIR
from ciciot.data import (
    build_test_split_from_global,
    compute_class_weights,
    ensure_results_dir,
    fit_standardizer,
    infer_num_classes,
    label_distribution,
)
from ciciot.metrics import evaluate_classification_detailed
from ciciot.tasks import THESIS_MULTICLASS_TASK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a centralized TabTransformer-style model on CICIoT2023")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR / "global.csv")
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
        from torch.utils.data import DataLoader, TensorDataset

        from ciciot.models.tabtransformer_torch import TabTransformerClassifier, TabTransformerConfig
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "TabTransformer training requires PyTorch. Install torch and rerun train_tabtransformer.py."
        ) from exc
    return torch, nn, DataLoader, TensorDataset, TabTransformerClassifier, TabTransformerConfig


def choose_device(torch_module, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def build_dataloader(torch_module, tensor_dataset_cls, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    features = torch_module.from_numpy(x.astype(np.float32))
    labels = torch_module.from_numpy(y.astype(np.int64))
    dataset = tensor_dataset_cls(features, labels)
    return dataset, batch_size, shuffle


def make_loader(torch_module, dataloader_cls, built_loader, device: str):
    dataset, batch_size, shuffle = built_loader
    pin_memory = device == "cuda"
    return dataloader_cls(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def compute_loss(logits, labels, criterion):
    return criterion(logits, labels)


def evaluate_model(torch_module, model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> tuple[float, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    criterion = torch_module.nn.CrossEntropyLoss()

    with torch_module.no_grad():
        for start in range(0, len(y), batch_size):
            end = start + batch_size
            batch_x = torch_module.from_numpy(x[start:end].astype(np.float32)).to(device)
            batch_y = torch_module.from_numpy(y[start:end].astype(np.int64)).to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch_module.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            total_loss += float(loss.item()) * len(batch_y)
            total_samples += len(batch_y)

    probabilities = np.concatenate(all_probs, axis=0)
    average_loss = total_loss / max(total_samples, 1)
    return average_loss, probabilities


def run_tabtransformer_experiment(args: argparse.Namespace) -> dict[str, object]:
    torch, nn, DataLoader, TensorDataset, TabTransformerClassifier, TabTransformerConfig = resolve_torch()

    dataset = build_test_split_from_global(
        args.data_path,
        max_rows=args.max_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        random_sample=not args.sequential_sample,
        stratified=not args.non_stratified,
    )
    all_labels = list(dataset.y_train) + list(dataset.y_val) + list(dataset.y_test)
    num_classes = infer_num_classes(all_labels)
    if not args.allow_class_mismatch and num_classes != THESIS_MULTICLASS_TASK.expected_num_classes:
        raise ValueError(
            f"Expected {THESIS_MULTICLASS_TASK.expected_num_classes} classes for the thesis multiclass setup, found {num_classes}. "
            "Increase --max-rows or use a different sample."
        )

    class_weighting_enabled = not args.disable_class_weighted
    standardization_enabled = not args.disable_standardize
    class_weights = compute_class_weights(dataset.y_train, num_classes) if class_weighting_enabled else None

    if standardization_enabled:
        standardizer = fit_standardizer(dataset.x_train)
        dataset.x_train = standardizer.transform(dataset.x_train)
        dataset.x_val = standardizer.transform(dataset.x_val)
        dataset.x_test = standardizer.transform(dataset.x_test)

    device = choose_device(torch, args.device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TabTransformerClassifier(
        TabTransformerConfig(
            num_features=dataset.x_train.shape[1],
            num_classes=num_classes,
            d_token=args.d_token,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = make_loader(
        torch,
        DataLoader,
        build_dataloader(torch, TensorDataset, dataset.x_train, dataset.y_train, args.batch_size, True),
        device,
    )

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    patience_left = args.patience

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=device == "cuda")
            batch_y = batch_y.to(device, non_blocking=device == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = compute_loss(logits, batch_y, criterion)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
            seen_samples += len(batch_y)

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        history_entry = {"epoch": float(epoch), "train_loss": running_loss / max(seen_samples, 1)}
        if should_eval:
            val_loss, _ = evaluate_model(torch, model, dataset.x_val, dataset.y_val, args.batch_size, device)
            history_entry["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    history.append(history_entry)
                    break
        history.append(history_entry)

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - start_time
    _, test_probabilities = evaluate_model(torch, model, dataset.x_test, dataset.y_test, args.batch_size, device)
    metrics = evaluate_classification_detailed(dataset.y_test, test_probabilities)

    split_summary = {
        "train_samples": int(len(dataset.y_train)),
        "val_samples": int(len(dataset.y_val)),
        "test_samples": int(len(dataset.y_test)),
        "train_distribution": label_distribution(dataset.y_train, num_classes),
        "val_distribution": label_distribution(dataset.y_val, num_classes),
        "test_distribution": label_distribution(dataset.y_test, num_classes),
    }
    return {
        "mode": "centralized_tabtransformer",
        "task": THESIS_MULTICLASS_TASK.to_dict(),
        "data_path": str(args.data_path),
        "max_rows": args.max_rows,
        "training_config": {
            "input_dim": int(dataset.x_train.shape[1]),
            "output_dim": int(num_classes),
            "d_token": args.d_token,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ffn_dim": args.ffn_dim,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "device": device,
        },
        "preprocessing": {
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
        "split_summary": split_summary,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "training_time_seconds": training_time,
        "history": history,
        "test_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    ensure_results_dir(RESULTS_DIR)
    output = run_tabtransformer_experiment(args)
    output_path = RESULTS_DIR / "centralized_tabtransformer_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
