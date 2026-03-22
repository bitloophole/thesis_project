from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ciciot.config import DATA_DIR, RESULTS_DIR, TrainingConfig
from ciciot.data import (
    build_test_split_from_global,
    compute_class_weights,
    ensure_results_dir,
    fit_standardizer,
    infer_num_classes,
    label_distribution,
    oversample_training_data,
)
from ciciot.metrics import evaluate_classification_detailed
from ciciot.models.mlp_numpy import NumpyMLP
from ciciot.tasks import THESIS_MULTICLASS_TASK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a centralized NumPy MLP on CICIoT2023")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR / "global.csv")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-class-weighted", action="store_true")
    parser.add_argument("--disable-standardize", action="store_true")
    parser.add_argument("--sequential-sample", action="store_true")
    parser.add_argument("--non-stratified", action="store_true")
    parser.add_argument("--allow-class-mismatch", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--oversample-train", action="store_true")
    parser.add_argument("--oversample-target-fraction", type=float, default=1.0)
    return parser.parse_args(argv)


def run_centralized_experiment(args: argparse.Namespace) -> dict[str, object]:
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
    oversampling_enabled = args.oversample_train
    original_train_distribution = label_distribution(dataset.y_train, num_classes)
    if oversampling_enabled:
        dataset.x_train, dataset.y_train = oversample_training_data(
            dataset.x_train,
            dataset.y_train,
            num_classes=num_classes,
            seed=args.seed,
            target_fraction=args.oversample_target_fraction,
        )
    effective_train_distribution = label_distribution(dataset.y_train, num_classes)
    class_weights = compute_class_weights(dataset.y_train, num_classes) if class_weighting_enabled else None

    if standardization_enabled:
        standardizer = fit_standardizer(dataset.x_train)
        dataset.x_train = standardizer.transform(dataset.x_train)
        dataset.x_val = standardizer.transform(dataset.x_val)
        dataset.x_test = standardizer.transform(dataset.x_test)

    config = TrainingConfig(
        input_dim=dataset.x_train.shape[1],
        output_dim=num_classes,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    model = NumpyMLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    start_time = time.time()
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        dataset.x_val,
        dataset.y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        patience=args.patience,
    )
    training_time = time.time() - start_time

    metrics = evaluate_classification_detailed(dataset.y_test, model.predict_proba(dataset.x_test))
    split_summary = {
        "train_samples": int(len(dataset.y_train)),
        "val_samples": int(len(dataset.y_val)),
        "test_samples": int(len(dataset.y_test)),
        "train_distribution": effective_train_distribution,
        "val_distribution": label_distribution(dataset.y_val, num_classes),
        "test_distribution": label_distribution(dataset.y_test, num_classes),
        "original_train_distribution": original_train_distribution,
    }
    output = {
        "mode": "centralized",
        "task": THESIS_MULTICLASS_TASK.to_dict(),
        "data_path": str(args.data_path),
        "max_rows": args.max_rows,
        "training_config": {
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dims": config.hidden_dims,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "focal_gamma": args.focal_gamma,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": config.seed,
        },
        "preprocessing": {
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "oversample_train": oversampling_enabled,
            "oversample_target_fraction": args.oversample_target_fraction,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "patience": args.patience,
        },
        "split_summary": split_summary,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "training_time_seconds": training_time,
        "history": history,
        "test_metrics": metrics,
    }
    return output


def main() -> None:
    args = parse_args()
    ensure_results_dir(RESULTS_DIR)
    output = run_centralized_experiment(args)

    output_path = RESULTS_DIR / "centralized_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
