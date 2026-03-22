from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ciciot.config import DATA_DIR, FEDERATED_DIR, RESULTS_DIR, FederatedConfig, TrainingConfig
from ciciot.data import (
    build_test_split_from_global,
    compute_class_weights,
    ensure_results_dir,
    fit_standardizer,
    infer_num_classes,
    label_distribution,
    load_client_frames,
    stratified_train_val_split,
    split_features_labels,
    train_val_split,
)
from ciciot.metrics import evaluate_classification_detailed
from ciciot.models.mlp_numpy import NumpyMLP
from ciciot.tasks import THESIS_MULTICLASS_TASK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated FedAvg on CICIoT2023")
    parser.add_argument("--global-data-path", type=Path, default=DATA_DIR / "global.csv")
    parser.add_argument("--client-dir", type=Path, default=FEDERATED_DIR)
    parser.add_argument("--test-rows", type=int, default=100000)
    parser.add_argument("--client-max-rows", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--skip-train-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-class-weighted", action="store_true")
    parser.add_argument("--disable-standardize", action="store_true")
    parser.add_argument("--sequential-sample", action="store_true")
    parser.add_argument("--non-stratified", action="store_true")
    parser.add_argument("--allow-class-mismatch", action="store_true")
    return parser.parse_args(argv)


def average_parameters(weighted_params: list[tuple[int, list[np.ndarray]]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for num_examples, _ in weighted_params)
    averaged: list[np.ndarray] = []
    for param_idx in range(len(weighted_params[0][1])):
        weighted_sum = sum(num_examples * params[param_idx] for num_examples, params in weighted_params)
        averaged.append(weighted_sum / total_examples)
    return averaged


def run_federated_experiment(args: argparse.Namespace) -> dict[str, object]:
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

    training_config = TrainingConfig(
        input_dim=test_split.x_train.shape[1],
        output_dim=num_classes,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.local_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    federated_config = FederatedConfig(rounds=args.rounds, local_epochs=args.local_epochs, seed=args.seed)

    client_frames = load_client_frames(
        args.client_dir,
        max_rows_per_client=args.client_max_rows,
        random_sample=not args.sequential_sample,
        seed=args.seed,
    )
    global_model = NumpyMLP(
        input_dim=training_config.input_dim,
        hidden_dims=training_config.hidden_dims,
        output_dim=training_config.output_dim,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        seed=training_config.seed,
    )

    round_history: list[dict[str, float]] = []
    client_summaries: list[dict[str, object]] = []
    start_time = time.time()

    for round_idx in range(1, federated_config.rounds + 1):
        client_updates: list[tuple[int, list[np.ndarray]]] = []
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

            local_model = NumpyMLP(
                input_dim=training_config.input_dim,
                hidden_dims=training_config.hidden_dims,
                output_dim=training_config.output_dim,
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                seed=training_config.seed + round_idx,
                class_weights=class_weights,
                focal_gamma=args.focal_gamma,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            local_model.set_parameters(global_model.get_parameters())
            local_model.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=federated_config.local_epochs,
                batch_size=training_config.batch_size,
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
                        "train_distribution": label_distribution(y_train, num_classes),
                        "val_distribution": label_distribution(y_val, num_classes),
                    }
                )

        global_model.set_parameters(average_parameters(client_updates))
        test_metrics = evaluate_classification_detailed(test_split.y_test, global_model.predict_proba(test_split.x_test))
        round_history.append({"round": float(round_idx), **test_metrics})
        print(json.dumps({"round": round_idx, "metrics": test_metrics}))

    training_time = time.time() - start_time
    split_summary = {
        "train_samples": int(len(test_split.y_train)),
        "val_samples": int(len(test_split.y_val)),
        "test_samples": int(len(test_split.y_test)),
        "train_distribution": label_distribution(test_split.y_train, num_classes),
        "val_distribution": label_distribution(test_split.y_val, num_classes),
        "test_distribution": label_distribution(test_split.y_test, num_classes),
    }
    output = {
        "mode": "federated",
        "task": THESIS_MULTICLASS_TASK.to_dict(),
        "global_data_path": str(args.global_data_path),
        "client_dir": str(args.client_dir),
        "test_rows": args.test_rows,
        "client_max_rows": args.client_max_rows,
        "training_config": {
            "input_dim": training_config.input_dim,
            "output_dim": training_config.output_dim,
            "hidden_dims": training_config.hidden_dims,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "weight_decay": training_config.weight_decay,
            "focal_gamma": args.focal_gamma,
            "gradient_clip_norm": args.gradient_clip_norm,
            "eval_every": args.eval_every,
            "skip_train_loss": args.skip_train_loss,
            "local_epochs": federated_config.local_epochs,
            "rounds": federated_config.rounds,
            "seed": training_config.seed,
        },
        "preprocessing": {
            "random_sample": not args.sequential_sample,
            "stratified_split": not args.non_stratified,
            "standardize": standardization_enabled,
            "class_weighted": class_weighting_enabled,
            "patience": args.patience,
        },
        "split_summary": split_summary,
        "client_summaries": client_summaries,
        "training_time_seconds": training_time,
        "round_history": round_history,
        "final_test_metrics": round_history[-1] if round_history else {},
    }
    return output


def main() -> None:
    args = parse_args()
    ensure_results_dir(RESULTS_DIR)
    output = run_federated_experiment(args)

    output_path = RESULTS_DIR / "federated_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
