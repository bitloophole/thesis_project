from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from train_centralized import parse_args, run_centralized_experiment


def build_candidates(base_args: argparse.Namespace) -> list[argparse.Namespace]:
    hidden_grid = [
        [256, 128, 64],
        [512, 256, 128],
        [256, 256, 128],
    ]
    learning_rates = [0.001, 0.0005]
    weight_decays = [1e-4, 5e-4]
    focal_gammas = [0.0, 1.5]

    candidates: list[argparse.Namespace] = []
    for hidden_dims, learning_rate, weight_decay, focal_gamma in itertools.product(
        hidden_grid,
        learning_rates,
        weight_decays,
        focal_gammas,
    ):
        candidate = deepcopy(base_args)
        candidate.hidden_dims = hidden_dims
        candidate.learning_rate = learning_rate
        candidate.weight_decay = weight_decay
        candidate.focal_gamma = focal_gamma
        candidates.append(candidate)
    return candidates


def parse_sweep_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a centralized MLP sweep for CICIoT2023")
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--sort-metric", type=str, default="f1_macro")
    parser.add_argument("--output-path", type=Path, default=Path("results") / "centralized_sweep.json")
    args, remaining = parser.parse_known_args()
    args.training_argv = remaining
    return args


def main() -> None:
    sweep_args = parse_sweep_args()
    base_args = parse_args(sweep_args.training_argv)
    candidates = build_candidates(base_args)[: sweep_args.max_trials]

    results: list[dict[str, object]] = []
    for trial_index, candidate in enumerate(candidates, start=1):
        result = run_centralized_experiment(candidate)
        result["trial_index"] = trial_index
        results.append(result)
        score = result["test_metrics"][sweep_args.sort_metric]
        print(f"trial={trial_index} {sweep_args.sort_metric}={score:.6f} hidden={candidate.hidden_dims} lr={candidate.learning_rate} wd={candidate.weight_decay} focal={candidate.focal_gamma}")

    ranked = sorted(results, key=lambda item: item["test_metrics"][sweep_args.sort_metric], reverse=True)
    summary = {
        "sort_metric": sweep_args.sort_metric,
        "num_trials": len(ranked),
        "best_trial": ranked[0] if ranked else None,
        "results": ranked,
    }
    sweep_args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved sweep results to {sweep_args.output_path}")


if __name__ == "__main__":
    main()
