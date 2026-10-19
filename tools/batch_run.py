"""
Run random hyperparameter batch for the 3DBall PPO experiments.

Each sampled config is paired with every seed in --seeds and logged to a CSV
file via run_and_log_experiment. If run already present in log, we skip it
"""

import argparse
import csv
import random
from pathlib import Path

from collect_run import CSV_FILE, run_and_log_experiment

# Search space for the batch
HYPERPARAM_GRID = {
    "batch_size": [32, 64, 128, 256],
    "buffer_size": [2048, 4096, 8192, 16384],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden_units": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "epsilon": [0.1, 0.2, 0.3],
    "beta": [1e-3, 5e-3, 1e-2],
    "num_epoch": [3, 5, 8],
}


def sample_configs(count: int, rng: random.Random) -> list[dict]:
    #Distinct count of configs, each with unique random params from grid
    seen = set()
    while len(seen) < count:
        config = tuple(rng.choice(values) for values in HYPERPARAM_GRID.values())
        seen.add(config)
    return [dict(zip(HYPERPARAM_GRID, c)) for c in seen]


def already_logged_run_ids(csv_path: Path) -> set[str]:
    # Runs ids from previous runs so we dont repeat
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random hyperparameter batch for 3DBall PPO runs.",
    )
    parser.add_argument("--n-configs", type=int, default=8,
                        help="how many distinct configs to sample")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                        help="seeds to run each config with")
    parser.add_argument("--batch-seed", type=int, default=42,
                        help="seed for sampling the batch itself")
    parser.add_argument("--config", default="config/ppo/3DBall.yaml",
                        help="path to the training config")
    parser.add_argument("--max-steps", type=int, default=500_000,
                        help="training steps passed as an override")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="success threshold used by run_and_log_experiment")
    parser.add_argument("--env-path", default=None,
                        help="optional Unity environment path")
    return parser.parse_args()


def main() -> None:
    print("[DEBUG] batch script started...")
    args = parse_args()
    print(f"[DEBUG] Configs to sample: {args.n_configs}, Seeds: {args.seeds}")

    rng = random.Random(args.batch_seed)
    configs = sample_configs(args.n_configs, rng)
    done = already_logged_run_ids(CSV_FILE)
    print(f"[DEBUG] Already logged runs: {done}")

    for config_idx, config in enumerate(configs):
        for seed in args.seeds:
            run_id = f"3DBall_c{config_idx:02d}_s{seed}"
            print(f"[DEBUG] Checking run_id: {run_id}")
            if run_id in done:
                print(f"[SKIP] {run_id} already logged")
                continue

            print(f"[DEBUG] Launching run_and_log_experiment for {run_id}")
            run_and_log_experiment(
                args.config,
                run_id,
                seed=seed,
                tuneables={**config, "max_steps": args.max_steps},
                threshold=args.threshold,
                env_path=args.env_path,
            )


if __name__ == "__main__":
    main()