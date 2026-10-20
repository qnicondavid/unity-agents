import argparse
import csv
import random

from batch_run import sample_configs
from collect_run import CSV_FILE, RESULTS_DIR, run_and_log_experiment


def logged_runs() -> dict[str, dict]:
    if not CSV_FILE.exists():
        return {}
    with open(CSV_FILE, newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def check_pairing(configs: list[dict], logged: dict[str, dict], pair_with: str, seeds: list[int]) -> int:
    checked = 0
    for idx, config in enumerate(configs):
        for seed in seeds:
            run_id = f"{pair_with}_c{idx:02d}_s{seed}"
            if run_id not in logged:
                continue
            for name, value in config.items():
                if float(logged[run_id][name]) != float(value):
                    raise ValueError(f"{run_id} has {name} = {logged[run_id][name]}, but config {idx} has {value}")
            checked += 1
    return checked


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--batch-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--pair-with", default="3DBall")
    parser.add_argument("--pair-seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = sample_configs(args.n_configs, random.Random(args.batch_seed))
    logged = logged_runs()
    checked = check_pairing(configs, logged, args.pair_with, args.pair_seeds)
    print(f"pairing checked against {checked} runs of {args.pair_with}")

    started = 0
    for seed in args.seeds:
        for idx, config in enumerate(configs):
            run_id = f"{args.name}_c{idx:02d}_s{seed}"
            if run_id in logged:
                continue
            if (RESULTS_DIR / run_id).exists():
                print(f"[SKIP] {run_id}: results/{run_id} exists but the run is not logged")
                continue
            if args.limit is not None and started >= args.limit:
                print(f"stopped after {started} runs (--limit)")
                return 0
            run_and_log_experiment(args.config, run_id, seed=seed,
                                   tuneables={**config, "max_steps": args.max_steps},
                                   threshold=args.threshold, env_path=args.env_path)
            started += 1
    print(f"{started} runs started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
