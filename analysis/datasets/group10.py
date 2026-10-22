import argparse
from pathlib import Path

import pandas as pd
import yaml

from analysis.datasets.outcomes import compute_outcomes

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG = REPO_ROOT / "data" / "group10_2026.csv"
CONFIGS = REPO_ROOT / "data" / "configs"
CURVES = REPO_ROOT / "data" / "curves"
OUTPUT = REPO_ROOT / "data" / "group10.csv"

BUDGETS = (500_000,)

SETTINGS = [
    "batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers",
    "beta", "epsilon", "lambd", "num_epoch", "time_horizon",
    "learning_rate_schedule", "normalize",
]
HARDWARE = ["cpu_model", "cpu_cores", "ram_gb_total", "gpu_model"]


def read_settings(path: Path, behavior_name: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    behavior = config["behaviors"][behavior_name]
    values = {**behavior, **behavior["hyperparameters"], **behavior["network_settings"]}
    settings = {name: values[name] for name in SETTINGS}
    ball = "random" if config.get("environment_parameters") else "fixed"
    return {"training_type": behavior["trainer_type"], "ball": ball, **settings,
            "max_steps": behavior["max_steps"]}


def read_runs(log: Path, configs: Path) -> pd.DataFrame:
    logged = pd.read_csv(log, keep_default_na=False)
    settings = pd.DataFrame([read_settings(configs / f"{run.run_id}.yaml", run.environment_name)
                             for run in logged.itertuples()])
    runs = pd.concat([logged[["run_id", "environment_name", "seed", *HARDWARE]], settings], axis=1)
    return runs.rename(columns={"run_id": "RunID", "environment_name": "game_name"})


def read_curves(curves: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(curves.glob("*.csv")):
        curve = pd.read_csv(path, usecols=["step", "environment_cumulative_reward"])
        frames.append(curve.assign(RunID=path.stem))
    training = pd.concat(frames, ignore_index=True)
    return training.rename(columns={"step": "Step", "environment_cumulative_reward": "cumulative_reward"})


def select_runs(runs: pd.DataFrame) -> pd.DataFrame:
    is_ppo = runs["training_type"] == "ppo"
    is_3dball = runs["game_name"] == "3DBall"
    in_budget = runs["max_steps"].isin(BUDGETS)
    return runs[is_ppo & is_3dball & in_budget]


def build_examples(runs: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["RunID", "training_type", "game_name", "ball", *SETTINGS, "max_steps", "seed", *HARDWARE]
    examples = runs[columns].join(outcomes, on="RunID")
    return examples.convert_dtypes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    runs = select_runs(read_runs(LOG, CONFIGS))
    training = read_curves(CURVES)
    outcomes = compute_outcomes(runs, training)
    examples = build_examples(runs, outcomes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.out, index=False, lineterminator="\n")
    print(f"{len(examples)} runs written to {args.out}")
    print(examples.groupby("ball")["reached_95"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
