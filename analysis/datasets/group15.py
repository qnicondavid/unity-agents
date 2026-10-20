import argparse
import tempfile
from pathlib import Path

import pandas as pd
import py7zr

from analysis.datasets.outcomes import compute_outcomes

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (REPO_ROOT / "external" / "BCS2720-project-data-collection" / "AY 2025-2026"
           / "Group 15" / "Group15_data.7z")
OUTPUT = REPO_ROOT / "data" / "group15.csv"

BUDGETS = (500_000, 1_000_000, 2_000_000)

SETTINGS = [
    "batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers",
    "beta", "epsilon", "lambd", "num_epoch", "time_horizon",
    "learning_rate_schedule", "normalize",
]
HARDWARE = ["os", "cpu", "physical_cores", "system_ram_gb", "gpu", "GPU_ram_mb"]


def read_archive(archive: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(path=tmp)
        static = pd.read_csv(Path(tmp) / "static_rows.csv")
        training = pd.read_csv(Path(tmp) / "training_rows.csv",
                               usecols=["RunID", "Step", "cumulative_reward"])
    return static, training


def select_runs(static: pd.DataFrame) -> pd.DataFrame:
    is_ppo = static["training_type"] == "ppo"
    is_3dball = static["game_name"] == "3DBall"
    in_budget = static["max_steps"].isin(BUDGETS)
    return static[is_ppo & is_3dball & in_budget]


def build_examples(runs: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["RunID", "training_type", "game_name", *SETTINGS, "max_steps", *HARDWARE]
    examples = runs[columns].join(outcomes, on="RunID")
    examples = examples.rename(columns={"system_ram_gb": "system_ram_bytes"})
    return examples.convert_dtypes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    static, training = read_archive(args.archive)
    runs = select_runs(static)
    outcomes = compute_outcomes(runs, training)
    examples = build_examples(runs, outcomes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.out, index=False, lineterminator="\n")
    print(f"{len(examples)} runs written to {args.out}")
    print(outcomes["reached_95"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
