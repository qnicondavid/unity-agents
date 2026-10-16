import argparse
import tempfile
from pathlib import Path

import pandas as pd
import py7zr

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = (REPO_ROOT / "external" / "BCS2720-project-data-collection" / "AY 2025-2026"
           / "Group 15" / "Group15_data.7z")
OUTPUT = REPO_ROOT / "data" / "group15.csv"

THRESHOLD = 95
BUDGETS = (500_000, 1_000_000, 2_000_000)
SUMMARY_FREQ = 12_000

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


def first_step_at(training: pd.DataFrame, threshold: float) -> pd.Series:
    above = training[training["cumulative_reward"] >= threshold]
    return above.groupby("RunID")["Step"].min()


def last_logged_step(training: pd.DataFrame) -> pd.Series:
    return training.groupby("RunID")["Step"].max()


def expected_last_step(max_steps: int) -> int:
    return (max_steps // SUMMARY_FREQ) * SUMMARY_FREQ


def compute_outcomes(runs: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    runs = runs.set_index("RunID")
    first = first_step_at(training, THRESHOLD).reindex(runs.index)
    last = last_logged_step(training).reindex(runs.index)
    ended = last >= expected_last_step(runs["max_steps"])

    reached = pd.Series(pd.NA, index=runs.index, dtype="boolean")
    reached[first.notna()] = True
    reached[first.isna() & ended] = False

    return pd.DataFrame({
        "last_logged_step": last,
        "reached_95": reached,
        "steps_to_95": first.astype("Int64"),
    })


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
