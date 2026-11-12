import argparse
from pathlib import Path

import pandas as pd

from analysis.core import cv, experiment
from analysis.gpr import gpr
from analysis.gpr.run import METRIC, check_kernel, evaluate, own_gpr, transform, write_length_scales

KEYS = ["kernel", "ard"]


def check_grid(config: dict) -> None:
    for kind in config["tuning"]["kernel"]:
        check_kernel(kind)


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    rows, targets = transform(rows, targets, config)
    train, predict = own_gpr(candidate["kernel"], candidate["ard"], config["gpr"].get("steps", 200))
    predictions = cv.cross_validation(rows, targets, folds, train, predict)
    return experiment.fold_scores(targets, predictions, folds, cv.mean_absolute_error)


# ties go to one shared length scale, then to the smoother kernel
def simplicity(candidate: dict) -> tuple:
    return (candidate["ard"], gpr.KERNELS.index(candidate["kernel"]))


def main(argv: list[str] | None = None) -> int:
    code = experiment.run_tuning(argv, "gpr", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 better=min, check=check_grid)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    config = experiment.read_config(args.config)
    folder = experiment.RESULTS / "gpr" / args.config.stem / "tuned_best_mean"
    split_by = experiment.split_column(config)
    chosen = pd.read_csv(folder / "summary.csv")
    steps = config["gpr"].get("steps", 200)
    settings = {row[split_by]: (row["kernel"], bool(row["ard"]), steps) for _, row in chosen.iterrows()}
    write_length_scales(config, settings, folder)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
