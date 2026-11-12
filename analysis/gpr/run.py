import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.core import cv, experiment
from analysis.gpr import gpr

METRIC = "mean_absolute_error"
NU = {"matern32": 1.5, "matern52": 2.5}


def own_gpr(kind: str, ard: bool, steps: int) -> tuple:
    def train(rows, targets):
        return gpr.train_gpr(rows, targets, kind, ard, steps)

    def predict(model, rows):
        return gpr.predict(model, rows)

    return train, predict


def scikit_learn_gpr(kind: str, ard: bool) -> tuple:
    def train(rows, targets):
        scales = np.ones(len(rows[0])) if ard else 1.0
        base = RBF(scales) if kind == "rbf" else Matern(scales, nu=NU[kind])
        model = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * base + WhiteKernel(0.1),
                                         normalize_y=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return make_pipeline(StandardScaler(), model).fit(np.array(rows), np.array(targets))

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def transform(rows: list, targets: list, config: dict) -> tuple[list, list]:
    logged = {config["features"].index(name) for name in config.get("log_features", [])}
    rows = [[math.log(value) if i in logged else value for i, value in enumerate(row)] for row in rows]
    if config.get("log_target", False):
        targets = [math.log(target) for target in targets]
    return rows, targets


def check_kernel(kind: str) -> None:
    if kind not in gpr.KERNELS:
        raise ValueError(f"kernel is {kind!r}, but it must be one of {', '.join(gpr.KERNELS)}")


def settings_of(config: dict) -> tuple[str, bool, int]:
    settings = config["gpr"]
    kind = settings.get("kernel", "matern52")
    check_kernel(kind)
    return kind, settings.get("ard", True), settings.get("steps", 200)


def summarize(targets: list, predictions: dict, folds: list) -> dict:
    return {"scores": {name: experiment.fold_scores(targets, predicted, folds, cv.mean_absolute_error)
                       for name, predicted in predictions.items()}}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    rows, targets = transform(rows, targets, config)
    kind, ard, steps = settings_of(config)
    models = {"gpr": own_gpr(kind, ard, steps), "scikit_learn": scikit_learn_gpr(kind, ard)}
    return experiment.evaluate(rows, targets, groups, reports, config["cross_validation"], models,
                               summarize)


# Fits our GPR on all runs of each split and writes what it learned to length_scales.csv:
# a short length scale means the result changes quickly with that setting, so it matters more.
def write_length_scales(config: dict, settings: dict, folder: Path) -> None:
    split_by = experiment.split_column(config)
    table = []
    for value, rows, targets, _, _ in experiment.load_groups(config):
        kind, ard, steps = settings[value]
        rows, targets = transform(rows, targets, config)
        learned = gpr.learned_settings(gpr.train_gpr(rows, targets, kind, ard, steps),
                                       config["features"])
        table.append({split_by: value, "kernel": kind, "ard": ard,
                      "signal_sd": round(learned["signal_sd"], 6),
                      "noise_sd": round(learned["noise_sd"], 6),
                      **{name: round(scale, 6) for name, scale in learned["length_scales"].items()}})
    pd.DataFrame(table).to_csv(folder / "length_scales.csv", index=False, lineterminator="\n")
    print(f"length scales written to {folder / 'length_scales.csv'}")


def main(argv: list[str] | None = None) -> int:
    code = experiment.run(argv, "gpr", METRIC, evaluate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = experiment.read_config(args.config)
    values = [part[0] for part in experiment.load_groups(config)]
    write_length_scales(config, {value: settings_of(config) for value in values},
                        experiment.RESULTS / "gpr" / args.config.stem)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
