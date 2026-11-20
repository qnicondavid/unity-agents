import math

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from analysis.core import cv, experiment
from analysis.gpr import gpr
from analysis.knn import knn

METRIC = "mean_absolute_error"
DISTANCES = ("standard", "gpr")


def gpr_scales(rows: list, targets: list, kernel: str, steps: int) -> list:
    model = gpr.train_gpr(rows, targets, kernel, True, steps)
    return gpr.unpack(model["log_params"])[0].tolist()


def own_knn(k: int, p: float, distance: str, kernel: str, steps: int) -> tuple:
    def train(rows, targets):
        scales = gpr_scales(rows, targets, kernel, steps) if distance == "gpr" else None
        return knn.train_knn(rows, targets, k, p, scales)

    def predict(model, rows):
        return knn.predict(model, rows)

    return train, predict


def scikit_learn_knn(k: int, p: float, distance: str, kernel: str, steps: int) -> tuple:
    def train(rows, targets):
        scales = np.array(gpr_scales(rows, targets, kernel, steps)) if distance == "gpr" else 1.0
        if math.isinf(p):
            model = KNeighborsRegressor(n_neighbors=k, metric="chebyshev")
        else:
            model = KNeighborsRegressor(n_neighbors=k, p=p)
        rescale = FunctionTransformer(lambda X: X / scales)
        return make_pipeline(StandardScaler(), rescale, model).fit(np.array(rows), np.array(targets))

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def transform(rows: list, targets: list, config: dict) -> tuple[list, list]:
    logged = {config["features"].index(name) for name in config.get("log_features", [])}
    rows = [[math.log(value) if i in logged else value for i, value in enumerate(row)] for row in rows]
    if config.get("log_target", False):
        targets = [math.log(target) for target in targets]
    return rows, targets


def check_settings(k: int, p: float) -> None:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k is {k!r}, but it must be a whole number of at least 1")
    if p < 1:
        raise ValueError(f"p is {p}, but it must be at least 1")


def check_distance(distance: str, kernel: str) -> None:
    if distance not in DISTANCES:
        raise ValueError(f"distance is {distance!r}, but it must be \"standard\" or \"gpr\"")
    if kernel not in gpr.KERNELS:
        raise ValueError(f"kernel is {kernel!r}, but it must be one of {', '.join(gpr.KERNELS)}")


def settings_of(config: dict) -> tuple[int, float]:
    settings = config["knn"]
    k = settings.get("k", 5)
    p = settings.get("p", 2)
    check_settings(k, p)
    return k, p


def distance_of(config: dict) -> tuple[str, str, int]:
    settings = config["knn"]
    distance = settings.get("distance", "standard")
    kernel = settings.get("kernel", "matern32")
    check_distance(distance, kernel)
    return distance, kernel, settings.get("steps", 200)


def summarize(targets: list, predictions: dict, folds: list) -> dict:
    return {"scores": {name: experiment.fold_scores(targets, predicted, folds, cv.mean_absolute_error)
                       for name, predicted in predictions.items()}}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    rows, targets = transform(rows, targets, config)
    k, p = settings_of(config)
    distance = distance_of(config)
    models = {"knn": own_knn(k, p, *distance), "scikit_learn": scikit_learn_knn(k, p, *distance)}
    return experiment.evaluate(rows, targets, groups, reports, config["cross_validation"], models,
                               summarize)


def main(argv: list[str] | None = None) -> int:
    return experiment.run(argv, "knn", METRIC, evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
