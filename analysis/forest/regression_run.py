import math

import numpy as np
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from analysis.core import cv, experiment
from analysis.forest import regression

METRIC = "mean_absolute_error"
SAMPLINGS = ("split", "tree")


def own_forest(trees: int, m: int, max_depth: int | None, min_leaf: int, sampling: str,
               seed: int) -> tuple:
    def train(rows, targets):
        return regression.build_forest(rows, targets, trees, m, max_depth, min_leaf, sampling, seed)

    def predict(model, rows):
        return [regression.predict_forest(model, row) for row in rows]

    return train, predict


def scikit_learn_forest(trees: int, m: int, max_depth: int | None, min_leaf: int, sampling: str,
                        seed: int) -> tuple:
    def train(rows, targets):
        if sampling == "split":
            model = RandomForestRegressor(n_estimators=trees, max_features=m, max_depth=max_depth,
                                          min_samples_leaf=min_leaf, random_state=seed)
        else:
            tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf)
            model = BaggingRegressor(estimator=tree, n_estimators=trees, max_features=m,
                                     random_state=seed)
        return model.fit(np.array(rows), np.array(targets))

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def transform(rows: list, targets: list, config: dict) -> tuple[list, list]:
    logged = {config["features"].index(name) for name in config.get("log_features", [])}
    rows = [[math.log(value) if i in logged else value for i, value in enumerate(row)] for row in rows]
    if config.get("log_target", False):
        targets = [math.log(target) for target in targets]
    return rows, targets


def check_settings(m: int, min_leaf: int, sampling: str, features: int) -> None:
    if m > features:
        raise ValueError(f"attributes_per_tree is {m}, but there are only {features} features")
    if min_leaf < 1:
        raise ValueError(f"min_leaf is {min_leaf}, but it must be at least 1")
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling is {sampling!r}, but it must be \"split\" or \"tree\"")


def summarize(targets: list, predictions: dict, folds: list) -> dict:
    return {"scores": {name: experiment.fold_scores(targets, predicted, folds, cv.mean_absolute_error)
                       for name, predicted in predictions.items()}}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    rows, targets = transform(rows, targets, config)
    settings = config["forest"]
    m = settings["attributes_per_tree"]
    min_leaf = settings.get("min_leaf", 1)
    sampling = settings.get("sampling", "split")
    check_settings(m, min_leaf, sampling, len(rows[0]))
    parameters = (settings["trees"], m, settings.get("max_depth"), min_leaf, sampling, settings["seed"])
    models = {"forest": own_forest(*parameters), "scikit_learn": scikit_learn_forest(*parameters)}
    return experiment.evaluate(rows, targets, groups, reports, config["cross_validation"], models,
                               summarize)


def main(argv: list[str] | None = None) -> int:
    return experiment.run(argv, "forest", METRIC, evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
