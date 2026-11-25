import math

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from analysis.boosting import boosting
from analysis.core import cv, experiment

METRIC = "mean_absolute_error"


def own_boosting(trees: int, learning_rate: float, max_depth: int, min_leaf: int, subsample: float,
                 seed: int) -> tuple:
    def train(rows, targets):
        return boosting.build_boosting(rows, targets, trees, learning_rate, max_depth, min_leaf,
                                       subsample, seed)

    def predict(model, rows):
        return [boosting.predict_boosting(model, row) for row in rows]

    return train, predict


def scikit_learn_boosting(trees: int, learning_rate: float, max_depth: int, min_leaf: int,
                          subsample: float, seed: int) -> tuple:
    def train(rows, targets):
        model = GradientBoostingRegressor(loss="squared_error", learning_rate=learning_rate,
                                          n_estimators=trees, max_depth=max_depth,
                                          min_samples_leaf=min_leaf, subsample=subsample,
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


def settings_of(settings: dict) -> tuple:
    return (settings["trees"], settings["learning_rate"], settings.get("max_depth", 3),
            settings.get("min_leaf", 1), settings.get("subsample", 1.0), settings["seed"])


def summarize(targets: list, predictions: dict, folds: list) -> dict:
    return {"scores": {name: experiment.fold_scores(targets, predicted, folds, cv.mean_absolute_error)
                       for name, predicted in predictions.items()}}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    rows, targets = transform(rows, targets, config)
    parameters = settings_of(config["boosting"])
    boosting.check_settings(*parameters[:5])
    models = {"boosting": own_boosting(*parameters), "scikit_learn": scikit_learn_boosting(*parameters)}
    return experiment.evaluate(rows, targets, groups, reports, config["cross_validation"], models,
                               summarize)


def main(argv: list[str] | None = None) -> int:
    return experiment.run(argv, "boosting", METRIC, evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
