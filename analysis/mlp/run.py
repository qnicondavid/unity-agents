import math
import statistics
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.core import cv, experiment
from analysis.mlp import network

METRIC = "mean_absolute_error"


def own_network(settings: dict, seed: int) -> tuple:
    def train(rows, targets):
        return network.train_network(rows, targets, settings["hidden_sizes"], settings["learning_rate"],
                                     settings["batch_size"], settings["epochs"], seed,
                                     dropout=settings.get("dropout", 0.0))

    def predict(model, rows):
        return network.predict(model, rows)

    return train, predict


def scikit_learn_network(settings: dict, seed: int) -> tuple:
    def train(rows, targets):
        regressor = MLPRegressor(hidden_layer_sizes=tuple(settings["hidden_sizes"]), solver="sgd",
                                 learning_rate_init=settings["learning_rate"],
                                 batch_size=settings["batch_size"], max_iter=settings["epochs"],
                                 n_iter_no_change=settings["epochs"], momentum=0.0,
                                 nesterovs_momentum=False, alpha=0.0, random_state=seed)
        model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), regressor),
                                           transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(np.array(rows), np.array(targets))
        return model

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def ridge() -> tuple:
    def train(rows, targets):
        return make_pipeline(StandardScaler(), Ridge()).fit(np.array(rows), np.array(targets))

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def median_baseline(column: int | None) -> tuple:
    def key(row):
        return None if column is None else row[column]

    def train(rows, targets):
        groups = {}
        for row, target in zip(rows, targets):
            groups.setdefault(key(row), []).append(target)
        medians = {value: statistics.median(group) for value, group in groups.items()}
        return medians, statistics.median(targets)

    def predict(model, rows):
        medians, overall = model
        return [medians.get(key(row), overall) for row in rows]

    return train, predict


def transform(rows: list, targets: list, config: dict) -> tuple[list, list]:
    logged = {config["features"].index(name) for name in config.get("log_features", [])}
    rows = [[math.log(value) if i in logged else value for i, value in enumerate(row)] for row in rows]
    if config.get("log_target", False):
        targets = [math.log(target) for target in targets]
    return rows, targets


def summarize(targets: list, predictions: dict, folds: list) -> dict:
    runs = {}
    for (name, _), predicted in predictions.items():
        runs.setdefault(name, []).append(experiment.fold_scores(targets, predicted, folds,
                                                                cv.mean_absolute_error))
    scores = {name: [sum(values) / len(values) for values in zip(*seeds)] for name, seeds in runs.items()}
    return {"scores": scores}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    rows, targets = transform(rows, targets, config)
    settings = config["mlp"]
    median_by = config.get("median_by")
    column = config["features"].index(median_by) if median_by else None
    models = {}
    for seed in settings["seeds"]:
        models[("mlp", seed)] = own_network(settings, seed)
        models[("scikit_learn", seed)] = scikit_learn_network(settings, seed)
    models[("ridge", None)] = ridge()
    models[("median", None)] = median_baseline(column)
    return experiment.evaluate(rows, targets, groups, reports, config["cross_validation"], models,
                               summarize)


def main(argv: list[str] | None = None) -> int:
    return experiment.run(argv, "mlp", METRIC, evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
