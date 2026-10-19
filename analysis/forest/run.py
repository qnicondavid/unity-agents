import numpy as np
from sklearn.ensemble import RandomForestClassifier

from analysis.core import cv, experiment
from analysis.forest import forest

METRIC = "balanced_accuracy"


def own_forest(trees: int, m: int, max_depth: int | None, seed: int) -> tuple:
    def train(rows, labels):
        return forest.build_forest(rows, labels, trees, m, max_depth, seed)

    def predict(model, rows):
        return [forest.predict_forest(model, row) for row in rows]

    return train, predict


def scikit_learn_forest(trees: int, m: int, max_depth: int | None, seed: int) -> tuple:
    def train(rows, labels):
        model = RandomForestClassifier(n_estimators=trees, max_features=m, criterion="entropy",
                                       max_depth=max_depth, random_state=seed)
        return model.fit(np.array(rows), np.array(labels))

    def predict(model, rows):
        return model.predict(np.array(rows)).tolist()

    return train, predict


def summarize(labels: list, predictions: dict, folds: list) -> dict:
    tested = [p for fold in folds for p in fold]
    scores = {name: experiment.fold_scores(labels, predicted, folds, cv.balanced_accuracy)
              for name, predicted in predictions.items()}
    confusion = cv.confusion_matrix([labels[p] for p in tested],
                                    [predictions["forest"][p] for p in tested])
    return {"scores": scores, "confusion": confusion}


def evaluate(rows: list, targets: list, groups: list, reports: list | None, config: dict) -> dict:
    settings = config["forest"]
    m = settings["attributes_per_tree"]
    if m > len(rows[0]):
        raise ValueError(f"attributes_per_tree is {m}, but there are only {len(rows[0])} features")
    parameters = (settings["trees"], m, settings.get("max_depth"), settings["seed"])
    models = {"forest": own_forest(*parameters), "scikit_learn": scikit_learn_forest(*parameters)}
    labels = [str(t) for t in targets]
    return experiment.evaluate(rows, labels, groups, reports, config["cross_validation"], models,
                               summarize)


def main(argv: list[str] | None = None) -> int:
    return experiment.run(argv, "forest", METRIC, evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
