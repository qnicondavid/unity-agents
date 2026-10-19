import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import cv
import experiment
import forest

RESULTS = experiment.RESULTS / "forest"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = experiment.read_config(args.config)
    split_by = experiment.split_column(config)
    report_by = config.get("report_by")
    results = []
    for value, rows, targets, groups, reports in experiment.load_groups(config):
        result = evaluate(rows, targets, groups, reports, config)
        results.append((value, result))
        counts = ", ".join(f"{label} {n}" for label, n in sorted(Counter(map(str, targets)).items()))
        scores = experiment.scores_line(result, METRIC)
        print(f"{split_by} = {value}: {len(rows)} runs ({counts}) | {scores}")
        experiment.print_breakdown(report_by, result, METRIC)

    out = experiment.write_results(args.config.stem, split_by, results, RESULTS, METRIC, report_by)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
