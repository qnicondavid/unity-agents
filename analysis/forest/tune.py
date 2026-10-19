from analysis.core import cv, experiment
from analysis.forest.run import METRIC, evaluate, own_forest

KEYS = ["trees", "attributes_per_tree", "max_depth"]


def check_grid(config: dict) -> None:
    too_large = sorted({m for m in config["tuning"]["attributes_per_tree"] if m > len(config["features"])})
    if too_large:
        raise ValueError(f"attributes_per_tree {too_large} exceed the {len(config['features'])} features")


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    labels = [str(t) for t in targets]
    train, predict = own_forest(candidate["trees"], candidate["attributes_per_tree"],
                                candidate["max_depth"], config["forest"]["seed"])
    predictions = cv.cross_validation(rows, labels, folds, train, predict)
    return experiment.fold_scores(labels, predictions, folds, cv.balanced_accuracy)


def simplicity(candidate: dict) -> tuple:
    depth = candidate["max_depth"]
    return (float("inf") if depth is None else depth, candidate["attributes_per_tree"], -candidate["trees"])


def main(argv: list[str] | None = None) -> int:
    return experiment.run_tuning(argv, "forest", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 check=check_grid)


if __name__ == "__main__":
    raise SystemExit(main())
