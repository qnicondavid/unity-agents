from analysis.core import cv, experiment
from analysis.forest.regression_run import METRIC, SAMPLINGS, evaluate, own_forest, transform

KEYS = ["trees", "attributes_per_tree", "max_depth", "min_leaf", "sampling"]


def check_grid(config: dict) -> None:
    tuning = config["tuning"]
    features = len(config["features"])
    too_large = sorted({m for m in tuning["attributes_per_tree"] if m > features})
    if too_large:
        raise ValueError(f"attributes_per_tree {too_large} exceed the {features} features")
    too_small = sorted({leaf for leaf in tuning["min_leaf"] if leaf < 1})
    if too_small:
        raise ValueError(f"min_leaf {too_small} must be at least 1")
    unknown = sorted({s for s in tuning["sampling"] if s not in SAMPLINGS})
    if unknown:
        raise ValueError(f"sampling {unknown} must be \"split\" or \"tree\"")


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    rows, targets = transform(rows, targets, config)
    train, predict = own_forest(candidate["trees"], candidate["attributes_per_tree"],
                                candidate["max_depth"], candidate["min_leaf"], candidate["sampling"],
                                config["forest"]["seed"])
    predictions = cv.cross_validation(rows, targets, folds, train, predict)
    return experiment.fold_scores(targets, predictions, folds, cv.mean_absolute_error)


def simplicity(candidate: dict) -> tuple:
    depth = candidate["max_depth"]
    return (float("inf") if depth is None else depth, -candidate["min_leaf"],
            candidate["attributes_per_tree"], -candidate["trees"], candidate["sampling"] != "split")


def main(argv: list[str] | None = None) -> int:
    return experiment.run_tuning(argv, "forest", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 better=min, check=check_grid)


if __name__ == "__main__":
    raise SystemExit(main())
