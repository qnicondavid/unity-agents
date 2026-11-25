from analysis.boosting import boosting
from analysis.boosting.run import METRIC, evaluate, own_boosting, transform
from analysis.core import cv, experiment

KEYS = ["trees", "learning_rate", "max_depth", "min_leaf", "subsample"]


def check_grid(config: dict) -> None:
    tuning = config["tuning"]
    for candidate in experiment.grid(tuning, KEYS):
        boosting.check_settings(candidate["trees"], candidate["learning_rate"], candidate["max_depth"],
                                candidate["min_leaf"], candidate["subsample"])


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    rows, targets = transform(rows, targets, config)
    train, predict = own_boosting(candidate["trees"], candidate["learning_rate"], candidate["max_depth"],
                                  candidate["min_leaf"], candidate["subsample"],
                                  config["boosting"]["seed"])
    predictions = cv.cross_validation(rows, targets, folds, train, predict)
    return experiment.fold_scores(targets, predictions, folds, cv.mean_absolute_error)


# ties go to the shallower tree, then larger leaves, fewer trees, the lower learning rate
# and finally no subsampling
def simplicity(candidate: dict) -> tuple:
    return (candidate["max_depth"], -candidate["min_leaf"], candidate["trees"],
            candidate["learning_rate"], -candidate["subsample"])


def main(argv: list[str] | None = None) -> int:
    return experiment.run_tuning(argv, "boosting", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 better=min, check=check_grid)


if __name__ == "__main__":
    raise SystemExit(main())
