from analysis.core import cv, experiment
from analysis.knn.run import METRIC, check_settings, distance_of, evaluate, own_knn, transform

KEYS = ["k", "p"]


def check_grid(config: dict) -> None:
    distance_of(config)
    for k in config["tuning"]["k"]:
        for p in config["tuning"]["p"]:
            check_settings(k, p)


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    rows, targets = transform(rows, targets, config)
    train, predict = own_knn(candidate["k"], candidate["p"], *distance_of(config))
    predictions = cv.cross_validation(rows, targets, folds, train, predict)
    return experiment.fold_scores(targets, predictions, folds, cv.mean_absolute_error)


def simplicity(candidate: dict) -> tuple:
    return (-candidate["k"], candidate["p"] != 2)


def main(argv: list[str] | None = None) -> int:
    return experiment.run_tuning(argv, "knn", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 better=min, check=check_grid)


if __name__ == "__main__":
    raise SystemExit(main())
