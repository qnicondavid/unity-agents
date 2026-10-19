from analysis.core import cv, experiment
from analysis.mlp.run import METRIC, evaluate, own_network, transform

KEYS = ["hidden_sizes", "learning_rate", "batch_size", "epochs"]


def score_candidate(rows: list, targets: list, folds: list, candidate: dict, config: dict) -> list[float]:
    rows, targets = transform(rows, targets, config)
    settings = {**config["mlp"], **candidate}
    runs = []
    for seed in settings["seeds"]:
        train, predict = own_network(settings, seed)
        predictions = cv.cross_validation(rows, targets, folds, train, predict)
        runs.append(experiment.fold_scores(targets, predictions, folds, cv.mean_absolute_error))
    return [sum(values) / len(values) for values in zip(*runs)]


def simplicity(candidate: dict) -> tuple:
    return (len(candidate["hidden_sizes"]), sum(candidate["hidden_sizes"]), candidate["epochs"],
            -candidate["batch_size"], candidate["learning_rate"])


def main(argv: list[str] | None = None) -> int:
    return experiment.run_tuning(argv, "mlp", METRIC, evaluate, KEYS, score_candidate, simplicity,
                                 better=min)


if __name__ == "__main__":
    raise SystemExit(main())
