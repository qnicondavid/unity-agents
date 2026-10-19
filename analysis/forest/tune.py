import argparse
import os
from pathlib import Path

from analysis.core import cv, experiment
from analysis.forest.run import METRIC, RESULTS, evaluate, own_forest

KEYS = ["trees", "attributes_per_tree", "max_depth"]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 61))
    args = parser.parse_args(argv)

    config = experiment.read_config(args.config)
    too_large = sorted({m for m in config["tuning"]["attributes_per_tree"] if m > len(config["features"])})
    if too_large:
        raise ValueError(f"attributes_per_tree {too_large} exceed the {len(config['features'])} features")

    split_by = experiment.split_column(config)
    report_by = config.get("report_by")
    outcomes = []
    tuned = experiment.tune(config, args.workers, KEYS, score_candidate)
    for value, rows, targets, groups, reports, scored in tuned:
        chosen = experiment.choose(scored, simplicity)
        settings = experiment.with_settings(config, "forest", chosen, KEYS)
        result = evaluate(rows, targets, groups, reports, settings)
        outcomes.append((value, scored, chosen, result))
        depth = "none" if chosen["max_depth"] is None else chosen["max_depth"]
        print(f"{split_by} = {value}: chose {chosen['trees']} trees, m = {chosen['attributes_per_tree']}, "
              f"depth {depth} (best of {len(scored)}) | tuning {chosen['mean']:.3f} (se {chosen['se']:.3f}) | "
              f"report: {experiment.scores_line(result, METRIC)}")
        experiment.print_breakdown(report_by, result, METRIC)

    folder = experiment.write_results("tuned_best_mean", split_by, [(o[0], o[3]) for o in outcomes],
                                      RESULTS / args.config.stem, METRIC, report_by)
    experiment.write_tuning(split_by, outcomes, KEYS, METRIC, folder)
    print(f"results written to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
