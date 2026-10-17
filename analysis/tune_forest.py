import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

import forest
from run_forest import RESULTS, evaluate, load_groups, mean_and_se, read_config, write_results


def candidates(config: dict) -> list[dict]:
    tuning = config["tuning"]
    ans = []
    for trees, m, depth in product(tuning["trees"], tuning["attributes_per_tree"], tuning["max_depth"]):
        ans.append({"trees": trees, "attributes_per_tree": m,
                    "max_depth": None if depth == "none" else int(depth)})
    return ans


def score_candidate(rows: list, labels: list, folds: list, candidate: dict, seed: int) -> list[float]:
    def train(training_rows, training_labels):
        return forest.build_forest(training_rows, training_labels, candidate["trees"],
                                   candidate["attributes_per_tree"], candidate["max_depth"], seed)

    _, scores, _ = forest.cross_validation(rows, labels, folds, train, forest.predict_forest)
    return scores


def simplicity(candidate: dict) -> tuple:
    depth = candidate["max_depth"]
    return (float("inf") if depth is None else depth, candidate["attributes_per_tree"], -candidate["trees"])


def choose(scored: list[dict]) -> dict:
    top = max(c["mean"] for c in scored)
    return min((c for c in scored if c["mean"] == top), key=simplicity)


def with_settings(config: dict, candidate: dict) -> dict:
    settings = {"trees": candidate["trees"], "attributes_per_tree": candidate["attributes_per_tree"],
                "seed": config["forest"]["seed"]}
    if candidate["max_depth"] is not None:
        settings["max_depth"] = candidate["max_depth"]
    report = {"folds": config["tuning"]["report_folds"], "seed": config["tuning"]["report_seed"]}
    return {**config, "forest": settings, "cross_validation": report}


def tune(config: dict, workers: int) -> list[tuple]:
    tuning = config["tuning"]
    seed = config["forest"]["seed"]
    options = candidates(config)
    too_large = sorted({c["attributes_per_tree"] for c in options
                        if c["attributes_per_tree"] > len(config["features"])})
    if too_large:
        raise ValueError(f"attributes_per_tree {too_large} exceed the {len(config['features'])} features")

    splits = load_groups(config)
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for value, rows, labels, groups in splits:
            if tuning["folds"] > len(set(groups)):
                raise ValueError(f"{tuning['folds']} tuning folds, but only {len(set(groups))} configurations")
            folds = forest.make_group_folds(groups, tuning["folds"], tuning["seed"])
            for candidate in options:
                jobs.append((value, candidate, pool.submit(score_candidate, rows, labels, folds, candidate, seed)))
        scored = {}
        for done, (value, candidate, job) in enumerate(jobs, start=1):
            mean, se = mean_and_se(job.result())
            scored.setdefault(value, []).append({**candidate, "mean": mean, "se": se})
            if done % 50 == 0 or done == len(jobs):
                print(f"{done} of {len(jobs)} candidate runs done", flush=True)
    return [(value, rows, labels, groups, scored[value]) for value, rows, labels, groups in splits]


def write_tuning(config: dict, outcomes: list, folder: Path) -> None:
    split_by = config["split_by"]
    candidate_rows, summary_rows = [], []
    for value, scored, chosen, result in outcomes:
        ranks = pd.Series([c["mean"] for c in scored]).rank(ascending=False, method="min").astype(int)
        for c, rank in zip(scored, ranks):
            candidate_rows.append({
                split_by: value,
                "trees": c["trees"],
                "attributes_per_tree": c["attributes_per_tree"],
                "max_depth": "none" if c["max_depth"] is None else c["max_depth"],
                "tuning_balanced_accuracy": round(c["mean"], 6),
                "tuning_se": round(c["se"], 6),
                "rank": rank,
                "chosen": c is chosen,
            })
        own, own_se = mean_and_se(result["forest"])
        ref, ref_se = mean_and_se(result["scikit_learn"])
        summary_rows.append({
            split_by: value,
            "trees": chosen["trees"],
            "attributes_per_tree": chosen["attributes_per_tree"],
            "max_depth": "none" if chosen["max_depth"] is None else chosen["max_depth"],
            "tuning_balanced_accuracy": round(chosen["mean"], 6),
            "tuning_se": round(chosen["se"], 6),
            "forest_balanced_accuracy": round(own, 6),
            "forest_se": round(own_se, 6),
            "scikit_learn_balanced_accuracy": round(ref, 6),
            "scikit_learn_se": round(ref_se, 6),
        })
    pd.DataFrame(candidate_rows).to_csv(folder / "candidates.csv", index=False, lineterminator="\n")
    pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 61))
    args = parser.parse_args(argv)

    config = read_config(args.config)
    outcomes = []
    for value, rows, labels, groups, scored in tune(config, args.workers):
        chosen = choose(scored)
        result = evaluate(rows, labels, groups, with_settings(config, chosen))
        outcomes.append((value, scored, chosen, result))
        own, own_se = mean_and_se(result["forest"])
        ref, ref_se = mean_and_se(result["scikit_learn"])
        depth = "none" if chosen["max_depth"] is None else chosen["max_depth"]
        print(f"{config['split_by']} = {value}: chose {chosen['trees']} trees, m = {chosen['attributes_per_tree']}, "
              f"depth {depth} (best of {len(scored)}) | tuning {chosen['mean']:.3f} (se {chosen['se']:.3f}) | "
              f"report: forest {own:.3f} (se {own_se:.3f}), scikit-learn {ref:.3f} (se {ref_se:.3f})")

    folder = write_results("tuned_best_mean", config["split_by"], [(o[0], o[3]) for o in outcomes],
                           RESULTS / args.config.stem)
    write_tuning(config, outcomes, folder)
    print(f"results written to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
