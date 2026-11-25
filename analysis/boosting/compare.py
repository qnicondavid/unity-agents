# Compares the tuned boosting with our other models for steps_to_95, fold by fold.
#
# All the tuned configs for our runs report on the same folds (report_folds 10, report_seed 11,
# grouped by the 8 settings, split by ball), so their fold scores pair up. This script reads the
# report run of each tuning (tuned_best_mean/folds.csv) and checks that the test folds have the
# same sizes before it pairs them.

import argparse

import numpy as np
import pandas as pd

from analysis.core import experiment

# (label, model folder, config name, score column); the boosting's config name comes from --boosting
MODELS = [
    ("boosting", "boosting", None, "boosting_mean_absolute_error"),
    ("mlp", "mlp", "group10_steps_to_95_edges", "mlp_mean_absolute_error"),
    ("forest", "forest", "group10_steps_to_95", "forest_mean_absolute_error"),
    ("gpr", "gpr", "group10_steps_to_95", "gpr_mean_absolute_error"),
]


def read_folds(boosting_config: str) -> pd.DataFrame:
    ans = None
    for label, model, config, column in MODELS:
        config = config or boosting_config
        path = experiment.RESULTS / model / config / "tuned_best_mean" / "folds.csv"
        table = pd.read_csv(path)[["ball", "fold", "test_runs", column]].rename(columns={column: label})
        if ans is None:
            ans = table
            continue
        ans = ans.merge(table, on=["ball", "fold"], suffixes=("", f"_{label}"))
        if not (ans["test_runs"] == ans[f"test_runs_{label}"]).all():
            raise ValueError(f"{path} was scored on other folds")
        ans = ans.drop(columns=f"test_runs_{label}")
    return ans


def compare(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ball, part in folds.groupby("ball"):
        for label, _, _, _ in MODELS:
            mean, se = experiment.mean_and_se(part[label].tolist())
            row = {"ball": ball, "model": label, "mean_absolute_error": round(mean, 6), "se": round(se, 6)}
            if label != "boosting":
                # positive: boosting's error is lower
                gap = (part[label] - part["boosting"]).tolist()
                gap_mean, gap_se = experiment.mean_and_se(gap)
                row["minus_boosting"] = round(gap_mean, 6)
                row["minus_boosting_se"] = round(gap_se, 6)
                row["folds_boosting_lower"] = int(np.sum(np.array(gap) > 0))
                row["folds"] = len(gap)
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boosting", default="group10_steps_to_95_edges",
                        help="the boosting config whose tuning to compare")
    args = parser.parse_args(argv)
    table = compare(read_folds(args.boosting))
    out = experiment.RESULTS / "boosting" / args.boosting / "comparison.csv"
    table.to_csv(out, index=False, lineterminator="\n")
    print(table.to_string(index=False))
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
