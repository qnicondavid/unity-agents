import numpy as np
import pandas as pd

from analysis.core import experiment

DATA = experiment.REPO_ROOT / "data" / "group10.csv"
OUT = experiment.RESULTS / "temperature"
CELL = ["ball", "learning_rate", "batch_size"]


def load(path):
    data = pd.read_csv(path)
    data = data[data["reached_95"].notna()].copy()
    data["reached"] = data["reached_95"].astype(str).str.lower().eq("true")
    data["pair"] = data["RunID"].str.replace(r"_s[23]$", "", regex=True)
    return data


def seed_gaps(data):
    reached = data[data["reached"]].copy()
    reached["ln_steps"] = np.log(reached["steps_to_95"])
    pairs = reached.pivot_table(index=CELL + ["pair"], columns="seed", values="ln_steps")
    return (pairs[2] - pairs[3]).abs().dropna().rename("seed_gap").reset_index()


def summarize(data, gaps, keys):
    runs = data.groupby(keys).agg(runs=("reached", "size"),
                                  never_reached=("reached", lambda r: 1 - r.mean()))
    steps = data[data["reached"]].groupby(keys)["steps_to_95"].median().rename("median_steps")
    gap = gaps.groupby(keys).agg(pairs=("seed_gap", "size"), median_seed_gap=("seed_gap", "median"))
    table = runs.join(steps).join(gap).reset_index()
    table[["never_reached", "median_seed_gap"]] = table[["never_reached", "median_seed_gap"]].round(6)
    return table


def grid(part, column, scale, digits):
    view = part.pivot(index="batch_size", columns="learning_rate", values=column) * scale
    return view.to_string(float_format=lambda v: f"{v:.{digits}f}")


def main():
    data = load(DATA)
    gaps = seed_gaps(data)
    cells = summarize(data, gaps, CELL)
    cells["lr_over_b"] = cells["learning_rate"] / cells["batch_size"]
    cells["lr_over_sqrt_b"] = cells["learning_rate"] / np.sqrt(cells["batch_size"])
    by_lr = summarize(data, gaps, ["ball", "learning_rate"])
    best = cells.loc[cells.groupby(["ball", "batch_size"])["median_steps"].idxmin(),
                     ["ball", "batch_size", "learning_rate", "median_steps"]]
    formats = {"never_reached": lambda v: f"{100 * v:.1f}%",
               "median_steps": lambda v: f"{v / 1000:.0f}k",
               "median_seed_gap": lambda v: f"{v:.3f}"}
    for ball in sorted(cells["ball"].unique()):
        part = cells[cells["ball"] == ball]
        print(f"ball = {ball}")
        print("median steps to 95, thousands (rows: batch_size, columns: learning_rate)")
        print(grid(part, "median_steps", 0.001, 0))
        print("share that never reached 95, %")
        print(grid(part, "never_reached", 100, 1))
        print("median seed gap |ln s2 - ln s3|")
        print(grid(part, "median_seed_gap", 1, 3))
        print("all batch sizes together")
        print(by_lr[by_lr["ball"] == ball].drop(columns="ball").to_string(index=False, formatters=formats))
        print("best learning_rate for each batch_size")
        for row in best[best["ball"] == ball].itertuples(index=False):
            print(f"  batch_size {row.batch_size}: learning_rate {row.learning_rate} "
                  f"(median {row.median_steps / 1000:.0f}k steps)")
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "cells.csv", index=False, lineterminator="\n")
    by_lr.to_csv(OUT / "by_learning_rate.csv", index=False, lineterminator="\n")
    best.to_csv(OUT / "best_learning_rate.csv", index=False, lineterminator="\n")
    print(f"results written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
