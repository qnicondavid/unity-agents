"""Builds the data files the GUI reads, in gui/data/.

From the repository root, with .venv active:

    python -m gui.export_data                 # everything; retrains the models (about 15 minutes)
    python -m gui.export_data --skip-models   # datasets, curves and results only

Each file is a script that sets a key on window.UA_DATA, so index.html can load it from disk
with a plain <script> tag (a browser refuses to fetch a .json or .csv from a file:// page).
"""

import argparse
import csv
import json
import math
import os
import statistics
import tomllib
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "gui" / "data"
DATA = ROOT / "data"
RESULTS = ROOT / "analysis" / "results"
CONFIGS = ROOT / "analysis" / "configs"

FEATURES = ["batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers", "beta",
            "epsilon", "num_epoch"]
LOGGED = {"batch_size", "buffer_size", "learning_rate", "beta"}
MACHINES = {"Model 197": "m01", "Model 165": "m02", "arm": "m03", "Model 186": "m05"}
METRICS = {
    "environment_episode_length": "episode_length",
    "policy_entropy": "entropy",
    "losses_policy_loss": "policy_loss",
    "losses_value_loss": "value_loss",
    "policy_learning_rate": "learning_rate",
    "policy_extrinsic_value_estimate": "value_estimate",
}
REPORT_FOLDS, REPORT_SEED = 10, 11


def write_js(name: str, key: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    (OUT / f"{name}.js").write_text(
        f"window.UA_DATA=window.UA_DATA||{{}};window.UA_DATA.{key}={text};\n", encoding="utf-8")
    print(f"wrote gui/data/{name}.js ({len(text) / 1e6:.2f} MB)")


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def table(frame: pd.DataFrame) -> dict:
    return {"columns": list(frame.columns),
            "rows": [[clean(v) for v in row] for row in frame.itertuples(index=False)]}


def sig(value: float, digits: int = 4):
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0
    return float(f"{value:.{digits}g}")


def machine_of(cpu: str) -> str:
    for key, machine in MACHINES.items():
        if key in str(cpu):
            return machine
    return "?"


def batch_of(run_id: str) -> str:
    for part in run_id.split("_"):
        if part.startswith("b") and part[1:].isdigit():
            return part[1:]
    return "first"


# Datasets ---------------------------------------------------------------------------------------

def export_runs() -> None:
    g10 = pd.read_csv(DATA / "group10.csv")
    g10["machine"] = g10["cpu_model"].map(machine_of)
    g10["batch"] = g10["RunID"].map(batch_of)
    keep = ["RunID", "ball", *FEATURES, "seed", "batch", "machine", "last_logged_step",
            "reached_95", "steps_to_95"]
    g15 = pd.read_csv(DATA / "group15.csv")
    g15_keep = ["RunID", "batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers",
                "beta", "epsilon", "lambd", "num_epoch", "time_horizon", "learning_rate_schedule",
                "normalize", "max_steps", "os", "cpu", "last_logged_step", "reached_95",
                "steps_to_95"]
    log = pd.read_csv(DATA / "group10_2026.csv")
    log_keep = ["run_id", "environment_name", "seed", "max_steps", "reward_threshold",
                "steps_to_threshold", "reached_threshold", "final_mean_reward"]
    same = pd.read_csv(DATA / "same_config_diff_machines.csv")
    same["machine"] = same["cpu_model"].map(machine_of)
    machines = pd.read_csv(ROOT / "docs" / "hardware" / "machines.csv")
    machines = machines.drop(columns=["owner"])
    survey = ROOT / "docs" / "shared-data" / "survey-output"
    write_js("runs", "runs", {
        "group10": table(g10[keep]),
        "group15": table(g15[g15_keep]),
        "log": table(log[log_keep]),
        "same_config": table(same[["run_id", "seed", "machine", "reached_threshold",
                                   "final_mean_reward", "steps_to_threshold", *FEATURES]]),
        "machines": table(machines),
        "survey_files": table(pd.read_csv(survey / "files.csv")),
        "survey_matrix": table(pd.read_csv(survey / "matrix.csv")),
    })


def export_curves() -> None:
    rewards, steps_by_run, durations, metrics = {}, {}, {}, {}
    for path in sorted((DATA / "curves").glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        run = path.stem
        steps_by_run[run] = [int(float(r["step"])) for r in rows]
        rewards[run] = [round(float(r["environment_cumulative_reward"]), 1) for r in rows]
        durations[run] = round(float(rows[-1]["wall_time"]) - float(rows[0]["wall_time"]))
        metrics[run] = {short: [sig(float(r[name])) if r.get(name) not in (None, "") else None
                                for r in rows]
                        for name, short in METRICS.items()}
    common = statistics.mode(tuple(s) for s in steps_by_run.values())
    other_steps = {run: s for run, s in steps_by_run.items() if tuple(s) != common}
    write_js("curves", "curves", {"steps": list(common), "other_steps": other_steps,
                                  "reward": rewards, "duration": durations})
    write_js("metrics", "metrics", {"names": list(METRICS.values()), "runs": metrics})


# Results ----------------------------------------------------------------------------------------

def export_results() -> None:
    files = {}
    for path in sorted(RESULTS.rglob("*.csv")):
        key = path.relative_to(RESULTS).as_posix()
        files[key] = table(pd.read_csv(path))
    configs = {}
    for path in sorted(CONFIGS.rglob("*.toml")):
        with open(path, "rb") as f:
            configs[path.relative_to(CONFIGS).as_posix()] = tomllib.load(f)
    write_js("results", "results", {"files": files, "configs": configs})


# Models -----------------------------------------------------------------------------------------

def read_config(model: str) -> dict:
    with open(CONFIGS / model / "group10_steps_to_95.toml", "rb") as f:
        return tomllib.load(f)


def tuned(model: str, ball: str) -> dict:
    frame = pd.read_csv(RESULTS / model / "group10_steps_to_95" / "tuned_best_mean" / "summary.csv")
    return frame[frame["ball"] == ball].iloc[0].to_dict()


def mlp_settings(ball: str) -> dict:
    # The edges grid holds the final MLP settings for our runs (analysis/README.md).
    frame = pd.read_csv(RESULTS / "mlp" / "group10_steps_to_95_edges" / "tuned_best_mean" / "summary.csv")
    row = frame[frame["ball"] == ball].iloc[0]
    return {"hidden_sizes": json.loads(row["hidden_sizes"]), "learning_rate": float(row["learning_rate"]),
            "batch_size": int(row["batch_size"]), "epochs": int(row["epochs"]),
            "dropout": float(row["dropout"])}


def forest_settings(ball: str) -> dict:
    row = tuned("forest", ball)
    depth = row["max_depth"]
    return {"trees": int(row["trees"]), "m": int(row["attributes_per_tree"]),
            "max_depth": None if str(depth) == "none" else int(float(depth)),
            "min_leaf": int(row["min_leaf"]), "sampling": row["sampling"]}


def gpr_settings(ball: str) -> dict:
    row = tuned("gpr", ball)
    return {"kind": row["kernel"], "ard": bool(row["ard"])}


def log_rows(rows: list) -> list:
    return [[math.log(v) if FEATURES[i] in LOGGED else v for i, v in enumerate(row)] for row in rows]


def fit_predict(job: dict) -> dict:
    kind, settings = job["model"], job["settings"]
    train_rows, train_y, test_rows = job["train_rows"], job["train_y"], job["test_rows"]
    if kind == "gpr":
        from analysis.gpr import gpr
        model = gpr.train_gpr(log_rows(train_rows), train_y, settings["kind"], settings["ard"], 200)
        mean, std = gpr.predict(model, log_rows(test_rows), return_std=True)
        return {**job["key"], "mean": mean, "std": std}
    if kind == "mlp":
        from analysis.mlp import network
        model = network.train_network(log_rows(train_rows), train_y, settings["hidden_sizes"],
                                      settings["learning_rate"], settings["batch_size"],
                                      settings["epochs"], job["seed"], dropout=settings["dropout"])
        return {**job["key"], "mean": network.predict(model, log_rows(test_rows))}
    if kind == "forest":
        from analysis.forest import regression
        model = regression.build_forest(train_rows, train_y, settings["trees"], settings["m"],
                                        settings["max_depth"], settings["min_leaf"],
                                        settings["sampling"], 1)
        return {**job["key"], "mean": [regression.predict_forest(model, row) for row in test_rows]}
    raise ValueError(kind)


def ridge_predict(train_rows, train_y, test_rows) -> list:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), Ridge()).fit(np.array(log_rows(train_rows)), np.array(train_y))
    return model.predict(np.array(log_rows(test_rows))).tolist()


def export_models(workers: int) -> None:
    from analysis.core import cv

    data = pd.read_csv(DATA / "group10.csv")
    data = data[data["steps_to_95"].notna()].reset_index(drop=True)
    values = {name: sorted(pd.read_csv(DATA / "group10.csv")[name].unique().tolist()) for name in FEATURES}
    grid = [list(combo) for combo in product(*(values[name] for name in FEATURES))]
    seeds = read_config("mlp")["mlp"]["seeds"]

    jobs, parts = [], {}
    for ball in ("fixed", "random"):
        part = data[data["ball"] == ball].reset_index(drop=True)
        rows = part[FEATURES].astype(float).values.tolist()
        y = [math.log(v) for v in part["steps_to_95"]]
        groups = [tuple(g) for g in part[FEATURES].astype(str).values.tolist()]
        folds = cv.make_group_folds(groups, REPORT_FOLDS, REPORT_SEED)
        parts[ball] = {"ids": part["RunID"].tolist(), "rows": rows, "y": y, "folds": folds}
        settings = {"gpr": gpr_settings(ball), "mlp": mlp_settings(ball), "forest": forest_settings(ball)}
        targets = [("grid", None, list(range(len(rows))), grid)]
        for f, fold in enumerate(folds):
            test = set(fold)
            train = [i for i in range(len(rows)) if i not in test]
            targets.append(("oof", f, train, [rows[i] for i in fold]))
        for where, f, train, test_rows in targets:
            for model in ("gpr", "mlp", "forest"):
                for seed in (seeds if model == "mlp" else [None]):
                    jobs.append({"key": {"ball": ball, "where": where, "fold": f, "model": model, "seed": seed},
                                 "model": model, "settings": settings[model], "seed": seed,
                                 "train_rows": [rows[i] for i in train], "train_y": [y[i] for i in train],
                                 "test_rows": test_rows})
        parts[ball]["settings"] = settings

    print(f"training {len(jobs)} models with {workers} workers")
    done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(fit_predict, jobs, chunksize=1)):
            done.append(result)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1} of {len(jobs)} done", flush=True)

    out = {"features": FEATURES, "values": values, "logged": sorted(LOGGED), "balls": {},
           "report_folds": REPORT_FOLDS, "report_seed": REPORT_SEED}
    for ball, part in parts.items():
        n = len(part["rows"])
        oof = {m: [None] * n for m in ("gpr", "gpr_std", "mlp", "forest", "ridge", "median")}
        fold_of = [None] * n
        for f, fold in enumerate(part["folds"]):
            for i in fold:
                fold_of[i] = f
            test = set(fold)
            train = [i for i in range(n) if i not in test]
            ridge = ridge_predict([part["rows"][i] for i in train], [part["y"][i] for i in train],
                                  [part["rows"][i] for i in fold])
            median = statistics.median(part["y"][i] for i in train)
            for j, i in enumerate(fold):
                oof["ridge"][i] = ridge[j]
                oof["median"][i] = median
        grid_pred = {}
        mlp_oof, mlp_grid = {}, []
        for r in done:
            if r["ball"] != ball:
                continue
            if r["where"] == "grid":
                if r["model"] == "mlp":
                    mlp_grid.append(r["mean"])
                else:
                    grid_pred[r["model"]] = r["mean"]
                    if r["model"] == "gpr":
                        grid_pred["gpr_std"] = r["std"]
                continue
            fold = part["folds"][r["fold"]]
            if r["model"] == "mlp":
                mlp_oof.setdefault(r["fold"], []).append(r["mean"])
                continue
            for j, i in enumerate(fold):
                oof[r["model"]][i] = r["mean"][j]
                if r["model"] == "gpr":
                    oof["gpr_std"][i] = r["std"][j]
        for f, runs in mlp_oof.items():
            for j, i in enumerate(part["folds"][f]):
                oof["mlp"][i] = sum(run[j] for run in runs) / len(runs)
        grid_pred["mlp"] = [sum(v) / len(v) for v in zip(*mlp_grid)]
        grid_pred["ridge"] = ridge_predict(part["rows"], part["y"], grid)
        scores = {}
        for m in ("gpr", "mlp", "forest", "ridge", "median"):
            per_fold = [statistics.mean(abs(part["y"][i] - oof[m][i]) for i in fold) for fold in part["folds"]]
            scores[m] = {"mae": statistics.mean(per_fold),
                         "se": statistics.stdev(per_fold) / math.sqrt(len(per_fold)), "folds": per_fold}
        print(ball, {m: round(s["mae"], 4) for m, s in scores.items()})
        out["balls"][ball] = {
            "ids": part["ids"], "fold": fold_of, "settings": part["settings"], "scores": scores,
            "oof": {m: [round(v, 4) for v in vals] for m, vals in oof.items()},
            "grid": {m: [round(v, 4) for v in vals] for m, vals in grid_pred.items()},
        }
    write_js("models", "models", out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true", help="keep gui/data/models.js as it is")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    export_runs()
    export_curves()
    export_results()
    if not args.skip_models:
        export_models(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
