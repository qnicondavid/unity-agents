import tomllib
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import cv

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis" / "results"


def read_config(path: Path) -> dict:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config["data"] = REPO_ROOT / config["data"]
    return config


def split_column(config: dict) -> str:
    return config.get("split_by", "split")


def encode(column: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        return column.astype(float)
    text = column.astype(str)
    codes = {value: float(i) for i, value in enumerate(sorted(text.unique()))}
    return text.map(codes)


def load_groups(config: dict) -> list[tuple]:
    data = pd.read_csv(config["data"])
    data = data[data[config["class_attribute"]].notna()].reset_index(drop=True)
    if data[config["features"]].isna().any().any():
        raise ValueError("some features have missing values")

    features = pd.DataFrame({name: encode(data[name]) for name in config["features"]})
    split_by = config.get("split_by")
    report_by = config.get("report_by")
    values = sorted(data[split_by].unique().tolist()) if split_by else ["all"]
    ans = []
    for value in values:
        part = data[split_by] == value if split_by else pd.Series(True, index=data.index)
        rows = features[part].values.tolist()
        targets = data.loc[part, config["class_attribute"]].tolist()
        groups = [tuple(g) for g in data.loc[part, config["group_by"]].astype(str).values.tolist()]
        reports = data.loc[part, report_by].tolist() if report_by else None
        ans.append((value, rows, targets, groups, reports))
    return ans


def group_folds(groups: list, k: int, seed: int) -> list:
    if k > len(set(groups)):
        raise ValueError(f"{k} folds, but only {len(set(groups))} configurations")
    return cv.make_group_folds(groups, k, seed)


def fold_scores(targets: list, predictions: list, folds: list, score) -> list[float]:
    return [score([targets[p] for p in fold], [predictions[p] for p in fold]) for fold in folds]


def report_folds(folds: list, reports: list) -> dict:
    ans = {}
    for value in sorted(set(reports)):
        part = [[p for p in fold if reports[p] == value] for fold in folds]
        if not all(part):
            raise ValueError(f"a test fold has no runs with the report value {value}")
        ans[value] = part
    return ans


def evaluate(rows: list, targets: list, groups: list, reports: list | None, cv_config: dict,
             models: dict, summarize) -> dict:
    folds = group_folds(groups, cv_config["folds"], cv_config["seed"])
    predictions = {name: cv.cross_validation(rows, targets, folds, train, predict)
                   for name, (train, predict) in models.items()}
    result = {"fold_sizes": [len(fold) for fold in folds], **summarize(targets, predictions, folds)}
    if reports is not None:
        result["by"] = {}
        for value, part in report_folds(folds, reports).items():
            result["by"][value] = {"fold_sizes": [len(fold) for fold in part],
                                   **summarize(targets, predictions, part)}
    return result


def mean_and_se(scores: list) -> tuple[float, float]:
    return float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(scores)))


def scores_line(result: dict, metric: str) -> str:
    parts = []
    for name, scores in result["scores"].items():
        mean, se = mean_and_se(scores)
        parts.append(f"{name} {mean:.3f} (se {se:.3f})")
    return f"{metric}: " + ", ".join(parts)


def print_breakdown(report_by: str | None, result: dict, metric: str) -> None:
    for value, part in result.get("by", {}).items():
        print(f"    {report_by} = {value}: {sum(part['fold_sizes'])} runs | {scores_line(part, metric)}")


def write_tables(out: Path, suffix: str, entries: list, metric: str) -> None:
    fold_rows, confusion_rows = [], []
    for keys, result in entries:
        for f in range(len(result["fold_sizes"])):
            row = {**keys, "fold": f + 1, "test_runs": result["fold_sizes"][f]}
            for name, scores in result["scores"].items():
                row[f"{name}_{metric}"] = round(scores[f], 6)
            fold_rows.append(row)
        for true_label, counts in result.get("confusion", {}).items():
            for predicted_label, count in counts.items():
                confusion_rows.append({**keys, "true": true_label,
                                       "predicted": predicted_label, "count": count})
    tables = {f"folds{suffix}.csv": fold_rows, f"confusion{suffix}.csv": confusion_rows}
    for file_name, table in tables.items():
        if table:
            pd.DataFrame(table).to_csv(out / file_name, index=False, lineterminator="\n")


def write_results(name: str, split_by: str, results: list, folder: Path, metric: str,
                  report_by: str | None = None) -> Path:
    out = folder / name
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out, "", [({split_by: value}, result) for value, result in results], metric)
    if report_by:
        parts = [({split_by: value, report_by: key}, part)
                 for value, result in results for key, part in result["by"].items()]
        write_tables(out, f"_by_{report_by}", parts, metric)
    return out


def grid(tuning: dict, keys: list[str]) -> list[dict]:
    ans = []
    for values in product(*(tuning[key] for key in keys)):
        ans.append({key: None if value == "none" else value for key, value in zip(keys, values)})
    return ans


def choose(scored: list[dict], simplicity, better=max) -> dict:
    top = better(c["mean"] for c in scored)
    return min((c for c in scored if c["mean"] == top), key=simplicity)


def with_settings(config: dict, section: str, candidate: dict, keys: list[str]) -> dict:
    settings = {key: value for key, value in config[section].items() if key not in keys}
    settings.update({key: candidate[key] for key in keys if candidate[key] is not None})
    report = {"folds": config["tuning"]["report_folds"], "seed": config["tuning"]["report_seed"]}
    return {**config, section: settings, "cross_validation": report}


def tune(config: dict, workers: int, keys: list[str], score_candidate) -> list[tuple]:
    tuning = config["tuning"]
    options = grid(tuning, keys)
    parts = load_groups(config)
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for value, rows, targets, groups, _ in parts:
            folds = group_folds(groups, tuning["folds"], tuning["seed"])
            for candidate in options:
                job = pool.submit(score_candidate, rows, targets, folds, candidate, config)
                jobs.append((value, candidate, job))
        scored = {}
        for done, (value, candidate, job) in enumerate(jobs, start=1):
            mean, se = mean_and_se(job.result())
            scored.setdefault(value, []).append({**candidate, "mean": mean, "se": se})
            if done % 50 == 0 or done == len(jobs):
                print(f"{done} of {len(jobs)} candidate runs done", flush=True)
    return [(value, rows, targets, groups, reports, scored[value])
            for value, rows, targets, groups, reports in parts]


def settings_row(candidate: dict, keys: list[str]) -> dict:
    return {key: "none" if candidate[key] is None else candidate[key] for key in keys}


def write_tuning(split_by: str, outcomes: list, keys: list[str], metric: str, folder: Path,
                 better=max) -> None:
    candidate_rows, summary_rows = [], []
    for value, scored, chosen, result in outcomes:
        means = pd.Series([c["mean"] for c in scored])
        ranks = means.rank(ascending=better is min, method="min").astype(int)
        for c, rank in zip(scored, ranks):
            candidate_rows.append({
                split_by: value,
                **settings_row(c, keys),
                f"tuning_{metric}": round(c["mean"], 6),
                "tuning_se": round(c["se"], 6),
                "rank": rank,
                "chosen": c is chosen,
            })
        row = {split_by: value, **settings_row(chosen, keys),
               f"tuning_{metric}": round(chosen["mean"], 6), "tuning_se": round(chosen["se"], 6)}
        for name, scores in result["scores"].items():
            mean, se = mean_and_se(scores)
            row[f"{name}_{metric}"] = round(mean, 6)
            row[f"{name}_se"] = round(se, 6)
        summary_rows.append(row)
    pd.DataFrame(candidate_rows).to_csv(folder / "candidates.csv", index=False, lineterminator="\n")
    pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False, lineterminator="\n")
