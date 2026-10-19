import argparse
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import forest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis" / "results" / "forest"


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
        labels = data.loc[part, config["class_attribute"]].astype(str).tolist()
        groups = [tuple(g) for g in data.loc[part, config["group_by"]].astype(str).values.tolist()]
        reports = data.loc[part, report_by].tolist() if report_by else None
        ans.append((value, rows, labels, groups, reports))
    return ans


def fold_scores(labels: list, predicted: list, folds: list) -> list[float]:
    return [forest.balanced_accuracy([labels[p] for p in fold], [predicted[p] for p in fold])
            for fold in folds]


def breakdown(labels: list, own_predictions: list, reference_predictions: list, folds: list,
              reports: list) -> dict:
    ans = {}
    for value in sorted(set(reports)):
        part = [[p for p in fold if reports[p] == value] for fold in folds]
        if not all(part):
            raise ValueError(f"a test fold has no runs with the report value {value}")
        inside = [p for fold in part for p in fold]
        ans[value] = {
            "fold_sizes": [len(fold) for fold in part],
            "forest": fold_scores(labels, own_predictions, part),
            "scikit_learn": fold_scores(labels, reference_predictions, part),
            "confusion": forest.confusion_matrix([labels[p] for p in inside],
                                                 [own_predictions[p] for p in inside]),
        }
    return ans


def evaluate(rows: list, labels: list, groups: list, config: dict,
             reports: list | None = None) -> dict:
    settings = config["forest"]
    cv = config["cross_validation"]
    m = settings["attributes_per_tree"]
    max_depth = settings.get("max_depth")
    if m > len(rows[0]):
        raise ValueError(f"attributes_per_tree is {m}, but there are only {len(rows[0])} features")
    if cv["folds"] > len(set(groups)):
        raise ValueError(f"{cv['folds']} folds, but only {len(set(groups))} configurations")

    folds = forest.make_group_folds(groups, cv["folds"], cv["seed"])

    def train(training_rows, training_labels):
        return forest.build_forest(training_rows, training_labels, settings["trees"], m,
                                   max_depth, settings["seed"])

    _, own, predictions = forest.cross_validation(rows, labels, folds, train, forest.predict_forest)

    X, y = np.array(rows), np.array(labels)
    reference, reference_predictions = [], [None] * len(rows)
    for f in range(len(folds)):
        training = [pos for j in range(len(folds)) if j != f for pos in folds[j]]
        model = RandomForestClassifier(n_estimators=settings["trees"], max_features=m,
                                       criterion="entropy", max_depth=max_depth,
                                       random_state=settings["seed"])
        model.fit(X[training], y[training])
        predicted = model.predict(X[folds[f]]).tolist()
        reference.append(forest.balanced_accuracy(y[folds[f]].tolist(), predicted))
        for pos, label in zip(folds[f], predicted):
            reference_predictions[pos] = label

    result = {
        "fold_sizes": [len(fold) for fold in folds],
        "forest": own,
        "scikit_learn": reference,
        "confusion": forest.confusion_matrix(labels, predictions),
    }
    if reports is not None:
        result["by"] = breakdown(labels, predictions, reference_predictions, folds, reports)
    return result


def mean_and_se(scores: list) -> tuple[float, float]:
    return float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(scores)))


def write_tables(out: Path, suffix: str, entries: list) -> None:
    fold_rows, confusion_rows = [], []
    for keys, result in entries:
        for f in range(len(result["fold_sizes"])):
            fold_rows.append({
                **keys,
                "fold": f + 1,
                "test_runs": result["fold_sizes"][f],
                "forest_balanced_accuracy": round(result["forest"][f], 6),
                "scikit_learn_balanced_accuracy": round(result["scikit_learn"][f], 6),
            })
        for true_label, row in result["confusion"].items():
            for predicted_label, count in row.items():
                confusion_rows.append({**keys, "true": true_label,
                                       "predicted": predicted_label, "count": count})
    pd.DataFrame(fold_rows).to_csv(out / f"folds{suffix}.csv", index=False, lineterminator="\n")
    pd.DataFrame(confusion_rows).to_csv(out / f"confusion{suffix}.csv", index=False, lineterminator="\n")


def write_results(name: str, split_by: str, results: list, folder: Path,
                  report_by: str | None = None) -> Path:
    out = folder / name
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out, "", [({split_by: value}, result) for value, result in results])
    if report_by:
        parts = [({split_by: value, report_by: key}, part)
                 for value, result in results for key, part in result["by"].items()]
        write_tables(out, f"_by_{report_by}", parts)
    return out


def print_breakdown(report_by: str | None, result: dict) -> None:
    for value, part in result.get("by", {}).items():
        counts = ", ".join(f"{label} {sum(row.values())}" for label, row in part["confusion"].items())
        own, own_se = mean_and_se(part["forest"])
        ref, ref_se = mean_and_se(part["scikit_learn"])
        print(f"    {report_by} = {value}: {sum(part['fold_sizes'])} runs ({counts}) | "
              f"forest {own:.3f} (se {own_se:.3f}), scikit-learn {ref:.3f} (se {ref_se:.3f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = read_config(args.config)
    split_by = split_column(config)
    report_by = config.get("report_by")
    results = []
    for value, rows, labels, groups, reports in load_groups(config):
        result = evaluate(rows, labels, groups, config, reports)
        results.append((value, result))
        counts = ", ".join(f"{label} {n}" for label, n in sorted(Counter(labels).items()))
        own, own_se = mean_and_se(result["forest"])
        ref, ref_se = mean_and_se(result["scikit_learn"])
        print(f"{split_by} = {value}: {len(rows)} runs ({counts}) | balanced accuracy: "
              f"forest {own:.3f} (se {own_se:.3f}), scikit-learn {ref:.3f} (se {ref_se:.3f})")
        print_breakdown(report_by, result)

    out = write_results(args.config.stem, split_by, results, RESULTS, report_by)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
