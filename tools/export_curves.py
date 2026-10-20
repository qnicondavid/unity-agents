import csv
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CSV_FILE = REPO_ROOT / "data" / "group10_2026.csv"
CURVES_DIR = REPO_ROOT / "data" / "curves"


def column_name(tag: str) -> str:
    return tag.lower().replace("/", "_").replace(" ", "_")


def read_scalars(event_dir: Path) -> dict[str, list]:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}


def curve_rows(scalars: dict[str, list]) -> tuple[list[str], list[list]]:
    tags = sorted(scalars)
    by_step = {}
    for tag in tags:
        for event in scalars[tag]:
            row = by_step.setdefault(event.step, {"wall_time": event.wall_time})
            row["wall_time"] = min(row["wall_time"], event.wall_time)
            row[tag] = event.value
    header = ["step", "wall_time", *(column_name(tag) for tag in tags)]
    rows = []
    for step in sorted(by_step):
        row = by_step[step]
        values = [str(np.float32(row[tag])) if tag in row else "" for tag in tags]
        rows.append([step, f"{row['wall_time']:.3f}", *values])
    return header, rows


def export_run(run_id: str, behavior_name: str) -> Path | None:
    event_dir = RESULTS_DIR / run_id / behavior_name
    if not event_dir.exists():
        return None
    scalars = read_scalars(event_dir)
    if not scalars:
        return None
    header, rows = curve_rows(scalars)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    out = CURVES_DIR / f"{run_id}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return out


def main() -> int:
    with open(CSV_FILE, newline="") as f:
        runs = [(row["run_id"], row["environment_name"]) for row in csv.DictReader(f)]
    written, existing, missing = [], [], []
    for run_id, behavior_name in runs:
        if (CURVES_DIR / f"{run_id}.csv").exists():
            existing.append(run_id)
        elif export_run(run_id, behavior_name) is None:
            missing.append(run_id)
        else:
            written.append(run_id)
    print(f"{len(written)} curves written to {CURVES_DIR}")
    print(f"{len(existing)} already exported")
    print(f"{len(missing)} without results: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
