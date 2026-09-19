#!/usr/bin/env python3
"""
Survey the BCS2720 shared data-collection repository.

Helps in answering the following questions:
    - which groups submitted data, and in what shape
    - which column names mean the same thing under different spellings
    - which files are per-run and which are per-logging-step
    (these two cannot be concatenated without aggregating first)

Usage:
    python survey_shared_data.py <path to "AY 2025-2026"> [-o outdir]

Writes to outdir (default ./survey_out):
    files.csv    one row per CSV found, with shape and granularity
    matrix.csv   column name x group presence matrix
    report.md    human-readable summary

Only reads.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RUN_KEYS = ("run_id", "runnr", "run", "config_name")

STEP_KEYS = ("step", "nrsteps", "steps", "wall time")

SYNONYMS: dict[str, tuple[str, ...]] = {
    "ram_total":        ("ram_total_gb","ram_mb","ram_gb","total_ram","total_ram_mb","ram_total_bytes"),
    "cpu_cores_logical":("cpu_cores_logical","cpu_logical_cores","cpu_threads"),
    "cpu_clock":        ("cpu_clock_ghz","cpu_frequency","cpu_frequency_khz"),
    "gpu_model":        ("gpu_model","gpu_name","gpu_0_name","gpu_1_name"),
    "gpu_vram":         ("gpu_0_vram_gb","gpu_1_vram_gb","gpu_memory_gb"),
    "os_name":          ("os_name","host_os_name","operating_system","os"),
    "ram_used_avg":     ("average_ram","avg_ram_usage","memory_usage_avg_mb"),
    "ram_used_peak":    ("peak_ram_mb","peak_ram_usage","memory_usage_peak_mb"),
    "cpu_used_avg":     ("average_cpu","avg_cpu_usage","cpu_usage_avg_percent"),
    "cpu_used_peak":    ("peak_cpu_usage","cpu_usage_peak_percent"),
    "run_duration":     ("total_time","training_duration_seconds","duration_sec","train_duration_s",
                         "time_elapsed","time_elapased_seconds","training_time",
                         "wallclock_seconds_total","total_duration"),
    "time_to_threshold":("time_to_convergence","time_to_threshold"),
    "steps_to_threshold":("steps_to_threshold","steps_to_convergence"),
    "algorithm":        ("algo","algo_name","algorithm","drl_algorithm","trainer_type"),
    "environment":      ("env_name","environment","enviroment"),
    "batch_size":       ("batch_size","bath_size"),
    "lambda":           ("lambd","lambda"),
    "num_epoch":        ("num_epoch","epoch","epochs"),
    "num_layers":       ("num_layers","nn_arch_depth"),
    "gamma":            ("gamma","extrinsic_gamma"),
    "strength":         ("strength","extrinsic_strength"),
    "num_agents":       ("num_agents","num_parallel_agents"),
    "run_id":           ("run_id","runNr"),
    "step":             ("step","Step","nrSteps"),
    "policy_loss":      ("losses_policy_loss","mean_policy_loss","p_loss_mean","final_mean_policy_loss"),
    "value_loss":       ("losses_value_loss","mean_value_loss","v_loss_mean","final_mean_value_loss"),
    "entropy":          ("policy_entropy","entropy_mean","mean_entropy","final_mean_entropy"),
    "reward_mean":      ("mean_reward","meanReward","reward_mean","cumulative_reward_mean","episodic_reward_mean"),
    "reward_final":     ("final_mean_reward","final_reward_mean","final_cumulative_mean_reward","final_perf"),
    "reward_std":       ("std_reward","final_std_reward"),
}


def sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    counts = {d: head.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def group_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "?"


def classify(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    key = next((lower[k] for k in RUN_KEYS if k in lower), None)
    if key is not None:
        distinct = df[key].nunique(dropna=True)
        if distinct == 0:
            return "unknown"
        return "per-run" if len(df) <= distinct * 1.5 else "per-step"
    if any(k in lower for k in STEP_KEYS):
        return "per-step"
    return "unknown"


def survey(root: Path) -> tuple[pd.DataFrame, dict[str, set[str]], list[str]]:
    rows: list[dict] = []
    matrix: dict[str, set[str]] = defaultdict(set)
    warnings: list[str] = []

    for path in sorted(root.rglob("*.csv")):
        grp = group_of(path, root)
        delim = sniff_delimiter(path)
        try:
            df = pd.read_csv(path, sep=delim, low_memory=False)
        except Exception as exc:
            warnings.append(f"{path.relative_to(root)}: unreadable ({exc.__class__.__name__})")
            continue

        for col in df.columns:
            matrix[str(col).strip()].add(grp)

        rows.append({
            "group": grp,
            "file": str(path.relative_to(root)).replace("\\", "/"),
            "delimiter": {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delim],
            "rows": len(df),
            "columns": len(df.columns),
            "granularity": classify(df),
            "size_kb": round(path.stat().st_size / 1024, 1),
        })

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in (".zip", ".7z", ".rar", ".gz"):
            warnings.append(
                f"{path.relative_to(root)}: archive, needs unpacking before it counts as data"
            )

    return pd.DataFrame(rows), matrix, warnings


def write_outputs(files: pd.DataFrame, matrix: dict[str, set[str]],
                  warnings: list[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    groups = sorted({g for gs in matrix.values() for g in gs},
                    key=lambda s: (len(s), s))

    files.to_csv(outdir / "files.csv", index=False)

    with (outdir / "matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["column", "n_groups", *groups])
        for col in sorted(matrix, key=lambda c: (-len(matrix[c]), c)):
            present = matrix[col]
            w.writerow([col, len(present), *("x" if g in present else "" for g in groups)])

    shared = {c: g for c, g in matrix.items() if len(g) > 1}
    orphans = {c: g for c, g in matrix.items() if len(g) == 1}

    lines = [
        "# Shared data survey",
        "",
        f"- groups with at least one readable CSV: **{files['group'].nunique()}**",
        f"- CSV files: **{len(files)}**",
        f"- total rows: **{int(files['rows'].sum()):,}**",
        f"- distinct column names: **{len(matrix)}**",
        f"- names used by more than one group: **{len(shared)}**",
        f"- names used by exactly one group: **{len(orphans)}**",
        "",
        "## Granularity split",
        "",
    ]
    for kind, n in files["granularity"].value_counts().items():
        sub = files[files["granularity"] == kind]
        lines.append(f"- **{kind}**: {n} files, {int(sub['rows'].sum()):,} rows")

    lines += ["", "## Same quantity, different spelling", ""]
    for canon, aliases in SYNONYMS.items():
        seen = {a: sorted(matrix[a]) for a in aliases if a in matrix}
        if len(seen) > 1:
            lines.append(f"**{canon}**")
            for alias, gs in sorted(seen.items()):
                lines.append(f"  - `{alias}`: {', '.join(gs)}")
            lines.append("")

    lines += ["", "## Non-comma delimiters", ""]
    odd = files[files["delimiter"] != "comma"]
    lines += [f"- `{r.file}`: {r.delimiter}" for r in odd.itertuples()] or ["- none"]

    if warnings:
        lines += ["", "## Needs attention", ""]
        lines += [f"- {w}" for w in warnings]

    lines += [
        "",
        "## Columns used by exactly one group",
        "",
        "Candidates for further synonyms, or genuinely unique measurements.",
        "",
    ]
    for col in sorted(orphans):
        lines.append(f"- `{col}`: {next(iter(orphans[col]))}")

    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help='path to an "AY ...." folder')
    ap.add_argument("-o", "--outdir", type=Path, default=Path("survey_out"))
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"not a directory: {args.root}", file=sys.stderr)
        return 1

    files, matrix, warnings = survey(args.root)
    if files.empty:
        print("no readable CSVs found", file=sys.stderr)
        return 1

    write_outputs(files, matrix, warnings, args.outdir)
    print(f"{len(files)} files, {len(matrix)} distinct columns -> {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
