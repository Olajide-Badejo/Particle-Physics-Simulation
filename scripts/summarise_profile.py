#!/usr/bin/env python3
"""Fold the per kernel Nsight Compute exports into one table.

Nsight writes one long CSV per kernel with a preamble, one row per metric and
the full kernel signature repeated on every row. The report wants one row per
variant with one column per metric, so that is what this produces.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VARIANTS = ("naive", "tiled", "fused")

# Metric name to the short column name used in the report.
COLUMNS = {
    "gpu__time_duration.sum": "duration_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "l1tex_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_load_sectors",
    "smsp__inst_executed_pipe_lsu.sum": "lsu_instructions",
    "smsp__inst_executed_pipe_fma.sum": "fma_instructions",
    "smsp__issue_active.avg.pct_of_peak_sustained_elapsed": "issue_active_pct",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "fadd_thread_inst",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_thread_inst",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "fmul_thread_inst",
}


def parse(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return {}
    values: dict[str, str] = {}
    for row in csv.DictReader(lines[start:]):
        name = row.get("Metric Name", "")
        if name in COLUMNS:
            raw = row.get("Metric Value", "").replace(",", "")
            values[COLUMNS[name]] = raw
    return values


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "experiments" / "results")
    parser.add_argument("--particles", type=int, required=True,
                        help="the particle count the exports were profiled at, which the report "
                             "needs to turn counters into per interaction figures")
    args = parser.parse_args()

    rows = []
    for variant in VARIANTS:
        values = parse(args.results / f"ncu_{variant}.csv")
        if not values:
            continue
        values["variant"] = variant
        values["num_particles"] = str(args.particles)
        rows.append(values)

    if not rows:
        print("no profiler exports found, nothing to summarise")
        return 0

    fields = ["variant"] + list(COLUMNS.values()) + ["num_particles"]
    out = args.results / "profile_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {out}")

    for row in rows:
        print(f"  {row['variant']:6s} l1tex {row.get('l1tex_throughput_pct', '?'):>6s}%  "
              f"dram {row.get('dram_throughput_pct', '?'):>5s}%  "
              f"issue {row.get('issue_active_pct', '?'):>6s}%  "
              f"load sectors {row.get('global_load_sectors', '?'):>14s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
