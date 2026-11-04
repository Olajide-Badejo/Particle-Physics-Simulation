#!/usr/bin/env python3
"""Fold the per kernel Nsight Compute exports into one table.

Nsight writes one long CSV per kernel with a preamble, one row per metric and
the full kernel signature repeated on every row. The report wants one row per
variant and particle count with one column per metric, so that is what this
produces.

`scripts/profile_kernels.sh` writes `profile_runs.csv` beside the exports, which
says what each one is and which commit made it, and this copies that onto the
rows. Any metric missing from an export named there is an error, because a table
with a hole in it reads as a measurement. A directory with no such file holds the
three exports of an older profile at one size, given by `--particles`, and gets
the table that profile always had.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VARIANTS = ("naive", "tiled", "fused")

# Metric name to the short column name used in the report. The percentages here
# are of peak sustained over elapsed cycles.
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

# Warps stalled for each reason, per cycle with an issue active. Percentages in
# this group are over active cycles and carry that in the column name, so they are
# never read against the elapsed ones above.
STALL_COLUMNS = {
    "smsp__inst_executed.sum": "instructions",
    "smsp__inst_executed_pipe_xu.sum": "xu_instructions",
    "smsp__inst_executed_pipe_alu.sum": "alu_instructions",
    "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": "xu_pct_active",
    "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "fma_pct_active",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "lsu_pct_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_pct_active",
    "smsp__warps_active.avg.pct_of_peak_sustained_active": "warps_pct_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "stall_wait",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": "stall_not_selected",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio":
        "stall_short_scoreboard",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":
        "stall_long_scoreboard",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio":
        "stall_math_pipe_throttle",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio_throttle",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": "stall_lg_throttle",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio":
        "stall_no_instruction",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shared_load_bank_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "shared_store_bank_conflicts",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct": "l1_global_load_hit_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__occupancy_limit_registers": "occupancy_limit_registers",
    "launch__waves_per_multiprocessor": "waves_per_multiprocessor",
}

RUN_COLUMNS = ("tile_size", "block_size", "commit", "dirty")


def parse(path: Path, columns: dict[str, str]) -> dict[str, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return {}
    values: dict[str, str] = {}
    for row in csv.DictReader(lines[start:]):
        name = row.get("Metric Name", "")
        if name in columns:
            raw = row.get("Metric Value", "").replace(",", "")
            values[columns[name]] = raw
    return values


def rows_from_manifest(results: Path, manifest: Path) -> list[dict[str, str]]:
    columns = {**COLUMNS, **STALL_COLUMNS}
    rows = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    for run in runs:
        values = parse(results / run["file"], columns)
        missing = [name for name, column in columns.items() if not values.get(column)]
        if missing:
            raise SystemExit(f"error: {run['file']} has no value for: {', '.join(missing)}. "
                             "Nsight Compute did not return every metric asked for, so this "
                             "profile is not usable.")
        values["variant"] = run["variant"]
        values["num_particles"] = run["num_particles"]
        for column in RUN_COLUMNS:
            values[column] = run[column]
        rows.append(values)
    return rows


def rows_from_old_profile(results: Path, particles: int) -> list[dict[str, str]]:
    rows = []
    for variant in VARIANTS:
        values = parse(results / f"ncu_{variant}.csv", COLUMNS)
        if not values:
            continue
        values["variant"] = variant
        values["num_particles"] = str(particles)
        rows.append(values)
    return rows


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "experiments" / "results")
    parser.add_argument("--particles", type=int,
                        help="the particle count of an older profile with no profile_runs.csv, "
                             "which the report needs to turn counters into per interaction figures")
    args = parser.parse_args()

    manifest = args.results / "profile_runs.csv"
    fields = ["variant"] + list(COLUMNS.values()) + ["num_particles"]
    if manifest.exists():
        rows = rows_from_manifest(args.results, manifest)
        fields += [*RUN_COLUMNS[:2], *STALL_COLUMNS.values(), *RUN_COLUMNS[2:]]
    elif args.particles is None:
        parser.error("--particles is required when there is no profile_runs.csv")
    else:
        rows = rows_from_old_profile(args.results, args.particles)

    if not rows:
        print("no profiler exports found, nothing to summarise")
        return 0

    out = args.results / "profile_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {out.name}")

    for row in rows:
        print(f"  {row['variant']:6s} N={row['num_particles']:>7s} "
              f"l1tex {row.get('l1tex_throughput_pct', '?'):>6s}%  "
              f"dram {row.get('dram_throughput_pct', '?'):>5s}%  "
              f"issue {row.get('issue_active_pct', '?'):>6s}%  "
              f"load sectors {row.get('global_load_sectors', '?'):>14s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
