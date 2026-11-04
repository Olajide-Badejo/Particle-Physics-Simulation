#!/usr/bin/env python3
"""Turn an Nsight Systems trace of the SPH step into a per pass time split.

The input is the `cuda_gpu_trace` report, with one row per kernel and per memory
operation. A step runs from the start of one assign_cells kernel to the start of the
next. Each pass is the time its kernels took:

    assign_cells         assign_cells_kernel
    sort                 every kernel in cub::, and the memset CUB launches between
                         its passes. A memset belongs to the sort only if it falls
                         between the assign_cells and the cell_bounds of its step.
    cell_bounds          cell_bounds_kernel
    gather               gather_sorted_kernel
    density              density_kernel
    force_and_integrate  forces_and_integrate_kernel, one kernel doing both
    between_kernels      whatever of the step no kernel accounts for: launching,
                         the front end, and the tracer's own cost

Anything else inside a measured step is an error, as is a pass with no launches or
a pass that takes no time, because a table with a hole in it reads as a measurement.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

KERNEL_PASSES = {
    "assign_cells_kernel": "assign_cells",
    "cell_bounds_kernel": "cell_bounds",
    "gather_sorted_kernel": "gather",
    "density_kernel": "density",
    "forces_and_integrate_kernel": "force_and_integrate",
}
PASSES = ("assign_cells", "sort", "cell_bounds", "gather", "density", "force_and_integrate")
BETWEEN = "between_kernels"
FIELDS = ("pass", "launches_per_step", "ms_per_step", "ms_stddev", "percent_of_step",
          "num_particles", "block_size", "warmup_steps", "measured_steps", "traced_step_ms",
          "untraced_step_ms", "source", "commit", "dirty")


class SplitError(Exception):
    pass


def classify(name: str) -> str | None:
    """The pass a kernel belongs to, by name, or None for anything else."""
    bare = name.split("(")[0]
    if "cub::" in bare:
        return "sort"
    for kernel, pass_name in KERNEL_PASSES.items():
        if bare.endswith(kernel) or f"{kernel}<" in bare:
            return pass_name
    return None


def read_trace(path: Path) -> list[tuple[int, int, str]]:
    """(start, duration, name) in time order, in nanoseconds."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and not {"Start (ns)", "Duration (ns)", "Name"} <= rows[0].keys():
        raise SplitError(f"{path.name} is not a cuda_gpu_trace report")
    events = [(int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"]) for row in rows]
    events.sort()
    return events


def split_steps(events: list[tuple[int, int, str]], warmup: int,
                measured: int) -> list[dict[str, tuple[int, int]]]:
    """One dict per measured step: pass to (nanoseconds, launches), plus the step's length."""
    starts = [start for start, _, name in events if classify(name) == "assign_cells"]
    if len(starts) < warmup + measured + 1:
        raise SplitError(f"the trace holds {len(starts)} assign_cells kernels, and {warmup} of "
                         f"warmup plus {measured} measured steps need {warmup + measured + 1}")

    steps = []
    position = 0
    for index in range(warmup, warmup + measured):
        begin, end = starts[index], starts[index + 1]
        step = {name: (0, 0) for name in PASSES}
        seen_bounds = False
        while position < len(events) and events[position][0] < begin:
            position += 1
        while position < len(events) and events[position][0] < end:
            start, duration, name = events[position]
            position += 1
            pass_name = classify(name)
            if pass_name == "cell_bounds":
                seen_bounds = True
            if pass_name is None and name.startswith("[CUDA memset") and not seen_bounds:
                pass_name = "sort"
            if pass_name is None:
                raise SplitError(f"step {index} holds something no pass accounts for: {name[:80]}")
            total, launches = step[pass_name]
            step[pass_name] = (total + duration, launches + 1)
        step["__length__"] = (end - begin, 1)
        steps.append(step)
    return steps


def summarise(steps: list[dict[str, tuple[int, int]]]) -> list[dict[str, float]]:
    def row(pass_name: str, values: list[int], launches: float) -> dict[str, float]:
        return {"pass": pass_name, "launches_per_step": launches,
                "ms_per_step": statistics.fmean(values) / 1.0e6,
                "ms_stddev": statistics.pstdev(values) / 1.0e6}

    rows = []
    for pass_name in PASSES:
        values = [step[pass_name][0] for step in steps]
        launches = [step[pass_name][1] for step in steps]
        if min(launches) == 0 or min(values) <= 0:
            raise SplitError(f"the {pass_name} pass is missing or took no time in at least one "
                             "measured step")
        rows.append(row(pass_name, values, statistics.fmean(launches)))
    between = [step["__length__"][0] - sum(step[name][0] for name in PASSES) for step in steps]
    if statistics.fmean(between) < 0:
        raise SplitError("the kernels add up to more than the step, so they overlapped")
    rows.append(row(BETWEEN, between, 0.0))

    total = sum(entry["ms_per_step"] for entry in rows)
    for entry in rows:
        entry["percent_of_step"] = 100.0 * entry["ms_per_step"] / total
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path, help="the cuda_gpu_trace report, as CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dirty", required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    parser.add_argument("--untraced-ms", type=float, required=True,
                        help="the step cost of the same run with no tracer")
    args = parser.parse_args()

    try:
        rows = summarise(split_steps(read_trace(args.trace), args.warmup, args.measured))
    except SplitError as error:
        raise SystemExit(f"error: {error}") from error

    traced = sum(entry["ms_per_step"] for entry in rows)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for entry in rows:
            writer.writerow({
                "pass": entry["pass"],
                "launches_per_step": f"{entry['launches_per_step']:.2f}",
                "ms_per_step": f"{entry['ms_per_step']:.6f}",
                "ms_stddev": f"{entry['ms_stddev']:.6f}",
                "percent_of_step": f"{entry['percent_of_step']:.2f}",
                "num_particles": args.particles,
                "block_size": args.block_size,
                "warmup_steps": args.warmup,
                "measured_steps": args.measured,
                "traced_step_ms": f"{traced:.6f}",
                "untraced_step_ms": f"{args.untraced_ms:.6f}",
                "source": "nsys",
                "commit": args.commit,
                "dirty": args.dirty,
            })
            print(f"  {entry['pass']:20s} {entry['ms_per_step']:9.4f} ms "
                  f"{entry['percent_of_step']:6.2f} %")
    print(f"  step {traced:.4f} ms traced, {args.untraced_ms:.4f} ms with no tracer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
