#!/usr/bin/env python3
"""Check that the committed results can be traced back to this repository.

Four checks, all over the files in `experiments/results`:

1. `validation_gates.csv` holds exactly the gates registered in
   `tests/validation/gates.txt`: every registered gate is present, and nothing
   else is. A missing gate means the report would quote a table with a hole in
   it; an extra one means a test records a gate nobody registered.
2. Every value in a column named `commit`, in any CSV under
   `experiments/results`, resolves to a commit object in this repository
   (`git cat-file -e <hash>^{commit}`). A hash that does not resolve cannot be
   checked out, so the number on that row cannot be reproduced from the tree
   that claims to have produced it.
3. No benchmark row came from a tree with uncommitted code or from a smoke test:
   any `dirty` value other than 0, any commit carrying `-dirty`, and any
   `protocol` value other than `full` fail. The files `bench_all` writes must
   have those columns, because a check that passes when the column is missing
   checks nothing.
4. Profiler files carry their stamp. `sph_pipeline_split.csv` and
   `profile_runs.csv` must have the columns saying what they measured, and every
   `ncu_*.csv` export must be named in `profile_runs.csv`, and the other way round,
   because the exports themselves are Nsight's format and hold no commit. A
   `profile_summary.csv` with stall columns needs both that file and a commit. One
   with neither is the older profile from `v1.1`, which recorded no commit.

`--gates-only` runs the first check alone, which needs no git history and so
works on a shallow clone. Each failure names the file, the row and the value.
Exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every file bench_all writes.
BENCHMARK_FILES = {
    "benchmarks.csv",
    "benchmark_trials.csv",
    "benchmark_repeatability.csv",
    "tile_sweep.csv",
    "tile_sweep_trials.csv",
}

# What each profiler file has to record about itself.
PROFILE_COLUMNS = {
    "profile_runs.csv": ("file", "commit", "dirty", "variant", "num_particles", "tile_size",
                         "block_size"),
    "sph_pipeline_split.csv": ("commit", "dirty", "num_particles", "block_size", "warmup_steps",
                               "measured_steps", "pass", "ms_per_step"),
}


def read_gate_registry(path: Path) -> list[str]:
    """Registered gate names in order. Blank lines and # comments are skipped."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def check_gates(results: Path, registry_path: Path) -> list[str]:
    failures: list[str] = []
    if not registry_path.exists():
        return [f"{relative(registry_path)}: the gate registry does not exist"]
    registry = read_gate_registry(registry_path)
    if not registry:
        return [f"{relative(registry_path)}: no gates are registered"]

    gates_csv = results / "validation_gates.csv"
    if not gates_csv.exists():
        return [f"{relative(gates_csv)}: file does not exist, so every registered gate is missing"]

    seen: dict[str, int] = {}
    with gates_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "gate" not in reader.fieldnames:
            return [f"{relative(gates_csv)}: no 'gate' column in the header"]
        for row_number, row in enumerate(reader, start=1):
            gate = (row.get("gate") or "").strip()
            where = f"{relative(gates_csv)} row {row_number} (line {reader.line_num})"
            if gate in seen:
                failures.append(f"{where}: gate '{gate}' already appears in row {seen[gate]}")
                continue
            seen[gate] = row_number
            if gate not in registry:
                failures.append(
                    f"{where}: gate '{gate}' is not registered in {relative(registry_path)}"
                )

    for gate in registry:
        if gate not in seen:
            failures.append(
                f"{relative(gates_csv)}: registered gate '{gate}' is missing "
                f"(listed in {relative(registry_path)})"
            )
    return failures


def commit_resolves(value: str, cache: dict[str, bool]) -> bool:
    if value not in cache:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{value}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        cache[value] = result.returncode == 0
    return cache[value]


def check_commits(results: Path) -> list[str]:
    failures: list[str] = []
    cache: dict[str, bool] = {}
    for path in sorted(results.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "commit" not in reader.fieldnames:
                continue
            if path.name in BENCHMARK_FILES:
                for column in ("dirty", "protocol"):
                    if column not in reader.fieldnames:
                        failures.append(f"{relative(path)}: no '{column}' column in the header")
            for row_number, row in enumerate(reader, start=1):
                value = (row.get("commit") or "").strip()
                where = f"{relative(path)} row {row_number} (line {reader.line_num})"
                dirty = row.get("dirty")
                if dirty is not None and dirty.strip() != "0":
                    failures.append(f"{where}: dirty is '{dirty.strip()}', so the row came from "
                                    "a tree with uncommitted code")
                protocol = row.get("protocol")
                if protocol is not None and protocol.strip() != "full":
                    failures.append(f"{where}: protocol is '{protocol.strip()}', and only a full "
                                    "sweep produces results")
                if not value:
                    failures.append(f"{where}: commit is empty")
                elif value.endswith("-dirty"):
                    failures.append(f"{where}: commit '{value}' is from a dirty tree")
                elif not commit_resolves(value, cache):
                    failures.append(
                        f"{where}: commit '{value}' does not resolve to a commit in this repository"
                    )
    return failures


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check_profile(results: Path) -> list[str]:
    failures: list[str] = []
    for name, columns in PROFILE_COLUMNS.items():
        path = results / name
        if path.exists():
            fields, rows = read_rows(path)
            failures += [f"{relative(path)}: no '{column}' column in the header"
                         for column in columns if column not in fields]
            if not rows:
                failures.append(f"{relative(path)}: the file has no rows")

    runs_path = results / "profile_runs.csv"
    if runs_path.exists():
        fields, rows = read_rows(runs_path)
        listed = {row.get("file", "") for row in rows}
        present = {path.name for path in results.glob("ncu_*.csv")}
        failures += [f"{relative(runs_path)}: names {name}, which does not exist"
                     for name in sorted(listed - present)]
        failures += [f"{relative(results / name)}: not named in profile_runs.csv, so nothing "
                     "says which commit made it"
                     for name in sorted(present - listed)]

    summary = results / "profile_summary.csv"
    if summary.exists():
        fields, _ = read_rows(summary)
        if any(field.startswith("stall_") for field in fields):
            if not runs_path.exists():
                failures.append(f"{relative(summary)}: has stall columns but "
                                "profile_runs.csv does not exist")
            failures += [f"{relative(summary)}: no '{column}' column in the header"
                         for column in ("commit", "dirty") if column not in fields]
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gates-only", action="store_true",
                        help="check only that validation_gates.csv matches the gate registry")
    parser.add_argument("--results", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "tests" / "validation" / "gates.txt")
    args = parser.parse_args()

    failures = check_gates(args.results, args.registry)
    if not args.gates_only:
        try:
            failures += check_commits(args.results)
            failures += check_profile(args.results)
        except FileNotFoundError:
            failures.append("git is not on PATH, so no commit hash could be checked")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"provenance check failed: {len(failures)} problem(s)")
        return 1
    print("provenance check passed" + (" (gates only)" if args.gates_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
