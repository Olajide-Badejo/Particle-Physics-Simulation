#!/usr/bin/env python3
"""Check that the committed results can be traced back to this repository.

Two checks, both over the files in `experiments/results`:

1. `validation_gates.csv` holds exactly the gates registered in
   `tests/validation/gates.txt`: every registered gate is present, and nothing
   else is. A missing gate means the report would quote a table with a hole in
   it; an extra one means a test records a gate nobody registered.
2. Every value in a column named `commit`, in any CSV under
   `experiments/results`, resolves to a commit object in this repository
   (`git cat-file -e <hash>^{commit}`). A hash that does not resolve cannot be
   checked out, so the number on that row cannot be reproduced from the tree
   that claims to have produced it.

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
            for row_number, row in enumerate(reader, start=1):
                value = (row.get("commit") or "").strip()
                where = f"{relative(path)} row {row_number} (line {reader.line_num})"
                if not value:
                    failures.append(f"{where}: commit is empty")
                elif not commit_resolves(value, cache):
                    failures.append(
                        f"{where}: commit '{value}' does not resolve to a commit in this repository"
                    )
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
