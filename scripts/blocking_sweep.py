#!/usr/bin/env python3
"""Measure register blocking against the kernels it would replace.

Runs every configuration in interleaved rounds, each trial a fresh process, and
records solver ms/step per trial with the commit that produced it. The rounds
rotate, so session drift falls on every configuration alike.

Blocking at a fixed block size also shrinks the number of blocks, which starves
the machine of work whatever the kernel does. So the sweep holds the thread count
constant as well (blocking factor times block size), and reports both.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# Particle count and steps per trial. The step counts keep a trial near a second
# at the larger sizes, so a round of the whole sweep stays affordable.
SIZES = {10000: 400, 100000: 100, 200000: 40}
# Blocking factor and block size. The first four hold the thread count constant.
SETTINGS = ((1, 256), (2, 128), (4, 64), (8, 32), (2, 256))
VARIANTS = ("tiled", "fused")
FIELDS = ("kernels", "variant", "num_particles", "blocking_factor", "block_size", "tile_size",
          "round", "steps", "ms_per_step", "commit", "dirty", "protocol")


def solver_ms(binary: str, arguments: list[str]) -> float:
    done = subprocess.run([binary, "run", *arguments, "--diag-every", "0", "--snapshot-every", "0",
                           "--diagnostics", ""],
                          capture_output=True, text=True, errors="replace", check=False)
    text = done.stdout + done.stderr
    found = re.search(r"solver ms/step:\s*([0-9.]+)", text)
    if done.returncode != 0 or found is None:
        raise SystemExit(f"blocking_sweep: {' '.join(arguments)} failed: {text[-400:]}")
    return float(found.group(1))


def version_stamp(binary: str) -> tuple[str, int]:
    text = subprocess.run([binary, "--version"], capture_output=True, text=True,
                          check=True).stdout
    found = re.search(r"\(([0-9a-f]+)(-dirty)?\)", text)
    if found is None:
        raise SystemExit(f"blocking_sweep: {binary} has no commit stamp: {text.strip()}")
    return found.group(1), 1 if found.group(2) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="the build with blocking")
    parser.add_argument("--preset", type=Path, required=True, help="its Plummer preset")
    parser.add_argument("--baseline", required=True, help="the build before blocking")
    parser.add_argument("--baseline-preset", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True,
                        help="the commit of that build, which reports none of its own")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/blocking_sweep.csv"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    commit, dirty = version_stamp(args.binary)
    if dirty and not args.allow_dirty:
        raise SystemExit("blocking_sweep: this build has uncommitted code; commit and rebuild")

    base = json.loads(args.preset.read_text(encoding="utf-8"))
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for count, steps in SIZES.items():
            cases = []
            for variant in VARIANTS:
                cases.append(("before", variant, 1, 256, args.baseline,
                              [str(args.baseline_preset), "--variant", variant,
                               "--particles", str(count), "--steps", str(steps)],
                              args.baseline_commit, 0))
                for factor, block in SETTINGS:
                    config = json.loads(json.dumps(base))
                    config["nbody"]["variant"] = variant
                    config["nbody"]["blocking_factor"] = factor
                    config["run"]["block_size"] = block
                    config["run"]["steps"] = steps
                    config["initial_condition"]["num_particles"] = count
                    path = Path(scratch) / f"{variant}_{factor}_{block}_{count}.json"
                    path.write_text(json.dumps(config), encoding="utf-8")
                    cases.append(("blocking", variant, factor, block, args.binary, [str(path)],
                                  commit, dirty))
            for case in cases:  # one discarded run each, so no trial pays for a cold start
                solver_ms(case[4], case[5])
            for trial in range(args.rounds):
                shift = trial % len(cases)
                for kernels, variant, factor, block, binary, arguments, stamp, stamp_dirty in (
                        cases[shift:] + cases[:shift]):
                    rows.append({
                        "kernels": kernels, "variant": variant, "num_particles": count,
                        "blocking_factor": factor, "block_size": block,
                        "tile_size": base["nbody"]["tile_size"], "round": trial + 1,
                        "steps": steps, "ms_per_step": solver_ms(binary, arguments),
                        "commit": stamp, "dirty": stamp_dirty, "protocol": "full",
                    })
            print(f"measured {count} particles", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for count in SIZES:
        print(f"N = {count}")
        for variant in VARIANTS:
            selected = [r for r in rows if r["num_particles"] == count and r["variant"] == variant]
            reference = {r["round"]: r["ms_per_step"] for r in selected if r["kernels"] == "before"}
            keys = sorted({(r["kernels"], r["blocking_factor"], r["block_size"]) for r in selected})
            for kernels, factor, block in keys:
                trials = [r for r in selected
                          if (r["kernels"], r["blocking_factor"], r["block_size"]) == (kernels, factor, block)]
                ratio = statistics.median(r["ms_per_step"] / reference[r["round"]] for r in trials)
                median = statistics.median(r["ms_per_step"] for r in trials)
                print(f"  {variant} {kernels:8s} factor {factor} block {block:3d}: "
                      f"{median:9.4f} ms, {ratio:.4f} of before")
    return 0


if __name__ == "__main__":
    sys.exit(main())
