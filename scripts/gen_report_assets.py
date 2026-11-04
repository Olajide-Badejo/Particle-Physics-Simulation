#!/usr/bin/env python3
"""Turn the recorded results into the tables and macros the reports include.

This exists so that no number in either PDF is ever typed by hand. Every table
and every figure quoted in the prose comes out of this script, which reads only
the CSV files that `bench_all`, `gpu_probe` and the validation suite wrote. Run
it twice on the same inputs and it produces byte identical output.

The prose in the reports quotes measured values through LaTeX macros defined in
`numbers.tex`, so a sentence such as "tiling is worth a factor of X" cannot
drift away from the table beside it: both come from the same row.

The same values go to `docs/generated_numbers.md` as a table, for the Markdown
documents to quote. `scripts/check_doc_numbers.py` holds them to it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# The stall columns of profile_summary.csv and the word each takes in a macro name,
# and the sizes a macro can name. A macro name is letters only.
STALL_COLUMNS = {
    "stall_wait": "Wait",
    "stall_not_selected": "NotSelected",
    "stall_short_scoreboard": "ShortScoreboard",
    "stall_long_scoreboard": "LongScoreboard",
    "stall_math_pipe_throttle": "MathPipeThrottle",
    "stall_mio_throttle": "MioThrottle",
    "stall_lg_throttle": "LgThrottle",
    "stall_barrier": "Barrier",
    "stall_no_instruction": "NoInstruction",
}
PROFILE_SIZE_NAMES = {200000: "TwoHundredK", 100000: "HundredK", 10000: "TenK"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# What the v1.1 benchmark protocol adds to a summary row. A file without them
# was measured in separate blocks and cannot support a quoted difference, so the
# report is not built from it.
DISPERSION_COLUMNS = ("trials", "ms_min", "ms_median", "ms_mean", "ms_stddev", "ms_iqr", "dirty")
RERUN_HINT = (
    "Rerun bench_all under the current protocol. This script has no fallback for older "
    "results, because the report quotes these figures."
)


def require_summary(path: Path) -> None:
    """Stop unless a summary CSV exists and carries the dispersion columns."""
    if not path.exists():
        raise SystemExit(f"error: {path} does not exist. {RERUN_HINT}")
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    missing = [column for column in DISPERSION_COLUMNS if column not in header]
    if missing:
        raise SystemExit(
            f"error: {path} has no dispersion column(s) {', '.join(missing)}. {RERUN_HINT}"
        )


def require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(f"error: {path} does not exist, so there is no {what}. {RERUN_HINT}")


def require_benchmark_inputs(results: Path) -> None:
    """Fail before anything is written, so a refused run leaves the report as it was."""
    require_summary(results / "benchmarks.csv")
    require_file(results / "benchmark_trials.csv", "raw trial data")
    require_file(results / "benchmark_repeatability.csv", "session drift figure")
    if (results / "tile_sweep.csv").exists():
        require_summary(results / "tile_sweep.csv")
        require_file(results / "tile_sweep_trials.csv", "raw trial data for the tile sweep")


def quantile(values: list[float], fraction: float) -> float:
    """Linear interpolation on the sorted sample, as in include/cps/core/bench_rows.hpp."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def trials_by_round(trials: list[dict[str, str]], variant: str, count: int,
                    tile: int | None = None) -> dict[int, float]:
    """One configuration's time in each round. A repeated round is an error."""
    times: dict[int, float] = {}
    for row in trials:
        if (row["variant"] != variant or int(row["num_particles"]) != count
                or (tile is not None and int(row["tile_size"]) != tile)):
            continue
        number = int(row["round"])
        if number in times:
            raise SystemExit(
                f"error: the raw trials hold round {number} of {variant} at N={count} twice, so "
                "a sweep was interrupted between writing its trials and its rows. Rerun it."
            )
        times[number] = float(row["ms_per_step"])
    return times


def paired_ratio(numerator: dict[int, float],
                 denominator: dict[int, float]) -> tuple[float, float, int]:
    """Median and IQR of the per round time ratio, and the number of rounds.

    Two configurations measured in the same round saw the same machine, so the
    ratio inside a round cancels the session drift that a ratio of two medians
    keeps.
    """
    if not numerator or numerator.keys() != denominator.keys():
        raise SystemExit("error: the raw trials of a compared pair do not cover the same rounds")
    ratios = [numerator[number] / denominator[number] for number in numerator]
    return (quantile(ratios, 0.5), quantile(ratios, 0.75) - quantile(ratios, 0.25), len(ratios))


def read_gate_registry(path: Path) -> list[str]:
    """Registered gate names from tests/validation/gates.txt, in order.

    The same parser lives in scripts/check_provenance.py. It is repeated rather
    than imported so each script still runs on its own.
    """
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


# Macro names cannot hold digits, so a particle count in a name is spelled out.
COUNT_WORDS = {
    10000: "TenThousand", 20000: "TwentyThousand", 25000: "TwentyFiveThousand",
    50000: "FiftyThousand", 75000: "SeventyFiveThousand", 100000: "HundredThousand",
    150000: "HundredFiftyThousand", 200000: "TwoHundredThousand",
}


def count_word(count: int) -> str:
    if count not in COUNT_WORDS:
        raise SystemExit(f"error: particle count {count} has no name in COUNT_WORDS. Add one.")
    return COUNT_WORDS[count]


def split_modes(values: list[float]) -> tuple[list[float], list[float]]:
    """A sample cut at its widest gap, into the values below and above it."""
    ordered = sorted(values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    cut = gaps.index(max(gaps)) + 1
    return ordered[:cut], ordered[cut:]


def latex_escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                     ("#", r"\#"), ("$", r"\$")):
        text = text.replace(old, new)
    return text


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def sci(value: float) -> str:
    """Scientific notation as LaTeX, avoiding the letter e in prose."""
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0 ** exponent)
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


def tex_to_markdown(value: str) -> str:
    """A macro's value as plain text, for a Markdown table cell."""
    match = re.fullmatch(r"\$(-?[0-9.]+) \\times 10\^\{(-?[0-9]+)\}\$", value)
    if match:
        return f"{match.group(1)}e{match.group(2)}"
    for old, new in ((r"\_", "_"), (r"\%", "%"), (r"\&", "&"), (r"\#", "#"), (r"\$", "$"),
                     (r"\textbackslash{}", "\\")):
        value = value.replace(old, new)
    return value.replace("|", r"\|")


class Assets:
    def __init__(self, results: Path, out_root: Path, registry: Path, scratch: bool) -> None:
        self.results = results
        # A results directory other than the committed one, usually holding only
        # what bench_all wrote. The report is never built from one.
        self.scratch = scratch
        self.registry = registry
        self.tables = out_root / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.out_root = out_root
        self.macros: dict[str, str] = {}
        # The file or files, comma separated, that the current step's macros are
        # computed from.
        self.source = ""
        self.sources: dict[str, str] = {}

    def macro(self, name: str, value: str) -> None:
        # A LaTeX control sequence is letters only. A digit silently ends the
        # name and the rest becomes body text, which shows up much later as a
        # baffling "missing begin document" error in an unrelated place.
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must contain only letters")
        self.macros[name] = value
        self.sources[name] = self.source

    def write_table(self, name: str, body: str) -> None:
        path = self.tables / f"{name}.tex"
        path.write_text(body, encoding="utf-8")
        print(f"  wrote tables/{name}.tex")

    # -- device -----------------------------------------------------------
    def device(self) -> dict:
        path = self.results / "device.json"
        if not path.exists():
            return {}
        info = json.loads(path.read_text(encoding="utf-8"))
        self.macro("GpuName", latex_escape(str(info.get("name", "unknown"))))
        self.macro("GpuCompute", str(info.get("compute_capability", "")))
        self.macro("GpuSms", str(info.get("multiprocessor_count", "")))
        self.macro("GpuBusBits", str(info.get("memory_bus_width_bits", "")))
        self.macro("GpuPeakBandwidth", fmt(float(info.get("peak_bandwidth_gb_s", 0.0)), 1))
        self.macro("GpuMeasuredBandwidth", fmt(float(info.get("measured_copy_gb_s", 0.0)), 1))
        peak = float(info.get("peak_bandwidth_gb_s", 0.0))
        measured = float(info.get("measured_copy_gb_s", 0.0))
        if peak > 0.0:
            self.macro("GpuBandwidthEfficiency", fmt(100.0 * measured / peak, 1))
        # Four bytes per float, for the arithmetic against memory comparison.
        self.macro("MeasuredFloatsBillions", f"{measured / 4.0:.0f}")
        self.macro("GpuSharedMemoryKib", f"{int(info.get('shared_memory_per_block_bytes', 0)) // 1024}")
        return info

    # -- N-body ladder ----------------------------------------------------
    def nbody(self, rows: list[dict[str, str]]) -> None:
        by_variant: dict[str, dict[int, dict[str, float]]] = {}
        for row in rows:
            if row["solver"] != "nbody":
                continue
            by_variant.setdefault(row["variant"], {})[int(row["num_particles"])] = {
                "ms": float(row["ms_per_step"]),
                "gflops": float(row["gflops"]),
                "gbs": float(row["model_gb_s"]),
                "interactions": float(row["interactions_per_second"]),
                "bytes": float(row["model_bytes_per_step"]),
            }
        if "naive" not in by_variant:
            return

        counts = sorted(by_variant["naive"].keys())
        lines = [
            r"\begin{tabular}{r rrr rr r}",
            r"\toprule",
            r"& \multicolumn{3}{c}{milliseconds per step} & \multicolumn{2}{c}{speedup} "
            r"& tiled \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
            r"particles & naive & tiled & fused & tiled & fused & TFLOP/s \\",
            r"\midrule",
        ]
        for count in counts:
            naive = by_variant["naive"][count]["ms"]
            tiled = by_variant["tiled"][count]["ms"]
            fused = by_variant["fused"][count]["ms"]
            gflops = by_variant["tiled"][count]["gflops"] / 1000.0
            lines.append(
                f"{count:,} & {naive:.3f} & {tiled:.3f} & {fused:.3f} & "
                f"{naive / tiled:.2f} & {naive / fused:.2f} & {gflops:.2f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("nbody_ladder", "\n".join(lines) + "\n")

        # Per rung deltas, which is what the ladder is actually for.
        delta_lines = [
            r"\begin{tabular}{r rr rr}",
            r"\toprule",
            r"& \multicolumn{2}{c}{naive to tiled} & \multicolumn{2}{c}{tiled to fused} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"particles & time saved & speedup & time saved & speedup \\",
            r"\midrule",
        ]
        for count in counts:
            naive = by_variant["naive"][count]["ms"]
            tiled = by_variant["tiled"][count]["ms"]
            fused = by_variant["fused"][count]["ms"]
            delta_lines.append(
                f"{count:,} & {100.0 * (naive - tiled) / naive:+.1f}\\% & {naive / tiled:.3f} & "
                f"{100.0 * (tiled - fused) / tiled:+.1f}\\% & {tiled / fused:.3f} \\\\"
            )
        delta_lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("nbody_rung_deltas", "\n".join(delta_lines) + "\n")

        # Every cell of the ladder, for the Markdown table that quotes it.
        for count in counts:
            word = count_word(count)
            naive = by_variant["naive"][count]["ms"]
            tiled = by_variant["tiled"][count]["ms"]
            fused = by_variant["fused"][count]["ms"]
            self.macro(f"NaiveMsAt{word}", f"{naive:.3f}")
            self.macro(f"TiledMsAt{word}", f"{tiled:.3f}")
            self.macro(f"FusedMsAt{word}", f"{fused:.3f}")
            self.macro(f"TilingSpeedupAt{word}", f"{naive / tiled:.2f}")
            self.macro(f"TiledGigaInteractionsAt{word}",
                       f"{by_variant['tiled'][count]['interactions'] / 1.0e9:.0f}")

        largest = counts[-1]
        naive = by_variant["naive"][largest]["ms"]
        tiled = by_variant["tiled"][largest]["ms"]
        fused = by_variant["fused"][largest]["ms"]
        self.macro("LadderMaxN", f"{largest:,}")
        self.macro("NaiveMsAtMaxN", f"{naive:.2f}")
        self.macro("TiledMsAtMaxN", f"{tiled:.2f}")
        self.macro("FusedMsAtMaxN", f"{fused:.2f}")
        self.macro("TilingSpeedupAtMaxN", f"{naive / tiled:.2f}")
        self.macro("FusionSpeedupAtMaxN", f"{tiled / fused:.3f}")
        self.macro("FusionGainPercentAtMaxN", f"{100.0 * (tiled - fused) / tiled:+.1f}")
        self.macro("TiledTflopsAtMaxN", f"{by_variant['tiled'][largest]['gflops'] / 1000.0:.2f}")
        self.tiled_interactions_at_max = by_variant["tiled"][largest]["interactions"]
        self.macro("TiledGigaInteractionsAtMaxN", f"{self.tiled_interactions_at_max / 1.0e9:.0f}")
        # What bench_all converts interactions to flops with, read back from its rows.
        self.macro("ConventionalFlopsPerInteraction",
                   f"{by_variant['tiled'][largest]['gflops'] * 1.0e9 / self.tiled_interactions_at_max:.0f}")
        self.source = "benchmark_trials.csv"
        self.paired_ladder(counts)
        self.source = "benchmarks.csv"

        smallest = counts[0]
        self.macro("LadderMinN", f"{smallest:,}")
        self.macro(
            "FusionGainPercentAtMinN",
            f"{100.0 * (by_variant['tiled'][smallest]['ms'] - by_variant['fused'][smallest]['ms']) / by_variant['tiled'][smallest]['ms']:+.1f}",
        )
        best = max(
            (by_variant[v][c]["gflops"] for v in by_variant for c in by_variant[v]),
            default=0.0,
        )
        self.macro("PeakTflops", f"{best / 1000.0:.2f}")

        # What the traffic fusion removes would cost at streaming bandwidth. The
        # byte counts come from the model bench_all records, which reads the code:
        # both of the split path's integrator launches read the acceleration array,
        # and the fused kernel reads it in neither.
        measured_bandwidth = float(self.macros.get("GpuMeasuredBandwidth", "0").replace(",", ""))
        if measured_bandwidth > 0.0:
            self.source = "benchmarks.csv,device.json"
            per_ms = 1.0e3 / (measured_bandwidth * 1.0e9)
            reads_ms = 2.0 * 12.0 * largest * per_ms
            self.macro("AccelReadsMs", f"{reads_ms:.4f}")
            self.macro("AccelReadsPercent", f"{100.0 * reads_ms / tiled:.3f}")
            tiled_bytes = by_variant["tiled"][largest]["bytes"]
            fused_bytes = by_variant["fused"][largest]["bytes"]
            self.macro("ModelBytesPerParticleSplit", f"{tiled_bytes / largest:.0f}")
            self.macro("ModelBytesPerParticleFused", f"{fused_bytes / largest:.0f}")
            self.macro("ModelBytesPerParticleSaved", f"{(tiled_bytes - fused_bytes) / largest:.0f}")
            saved_ms = (tiled_bytes - fused_bytes) * per_ms
            self.macro("ModelSavingMsAtMaxN", f"{saved_ms:.4f}")
            self.macro("ModelSavingPercentAtMaxN", f"{100.0 * saved_ms / tiled:.3f}")
            smallest_tiled = by_variant["tiled"][smallest]
            saved_small_ms = (smallest_tiled["bytes"] - by_variant["fused"][smallest]["bytes"]) * per_ms
            self.macro("ModelSavingPercentAtMinN", f"{100.0 * saved_small_ms / smallest_tiled['ms']:.2f}")

    def paired_ladder(self, counts: list[int]) -> None:
        """Per round time ratios for each variant pair, a ratio below 1 favouring the first."""
        trials = [r for r in read_csv(self.results / "benchmark_trials.csv")
                  if r["solver"] == "nbody"]
        lines = [
            r"\begin{tabular}{r rr rr}",
            r"\toprule",
            r"& \multicolumn{2}{c}{fused / tiled} & \multicolumn{2}{c}{tiled / naive} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"particles & median & IQR & median & IQR \\",
            r"\midrule",
        ]
        for count in counts:
            naive = trials_by_round(trials, "naive", count)
            tiled = trials_by_round(trials, "tiled", count)
            fused = trials_by_round(trials, "fused", count)
            fused_median, fused_iqr, rounds = paired_ratio(fused, tiled)
            tiled_median, tiled_iqr, _ = paired_ratio(tiled, naive)
            word = count_word(count)
            self.macro(f"FusedOverTiledRatioAt{word}", f"{fused_median:.4f}")
            self.macro(f"FusedOverTiledIqrAt{word}", f"{fused_iqr:.4f}")
            self.macro(f"TiledOverNaiveRatioAt{word}", f"{tiled_median:.4f}")
            lines.append(
                f"{count:,} & {fused_median:.4f} & {fused_iqr:.4f} & "
                f"{tiled_median:.4f} & {tiled_iqr:.4f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("paired_ratios", "\n".join(lines) + "\n")

        # The loop ended on the largest particle count.
        self.macro("FusedOverTiledRatioAtMaxN", f"{fused_median:.4f}")
        self.macro("FusedOverTiledRatioIqrAtMaxN", f"{fused_iqr:.4f}")
        self.macro("TiledOverNaiveRatioAtMaxN", f"{tiled_median:.4f}")
        self.macro("TiledOverNaiveRatioIqrAtMaxN", f"{tiled_iqr:.4f}")
        self.macro("PairedRoundsAtMaxN", str(rounds))
        # The tiled trials at the largest size fall into two groups, which is why
        # the ratio there is not resolved.
        low, high = split_modes(list(tiled.values()))
        self.macro("TiledLowModeMsAtMaxN", f"{quantile(low, 0.5):.1f}")
        self.macro("TiledHighModeMsAtMaxN", f"{quantile(high, 0.5):.1f}")
        self.macro("TiledHighModeRoundsAtMaxN", str(len(high)))

    # -- session drift ----------------------------------------------------
    def repeatability(self) -> None:
        path = self.results / "benchmark_repeatability.csv"
        rows = read_csv(path)
        checkpoints = {row["checkpoint"]: row for row in rows}
        missing = [name for name in ("start", "middle", "end") if name not in checkpoints]
        if missing or len(rows) != 3:
            raise SystemExit(
                f"error: {path} must hold one row each for start, middle and end, and it lacks "
                f"{', '.join(missing) or 'that shape'}. {RERUN_HINT}"
            )
        medians = [float(row["ms_median"]) for row in rows]
        spread = (max(medians) - min(medians)) / quantile(medians, 0.5)
        self.macro("RepeatabilitySpread", f"{100.0 * spread:.2f}")
        self.macro("RepeatabilityVariant", latex_escape(rows[0]["variant"]))
        self.macro("RepeatabilityParticles", f"{int(rows[0]['num_particles']):,}")

    # -- fluid ------------------------------------------------------------
    def sph(self, rows: list[dict[str, str]]) -> None:
        points = sorted(
            (int(r["num_particles"]), float(r["ms_per_step"]), float(r["model_gb_s"]))
            for r in rows
            if r["solver"] == "sph"
        )
        if not points:
            return
        lines = [
            r"\begin{tabular}{r rrr}",
            r"\toprule",
            r"particles & milliseconds per step & steps per second & million particles per second \\",
            r"\midrule",
        ]
        for count, ms, _gbs in points:
            lines.append(
                f"{count:,} & {ms:.4f} & {1000.0 / ms:,.0f} & {count / (ms * 1.0e-3) / 1.0e6:.1f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("sph_scaling", "\n".join(lines) + "\n")

        for count, ms, _gbs in points:
            word = count_word(count)
            self.macro(f"SphStepMsAt{word}", f"{ms:.4f}")
            self.macro(f"SphStepsPerSecondAt{word}", f"{1000.0 / ms:,.0f}")

        at_50k = [p for p in points if p[0] == 50000]
        if at_50k:
            self.macro("SphMsAtFiftyThousand", f"{at_50k[0][1]:.3f}")
            self.macro("SphFpsAtFiftyThousand", f"{1000.0 / at_50k[0][1]:,.0f}")
            # Against one frame at sixty frames per second.
            self.macro("SphInteractiveMargin", f"{(1000.0 / 60.0) / at_50k[0][1]:.0f}")
        self.macro("SphCountRatio", f"{points[-1][0] / points[0][0]:.0f}")
        self.macro("SphTimeRatio", f"{points[-1][1] / points[0][1]:.1f}")
        # The fluid moves during a measurement, so its trials spread by more than
        # the drift does. How far, and how far the fluid got.
        fluid = [r for r in rows if r["solver"] == "sph"]
        spreads = [100.0 * float(r["ms_iqr"]) / float(r["ms_median"]) for r in fluid]
        self.macro("SphIqrLeastPercent", f"{min(spreads):.1f}")
        self.macro("SphIqrMostPercent", f"{max(spreads):.1f}")
        timed = [int(r["timed_steps"]) * int(r["trials"]) for r in fluid]
        self.macro("SphTimedStepsLeast", f"{min(timed):,}")
        self.macro("SphTimedStepsMost", f"{max(timed):,}")
        largest = points[-1]
        self.macro("SphMaxN", f"{largest[0]:,}")
        self.macro("SphMsAtMaxN", f"{largest[1]:.3f}")
        self.macro("SphFpsAtMaxN", f"{1000.0 / largest[1]:,.0f}")

    # -- tile sweep -------------------------------------------------------
    def tile_sweep(self) -> None:
        rows = read_csv(self.results / "tile_sweep.csv")
        if not rows:
            return
        tiles = sorted({int(r["tile_size"]) for r in rows})
        lines = [
            r"\begin{tabular}{r rr rr}",
            r"\toprule",
            r"& \multicolumn{2}{c}{tiled} & \multicolumn{2}{c}{fused} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"tile width & ms per step & TFLOP/s & ms per step & TFLOP/s \\",
            r"\midrule",
        ]
        best_tile, best_ms = None, None
        for tile in tiles:
            cells = []
            for variant in ("tiled", "fused"):
                match = [r for r in rows if int(r["tile_size"]) == tile and r["variant"] == variant]
                if match:
                    ms = float(match[0]["ms_per_step"])
                    cells.append(f"{ms:.3f} & {float(match[0]['gflops']) / 1000.0:.2f}")
                    if variant == "tiled" and (best_ms is None or ms < best_ms):
                        best_ms, best_tile = ms, tile
                else:
                    cells.append("-- & --")
            lines.append(f"{tile} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("tile_sweep", "\n".join(lines) + "\n")

        tile_32 = [r for r in rows if int(r["tile_size"]) == 32 and r["variant"] == "tiled"]
        if tile_32 and best_ms is not None:
            tile_32_ms = float(tile_32[0]["ms_per_step"])
            self.macro("BestTileWidth", str(best_tile))
            self.macro("BestTileMs", f"{best_ms:.3f}")
            self.macro("TileThirtyTwoMs", f"{tile_32_ms:.3f}")
            self.macro("TileSweepParticles", f"{int(tile_32[0]['num_particles']):,}")
            self.macro("TileWidthGainPercent", f"{100.0 * (tile_32_ms - best_ms) / tile_32_ms:.1f}")
            # What deciding the default rests on: the fastest width against a
            # width of 32, round by round.
            self.source = "tile_sweep_trials.csv"
            trials = read_csv(self.results / "tile_sweep_trials.csv")
            count = int(tile_32[0]["num_particles"])
            ratio, ratio_iqr, _ = paired_ratio(trials_by_round(trials, "tiled", count, best_tile),
                                               trials_by_round(trials, "tiled", count, 32))
            self.macro("BestTileOverTileThirtyTwoRatio", f"{ratio:.4f}")
            self.macro("BestTileOverTileThirtyTwoRatioIqr", f"{ratio_iqr:.4f}")

    # -- register blocking, measured and removed --------------------------
    def blocking_sweep(self) -> None:
        # The kernels this measured are gone, at a commit that stays reachable,
        # so these numbers only ever describe that one sweep.
        rows = read_csv(self.results / "blocking_sweep.csv")
        if not rows:
            raise SystemExit("error: blocking_sweep.csv is missing or empty")
        factors = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
        sizes = {10000: "TenK", 100000: "HundredK", 200000: "TwoHundredK"}
        self.macro("BlockingCommit", next(r["commit"] for r in rows if r["kernels"] == "blocking"))
        self.macro("BlockingRounds", str(max(int(r["round"]) for r in rows)))

        def trials(kernels: str, variant: str, count: int, factor: int, block: int) -> dict[int, float]:
            return {int(r["round"]): float(r["ms_per_step"]) for r in rows
                    if r["kernels"] == kernels and r["variant"] == variant
                    and int(r["num_particles"]) == count and int(r["blocking_factor"]) == factor
                    and int(r["block_size"]) == block}

        best_ms = None
        for count, size in sizes.items():
            for variant in ("tiled", "fused"):
                before = trials("before", variant, count, 1, 256)
                # Blocks multiplied by threads per block held constant, then the
                # one fixed block case, which shows what starving the machine costs.
                for factor, block, tag in ((1, 256, ""), (2, 128, ""), (4, 64, ""), (8, 32, ""),
                                           (2, 256, "FixedBlock")):
                    after = trials("blocking", variant, count, factor, block)
                    ratio = statistics.median(after[k] / before[k] for k in after)
                    self.macro(f"Blocking{variant.capitalize()}{tag}Factor{factors[factor]}{size}",
                               f"{ratio:.4f}")
                    if count == max(sizes):
                        median = statistics.median(after.values())
                        best_ms = median if best_ms is None else min(best_ms, median)
        fused = statistics.median(trials("before", "fused", max(sizes), 1, 256).values())
        self.macro("BlockingBestOverFusedAtMaxN", f"{best_ms / fused:.4f}")

        # Why it could not pay: the instruction mix of the kernel it would replace.
        self.source = "profile_summary.csv"
        tiled = [r for r in read_csv(self.results / "profile_summary.csv")
                 if r["variant"] == "tiled" and int(r["num_particles"]) == max(sizes)][0]
        self.macro("TiledFmaPerLoadAtMaxN",
                   f"{float(tiled['fma_instructions']) / float(tiled['lsu_instructions']):.1f}")

    # -- validation gates -------------------------------------------------
    def gates(self) -> None:
        # A missing gate would leave a hole in the table and a macro the prose
        # quotes undefined, so the report is never built from a partial file.
        path = self.results / "validation_gates.csv"
        if self.scratch and not path.exists():
            print(f"  no validation_gates.csv in {self.results}, skipping the gates")
            return
        if not self.registry.exists():
            raise SystemExit(f"error: the gate registry {self.registry} does not exist")
        rows = read_csv(path)
        present = {row["gate"] for row in rows}
        missing = [gate for gate in read_gate_registry(self.registry) if gate not in present]
        if missing:
            raise SystemExit(
                f"error: {path} is missing {len(missing)} registered gate(s): "
                + ", ".join(missing)
                + f". Every gate in {self.registry} must have a row; rerun the validation "
                "suite with CPS_RECORD_GATES=1 to record them."
            )
        pretty = {
            "two_body_energy_drift_10000_steps":
                "Two body orbit, relative energy drift over 10,000 steps",
            "two_body_orbit_out_of_round": "Two body orbit, departure from circular",
            "plummer_momentum_drift_1000_steps":
                "Plummer 10k, relative momentum drift over 1,000 steps",
            "force_vs_cpu_reference_100_particles":
                "Force against a double precision CPU reference, 100 particles",
            "pairwise_force_antisymmetry_residual": "Pairwise force antisymmetry residual",
            "plummer_radial_density_chi_square_p_value":
                "Plummer radial density, chi squared p value",
            "sph_poly6_kernel_normalisation_error": "SPH poly6 kernel normalisation error",
            "sph_spiky_kernel_normalisation_error": "SPH spiky kernel normalisation error",
            "sph_interior_density_mean_error": "SPH interior density, mean error",
            "sph_interior_density_rms_deviation": "SPH interior density, RMS deviation",
            "sph_interior_density_max_deviation": "SPH interior density, maximum deviation",
            "verlet_energy_slope_error_eccentric":
                "Velocity Verlet energy order, eccentric orbit, distance of slope from 2",
            "verlet_energy_slope_error_circular":
                "Velocity Verlet energy order, circular orbit, distance of slope from 4",
            "sph_neighbour_search_vs_brute_force":
                "SPH neighbour search against a brute force sum",
        }
        lines = [
            r"\begin{tabular}{l r r c}",
            r"\toprule",
            r"gate & measured & limit & verdict \\",
            r"\midrule",
        ]
        all_pass = True
        for row in rows:
            name = pretty.get(row["gate"], latex_escape(row["gate"]))
            measured = float(row["measured"])
            limit = float(row["limit"])
            passed = row["passed"] == "1"
            all_pass = all_pass and passed
            relation = "$<$" if row["comparison"] == "less_than" else "$>$"
            measured_text = f"{measured:.4f}" if 1.0e-3 <= abs(measured) < 1.0e4 else sci(measured)
            limit_text = f"{limit:.4f}" if 1.0e-3 <= abs(limit) < 1.0e4 else sci(limit)
            verdict = r"\textbf{pass}" if passed else r"\textbf{FAIL}"
            lines.append(f"{name} & {measured_text} & {relation} {limit_text} & {verdict} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("validation_gates", "\n".join(lines) + "\n")

        self.macro("GateCount", str(len(rows)))
        self.macro("AllGatesPass", "true" if all_pass else "false")
        for row in rows:
            if row["gate"] == "two_body_energy_drift_10000_steps":
                self.macro("EnergyDriftMeasured", sci(float(row["measured"])))
            if row["gate"] == "plummer_momentum_drift_1000_steps":
                self.macro("MomentumDriftMeasured", sci(float(row["measured"])))
            if row["gate"] == "plummer_radial_density_chi_square_p_value":
                self.macro("PlummerPValue", f"{float(row['measured']):.4f}")
            if row["gate"] == "sph_poly6_kernel_normalisation_error":
                self.macro("PolySixNormalisationError", sci(float(row["measured"])))
            if row["gate"] == "sph_interior_density_mean_error":
                self.macro("SphDensityErrorPercent", f"{100.0 * float(row['measured']):.2f}")
            if row["gate"] == "sph_interior_density_rms_deviation":
                self.macro("SphDensityRmsPercent", f"{100.0 * float(row['measured']):.2f}")
            if row["gate"] == "sph_interior_density_max_deviation":
                self.macro("SphDensityMaxPercent", f"{100.0 * float(row['measured']):.2f}")
            if row["gate"] == "two_body_orbit_out_of_round":
                self.macro("OrbitOutOfRoundMeasured", sci(float(row["measured"])))
            if row["gate"] == "verlet_energy_slope_error_eccentric":
                self.macro("VerletSlopeErrorEccentric", f"{float(row['measured']):.4f}")
            if row["gate"] == "verlet_energy_slope_error_circular":
                self.macro("VerletSlopeErrorCircular", f"{float(row['measured']):.4f}")
            if row["gate"] == "force_vs_cpu_reference_100_particles":
                self.macro("ForceReferenceError", sci(float(row["measured"])))
            if row["gate"] == "pairwise_force_antisymmetry_residual":
                self.macro("AntisymmetryResidual", sci(float(row["measured"])))
            if row["gate"] == "sph_spiky_kernel_normalisation_error":
                self.macro("SpikyNormalisationError", sci(float(row["measured"])))
            if row["gate"] == "sph_neighbour_search_vs_brute_force":
                self.macro("NeighbourSearchError", sci(float(row["measured"])))

    # -- hardware counters ------------------------------------------------
    def profile(self) -> None:
        all_rows = read_csv(self.results / "profile_summary.csv")
        if not all_rows:
            return
        order = {"naive": 0, "tiled": 1, "fused": 2}
        all_rows.sort(key=lambda r: order.get(r["variant"], 9))
        self.profile_stalls(all_rows)
        # The counter table and its macros are for the largest size profiled. An
        # older summary holds one size, and one written before the column existed
        # was profiled at 100,000.
        headline = max(int(r.get("num_particles") or 100000) for r in all_rows)
        rows = [r for r in all_rows if int(r.get("num_particles") or 100000) == headline]

        lines = [
            r"\begin{tabular}{l rrrr rr}",
            r"\toprule",
            r"& \multicolumn{4}{c}{throughput, percent of peak} "
            r"& \multicolumn{2}{c}{per launch} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
            r"variant & SM & L1 and load store & L2 & device memory "
            r"& load sectors & warp issue \\",
            r"\midrule",
        ]
        for row in rows:
            sectors = float(row["global_load_sectors"])
            lines.append(
                f"{row['variant']} & {float(row['sm_throughput_pct']):.2f} & "
                f"{float(row['l1tex_throughput_pct']):.2f} & "
                f"{float(row['l2_throughput_pct']):.2f} & "
                f"{float(row['dram_throughput_pct']):.2f} & "
                f"{sectors / 1.0e6:,.1f}M & {float(row['issue_active_pct']):.2f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("profile_counters", "\n".join(lines) + "\n")

        keyed = {r["variant"]: r for r in rows}
        # Every cell of the counter table, for the Markdown table that quotes it.
        for row in rows:
            variant = row["variant"].capitalize()
            self.macro(f"{variant}SmPct", f"{float(row['sm_throughput_pct']):.2f}")
            self.macro(f"{variant}LoneTexCellPct", f"{float(row['l1tex_throughput_pct']):.2f}")
            self.macro(f"{variant}LtwoPct", f"{float(row['l2_throughput_pct']):.2f}")
            self.macro(f"{variant}DramCellPct", f"{float(row['dram_throughput_pct']):.2f}")
            self.macro(f"{variant}SectorsMillions", f"{float(row['global_load_sectors']) / 1.0e6:,.1f}")
            self.macro(f"{variant}IssueCellPct", f"{float(row['issue_active_pct']):.2f}")
        if "naive" in keyed and "tiled" in keyed:
            naive, tiled = keyed["naive"], keyed["tiled"]
            self.macro("NaiveLoneTexPct", f"{float(naive['l1tex_throughput_pct']):.1f}")
            self.macro("TiledLoneTexPct", f"{float(tiled['l1tex_throughput_pct']):.1f}")
            self.macro("NaiveDramPct", f"{float(naive['dram_throughput_pct']):.2f}")
            self.macro("TiledDramPct", f"{float(tiled['dram_throughput_pct']):.2f}")
            self.macro("NaiveIssuePct", f"{float(naive['issue_active_pct']):.1f}")
            self.macro("TiledIssuePct", f"{float(tiled['issue_active_pct']):.1f}")
            sector_ratio = float(naive["global_load_sectors"]) / float(tiled["global_load_sectors"])
            self.macro("SectorReduction", f"{sector_ratio:.1f}")
            lsu_ratio = float(naive["lsu_instructions"]) / float(tiled["lsu_instructions"])
            self.macro("LsuReduction", f"{lsu_ratio:.1f}")

            # The conventional twenty operations per interaction, checked
            # against what the hardware actually retired.
            fadd = float(tiled["fadd_thread_inst"])
            ffma = float(tiled["ffma_thread_inst"])
            fmul = float(tiled["fmul_thread_inst"])
            # A summary written before this column existed was profiled at 100,000.
            particles = float(tiled.get("num_particles") or 100000)
            interactions = particles * particles
            measured_flops = (fadd + fmul + 2.0 * ffma) / interactions
            self.macro("MeasuredFlopsPerInteraction", f"{measured_flops:.1f}")
            self.macro("ProfileParticles", f"{int(particles):,}")
            # The headline throughput restated at the operation count the counters give.
            interactions_per_second = getattr(self, "tiled_interactions_at_max", 0.0)
            if interactions_per_second > 0.0:
                self.macro("TiledTflopsAtMeasuredFlops",
                           f"{interactions_per_second * measured_flops / 1.0e12:.2f}")

    def profile_stalls(self, rows: list[dict[str, str]]) -> None:
        """The stall breakdown, for each variant at each size, once it has been profiled."""
        if not rows or "stall_wait" not in rows[0]:
            return
        ordered = sorted(rows, key=lambda r: -int(r["num_particles"]))
        lines = [
            r"\begin{tabular}{r l rrrrrrrrr rr}",
            r"\toprule",
            r"& & \multicolumn{9}{c}{warps stalled per cycle with an issue active} "
            r"& \multicolumn{2}{c}{L1, L2 hit} \\",
            r"\cmidrule(lr){3-11}\cmidrule(lr){12-13}",
            r"particles & variant & wait & not sel. & short sb. & long sb. & math & MIO & LG "
            r"& barrier & no inst. & percent & percent \\",
            r"\midrule",
        ]
        for row in ordered:
            cells = " & ".join(f"{float(row[column]):.2f}" for column in STALL_COLUMNS)
            lines.append(f"{int(row['num_particles']):,} & {row['variant']} & {cells} & "
                         f"{float(row['l1_global_load_hit_pct']):.2f} & "
                         f"{float(row['l2_hit_pct']):.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("profile_stalls", "\n".join(lines) + "\n")

        for row in ordered:
            size = PROFILE_SIZE_NAMES.get(int(row["num_particles"]))
            if size is None:
                continue
            variant = row["variant"].capitalize()
            for column, word in STALL_COLUMNS.items():
                self.macro(f"{variant}Stall{word}{size}", f"{float(row[column]):.2f}")
            self.macro(f"{variant}RegistersPerThread{size}",
                       f"{float(row['registers_per_thread']):.0f}")
            self.macro(f"{variant}WavesPerMultiprocessor{size}",
                       f"{float(row['waves_per_multiprocessor']):.2f}")
            self.macro(f"{variant}IssueActivePct{size}", f"{float(row['issue_pct_active']):.2f}")

    def sph_split(self) -> None:
        rows = read_csv(self.results / "sph_pipeline_split.csv")
        if not rows:
            return
        self.macro("SphSplitParticles", f"{int(rows[0]['num_particles']):,}")
        self.macro("SphSplitWarmupSteps", f"{int(rows[0]['warmup_steps']):,}")
        self.macro("SphSplitMeasuredSteps", f"{int(rows[0]['measured_steps']):,}")
        self.macro("SphSplitTracedStepMs", f"{float(rows[0]['traced_step_ms']):.3f}")
        self.macro("SphSplitUntracedStepMs", f"{float(rows[0]['untraced_step_ms']):.3f}")
        for row in rows:
            word = "".join(part.capitalize() for part in row["pass"].split("_"))
            self.macro(f"SphSplit{word}Pct", f"{float(row['percent_of_step']):.1f}")
            self.macro(f"SphSplit{word}Ms", f"{float(row['ms_per_step']):.4f}")

    # -- debug report -----------------------------------------------------
    def debug_entries(self, sections: Path) -> None:
        """How many entries the debug report holds, so prose cannot miscount them."""
        files = sorted(sections.glob("*.tex"))
        if not files:
            return
        self.macro("DebugEntryCount", str(sum(
            len(re.findall(r"^\\entry\{", path.read_text(encoding="utf-8"), re.MULTILINE))
            for path in files)))

    # -- provenance -------------------------------------------------------
    def provenance(self, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        first = rows[0]
        self.macro("BenchCommit", latex_escape(first.get("commit", "unknown")))
        self.macro("BenchTimestamp", latex_escape(first.get("timestamp", "")))
        self.macro("BenchSeed", latex_escape(first.get("seed", "")))
        runtime = first.get("cuda_runtime", "0")
        try:
            value = int(runtime)
            self.macro("CudaRuntime", f"{value // 1000}.{(value % 1000) // 10}")
        except ValueError:
            self.macro("CudaRuntime", latex_escape(runtime))
        self.macro("BenchFastMath", "on" if first.get("fast_math") == "1" else "off")

    def write_macros(self) -> None:
        lines = [
            "% Generated by scripts/gen_report_assets.py. Do not edit by hand.",
            "% Every value here comes from a file written by a run.",
        ]
        for name in sorted(self.macros):
            lines.append(f"\\newcommand{{\\{name}}}{{{self.macros[name]}}}")
        path = self.out_root / "numbers.tex"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote numbers.tex with {len(self.macros)} macros")

    def write_numbers(self, path: Path, root: Path) -> None:
        """Write the table scripts/check_doc_numbers.py compares the Markdown files with."""
        results = self.results.resolve()
        folder = results.relative_to(root).as_posix() if results.is_relative_to(root) else str(results)
        lines = [
            "# Generated numbers",
            "",
            "Generated by `scripts/gen_report_assets.py`. Do not edit by hand.",
            "",
            "To quote one of these in a Markdown file, mark it with the metric's name, as in",
            "`<!-- num:RepeatabilitySpread -->1.2<!-- /num -->`. `scripts/check_doc_numbers.py`",
            "fails when a marked value disagrees with this table, and `--fix` rewrites it.",
            "",
            "| Metric | Value | Source |",
            "|---|---|---|",
        ]
        for name in sorted(self.macros):
            # A source with a folder of its own is not in the results directory.
            sources = ", ".join(item if "/" in item else f"{folder}/{item}"
                                for item in self.sources[name].split(","))
            lines.append(f"| {name} | {tex_to_markdown(self.macros[name])} | {sources} |")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {path.name} with {len(self.macros)} metrics")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate report tables and macros.")
    default_results = root / "experiments" / "results"
    parser.add_argument("results_dir", type=Path, nargs="?",
                        help="results to read (default experiments/results). Anything else is "
                             "a scratch directory, whose output goes to report_assets inside it "
                             "unless --out says otherwise")
    parser.add_argument("--results", type=Path, help="the same as the positional argument")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gates", type=Path, default=root / "tests" / "validation" / "gates.txt",
                        help="the registry of gates validation_gates.csv must contain")
    args = parser.parse_args()

    results = args.results_dir or args.results or default_results
    scratch = results.resolve() != default_results.resolve()
    out = args.out or (results / "report_assets" if scratch else root / "report")

    print(f"reading results from {results}")
    print(f"writing assets to {out}")
    require_benchmark_inputs(results)
    assets = Assets(results, out, args.gates, scratch)
    benchmarks = read_csv(results / "benchmarks.csv")

    steps = [
        ("device.json", lambda: assets.device()),
        ("benchmarks.csv", lambda: assets.nbody(benchmarks)),
        ("benchmark_repeatability.csv", lambda: assets.repeatability()),
        ("benchmarks.csv", lambda: assets.sph(benchmarks)),
        ("tile_sweep.csv", lambda: assets.tile_sweep()),
        ("validation_gates.csv", lambda: assets.gates()),
        ("profile_summary.csv", lambda: assets.profile()),
        ("sph_pipeline_split.csv", lambda: assets.sph_split()),
        ("blocking_sweep.csv", lambda: assets.blocking_sweep()),
        ("report_debug/sections", lambda: assets.debug_entries(root / "report_debug" / "sections")),
        ("benchmarks.csv", lambda: assets.provenance(benchmarks)),
    ]
    for source, action in tqdm(steps, desc="assets", unit="step"):
        assets.source = source
        action()

    assets.write_macros()
    # The committed table belongs to the committed report. Any other output
    # directory keeps its own copy, so a scratch run leaves docs alone.
    default_out = out.resolve() == (root / "report").resolve()
    numbers = root / "docs" / "generated_numbers.md" if default_out else out / "generated_numbers.md"
    assets.write_numbers(numbers, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
