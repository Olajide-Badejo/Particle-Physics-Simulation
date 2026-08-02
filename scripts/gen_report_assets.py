#!/usr/bin/env python3
"""Turn the recorded results into the tables and macros the reports include.

This exists so that no number in either PDF is ever typed by hand. Every table
and every figure quoted in the prose comes out of this script, which reads only
the CSV files that `bench_all`, `gpu_probe` and the validation suite wrote. Run
it twice on the same inputs and it produces byte identical output.

The prose in the reports quotes measured values through LaTeX macros defined in
`numbers.tex`, so a sentence such as "tiling is worth a factor of X" cannot
drift away from the table beside it: both come from the same row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


class Assets:
    def __init__(self, results: Path, out_root: Path) -> None:
        self.results = results
        self.tables = out_root / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.out_root = out_root
        self.macros: dict[str, str] = {}

    def macro(self, name: str, value: str) -> None:
        # A LaTeX control sequence is letters only. A digit silently ends the
        # name and the rest becomes body text, which shows up much later as a
        # baffling "missing begin document" error in an unrelated place.
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must contain only letters")
        self.macros[name] = value

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

        # Arithmetic intensity argument, computed rather than asserted: how much
        # of the step at the largest size the fused round trip could possibly
        # account for.
        return_trip_bytes = 2.0 * 12.0 * largest  # write then read the acceleration array
        measured_bandwidth = float(self.macros.get("GpuMeasuredBandwidth", "0").replace(",", ""))
        if measured_bandwidth > 0.0:
            trip_ms = return_trip_bytes / (measured_bandwidth * 1.0e9) * 1.0e3
            self.macro("FusionCeilingMs", f"{trip_ms:.4f}")
            self.macro("FusionCeilingPercent", f"{100.0 * trip_ms / tiled:.3f}")

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

        at_50k = [p for p in points if p[0] == 50000]
        if at_50k:
            self.macro("SphMsAtFiftyThousand", f"{at_50k[0][1]:.3f}")
            self.macro("SphFpsAtFiftyThousand", f"{1000.0 / at_50k[0][1]:,.0f}")
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

        specified = [r for r in rows if int(r["tile_size"]) == 32 and r["variant"] == "tiled"]
        if specified and best_ms is not None:
            spec_ms = float(specified[0]["ms_per_step"])
            self.macro("BestTileWidth", str(best_tile))
            self.macro("BestTileMs", f"{best_ms:.3f}")
            self.macro("SpecifiedTileMs", f"{spec_ms:.3f}")
            self.macro("TileWidthGainPercent", f"{100.0 * (spec_ms - best_ms) / spec_ms:.1f}")

    # -- validation gates -------------------------------------------------
    def gates(self) -> None:
        rows = read_csv(self.results / "validation_gates.csv")
        if not rows:
            return
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
            "sph_interior_density_relative_error": "SPH interior density, relative error",
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
            if row["gate"] == "sph_interior_density_relative_error":
                self.macro("SphDensityErrorPercent", f"{100.0 * float(row['measured']):.2f}")

    # -- hardware counters ------------------------------------------------
    def profile(self) -> None:
        rows = read_csv(self.results / "profile_summary.csv")
        if not rows:
            return
        order = {"naive": 0, "tiled": 1, "fused": 2}
        rows.sort(key=lambda r: order.get(r["variant"], 9))

        lines = [
            r"\begin{tabular}{l rrr rr}",
            r"\toprule",
            r"& \multicolumn{3}{c}{throughput, percent of peak} "
            r"& \multicolumn{2}{c}{per launch} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
            r"variant & L1 and load store & L2 & device memory "
            r"& global load sectors & warp issue \\",
            r"\midrule",
        ]
        for row in rows:
            sectors = float(row["global_load_sectors"])
            lines.append(
                f"{row['variant']} & {float(row['l1tex_throughput_pct']):.2f} & "
                f"{float(row['l2_throughput_pct']):.2f} & "
                f"{float(row['dram_throughput_pct']):.2f} & "
                f"{sectors / 1.0e6:,.1f}M & {float(row['issue_active_pct']):.2f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        self.write_table("profile_counters", "\n".join(lines) + "\n")

        keyed = {r["variant"]: r for r in rows}
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
            interactions = 100000.0 * 100000.0
            self.macro("MeasuredFlopsPerInteraction",
                       f"{(fadd + fmul + 2.0 * ffma) / interactions:.1f}")

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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate report tables and macros.")
    parser.add_argument("--results", type=Path, default=root / "experiments" / "results")
    parser.add_argument("--out", type=Path, default=root / "report")
    args = parser.parse_args()

    print(f"reading results from {args.results}")
    assets = Assets(args.results, args.out)
    benchmarks = read_csv(args.results / "benchmarks.csv")

    steps = [
        ("device properties", lambda: assets.device()),
        ("N-body ladder", lambda: assets.nbody(benchmarks)),
        ("fluid scaling", lambda: assets.sph(benchmarks)),
        ("tile width sweep", lambda: assets.tile_sweep()),
        ("validation gates", lambda: assets.gates()),
        ("hardware counters", lambda: assets.profile()),
        ("provenance", lambda: assets.provenance(benchmarks)),
    ]
    for _label, action in tqdm(steps, desc="assets", unit="step"):
        action()

    assets.write_macros()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
