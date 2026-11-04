#!/usr/bin/env python3
"""Draw every figure in the report from the recorded result files.

Nothing here invents a number. Each figure is a view of a CSV written by
`bench_all` or by the validation suite, so regenerating the figures after a
rerun is the only way the report ever changes.

Two conventions are enforced throughout.

Every axis states its unit. A bare "particles" or "simulation time" forces the
reader to guess, and a guess about units is how a result gets misread. Where a
quantity is genuinely dimensionless, the axis says so rather than borrowing a
unit it does not have: the gravitational runs use code units with the
gravitational constant set to one, so their time axis is not in seconds and is
not labelled as though it were.

Colour is checked rather than judged. `make plots` runs the three variant
colours through `scripts/validate_palette.py`, which computes their OKLab separation
under simulated protanopia and deuteranopia. All hard checks pass. The aqua
sits at 2.82 to 1 against a white page, below the 3 to 1 guideline; the
documented relief is a second channel carrying the same information, which here
is the marker shape on every series plus the full result tables printed
alongside the figures in the report.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated categorical slots one to three. Colour follows the entity: tiled is
# orange in every figure it appears in, whatever else is on the axes.
SERIES = {
    "naive": {"color": "#2a78d6", "marker": "o", "label": "naive"},
    "tiled": {"color": "#eb6834", "marker": "s", "label": "tiled"},
    "fused": {"color": "#1baf7a", "marker": "^", "label": "fused"},
}
# The passes of the SPH step, in the order they run and so in the order they stack.
# Slots one to eight of the same validated palette. Only neighbours in the stack have
# to be told apart by colour, so `make plots` checks them with --pairs adjacent; the
# legend also gives every pass its name and value.
SPH_PASSES = {
    "hash": ("hash", "#2a78d6"),
    "sort": ("sort", "#eb6834"),
    "clear": ("clear", "#1baf7a"),
    "cell_bounds": ("cell bounds", "#eda100"),
    "gather": ("gather", "#e87ba4"),
    "density": ("density", "#008300"),
    "force_and_integrate": ("force and integrate", "#6250d6"),
    "between_kernels": ("between kernels", "#e34948"),
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LEGEND_FACE = "#f0efec"
LEGEND_EDGE = "#dcdad2"

# Code units. The gravitational runs set the gravitational constant, the total
# mass and the initial separation to one, so time carries no second.
CODE_TIME = "simulation time (code units, $G=1$)"

FIGSIZE = (7.0, 4.2)
DPI = 200


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def strip_spines(ax) -> None:
    """Axes are chrome, not data. Only the two that carry scale remain."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def add_legend(ax, loc: str = "best"):
    """Legend on a panel a shade darker than the page, so it reads as a unit."""
    legend = ax.legend(loc=loc, frameon=True, framealpha=1.0, borderpad=0.6,
                       facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE)
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return legend


def thousands(value, _pos) -> str:
    """Particle counts read better as 10k than as 10000."""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def caption(fig, text: str) -> None:
    """Wrapped note under the axes.

    The wrap is not cosmetic. Figures are saved with a tight bounding box, so a
    single long caption line widens the saved image to fit it and squashes the
    plot into the middle of a very wide canvas.
    """
    wrapped = textwrap.fill(" ".join(text.split()), width=88)
    # multialignment centres each wrapped line individually; ha alone would
    # only centre the block and leave the lines ragged on the right.
    fig.text(0.5, -0.03, wrapped, ha="center", va="top", multialignment="center",
             fontsize=8, color=INK_MUTED, linespacing=1.5)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_dispersion(path: Path, rows: list[dict[str, str]]) -> None:
    """Stop on results that predate the dispersion columns the error bars come from."""
    if rows and not {"ms_median", "ms_iqr"} <= rows[0].keys():
        raise SystemExit(
            f"error: {path} has no ms_median and ms_iqr columns, so its figures cannot show "
            "their error bars. Rerun bench_all under the current protocol."
        )


def half_iqr(rows: list[dict[str, str]]) -> dict[str, dict[int, float]]:
    """Half the interquartile range, by variant and particle count."""
    out: dict[str, dict[int, float]] = {}
    for row in rows:
        out.setdefault(row["variant"], {})[int(row["num_particles"])] = float(row["ms_iqr"]) / 2.0
    return out


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        # A PDF otherwise records when it was made, so an unchanged figure would
        # read as changed on every run.
        metadata = {"CreationDate": None} if suffix == "pdf" else None
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=DPI, metadata=metadata)
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


def nbody_series(rows: list[dict[str, str]], field: str) -> dict[str, tuple[list, list]]:
    """Group the N-body rows by variant, sorted by particle count."""
    out: dict[str, tuple[list, list]] = {}
    for variant in ("naive", "tiled", "fused"):
        points = sorted(
            (int(r["num_particles"]), float(r[field]))
            for r in rows
            if r["solver"] == "nbody" and r["variant"] == variant
        )
        if points:
            out[variant] = ([p[0] for p in points], [p[1] for p in points])
    return out


def plot_nbody_time(rows, out_dir: Path) -> None:
    """Step cost against particle count for the three force variants."""
    data = nbody_series(rows, "ms_per_step")
    if not data:
        return
    spread = half_iqr([r for r in rows if r["solver"] == "nbody"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant, (xs, ys) in data.items():
        style = SERIES[variant]
        ax.errorbar(xs, ys, yerr=[spread[variant][x] for x in xs], color=style["color"],
                    marker=style["marker"], markersize=5, linewidth=1.8, capsize=2,
                    label=style["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of particles $N$ (count)")
    ax.set_ylabel("time per step (milliseconds)")
    ax.set_title("N-body step cost against particle count")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    add_legend(ax, "upper left")
    strip_spines(ax)
    caption(fig,
            "Both axes logarithmic. The slope near two is the all pairs sum; "
            "the vertical offset between the lines is what the memory hierarchy is worth. "
            "Points are medians; bars span the interquartile range of the trials.")
    save(fig, out_dir, "fig_nbody_time")


def plot_nbody_speedup(rows, out_dir: Path) -> None:
    """How many times faster each rung is than the naive baseline."""
    times = nbody_series(rows, "ms_per_step")
    if "naive" not in times:
        return
    baseline = dict(zip(times["naive"][0], times["naive"][1]))
    spread = half_iqr([r for r in rows if r["solver"] == "nbody"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    first_x = None
    for variant in ("tiled", "fused"):
        if variant not in times:
            continue
        xs, ys = times[variant]
        valid_x = [x for x in xs if x in baseline]
        speedups = [baseline[x] / y for x, y in zip(xs, ys) if x in baseline]
        # Relative spreads add in quadrature for a ratio of two independent times.
        bars = [
            baseline[x] / y * math.hypot(spread["naive"][x] / baseline[x], spread[variant][x] / y)
            for x, y in zip(xs, ys)
            if x in baseline
        ]
        style = SERIES[variant]
        ax.errorbar(valid_x, speedups, yerr=bars, color=style["color"], marker=style["marker"],
                    markersize=5, linewidth=1.8, capsize=2, label=style["label"])
        first_x = first_x or valid_x[0]

    ax.axhline(1.0, color=AXIS, linewidth=1.0, linestyle="--")
    ax.annotate("naive baseline", xy=(first_x, 1.0), xytext=(0, 6),
                textcoords="offset points", color=INK_MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("number of particles $N$ (count)")
    # A ratio of two times, so the unit is a pure number.
    ax.set_ylabel("speedup against naive (ratio, dimensionless)")
    ax.set_title("What each rung of the ladder is worth")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(bottom=0.0)
    add_legend(ax, "lower right")
    strip_spines(ax)
    caption(fig,
            "A value of 2.0 means the step took half as long as the naive kernel at the "
            "same particle count. The dashed line is the naive kernel itself. Bars combine "
            "the interquartile ranges of the two times.")
    save(fig, out_dir, "fig_nbody_speedup")


def plot_nbody_throughput(rows, out_dir: Path) -> None:
    """Sustained single precision arithmetic rate."""
    data = nbody_series(rows, "gflops")
    if not data:
        return
    times = nbody_series(rows, "ms_per_step")
    spread = half_iqr([r for r in rows if r["solver"] == "nbody"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant, (xs, ys) in data.items():
        style = SERIES[variant]
        # Throughput is proportional to the inverse of the time, so it shares
        # the time's relative spread.
        ms = dict(zip(*times[variant]))
        bars = [y / 1000.0 * spread[variant][x] / ms[x] for x, y in zip(xs, ys)]
        ax.errorbar(xs, [y / 1000.0 for y in ys], yerr=bars, color=style["color"],
                    marker=style["marker"], markersize=5, linewidth=1.8, capsize=2,
                    label=style["label"])
    ax.set_xscale("log")
    ax.set_xlabel("number of particles $N$ (count)")
    ax.set_ylabel("sustained throughput (TFLOP/s, single precision)")
    ax.set_title("Sustained arithmetic throughput")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    add_legend(ax, "lower right")
    strip_spines(ax)
    caption(fig,
            "Twenty floating point operations per interaction, the conventional count. "
            "The rise with particle count is the GPU filling up.")
    save(fig, out_dir, "fig_nbody_throughput")


def plot_sph_scaling(rows, out_dir: Path) -> None:
    """Fluid step cost against particle count."""
    points = sorted(
        (int(r["num_particles"]), float(r["ms_per_step"]), float(r["ms_iqr"]) / 2.0)
        for r in rows
        if r["solver"] == "sph"
    )
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # One series, so the title names it and no legend box is needed.
    ax.errorbar(xs, ys, yerr=[p[2] for p in points], color=SERIES["naive"]["color"], marker="o",
                markersize=5, linewidth=1.8, capsize=2)
    ax.set_xlabel("number of particles $N$ (count)")
    ax.set_ylabel("time per step (milliseconds)")
    ax.set_title("Fluid step cost against particle count")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    ax.axhline(16.7, color=AXIS, linewidth=1.0, linestyle="--")
    ax.annotate("16.7 ms, sixty frames per second", xy=(xs[0], 16.7), xytext=(0, 6),
                textcoords="offset points", color=INK_MUTED, fontsize=8)
    ax.set_ylim(0, max(20.0, max(ys) * 1.3))
    strip_spines(ax)
    caption(fig,
            "The whole sweep sits far below the interactive budget, so the dashed line "
            "is a reference rather than a constraint. Bars span the interquartile range "
            "of the trials.")
    save(fig, out_dir, "fig_sph_scaling")


def plot_tile_sweep(path: Path, out_dir: Path) -> None:
    """Step cost against the width of the shared memory tile."""
    rows = read_csv(path)
    if not rows:
        return
    require_dispersion(path, rows)
    tiles = sorted({int(r["tile_size"]) for r in rows})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    width = 0.38
    positions = range(len(tiles))
    tallest = 0.0
    for offset, variant in ((-width / 2, "tiled"), (width / 2, "fused")):
        values = []
        bars = []
        for tile in tiles:
            match = [r for r in rows if int(r["tile_size"]) == tile and r["variant"] == variant]
            values.append(float(match[0]["ms_per_step"]) if match else 0.0)
            bars.append(float(match[0]["ms_iqr"]) / 2.0 if match else 0.0)
        tallest = max(tallest, max(values))
        if variant == "tiled":
            fastest_tile = tiles[values.index(min(values))]
        style = SERIES[variant]
        # A 2 point surface gap between adjacent bars, drawn as a white edge.
        ax.bar([p + offset for p in positions], values, width, yerr=bars,
               color=style["color"], label=style["label"], edgecolor="white", linewidth=2,
               error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 2})

    # Bars are read by area, so the baseline stays at zero even though it
    # compresses the differences; the exact values are in the table beside this
    # figure. The headroom keeps the legend clear of the tallest bar.
    ax.set_ylim(0.0, tallest * 1.22)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(t) for t in tiles])
    ax.set_xlabel("shared memory tile width (particles staged per tile)")
    ax.set_ylabel("time per step (milliseconds)")
    ax.set_title(f"Tile width sweep at {int(rows[0]['num_particles']):,} particles")
    add_legend(ax, "upper right")
    ax.grid(axis="x", visible=False)
    strip_spines(ax)
    caption(fig,
            "Lower is better. The leftmost pair is the width of 32 that v1.0.0 defaulted "
            f"to; the fastest tiled median is at width {fastest_tile}. Bars span the "
            "interquartile range of the trials. Both variants time a whole step.")
    save(fig, out_dir, "fig_tile_sweep")


def plot_energy_drift(path: Path, out_dir: Path) -> None:
    """Relative energy error of the two body orbit."""
    rows = read_csv(path)
    if not rows:
        return
    times = [float(r["time"]) for r in rows]
    drift = [float(r["relative_energy_drift"]) for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(times, drift, color=SERIES["naive"]["color"], linewidth=1.5)
    ax.axhline(1.0e-3, color="#d03b3b", linewidth=1.2, linestyle="--")
    ax.annotate("gate: 0.1 percent", xy=(times[0], 1.0e-3), xytext=(0, 6),
                textcoords="offset points", color="#d03b3b", fontsize=8, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel(CODE_TIME)
    ax.set_ylabel(r"relative energy drift $|E(t)-E_0|\,/\,|E_0|$ (dimensionless)")
    ax.set_title("Two body circular orbit: energy conservation over 10,000 steps")
    strip_spines(ax)
    caption(fig,
            "Time is in the code units of the run, where the gravitational constant, the "
            "total mass and the initial separation are all one, so one orbit is about 4.44 "
            "units. The error rises and then settles onto a bounded plateau instead of "
            "growing without limit, which is the property a symplectic integrator "
            "guarantees and a non symplectic one does not.")
    save(fig, out_dir, "fig_energy_drift")


def plot_momentum_drift(path: Path, out_dir: Path) -> None:
    """Relative momentum error of the Plummer sphere."""
    rows = read_csv(path)
    if not rows:
        return
    times = [float(r["time"]) for r in rows]
    drift = [float(r["relative_momentum_drift"]) for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(times, drift, color=SERIES["naive"]["color"], linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(CODE_TIME)
    ax.set_ylabel(r"relative momentum drift $|\Delta \mathbf{p}|\,/\,\sum_i m_i |\mathbf{v}_i|$"
                  "\n(dimensionless)")
    ax.set_title("Plummer sphere, 10,000 particles: momentum conservation")
    strip_spines(ax)
    caption(fig,
            "Drift is measured against the total absolute momentum in the system, because the "
            "vector total is zero by construction and dividing by it would be dividing by noise. "
            "The rise is single precision rounding accumulating in the force sum.")
    save(fig, out_dir, "fig_momentum_drift")


def plot_plummer_density(path: Path, out_dir: Path) -> None:
    """Sampled radial density against the analytic Plummer profile."""
    rows = read_csv(path)
    if not rows:
        return
    centres = [float(r["shell_centre"]) for r in rows]
    observed = [float(r["observed"]) for r in rows]
    expected = [float(r["expected"]) for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(centres, observed, color=SERIES["naive"]["color"], marker="o", markersize=4,
            linewidth=0, label="sampled")
    ax.plot(centres, expected, color=SERIES["tiled"]["color"], linewidth=1.8,
            label="Plummer profile")
    ax.set_xscale("log")
    ax.set_xlabel("shell radius $r/a$ (multiples of the scale radius, dimensionless)")
    ax.set_ylabel("particles in shell (count)")
    ax.set_title("Plummer sphere: sampled radial density against the analytic profile")
    add_legend(ax, "upper right")
    strip_spines(ax)
    caption(fig,
            "Shells hold equal expected counts, so every point carries the same weight in the "
            "chi squared statistic.")
    save(fig, out_dir, "fig_plummer_density")


def plot_sph_split(path: Path, out_dir: Path) -> None:
    """One stacked bar: where a fluid step goes, pass by pass."""
    rows = read_csv(path)
    if not rows:
        return
    by_pass = {row["pass"]: row for row in rows}
    unknown = by_pass.keys() - SPH_PASSES.keys()
    if unknown:
        raise SystemExit(f"error: {path} holds a pass this figure does not know: {sorted(unknown)}")

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    left = 0.0
    for name, (label, color) in SPH_PASSES.items():
        if name not in by_pass:
            continue
        value = float(by_pass[name]["ms_per_step"])
        percent = float(by_pass[name]["percent_of_step"])
        # A 2 point surface gap between segments, drawn as a white edge.
        ax.barh([0], [value], 0.55, left=left, color=color, edgecolor="white", linewidth=2,
                label=f"{label}, {percent:.1f}%")
        left += value

    ax.set_xlim(0.0, left)
    ax.set_yticks([])
    ax.set_xlabel("time per step (milliseconds)")
    ax.set_title(f"SPH step at {int(rows[0]['num_particles']):,} particles")
    ax.grid(axis="y", visible=False)
    strip_spines(ax)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=True,
                       facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE, fontsize=7.5)
    legend.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(bottom=0.34)
    caption(fig,
            f"Mean of {int(rows[0]['measured_steps']):,} steps after "
            f"{int(rows[0]['warmup_steps']):,} of warmup, from kernel records in Nsight Systems. "
            f"The traced step costs {float(rows[0]['traced_step_ms']):.3f} ms against "
            f"{float(rows[0]['untraced_step_ms']):.3f} ms with no tracer, so the between kernels "
            "share includes the tracer's own cost.")
    save(fig, out_dir, "fig_sph_split")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the report figures.")
    root = Path(__file__).resolve().parent.parent
    default_results = root / "experiments" / "results"
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument("--out", type=Path,
                        help="default report/figures, or figures inside a --results directory "
                             "other than experiments/results")
    parser.add_argument("--print-palette", action="store_true",
                        help="print the series colours for validate_palette.py, and draw nothing")
    parser.add_argument("--print-pass-palette", action="store_true",
                        help="print the SPH pass colours in stack order, and draw nothing")
    args = parser.parse_args()
    if args.print_palette:
        print(",".join(series["color"] for series in SERIES.values()))
        return 0
    if args.print_pass_palette:
        print(",".join(color for _, color in SPH_PASSES.values()))
        return 0
    if args.out is None:
        scratch = args.results.resolve() != default_results.resolve()
        args.out = args.results / "figures" if scratch else root / "report" / "figures"

    configure_style()
    print(f"reading results from {args.results}")
    print(f"writing figures to {args.out}")

    benchmarks = read_csv(args.results / "benchmarks.csv")
    if benchmarks:
        require_dispersion(args.results / "benchmarks.csv", benchmarks)
        plot_nbody_time(benchmarks, args.out)
        plot_nbody_speedup(benchmarks, args.out)
        plot_nbody_throughput(benchmarks, args.out)
        plot_sph_scaling(benchmarks, args.out)
    else:
        print("  no benchmarks.csv, skipping the performance figures")

    plot_tile_sweep(args.results / "tile_sweep.csv", args.out)
    plot_sph_split(args.results / "sph_pipeline_split.csv", args.out)
    plot_energy_drift(args.results / "energy_drift_two_body.csv", args.out)
    plot_momentum_drift(args.results / "momentum_drift_plummer.csv", args.out)
    plot_plummer_density(args.results / "plummer_radial_density.csv", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
