#!/usr/bin/env python3
"""Check a categorical chart palette against the computable colour criteria.

The figures in this report are read by people, and roughly one man in twelve
has some form of red green colour vision deficiency. Whether two series can
actually be told apart is measurable, so it is measured here rather than judged
by eye.

Five checks are computed:

1. Lightness band: OKLCH L inside the band for the mode.
2. Chroma floor: OKLCH C at or above the floor, below which a hue reads gray.
3. Colour vision deficiency separation: OKLab delta E times 100 between slots
   under simulated protanopia and deuteranopia, using the Machado, Oliveira and
   Fernandes (2009) transforms at severity one.
4. Normal vision floor: the same distance without simulation, because readers
   with full colour vision have to tell the series apart too.
5. Contrast against the surface the chart is drawn on.

Exit status is 0 when nothing hard fails and 1 otherwise. A separation in the
warn band is legal only when the chart also carries a second channel, which in
this project means direct labels and distinct markers on every series.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

# Thresholds, matching the reference implementation.
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET = 8.0
CVD_FLOOR = 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def hex_to_srgb(value: str) -> tuple[float, float, float]:
    text = value.strip().lstrip("#")
    return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def linear(value: str) -> tuple[float, float, float]:
    return tuple(srgb_to_linear(c) for c in hex_to_srgb(value))


def relative_luminance(value: str) -> float:
    r, g, b = linear(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def oklab_from_linear(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(value: str) -> tuple[float, float]:
    lightness, a, b = oklab_from_linear(linear(value))
    return lightness, math.hypot(a, b)


def simulate(value: str, kind: str) -> tuple[float, float, float]:
    r, g, b = linear(value)
    matrix = MACHADO[kind]
    out = []
    for row in matrix:
        out.append(min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)))
    return tuple(out)


def delta_e(first: str, second: str, kind: str | None = None) -> float:
    a = oklab_from_linear(simulate(first, kind) if kind else linear(first))
    b = oklab_from_linear(simulate(second, kind) if kind else linear(second))
    return 100.0 * math.dist(a, b)


def validate(palette: list[str], mode: str, surface: str, pairs: str) -> bool:
    low, high = BAND[mode]
    ok = True
    print(f"\npalette ({mode}, surface {surface}, {pairs} pairs): {len(palette)} slots")

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not low <= oklch(c)[0] <= high]
    ok = ok and not offband
    print(f"  [{'PASS' if not offband else 'FAIL'}] lightness band        "
          f"{offband if offband else f'all inside L {low} to {high}'}")

    low_chroma = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok = ok and not low_chroma
    print(f"  [{'PASS' if not low_chroma else 'FAIL'}] chroma floor          "
          f"{low_chroma if low_chroma else f'all at or above {CHROMA_FLOOR}'}")

    count = len(palette)
    if pairs == "all":
        pairlist = list(itertools.combinations(range(count), 2))
    else:
        pairlist = [(i, i + 1) for i in range(count - 1)]

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            distance = delta_e(palette[i], palette[j], kind)
            if worst is None or distance < worst[0]:
                worst = (distance, kind, palette[i], palette[j])
    tritan = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99.0)

    worst_distance = worst[0] if worst else 99.0
    state = "PASS" if worst_distance >= CVD_TARGET else ("WARN" if worst_distance >= CVD_FLOOR else "FAIL")
    ok = ok and state != "FAIL"
    print(f"  [{state}] cvd separation        worst {worst[2]} against {worst[3]} "
          f"delta E {worst_distance:.1f} ({worst[1]}), tritan {tritan:.1f}")

    normal_worst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j])
                        for i, j in pairlist), default=(99.0, "", ""))
    normal_state = "PASS" if normal_worst[0] >= NORMAL_FLOOR else "FAIL"
    ok = ok and normal_state == "PASS"
    print(f"  [{normal_state}] normal vision floor   worst {normal_worst[1]} against "
          f"{normal_worst[2]} delta E {normal_worst[0]:.1f}")

    poor = [(c, round(contrast(c, surface), 2)) for c in palette
            if contrast(c, surface) < CONTRAST_MIN]
    print(f"  [{'PASS' if not poor else 'WARN'}] contrast vs surface   "
          f"{poor if poor else f'all at or above {CONTRAST_MIN} to 1'}")
    if poor:
        print("         relief required: every series also carries a direct label "
              "and its own marker shape")

    print(f"\n  result: {'all checks pass' if ok else 'failed, fix the marked checks'}\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("palette", help="comma separated hex values")
    parser.add_argument("--mode", choices=("light", "dark"), default="light")
    parser.add_argument("--surface", default=None)
    parser.add_argument("--pairs", choices=("adjacent", "all"), default="adjacent")
    args = parser.parse_args()

    palette = [c.strip() for c in args.palette.split(",") if c.strip()]
    surface = args.surface or DEFAULT_SURFACE[args.mode]
    return 0 if validate(palette, args.mode, surface, args.pairs) else 1


if __name__ == "__main__":
    sys.exit(main())
