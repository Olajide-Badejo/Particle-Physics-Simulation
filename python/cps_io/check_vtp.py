#!/usr/bin/env python3
"""Check VTK frames and series: XML, array sizes against the appended block, and ids.

usage: python3 -I python/cps_io/check_vtp.py <frame.vtp or series.pvd> ...

A .pvd stands for its frames, whose times must rise and match the index. Every frame
named must read cleanly and hold a permutation of 0 to N - 1, and all of them the same
N, so a particle can be followed from the first frame to the last. Exits 0 if all is
well and 1, naming the file, if not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# -I leaves the script's own directory off the path, which also keeps anything else
# in this directory from being imported by accident.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cps_io import FormatError, read_pvd, read_vtp  # noqa: E402


def check_frames(paths: list[Path]) -> int:
    """Raises FormatError on the first problem. Returns the particle count."""
    count = None
    for path in paths:
        frame = read_vtp(path)
        if count is None:
            count = frame.id.size
        if frame.id.size != count:
            raise FormatError(f"VTK frame {path} has {frame.id.size} points but the first frame "
                              f"has {count}")
        if not np.array_equal(np.sort(frame.id), np.arange(count)):
            raise FormatError(f"VTK frame {path} does not hold each id from 0 to {count - 1} once")
    return count or 0


def check_series(path: Path) -> list[Path]:
    """Checks the index and returns its frames."""
    entries = read_pvd(path)
    previous = None
    for entry in entries:
        if not entry.path.exists():
            raise FormatError(f"VTK series {path} names {entry.file}, which does not exist")
        if previous is not None and entry.time <= previous:
            raise FormatError(f"VTK series {path}: the time of {entry.file} does not rise")
        previous = entry.time
        frame = read_vtp(entry.path)
        if frame.time != entry.time:
            raise FormatError(f"VTK series {path} gives {entry.file} the time {entry.time}, but "
                              f"the frame says {frame.time}")
    return [entry.path for entry in entries]


def main(arguments: list[str]) -> int:
    if not arguments:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        frames: list[Path] = []
        for argument in arguments:
            path = Path(argument)
            frames.extend(check_series(path) if path.suffix == ".pvd" else [path])
        count = check_frames(frames)
    except (FormatError, OSError) as error:
        print(f"check_vtp: {error}", file=sys.stderr)
        return 1
    print(f"check_vtp: {len(frames)} frames of {count} particles are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
