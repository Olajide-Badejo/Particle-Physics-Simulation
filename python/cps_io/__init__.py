"""Read what the simulator writes: CPS1 and CPS2 snapshots, VTK frames and series.

NumPy is the only dependency. Each reader raises FormatError, naming the file, on
anything it cannot vouch for, and arrays come back with the bits that are on disk.
"""

from .errors import FormatError
from .snapshot import Snapshot, read_snapshot
from .vtp import SeriesEntry, VtpFrame, read_pvd, read_vtp

__all__ = [
    "FormatError",
    "SeriesEntry",
    "Snapshot",
    "VtpFrame",
    "read_pvd",
    "read_snapshot",
    "read_vtp",
]
