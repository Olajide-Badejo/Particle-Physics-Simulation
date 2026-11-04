"""Read CPS2 and CPS1 snapshots. include/cps/core/snapshot.hpp defines the layout."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import FormatError

_MAGIC_V2 = b"CPS2"
_MAGIC_V1 = b"CPS1"
_HEADER_V2 = struct.Struct("<4sIII QdQQQ II")
_ENTRY = struct.Struct("<16sII QQ")
_HEADER_V1 = struct.Struct("<4sIII")
_FLOAT_ARRAYS = ("x", "y", "z", "vx", "vy", "vz", "mass")
_DTYPES = {
    1: "<u1", 2: "<u2", 3: "<u4", 4: "<u8",
    5: "<i1", 6: "<i2", 7: "<i4", 8: "<i8",
    9: "<f4", 10: "<f8",
}
_FNV_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = (1 << 64) - 1


@dataclass
class Snapshot:
    """One snapshot. ``arrays`` holds every array in the file by name, plus ``id``.

    ``id_in_file`` is False for a CPS1 file or a CPS2 file with no id array. ``id`` is
    then 0 to count - 1 in file order, as the simulator assumes when it loads one.
    A CPS1 file has no clock or physics record: ``time`` is NaN and the rest are empty.
    """

    format_version: int
    step: int
    time: float
    flags: int
    count: int
    fingerprint: int = 0
    fingerprint_version: int = 0
    physics: str = ""
    id_in_file: bool = False
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def half_step_velocity(self) -> bool:
        return bool(self.flags & 1)

    def __getattr__(self, name: str) -> np.ndarray:
        # Only reached for what is not a field, so snap.x works as well as snap.arrays["x"].
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _fnv1a_64(data: bytes) -> int:
    value = _FNV_BASIS
    for byte in data:
        value = ((value ^ byte) * _FNV_PRIME) & _FNV_MASK
    return value


def _check_id(ids: np.ndarray, where: str) -> None:
    count = ids.size
    if count == 0:
        return
    if int(ids.max()) >= count:
        raise FormatError(f"{where}: array 'id' holds {int(ids.max())}, which is not below the "
                          f"particle count {count}")
    if np.unique(ids).size != count:
        raise FormatError(f"{where}: array 'id' holds a value more than once")


def _read_v1(data: bytes, where: str) -> Snapshot:
    if len(data) < _HEADER_V1.size:
        raise FormatError(f"{where} is truncated: {len(data)} bytes is short of the CPS1 header")
    _, count, step, flags = _HEADER_V1.unpack_from(data)
    expected = _HEADER_V1.size + len(_FLOAT_ARRAYS) * 4 * count
    if len(data) != expected:
        raise FormatError(f"{where} has a size mismatch: its CPS1 header claims {count} particles, "
                          f"which needs {expected} bytes, but the file has {len(data)}")
    snapshot = Snapshot(1, step, math.nan, flags, count)
    for i, name in enumerate(_FLOAT_ARRAYS):
        snapshot.arrays[name] = np.frombuffer(data, "<f4", count, _HEADER_V1.size + i * 4 * count)
    snapshot.arrays["id"] = np.arange(count, dtype="<u4")
    return snapshot


def _read_v2(data: bytes, where: str, verify_checksum: bool) -> Snapshot:
    if len(data) < _HEADER_V2.size:
        raise FormatError(f"{where} is truncated: {len(data)} bytes is short of the CPS2 header")
    (_, version, flags, array_count, step, time, count, fingerprint, checksum,
     fingerprint_version, _reserved) = _HEADER_V2.unpack_from(data)
    if version != 2:
        raise FormatError(f"{where} has format version {version}, not CPS2 version 2")

    table_end = _HEADER_V2.size + array_count * _ENTRY.size
    if table_end > len(data):
        raise FormatError(f"{where} is truncated: its array table needs {table_end} bytes, but "
                          f"the file has {len(data)}")
    if verify_checksum and _fnv1a_64(data[table_end:]) != checksum:
        raise FormatError(f"{where} is corrupt: its payload checksum does not match the header")

    snapshot = Snapshot(2, step, time, flags, count, fingerprint, fingerprint_version)
    for i in range(array_count):
        raw_name, dtype, _, elements, offset = _ENTRY.unpack_from(
            data, _HEADER_V2.size + i * _ENTRY.size)
        name = raw_name.split(b"\0", 1)[0].decode("ascii", "replace")
        if dtype not in _DTYPES:
            raise FormatError(f"{where}: array '{name}' has unknown dtype code {dtype}")
        item = np.dtype(_DTYPES[dtype]).itemsize
        if offset < table_end or offset > len(data) or elements > (len(data) - offset) // item:
            raise FormatError(f"{where}: array '{name}' lies outside the payload")
        array = np.frombuffer(data, _DTYPES[dtype], elements, offset)
        if name == "physics":
            snapshot.physics = array.tobytes().decode("utf-8")
        else:
            snapshot.arrays[name] = array

    for name in _FLOAT_ARRAYS:
        array = snapshot.arrays.get(name)
        if array is None:
            raise FormatError(f"{where} is missing the required array '{name}'")
        if array.dtype != np.dtype("<f4") or array.size != count:
            raise FormatError(f"{where}: array '{name}' must be f32 with {count} elements")

    ids = snapshot.arrays.get("id")
    if ids is None:
        snapshot.arrays["id"] = np.arange(count, dtype="<u4")
    else:
        if ids.dtype != np.dtype("<u4") or ids.size != count:
            raise FormatError(f"{where}: array 'id' must be u32 with {count} elements")
        _check_id(ids, where)
        snapshot.id_in_file = True
    return snapshot


def read_snapshot(path: str | Path, verify_checksum: bool = True) -> Snapshot:
    """Reads a snapshot of either version. The checksum walk is pure Python and costs
    about a second per ten megabytes, so a caller that has read the file before may
    turn it off."""
    where = f"snapshot {path}"
    data = Path(path).read_bytes()
    magic = data[:4]
    if magic == _MAGIC_V2:
        return _read_v2(data, where, verify_checksum)
    if magic == _MAGIC_V1:
        return _read_v1(data, where)
    raise FormatError(f"{path} is not a snapshot of this simulation: magic {magic!r}")
