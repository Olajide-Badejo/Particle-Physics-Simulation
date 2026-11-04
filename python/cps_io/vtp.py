"""Read the VTK frames and series the simulator writes. include/cps/io/vtk.hpp has the layout.

This understands the one layout the writer produces, and checks all of it, rather than
being a general VTK reader. For anything else there is VTK itself.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import FormatError

_APPENDED_OPEN = b'<AppendedData encoding="raw">'
_APPENDED_TAIL = b"\n  </AppendedData>\n</VTKFile>\n"
_BLOCK_HEADER_BYTES = 8
_TYPES = {"Float32": "<f4", "UInt32": "<u4", "Int64": "<i8"}


@dataclass
class VtpFrame:
    """One frame. Points are in the state's slot order; ``id`` follows a particle."""

    path: Path
    time: float
    step: int
    velocity_convention: str
    points: np.ndarray  # (N, 3)
    velocity: np.ndarray  # (N, 3)
    mass: np.ndarray
    id: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray


@dataclass
class SeriesEntry:
    time: float
    file: str
    path: Path


def _split(data: bytes, where: str) -> tuple[ET.Element, int]:
    """The parsed XML, with the appended block replaced by nothing, and where the block
    starts. Refuses a file whose tail is not exactly what the writer ends with, which is
    what catches a truncated one before any offset is believed."""
    opening = data.find(_APPENDED_OPEN)
    if opening < 0:
        raise FormatError(f"{where} has no raw appended data block")
    underscore = data.find(b"_", opening + len(_APPENDED_OPEN))
    if underscore < 0 or data[opening + len(_APPENDED_OPEN):underscore].strip():
        raise FormatError(f"{where} has no underscore at the start of its appended data")
    if not data.endswith(_APPENDED_TAIL):
        raise FormatError(f"{where} is truncated: it does not end with the closing tags")
    try:
        root = ET.fromstring(data[:opening] + b"<AppendedData/></VTKFile>")
    except ET.ParseError as error:
        raise FormatError(f"{where} is not valid XML: {error}") from error
    return root, underscore + 1


def _field_data(root: ET.Element, where: str) -> tuple[float, int, str]:
    time = step = convention = None
    for element in root.findall("./PolyData/FieldData/*"):
        name = element.get("Name")
        values = (element.text or "").split()
        try:
            if name == "TimeValue":
                time = float(values[0])
            elif name == "Step":
                step = int(values[0])
            elif name == "VelocityConvention":
                codes = [int(value) for value in values]
                if not codes or codes[-1] != 0:
                    raise ValueError("the text is not NUL terminated")
                convention = bytes(codes[:-1]).decode("ascii")
        except (IndexError, ValueError) as error:
            raise FormatError(f"{where}: field data '{name}' cannot be read: {error}") from error
    if time is None or step is None or convention is None:
        raise FormatError(f"{where} lacks TimeValue, Step or VelocityConvention in its field data")
    return time, step, convention


def read_vtp(path: str | Path) -> VtpFrame:
    """Reads a frame, checking that the arrays exactly fill the appended block."""
    path = Path(path)
    where = f"VTK frame {path}"
    data = path.read_bytes()
    root, base = _split(data, where)

    if root.tag != "VTKFile" or root.get("type") != "PolyData":
        raise FormatError(f"{where} is not VTK PolyData")
    if root.get("byte_order") != "LittleEndian" or root.get("header_type") != "UInt64":
        raise FormatError(f"{where} is not little endian with UInt64 block headers")
    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise FormatError(f"{where} has no Piece")
    try:
        count = int(piece.get("NumberOfPoints", ""))
        verts = int(piece.get("NumberOfVerts", ""))
    except ValueError as error:
        raise FormatError(f"{where} has a bad point or vertex count: {error}") from error
    if count != verts:
        raise FormatError(f"{where} has {count} points but {verts} vertices")

    block_end = len(data) - len(_APPENDED_TAIL)
    expected_offset = 0

    def array(element: ET.Element | None, label: str, dtype: str, components: int) -> np.ndarray:
        nonlocal expected_offset
        if element is None:
            raise FormatError(f"{where} has no array '{label}'")
        if _TYPES.get(element.get("type", "")) != dtype:
            raise FormatError(f"{where}: array '{label}' has type {element.get('type')}")
        if int(element.get("NumberOfComponents", "1")) != components:
            raise FormatError(f"{where}: array '{label}' has the wrong number of components")
        if element.get("format") != "appended":
            raise FormatError(f"{where}: array '{label}' is not appended")
        offset = int(element.get("offset", "-1"))
        if offset != expected_offset:
            raise FormatError(f"{where}: array '{label}' is at offset {offset}, expected "
                              f"{expected_offset}")
        start = base + offset
        size = count * components * np.dtype(dtype).itemsize
        if start + _BLOCK_HEADER_BYTES + size > block_end:
            raise FormatError(f"{where} is truncated: array '{label}' runs past the appended block")
        declared = int.from_bytes(data[start:start + _BLOCK_HEADER_BYTES], "little")
        if declared != size:
            raise FormatError(f"{where}: array '{label}' declares {declared} bytes but "
                              f"{count} points need {size}")
        expected_offset += _BLOCK_HEADER_BYTES + size
        values = np.frombuffer(data, dtype, count * components, start + _BLOCK_HEADER_BYTES)
        return values.reshape(count, components) if components > 1 else values

    def named(section: str, name: str) -> ET.Element | None:
        return piece.find(f"./{section}/DataArray[@Name='{name}']")

    points = array(piece.find("./Points/DataArray"), "Points", "<f4", 3)
    velocity = array(named("PointData", "velocity"), "velocity", "<f4", 3)
    mass = array(named("PointData", "mass"), "mass", "<f4", 1)
    ids = array(named("PointData", "id"), "id", "<u4", 1)
    connectivity = array(named("Verts", "connectivity"), "connectivity", "<i8", 1)
    offsets = array(named("Verts", "offsets"), "offsets", "<i8", 1)
    if base + expected_offset != block_end:
        raise FormatError(f"{where}: {block_end - base - expected_offset} bytes in the appended "
                          f"block belong to no array")
    if not np.array_equal(connectivity, np.arange(count)):
        raise FormatError(f"{where}: connectivity is not one vertex per point")
    if not np.array_equal(offsets, np.arange(1, count + 1)):
        raise FormatError(f"{where}: offsets are not one vertex per point")

    time, step, convention = _field_data(root, where)
    return VtpFrame(path, time, step, convention, points, velocity, mass, ids, connectivity, offsets)


def read_pvd(path: str | Path) -> list[SeriesEntry]:
    """Reads a series index. Frame paths come back resolved against its directory."""
    path = Path(path)
    where = f"VTK series {path}"
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as error:
        raise FormatError(f"{where} is not valid XML: {error}") from error
    if root.tag != "VTKFile" or root.get("type") != "Collection":
        raise FormatError(f"{where} is not a VTK collection")
    entries = []
    for dataset in root.findall("./Collection/DataSet"):
        file = dataset.get("file")
        try:
            time = float(dataset.get("timestep", ""))
        except ValueError as error:
            raise FormatError(f"{where}: a data set has a bad timestep: {error}") from error
        if not file:
            raise FormatError(f"{where}: a data set names no file")
        entries.append(SeriesEntry(time, file, path.parent / file))
    return entries
