#!/usr/bin/env python3
"""Test python/cps_io on the checked in fixtures. Run by ctest as cps_io_tool."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "tests" / "data"
SERIES = DATA / "vtk_4p"
TOOL = ROOT / "python" / "cps_io" / "check_vtp.py"

sys.dont_write_bytecode = True
try:
    import numpy as np
except ImportError:
    print("NumPy is not installed, so the Python readers cannot run", file=sys.stderr)
    sys.exit(77)

sys.path.insert(0, str(ROOT / "python"))
from cps_io import FormatError, read_pvd, read_snapshot, read_vtp  # noqa: E402

FRAME = SERIES / "frame_000000010.vtp"


def bits(values: list[float]) -> np.ndarray:
    return np.array(values, dtype="<f4")


class Snapshots(unittest.TestCase):
    def test_cps1_gets_ids_in_file_order_and_no_clock(self) -> None:
        snap = read_snapshot(DATA / "snapshot_v1_4p.bin")
        self.assertEqual(snap.format_version, 1)
        self.assertEqual(snap.step, 42)
        self.assertTrue(np.isnan(snap.time))
        self.assertFalse(snap.id_in_file)
        self.assertEqual(snap.id.tolist(), [0, 1, 2, 3])
        np.testing.assert_array_equal(snap.x, bits([-0.5, 0.5, -0.5, 0.5]))

    def test_cps2_with_no_id_array_gets_ids_in_file_order(self) -> None:
        snap = read_snapshot(DATA / "snapshot_v2_4p.bin")
        self.assertEqual(snap.format_version, 2)
        self.assertFalse(snap.id_in_file)
        self.assertEqual(snap.id.tolist(), [0, 1, 2, 3])
        self.assertTrue(snap.half_step_velocity)
        self.assertIn("fingerprint_version", snap.physics)

    def test_cps2_with_an_id_array_keeps_it(self) -> None:
        snap = read_snapshot(DATA / "snapshot_v2_4p_id.bin")
        self.assertTrue(snap.id_in_file)
        self.assertEqual(snap.id.tolist(), [2, 0, 3, 1])
        self.assertEqual(snap.step, 42)
        self.assertEqual(snap.count, 4)
        # Negative zero survives, which a value comparison would not show.
        self.assertEqual(snap.vz[0].tobytes(), np.float32(-0.0).tobytes())

    def test_a_flipped_byte_fails_the_checksum(self) -> None:
        data = bytearray((DATA / "snapshot_v2_4p_id.bin").read_bytes())
        data[-1] ^= 1
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "bad.bin"
            path.write_bytes(data)
            with self.assertRaisesRegex(FormatError, "checksum"):
                read_snapshot(path)
            read_snapshot(path, verify_checksum=False)

    def test_bad_ids_are_refused(self) -> None:
        good = (DATA / "snapshot_v2_4p_id.bin").read_bytes()
        offset = good.index(np.array([2, 0, 3, 1], dtype="<u4").tobytes())
        for bad, message in (([0, 1, 2, 4], "not below"), ([1, 0, 1, 3], "more than once")):
            data = bytearray(good)
            data[offset:offset + 16] = np.array(bad, dtype="<u4").tobytes()
            with tempfile.TemporaryDirectory() as scratch:
                path = Path(scratch) / "bad.bin"
                path.write_bytes(data)
                with self.assertRaisesRegex(FormatError, message):
                    read_snapshot(path, verify_checksum=False)

    def test_something_else_is_refused(self) -> None:
        with self.assertRaisesRegex(FormatError, "not a snapshot"):
            read_snapshot(FRAME)


class Frames(unittest.TestCase):
    def test_every_field_of_the_fixture(self) -> None:
        frame = read_vtp(FRAME)
        self.assertEqual(frame.step, 10)
        self.assertEqual(frame.time, 0.5)
        self.assertEqual(frame.velocity_convention, "half_step")
        np.testing.assert_array_equal(frame.points[1], bits([0.5, -0.5, 0.25]))
        np.testing.assert_array_equal(frame.velocity[3], bits([2.0, -0.375, -0.0625]))
        np.testing.assert_array_equal(frame.mass, bits([0.25, 0.5, 1.0, 2.0]))
        self.assertEqual(frame.id.tolist(), [2, 0, 3, 1])
        self.assertEqual(frame.connectivity.tolist(), [0, 1, 2, 3])
        self.assertEqual(frame.offsets.tolist(), [1, 2, 3, 4])

    def test_the_series_index(self) -> None:
        entries = read_pvd(SERIES / "series.pvd")
        self.assertEqual([entry.time for entry in entries], [0.0, 0.5])
        self.assertEqual(entries[1].path, FRAME)

    def test_truncation_is_refused(self) -> None:
        data = FRAME.read_bytes()
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "cut.vtp"
            for length in (len(data) - 1, len(data) // 2, 40):
                path.write_bytes(data[:length])
                with self.assertRaises(FormatError):
                    read_vtp(path)


class CheckTool(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)
        for source in SERIES.iterdir():
            shutil.copy(source, self.directory / source.name)
        self.frame = self.directory / FRAME.name
        self.index = self.directory / "series.pvd"

    def run_tool(self, *paths: Path) -> subprocess.CompletedProcess:
        # -I, as the tool is documented to run, so nothing but its own path set up works.
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, paths)],
                              capture_output=True, text=True, check=False)

    def test_a_good_series_passes_and_says_what_it_checked(self) -> None:
        result = self.run_tool(self.index)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 frames of 4 particles", result.stdout)

    def test_frames_can_be_named_directly(self) -> None:
        self.assertEqual(self.run_tool(self.frame).returncode, 0)

    def test_a_truncated_frame_fails(self) -> None:
        self.frame.write_bytes(self.frame.read_bytes()[:-10])
        result = self.run_tool(self.index)
        self.assertEqual(result.returncode, 1)
        self.assertIn("truncated", result.stderr)
        self.assertIn(self.frame.name, result.stderr)

    def test_a_block_with_a_wrong_size_fails(self) -> None:
        data = bytearray(self.frame.read_bytes())
        at = data.index(b"<AppendedData") + data[data.index(b"<AppendedData"):].index(b"_") + 1
        data[at] += 1
        self.frame.write_bytes(data)
        self.assertIn("declares", self.run_tool(self.frame).stderr)

    def test_bad_xml_fails(self) -> None:
        data = self.frame.read_bytes().replace(b"<Points>", b"<Points", 1)
        self.frame.write_bytes(data)
        result = self.run_tool(self.frame)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not valid XML", result.stderr)

    def test_a_missing_frame_fails(self) -> None:
        self.frame.unlink()
        result = self.run_tool(self.index)
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not exist", result.stderr)

    def test_a_frame_with_a_repeated_id_fails(self) -> None:
        data = bytearray(self.frame.read_bytes())
        ids = np.array([2, 0, 3, 1], dtype="<u4").tobytes()
        at = data.index(ids)
        data[at:at + 16] = np.array([2, 0, 3, 3], dtype="<u4").tobytes()
        self.frame.write_bytes(data)
        result = self.run_tool(self.index)
        self.assertEqual(result.returncode, 1)
        self.assertIn("each id", result.stderr)

    def test_frames_of_different_sizes_fail(self) -> None:
        # A one point frame, written out by hand from the fixture's own layout.
        small = self.directory / "small.vtp"
        small.write_bytes(_one_point_frame())
        result = self.run_tool(self.frame, small)
        self.assertEqual(result.returncode, 1)
        self.assertIn("has 1 points", result.stderr)


def _one_point_frame() -> bytes:
    arrays = [
        np.zeros(3, "<f4").tobytes(), np.zeros(3, "<f4").tobytes(), np.ones(1, "<f4").tobytes(),
        np.zeros(1, "<u4").tobytes(), np.zeros(1, "<i8").tobytes(), np.ones(1, "<i8").tobytes(),
    ]
    template = (SERIES / "frame_000000000.vtp").read_bytes()
    header = template[:template.index(b"_", template.index(b"<AppendedData")) + 1].decode("ascii")
    header = header.replace('NumberOfPoints="4"', 'NumberOfPoints="1"')
    header = header.replace('NumberOfVerts="4"', 'NumberOfVerts="1"')
    offset = 0
    lines = []
    sizes = iter(arrays)
    for line in header.split("\n"):
        if 'format="appended"' in line:
            start = line.index('offset="') + len('offset="')
            line = f'{line[:start]}{offset}"/>'
            offset += 8 + len(next(sizes))
        lines.append(line)
    body = b"".join(len(array).to_bytes(8, "little") + array for array in arrays)
    return "\n".join(lines).encode("ascii") + body + b"\n  </AppendedData>\n</VTKFile>\n"


if __name__ == "__main__":
    unittest.main(verbosity=1)
