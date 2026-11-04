#!/usr/bin/env python3
"""Test the profiler tooling on scratch files. Run by ctest as profile_tools."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
PATHS_TOOL = SCRIPTS / "check_no_local_paths.py"
SPLIT_TOOL = SCRIPTS / "summarise_sph_split.py"
PROVENANCE_TOOL = SCRIPTS / "check_provenance.py"

# Assembled here so the scanner does not flag this file when it scans the tree.
USERS = "Us" + "ers"
FAKE_PATHS = {
    "wsl mount": "/mnt/c/" + USERS + "/someone/project/build/bin/cps",
    "linux home": "/ho" + "me/someone/project/build/bin/cps",
    "windows backslash": "C:\\" + USERS + "\\someone\\project",
    "windows slash": "C:/" + USERS + "/someone/project",
}


def run(tool: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(tool), *args],
                          capture_output=True, text=True, check=False)


class Scratch(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)


class CheckNoLocalPaths(Scratch):
    def test_clean_tree_passes(self) -> None:
        (self.root / "clean.md").write_text("Installed under /usr/local/cuda/bin.\n", encoding="utf-8")
        result = run(PATHS_TOOL, str(self.root))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_each_shape_names_file_and_line(self) -> None:
        for label, path in FAKE_PATHS.items():
            with self.subTest(label):
                (self.root / "results.csv").write_text(f"first line\nConnected to {path}\n",
                                                       encoding="utf-8")
                result = run(PATHS_TOOL, str(self.root))
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("results.csv:2:", result.stdout)

    def test_a_placeholder_in_documentation_is_not_a_path(self) -> None:
        (self.root / "doc.md").write_text("Like /ho" + "me/<name>/project.\n", encoding="utf-8")
        self.assertEqual(run(PATHS_TOOL, str(self.root)).returncode, 0)

    def test_allow_list_leaves_a_file_out(self) -> None:
        (self.root / "example.md").write_text(FAKE_PATHS["linux home"] + "\n", encoding="utf-8")
        self.assertEqual(run(PATHS_TOOL, str(self.root)).returncode, 1)
        self.assertEqual(run(PATHS_TOOL, str(self.root), "--allow", "example.*").returncode, 0)

    def test_files_mode_scans_only_what_it_is_given(self) -> None:
        bad = self.root / "bad.csv"
        good = self.root / "good.csv"
        bad.write_text(FAKE_PATHS["wsl mount"] + "\n", encoding="utf-8")
        good.write_text("fine\n", encoding="utf-8")
        self.assertEqual(run(PATHS_TOOL, "--files", str(good)).returncode, 0)
        self.assertEqual(run(PATHS_TOOL, "--files", str(good), str(bad)).returncode, 1)

    def test_binary_file_is_not_read(self) -> None:
        (self.root / "data.bin").write_bytes(b"\0" + FAKE_PATHS["linux home"].encode())
        self.assertEqual(run(PATHS_TOOL, str(self.root)).returncode, 0)


SORT_KERNELS = ("void cub::_V_300304_SM_1200::detail::radix_sort::DeviceRadixSortHistogramKernel"
                "<cub::policy>(const unsigned int *)", "[CUDA memset]",
                "void cub::_V_300304_SM_1200::detail::radix_sort::DeviceRadixSortOnesweepKernel"
                "<cub::policy>(const unsigned int *)")
STEP = ("cps::<unnamed>::hash_particles_kernel(const float *)", *SORT_KERNELS,
        "cps::<unnamed>::clear_cell_bounds_kernel(unsigned int *)",
        "cps::<unnamed>::cell_bounds_kernel(const unsigned int *)",
        "cps::<unnamed>::gather_sorted_kernel(const float *)",
        "cps::<unnamed>::density_kernel(const float *)",
        "cps::<unnamed>::forces_and_integrate_kernel(const float *)")


class SphSplit(Scratch):
    def split(self, steps: list[tuple[str, ...]], warmup: int, measured: int
              ) -> subprocess.CompletedProcess:
        trace = self.root / "trace.csv"
        with trace.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Start (ns)", "Duration (ns)", "Name"])
            clock = 0
            for names in steps:
                for name in names:
                    writer.writerow([clock, 10_000, name])
                    clock += 20_000
        return run(SPLIT_TOOL, str(trace), "--out", str(self.root / "split.csv"),
                   "--commit", "abc", "--dirty", "0", "--particles", "100", "--block-size", "256",
                   "--warmup", str(warmup), "--measured", str(measured), "--untraced-ms", "0.1")

    def test_the_sort_gets_cub_and_its_memset_and_the_rest_is_between_kernels(self) -> None:
        result = self.split([STEP] * 4, 1, 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "split.csv").open(newline="", encoding="utf-8") as handle:
            rows = {row["pass"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), {"hash", "sort", "clear", "cell_bounds", "gather", "density",
                                     "force_and_integrate", "between_kernels"})
        self.assertAlmostEqual(float(rows["hash"]["ms_per_step"]), 0.01)
        self.assertAlmostEqual(float(rows["sort"]["ms_per_step"]), 0.03)
        self.assertEqual(float(rows["sort"]["launches_per_step"]), 3.0)
        self.assertAlmostEqual(float(rows["between_kernels"]["ms_per_step"]), 0.09)
        self.assertEqual(rows["hash"]["commit"], "abc")
        self.assertEqual(rows["hash"]["measured_steps"], "2")
        self.assertAlmostEqual(sum(float(r["percent_of_step"]) for r in rows.values()), 100.0,
                               delta=0.05)

    def test_warmup_is_left_out(self) -> None:
        slow = tuple(STEP) + ("cps::<unnamed>::something_only_at_startup(int)",)
        self.assertEqual(self.split([slow, STEP, STEP], 1, 1).returncode, 0)

    def test_an_unknown_kernel_is_an_error(self) -> None:
        odd = tuple(STEP) + ("cps::<unnamed>::a_new_pass_kernel(int)",)
        result = self.split([STEP, odd, STEP], 1, 1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("a_new_pass_kernel", result.stderr)
        self.assertFalse((self.root / "split.csv").exists())

    def test_a_missing_pass_is_an_error(self) -> None:
        without_density = tuple(name for name in STEP if "density" not in name)
        result = self.split([without_density] * 3, 1, 1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("density", result.stderr)

    def test_too_few_steps_is_an_error(self) -> None:
        self.assertEqual(self.split([STEP] * 3, 1, 5).returncode, 1)


class Provenance(Scratch):
    HEAD = subprocess.run(["git", "-C", str(SCRIPTS), "rev-parse", "--short=12", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()

    def setUp(self) -> None:
        super().setUp()
        (self.root / "gates.txt").write_text("a_gate\n", encoding="utf-8")
        (self.root / "validation_gates.csv").write_text("gate\na_gate\n", encoding="utf-8")

    def check(self) -> subprocess.CompletedProcess:
        return run(PROVENANCE_TOOL, "--results", str(self.root),
                   "--registry", str(self.root / "gates.txt"))

    def runs(self, dirty: str = "0") -> None:
        (self.root / "ncu_tiled.csv").write_text("x\n", encoding="utf-8")
        (self.root / "profile_runs.csv").write_text(
            "file,commit,dirty,variant,num_particles,tile_size,block_size,launch\n"
            f"ncu_tiled.csv,{self.HEAD},{dirty},tiled,10,256,256,3\n", encoding="utf-8")

    def test_an_old_profile_needs_no_stamp(self) -> None:
        if not self.HEAD:
            self.skipTest("not in a git checkout")
        (self.root / "ncu_tiled.csv").write_text("x\n", encoding="utf-8")
        (self.root / "profile_summary.csv").write_text("variant,duration_ns\ntiled,1\n",
                                                       encoding="utf-8")
        self.assertEqual(self.check().returncode, 0)

    def test_a_stamped_profile_passes(self) -> None:
        if not self.HEAD:
            self.skipTest("not in a git checkout")
        self.runs()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_dirty_profile_fails(self) -> None:
        if not self.HEAD:
            self.skipTest("not in a git checkout")
        self.runs("1")
        self.assertIn("dirty", self.check().stdout)
        self.assertEqual(self.check().returncode, 1)

    def test_an_export_the_runs_file_does_not_name_fails(self) -> None:
        if not self.HEAD:
            self.skipTest("not in a git checkout")
        self.runs()
        (self.root / "ncu_fused.csv").write_text("x\n", encoding="utf-8")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("ncu_fused.csv", result.stdout)

    def test_a_listed_export_that_is_gone_fails(self) -> None:
        if not self.HEAD:
            self.skipTest("not in a git checkout")
        self.runs()
        (self.root / "ncu_tiled.csv").unlink()
        self.assertEqual(self.check().returncode, 1)

    def test_stall_columns_need_the_runs_file_and_a_stamp(self) -> None:
        (self.root / "profile_summary.csv").write_text("variant,stall_wait\ntiled,1\n",
                                                       encoding="utf-8")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("profile_runs.csv", result.stdout)

    def test_the_split_needs_its_stamp(self) -> None:
        (self.root / "sph_pipeline_split.csv").write_text("pass,ms_per_step\nhash,1\n",
                                                          encoding="utf-8")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("commit", result.stdout)


if __name__ == "__main__":
    unittest.main()
