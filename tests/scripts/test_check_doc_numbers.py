#!/usr/bin/env python3
"""Test scripts/check_doc_numbers.py on scratch files. Run by ctest as doc_numbers_tool."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
TOOL = SCRIPTS / "check_doc_numbers.py"

sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))
from gen_report_assets import Assets  # noqa: E402

TABLE = """# Generated numbers

| Metric | Value | Source |
|---|---|---|
| Spread | 1.25 | experiments/results/benchmark_repeatability.csv |
| Device | a \\| b | experiments/results/device.json |
"""


class CheckDocNumbers(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.table = self.root / "numbers.md"
        self.table.write_text(TABLE, encoding="utf-8")
        self.doc = self.root / "doc.md"

    def run_tool(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), "--root", str(self.root), "--generated", str(self.table),
             *args],
            capture_output=True, text=True, check=False)

    def check(self, text: str, *args: str) -> subprocess.CompletedProcess:
        self.doc.write_text(text, encoding="utf-8")
        return self.run_tool(str(self.doc), *args)

    def test_match_passes(self) -> None:
        result = self.check("Drift is <!-- num:Spread -->1.25<!-- /num --> percent.\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 marked value", result.stdout)

    def test_escaped_pipe_in_the_table_is_read_as_a_pipe(self) -> None:
        result = self.check("<!-- num:Device -->a | b<!-- /num -->\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mismatch_names_file_line_and_metric(self) -> None:
        result = self.check("Title\n\nDrift <!-- num:Spread -->9.99<!-- /num -->.\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("doc.md:3: Spread", result.stderr)
        self.assertIn("9.99", result.stderr)
        self.assertIn("1.25", result.stderr)

    def test_unknown_metric_fails(self) -> None:
        result = self.check("<!-- num:Nonsense -->1<!-- /num -->\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("doc.md:1: Nonsense: unknown metric", result.stderr)

    def test_unclosed_mark_fails(self) -> None:
        result = self.check("<!-- num:Spread -->1.25\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("doc.md:1", result.stderr)

    def test_fix_rewrites_only_the_values(self) -> None:
        result = self.check("a <!-- num:Spread -->9<!-- /num --> b "
                            "<!-- num:Device -->x<!-- /num --> c\r\nnext\n", "--fix")
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.doc.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(handle.read(),
                             "a <!-- num:Spread -->1.25<!-- /num --> b "
                             "<!-- num:Device -->a | b<!-- /num --> c\r\nnext\n")
        self.assertEqual(self.run_tool(str(self.doc)).returncode, 0)

    def test_fix_still_fails_on_an_unknown_metric(self) -> None:
        text = "<!-- num:Nonsense -->1<!-- /num -->\n"
        result = self.check(text, "--fix")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.doc.read_text(encoding="utf-8"), text)

    def test_examples_in_code_are_not_read(self) -> None:
        result = self.check("Write `<!-- num:Nonsense -->1<!-- /num -->` like so.\n\n"
                            "```\n<!-- num:Nonsense -->1<!-- /num -->\n```\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 marked value", result.stdout)

    def test_a_real_metric_inside_a_code_span_fails(self) -> None:
        result = self.check("Title\n\nAt `<!-- num:Spread -->9.99<!-- /num -->` here.\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("doc.md:3: Spread", result.stderr)
        self.assertIn("code span", result.stderr)

    def test_no_marks_pass_without_a_table(self) -> None:
        self.table.unlink()
        self.assertEqual(self.check("Nothing marked here.\n").returncode, 0)

    def test_a_mark_without_a_table_fails(self) -> None:
        self.table.unlink()
        result = self.check("<!-- num:Spread -->1.25<!-- /num -->\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no generated table", result.stderr)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_default_scan_leaves_the_changelog_alone(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "CHANGELOG.md").write_text("<!-- num:Nonsense -->1<!-- /num -->\n",
                                                encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "CHANGELOG.md"], check=True)
        result = self.run_tool()
        self.assertNotIn("Nonsense", result.stderr)

    def test_default_scan_reads_only_tracked_markdown(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "tracked.md").write_text("<!-- num:Spread -->9<!-- /num -->\n",
                                              encoding="utf-8")
        (self.root / "untracked.md").write_text("<!-- num:Nonsense -->1<!-- /num -->\n",
                                                encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "tracked.md"], check=True)
        result = self.run_tool()
        self.assertEqual(result.returncode, 1)
        self.assertIn("tracked.md:1: Spread", result.stderr)
        self.assertNotIn("Nonsense", result.stderr)

    def test_table_written_by_the_generator_is_readable(self) -> None:
        assets = Assets(self.root, self.root / "assets", self.root / "gates.txt", True)
        assets.source = "benchmarks.csv,device.json"
        assets.macro("Drift", "$1.23 \\times 10^{-4}$")
        assets.macro("Commit", "abc\\_def")
        assets.macro("Pipes", "a|b")
        assets.macro("Empty", "")
        assets.write_numbers(self.table, self.root)
        result = self.check("<!-- num:Drift -->1.23e-4<!-- /num --> "
                            "<!-- num:Commit -->abc_def<!-- /num --> "
                            "<!-- num:Pipes -->a|b<!-- /num --> "
                            "<!-- num:Empty --><!-- /num -->\n")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
