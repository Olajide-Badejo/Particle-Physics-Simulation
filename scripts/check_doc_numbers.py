#!/usr/bin/env python3
"""Check the numbers quoted in the Markdown files against the generated table.

A Markdown file marks a generated number with a pair of comments naming the
metric, which do not show when the page is rendered:

    <!-- num:RepeatabilitySpread -->1.2<!-- /num -->

Each mark must sit on one line, name a metric in `docs/generated_numbers.md`
(which `scripts/gen_report_assets.py` writes) and hold exactly its value. Marks
inside fenced code or an inline code span are examples and are not read.

Each failure names the file, the line and the metric. With `--fix` the marked
values are rewritten from the table instead; a mark that cannot be fixed still
fails. Exit status is 0 when every mark holds and 1 otherwise. A tree with no
marks passes whether or not the table exists yet.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MARK = re.compile(r"<!--\s*num:(\S*)\s*-->(.*?)<!--\s*/num\s*-->")
# What is left of a mark after the well formed ones are taken out.
STRAY_MARK = re.compile(r"<!--\s*/?num\b")
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)")
TABLE_ROW = re.compile(r"\| ([A-Za-z]+) \| (.*) \| [^|]* \|")


def read_table(path: Path) -> dict[str, str]:
    """Metric name to value, from the table gen_report_assets.py writes."""
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW.fullmatch(line.strip())
        if match:
            table[match.group(1)] = match.group(2).strip().replace(r"\|", "|")
    return table


def tracked_markdown(root: Path) -> list[Path]:
    # The changelog records what each release measured, so a later measurement
    # must not rewrite it. It takes no marks and is not scanned.
    names = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.md",
                            ":(exclude)CHANGELOG.md"],
                           check=True, capture_output=True).stdout.decode("utf-8").split("\0")
    return [root / name for name in names if name and (root / name).is_file()]


def find_marks(line: str) -> tuple[list[re.Match], str]:
    """The marks on a line outside code spans, and the line without them or the code."""
    spans = [span.span() for span in CODE_SPAN.finditer(line)]
    marks = [mark for mark in MARK.finditer(line)
             if not any(start <= mark.start() < end for start, end in spans)]
    rest = line
    # Code inside a mark is part of its value, so it goes along with the mark.
    for start, end in sorted(spans + [mark.span() for mark in marks], reverse=True):
        rest = rest[:start] + rest[end:]
    return marks, rest


def process(text: str, table: dict[str, str] | None, shown: str, fix: bool,
            failures: list[str]) -> tuple[str, int]:
    """Check one file's text. Returns the text, with any fixes applied, and its mark count."""
    lines = []
    count = 0
    fence = None
    for number, line in enumerate(text.split("\n"), start=1):
        opening = FENCE.match(line)
        if opening:
            run = opening.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
        if opening or fence is not None:
            lines.append(line)
            continue
        marks, rest = find_marks(line)
        if STRAY_MARK.search(rest):
            failures.append(f"{shown}:{number}: a num mark is not closed on the same line")
        count += len(marks)
        # Backwards, so that rewriting one value leaves the earlier offsets valid.
        for mark in reversed(marks):
            metric, value = mark.group(1), mark.group(2)
            if table is None:
                failures.append(f"{shown}:{number}: {metric}: there is no generated table to "
                                "compare with; run scripts/gen_report_assets.py")
            elif metric not in table:
                failures.append(f"{shown}:{number}: {metric}: unknown metric")
            elif value != table[metric]:
                if fix:
                    start, end = mark.span(2)
                    line = line[:start] + table[metric] + line[end:]
                else:
                    failures.append(f"{shown}:{number}: {metric}: the document says {value!r} "
                                    f"and the generated table says {table[metric]!r}")
        lines.append(line)
    return "\n".join(lines), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", type=Path, nargs="*",
                        help="the files to check (default the tracked Markdown files)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="the repository to scan, and whose table to read by default")
    parser.add_argument("--generated", type=Path,
                        help="the generated table (default docs/generated_numbers.md)")
    parser.add_argument("--fix", action="store_true", help="rewrite the marked values")
    args = parser.parse_args()

    root = args.root.resolve()
    generated = args.generated or root / "docs" / "generated_numbers.md"
    table = read_table(generated) if generated.exists() else None
    files = args.files or tracked_markdown(root)

    failures: list[str] = []
    total = 0
    for path in files:
        if path.resolve() == generated.resolve():
            continue
        # Without newline translation, so that a fix changes only the values.
        with path.open(newline="", encoding="utf-8") as handle:
            text = handle.read()
        resolved = path.resolve()
        shown = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(path)
        fixed, count = process(text, table, shown, args.fix, failures)
        total += count
        if fixed != text:
            with path.open("w", newline="", encoding="utf-8") as handle:
                handle.write(fixed)
            print(f"  fixed {shown}")

    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"{total} marked value(s) in {len(files)} file(s), {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
