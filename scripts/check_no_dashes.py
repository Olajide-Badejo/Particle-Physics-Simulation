#!/usr/bin/env python3
"""Repository style gate: no em dashes and no en dashes anywhere.

The rule covers three surfaces:

1. Every text file in the repository is scanned for the literal characters
   U+2014 (em dash) and U+2013 (en dash).
2. LaTeX sources are additionally scanned for the ligatures ``--`` and ``---``,
   which TeX silently typesets as an en dash and an em dash. Verbatim-like
   environments and inline verbatim are excluded, because a command line flag
   such as the fast math switch legitimately contains two hyphens and TeX does
   not apply the ligature there.
3. Compiled PDFs are scanned through ``pdftotext`` when it is available, which
   is the only way to prove the rule survived the whole report pipeline.

Exit status is 0 when the repository is clean and 1 when any hit is found.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EM_DASH = "\u2014"
EN_DASH = "\u2013"
BAD_CHARS = {EM_DASH: "U+2014 em dash", EN_DASH: "U+2013 en dash"}

SKIP_DIRS = {
    ".git",
    "build",
    "_deps",
    "__pycache__",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}

# Extensions scanned as text. Anything else is treated as opaque.
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".bib",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    "",
}

TEXT_NAMES = {"Makefile", "LICENSE", "CMakeLists.txt", ".gitignore"}

# LaTeX regions where two consecutive hyphens are not a ligature.
VERBATIM_BLOCK = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted|Verbatim|semiverbatim)\}.*?"
    r"\\end\{(verbatim|lstlisting|minted|Verbatim|semiverbatim)\}",
    re.DOTALL,
)
INLINE_VERBATIM = re.compile(r"\\(verb|lstinline)(\*?)(.)(.*?)\3")
TEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
HYPHEN_LIGATURE = re.compile(r"-{2,}")


def is_text_file(path: Path) -> bool:
    if path.name in TEXT_NAMES:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def blank_out(match: re.Match[str]) -> str:
    """Replace a matched region with newline preserving filler."""
    return "".join("\n" if ch == "\n" else " " for ch in match.group(0))


def check_text(path: Path, rel: str, hits: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch, label in BAD_CHARS.items():
            col = line.find(ch)
            if col >= 0:
                hits.append(f"{rel}:{lineno}:{col + 1}: {label} in {line.strip()[:80]!r}")


def check_tex_ligatures(path: Path, rel: str, hits: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    text = VERBATIM_BLOCK.sub(blank_out, text)
    text = INLINE_VERBATIM.sub(blank_out, text)
    text = TEX_COMMENT.sub(blank_out, text)
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in HYPHEN_LIGATURE.finditer(line):
            run = match.group(0)
            label = "--- em dash ligature" if len(run) >= 3 else "-- en dash ligature"
            hits.append(f"{rel}:{lineno}:{match.start() + 1}: {label}")


def check_pdf(path: Path, rel: str, hits: list[str]) -> bool:
    """Return True when the PDF was actually inspected."""
    if shutil.which("pdftotext") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        try:
            subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(path), str(out)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return False
        text = out.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch, label in BAD_CHARS.items():
            if ch in line:
                hits.append(f"{rel} (page text line {lineno}): {label} in {line.strip()[:80]!r}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if any en dash or em dash is present.")
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="repository root to scan",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args()

    root = args.root.resolve()
    hits: list[str] = []
    scanned = 0
    pdfs_scanned = 0
    pdfs_skipped = 0

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            if check_pdf(path, rel, hits):
                pdfs_scanned += 1
            else:
                pdfs_skipped += 1
            continue
        if not is_text_file(path):
            continue
        scanned += 1
        check_text(path, rel, hits)
        if path.suffix.lower() in {".tex", ".bib"}:
            check_tex_ligatures(path, rel, hits)

    if hits:
        print(f"check_no_dashes: {len(hits)} violation(s) found")
        for hit in hits:
            print(f"  {hit}")
        return 1

    if not args.quiet:
        note = ""
        if pdfs_skipped:
            note = f", {pdfs_skipped} pdf(s) skipped because pdftotext is unavailable"
        print(
            f"check_no_dashes: clean across {scanned} text file(s) "
            f"and {pdfs_scanned} pdf(s){note}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
