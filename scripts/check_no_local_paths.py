#!/usr/bin/env python3
"""Fail if a file names a place on somebody's own machine.

A profiler or a build tool will happily print the absolute path of whatever it
ran, and once that lands in a committed file it publishes the author's account
name and folder layout. This looks for four shapes of path: a home directory on
Linux, a Users folder under a Windows drive with either kind of slash, and that
same folder as WSL mounts it.

Every text file git knows about is scanned, whatever its extension, along with
any new file git does not ignore. Compiled PDFs are scanned through `pdftotext`.
As in the dash gate, a PDF that could not be read is an error rather than a skip,
so the check cannot pass without looking.

A legitimate example in the documentation goes on the allow-list: `--allow`,
which may be repeated, takes a glob for files to leave out, and ALLOWED below
holds the permanent ones. There are none today.

`--files` scans the named files instead of a tree, which is how the profiling
scripts check what they are about to publish. Each hit names the file and the
line. Exit status is 0 when nothing is found and 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOCAL_PATH = re.compile(
    r"/mnt/[A-Za-z]/Users/"
    r"|/home/[A-Za-z0-9._-]+/"
    r"|\b[A-Za-z]:[\\/]+Users[\\/]",
    re.IGNORECASE,
)

ALLOWED: tuple[str, ...] = ()


def known_files(root: Path) -> list[Path] | None:
    """Files git does not ignore, or None outside a work tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    names = result.stdout.decode("utf-8", errors="replace").split("\0")
    return sorted(root / name for name in names if name)


def scan_text(text: str, name: str, hits: list[str]) -> None:
    for number, line in enumerate(text.splitlines(), start=1):
        match = LOCAL_PATH.search(line)
        if match:
            hits.append(f"{name}:{number}: local path starting {match.group(0)!r}")


def scan_pdf(path: Path, name: str, hits: list[str]) -> bool:
    """Return True when the PDF was actually read."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "out.txt"
        try:
            subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), str(out)],
                           check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError):
            return False
        scan_text(out.read_text(encoding="utf-8", errors="replace"), f"{name} (page text)", hits)
    return True


def scan_file(path: Path, name: str, hits: list[str], unread_pdfs: list[str]) -> bool:
    """Scan one file. Returns False when it is binary and so was not looked at."""
    if path.suffix.lower() == ".pdf":
        if not scan_pdf(path, name, hits):
            unread_pdfs.append(name)
        return True
    data = path.read_bytes()
    if b"\0" in data:
        return False
    scan_text(data.decode("utf-8", errors="replace"), name, hits)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="repository to scan (default this one)")
    parser.add_argument("--allow", action="append", default=[], metavar="GLOB",
                        help="leave out files whose path from the root matches this")
    parser.add_argument("--files", nargs="+", type=Path, metavar="FILE",
                        help="scan these files instead of a tree")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.files:
        targets = [(path, path.name) for path in args.files]
    else:
        paths = known_files(root)
        if paths is None:
            paths = sorted(path for path in root.rglob("*")
                           if ".git" not in path.relative_to(root).parts)
        targets = [(path, path.relative_to(root).as_posix()) for path in paths]

    allowed = (*ALLOWED, *args.allow)
    hits: list[str] = []
    unread_pdfs: list[str] = []
    scanned = 0
    for path, name in targets:
        if not path.is_file() or any(fnmatch.fnmatch(name, glob) for glob in allowed):
            continue
        if scan_file(path, name, hits, unread_pdfs):
            scanned += 1

    for hit in hits:
        print(f"FAIL {hit}")
    if unread_pdfs:
        reason = "pdftotext is not on PATH" if shutil.which("pdftotext") is None else "pdftotext failed"
        print(f"FAIL {reason}, so {len(unread_pdfs)} pdf(s) could not be inspected: "
              f"{', '.join(unread_pdfs)}")
    if hits or unread_pdfs:
        print(f"check_no_local_paths: {len(hits)} local path(s) found")
        return 1
    print(f"check_no_local_paths: clean across {scanned} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
