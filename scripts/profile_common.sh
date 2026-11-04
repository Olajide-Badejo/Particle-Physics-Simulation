# Shared by profile_kernels.sh and profile_sph_split.sh, which source it.
#
# A profile is a result like any other, so it has to name the commit that made it.
# The binary states its commit through `cps info`, and that decides the stamp. A
# run is stamped dirty if the binary says so, if the binary was built from another
# commit, or if the code in the checkout has changed since. The list of paths that
# do not count is the one in cmake/write_version_header.cmake: results and the
# reports built from them are tracked, so without it one profile would block the
# next.
#
# Expects ROOT and USAGE to be set.

BIN="${CPS_BIN:-$ROOT/build/bin/cps}"
DEFAULT_OUT="$ROOT/experiments/results"
OUT="$DEFAULT_OUT"
ALLOW_DIRTY=0

parse_common_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --out)
        [ $# -ge 2 ] || { echo "error: --out needs a directory" >&2; exit 2; }
        OUT="$2"
        shift 2
        ;;
      --allow-dirty)
        ALLOW_DIRTY=1
        shift
        ;;
      -h|--help)
        echo "$USAGE"
        exit 0
        ;;
      *)
        echo "error: unknown argument $1" >&2
        echo "$USAGE" >&2
        exit 2
        ;;
    esac
  done
}

out_is_default() {
  [ -d "$OUT" ] && [ "$(cd "$OUT" && pwd -P)" = "$(cd "$DEFAULT_OUT" && pwd -P)" ]
}

# Sets COMMIT and DIRTY, or refuses the run.
stamp_tree() {
  if [ ! -x "$BIN" ]; then
    echo "error: build the project first, the program is missing: $BIN" >&2
    exit 1
  fi
  local stamp head
  stamp="$("$BIN" info | awk '$1 == "commit" { print $2; exit }')"
  COMMIT="${stamp%-dirty}"
  DIRTY=0
  [ "$COMMIT" != "$stamp" ] && DIRTY=1
  [ -z "$COMMIT" ] && COMMIT=unknown
  [ "$COMMIT" = "unknown" ] && DIRTY=1
  head="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo none)"
  [ "$COMMIT" != "$head" ] && DIRTY=1
  git -C "$ROOT" diff --quiet HEAD -- . \
      ":(exclude)experiments/results" \
      ":(exclude)report/figures" \
      ":(exclude)report/tables" \
      ":(exclude)report/numbers.tex" \
      ":(exclude)docs/generated_numbers.md" \
      ":(exclude)report/main.pdf" \
      ":(exclude)report_debug/debug_report.pdf" 2>/dev/null || DIRTY=1

  if [ "$DIRTY" = "1" ] && out_is_default; then
    echo "error: refusing to write dirty rows into experiments/results (binary $stamp, HEAD $head)." >&2
    echo "Commit the change and rebuild, or use --out with --allow-dirty for a scratch run." >&2
    exit 1
  fi
  if [ "$DIRTY" = "1" ] && [ "$ALLOW_DIRTY" != "1" ]; then
    echo "error: the tree is dirty or the binary is stale (binary $stamp, HEAD $head)." >&2
    echo "Commit the change and rebuild, or pass --allow-dirty to stamp the rows dirty=1." >&2
    exit 1
  fi
}

# Nothing goes out with a path from this machine in it. The scanner covers home
# directories and the like; the literal checks cover a checkout anywhere else.
check_no_paths() {
  python3 "$ROOT/scripts/check_no_local_paths.py" --files "$@" >&2 || {
    echo "error: a file about to be published names a local path, so nothing was written" >&2
    exit 1
  }
  local literal
  for literal in "$ROOT" "${HOME:-$ROOT}"; do
    if grep -lF -- "$literal" "$@" >&2; then
      echo "error: the files above name a local path, so nothing was written" >&2
      exit 1
    fi
  done
}

# Moves the finished files out of the run's scratch directory, all together.
publish() {
  local work="$1"
  shift
  mkdir -p "$OUT"
  local name
  for name in "$@"; do
    mv -f "$work/$name" "$OUT/$name"
    echo "  wrote $name"
  done
}
