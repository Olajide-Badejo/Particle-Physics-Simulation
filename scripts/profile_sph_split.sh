#!/usr/bin/env bash
# Where the time in one SPH step goes, pass by pass, at 100,000 particles.
#
# Nsight Systems records every launch without serialising them, so the kernels run
# here as they do in a real run. Its kernel records are the source; the NVTX ranges
# would only say how long the host took to launch each one, because the stream is
# non-blocking. Under WSL2 this was confirmed to trace CUDA kernels with Nsight
# Systems 2026.1.3 at kernel paranoid level 2. CPU sampling is off, because it is
# neither needed nor available at that level.
#
# The run takes WARMUP steps, which are discarded, then MEASURED steps, one more to
# close the last of them. The first steps cost more than the rest while the runtime
# settles and the lattice first gets sorted. The fluid has only just started to move
# by the end of this window, so this is the split near the start of a dam break, and
# a later stage of the flow, with more neighbours per particle, may differ. A second
# run of the same length with no tracer gives the step cost the trace can be judged
# against, since tracing adds to the launches.
#
# Both runs have graphs off. The split is per kernel and per launch, and a captured
# step is one launch, so this measures the stream path and says nothing about the
# time between kernels in a graph.
#
# The summary fails, rather than writing a partial table, if any pass is absent or
# if the trace holds a kernel it cannot attribute. See summarise_sph_split.py.
#
# A machine with no Nsight Systems cannot produce this, which is a failure unless
# the caller says otherwise with CPS_ALLOW_NO_NSYS=1.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USAGE="usage: profile_sph_split.sh [--out DIR] [--allow-dirty]

  --out DIR      where sph_pipeline_split.csv goes (default experiments/results)
  --allow-dirty  profile a tree with uncommitted code, stamping the rows dirty=1.
                 Refused for experiments/results."
# shellcheck source=scripts/profile_common.sh
source "$ROOT/scripts/profile_common.sh"
parse_common_args "$@"

NSYS="${NSYS:-nsys}"
if ! NSYS_PATH="$(command -v "$NSYS")"; then
  if [ "${CPS_ALLOW_NO_NSYS:-0}" = "1" ]; then
    echo "nsight systems is not available as $NSYS, skipping the split (CPS_ALLOW_NO_NSYS=1)" >&2
    exit 0
  fi
  echo "error: nsight systems is not available as $NSYS, so nothing was profiled." >&2
  echo "Set NSYS to its path, or CPS_ALLOW_NO_NSYS=1 to skip the split on purpose." >&2
  exit 1
fi

stamp_tree

PARTICLES=100000
WARMUP=200
MEASURED=1000
STEPS=$((WARMUP + MEASURED + 1))
RUN=("$BIN" run dam_break_50k --particles "$PARTICLES" --steps "$STEPS"
     --snapshot-every 0 --diag-every 0 --diagnostics "" --vtk-every 0 --graphs off)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

"$BIN" run dam_break_50k --print-config > config.json
BLOCK="$(python3 -c 'import json; print(json.load(open("config.json"))["run"]["block_size"])')"

echo "tracing the SPH step at $PARTICLES particles, $MEASURED steps after $WARMUP of warmup"
if ! "$NSYS_PATH" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
                  --stats=false --force-overwrite=true -o trace "${RUN[@]}" > nsys.log 2>&1; then
  echo "error: nsight systems failed:" >&2
  tail -n 20 nsys.log >&2
  exit 1
fi
if ! "$NSYS_PATH" stats --force-export=true --report cuda_gpu_trace --format csv \
                  --output trace trace.nsys-rep > stats.log 2>&1; then
  echo "error: nsight systems could not export the trace:" >&2
  tail -n 20 stats.log >&2
  exit 1
fi
if [ ! -s trace_cuda_gpu_trace.csv ]; then
  echo "error: the trace holds no CUDA activity, so kernels are not being recorded here." >&2
  tail -n 20 nsys.log stats.log >&2
  exit 1
fi

echo "timing the same run with no tracer"
"${RUN[@]}" > untraced.log 2> /dev/null
UNTRACED="$(awk '/solver ms\/step:/ { print $3; exit }' untraced.log)"
if [ -z "$UNTRACED" ]; then
  echo "error: the run with no tracer reported no step time:" >&2
  tail -n 20 untraced.log >&2
  exit 1
fi

python3 "$ROOT/scripts/summarise_sph_split.py" trace_cuda_gpu_trace.csv \
  --out sph_pipeline_split.csv --commit "$COMMIT" --dirty "$DIRTY" \
  --particles "$PARTICLES" --block-size "$BLOCK" --warmup "$WARMUP" --measured "$MEASURED" \
  --untraced-ms "$UNTRACED"
check_no_paths sph_pipeline_split.csv
publish "$WORK" sph_pipeline_split.csv
