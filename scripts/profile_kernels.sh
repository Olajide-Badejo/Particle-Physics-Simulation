#!/usr/bin/env bash
# Measure what the force kernels actually move through DRAM.
#
# The benchmark CSV carries a traffic figure derived from the algorithm, which
# is a model and could be wrong. This asks the hardware counters instead, and
# the two numbers are compared in the debug report. Nsight Compute replays each
# kernel several times to collect counters, so this is slow and is kept out of
# the normal build.
#
# A machine with no Nsight Compute cannot produce these numbers, which is a
# failure unless the caller says otherwise with CPS_ALLOW_NO_NCU=1.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/bin/cuda_physics_sim"
CONFIG="$ROOT/presets/plummer_10k.json"
OUT="$ROOT/experiments/results"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
PARTICLES="${PARTICLES:-200000}"

if [ ! -x "$BIN" ]; then
  echo "build the project first: $BIN is missing" >&2
  exit 1
fi
if [ ! -x "$NCU" ]; then
  if [ "${CPS_ALLOW_NO_NCU:-0}" = "1" ]; then
    echo "nsight compute is not available at $NCU, skipping the profile (CPS_ALLOW_NO_NCU=1)" >&2
    exit 0
  fi
  echo "error: nsight compute is not available at $NCU, so nothing was profiled." >&2
  echo "Set NCU to its path, or CPS_ALLOW_NO_NCU=1 to skip the profile on purpose." >&2
  exit 1
fi

mkdir -p "$OUT"

# Chosen to answer one question: where does the naive kernel actually lose the
# time, given that all three variants issue identical arithmetic.
#
#   l1tex__throughput          the first level cache and load store path
#   lts__throughput            the second level cache
#   gpu__dram_throughput       device memory
#   ..._op_ld.sum              global load sectors, the traffic tiling removes
#   inst_executed_pipe_lsu     load store instructions issued
#   inst_executed_pipe_fma     arithmetic instructions, expected to be equal
#   issue_active               how often a warp had something to issue
METRICS=gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed_pipe_fma.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum

for VARIANT in naive tiled fused; do
  case "$VARIANT" in
    naive) PATTERN="regex:compute_forces_naive" ;;
    tiled) PATTERN="regex:compute_forces_tiled" ;;
    fused) PATTERN="regex:nbody_fused_step" ;;
  esac
  echo "profiling $VARIANT at $PARTICLES particles"
  if ! "$NCU" --target-processes all \
              --kernel-name "$PATTERN" \
              --launch-skip 2 --launch-count 1 \
              --metrics "$METRICS" \
              --csv \
              "$BIN" --config "$CONFIG" --variant "$VARIANT" --particles "$PARTICLES" \
                     --steps 6 --snapshot-every 0 --diag-every 0 --diagnostics "" \
              > "$OUT/ncu_${VARIANT}.csv" 2> "$OUT/ncu_${VARIANT}.log"; then
    echo "error: nsight compute failed on $VARIANT, see $OUT/ncu_${VARIANT}.log" >&2
    exit 1
  fi
  if [ ! -s "$OUT/ncu_${VARIANT}.csv" ]; then
    echo "error: no counters collected for $VARIANT, see $OUT/ncu_${VARIANT}.log" >&2
    exit 1
  fi
  echo "  wrote $OUT/ncu_${VARIANT}.csv"
done

# Fold the three per kernel files into one tidy table the report can read.
python3 "$ROOT/scripts/summarise_profile.py" --results "$OUT" --particles "$PARTICLES"
