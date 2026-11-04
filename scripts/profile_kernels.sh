#!/usr/bin/env bash
# Measure what the force kernels actually move through DRAM, and where their warps
# wait.
#
# The benchmark CSV carries a traffic figure derived from the algorithm, which
# is a model and could be wrong. This asks the hardware counters instead, and
# the two numbers are compared in the debug report. Nsight Compute replays each
# kernel several times to collect counters, so this is slow and is kept out of
# the normal build. It also serialises the launches, so nothing here sees what
# happens between kernels or how one launch differs from the next.
#
# Every variant is profiled at each particle count, at the default tile width and
# on the third launch of its kernel:
#
#   ncu_<variant>.csv          200,000 particles, the headline size
#   ncu_<variant>_<N>.csv      every other size
#   profile_runs.csv           one row per export: commit, dirty, particles, tile
#                              width and block size. The naive kernel has no tile,
#                              which is 0 here as it is in benchmarks.csv.
#   profile_summary.csv        one row per variant and size, one column per metric
#
# Nothing is written until every run has worked, and an export keeps only what
# Nsight Compute reports: its ==PROF== lines name the program by its full path.
#
# A machine with no Nsight Compute cannot produce these numbers, which is a
# failure unless the caller says otherwise with CPS_ALLOW_NO_NCU=1.
#
# PARTICLES and VARIANTS narrow a run to a scratch directory, for a repeat of one
# configuration.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USAGE="usage: profile_kernels.sh [--out DIR] [--allow-dirty]

  --out DIR      where the files go (default experiments/results)
  --allow-dirty  profile a tree with uncommitted code, stamping the rows dirty=1.
                 Refused for experiments/results."
# shellcheck source=scripts/profile_common.sh
source "$ROOT/scripts/profile_common.sh"
parse_common_args "$@"

HEADLINE=200000
NCU="${NCU:-ncu}"
if ! NCU_PATH="$(command -v "$NCU")"; then
  if [ "${CPS_ALLOW_NO_NCU:-0}" = "1" ]; then
    echo "nsight compute is not available as $NCU, skipping the profile (CPS_ALLOW_NO_NCU=1)" >&2
    exit 0
  fi
  echo "error: nsight compute is not available as $NCU, so nothing was profiled." >&2
  echo "Set NCU to its path, or CPS_ALLOW_NO_NCU=1 to skip the profile on purpose." >&2
  exit 1
fi

stamp_tree

if out_is_default && { [ -n "${PARTICLES:-}" ] || [ -n "${VARIANTS:-}" ]; }; then
  echo "error: PARTICLES and VARIANTS are for a scratch run, use --out" >&2
  exit 1
fi
PARTICLES="${PARTICLES:-200000 100000 10000}"
VARIANTS="${VARIANTS:-naive tiled fused}"

# The first group was chosen to answer one question: where does the naive kernel
# actually lose the time, given that all three variants issue identical arithmetic.
#
#   l1tex__throughput          the first level cache and load store path
#   lts__throughput            the second level cache
#   gpu__dram_throughput       device memory
#   ..._op_ld.sum              global load sectors, the traffic tiling removes
#   inst_executed_pipe_lsu     load store instructions issued
#   inst_executed_pipe_fma     arithmetic instructions, expected to be equal
#   issue_active               how often a warp had something to issue
#
# The second is the set from the V2 design, section 9.2, which says why a warp did
# not issue. Stall ratios are warps per cycle with an issue active, the
# _per_issue_active.ratio form. Percentages here are of _active, and the first
# group's are of _elapsed; summarise_profile.py names the two apart, because they
# have different denominators and do not belong in one column.
#
# Every name below was run against Nsight Compute 2026.2.1 on sm_120, and nothing
# in that set was substituted. The launch__ names are accepted and return values,
# but `ncu --query-metrics` does not list them. summarise_profile.py fails if any
# metric here comes back missing, so a name a later version drops is an error.
#
# The two hit rates are additions, for comparing kernels that differ in their L1
# traffic. So is smsp__issue_active over _active, which puts issue on the same
# denominator as the pipe percentages.
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
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__inst_executed.sum,\
smsp__inst_executed_pipe_xu.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__waves_per_multiprocessor

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# The tile width and block size are whatever the preset gives this build, read back
# rather than assumed.
"$BIN" run plummer_10k --print-config > config.json
TILE="$(python3 -c 'import json; print(json.load(open("config.json"))["nbody"]["tile_size"])')"
BLOCK="$(python3 -c 'import json; print(json.load(open("config.json"))["run"]["block_size"])')"
LAUNCH=3

FILES=()
echo "file,commit,dirty,variant,num_particles,tile_size,block_size,launch" > profile_runs.csv
for N in $PARTICLES; do
  for VARIANT in $VARIANTS; do
    case "$VARIANT" in
      naive) PATTERN="regex:compute_forces_naive" ; RUN_TILE=0 ;;
      tiled) PATTERN="regex:compute_forces_tiled" ; RUN_TILE="$TILE" ;;
      fused) PATTERN="regex:nbody_fused_step" ; RUN_TILE="$TILE" ;;
      *) echo "error: unknown variant $VARIANT" >&2; exit 2 ;;
    esac
    if [ "$N" = "$HEADLINE" ]; then
      FILE="ncu_${VARIANT}.csv"
    else
      FILE="ncu_${VARIANT}_${N}.csv"
    fi
    echo "profiling $VARIANT at $N particles"
    if ! "$NCU_PATH" --target-processes all \
                     --kernel-name "$PATTERN" \
                     --launch-skip $((LAUNCH - 1)) --launch-count 1 \
                     --metrics "$METRICS" \
                     --csv \
                     "$BIN" run plummer_10k --variant "$VARIANT" --particles "$N" \
                            --steps 6 --snapshot-every 0 --diag-every 0 --diagnostics "" \
                     > raw.csv 2> raw.log; then
      echo "error: nsight compute failed on $VARIANT at $N particles:" >&2
      tail -n 20 raw.csv raw.log >&2
      exit 1
    fi
    grep -v '^==PROF==' raw.csv > "$FILE" || true
    if ! grep -q '^"ID"' "$FILE"; then
      echo "error: no counters collected for $VARIANT at $N particles:" >&2
      tail -n 20 raw.csv raw.log >&2
      exit 1
    fi
    echo "$FILE,$COMMIT,$DIRTY,$VARIANT,$N,$RUN_TILE,$BLOCK,$LAUNCH" >> profile_runs.csv
    FILES+=("$FILE")
  done
done

# Fold the per kernel files into one tidy table the report can read.
python3 "$ROOT/scripts/summarise_profile.py" --results "$WORK"
FILES+=(profile_runs.csv profile_summary.csv)
check_no_paths "${FILES[@]}"
publish "$WORK" "${FILES[@]}"
