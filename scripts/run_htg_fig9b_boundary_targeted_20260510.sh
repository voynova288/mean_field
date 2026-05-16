#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-results/HTG/htg_fig9b_d3_boundary_targeted_20260510_001}"
shift || true

PREFIX="fig9b_d3_boundary_targeted"
THREADS_PER_SHARD="${HTG_FIG9B_D3_BOUNDARY_THREADS_PER_SHARD:-4}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-64}"
MAX_SHARDS=$((SLURM_CPUS / THREADS_PER_SHARD))
if (( MAX_SHARDS < 1 )); then
  echo "Need at least ${THREADS_PER_SHARD} CPUs for one boundary shard, got ${SLURM_CPUS}" >&2
  exit 2
fi
if (( MAX_SHARDS > 16 )); then
  DEFAULT_SHARDS=16
else
  DEFAULT_SHARDS="${MAX_SHARDS}"
fi
N_SHARDS="${HTG_FIG9B_D3_BOUNDARY_SHARDS:-${DEFAULT_SHARDS}}"
SEEDS_PER_FLAVOR_CLASS="${HTG_FIG9B_D3_BOUNDARY_SEEDS_PER_FLAVOR_CLASS:-76}"
DRY_RUN=false
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done

if (( N_SHARDS < 1 )); then
  echo "N_SHARDS must be positive" >&2
  exit 2
fi
if (( SLURM_CPUS < N_SHARDS * THREADS_PER_SHARD )); then
  echo "Need at least $((N_SHARDS * THREADS_PER_SHARD)) CPUs, got ${SLURM_CPUS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/_shards"

run_shard() {
  local shard="$1"
  local cpu_set="$2"
  shift 2
  local shard_dir="${OUTPUT_DIR}/_shards/shard_${shard}"
  mkdir -p "${shard_dir}"
  echo "[boundary-packed] shard=${shard} cpu_set=${cpu_set} threads=${THREADS_PER_SHARD} output=${shard_dir}"
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "${cpu_set}" env \
      OMP_NUM_THREADS="${THREADS_PER_SHARD}" \
      OPENBLAS_NUM_THREADS="${THREADS_PER_SHARD}" \
      MKL_NUM_THREADS="${THREADS_PER_SHARD}" \
      BLIS_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMEXPR_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMBA_NUM_THREADS="${THREADS_PER_SHARD}" \
      python3 -m mean_field.devtools.run_htg_fig9b_boundary_targeted_diagnostic \
        --output-dir "${shard_dir}" \
        --artifact-prefix "${PREFIX}" \
        --shard-index "${shard}" \
        --shard-count "${N_SHARDS}" \
        --seeds-per-flavor-class "${SEEDS_PER_FLAVOR_CLASS}" \
        "$@"
  else
    env \
      OMP_NUM_THREADS="${THREADS_PER_SHARD}" \
      OPENBLAS_NUM_THREADS="${THREADS_PER_SHARD}" \
      MKL_NUM_THREADS="${THREADS_PER_SHARD}" \
      BLIS_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMEXPR_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMBA_NUM_THREADS="${THREADS_PER_SHARD}" \
      python3 -m mean_field.devtools.run_htg_fig9b_boundary_targeted_diagnostic \
        --output-dir "${shard_dir}" \
        --artifact-prefix "${PREFIX}" \
        --shard-index "${shard}" \
        --shard-count "${N_SHARDS}" \
        --seeds-per-flavor-class "${SEEDS_PER_FLAVOR_CLASS}" \
        "$@"
  fi
}

echo "[boundary-packed] output_dir=${OUTPUT_DIR}"
echo "[boundary-packed] prefix=${PREFIX}"
echo "[boundary-packed] shards=${N_SHARDS} threads_per_shard=${THREADS_PER_SHARD} slurm_cpus=${SLURM_CPUS}"
echo "[boundary-packed] seeds_per_flavor_class=${SEEDS_PER_FLAVOR_CLASS}"
echo "[boundary-packed] target_thetas=1.60,1.65,1.70,1.75 target_wAA=75,80,85,90"
echo "[boundary-packed] candidates_per_parameter=$((2 * 4 * SEEDS_PER_FLAVOR_CLASS))"

shard_dirs=()
pids=()
for ((shard = 0; shard < N_SHARDS; shard++)); do
  start_cpu=$((shard * THREADS_PER_SHARD))
  end_cpu=$((start_cpu + THREADS_PER_SHARD - 1))
  shard_dirs+=("${OUTPUT_DIR}/_shards/shard_${shard}")
  run_shard "${shard}" "${start_cpu}-${end_cpu}" "$@" >"${OUTPUT_DIR}/_shards/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if wait "${pid}"; then
    :
  else
    status=$?
  fi
done
if (( status != 0 )); then
  exit "${status}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[boundary-packed] dry-run complete; skipping shard merge because no CSV artifacts are written"
  exit 0
fi

python3 scripts/merge_htg_fig9b_boundary_targeted_shards.py \
  --output-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}" \
  "${shard_dirs[@]}"
