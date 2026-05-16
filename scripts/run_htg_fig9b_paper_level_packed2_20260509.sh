#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-results/HTG/htg_fig9b_bandwidth_scan_8x10_paper_level_packed2_20260509_001}"
shift || true

PREFIX="fig9b_conduction_bandwidth_scan_8x10_paper_level"
N_SHARDS="${HTG_FIG9B_PACKED_SHARDS:-8}"
THREADS_PER_SHARD="${HTG_FIG9B_THREADS_PER_SHARD:-8}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-64}"
DRY_RUN=false
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done

if (( SLURM_CPUS < N_SHARDS * THREADS_PER_SHARD )); then
  echo "Need at least $((N_SHARDS * THREADS_PER_SHARD)) CPUs, got ${SLURM_CPUS}" >&2
  exit 2
fi

# Explicit calculation centers for Kwan Fig. 9b. These are denser than the
# visible major ticks, and plotted cell edges are derived metadata only.
thetas=(1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95)
waas=(40 47.5 55 60 65 70 75 80 85 90)
cases=()
case_index=0
for waa in "${waas[@]}"; do
  for theta in "${thetas[@]}"; do
    cases+=("${case_index}:${theta}:${waa}:theta${theta}_wAA${waa}")
    case_index=$((case_index + 1))
  done
done

seeds=()
for seed in $(seq 1 31); do
  seeds+=("${seed}")
done

seeds_arg="$(IFS=,; echo "${seeds[*]}")"
mkdir -p "${OUTPUT_DIR}/_shards"

run_shard() {
  local shard="$1"
  local cpu_set="$2"
  shift 2
  local shard_dir="${OUTPUT_DIR}/_shards/shard_${shard}"
  local shard_cases=()
  local entry idx theta waa label
  mkdir -p "${shard_dir}"
  for entry in "${cases[@]}"; do
    IFS=: read -r idx theta waa label <<<"${entry}"
    if (( idx % N_SHARDS == shard )); then
      shard_cases+=("${theta}:${waa}:${label}")
    fi
  done
  local cases_arg
  cases_arg="$(IFS=,; echo "${shard_cases[*]}")"
  echo "[packed] shard=${shard} cpu_set=${cpu_set} threads=${THREADS_PER_SHARD} cases=${#shard_cases[*]} output=${shard_dir}"
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "${cpu_set}" env \
      OMP_NUM_THREADS="${THREADS_PER_SHARD}" \
      OPENBLAS_NUM_THREADS="${THREADS_PER_SHARD}" \
      MKL_NUM_THREADS="${THREADS_PER_SHARD}" \
      BLIS_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMEXPR_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMBA_NUM_THREADS="${THREADS_PER_SHARD}" \
      python3 scripts/mean_field_tools.py run_htg_fig9b_exact_anchor_scan \
        --output-dir "${shard_dir}" \
        --cases "${cases_arg}" \
        --artifact-prefix "${PREFIX}" \
        --report-title "HTG Fig. 9b 8x10 Paper-Level Multi-Init Scan shard ${shard}" \
        --init-modes d3b,d3a,bm,fi,flavor,vp,sp,chern,perturbed,random \
        --seeds "${seeds_arg}" \
        --reproduction-mode paper-level \
        "$@"
  else
    env \
      OMP_NUM_THREADS="${THREADS_PER_SHARD}" \
      OPENBLAS_NUM_THREADS="${THREADS_PER_SHARD}" \
      MKL_NUM_THREADS="${THREADS_PER_SHARD}" \
      BLIS_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMEXPR_NUM_THREADS="${THREADS_PER_SHARD}" \
      NUMBA_NUM_THREADS="${THREADS_PER_SHARD}" \
      python3 scripts/mean_field_tools.py run_htg_fig9b_exact_anchor_scan \
        --output-dir "${shard_dir}" \
        --cases "${cases_arg}" \
        --artifact-prefix "${PREFIX}" \
        --report-title "HTG Fig. 9b 8x10 Paper-Level Multi-Init Scan shard ${shard}" \
        --init-modes d3b,d3a,bm,fi,flavor,vp,sp,chern,perturbed,random \
        --seeds "${seeds_arg}" \
        --reproduction-mode paper-level \
        "$@"
  fi
}

echo "[packed] output_dir=${OUTPUT_DIR}"
echo "[packed] prefix=${PREFIX}"
echo "[packed] shards=${N_SHARDS} threads_per_shard=${THREADS_PER_SHARD} slurm_cpus=${SLURM_CPUS}"
echo "[packed] total_cases=${#cases[*]} expected_wcond_shape=(10,8)"

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
  echo "[packed] dry-run complete; skipping shard merge because no TSV artifacts are written"
  exit 0
fi

python3 scripts/merge_htg_fig9b_packed_shards.py \
  --output-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}" \
  "${shard_dirs[@]}"
