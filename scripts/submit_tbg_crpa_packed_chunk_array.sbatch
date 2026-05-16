#!/bin/bash
# Usage:
#   sbatch --array=0-71%2 -c 64 --mem=0 scripts/submit_tbg_crpa_packed_chunk_array.sbatch \
#     <bm_solution> <output_root> <q_lg> <chunk_count> <chunks_per_task> [extra run_tbg_crpa_chunk args...]

#SBATCH -J crpa_chunk_pack
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -o logs/crpa_chunk_pack_%A_%a.out
#SBATCH -e logs/crpa_chunk_pack_%A_%a.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ $# -lt 5 ]]; then
  echo "Usage: sbatch --array=0-(N-1) scripts/submit_tbg_crpa_packed_chunk_array.sbatch <bm_solution> <output_root> <q_lg> <chunk_count> <chunks_per_task> [extra args...]" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a Slurm array job." >&2
  exit 2
fi

BM_SOLUTION="$1"
OUTPUT_ROOT="$2"
Q_LG="$3"
CHUNK_COUNT="$4"
CHUNKS_PER_TASK="$5"
shift 5

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

total_cpus="${SLURM_CPUS_PER_TASK:-1}"
if [[ "${CHUNKS_PER_TASK}" -lt 1 ]]; then
  echo "chunks_per_task must be positive, got ${CHUNKS_PER_TASK}" >&2
  exit 2
fi
threads_per_chunk=$((total_cpus / CHUNKS_PER_TASK))
if [[ "${threads_per_chunk}" -lt 1 ]]; then
  threads_per_chunk=1
fi

start_chunk=$((SLURM_ARRAY_TASK_ID * CHUNKS_PER_TASK))
stop_chunk=$((start_chunk + CHUNKS_PER_TASK))
if [[ "${stop_chunk}" -gt "${CHUNK_COUNT}" ]]; then
  stop_chunk="${CHUNK_COUNT}"
fi
if [[ "${start_chunk}" -ge "${CHUNK_COUNT}" ]]; then
  echo "Packed array task ${SLURM_ARRAY_TASK_ID} starts beyond chunk_count=${CHUNK_COUNT}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] packed_task=${SLURM_ARRAY_TASK_ID} chunks=${start_chunk}:$((stop_chunk - 1))/${CHUNK_COUNT}"
echo "[job] cpus_per_task=${total_cpus}"
echo "[job] chunks_per_task=${CHUNKS_PER_TASK}"
echo "[job] threads_per_chunk=${threads_per_chunk}"
echo "[job] extra_args=$*"

pids=()
for ((chunk_id = start_chunk; chunk_id < stop_chunk; chunk_id++)); do
  chunk_dir="${OUTPUT_ROOT}/chunk_${chunk_id}"
  mpl_dir="${MPLCONFIGDIR:-/tmp}/mplconfig_crpa_chunk_${SLURM_ARRAY_JOB_ID:-manual}_${chunk_id}"
  mkdir -p "${mpl_dir}"
  (
    export OMP_NUM_THREADS="${threads_per_chunk}"
    export OPENBLAS_NUM_THREADS="${threads_per_chunk}"
    export MKL_NUM_THREADS="${threads_per_chunk}"
    export BLIS_NUM_THREADS="${threads_per_chunk}"
    export NUMEXPR_NUM_THREADS="${threads_per_chunk}"
    export NUMBA_NUM_THREADS="${threads_per_chunk}"
    export MPLCONFIGDIR="${mpl_dir}"
    echo "[packed-chunk] start chunk=${chunk_id} threads=${threads_per_chunk} output=${chunk_dir}"
    python scripts/mean_field_tools.py run_tbg_crpa_chunk \
      --bm-solution "${BM_SOLUTION}" \
      --q-lg "${Q_LG}" \
      --chunk-index "${chunk_id}" \
      --chunk-count "${CHUNK_COUNT}" \
      --output-dir "${chunk_dir}" \
      "$@"
    echo "[packed-chunk] done chunk=${chunk_id}"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
