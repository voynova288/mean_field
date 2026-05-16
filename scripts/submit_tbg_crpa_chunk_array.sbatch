#!/bin/bash
# Usage:
#   sbatch --array=0-15%4 -c 4 --mem=24G scripts/submit_tbg_crpa_chunk_array.sbatch \
#     results/TBG_HF_cRPA/bm_lk8_lg11.npz \
#     results/TBG_HF_cRPA/crpa_lk8_lg11_q5_chunks \
#     5 16

#SBATCH -J crpa_chunk
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o logs/crpa_chunk_%A_%a.out
#SBATCH -e logs/crpa_chunk_%A_%a.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch --array=0-(N-1) scripts/submit_tbg_crpa_chunk_array.sbatch <bm_solution.npz> <output_root> <q_lg> <chunk_count> [extra run_tbg_crpa_chunk.py args...]" >&2
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
shift 4

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_chunk_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

CHUNK_DIR="${OUTPUT_ROOT}/chunk_${SLURM_ARRAY_TASK_ID}"

echo "[job] command=python scripts/mean_field_tools.py run_tbg_crpa_chunk --bm-solution ${BM_SOLUTION} --q-lg ${Q_LG} --chunk-index ${SLURM_ARRAY_TASK_ID} --chunk-count ${CHUNK_COUNT} --output-dir ${CHUNK_DIR} $*"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] array_task=${SLURM_ARRAY_TASK_ID}/${CHUNK_COUNT}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

python scripts/mean_field_tools.py run_tbg_crpa_chunk \
  --bm-solution "${BM_SOLUTION}" \
  --q-lg "${Q_LG}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-count "${CHUNK_COUNT}" \
  --output-dir "${CHUNK_DIR}" \
  "$@"
