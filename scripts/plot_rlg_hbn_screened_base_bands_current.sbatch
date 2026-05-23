#!/bin/bash
# Plot screened-base pre-HF bands from an existing Fig. 6 output root/cache.

#SBATCH -J rlg_screen_bands
#SBATCH -p regular
#SBATCH --account=hmt03
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/rlg_screen_bands_%j.out
#SBATCH -e logs/rlg_screen_bands_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 <output-root> <cache-dir> <output-dir> [panels]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
CACHE_DIR="$2"
OUTPUT_DIR="$3"
PANELS="${4:-xi0_V064meV,xi1_V064meV}"
PATH_MODE="${PATH_MODE:-mesh}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-48}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
DPI="${DPI:-180}"
YLIM_MEV="${YLIM_MEV:--120,120}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs "${OUTPUT_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_screen_bands_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[plot] output_root=${OUTPUT_ROOT}"
echo "[plot] cache_dir=${CACHE_DIR}"
echo "[plot] output_dir=${OUTPUT_DIR}"
echo "[plot] panels=${PANELS}"
echo "[plot] path_mode=${PATH_MODE}"
echo "[plot] cache_policy=${CACHE_POLICY}"
echo "[plot] hostname=$(hostname)"
echo "[plot] cpus=${SLURM_CPUS_PER_TASK:-}"

python scripts/plot_rlg_hbn_screened_base_bands.py \
  --output-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --panels "${PANELS}" \
  --cache-dir "${CACHE_DIR}" \
  --cache-policy "${CACHE_POLICY}" \
  --path-mode "${PATH_MODE}" \
  --points-per-segment "${POINTS_PER_SEGMENT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --ylim-mev="${YLIM_MEV}" \
  --dpi "${DPI}"
