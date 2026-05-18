#!/bin/bash
# Plot a current RLG/hBN HF checkpoint using nearest points from the saved SCF mesh.

#SBATCH -J rlg_ckpt_mesh_bands
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/rlg_ckpt_mesh_bands_%j.out
#SBATCH -e logs/rlg_ckpt_mesh_bands_%j.err

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch $0 <checkpoint.npz> <paper_hf_config.json> <panel> <output-dir> [points_per_segment]" >&2
  exit 2
fi

CHECKPOINT="$1"
CONFIG="$2"
PANEL="$3"
OUTPUT_DIR="$4"
POINTS_PER_SEGMENT="${5:-24}"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_ckpt_mesh_bands_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

echo "[job] checkpoint=${CHECKPOINT}"
echo "[job] config=${CONFIG}"
echo "[job] panel=${PANEL}"
echo "[job] output_dir=${OUTPUT_DIR}"
echo "[job] points_per_segment=${POINTS_PER_SEGMENT}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

python scripts/plot_rlg_hbn_checkpoint_mesh_bands.py \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --panel "${PANEL}" \
  --paper-target fig6 \
  --output-dir "${OUTPUT_DIR}" \
  --points-per-segment "${POINTS_PER_SEGMENT}" \
  --ylim-mev=-90,90 \
  --dpi 180
