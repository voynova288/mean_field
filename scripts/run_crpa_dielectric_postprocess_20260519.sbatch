#!/bin/bash
# Postprocess a merged TBG cRPA dielectric artifact after the chunk merge.
# Usage:
#   sbatch scripts/run_crpa_dielectric_postprocess_20260519.sbatch \
#     <crpa_dir> <validation_dir> <reference_crpa_dir> <reference_curve_tsv> <comparison_dir>

#SBATCH -J crpa_eps_post
#SBATCH -p regular256
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/crpa_eps_post_%j.out
#SBATCH -e logs/crpa_eps_post_%j.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ $# -ne 5 ]]; then
  echo "Usage: sbatch $0 <crpa_dir> <validation_dir> <reference_crpa_dir> <reference_curve_tsv> <comparison_dir>" >&2
  exit 2
fi

CRPA_DIR="$1"
VALIDATION_DIR="$2"
REFERENCE_CRPA_DIR="$3"
REFERENCE_CURVE_TSV="$4"
COMPARISON_DIR="$5"

mkdir -p "${REPO_ROOT}/logs" "${VALIDATION_DIR}" "${COMPARISON_DIR}"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_eps_post_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

echo "[crpa-eps-post] hostname=$(hostname)"
echo "[crpa-eps-post] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[crpa-eps-post] crpa_dir=${CRPA_DIR}"
echo "[crpa-eps-post] validation_dir=${VALIDATION_DIR}"
echo "[crpa-eps-post] reference_crpa_dir=${REFERENCE_CRPA_DIR}"
echo "[crpa-eps-post] reference_curve_tsv=${REFERENCE_CURVE_TSV}"
echo "[crpa-eps-post] comparison_dir=${COMPARISON_DIR}"

python scripts/mean_field_tools.py validate_tbg_crpa_artifact \
  --crpa-dir "${CRPA_DIR}" \
  --output-dir "${VALIDATION_DIR}" \
  --reference-crpa-dir "${REFERENCE_CRPA_DIR}"

python scripts/mean_field_tools.py compare_tbg_crpa_fig1e \
  --paper-curve "${REFERENCE_CURVE_TSV}" \
  --output-dir "${COMPARISON_DIR}" \
  --case "current=${CRPA_DIR}" \
  --case "previous_q11_reference=${REFERENCE_CRPA_DIR}" \
  --sample-q 0,0.02252,0.04202,0.05958,0.08120,0.10320,0.13512,0.15764,0.20268,0.3,0.4,0.6,0.8,1.0,1.2 \
  --x-max 1.2 \
  --y-min 3.5 \
  --y-max 21.5

echo "[crpa-eps-post] done"
