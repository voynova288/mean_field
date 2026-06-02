#!/bin/bash
#SBATCH -J tbg_zfig10_smoke
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/tbg_zhang_fig10_smoke_%j.out
#SBATCH -e logs/tbg_zhang_fig10_smoke_%j.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tbg_zhang_fig10_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/TBG_Zhang2022_fig10/smoke_${SLURM_JOB_ID:-manual}}"

echo "[job] hostname=$(hostname) job_id=${SLURM_JOB_ID:-manual} cpus=${SLURM_CPUS_PER_TASK:-} output_dir=${OUT_DIR}"

python scripts/mean_field_tools.py run_tbg_zhang_fig10 \
  --output-dir "${OUT_DIR}" \
  --theta-deg 1.05 \
  --vf 2135.4 \
  --w0 79.7 \
  --w1 97.4 \
  --epsilon-r 7 \
  --ds-angstrom 400 \
  --lg 3 \
  --kmesh 3 \
  --interaction-lg 3 \
  --path-points-per-segment 4 \
  --max-iter 2 \
  --mixing 0.5 \
  --precision 1e-4 \
  --energy-reference max_selected \
  --plot-sector partial \
  --ylim-ev=auto
