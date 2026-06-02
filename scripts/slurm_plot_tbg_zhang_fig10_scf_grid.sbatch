#!/bin/bash
#SBATCH -J tbg_zfig10_scfplot
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/tbg_zhang_fig10_scf_grid_plot_%j.out
#SBATCH -e logs/tbg_zhang_fig10_scf_grid_plot_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tbg_zhang_fig10_scf_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/TBG_Zhang2022_fig10/zhang_fig10_lg11_k12_eps7_ds400_hf_random_seed1_primitivev0_132678}"
Y_LIM="${Y_LIM:--0.4,0.1}"

printf '[job] hostname=%s job_id=%s cpus=%s result_dir=%s\n' "$(hostname)" "${SLURM_JOB_ID:-manual}" "${SLURM_CPUS_PER_TASK:-}" "${RESULT_DIR}"
printf '[job] date=%s repo=%s\n' "$(date --iso-8601=seconds)" "${REPO_ROOT}"

python scripts/mean_field_tools.py plot_tbg_zhang_fig10_scf_grid \
  --result-dir "${RESULT_DIR}" \
  --ylim-ev="${Y_LIM}"
