#!/bin/bash
#SBATCH -J fig1e_smallq_fit
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/fig1e_smallq_fit_%j.out
#SBATCH -e logs/fig1e_smallq_fit_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_fig1e_smallq_fit_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
python scripts/plot_crpa_fig1e_smallq_fit_vs_q9_20260519.py "$@"
