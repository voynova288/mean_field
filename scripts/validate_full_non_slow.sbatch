#!/bin/bash
#SBATCH -J mf_nonslow_tests
#SBATCH -p test
#SBATCH --account=hmt03
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o logs/mf_nonslow_tests_%j.out
#SBATCH -e logs/mf_nonslow_tests_%j.err

set -euo pipefail
REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
printf '[job] host=%s date=%s job=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-manual}"
python -m pytest -m 'not slow' -q
