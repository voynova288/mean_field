#!/bin/bash
#SBATCH -J rlg_active_window
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/rlg_active_window_%j.out
#SBATCH -e logs/rlg_active_window_%j.err

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

echo "[job] hostname=$(hostname)"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

/data/home/ziyuzhu/miniconda3/bin/python3 scripts/diagnose_rlg_hbn_active_window.py \
  --basis-cache results/RnG_hBN/cache_fig6_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix/basis/basis_c6e2812769cfcb370c037d38 \
  --path-npz results/RnG_hBN/diagnostics_xi0_bm_seed1_checkpoint_band_20260525_1127/xi0_V064meV/scf_exact_path/current_checkpoint_scf_exact_paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma_bands.npz \
  --output-dir results/RnG_hBN/diagnostics_xi0_active_window_gamma_mprime_20260525 \
  --segment gamma_mprime
