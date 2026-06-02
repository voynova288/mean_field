#!/bin/bash
#SBATCH -J rlg_occ_check
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/rlg_occ_check_%j.out
#SBATCH -e logs/rlg_occ_check_%j.err

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

ROOT="results/RnG_hBN/fig6_hf_bands_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix/tasks"
OUT="results/RnG_hBN/diagnostics_projection_occupations_3p3_20260525"

/data/home/ziyuzhu/miniconda3/bin/python3 scripts/diagnose_rlg_hbn_projection_occupations.py \
  --output-dir "${OUT}" \
  --label current_fig6_3p3 \
  --checkpoint "${ROOT}/task_0_xi0_V064meV_flavor_seed1/xi0_V064meV/runs/flavor_seed1/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_1_xi0_V064meV_flavor_seed2/xi0_V064meV/runs/flavor_seed2/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_2_xi0_V064meV_flavor_seed3/xi0_V064meV/runs/flavor_seed3/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_3_xi0_V064meV_flavor_seed4/xi0_V064meV/runs/flavor_seed4/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_4_xi0_V064meV_bm_seed1/xi0_V064meV/runs/bm_seed1/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_8_xi0_V064meV_perturbed_seed1/xi0_V064meV/runs/perturbed_seed1/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_9_xi0_V064meV_perturbed_seed2/xi0_V064meV/runs/perturbed_seed2/checkpoints/hf_checkpoint_latest.npz" \
  --checkpoint "${ROOT}/task_16_xi1_V064meV_flavor_seed1/xi1_V064meV/runs/flavor_seed1/checkpoints/hf_checkpoint_latest.npz"
