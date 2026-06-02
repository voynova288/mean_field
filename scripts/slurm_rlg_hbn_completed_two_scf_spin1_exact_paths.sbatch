#!/usr/bin/env bash
#SBATCH --job-name=rlg-hbn-two-scf-s1
#SBATCH --partition=test
#SBATCH --account=hmt03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/results/RnG_hBN/slurm_logs/%x-%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/results/RnG_hBN/slurm_logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg
export PYTHONPATH=/data/home/ziyuzhu/Mean_Field/src:${PYTHONPATH:-}

cd /data/home/ziyuzhu/Mean_Field

ROOT="${ROOT:-results/RnG_hBN/fig6_completed_two_scf_chern_20260525_103601}"
PATH_KIND="${PATH_KIND:-paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma}"
SPIN_INDEX="${SPIN_INDEX:-1}"
YLIM_MEV="${YLIM_MEV:--90,90}"

for PANEL in xi0_V064meV xi1_V064meV; do
  /data/home/ziyuzhu/miniconda3/bin/python3 scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
    --checkpoint "${ROOT}/${PANEL}/hf_ground_state.npz" \
    --config "${ROOT}/paper_hf_config.json" \
    --panel "${PANEL}" \
    --output-dir "${ROOT}/${PANEL}/scf_exact_path_spin${SPIN_INDEX}" \
    --path-kind "${PATH_KIND}" \
    --spin-index "${SPIN_INDEX}" \
    --ylim-mev="${YLIM_MEV}" \
    --dpi 180
done
