#!/usr/bin/env bash
#SBATCH --job-name=rlg-hbn-xi0-bm-s1-band
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

SPIN_INDEX="${SPIN_INDEX:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/RnG_hBN/diagnostics_xi0_bm_seed1_checkpoint_band_20260525_1127/xi0_V064meV/scf_exact_path_spin${SPIN_INDEX}}"

/data/home/ziyuzhu/miniconda3/bin/python3 scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
  --checkpoint results/RnG_hBN/diagnostics_xi0_bm_seed1_checkpoint_band_20260525_1127/xi0_V064meV/hf_ground_state.npz \
  --config results/RnG_hBN/diagnostics_xi0_bm_seed1_checkpoint_band_20260525_1127/paper_hf_config.json \
  --panel xi0_V064meV \
  --output-dir "${OUTPUT_DIR}" \
  --path-kind paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma \
  --spin-index "${SPIN_INDEX}" \
  --ylim-mev=-90,90 \
  --dpi 180
