#!/bin/bash
#SBATCH --job-name=sc_htg_fig12
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shift_current_htg_fig12_%j.out
#SBATCH --error=logs/shift_current_htg_fig12_%j.err

set -euo pipefail

cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src

RUN_ROOT="results/shift_current_htg/fig1_fig2_smoke_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

echo "[start] $(date)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} run_root=${RUN_ROOT}"
echo "threads OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS}"

python -m analysis.shift_current_htg.run_htg_bands_dos \
  --theta-deg 1.95 \
  --stacking ABA \
  --r 0.8 \
  --n-shells 2 \
  --points-per-segment 24 \
  --central-band-count 24 \
  --dos-mesh 8 \
  --dos-sigma-mev 5 \
  --output-dir "${RUN_ROOT}/bands_dos"

python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 \
  --stacking ABA \
  --r 0.8 \
  --n-shells 2 \
  --mesh-size 8 \
  --n-energy 241 \
  --eta-mev 1,2 \
  --pair central_flat:-1,0 \
  --output-dir "${RUN_ROOT}/central_flat"

python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/central_flat/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/central_flat/summary.json" \
  --group central_flat \
  --eta-mev 2 \
  --xmax 0.12 \
  --output "${RUN_ROOT}/central_flat/central_flat_eta2meV_spectra.png"

python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 \
  --stacking ABA \
  --r 0.8 \
  --n-shells 2 \
  --mesh-size 6 \
  --n-energy 241 \
  --eta-mev 2 \
  --pair-window central_window:-4,3 \
  --output-dir "${RUN_ROOT}/central_window"

python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/central_window/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/central_window/summary.json" \
  --group central_window \
  --eta-mev 2 \
  --xmax 0.12 \
  --output "${RUN_ROOT}/central_window/central_window_eta2meV_spectra.png"

echo "[done] $(date) run_root=${RUN_ROOT}"
