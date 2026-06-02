#!/bin/bash
#SBATCH --job-name=sc_htg_n3
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/shift_current_htg_nshell3_%j.out
#SBATCH --error=logs/shift_current_htg_nshell3_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src

RUN_ROOT="results/shift_current_htg/fig1_fig2_nshell3_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

echo "[start] $(date)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} run_root=${RUN_ROOT}"
echo "threads OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS}"

python -m analysis.shift_current_htg.run_htg_bands_dos \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --points-per-segment 20 --central-band-count 24 --dos-mesh 8 --dos-sigma-mev 5 \
  --output-dir "${RUN_ROOT}/bands_dos"

python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --mesh-size 12 --n-energy 241 --eta-mev 1,2 \
  --pair central_flat:-1,0 \
  --output-dir "${RUN_ROOT}/central_flat"

python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/central_flat/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/central_flat/summary.json" \
  --group central_flat --eta-mev 2 --xmax 0.12 --reflect-y \
  --output "${RUN_ROOT}/central_flat/central_flat_eta2meV_spectra_reflect_y.png"

echo "[done] $(date) run_root=${RUN_ROOT}"
