#!/bin/bash
#SBATCH --job-name=sc_htg_a24
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/shift_current_htg_active24_%j.out
#SBATCH --error=logs/shift_current_htg_active24_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src

RUN_ROOT="results/shift_current_htg/active24_nshell3_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT} cpus=${SLURM_CPUS_PER_TASK:-1}"

python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --mesh-size 8 --n-energy 501 --emin 0 --emax 1.0 --eta-mev 2 \
  --pair-window active24:-12,11 \
  --output-dir "${RUN_ROOT}/active24"

python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/active24/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/active24/summary.json" \
  --group active24 --eta-mev 2 --xmax 1.0 --reflect-y \
  --output "${RUN_ROOT}/active24/active24_eta2meV_spectra_reflect_y.png"

python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/active24/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/active24/summary.json" \
  --group active24 --eta-mev 2 --xmax 0.12 --reflect-y \
  --output "${RUN_ROOT}/active24/active24_eta2meV_spectra_reflect_y_thz.png"

echo "[done] $(date) run_root=${RUN_ROOT}"
