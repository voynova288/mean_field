#!/bin/bash
#SBATCH --job-name=sc_a24_allc
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/shift_current_htg_active24_allc_%j.out
#SBATCH --error=logs/shift_current_htg_active24_allc_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/active24_allcomponents_nshell3_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT} cpus=${SLURM_CPUS_PER_TASK:-1}"
python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --mesh-size 8 --n-energy 501 --emin 0 --emax 1.0 --eta-mev 2 \
  --pair-window active24:-12,11 \
  --component 'x;yy' --component 'x;xx' --component 'y;yx' --component 'y;xy' \
  --component 'y;xx' --component 'y;yy' --component 'x;xy' --component 'x;yx' \
  --output-dir "${RUN_ROOT}/active24_allc"
echo "[done] $(date) run_root=${RUN_ROOT}"
