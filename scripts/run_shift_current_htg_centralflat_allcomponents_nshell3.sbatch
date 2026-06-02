#!/bin/bash
#SBATCH --job-name=sc_htg_allc
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shift_current_htg_allcomponents_%j.out
#SBATCH --error=logs/shift_current_htg_allcomponents_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/centralflat_allcomponents_nshell3_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT}"
python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --mesh-size 12 --n-energy 241 --eta-mev 2 \
  --pair central_flat:-1,0 \
  --component 'x;yy' --component 'x;xx' --component 'y;yx' --component 'y;xy' \
  --component 'y;xx' --component 'y;yy' --component 'x;xy' --component 'x;yx' \
  --output-dir "${RUN_ROOT}/central_flat_allc"
echo "[done] $(date) run_root=${RUN_ROOT}"
