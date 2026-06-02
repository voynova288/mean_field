#!/bin/bash
#SBATCH --job-name=sc_htg_pathfix
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shift_current_htg_pathfix_%j.out
#SBATCH --error=logs/shift_current_htg_pathfix_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/bands_pathfix_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT}"
python -m analysis.shift_current_htg.run_htg_bands_dos \
  --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells 3 \
  --points-per-segment 40 --central-band-count 24 --dos-mesh 8 --dos-sigma-mev 5 \
  --output-dir "${RUN_ROOT}/bands_dos"
echo "[done] $(date) run_root=${RUN_ROOT}"
