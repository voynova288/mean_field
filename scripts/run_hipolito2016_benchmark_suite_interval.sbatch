#!/bin/bash
#SBATCH --job-name=hip_bench_interval
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "job=${SLURM_JOB_ID:-none}"

PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_benchmark_suite \
  --theta-count 72 \
  --transition-energy-intervals 900 \
  --workers ${SLURM_CPUS_PER_TASK:-8} \
  --output-dir results/shift_current_htg/hipolito2016_benchmark_suite_interval
