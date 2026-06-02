#!/bin/bash
#SBATCH --job-name=hip5a_full_tetra
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
# The Python script parallelizes over gaps with one process per gap.  Keep BLAS
# libraries single-threaded inside each process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "job=${SLURM_JOB_ID:-none}"
echo "cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "threads OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS}"
echo "output=results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m360_bin1mev"

PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_fig5a_full_bz_tetra \
  --mesh-size 360 \
  --energy-bin-width-mev 1.0 \
  --n-photon 1901 \
  --gamma-mev 1.0 \
  --workers ${SLURM_CPUS_PER_TASK:-10} \
  --output-dir results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m360_bin1mev
