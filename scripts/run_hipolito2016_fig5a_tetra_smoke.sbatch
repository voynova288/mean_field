#!/bin/bash
#SBATCH --job-name=hip5a_tetra_smoke
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "job=${SLURM_JOB_ID:-none}"
echo "threads=${OMP_NUM_THREADS}"

PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_fig5a_full_bz_tetra \
  --deltas-ev 0.2,0.5 \
  --mesh-size 12 \
  --energy-bin-width-mev 20 \
  --n-photon 51 \
  --emax 1.0 \
  --output-dir results/shift_current_htg/tmp_fig5a_tetra_smoke \
  --workers 2
