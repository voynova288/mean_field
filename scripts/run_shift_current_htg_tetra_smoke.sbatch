#!/bin/bash
#SBATCH --job-name=htg_tetra_smoke
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra_tetra \
  --n-shells 1 --mesh-size 4 --eta-mev 2 --n-energy 41 --emax 0.05 \
  --component 'x;yy' --component 'y;xx' \
  --pair central_flat:-1,0 \
  --output-dir results/shift_current_htg/tmp_htg_tetra_smoke
