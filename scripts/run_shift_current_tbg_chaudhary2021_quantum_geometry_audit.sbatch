#!/bin/bash
#SBATCH --job-name=tbg_chau21_qg_audit
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_quantum_geometry_audit \
  --theta-deg 0.8 --lg 7 --mesh-size 55 --mu-mesh-size 12 \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --component 'y;xx' \
  --ff-fillings='-2,0,2' --fd-mu-mev='-30,0,30' \
  --ff-target-mev 16.0 --fd-target-mev 21.6 --ff-eta-mev 2.0 --fd-eta-mev 10.0 \
  --fd-bands 1 --fd-mode same_side \
  --output-dir results/shift_current_tbg/chaudhary2021_quantum_geometry_audit_lg7_m55
