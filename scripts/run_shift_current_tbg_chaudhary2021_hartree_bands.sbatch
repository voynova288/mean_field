#!/bin/bash
#SBATCH --job-name=tbg_chau21_hartree_bands
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
  --theta-deg 0.8 --lg 7 --mesh-size 9 --path-points-per-segment 100 \
  --nus=-3,-2,-1,0,1,2,3 \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 15 --mixing 0.30 --precision 1e-6 --max-iter 45 \
  --bands-each-side 5 \
  --output-dir results/shift_current_tbg/chaudhary2021_hartree_bands_lg7_m9_eps15
