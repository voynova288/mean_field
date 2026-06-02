#!/bin/bash
#SBATCH --job-name=tbg_chau21_maps
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_integrand_maps \
  --theta-deg 0.8 --lg 7 --mesh-size 55 \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --group FF --labels=-2,0,2 --component 'y;xx' --target-energy-mev 16 --eta-mev 2 \
  --output-dir results/shift_current_tbg/chaudhary2021_integrand_map_ff_nu_m2_0_p2_lg7_m55

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_integrand_maps \
  --theta-deg 0.8 --lg 7 --mesh-size 55 \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --group FD --fd-bands 1 --fd-mode same_side --labels=-30,0,30 --mu-mev=-30,0,30 \
  --component 'y;xx' --target-energy-mev 21.6 --eta-mev 10 \
  --output-dir results/shift_current_tbg/chaudhary2021_integrand_map_fd_mu_m30_0_p30_lg7_m55
