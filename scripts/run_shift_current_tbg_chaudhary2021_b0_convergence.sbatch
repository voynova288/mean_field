#!/bin/bash
#SBATCH --job-name=tbg_chau21_b0_conv
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
echo "cpus=${SLURM_CPUS_PER_TASK:-1}"

COMMON=(
  --theta-deg 0.8
  --delta1-mev 5 --delta2-mev 5
  --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354
  --filling=-3,-2,-1,0,1,2,3
  --eta-mev 2
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25
  --fd-bands 10
  --path-points-per-segment 60 --band-window 8
  --c3-symmetrize-grid
)

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  "${COMMON[@]}" \
  --lg 7 --mesh-size 16 --mu-mesh-size 16 \
  --output-dir results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_lg7_m16_c3

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  "${COMMON[@]}" \
  --lg 9 --mesh-size 10 --mu-mesh-size 10 \
  --output-dir results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_lg9_m10_c3
