#!/bin/bash
#SBATCH --job-name=tbg_chau21_tensor
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

COMMON=(
  --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid --mu-mesh-size 10
  --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354
  --filling=-2,0,2 --eta-mev 2
  --emin 0 --emax 0.08 --n-energy 321 --energy-bin-width-mev 0.25
  --fd-bands 1 --fd-mode same_side --path-points-per-segment 30 --band-window 8
  --component 'x;xx' --component 'x;xy' --component 'x;yx' --component 'x;yy'
  --component 'y;xx' --component 'y;xy' --component 'y;yx' --component 'y;yy'
)

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  "${COMMON[@]}" --delta1-mev 5 --delta2-mev 5 \
  --output-dir results/shift_current_tbg/chaudhary2021_tensor_audit_delta5_5_lg7_m10

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  "${COMMON[@]}" --delta1-mev 5 --delta2-mev 10 \
  --output-dir results/shift_current_tbg/chaudhary2021_tensor_audit_delta5_10_lg7_m10
