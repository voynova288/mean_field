#!/bin/bash
#SBATCH --job-name=tbg_chau21_fd_same
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
  --fd-bands 10 --fd-mode same_side
  --path-points-per-segment 80 --band-window 8
  --c3-symmetrize-grid
)

for spec in "7 12" "7 16" "9 10"; do
  set -- $spec
  LG=$1
  MESH=$2
  OUT="results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd_same_lg${LG}_m${MESH}_c3"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
    "${COMMON[@]}" \
    --lg "$LG" --mesh-size "$MESH" --mu-mesh-size "$MESH" \
    --output-dir "$OUT"
  PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_fig2 \
    --input "$OUT/chaudhary2021_b0_noninteracting.npz" \
    --summary "$OUT/summary.json" \
    --component 'y;xx' \
    --output "$OUT/chaudhary2021_fig2_style.png"
done
