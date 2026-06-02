#!/bin/bash
#SBATCH --job-name=tbg_chau21_delta
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

for DELTA in 5 10 20; do
  OUT="results/shift_current_tbg/chaudhary2021_delta_scan_D${DELTA}_lg7_m12_c3"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
    --theta-deg 0.8 --lg 7 --mesh-size 12 --c3-symmetrize-grid --mu-mesh-size 12 \
    --delta1-mev "$DELTA" --delta2-mev "$DELTA" \
    --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --filling=0,30 --mu-mev=0,30 --eta-mev 2 \
    --emin 0 --emax 0.10 --n-energy 501 --energy-bin-width-mev 0.25 \
    --fd-bands 1 --fd-mode same_side --path-points-per-segment 50 --band-window 8 \
    --output-dir "$OUT"
  PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_fig2 \
    --input "$OUT/chaudhary2021_b0_noninteracting.npz" \
    --summary "$OUT/summary.json" \
    --component 'y;xx' --fillings=0,30 --fd-reference-filling 30 \
    --output "$OUT/chaudhary2021_delta_scan_style.png"
done
