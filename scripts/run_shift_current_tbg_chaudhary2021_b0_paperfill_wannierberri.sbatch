#!/bin/bash
#SBATCH --job-name=tbg_chau21_b0_wb40
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"
echo "threads OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMBA=$NUMBA_NUM_THREADS"

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
OUT="results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta40"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  --theta-deg 0.8 --lg 7 --mesh-size 16 --mu-mesh-size 16 --c3-symmetrize-grid \
  --filling="${FILLINGS}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --degeneracy 1 --filling-degeneracy 4 --eta-mev 2 \
  --response-formula wannierberri --sc-eta-mev 40 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --path-points-per-segment 100 --band-window 8 \
  --output-dir "${OUT}"

PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_fig2 \
  --input "${OUT}/chaudhary2021_b0_noninteracting.npz" \
  --summary "${OUT}/summary.json" \
  --component 'y;xx' --fillings="${FILLINGS}" \
  --output "${OUT}/chaudhary2021_fig2_paperfill_wannierberri_sceta40.png"

echo "completed $(date) output=${OUT}"
