#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_fd_decomp_e15e20
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"

FILLINGS="-3.95,-3.5,-2,2,3.5,3.95"

run_one() {
  local EPS="$1"
  local MU_MEV="$2"
  local HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps${EPS}_T15K"
  local OUT="results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps${EPS}_edgegap_lg7_m12_c3"
  echo "=== eps=${EPS} mu=${MU_MEV} ==="
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_fd_decomposition \
    --hartree-dir "${HARTREE_DIR}" \
    --theta-deg 0.8 --lg 7 --mesh-size 12 --c3-symmetrize-grid \
    --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r "${EPS}" --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
    --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
    --fd-bands 10 --fd-mode same_side --component 'y;xx' --gamma-radius-frac 0.23 \
    --output-dir "${OUT}"
}

run_one 15 "-40,-26.497083,-18.248633,17.204695,25.486193,40"
run_one 20 "-40,-23.20501,-15.89875,14.831656,22.195258,40"
