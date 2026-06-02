#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_fd_decomp_eps10
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

FILLINGS="-3.95,-3.5,-2,2,3.5,3.95"
# Interior labels use SCF chemical potentials from eps=10/T=15K paper-fill Hartree; edge labels are placed in flat-dispersive gaps.
MU_MEV="-45,-33.294384,-22.106306,21.120492,32.341258,45"
HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps10_T15K"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_fd_decomposition \
  --hartree-dir "${HARTREE_DIR}" \
  --theta-deg 0.8 --lg 7 --mesh-size 12 --c3-symmetrize-grid \
  --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' --gamma-radius-frac 0.23 \
  --output-dir results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps10_edgegap_lg7_m12_c3
