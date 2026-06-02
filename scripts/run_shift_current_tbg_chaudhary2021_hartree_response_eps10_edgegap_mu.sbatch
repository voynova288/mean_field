#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_resp_eps10_edgegap
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
# SCF mu for interior labels; edge labels placed in the Hartree flat-dispersive gap based on path-band midpoint diagnostics.
MU_MEV="-45,-33.294384,-28.931598,-25.285136,-22.106306,-18.617599,-14.089974,-8.6587054,0,7.56553,13.012393,17.584546,21.120492,24.306304,27.95643,32.341258,45"
HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps10_T15K"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
  --hartree-dir "${HARTREE_DIR}" \
  --theta-deg 0.8 --lg 7 --mesh-size 12 --c3-symmetrize-grid \
  --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --output-dir results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m12_c3_eps10_T15K_edgegap_mu
