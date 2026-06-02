#!/bin/bash
#SBATCH --job-name=tbg_chau21_hartree_eps_T15
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
cd /data/home/ziyuzhu/Mean_Field

for EPS in 10 20 30; do
  TAG="eps${EPS}_T15K"
  BANDS_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_lg7_m9_${TAG}"
  RESP_DIR="results/shift_current_tbg/chaudhary2021_hartree_response_lg7_m10_c3_${TAG}"

  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
    --theta-deg 0.8 --lg 7 --mesh-size 9 --path-points-per-segment 80 \
    --nus=-3,-2,-1,0,1,2,3 \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r "${EPS}" --temperature-k 15 --mixing 0.35 --precision 1e-5 --max-iter 80 \
    --bands-each-side 5 \
    --output-dir "${BANDS_DIR}"

  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
    --hartree-dir "${BANDS_DIR}" \
    --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
    --filling=-3,-2,-1,0,1,2,3 \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r "${EPS}" --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
    --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
    --fd-bands 10 --fd-mode same_side --component 'y;xx' \
    --output-dir "${RESP_DIR}"
done
