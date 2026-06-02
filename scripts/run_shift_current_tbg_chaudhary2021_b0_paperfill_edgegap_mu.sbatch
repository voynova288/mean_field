#!/bin/bash
#SBATCH --job-name=tbg_chau21_b0_pf_edgegap
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

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
# Filling-derived chemical potentials except edge labels clipped into the flat-dispersive gap.
# This tests whether Chaudhary's near-|nu|=4 FD convention is effectively a gap chemical potential.
MU_MEV="-30,-13.800533,-10.681777,-8.9511449,-7.74249,-6.8624881,-6.377587,-5.5606111,-0.54931615,4.4648074,5.2743852,5.7629047,6.6731938,7.9118387,9.6545384,12.848276,30"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  --theta-deg 0.8 --lg 7 --mesh-size 20 --mu-mesh-size 24 --c3-symmetrize-grid \
  --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --degeneracy 1 --filling-degeneracy 4 --eta-mev 2 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --path-points-per-segment 100 --band-window 8 \
  --output-dir results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_edgegap_mu_lg7_m20_c3
