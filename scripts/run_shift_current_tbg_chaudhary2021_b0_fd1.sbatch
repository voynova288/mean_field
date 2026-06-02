#!/bin/bash
#SBATCH --job-name=tbg_chau21_fd1
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

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
  --theta-deg 0.8 --lg 7 --mesh-size 16 --c3-symmetrize-grid --mu-mesh-size 16 \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --filling=-3,-2,-1,0,1,2,3 --eta-mev 2 \
  --emin 0 --emax 0.08 --n-energy 401 --energy-bin-width-mev 0.25 \
  --fd-bands 1 --fd-mode same_side --path-points-per-segment 60 --band-window 8 \
  --output-dir results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd1_lg7_m16_c3

PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_fig2 \
  --input results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd1_lg7_m16_c3/chaudhary2021_b0_noninteracting.npz \
  --summary results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd1_lg7_m16_c3/summary.json \
  --component 'y;xx' \
  --output results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd1_lg7_m16_c3/chaudhary2021_fig2_style.png
