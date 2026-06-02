#!/bin/bash
#SBATCH --job-name=htg_tetra_mao_retry
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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
echo "threads OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS}"

# Fig. 2-style central flat-band pair, shell-5, all tensor components.
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra_tetra \
  --theta-deg 1.95 --stacking ABA --domain h --r 0.8 --mass-mev 30 \
  --n-shells 5 --mesh-size 20 --c3-symmetrize-grid \
  --eta-mev 1,2 --emin 0 --emax 0.12 --n-energy 601 --energy-bin-width-mev 0.1 \
  --pair central_flat:-1,0 \
  --output-dir results/shift_current_htg/mao_retry_tetra_central_flat_shell5_m20

# Fig. 1(b)-style low-energy central-window diagnostic, shell-3, all tensor components.
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra_tetra \
  --theta-deg 1.95 --stacking ABA --domain h --r 0.8 --mass-mev 30 \
  --n-shells 3 --mesh-size 18 --c3-symmetrize-grid \
  --eta-mev 1,2 --emin 0 --emax 0.12 --n-energy 601 --energy-bin-width-mev 0.1 \
  --pair-window central_window:-4,3 \
  --output-dir results/shift_current_htg/mao_retry_tetra_central_window_shell3_m18

# Standard plots in code axes and with the previously diagnosed empirical tensor rotation.
for root in \
  results/shift_current_htg/mao_retry_tetra_central_flat_shell5_m20 \
  results/shift_current_htg/mao_retry_tetra_central_window_shell3_m18; do
  for group in central_flat central_window; do
    if [ -f "$root/htg_bandpair_spectra.npz" ]; then
      PYTHONPATH=src python -m analysis.shift_current_htg.plot_bandpair_spectra \
        --input "$root/htg_bandpair_spectra.npz" \
        --summary "$root/summary.json" \
        --group "$group" --eta-mev 2 --components 'x;yy' 'y;xx' --xmax 0.12 \
        --output "$root/${group}_eta2_code_axes.png" || true
      PYTHONPATH=src python -m analysis.shift_current_htg.plot_bandpair_spectra \
        --input "$root/htg_bandpair_spectra.npz" \
        --summary "$root/summary.json" \
        --group "$group" --eta-mev 2 --components 'x;yy' 'y;xx' --xmax 0.12 \
        --reflect-y --rotation-deg 4.8 \
        --output "$root/${group}_eta2_rot4p8_reflect_y.png" || true
    fi
  done
done
