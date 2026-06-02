#!/bin/bash
#SBATCH --job-name=sc_hbar
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/shift_current_htg_hbar_%j.out
#SBATCH --error=logs/shift_current_htg_hbar_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/mao_zeta0_hbar_check_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT}"
python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --domain hbar --r 0.8 --n-shells 3 \
  --mesh-size 8 --c3-symmetrize-grid --n-energy 501 --emin 0 --emax 0.12 --eta-mev 2 \
  --pair-window active24:-12,11 \
  --component 'x;yy' --component 'x;xx' --component 'y;yx' --component 'y;xy' \
  --component 'y;xx' --component 'y;yy' --component 'x;xy' --component 'x;yx' \
  --output-dir "${RUN_ROOT}/active24_allc"
for angle in 0 5 10; do
  python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "${RUN_ROOT}/active24_allc/htg_bandpair_spectra.npz" \
    --summary "${RUN_ROOT}/active24_allc/summary.json" \
    --group active24 --eta-mev 2 --xmax 0.12 --rotation-deg "${angle}" --reflect-y \
    --output "${RUN_ROOT}/active24_allc/active24_hbar_eta2meV_rot${angle}_reflect_y_thz.png"
done
echo "[done] $(date) run_root=${RUN_ROOT}"
