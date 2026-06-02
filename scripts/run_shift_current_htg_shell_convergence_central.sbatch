#!/bin/bash
#SBATCH --job-name=sc_shell_cf
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/shift_current_htg_shell_cf_%j.out
#SBATCH --error=logs/shift_current_htg_shell_cf_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/physics_audit_shell_central_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT}"
for shell in 2 3 4 5; do
  out="${RUN_ROOT}/central_shell${shell}"
  echo "[shell ${shell}] $(date)"
  python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
    --theta-deg 1.95 --stacking ABA --r 0.8 --n-shells "${shell}" \
    --mesh-size 8 --c3-symmetrize-grid --n-energy 241 --emin 0 --emax 0.12 --eta-mev 2 \
    --pair central_flat:-1,0 \
    --component 'x;yy' --component 'x;xx' --component 'y;yx' --component 'y;xy' \
    --component 'y;xx' --component 'y;yy' --component 'x;xy' --component 'x;yx' \
    --output-dir "${out}"
  python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "${out}/htg_bandpair_spectra.npz" --summary "${out}/summary.json" \
    --group central_flat --eta-mev 2 --xmax 0.12 --reflect-y \
    --output "${out}/central_shell${shell}_eta2_reflect_y.png"
done
echo "[done] $(date) run_root=${RUN_ROOT}"
