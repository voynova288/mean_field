#!/bin/bash
#SBATCH --job-name=sc_cf_s5m16_nc3
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/shift_current_htg_cf_s5m16_noc3_%j.out
#SBATCH --error=logs/shift_current_htg_cf_s5m16_noc3_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH=src
RUN_ROOT="results/shift_current_htg/physics_audit_central_shell5_mesh16_noc3_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
echo "[start] $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} run_root=${RUN_ROOT}"
python -m analysis.shift_current_htg.run_htg_bandpair_spectra \
  --theta-deg 1.95 --stacking ABA --domain h --r 0.8 --n-shells 5 \
  --mesh-size 16 --n-energy 501 --emin 0 --emax 0.12 --eta-mev 1,2 \
  --pair central_flat:-1,0 \
  --component 'x;yy' --component 'x;xx' --component 'y;yx' --component 'y;xy' \
  --component 'y;xx' --component 'y;yy' --component 'x;xy' --component 'x;yx' \
  --output-dir "${RUN_ROOT}/central_flat_allc"
for eta in 1 2; do
  for angle in 0 5 10 15 20 30; do
    python -m analysis.shift_current_htg.plot_bandpair_spectra \
      --input "${RUN_ROOT}/central_flat_allc/htg_bandpair_spectra.npz" \
      --summary "${RUN_ROOT}/central_flat_allc/summary.json" \
      --group central_flat --eta-mev "${eta}" --xmax 0.12 --reflect-y --rotation-deg "${angle}" \
      --output "${RUN_ROOT}/central_flat_allc/central_shell5_m16_noc3_eta${eta}_rot${angle}_reflect_y.png"
  done
done
echo "[done] $(date) run_root=${RUN_ROOT}"
