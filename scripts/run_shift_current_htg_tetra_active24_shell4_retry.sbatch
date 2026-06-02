#!/bin/bash
#SBATCH --job-name=htg_tetra_act24_s4
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
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

root="results/shift_current_htg/mao_retry_tetra_active24_shell4_m12"
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra_tetra \
  --theta-deg 1.95 --stacking ABA --domain h --r 0.8 --mass-mev 30 \
  --n-shells 4 --mesh-size 12 --c3-symmetrize-grid \
  --eta-mev 1,2 --emin 0 --emax 0.12 --n-energy 601 --energy-bin-width-mev 0.1 \
  --pair-window active24:-12,11 \
  --output-dir "$root"

for eta in 1 2; do
  PYTHONPATH=src python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "$root/htg_bandpair_spectra.npz" --summary "$root/summary.json" \
    --group active24 --eta-mev ${eta} --components 'x;yy' 'y;xx' --xmax 0.12 \
    --output "$root/active24_eta${eta}_code_axes.png"
  PYTHONPATH=src python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "$root/htg_bandpair_spectra.npz" --summary "$root/summary.json" \
    --group active24 --eta-mev ${eta} --components 'x;yy' 'y;xx' --xmax 0.12 \
    --reflect-y --rotation-deg 4.8 \
    --output "$root/active24_eta${eta}_rot4p8_reflect_y.png"
done
PYTHONPATH=src python -m analysis.shift_current_htg.analyze_bandpair_tensor_symmetry \
  --input "$root/htg_bandpair_spectra.npz" --summary "$root/summary.json" \
  --group active24 --eta-mev 2 \
  --output-dir "$root/tensor_symmetry_audit_eta2"
