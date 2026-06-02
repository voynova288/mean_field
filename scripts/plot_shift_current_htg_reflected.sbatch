#!/bin/bash
#SBATCH --job-name=sc_htg_plot
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/shift_current_htg_plot_%j.out
#SBATCH --error=logs/shift_current_htg_plot_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export PYTHONPATH=src
RUN_ROOT=${RUN_ROOT:-results/shift_current_htg/fig1_fig2_smoke_132069}

echo "[start] $(date) host=$(hostname) run_root=${RUN_ROOT}"
python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/central_flat/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/central_flat/summary.json" \
  --group central_flat \
  --eta-mev 2 \
  --xmax 0.12 \
  --reflect-y \
  --output "${RUN_ROOT}/central_flat/central_flat_eta2meV_spectra_reflect_y.png"
python -m analysis.shift_current_htg.plot_bandpair_spectra \
  --input "${RUN_ROOT}/central_window/htg_bandpair_spectra.npz" \
  --summary "${RUN_ROOT}/central_window/summary.json" \
  --group central_window \
  --eta-mev 2 \
  --xmax 0.12 \
  --reflect-y \
  --output "${RUN_ROOT}/central_window/central_window_eta2meV_spectra_reflect_y.png"
echo "[done] $(date)"
