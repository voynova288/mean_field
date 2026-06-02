#!/bin/bash
#SBATCH --job-name=sc_rotplot
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/shift_current_htg_rotplot_%j.out
#SBATCH --error=logs/shift_current_htg_rotplot_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export PYTHONPATH=src
ROOT=${ROOT:-results/shift_current_htg/active24_allcomponents_nshell3_132107/active24_allc}
INPUT="${ROOT}/htg_bandpair_spectra.npz"
SUMMARY="${ROOT}/summary.json"
echo "[start] $(date) host=$(hostname) input=${INPUT}"
for angle in -60 -30 0 30 60 90; do
  tag=${angle//-/_m}
  python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "${INPUT}" --summary "${SUMMARY}" --group active24 --eta-mev 2 \
    --xmax 0.12 --rotation-deg "${angle}" --reflect-y \
    --output "${ROOT}/active24_eta2meV_rot${tag}_reflect_y_thz.png"
done
echo "[done] $(date)"
