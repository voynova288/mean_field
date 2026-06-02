#!/bin/bash
#SBATCH --job-name=sc_c3ang
#SBATCH --account=hmt03
#SBATCH --partition=regular6430
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/shift_current_htg_c3ang_%j.out
#SBATCH --error=logs/shift_current_htg_c3ang_%j.err

set -euo pipefail
cd /data/home/ziyuzhu/Mean_Field
mkdir -p logs
export PYTHONPATH=src
ROOT=results/shift_current_htg/active24_allcomponents_c3_nshell3_132114/active24_allc
INPUT="${ROOT}/htg_bandpair_spectra.npz"
SUMMARY="${ROOT}/summary.json"
for angle in 5 10 15 20 25; do
  python -m analysis.shift_current_htg.plot_bandpair_spectra \
    --input "${INPUT}" --summary "${SUMMARY}" --group active24 --eta-mev 2 \
    --xmax 0.12 --rotation-deg "${angle}" --reflect-y \
    --output "${ROOT}/active24_c3_eta2meV_rot${angle}_reflect_y_thz.png"
done
