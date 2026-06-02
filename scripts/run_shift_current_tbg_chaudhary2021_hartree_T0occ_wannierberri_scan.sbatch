#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_T0occ_wb
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
echo "threads OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMBA=$NUMBA_NUM_THREADS"

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_states_paperfill_lg7_m9_eps15_T15K_firststar_linearpm"
BASE="results/shift_current_tbg"

SUM_OUT="${BASE}/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15states_T0occ_firststar_linearpm_sumrule"
if [[ -f "${SUM_OUT}/summary.json" && -f "${SUM_OUT}/spectra_histograms.npz" ]]; then
  echo "skip existing sum_rule T0occ: ${SUM_OUT}"
else
  echo "run sum_rule T0 response occupation"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
    --hartree-dir "${HARTREE_DIR}" \
    --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
    --filling="${FILLINGS}" \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 0 \
    --response-formula sum_rule \
    --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
    --fd-bands 10 --fd-mode same_side --component 'y;xx' \
    --output-dir "${SUM_OUT}"
fi

for SCETA in 5 10 20 40; do
  WB_OUT="${BASE}/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15states_T0occ_firststar_linearpm_wannierberri_sceta${SCETA}"
  if [[ -f "${WB_OUT}/summary.json" && -f "${WB_OUT}/spectra_histograms.npz" ]]; then
    echo "skip existing WB T0occ sc_eta=${SCETA}: ${WB_OUT}"
  else
    echo "run WB T0 response occupation sc_eta=${SCETA}"
    PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
      --hartree-dir "${HARTREE_DIR}" \
      --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
      --filling="${FILLINGS}" \
      --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
      --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 0 \
      --response-formula wannierberri --sc-eta-mev "${SCETA}" \
      --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
      --fd-bands 10 --fd-mode same_side --component 'y;xx' \
      --output-dir "${WB_OUT}"
  fi
done

PYTHONPATH=src python - <<'PY'
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key

root = Path('results/shift_current_tbg')
out = root / 'chaudhary2021_paperstyle_comparison' / 'hartree_T0occ_wannierberri_scan_summary.md'
fillings = [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]

def y(data, nu, group):
    key = f'nu={nu:g}|{group}|y;xx'
    return -np.asarray(data[f'spectrum_{_safe_key(key)}'], dtype=float)

def peak(path, group, lo, hi):
    data = np.load(path)
    e = np.asarray(data['photon_energies_ev'], dtype=float) * 1e3
    idxs = np.where((e >= lo) & (e <= hi))[0]
    best = None
    for nu in fillings:
        arr = y(data, nu, group)
        idx = idxs[int(np.argmax(np.abs(arr[idxs])))]
        item = (float(arr[idx]), float(e[idx]), float(nu))
        if best is None or abs(item[0]) > abs(best[0]):
            best = item
    return best

runs = [
    ('T15occ sum_rule firststar linearpm', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm'/'spectra_histograms.npz'),
    ('T0occ sum_rule firststar linearpm', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15states_T0occ_firststar_linearpm_sumrule'/'spectra_histograms.npz'),
]
for sc in [5,10,20,40]:
    runs.append((f'T15occ WB sc_eta={sc:g}', root/f'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta{sc}'/'spectra_histograms.npz'))
    runs.append((f'T0occ WB sc_eta={sc:g}', root/f'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15states_T0occ_firststar_linearpm_wannierberri_sceta{sc}'/'spectra_histograms.npz'))

lines = [
    '# Hartree response occupation-temperature scan on first-star/linearpm states',
    '',
    'The paper states self-consistent calculations at T=0 K.  This diagnostic keeps the converged T=15 K first-star/linearpm Hartree potentials fixed and changes only the response Fermi occupation from 15 K to 0 K.',
    'Paper-style global sign alignment is applied only for the table.',
    '',
    '| run | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV |',
    '|---|---:|---:|---:|',
]
for label, path in runs:
    if not path.exists():
        lines.append(f'| {label} | missing | missing | missing |')
        continue
    ff = peak(path, 'FF', 3, 40)
    fdl = peak(path, 'FD', 0, 40)
    fdh = peak(path, 'FD', 45, 80)
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) |')
lines += ['', 'Interpretation checkpoint: if T0 response occupation recovers the 60 meV Fig. 4(c) feature without reviving the low-energy pathology, then the remaining convention is thermal/edge occupation rather than Hartree density source.']
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
PY

echo "completed $(date)"
