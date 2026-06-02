#!/bin/bash
#SBATCH --job-name=tbg_chau21_wb_eta_scan
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
SCETAS=(5 10 20 40)

for SCETA in "${SCETAS[@]}"; do
  NONINT_OUT="results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta${SCETA}"
  HARTREE_OUT="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta${SCETA}"

  if [[ -f "${NONINT_OUT}/summary.json" && -f "${NONINT_OUT}/chaudhary2021_b0_noninteracting.npz" ]]; then
    echo "skip existing nonint sc_eta=${SCETA}: ${NONINT_OUT}"
  else
    echo "run nonint sc_eta=${SCETA}"
    PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_b0_noninteracting \
      --theta-deg 0.8 --lg 7 --mesh-size 16 --mu-mesh-size 16 --c3-symmetrize-grid \
      --filling="${FILLINGS}" \
      --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
      --degeneracy 1 --filling-degeneracy 4 --eta-mev 2 \
      --response-formula wannierberri --sc-eta-mev "${SCETA}" \
      --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
      --fd-bands 10 --fd-mode same_side --component 'y;xx' \
      --path-points-per-segment 100 --band-window 8 \
      --output-dir "${NONINT_OUT}"
  fi

  if [[ -f "${HARTREE_OUT}/summary.json" && -f "${HARTREE_OUT}/spectra_histograms.npz" ]]; then
    echo "skip existing hartree sc_eta=${SCETA}: ${HARTREE_OUT}"
  else
    echo "run hartree sc_eta=${SCETA}"
    PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
      --hartree-dir "${HARTREE_DIR}" \
      --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
      --filling="${FILLINGS}" \
      --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
      --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
      --response-formula wannierberri --sc-eta-mev "${SCETA}" \
      --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
      --fd-bands 10 --fd-mode same_side --component 'y;xx' \
      --output-dir "${HARTREE_OUT}"
  fi

done

PYTHONPATH=src python - <<'PY'
from pathlib import Path
import json
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key

root = Path('results/shift_current_tbg')
out = root / 'chaudhary2021_paperstyle_comparison' / 'wannierberri_sceta_scan_summary.md'
fillings = [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]

def arr(data, prefix, nu, group):
    key = f'nu={nu:g}|{group}|y;xx'
    return -np.asarray(data[f'{prefix}_{_safe_key(key)}'], dtype=float)

def peak(npz_path, prefix, group, lo, hi):
    data = np.load(npz_path)
    e = np.asarray(data['photon_energies_ev'], dtype=float) * 1e3
    idxs = np.where((e >= lo) & (e <= hi))[0]
    best = None
    for nu in fillings:
        y = arr(data, prefix, nu, group)
        idx = idxs[int(np.argmax(np.abs(y[idxs])))]
        item = (float(y[idx]), float(e[idx]), float(nu))
        if best is None or abs(item[0]) > abs(best[0]):
            best = item
    return best

runs = [
    ('sum_rule nonint', root/'chaudhary2021_b0_nonint_paperfill_lg7_m16_c3'/'chaudhary2021_b0_noninteracting.npz', 'sigma'),
    ('sum_rule Hartree firststar linearpm', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm'/'spectra_histograms.npz', 'spectrum'),
]
for sc in [5, 10, 20, 40]:
    runs.append((f'WB sc_eta={sc:g} nonint', root/f'chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta{sc}'/'chaudhary2021_b0_noninteracting.npz', 'sigma'))
    runs.append((f'WB sc_eta={sc:g} Hartree firststar linearpm', root/f'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta{sc}'/'spectra_histograms.npz', 'spectrum'))

lines = [
    '# WannierBerri sc_eta scan summary',
    '',
    'Paper-style global sign alignment is applied only for the table.  Hartree uses first-star, linear ±4 interpolation states at epsilon=15, T=15 K response occupation.',
    '',
    '| run | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV |',
    '|---|---:|---:|---:|',
]
for label, path, prefix in runs:
    if not path.exists():
        lines.append(f'| {label} | missing | missing | missing |')
        continue
    ff = peak(path, prefix, 'FF', 3, 40)
    fdl = peak(path, prefix, 'FD', 0, 40)
    fdh = peak(path, prefix, 'FD', 45, 80)
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) |')
lines += [
    '',
    'Interpretation checkpoint: decreasing `sc_eta` tests whether the WB principal-value damping is over-suppressing the high-energy FD Hartree structure while still controlling the 26 meV Gamma/doublet feature.',
]
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
PY

echo "completed $(date)"
