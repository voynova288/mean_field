#!/bin/bash
#SBATCH --job-name=tbg_chau21_wb40_plot
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"

NONINT="results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta40"
HARTREE="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta40"
OUT="results/shift_current_tbg/chaudhary2021_paperstyle_comparison_wannierberri_sceta40"

PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_paperstyle_comparison \
  --nonint-dir "${NONINT}" \
  --hartree-dir "${HARTREE}" \
  --output-dir "${OUT}" \
  --our-sign -1

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key

root = Path('results/shift_current_tbg')
nonint = root / 'chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta40'
hartree = root / 'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta40'
out = root / 'chaudhary2021_paperstyle_comparison_wannierberri_sceta40' / 'peak_window_summary.md'
fillings = [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]

def spectrum(npz, prefix, nu, group):
    key = f'nu={nu:g}|{group}|y;xx'
    return -np.asarray(npz[f'{prefix}_{_safe_key(key)}'], dtype=float)  # paper-style sign alignment

def peak(path, prefix, nu, group, lo, hi):
    data = np.load(path)
    e = np.asarray(data['photon_energies_ev'], dtype=float) * 1e3
    y = spectrum(data, prefix, nu, group)
    mask = (e >= lo) & (e <= hi)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None
    idx = idxs[int(np.argmax(np.abs(y[idxs])))]
    return float(y[idx]), float(e[idx]), float(nu)

rows = []
for label, directory, filename, prefix in [
    ('nonint_wb40', nonint, 'chaudhary2021_b0_noninteracting.npz', 'sigma'),
    ('hartree_firststar_linearpm_wb40', hartree, 'spectra_histograms.npz', 'spectrum'),
]:
    npz_path = directory / filename
    ff = max((peak(npz_path, prefix, nu, 'FF', 3, 40) for nu in fillings), key=lambda x: abs(x[0]))
    fdl = max((peak(npz_path, prefix, nu, 'FD', 0, 40) for nu in fillings), key=lambda x: abs(x[0]))
    fdh = max((peak(npz_path, prefix, nu, 'FD', 45, 80) for nu in fillings), key=lambda x: abs(x[0]))
    rows.append((label, ff, fdl, fdh))
lines = [
    '# WannierBerri sc_eta=40 paper-grid peak summary',
    '',
    'All values include the diagnostic global sign alignment used in the paper-style comparison plot.',
    '',
    '| run | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV |',
    '|---|---:|---:|---:|',
]
for label, ff, fdl, fdh in rows:
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) |')
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
PY

echo "completed $(date) output=${OUT}"
