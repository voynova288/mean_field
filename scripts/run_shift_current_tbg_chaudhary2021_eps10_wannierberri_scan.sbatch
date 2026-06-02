#!/bin/bash
#SBATCH --job-name=tbg_chau21_eps10_wb_scan
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
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
HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps10_T15K"
SCETAS=(5 10 20 40)

if [[ ! -f "${HARTREE_DIR}/summary.json" ]]; then
  echo "missing Hartree source directory: ${HARTREE_DIR}" >&2
  exit 2
fi

for SCETA in "${SCETAS[@]}"; do
  OUT="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15K_wannierberri_sceta${SCETA}"
  if [[ -f "${OUT}/summary.json" && -f "${OUT}/spectra_histograms.npz" ]]; then
    echo "skip existing eps10 T15occ WB sc_eta=${SCETA}: ${OUT}"
  else
    echo "run eps10 T15occ WB sc_eta=${SCETA}"
    PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
      --hartree-dir "${HARTREE_DIR}" \
      --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
      --filling="${FILLINGS}" \
      --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
      --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
      --response-formula wannierberri --sc-eta-mev "${SCETA}" \
      --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
      --fd-bands 10 --fd-mode same_side --component 'y;xx' \
      --output-dir "${OUT}"
  fi
done

# A small response-occupation check on the best-looking regularization range.
for SCETA in 10 20; do
  OUT="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15states_T0occ_wannierberri_sceta${SCETA}"
  if [[ -f "${OUT}/summary.json" && -f "${OUT}/spectra_histograms.npz" ]]; then
    echo "skip existing eps10 T0occ WB sc_eta=${SCETA}: ${OUT}"
  else
    echo "run eps10 T0occ WB sc_eta=${SCETA}"
    PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
      --hartree-dir "${HARTREE_DIR}" \
      --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
      --filling="${FILLINGS}" \
      --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
      --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 0 \
      --response-formula wannierberri --sc-eta-mev "${SCETA}" \
      --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
      --fd-bands 10 --fd-mode same_side --component 'y;xx' \
      --output-dir "${OUT}"
  fi
done

# Paper-style side-by-side plots for the most relevant sc_eta values.
for SCETA in 10 20; do
  NONINT="results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta${SCETA}"
  HARTREE="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15K_wannierberri_sceta${SCETA}"
  PLOT_OUT="results/shift_current_tbg/chaudhary2021_paperstyle_comparison_eps10_wannierberri_sceta${SCETA}"
  if [[ -f "${NONINT}/chaudhary2021_b0_noninteracting.npz" && -f "${HARTREE}/spectra_histograms.npz" ]]; then
    PYTHONPATH=src python -m analysis.shift_current_tbg.plot_chaudhary2021_paperstyle_comparison \
      --nonint-dir "${NONINT}" \
      --hartree-dir "${HARTREE}" \
      --output-dir "${PLOT_OUT}" \
      --our-sign -1
  else
    echo "skip paper-style plot for sc_eta=${SCETA}; missing ${NONINT} or ${HARTREE}"
  fi
done

PYTHONPATH=src python - <<'PY'
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key

root = Path('results/shift_current_tbg')
out = root / 'chaudhary2021_paperstyle_comparison' / 'eps10_wannierberri_sceta_scan_summary.md'
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
    ('paper visual target, Fig1_v3', None, None, 'FF_H≈9e3-1e4; FD_low≈1.1e4-1.2e4; FD_high≈7e3-8e3; FD_high/low≈0.6-0.7'),
    ('sum_rule Hartree eps10 all-shift T15occ', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15K'/'spectra_histograms.npz', 'spectrum', ''),
    ('sum_rule Hartree eps15 firststar linearpm T15occ', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm'/'spectra_histograms.npz', 'spectrum', ''),
]
for sc in [5, 10, 20, 40]:
    runs.append((f'WB sc_eta={sc:g} Hartree eps10 all-shift T15occ', root/f'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15K_wannierberri_sceta{sc}'/'spectra_histograms.npz', 'spectrum', ''))
for sc in [10, 20]:
    runs.append((f'WB sc_eta={sc:g} Hartree eps10 all-shift T0occ-from-T15states', root/f'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps10_T15states_T0occ_wannierberri_sceta{sc}'/'spectra_histograms.npz', 'spectrum', ''))

lines = [
    '# Epsilon=10 all-shift WannierBerri sc_eta scan',
    '',
    'Paper-style global sign alignment (`-1`) is applied only for this table.',
    'This scan tests whether the Hartree setting that best matched the FF amplitude (epsilon=10, all Hartree shifts) plus WannierBerri principal-value regularization can improve the FD high/low hierarchy.',
    '',
    '| run | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV | |high/low| | note |',
    '|---|---:|---:|---:|---:|---|',
]
for label, path, prefix, note in runs:
    if path is None:
        lines.append(f'| {label} | target | target | target | ~0.6-0.7 | {note} |')
        continue
    if not path.exists():
        lines.append(f'| {label} | missing | missing | missing | missing | {note} |')
        continue
    ff = peak(path, prefix, 'FF', 3, 40)
    fdl = peak(path, prefix, 'FD', 0, 40)
    fdh = peak(path, prefix, 'FD', 45, 80)
    ratio = abs(fdh[0]) / max(abs(fdl[0]), 1e-30)
    lines.append(
        f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | '
        f'{fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | '
        f'{fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) | {ratio:.3f} | {note} |'
    )
lines += [
    '',
    'Interpretation gate: a genuinely improved Fig. 4(c) candidate should keep the Hartree FF peak near the paper scale while raising the FD high/low ratio toward ~0.6-0.7 without creating a new low-energy artifact.',
]
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
PY

echo "completed $(date)"
