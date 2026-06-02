#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_firststar15
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

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
HARTREE_DIR="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar"
RESP_DIR="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
  --theta-deg 0.8 --lg 7 --mesh-size 9 --path-points-per-segment 100 \
  --nus="${FILLINGS}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 15 --temperature-k 15 --mixing 0.35 --precision 1e-5 --max-iter 80 \
  --density-mode flat --hartree-shift-mode first_star \
  --bands-each-side 5 \
  --output-dir "${HARTREE_DIR}"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
  --hartree-dir "${HARTREE_DIR}" \
  --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
  --filling="${FILLINGS}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --output-dir "${RESP_DIR}"

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np

def skey(x): return f'{x:g}'.replace('-','m').replace('.','p')
def peak(resp, fill, group, lo, hi):
    npz=np.load(resp/'spectra_histograms.npz')
    e=npz['photon_energies_ev']*1e3
    arr=-npz[f'spectrum_nu={skey(fill)}_{group}_y_xx']  # paper-style sign alignment used in epsilon scan note
    mask=(e>=lo)&(e<=hi)
    idx=np.where(mask)[0][np.argmax(np.abs(arr[mask]))]
    return float(arr[idx]), float(e[idx]), float(fill)
root=Path('results/shift_current_tbg')
all_band=root/'chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K'
fs_band=root/'chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar'
all_resp=root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K'
fs_resp=root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar'
sa=json.loads((all_band/'summary.json').read_text())
sf=json.loads((fs_band/'summary.json').read_text())
lines=['# Hartree first-star audit (Ref. 65 convention)', '', 'Choi Ref. 65 supplement keeps only first-star Hartree Fourier components and uses epsilon=15.  This compares our previous all-shift eps15 run with first-star-only eps15.', '', '## Band metrics', '', '| nu | all-shift BW lower/upper | first-star BW lower/upper | all mu (meV) | first-star mu (meV) | first-star rho count |', '|---:|---:|---:|---:|---:|---:|']
for nu in [-3.95,-2.0,0.0,2.0,3.95]:
    key=f'nu={nu:g}'
    ra=sa['results'][key]; rf=sf['results'][key]
    bwa=ra.get('central_bandwidths_mev',[float('nan'),float('nan')]); bwf=rf.get('central_bandwidths_mev',[float('nan'),float('nan')])
    lines.append(f"| {nu:g} | {bwa[0]:.2f}/{bwa[1]:.2f} | {bwf[0]:.2f}/{bwf[1]:.2f} | {1e3*float(ra.get('mu_ev',0)):.2f} | {1e3*float(rf.get('mu_ev',0)):.2f} | {int(rf.get('rho_q_count',0))} |")
lines += ['', '## Response peak windows (paper-style sign flip)', '', '| mode | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV |', '|---|---:|---:|---:|']
for label,resp in [('all_shift_eps15',all_resp),('first_star_eps15',fs_resp)]:
    ff=max((peak(resp,nu,'FF',3,40) for nu in [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]), key=lambda x: abs(x[0]))
    fdl=max((peak(resp,nu,'FD',0,40) for nu in [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]), key=lambda x: abs(x[0]))
    fdh=max((peak(resp,nu,'FD',45,80) for nu in [-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]), key=lambda x: abs(x[0]))
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) |')
lines += ['', 'Interpretation: first-star is the documented Ref. 65 Hartree truncation.  Use this audit to decide whether previous all-shift Hartree harmonics were overfitting/overdistorting outer-BZ FD geometry.']
out=root/'chaudhary2021_paperstyle_comparison'/'hartree_firststar_eps15_audit.md'
out.write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
