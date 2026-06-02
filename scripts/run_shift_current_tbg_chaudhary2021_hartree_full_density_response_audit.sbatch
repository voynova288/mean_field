#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_fullrho_resp
#SBATCH --account=hmt03
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.out
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /data/home/ziyuzhu/Mean_Field

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1} pwd=$PWD"

ROOT="results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K"
FILLINGS="-3.95,-2,0,2,3.95"
# Interior mu values are the converged full_fixed_cnp SCF chemical potentials from the density-mode audit;
# edge labels keep the same gap-mu diagnostic convention used in current Fig. 4 comparisons.
MU_MEV="-45,-14.878,0,13.805,45"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
  --hartree-dir "${ROOT}/full_fixed_cnp" \
  --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
  --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --output-dir "${ROOT}/full_fixed_cnp_response_edgegap_m10"

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key
root=Path('results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K')
resp=root/'full_fixed_cnp_response_edgegap_m10'
s=json.loads((resp/'summary.json').read_text())
npz=np.load(resp/'spectra_histograms.npz')
e=npz['photon_energies_ev']*1e3
def key(x): return f'{x:g}'.replace('-','m').replace('.','p')
fills=[-3.95,-2.0,0.0,2.0,3.95]
lines=['','## Full-density response audit (m10, edge-gap mu)', '', '| filling | group | peak 0--100 meV |', '|---:|---|---:|']
for fill in fills:
    fkey=key(float(fill))
    for group in ['FF','FD']:
        arr=npz[f'spectrum_nu={fkey}_{group}_y_xx']
        mask=(e>=0)&(e<=100)
        idx=np.where(mask)[0][np.argmax(np.abs(arr[mask]))]
        lines.append(f'| {float(fill):g} | {group} | {float(arr[idx]):.0f}@{float(e[idx]):.1f} |')
md=root/'density_mode_audit.md'
old=md.read_text()
md.write_text(old+'\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
