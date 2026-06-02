#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_deltaocc
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

ROOT="results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K"
NUS="-3.95,-2,0,2,3.95"
COMMON=(
  --theta-deg 0.8 --lg 7 --mesh-size 5 --path-points-per-segment 40
  --nus="${NUS}"
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354
  --epsilon-r 10 --temperature-k 15 --precision 1e-5 --bands-each-side 5
)

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
  "${COMMON[@]}" --density-mode full_delta_occ --mixing 0.25 --max-iter 70 \
  --output-dir "${ROOT}/full_delta_occ"

MU_MEV=$(PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K/full_delta_occ/summary.json').read_text())
vals=[]
for nu in [-3.95,-2.0,0.0,2.0,3.95]:
    if abs(nu)==3.95:
        vals.append(-45.0 if nu < 0 else 45.0)
    else:
        vals.append(1e3*float(s['results'][f'nu={nu:g}'].get('mu_ev',0.0)))
print(','.join(f'{v:.9g}' for v in vals))
PY
)
echo "delta-occ response mu_mev=${MU_MEV}"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
  --hartree-dir "${ROOT}/full_delta_occ" \
  --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
  --filling="${NUS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 10 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --output-dir "${ROOT}/full_delta_occ_response_edgegap_m10"

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np

def kfill(x): return f'nu={x:g}'
def skey(x): return f'{x:g}'.replace('-','m').replace('.','p')
def load_json(path): return json.loads(Path(path).read_text())
def peak(root, mode, fill, group):
    npz=np.load(root/f'{mode}_response_edgegap_m10'/'spectra_histograms.npz')
    e=npz['photon_energies_ev']*1e3
    arr=npz[f'spectrum_nu={skey(fill)}_{group}_y_xx']
    mask=(e>=0)&(e<=100)
    idx=np.where(mask)[0][np.argmax(np.abs(arr[mask]))]
    return float(arr[idx]), float(e[idx])
root=Path('results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K')
modes=['flat','full_delta_occ','full_fixed_cnp']
summ={m: load_json(root/m/'summary.json') for m in modes}
lines=[
    '# Hartree density-mode audit: three source conventions',
    '',
    'Purpose: compare three interpretations of density relative to charge neutrality:',
    '',
    '- `flat`: current two-flat-band occupation difference; dispersive screening folded into scalar epsilon.',
    '- `full_delta_occ`: full continuum eigenbasis, but subtract same-Hamiltonian CNP occupation so filled remote sea cancels; tests whether carriers leak outside the central two bands.',
    '- `full_fixed_cnp`: full occupied density minus fixed noninteracting CNP density; includes explicit remote-band polarization and can double-count screening.',
    '',
    'Settings: lg=7, Hartree mesh m=5, epsilon_r=10, T=15 K; response audit uses m=10, component y;xx, same-side FD, edge-gap mu for |nu|=3.95.',
    '',
    '## SCF / band summary',
    '',
    '| nu | mode | conv | iter | final_err | mu (meV) | BW lower/upper (meV) | max |rho_q| | rho count |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
]
for nu in [-3.95,-2.0,0.0,2.0,3.95]:
    for mode in modes:
        r=summ[mode]['results'][kfill(nu)]
        bw=r.get('central_bandwidths_mev',[float('nan'),float('nan')])
        lines.append(f"| {nu:g} | {mode} | {int(bool(r.get('converged',False)))} | {int(r.get('iterations',0))} | {float(r.get('final_error',0.0)):.3g} | {1e3*float(r.get('mu_ev',0.0)):.3f} | {bw[0]:.2f}/{bw[1]:.2f} | {float(r.get('max_abs_rho_q',0.0)):.3g} | {int(r.get('rho_q_count',0))} |")
lines += ['', '## Response peak comparison', '', '| filling | group | flat peak | full_delta_occ peak | full_fixed_cnp peak |', '|---:|---|---:|---:|---:|']
for fill in [-3.95,-2.0,0.0,2.0,3.95]:
    for group in ['FF','FD']:
        p=[]
        for mode in modes:
            v,e=peak(root,mode,fill,group)
            p.append(f'{v:.0f}@{e:.1f}')
        lines.append(f'| {fill:g} | {group} | {p[0]} | {p[1]} | {p[2]} |')
lines += [
    '',
    'Conclusion: `full_delta_occ` should be close to `flat` if the doped carriers remain in the central flat pair.  A large deviation would indicate carrier leakage/band-index convention problems.  `full_fixed_cnp` separately probes explicit remote-sea polarization.',
]
(root/'density_mode_audit_three_modes.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
