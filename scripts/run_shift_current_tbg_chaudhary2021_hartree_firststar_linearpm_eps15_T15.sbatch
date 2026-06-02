#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_fs15_linpm
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

FILLINGS="-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95"
SRC="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar"
LIN="results/shift_current_tbg/chaudhary2021_hartree_states_paperfill_lg7_m9_eps15_T15K_firststar_linearpm"
RESP="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm"

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.hartree import arrays_to_rho, rho_to_arrays
src=Path('results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar')
out=Path('results/shift_current_tbg/chaudhary2021_hartree_states_paperfill_lg7_m9_eps15_T15K_firststar_linearpm')
out.mkdir(parents=True, exist_ok=True)
summary=json.loads((src/'summary.json').read_text())
def state_name(nu): return f"hartree_state_nu_{f'{nu:g}'.replace('-','m').replace('.','p')}.npz"
def load_rho(nu):
    d=np.load(src/state_name(nu), allow_pickle=False)
    return arrays_to_rho(d['rho_shifts'], d['rho_values'])
rp=load_rho(3.95); rm=load_rho(-3.95); keys=set(rp)|set(rm)
edge=3.95
fillings=[-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]
res={}
for nu in fillings:
    rho={}
    for k in keys:
        # Ref. 65 Eq. (17)-style two-ended linear interpolation; because rho(+edge)+rho(-edge)
        # is nearly zero this is almost the same as an odd CNP-anchored interpolation.
        val=0.5*((rp.get(k,0)+rm.get(k,0)) + (nu/edge)*(rp.get(k,0)-rm.get(k,0)))
        if abs(val)>1e-14:
            rho[k]=val
    shifts, vals=rho_to_arrays(rho)
    src_key=f'nu={nu:g}'
    mu=float(summary['results'][src_key].get('mu_ev',0.0)) if src_key in summary['results'] else 0.0
    np.savez_compressed(
        out/state_name(nu),
        rho_shifts=shifts,
        rho_values=vals,
        iter_error=np.asarray([], dtype=float),
        iter_mu_ev=np.asarray([mu], dtype=float),
        density_mode=np.asarray('flat'),
        hartree_shift_mode=np.asarray('first_star_linearpm'),
        path_kdist=np.asarray([], dtype=float),
        path_energies_ev=np.asarray([], dtype=float),
    )
    res[src_key]={
        'nu': nu,
        'rho_q_count': len(rho),
        'max_abs_rho_q': max((abs(v) for v in rho.values()), default=0.0),
        'mu_ev': mu,
        'source': 'linear interpolation of first-star rho at nu=+/-3.95',
    }
meta={
    'status': 'Ref65-style linear interpolation diagnostic from first-star Hartree edge states; response-only state directory.',
    'source_dir': str(src),
    'edge_nu': edge,
    'results': res,
    'run': {**summary.get('run',{}), 'hartree_shift_mode':'first_star_linearpm'},
    'config': summary.get('config',{}),
}
(out/'summary.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
print(json.dumps(meta['run'], indent=2, sort_keys=True))
PY

MU_MEV=$(PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar/summary.json').read_text())
fills=[-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]
print(','.join(f"{1e3*float(s['results'][f'nu={nu:g}'].get('mu_ev',0.0)):.9g}" for nu in fills))
PY
)

echo "linearpm response mu_mev=${MU_MEV}"
PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
  --hartree-dir "${LIN}" \
  --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
  --filling="${FILLINGS}" --mu-mev="${MU_MEV}" \
  --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
  --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 15 \
  --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
  --fd-bands 10 --fd-mode same_side --component 'y;xx' \
  --output-dir "${RESP}"

PYTHONPATH=src python - <<'PY'
from pathlib import Path
import numpy as np

def skey(x): return f'{x:g}'.replace('-','m').replace('.','p')
def peak(resp, fill, group, lo, hi):
    npz=np.load(resp/'spectra_histograms.npz')
    e=npz['photon_energies_ev']*1e3
    arr=-npz[f'spectrum_nu={skey(fill)}_{group}_y_xx']
    mask=(e>=lo)&(e<=hi)
    idx=np.where(mask)[0][np.argmax(np.abs(arr[mask]))]
    return float(arr[idx]), float(e[idx]), float(fill)
root=Path('results/shift_current_tbg')
resp_sc=root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar'
resp_lin=root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm'
fills=[-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]
lines=['', '## First-star linear-from-edge diagnostic', '', '| mode | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV |', '|---|---:|---:|---:|']
for label,resp in [('first_star_self_consistent_each_nu',resp_sc),('first_star_linearpm_from_edges',resp_lin)]:
    ff=max((peak(resp,nu,'FF',3,40) for nu in fills), key=lambda x: abs(x[0]))
    fdl=max((peak(resp,nu,'FD',0,40) for nu in fills), key=lambda x: abs(x[0]))
    fdh=max((peak(resp,nu,'FD',45,80) for nu in fills), key=lambda x: abs(x[0]))
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) |')
md=root/'chaudhary2021_paperstyle_comparison'/'hartree_firststar_eps15_audit.md'
old=md.read_text() if md.exists() else '# Hartree first-star audit\n'
if '## First-star linear-from-edge diagnostic' not in old:
    md.write_text(old+'\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
