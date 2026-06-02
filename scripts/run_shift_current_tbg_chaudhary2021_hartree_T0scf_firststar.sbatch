#!/bin/bash
#SBATCH --job-name=tbg_chau21_T0scf_fs
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
INIT="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T15K_firststar"
BANDS="results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T0K_firststar_fromT15"
LIN="results/shift_current_tbg/chaudhary2021_hartree_states_paperfill_lg7_m9_eps15_T0K_firststar_linearpm"
RESP_SUM="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T0K_firststar_linearpm_sumrule"
RESP_WB10="results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T0K_firststar_linearpm_wannierberri_sceta10"

if [[ ! -f "${INIT}/summary.json" ]]; then
  echo "missing finite-T initial states: ${INIT}" >&2
  exit 2
fi

if [[ -f "${BANDS}/summary.json" ]]; then
  echo "skip existing T=0 SCF bands: ${BANDS}"
else
  echo "run true T=0 first-star SCF initialized from T=15 K first-star states"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
    --theta-deg 0.8 --lg 7 --mesh-size 9 --path-points-per-segment 100 \
    --nus="${FILLINGS}" \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r 15 --temperature-k 0 --mixing 0.20 --precision 1e-5 --max-iter 120 \
    --density-mode flat --hartree-shift-mode first_star --initial-state-dir "${INIT}" \
    --bands-each-side 5 \
    --output-dir "${BANDS}"
fi

if [[ -f "${LIN}/summary.json" ]]; then
  echo "skip existing T=0 first-star linearpm states: ${LIN}"
else
  echo "construct Ref65-style linearpm states from T=0 edge SCF states"
  PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import numpy as np
from analysis.shift_current_tbg.hartree import arrays_to_rho, rho_to_arrays
src=Path('results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T0K_firststar_fromT15')
out=Path('results/shift_current_tbg/chaudhary2021_hartree_states_paperfill_lg7_m9_eps15_T0K_firststar_linearpm')
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
        val=0.5*((rp.get(k,0)+rm.get(k,0)) + (nu/edge)*(rp.get(k,0)-rm.get(k,0)))
        if abs(val)>1e-14:
            rho[k]=val
    shifts, vals=rho_to_arrays(rho)
    src_key=f'nu={nu:g}'
    mu=float(summary['results'][src_key].get('mu_ev',0.0)) if src_key in summary['results'] else 0.0
    # Store the true T=0 SCF mu for this filling even though the rho is linearpm;
    # this isolates potential interpolation from Pauli-blocking convention.
    np.savez_compressed(
        out/state_name(nu),
        rho_shifts=shifts,
        rho_values=vals,
        iter_error=np.asarray([], dtype=float),
        iter_mu_ev=np.asarray([mu], dtype=float),
        density_mode=np.asarray('flat'),
        hartree_shift_mode=np.asarray('first_star_linearpm_T0edge'),
        path_kdist=np.asarray([], dtype=float),
        path_energies_ev=np.asarray([], dtype=float),
    )
    res[src_key]={
        'nu': nu,
        'rho_q_count': len(rho),
        'max_abs_rho_q': max((abs(v) for v in rho.values()), default=0.0),
        'mu_ev': mu,
        'source': 'linear interpolation of first-star T=0 edge rho at nu=+/-3.95; mu from true T=0 SCF at same nu',
    }
meta={
    'status': 'Ref65-style linear interpolation diagnostic from first-star true-T0 edge states; response-only state directory.',
    'source_dir': str(src),
    'edge_nu': edge,
    'results': res,
    'run': {**summary.get('run',{}), 'hartree_shift_mode':'first_star_linearpm_T0edge'},
    'config': summary.get('config',{}),
}
(out/'summary.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
print(json.dumps(meta['run'], indent=2, sort_keys=True))
PY
fi

if [[ -f "${RESP_SUM}/summary.json" && -f "${RESP_SUM}/spectra_histograms.npz" ]]; then
  echo "skip existing T=0 firststar linearpm sum_rule response: ${RESP_SUM}"
else
  echo "run T=0 firststar linearpm sum_rule response"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
    --hartree-dir "${LIN}" \
    --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
    --filling="${FILLINGS}" \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 0 \
    --response-formula sum_rule \
    --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
    --fd-bands 10 --fd-mode same_side --component 'y;xx' \
    --output-dir "${RESP_SUM}"
fi

if [[ -f "${RESP_WB10}/summary.json" && -f "${RESP_WB10}/spectra_histograms.npz" ]]; then
  echo "skip existing T=0 firststar linearpm WB10 response: ${RESP_WB10}"
else
  echo "run T=0 firststar linearpm WannierBerri sc_eta=10 response"
  PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_shift_current \
    --hartree-dir "${LIN}" \
    --theta-deg 0.8 --lg 7 --mesh-size 10 --c3-symmetrize-grid \
    --filling="${FILLINGS}" \
    --delta1-mev 5 --delta2-mev 5 --w-ab-mev 90 --w-aa-ratio 0.4 --kinetic-ev 2.1354 \
    --epsilon-r 15 --degeneracy 1 --eta-mev 2 --occupation-temperature-k 0 \
    --response-formula wannierberri --sc-eta-mev 10 \
    --emin 0 --emax 0.16 --n-energy 641 --energy-bin-width-mev 0.25 \
    --fd-bands 10 --fd-mode same_side --component 'y;xx' \
    --output-dir "${RESP_WB10}"
fi

PYTHONPATH=src python - <<'PY'
from pathlib import Path
import json
import numpy as np
from analysis.shift_current_tbg.run_chaudhary2021_noninteracting import _safe_key
root=Path('results/shift_current_tbg')
out=root/'chaudhary2021_paperstyle_comparison'/'hartree_T0scf_firststar_summary.md'
fills=[-3.95,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,3.95]
def arr(data,prefix,nu,group):
    key=f'nu={nu:g}|{group}|y;xx'
    return -np.asarray(data[f'{prefix}_{_safe_key(key)}'], dtype=float)
def peak(npz_path,prefix,group,lo,hi):
    data=np.load(npz_path)
    e=np.asarray(data['photon_energies_ev'], dtype=float)*1e3
    idxs=np.where((e>=lo)&(e<=hi))[0]
    best=None
    for nu in fills:
        y=arr(data,prefix,nu,group)
        idx=idxs[int(np.argmax(np.abs(y[idxs])))]
        item=(float(y[idx]), float(e[idx]), float(nu))
        if best is None or abs(item[0])>abs(best[0]): best=item
    return best
runs=[
    ('T15 firststar linearpm sum_rule', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm'/'spectra_histograms.npz', 'spectrum'),
    ('T15 firststar linearpm WB sc_eta=10', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta10'/'spectra_histograms.npz', 'spectrum'),
    ('true T0-SCF edge firststar linearpm sum_rule T0occ', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T0K_firststar_linearpm_sumrule'/'spectra_histograms.npz', 'spectrum'),
    ('true T0-SCF edge firststar linearpm WB sc_eta=10 T0occ', root/'chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T0K_firststar_linearpm_wannierberri_sceta10'/'spectra_histograms.npz', 'spectrum'),
]
lines=['# True T=0 SCF first-star Hartree diagnostic', '', 'Paper-style sign alignment (`-1`) is used only for this table.', '', '| run | FF max 3-40 meV | FD low max 0-40 meV | FD high max 45-80 meV | |high/low| |', '|---|---:|---:|---:|---:|']
for label,path,prefix in runs:
    if not path.exists():
        lines.append(f'| {label} | missing | missing | missing | missing |')
        continue
    ff=peak(path,prefix,'FF',3,40); fdl=peak(path,prefix,'FD',0,40); fdh=peak(path,prefix,'FD',45,80)
    ratio=abs(fdh[0])/max(abs(fdl[0]),1e-30)
    lines.append(f'| {label} | {ff[0]:.0f} (nu={ff[2]:g}, {ff[1]:.1f} meV) | {fdl[0]:.0f} (nu={fdl[2]:g}, {fdl[1]:.1f} meV) | {fdh[0]:.0f} (nu={fdh[2]:g}, {fdh[1]:.1f} meV) | {ratio:.3f} |')
# convergence note
bands=root/'chaudhary2021_hartree_bands_paperfill_lg7_m9_eps15_T0K_firststar_fromT15'/'summary.json'
if bands.exists():
    s=json.loads(bands.read_text())
    bad=[]
    for k,v in sorted(s.get('results',{}).items()):
        if not v.get('converged',True):
            bad.append(f"{k}: err={v.get('final_error')}")
    lines += ['', 'SCF convergence:', ('all fillings converged to requested precision' if not bad else '; '.join(bad))]
out.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
PY

echo "completed $(date)"
