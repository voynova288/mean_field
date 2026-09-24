#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, stat
from pathlib import Path
import numpy as np

ROOT = Path('/data/home/ziyuzhu/Mean_Field')
CAP = ROOT / 'results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25'
FINAL = CAP / 'runtime/output/job_513422'
CHECKPOINT = CAP / 'runtime/checkpoint/full385_v25_job_513422'
S = np.asarray([[1,-1],[1,2]], dtype=np.int64)
REPS = ((0,0),(1,0),(2,0))
AXES = ('electron_charge_e','hole_charge_e','spin_x_hbar','spin_y_hbar','spin_z_hbar')

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def reconstruct(coeff, labels, grid):
    x=(np.arange(grid,dtype=np.float64)+0.5)/grid
    px=np.exp(-2j*np.pi*np.outer(x,labels[:,0]))
    py=np.exp(-2j*np.pi*np.outer(x,labels[:,1]))
    return np.einsum('qi,qj,q->ij',px.T,py.T,coeff,optimize=True)

def partition_labels(primitive_lattice, grid, kind, translated_by=(0,0)):
    axis=(np.arange(grid,dtype=np.float64)+0.5)/grid
    y=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    primitive_fractional=y@S
    translation=np.asarray(translated_by,dtype=np.int64)
    if kind=='parallelogram':
        integer=np.floor(primitive_fractional-translation).astype(np.int64)
        labels=(integer[:,0]+integer[:,1])%3
    elif kind=='wigner_seitz':
        candidates=[]; owners=[]
        for owner,representative in enumerate(REPS):
            center0=np.asarray(representative,dtype=np.int64)+translation
            for m0 in range(-2,3):
                for m1 in range(-2,3):
                    candidates.append(center0+np.asarray((m0,m1))@S)
                    owners.append(owner)
        centers=np.asarray(candidates,dtype=np.float64)
        cart=(primitive_fractional[:,None,:]-centers[None,:,:])@primitive_lattice
        nearest=np.argmin(np.sum(cart*cart,axis=2),axis=1)
        labels=np.asarray(owners,dtype=np.int64)[nearest]
    else: raise ValueError(kind)
    return labels.reshape(grid,grid)

def fields(data, grid):
    area=float(data['area_nm2'])
    labels=np.asarray(data['supercell_labels'],np.int64)
    electron=reconstruct(np.asarray(data['charge_fourier']),labels,grid)/area
    hole=reconstruct(np.asarray(data['hole_charge_fourier']),labels,grid)/area
    pauli=np.asarray([reconstruct(row,labels,grid)/area for row in np.asarray(data['spin_pauli_fourier'])])
    out=np.concatenate((electron.real[None],hole.real[None],pauli.real/2),axis=0)
    imag=max(np.max(np.abs(electron.imag)),np.max(np.abs(hole.imag)),np.max(np.abs(pauli.imag)))
    return out,float(imag)

def integrate(f, labels, area):
    w=area/labels.size
    return np.asarray([[float(np.sum(f[a][labels==r])*w) for r in range(3)] for a in range(5)])

def maxabs(a): return float(np.max(np.abs(a)))
def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.ndarray): return native(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(complex,np.complexfloating)): return [float(x.real),float(x.imag)]
    return x

summary=json.loads((FINAL/'summary.json').read_text())
cfg=json.loads((CAP/'config.json').read_text())
manifest=json.loads((FINAL/'OUTPUT_SHA256.json').read_text())
sentinel=json.loads((FINAL/'COMPLETE').read_text())
cp_manifest=json.loads((CHECKPOINT/'FULL385_DIAGNOSTIC_V25_SHA256.json').read_text())
cp_sentinel=json.loads((CHECKPOINT/'FULL385_DIAGNOSTIC_V25_COMPLETE').read_text())

# Byte/mode/inventory closure.
expected_payload={'matched_R1.npz','matched_R2.npz','matched_R3.npz','matched_R4.npz','matched_R5.npz','matched_R6.npz','summary.json'}
final_names={p.name for p in FINAL.iterdir()}
hash_checks={name:(sha256(FINAL/name)==digest) for name,digest in manifest['files'].items()}
cp_hash_checks={name:(sha256(CHECKPOINT/name)==digest) for name,digest in cp_manifest['files'].items()}
cp_hash_checks['FULL385_DIAGNOSTIC_V25_SHA256.json']=(sha256(CHECKPOINT/'FULL385_DIAGNOSTIC_V25_SHA256.json')==cp_sentinel['manifest_sha256'])
mode_checks={p.name:oct(stat.S_IMODE(p.stat().st_mode)) for p in FINAL.iterdir()}
mode_checks['.']=oct(stat.S_IMODE(FINAL.stat().st_mode))
checkpoint_modes={p.name:oct(stat.S_IMODE(p.stat().st_mode)) for p in CHECKPOINT.iterdir()}
checkpoint_modes['.']=oct(stat.S_IMODE(CHECKPOINT.stat().st_mode))

with np.load(FINAL/'matched_R6.npz',allow_pickle=False) as z:
    r6={k:np.asarray(z[k]) for k in z.files}
with np.load(FINAL/'matched_R5.npz',allow_pickle=False) as z:
    r5={k:np.asarray(z[k]) for k in z.files}
with np.load(CHECKPOINT/'full385_diagnostic_v25.npz',allow_pickle=False) as z:
    cp={k:np.asarray(z[k]) for k in z.files}
labels=np.asarray(r6['supercell_labels'],np.int64)
shell=np.asarray(r6['shell'],np.int64)
num3=np.asarray(r6['numerator3'],np.int64)
area=float(r6['area_nm2'])
primitive_lattice=np.linalg.inv(S)@np.asarray(r6['supercell_lattice_nm'],np.float64)
q0=int(np.flatnonzero(np.all(labels==0,axis=1))[0])
index={tuple(map(int,row)):i for i,row in enumerate(labels)}
num_index={tuple(map(int,row)):i for i,row in enumerate(num3)}
reverse_coeff=max(maxabs(np.asarray(r6['charge_fourier'])[i]-np.conj(np.asarray(r6['charge_fourier'])[index[tuple(-row)]])) for i,row in enumerate(labels))
reverse_hole=max(float(abs(np.asarray(r6['hole_charge_fourier'])[i]-np.conj(np.asarray(r6['hole_charge_fourier'])[index[tuple(-row)]]))) for i,row in enumerate(labels))
reverse_spin=max(maxabs(np.asarray(r6['spin_pauli_fourier'])[:,i]-np.conj(np.asarray(r6['spin_pauli_fourier'])[:,index[tuple(-row)]])) for i,row in enumerate(labels))

# Inventory checks independently from the final NPZ.
metric=num3[:,0]**2+num3[:,0]*num3[:,1]+num3[:,1]**2
inventory={
 'count':int(len(labels)), 'unique':len({tuple(x) for x in labels})==len(labels),
 'labels_match_det3_map':bool(np.array_equal(labels,np.stack(((num3[:,0]-num3[:,1])//3,(num3[:,0]+2*num3[:,1])//3),axis=1))),
 'congruence_u_minus_v_mod3':bool(np.all((num3[:,0]-num3[:,1])%3==0)),
 'metric_le_9R2':bool(np.all(metric<=9*shell**2)),
 'reverse_closed':all(tuple(-x for x in row) in num_index for row in num3),
 'c3_closed':all((int(row[1]),int(-row[0]-row[1])) in num_index for row in num3),
 'shell_counts':[int(np.count_nonzero(shell<=R)) for R in range(1,7)],
 'sector_counts':{str(s):int(np.count_nonzero(num3[:,0]%3==s)) for s in range(3)},
}

# Reconstruct regional integrals independently at both grids.
f6_384,imag6_384=fields(r6,384); f6_192,imag6_192=fields(r6,192); f5_384,imag5_384=fields(r5,384)
regional={}; max_cyclic=0.; max_q0sum=0.; max_quad_charge=0.; max_quad_spin=0.; max_summary=0.
q0_expected=np.asarray([r6['charge_fourier'][q0].real,r6['hole_charge_fourier'][q0].real,*list(r6['spin_pauli_fourier'][:,q0].real/2)])
for kind in ('parallelogram','wigner_seitz'):
    regional[kind]={}
    ints={}
    for repi,rep in enumerate(REPS):
        p384=partition_labels(primitive_lattice,384,kind,rep)
        p192=partition_labels(primitive_lattice,192,kind,rep)
        i6=integrate(f6_384,p384,area); i5=integrate(f5_384,p384,area); i192=integrate(f6_192,p192,area)
        ints[repi]=i6
        max_q0sum=max(max_q0sum,maxabs(np.sum(i6,axis=1)-q0_expected))
        max_quad_charge=max(max_quad_charge,maxabs(i192[:2]-i6[:2]))
        max_quad_spin=max(max_quad_spin,maxabs(i192[2:]-i6[2:]))
        recorded=np.asarray(summary['gates']['shell_metrics'][-1]['regional_integrals'][kind]['physical_translation_representatives'][str(repi)])
        max_summary=max(max_summary,maxabs(recorded-i6))
        regional[kind][str(repi)]={'R6_grid384':i6,'R5_grid384':i5,'R6_grid192':i192,'R6_minus_R5':i6-i5,'grid384_minus_grid192':i6-i192}
    base=ints[0]
    for shift in (1,2):
        expected=np.asarray([[base[a,(r+shift)%3] for r in range(3)] for a in range(5)])
        max_cyclic=max(max_cyclic,maxabs(ints[shift]-expected))

# Field gates and shell convergence.
margin=f6_384[0]/2-np.sqrt(np.sum(f6_384[2:]**2,axis=0))
delta=f6_384-f5_384
field_metrics={
 'R6_grid384_max_imaginary_abs':imag6_384,
 'R6_grid192_max_imaginary_abs':imag6_192,
 'electron_total_e':float(np.mean(f6_384[0])*area),
 'hole_total_e':float(np.mean(f6_384[1])*area),
 'minimum_electron_charge_e_nm2':float(np.min(f6_384[0])),
 'minimum_hole_charge_e_nm2':float(np.min(f6_384[1])),
 'minimum_local_pauli_margin_hbar_nm2':float(np.min(margin)),
 'R5_R6_relative_l2_charge':float(np.linalg.norm(delta[0])/np.linalg.norm(f6_384[0])),
 'R5_R6_max_abs_charge_e_nm2':maxabs(delta[0]),
 'R5_R6_max_abs_spin_component_hbar_nm2':maxabs(delta[2:]),
 'R5_R6_max_vector_spin_norm_hbar_nm2':float(np.max(np.sqrt(np.sum(delta[2:]**2,axis=0)))),
 'R5_R6_max_regional_charge_e':max(maxabs(regional[k]['0']['R6_minus_R5'][:2]) for k in regional),
 'R5_R6_max_regional_spin_hbar':max(maxabs(regional[k]['0']['R6_minus_R5'][2:]) for k in regional),
 'quadrature_192_384_max_regional_charge_e':max_quad_charge,
 'quadrature_192_384_max_regional_spin_hbar':max_quad_spin,
 'partition_translation_cyclic_max_abs':max_cyclic,
 'partition_q0_sum_max_abs':max_q0sum,
 'summary_regional_integral_max_abs':max_summary,
}

# All six NPZ payloads versus R6 nesting and summary shell records.
shell_npz_consistency={}; shell_summary_max_abs=0.0; shell_regional_max_abs=0.0
for R in range(1,7):
    with np.load(FINAL/f"matched_R{R}.npz",allow_pickle=False) as z:
        zr={k:np.asarray(z[k]) for k in z.files}
    mask=shell<=R; sm=summary["gates"]["shell_metrics"][R-1]
    exact={
      "numerator3":bool(np.array_equal(zr["numerator3"],r6["numerator3"][mask])),
      "supercell_labels":bool(np.array_equal(zr["supercell_labels"],r6["supercell_labels"][mask])),
      "shell":bool(np.array_equal(zr["shell"],r6["shell"][mask])),
      "charge_fourier":bool(np.array_equal(zr["charge_fourier"],r6["charge_fourier"][mask])),
      "hole_charge_fourier":bool(np.array_equal(zr["hole_charge_fourier"],r6["hole_charge_fourier"][mask])),
      "spin_pauli_fourier":bool(np.array_equal(zr["spin_pauli_fourier"],r6["spin_pauli_fourier"][:,mask])),
      "axes":bool(np.array_equal(zr["axes"],r6["axes"])),
      "supercell_matrix":bool(np.array_equal(zr["supercell_matrix"],r6["supercell_matrix"])),
      "supercell_lattice_nm":bool(np.array_equal(zr["supercell_lattice_nm"],r6["supercell_lattice_nm"])),
      "area_nm2":bool(np.array_equal(zr["area_nm2"],r6["area_nm2"])),
    }
    fr,im=fields(zr,384); mar=fr[0]/2-np.sqrt(np.sum(fr[2:]**2,axis=0))
    vals={"channels":len(zr["shell"]),"electron_total_e":float(np.mean(fr[0])*area),"hole_total_e":float(np.mean(fr[1])*area),"minimum_electron_charge_e_nm2":float(np.min(fr[0])),"minimum_hole_charge_e_nm2":float(np.min(fr[1])),"minimum_local_pauli_margin_hbar_nm2":float(np.min(mar)),"maximum_imaginary_abs":im}
    scalar_diff=max(abs(float(vals[k])-float(sm[k])) for k in vals)
    regdiff=0.0
    for kind in ("parallelogram","wigner_seitz"):
      for repi,rep in enumerate(REPS):
        got=integrate(fr,partition_labels(primitive_lattice,384,kind,rep),area)
        want=np.asarray(sm["regional_integrals"][kind]["physical_translation_representatives"][str(repi)])
        regdiff=max(regdiff,maxabs(got-want))
    shell_summary_max_abs=max(shell_summary_max_abs,scalar_diff); shell_regional_max_abs=max(shell_regional_max_abs,regdiff)
    shell_npz_consistency[str(R)]={"exact_nested_arrays":exact,"scalar_summary_max_abs":scalar_diff,"regional_summary_max_abs":regdiff}

# Checkpoint/final array identity.
checkpoint_final={
 'numerator3':bool(np.array_equal(cp['numerator3'],r6['numerator3'])),
 'supercell_labels':bool(np.array_equal(cp['supercell_labels'],r6['supercell_labels'])),
 'shell':bool(np.array_equal(cp['shell'],r6['shell'])),
 'axes':bool(np.array_equal(cp['axes'],r6['axes'])),
 'charge_fourier':bool(np.array_equal(cp['charge_fourier'],r6['charge_fourier'])),
 'hole_charge_fourier':bool(np.array_equal(cp['hole_charge_fourier'],r6['hole_charge_fourier'])),
 'spin_pauli_fourier':bool(np.array_equal(cp['spin_pauli_fourier'],r6['spin_pauli_fourier'])),
}

# Primitive-127 baseline replay from checkpoint, independent of summary.
with np.load(Path(cfg['primitive_baseline']['path']),allow_pickle=False) as z:
    old_g=np.asarray(z['integer_g'],np.int64)[:,:2]; old_base=np.asarray(z['density_fourier_reverse_projected'],np.complex128)
baseline=np.asarray(cp['primitive_baseline'],np.complex128)
baseline_error=max(float(abs(baseline[num_index[(int(3*g[0]),int(3*g[1]))]]-v)) for g,v in zip(old_g,old_base,strict=True))

# Seven-Q direct physical-label replay, without a second reversal.
with np.load(Path(cfg['accepted_seven_q']['path']),allow_pickle=False) as z:
    old={k:np.asarray(z[k]) for k in z.files}
fi=[str(x) for x in old['filling_names']].index('n2_B1'); ei=int(np.flatnonzero(old['dielectric_constants']==10)[0])
old_h=old['charge_hole_fourier'][fi,ei]
old_c=old['complete_active_seven_mode_charge_fourier']-old_h
old_s=old['spin_pauli_fourier'][fi,ei]
errs={'charge':[],'hole':[],'sx':[],'sy':[],'sz':[]}
for oi,label in enumerate(np.asarray(old['labels'],np.int64)):
    ni=index[tuple(map(int,label))]
    errs['charge'].append(abs(r6['charge_fourier'][ni]-old_c[oi])); errs['hole'].append(abs(r6['hole_charge_fourier'][ni]-old_h[oi]))
    for a,name in enumerate(('sx','sy','sz')): errs[name].append(abs(r6['spin_pauli_fourier'][a,ni]-old_s[a,oi]))
seven={k:float(max(v)) for k,v in errs.items()}

# Hypothesis diagnostics for every choice of the nominal near-zero C region.
def hypothesis_rows(mat):
    N=mat[1]; M=mat[2:].T
    out={}
    for c in range(3):
        a,b=[i for i in range(3) if i!=c]
        ell=M[a]-M[b]; ell=ell/np.linalg.norm(ell)
        para=M@ell; trans=M-para[:,None]*ell[None,:]
        out[f'C=owner{c}']={
          'A_owner':a,'B_owner':b,'C_owner':c,'N_hole':[float(N[a]),float(N[b]),float(N[c])],
          'N_A_minus_N_B':float(N[a]-N[b]),'M_norm_hbar':[float(np.linalg.norm(M[a])),float(np.linalg.norm(M[b])),float(np.linalg.norm(M[c]))],
          'longitudinal_axis_xyz':ell,'M_parallel_hbar':para,'A_B_parallel_sum':float(para[a]+para[b]),
          'M_transverse_norm_hbar':[float(np.linalg.norm(x)) for x in trans],
        }
    return out
hyp={k:hypothesis_rows(v['0']['R6_grid384']) for k,v in regional.items()}

report={
 'schema':'ptse2_v25_job513422_independent_postflight_recompute/v1',
 'inputs':{'matched_R6_sha256':sha256(FINAL/'matched_R6.npz'),'matched_R5_sha256':sha256(FINAL/'matched_R5.npz'),'summary_sha256':sha256(FINAL/'summary.json'),'checkpoint_npz_sha256':sha256(CHECKPOINT/'full385_diagnostic_v25.npz')},
 'publication':{'final_exact_names':sorted(final_names),'final_allowlist_exact':final_names==expected_payload|{'OUTPUT_SHA256.json','COMPLETE'},'payload_hash_checks':hash_checks,'sentinel_manifest_hash':sha256(FINAL/'OUTPUT_SHA256.json')==sentinel['output_manifest_sha256'],'sentinel_summary_hash':sha256(FINAL/'summary.json')==sentinel['summary_sha256'],'sentinel_R6_hash':sha256(FINAL/'matched_R6.npz')==sentinel['matched_R6_sha256'],'sentinel_checkpoint_hash':sha256(CHECKPOINT/'FULL385_DIAGNOSTIC_V25_COMPLETE')==sentinel['diagnostic_checkpoint_sentinel_sha256'],'final_modes':mode_checks,'checkpoint_hash_checks':cp_hash_checks,'checkpoint_modes':checkpoint_modes,'final_sentinel_last_claim':sentinel['sentinel_last'],'checkpoint_sentinel_last_claim':cp_sentinel['sentinel_last']},
 'inventory':inventory,
 'fourier':{'q0_index':q0,'q0_charge':r6['charge_fourier'][q0],'q0_hole':r6['hole_charge_fourier'][q0],'q0_spin_pauli':r6['spin_pauli_fourier'][:,q0],'reverse_charge_max_abs':reverse_coeff,'reverse_hole_max_abs':reverse_hole,'reverse_spin_max_abs':reverse_spin,'primitive127_baseline_max_abs':baseline_error,'seven_q_physical_label_direct_max_abs':seven},
 'shell_npz_summary_consistency':{'per_shell':shell_npz_consistency,'all_scalar_summary_max_abs':shell_summary_max_abs,'all_regional_summary_max_abs':shell_regional_max_abs},
 'checkpoint_final_array_identity':checkpoint_final,
 'field_metrics':field_metrics,
 'regional_integrals':regional,
 'hypothesis_diagnostics':hyp,
 'summary_selected_gate_differences':{
    'baseline':baseline_error-summary['gates']['primitive_127_baseline_replay_max_abs'],
    'quadrature_charge':max_quad_charge-summary['gates']['quadrature_192_to_384_regional_charge_max_abs_e'],
    'quadrature_spin':max_quad_spin-summary['gates']['quadrature_192_to_384_regional_spin_max_abs_hbar'],
    'partition_cyclic':max_cyclic-summary['gates']['physical_partition_translation_cyclic_max_abs'],
    'partition_q0_sum':max_q0sum-summary['gates']['physical_partition_q0_sum_max_abs'],
 },
}
print(json.dumps(native(report),indent=2,sort_keys=True,allow_nan=False))
