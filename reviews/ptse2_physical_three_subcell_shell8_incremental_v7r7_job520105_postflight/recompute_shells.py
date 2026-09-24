#!/usr/bin/env python3
"""Independent NPZ-first postflight for shell8 incremental V7R7 job 520105.

Does not import or execute capsule formula_core.py/analyze_shells.py and does not
read production ANALYSIS_ARRAYS.npz or POINTWISE_FIELD_DIFFERENCES.npz.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, stat, struct, sys
from pathlib import Path
sys.path.insert(0, "/data/home/ziyuzhu/miniconda3/envs/moirekp/lib/python3.11/site-packages")
import numpy as np

ROOT=Path('/data/home/ziyuzhu/Mean_Field')
CAP=ROOT/'results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell8_incremental_v7r7'
MERGE=CAP/'runtime/checkpoint/r6_r7_r8_merge_job_520105'
FINAL=CAP/'runtime/output/job_520105'
JOB=CAP/'runtime/job_520105'
RUNTIME=CAP/'runtime'
OBS=('electron_charge_e','hole_charge_e','spin_x_hbar','spin_y_hbar','spin_z_hbar')
SUPERCELL=np.array([[1,-1],[1,2]],dtype=np.int64)
REPS=((0,0),(1,0),(2,0))
GRID=384

def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(8<<20),b''): d.update(b)
 return d.hexdigest()
def mode(path): return f'{stat.S_IMODE(path.lstat().st_mode):04o}'
def cjson(x): return (json.dumps(native(x),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def native(x):
 if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [native(v) for v in x]
 if isinstance(x,np.ndarray): return native(x.tolist())
 if isinstance(x,(np.integer,)): return int(x)
 if isinstance(x,(np.floating,)): return float(x)
 if isinstance(x,(np.bool_,)): return bool(x)
 if isinstance(x,(complex,np.complexfloating)): return {'real':float(x.real),'imag':float(x.imag)}
 return x

def shell_of(u,v):
 q=u*u+u*v+v*v
 for r in range(9):
  if q<=9*r*r:return r
 raise ValueError((u,v))
def inventory(radius):
 b=math.ceil(math.sqrt(12*radius*radius)); rows=[]
 for u in range(-b,b+1):
  for v in range(-b,b+1):
   q=u*u+u*v+v*v
   if (u-v)%3==0 and q<=9*radius*radius:
    rows.append((shell_of(u,v),q,u%3,u,v,(u-v)//3,(u+2*v)//3))
 return sorted(rows)
def load(r):
 p=MERGE/f'matched_R{r}.npz'
 with np.load(p,allow_pickle=False) as z: d={k:np.asarray(z[k]) for k in z.files}; keys=z.files
 expected=['schema','numerator3','supercell_labels','shell','axes','charge_fourier','hole_charge_fourier','spin_pauli_fourier','spin_physical_fourier','supercell_matrix','supercell_lattice_nm','area_nm2']
 if keys!=expected: raise ValueError(f'R{r} keys/order')
 return d

def cov_metrics(spin,mask,area):
 s=spin[:,mask]; cov=np.real(s@s.conj().T)/area**2
 vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
 axis=vecs[:,0].copy()
 if axis[0]<0:axis*=-1
 return {'matrix':cov,'eigenvalues':vals,'fractions':vals/np.trace(cov),'axis':axis,'trace':float(np.trace(cov))}
def z2(charge,spin,labels,mask,axis):
 by={tuple(map(int,x)):i for i,x in enumerate(labels)}; num=0j; den=0.; pairs=pos=zero=0; cr=sr=0.
 for i in np.flatnonzero(mask):
  a,b=map(int,labels[i])
  if not (a>0 or (a==0 and b>0)):continue
  j=by[(-a,-b)]; pairs+=1
  sp=complex(axis@spin[:,i]); sm=complex(axis@spin[:,j]); cr=max(cr,abs(charge[j]-charge[i].conjugate())); sr=max(sr,abs(sm-sp.conjugate()))
  for n,s in ((charge[i],sp),(charge[j],sm)):
   w=float(abs(n)*abs(s))
   if w==0:zero+=1;continue
   rel=n*s.conjugate();num+=w*(rel/abs(rel))**2;den+=w;pos+=1
 return {'pair_count':pairs,'positive_weight_terms':pos,'exact_zero_weight_terms':zero,'denominator':den,'z2':num/den,'charge_reverse_max':cr,'spin_reverse_max':sr}
def fourier_metrics(d):
 sectors=d['numerator3'][:,0]%3; tsb=sectors!=0; area=float(d['area_nm2']); c=d['charge_fourier'];h=d['hole_charge_fourier'];s=d['spin_physical_fourier']
 cv=cov_metrics(s,tsb,area); ph=z2(c,s,d['supercell_labels'],tsb,cv['axis'])
 return {'electron_tsb_rms':float(np.sqrt(np.sum(abs(c[tsb])**2))/area),'hole_tsb_rms':float(np.sqrt(np.sum(abs(h[tsb])**2))/area),'physical_spin_tsb_rms':float(np.sqrt(np.sum(abs(s[:,tsb])**2))/area),'covariance':cv,'phase':ph}
def reconstruct(d):
 labels=d['supercell_labels'];area=float(d['area_nm2']);coef=np.concatenate((d['charge_fourier'][None],d['hole_charge_fourier'][None],d['spin_physical_fourier']),axis=0)
 x=(np.arange(GRID,dtype=float)+.5)/GRID; px=np.exp(-2j*np.pi*np.outer(x,labels[:,0]));py=np.exp(-2j*np.pi*np.outer(x,labels[:,1]))
 return np.asarray([(px*row[None])@py.T/area for row in coef])

def signed_area(p):return .5*float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
def ccw(p):return p if signed_area(p)>=0 else p[::-1].copy()
def clip(p,n,b,tol=2e-13):
 if len(p)==0:return p
 out=[];prev=p[-1];pv=float(prev@n-b);pin=pv<=tol
 for cur in p:
  cv=float(cur@n-b);cin=cv<=tol
  if cin!=pin:
   den=pv-cv
   if abs(den)<1e-30:raise ValueError('clip degeneracy')
   out.append(prev+(pv/den)*(cur-prev))
  if cin:out.append(cur.copy())
  prev,pv,pin=cur,cv,cin
 if len(out)<3:return np.empty((0,2))
 r=np.asarray(out);clean=[r[0]]
 for q in r[1:]:
  if np.linalg.norm(q-clean[-1])>1e-12:clean.append(q)
 if len(clean)>2 and np.linalg.norm(clean[0]-clean[-1])<=1e-12:clean.pop()
 return np.asarray(clean) if len(clean)>=3 else np.empty((0,2))
def fundamental():return ccw(np.asarray([[0.,0.],SUPERCELL[0],SUPERCELL[0]+SUPERCELL[1],SUPERCELL[1]]))
def para_polys():
 f=fundamental();res={0:[],1:[],2:[]};v=np.rint(f).astype(int)
 for n0 in range(v[:,0].min()-1,v[:,0].max()+1):
  for n1 in range(v[:,1].min()-1,v[:,1].max()+1):
   p=f.copy()
   for n,b in ((np.array((-1.,0.)),-n0),(np.array((1.,0.)),n0+1),(np.array((0.,-1.)),-n1),(np.array((0.,1.)),n1+1)):p=clip(p,n,float(b))
   if len(p)>=3 and abs(signed_area(p))>1e-13:res[(n0+n1)%3].append(ccw(p))
 return res
def ws_polys(primitive,span=8):
 f=fundamental();metric=primitive@primitive.T;centers=[]
 for owner,rep in enumerate(REPS):
  for m0 in range(-span,span+1):
   for m1 in range(-span,span+1):centers.append((owner,np.asarray(rep,dtype=float)+np.asarray((m0,m1))@SUPERCELL))
 res={0:[],1:[],2:[]}
 for owner,c in centers:
  p=f.copy();cn=float(c@metric@c)
  for _,o in centers:
   d=o-c
   if np.all(d==0):continue
   p=clip(p,metric@d,(float(o@metric@o)-cn)/2)
   if len(p)==0:break
  if len(p)>=3 and abs(signed_area(p))>1e-13:res[owner].append(ccw(p))
 return res
def poly_int(p,k):
 p=ccw(np.asarray(p));k2=float(k@k)
 if k2<1e-28:return complex(abs(signed_area(p)),0)
 total=0j
 for a,b in zip(p,np.roll(p,-1,axis=0)):
  d=b-a;kd=float(k@d);avg=np.exp(-1j*float(k@(a+b)/2))*np.sinc(kd/(2*np.pi));total+=1j*float(k[0]*d[1]-k[1]*d[0])*avg/k2
 return total
def region_values(d,polys):
 primitive=np.linalg.inv(d['supercell_matrix'])@d['supercell_lattice_nm'];superlat=d['supercell_lattice_nm'];inv=np.linalg.inv(superlat);area=abs(float(np.linalg.det(superlat)));labels=d['supercell_labels']
 weights=np.empty((3,len(labels)),complex)
 for i,label in enumerate(labels):
  wave=2*np.pi*inv@label.astype(float)
  for o in range(3):weights[o,i]=sum(poly_int(p@primitive,wave) for p in polys[o])/area
 coeff=np.concatenate((d['charge_fourier'][None],d['hole_charge_fourier'][None],d['spin_physical_fourier']),axis=0)
 q0=int(np.flatnonzero(np.all(labels==0,axis=1))[0]);expected=np.zeros(len(labels),complex);expected[q0]=1
 return {'values':coeff@weights.T,'closure':float(np.max(abs(np.sum(weights,axis=0)-expected))),'q0_owner':float(np.max(abs(weights[:,q0]-1/3))),'weights':weights}
def ult(values,principal):
 spin=values[2:5].real.T;A,B,C=spin[0],spin[2],spin[1];U=A+B+C;L=(A-B)/2;T=(A+B-2*C)/6;ln=np.linalg.norm(L);axis=(A-B)/np.linalg.norm(A-B);cp=float(C@axis)
 rec=np.stack((U/3+L+T,U/3-L+T,U/3-2*T))
 return {'A':A,'B':B,'C':C,'U':U,'L':L,'T':T,'U_norm':float(np.linalg.norm(U)),'L_norm':float(ln),'T_norm':float(np.linalg.norm(T)),'T_norm_over_L_norm':float(np.linalg.norm(T)/ln),'M_C_norm':float(np.linalg.norm(C)),'M_C_parallel':cp,'M_C_perp_norm':float(np.linalg.norm(C-cp*axis)),'inverse_reconstruction_max_abs':float(np.max(abs(rec-np.stack((A,B,C))))),'principal_axis_absolute_dot_L':float(abs((L/ln)@principal))}
def read_csv(p):
 with p.open(newline='') as h:return list(csv.DictReader(h))
def png_info(p):
 b=p.read_bytes(); assert b[:8]==b'\x89PNG\r\n\x1a\n';w,h,depth,color,comp,filt,inter=struct.unpack('>IIBBBBB',b[16:29]);return {'width':w,'height':h,'bit_depth':depth,'color_type':color,'compression':comp,'filter':filt,'interlace':inter,'sha256':sha(p),'size_bytes':p.stat().st_size,'mode':mode(p)}

def manifest_check(directory,name):
 mpath=directory/name;m=json.loads(mpath.read_text());expected=set(m['files']);actual={p.name for p in directory.iterdir()};rows={}
 for n,rec in m['files'].items():
  p=directory/n;st=p.lstat();rows[n]={'regular_nonsymlink':stat.S_ISREG(st.st_mode) and not p.is_symlink(),'mode':mode(p),'size_matches':st.st_size==rec['size_bytes'],'sha_matches':sha(p)==rec['sha256'],'nlink':st.st_nlink}
 return m,{'expected_names':sorted(expected),'actual_names':sorted(actual),'exact_namespace':actual==expected|{name},'directory_mode':mode(directory),'manifest_mode':mode(mpath),'manifest_sha256':sha(mpath),'files':rows,'all_files_pass_0400':all(x['regular_nonsymlink'] and x['mode']=='0400' and x['size_matches'] and x['sha_matches'] for x in rows.values())}

def map_checks():
 rrpath=JOB/'RUNTIME_RECEIPT.json';rr=json.loads(rrpath.read_text());out={'runtime_receipt_sha256':sha(rrpath),'runtime_receipt_passed':rr['passed'],'profiles':{}}
 for profile,bundle_path in [('analyze_merge',MERGE/'RUNTIME_MAP_RECEIPTS.json'),('analyze_final',FINAL/'RUNTIME_MAP_RECEIPTS.json')]:
  b=json.loads(bundle_path.read_text());rows=[]
  for group in ('after_imports','lazy_prepublish'):
   for rec in b[group]:
    ext=Path(rec['external_path']);phase='after_imports' if rec['phase']=='after_imports' else 'lazy';core=rr['profiles'][profile][f'{phase}_core_records'];extras=rec['audited_extras'];ordered=sorted(core+extras,key=lambda x:(str(x['realpath']),str(x['path'])))
    er=[]
    for x in extras:
     p=Path(x['path']);st=p.lstat();er.append(not p.is_symlink() and stat.S_ISREG(st.st_mode) and st.st_size==x['size_bytes'] and sha(p)==x['sha256'])
    rows.append({'stage':rec['stage'],'external_path':str(ext),'external_sha_matches':sha(ext)==rec['external_sha256'],'runtime_receipt_sha_matches':rec['runtime_receipt_sha256']==sha(rrpath),'required_core_sha_matches_runtime_receipt':rec['required_core_sha256']==rr['profiles'][profile][f'{phase}_core_sha256'],'required_core_count_matches':rec['required_core_count']==len(core),'extras_digest_recomputed':rec['audited_extras_sha256']==hashlib.sha256(cjson(extras)).hexdigest(),'full_map_digest_recomputed':rec['full_map_sha256']==hashlib.sha256(cjson(ordered)).hexdigest(),'extras_files_currently_match':all(er),'required_core_exact_subset':rec['required_core_exact_subset'],'job_host_rank':(rec['job_id'],rec['hostname'],rec['rank'])})
  out['profiles'][profile]={'bundle_sha256':sha(bundle_path),'profile':b['profile'],'rank_count':b['rank_count'],'openmx_process_receipts':b['openmx_process_receipts'],'rows':rows}
 out['all_pass']=all(all(all(v for k,v in row.items() if k in {'external_sha_matches','runtime_receipt_sha_matches','required_core_sha_matches_runtime_receipt','required_core_count_matches','extras_digest_recomputed','full_map_digest_recomputed','extras_files_currently_match','required_core_exact_subset'}) for row in p['rows']) for p in out['profiles'].values())
 return out

shells={r:load(r) for r in (6,7,8)}
invs={};sets={}
for r,d in shells.items():
 rows=inventory(r);uv=np.asarray([(x[3],x[4]) for x in rows]);lab=np.asarray([(x[5],x[6]) for x in rows]);sh=np.asarray([x[0] for x in rows]);keys=set(map(tuple,uv));sets[r]=keys
 invs[r]={'count':len(rows),'sectors':[sum(x[2]==q for x in rows) for q in range(3)],'npz_exact_inventory':bool(np.array_equal(uv,d['numerator3']) and np.array_equal(lab,d['supercell_labels']) and np.array_equal(sh,d['shell'])),'unique_uv':len(keys)==len(rows),'unique_labels':len(set(map(tuple,lab)))==len(rows),'reverse_closed':all((-u,-v) in keys for u,v in keys),'c3_closed':all((v,-u-v) in keys for u,v in keys),'spin_pauli_exact_twice_physical':bool(np.array_equal(d['spin_pauli_fourier'],2*d['spin_physical_fourier'])),'area_nm2':float(d['area_nm2']),'supercell_matrix_exact':bool(np.array_equal(d['supercell_matrix'],SUPERCELL))}
invs['nested']={'R6_proper_R7':sets[6]<sets[7],'R7_proper_R8':sets[7]<sets[8],'new_R7':len(sets[7]-sets[6]),'new_R8':len(sets[8]-sets[7]),'new306':len(sets[8]-sets[6]),'prefix_arrays_exact':all(np.array_equal(shells[6][k],shells[7][k][:385]) and np.array_equal(shells[6][k],shells[8][k][:385]) for k in ('numerator3','supercell_labels','shell','charge_fourier','hole_charge_fourier')) and np.array_equal(shells[6]['spin_physical_fourier'],shells[7]['spin_physical_fourier'][:,:385]) and np.array_equal(shells[6]['spin_physical_fourier'],shells[8]['spin_physical_fourier'][:,:385])}
metrics={r:fourier_metrics(d) for r,d in shells.items()}
fields={r:reconstruct(d) for r,d in shells.items()}
point={}
for a,b in ((6,7),(7,8),(6,8)):
 delta=fields[b]-fields[a];point[f'R{a}_to_R{b}']={name:{'maximum_abs':float(np.max(abs(delta[i].real))),'rms':float(np.sqrt(np.mean(delta[i].real**2))),'maximum_imaginary_abs':float(np.max(abs(delta[i].imag)))} for i,name in enumerate(OBS)}
primitive=np.linalg.inv(shells[8]['supercell_matrix'])@shells[8]['supercell_lattice_nm'];pp=para_polys();wp=ws_polys(primitive,8)
regions={};ults={}
for r,d in shells.items():
 regions[r]={'parallelogram':region_values(d,pp),'wigner_seitz':region_values(d,wp)};ults[r]=ult(regions[r]['wigner_seitz']['values'],metrics[r]['covariance']['axis'])
# Production comparisons from CSVs only after independent formation.
frows={int(x['R']):x for x in read_csv(FINAL/'FOURIER_TSB_CONVERGENCE.csv')};crows={int(x['R']):x for x in read_csv(FINAL/'SPIN_COVARIANCE_CONVERGENCE.csv')};prows=read_csv(FINAL/'POINTWISE_FIELD_DIFFERENCES.csv');rrows=read_csv(FINAL/'REGIONAL_INTEGRALS_EXACT.csv');urows=read_csv(FINAL/'WS_ULT_MC_CONVERGENCE.csv')
comp={'fourier_max_abs':0.,'covariance_max_abs':0.,'pointwise_max_abs':0.,'regional_max_abs':0.,'ult_max_abs':0.,'z2':{}}
for r in (6,7,8):
 m=metrics[r];fr=frows[r];cr=crows[r]
 for key in ('electron_tsb_rms','hole_tsb_rms','physical_spin_tsb_rms'):comp['fourier_max_abs']=max(comp['fourier_max_abs'],abs(m[key]-float(fr[key])))
 z=m['phase']['z2'];comp['z2'][r]={'independent':z,'production':complex(float(fr['z2_real']),float(fr['z2_imag'])),'absolute_difference':abs(z-complex(float(fr['z2_real']),float(fr['z2_imag'])))}
 for i,k in enumerate(('lambda1','lambda2','lambda3')):comp['covariance_max_abs']=max(comp['covariance_max_abs'],abs(m['covariance']['eigenvalues'][i]-float(cr[k])))
 for i,k in enumerate(('fraction1','fraction2','fraction3')):comp['covariance_max_abs']=max(comp['covariance_max_abs'],abs(m['covariance']['fractions'][i]-float(cr[k])))
 for i,k in enumerate(('axis_x','axis_y','axis_z')):comp['covariance_max_abs']=max(comp['covariance_max_abs'],abs(m['covariance']['axis'][i]-float(cr[k])))
for row in prows:
 for k in ('maximum_abs','rms','maximum_imaginary_abs'):comp['pointwise_max_abs']=max(comp['pointwise_max_abs'],abs(point[row['shell_pair']][row['observable']][k]-float(row[k])))
for row in rrows:
 r=int(row['R']);part=row['partition'];owner=int(row['owner'][-1]);obs=OBS.index(row['observable']);comp['regional_max_abs']=max(comp['regional_max_abs'],abs(regions[r][part]['values'][obs,owner].real-float(row['value'])))
for row in urows:
 if row['defined']!='True':continue
 r=int(row['R']);q=row['quantity'];base=q[:-2];base='C' if base=='M_C' else base;v=ults[r][base]['xyz'.index(q[-1])] if q[-2:] in ('_x','_y','_z') else ults[r][q];comp['ult_max_abs']=max(comp['ult_max_abs'],abs(float(v)-float(row['value'])))
merge_manifest,merge_closure=manifest_check(MERGE,'OUTPUT_MANIFEST.json');final_manifest,final_closure=manifest_check(FINAL,'OUTPUT_MANIFEST.json')
ms=RUNTIME/'checkpoint/R6_R7_R8_MERGE_COMPLETE_520105';fs=RUNTIME/'SHELL8_ANALYSIS_COMPLETE_520105';msj=json.loads(ms.read_text());fsj=json.loads(fs.read_text())
artifact={'merge':merge_closure,'final':final_closure,'merge_sentinel':{'mode':mode(ms),'sha256':sha(ms),'nlink':ms.lstat().st_nlink,'manifest_binding':msj['manifest_sha256']==sha(MERGE/'OUTPUT_MANIFEST.json'),'staging_same_inode':ms.lstat().st_ino==(RUNTIME/'.r6_r7_r8_merge_staging.520105/R6_R7_R8_MERGE_COMPLETE_520105').lstat().st_ino},'final_sentinel':{'mode':mode(fs),'sha256':sha(fs),'nlink':fs.lstat().st_nlink,'manifest_binding':fsj['output_manifest_sha256']==sha(FINAL/'OUTPUT_MANIFEST.json'),'summary_binding':fsj['summary_sha256']==sha(FINAL/'SHELL_CONVERGENCE.json'),'report_binding':fsj['report_sha256']==sha(FINAL/'REPORT.md'),'merge_binding':fsj['merge_checkpoint_sentinel_sha256']==sha(ms),'staging_same_inode':fs.lstat().st_ino==(RUNTIME/'.shell8_analysis_staging.520105/SHELL8_ANALYSIS_COMPLETE_520105').lstat().st_ino},'stderr_zero_bytes':(RUNTIME/'slurm/ptse2_shell8_merge_v7r7_520105.err').stat().st_size==0,'runtime_map_receipts':map_checks(),'figures':{p.name:png_info(p) for p in FINAL.glob('*.png')}}
result={'schema':'ptse2_shell8_v7r7_job520105_independent_postflight/v1','method':{'production_modules_imported':False,'production_analysis_npz_read':False,'inputs':[str(MERGE/f'matched_R{r}.npz') for r in (6,7,8)],'grid':GRID,'fourier_convention':'f=A^-1 sum_Q f_Q exp[-2pi i(l1*x+l2*y)]','regional_method':'independent closed-form polygon Fourier integral'},'inventories':invs,'metrics':metrics,'pointwise_differences':point,'regional':{r:{p:{'values':x['values'],'closure':x['closure'],'q0_owner':x['q0_owner']} for p,x in v.items()} for r,v in regions.items()},'ult':ults,'production_absolute_differences':comp,'artifact_closure':artifact}
print(json.dumps(native(result),indent=2,sort_keys=True,allow_nan=False))
