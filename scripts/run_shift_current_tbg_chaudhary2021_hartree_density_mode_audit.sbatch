#!/bin/bash
#SBATCH --job-name=tbg_chau21_H_density_audit
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
  "${COMMON[@]}" --density-mode flat --mixing 0.35 --max-iter 60 \
  --output-dir "${ROOT}/flat"

PYTHONPATH=src python -m analysis.shift_current_tbg.run_chaudhary2021_hartree_bands \
  "${COMMON[@]}" --density-mode full_fixed_cnp --mixing 0.15 --max-iter 60 \
  --output-dir "${ROOT}/full_fixed_cnp"

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
root = Path('results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K')
flat = json.loads((root/'flat'/'summary.json').read_text())
full = json.loads((root/'full_fixed_cnp'/'summary.json').read_text())
lines = [
    '# Hartree density-mode audit',
    '',
    'Purpose: test whether Chaudhary Eq. (6)--(8) should be interpreted as the current flat-only excess density or as a full occupied density minus a fixed noninteracting CNP reference.  The latter includes remote-band wavefunction polarization and may double-count screening if epsilon_r is already renormalized, so it is a diagnostic, not a final model.',
    '',
    'Settings: lg=7, Hartree mesh m=5, epsilon_r=10, T=15 K, same b0 model as current paper comparisons.',
    '',
    '| nu | mode | conv | iter | final_err | mu (meV) | BW lower/upper (meV) | max |rho_q| | rho count |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
]
for nu in [-3.95,-2.0,0.0,2.0,3.95]:
    key=f'nu={nu:g}'
    for mode,s in [('flat',flat),('full_fixed_cnp',full)]:
        r=s['results'][key]
        bw=r.get('central_bandwidths_mev',[float('nan'),float('nan')])
        lines.append(f"| {nu:g} | {mode} | {int(bool(r.get('converged', False)))} | {int(r.get('iterations',0))} | {float(r.get('final_error',0.0)):.3g} | {1e3*float(r.get('mu_ev',0.0)):.3f} | {bw[0]:.2f}/{bw[1]:.2f} | {float(r.get('max_abs_rho_q',0.0)):.3g} | {int(r.get('rho_q_count',0))} |")
lines += [
    '',
    'Interpretation rule:',
    '',
    '- If `full_fixed_cnp` produces much larger/nonconvergent rho or qualitatively wrong band inversion on this coarse audit, do not use it as final; treat dispersive-band screening as already folded into epsilon_r as the paper text suggests.',
    '- If it gives controlled changes localized to outer-BZ/high-energy FD physics, then run a response diagnostic next.',
]
(root/'density_mode_audit.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps({'flat': flat, 'full_fixed_cnp': full}, indent=2, sort_keys=True)+'\n')
print('\n'.join(lines))
PY
