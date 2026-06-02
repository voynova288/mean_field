#!/bin/bash
#SBATCH -J tmbg_pol_chern
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/tmbg_polshyn_chern_probe_%j.out
#SBATCH -e logs/tmbg_polshyn_chern_probe_%j.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/TMBG_Polshyn2021_figS1/chern_probe_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

echo "[job] hostname=$(hostname) job_id=${SLURM_JOB_ID:-manual} out_dir=${OUT_DIR}"
python - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
from mean_field.systems.tmbg.topology_sewn import central_band_cherns

out_dir = Path(os.environ['OUT_DIR'])
theta = float(os.environ.get('THETA_DEG', '1.29'))
parameter_set = os.environ.get('PARAMETER_SET', 'polshyn2020')
delta = float(os.environ.get('INTERLAYER_POTENTIAL', '-0.033'))
staggered = float(os.environ.get('STAGGERED_POTENTIAL', '0.0'))
stacking = os.environ.get('BLG_STACKING', 'BA')
n_shells = int(os.environ.get('N_SHELLS', '5'))
mesh = int(os.environ.get('TOPOLOGY_MESH', '12'))
valley = int(os.environ.get('VALLEY', '1'))
orientation = os.environ.get('SEWN_ORIENTATION', 'raw')
if parameter_set == 'polshyn2020':
    params = TMBGParameters.polshyn2020(
        interlayer_potential=delta,
        staggered_potential=staggered,
        blg_stacking=stacking,
    )
else:
    params = TMBGParameters.full(
        interlayer_potential=delta,
        staggered_potential=staggered,
        blg_stacking=stacking,
    )
model = TMBGModel.from_config(theta, n_shells=n_shells, params=params)
result = central_band_cherns(model, mesh=mesh, valley=valley, orientation=orientation, subset_pad=2)
result.update({
    'parameter_set': parameter_set,
    'interlayer_potential_ev': delta,
    'staggered_potential_ev': staggered,
    'blg_stacking': stacking,
    'bernal_convention': params.bernal_convention,
})
path = out_dir / 'central_band_cherns.json'
path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
PY
