#!/bin/bash
# Formula-level cRPA Hartree regression plus saved-state remote-bare diagnostic.

#SBATCH -J crpa_formula_diag
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/crpa_hartree_formula_remote_diag_20260523_%j.out
#SBATCH -e logs/crpa_hartree_formula_remote_diag_20260523_%j.err

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
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_formula_diag_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

ROOT="${ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_fockshellfix_lk24}"
NO_REMOTE_RUN="${NO_REMOTE_RUN:-${ROOT}/theta_105_nu_-3000_crpa_no_remote_projector_lk24_lg9_iter5}"
REMOTE_RUN="${REMOTE_RUN:-${ROOT}/theta_105_nu_-3000_crpa_remote_scale_1_lk24_lg9_iter5}"
NO_REMOTE_STATE="${NO_REMOTE_STATE:-${NO_REMOTE_RUN}/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz}"
REMOTE_STATE="${REMOTE_STATE:-${REMOTE_RUN}/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz}"
PATH_TSV="${PATH_TSV:-${REMOTE_RUN}/path_bands/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_hf_scf_path.tsv}"
DIAG_OUT="${DIAG_OUT:-${ROOT}/diagnostics_remote_bare_20260523}"

echo "[job] hostname=$(hostname)"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] root=${ROOT}"

python -m pytest -q tests/test_crpa_core.py \
  -k 'crpa_hartree_full_matrix_uses_output_shift_as_first_index or crpa_path_hartree_full_matrix_uses_output_shift_as_first_index or remote_reference_eq19_direct_terms_match_half_reference_delta or crpa_split_with_identity_screening_matches_bare_delta_hf'

python scripts/diagnose_crpa_remote_bare_from_states.py \
  --no-remote-state "${NO_REMOTE_STATE}" \
  --remote-state "${REMOTE_STATE}" \
  --path-tsv "${PATH_TSV}" \
  --output-dir "${DIAG_OUT}"

python scripts/diagnose_crpa_remote_bare_components.py \
  --no-remote-state "${NO_REMOTE_STATE}" \
  --remote-state "${REMOTE_STATE}" \
  --path-tsv "${PATH_TSV}" \
  --output-dir "${DIAG_OUT}"
