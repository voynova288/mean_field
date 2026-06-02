#!/bin/bash
# Saved-state diagnostic for fixed cRPA reference one-body bandwidths.

#SBATCH -J crpa_ref_h0diag
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -o logs/crpa_fixed_reference_h0_diag_20260524_%j.out
#SBATCH -e logs/crpa_fixed_reference_h0_diag_20260524_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_ref_h0diag_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

NO_REMOTE_STATE="${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_no_remote_lk24_iter200_from60/theta_105_nu_-3000_crpa_no_remote_projector_lk24_lg9_iter200_from60/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz"
ACTIVE_CNP_STATE="${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_hartreefix_active_cnp_ref_lk24_fullchi0_iter200_from60/theta_105_nu_-3000_crpa_active_cnp_fock_ref_fullchi0_hartreefix_lk24_lg9_iter200_from60/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz"
SCF_PATH_TSV="${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_no_remote_lk24_iter200_from60/theta_105_nu_-3000_crpa_no_remote_projector_lk24_lg9_iter200_from60/path_bands/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_hf_scf_path.tsv"
OUT_DIR="${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_hartreefix_active_cnp_ref_lk24_fullchi0_iter200_from60/diagnostics_fixed_reference_20260524/active_cnp_fock_reference_vs_no_remote"

echo "[job] hostname=$(hostname)"
echo "[job] no_remote_state=${NO_REMOTE_STATE}"
echo "[job] active_cnp_state=${ACTIVE_CNP_STATE}"
echo "[job] scf_path_tsv=${SCF_PATH_TSV}"
echo "[job] out_dir=${OUT_DIR}"

python scripts/diagnose_crpa_remote_bare_from_states.py \
  --no-remote-state "${NO_REMOTE_STATE}" \
  --remote-state "${ACTIVE_CNP_STATE}" \
  --path-tsv "${SCF_PATH_TSV}" \
  --output-dir "${OUT_DIR}"

echo "[job] complete"
