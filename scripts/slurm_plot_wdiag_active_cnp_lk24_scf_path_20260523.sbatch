#!/bin/bash
# Postprocess the lk24 wdiag active-CNP cRPA gate into SCF-grid path artifacts.

#SBATCH -J mf_wdiag_scf
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/wdiag_active_cnp_lk24_scf_path_20260523_%j.out
#SBATCH -e logs/wdiag_active_cnp_lk24_scf_path_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_wdiag_active_cnp_scf_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

STATE="${STATE:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_hartreefix_wdiag_active_cnp_ref_lk24_fullchi0/theta_105_nu_-3000_crpa_fock_wdiag_rscale_1p0_lk24_lg9_iter5/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz}"
RUN_DIR="$(dirname "$(dirname "${STATE}")")"
PATH_DIR="${RUN_DIR}/path_bands"
TSV="${PATH_DIR}/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_hf_scf_path.tsv"
SUMMARY_JSON="${PATH_DIR}/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_hf_scf_path_summary.json"

echo "[job] hostname=$(hostname)"
echo "[job] state=${STATE}"
echo "[job] path_dir=${PATH_DIR}"

python scripts/plot_tbg_hf_scf_grid_path_from_state.py \
  --state "${STATE}" \
  --path-kind gamma-m-k-gamma-kprime \
  --points-per-segment 120 \
  --path-tolerance 1.0e-12

python scripts/summarize_tbg_hf_scf_path_tsv.py \
  --tsv "${TSV}" \
  --nu -3 \
  --output-json "${SUMMARY_JSON}"

echo "[job] wrote ${TSV}"
echo "[job] wrote ${SUMMARY_JSON}"
