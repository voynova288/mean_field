#!/bin/bash
# Produce SCF-grid-only line plot and bandwidth report for the converged nu=-3 cRPA-HF run.

#SBATCH -J crpa_scf_bw
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/crpa_converged_scf_line_bandwidth_20260523_%j.out
#SBATCH -e logs/crpa_converged_scf_line_bandwidth_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_scf_bw_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_hartreefix_active_cnp_ref_lk24_fullchi0_iter200_from60/theta_105_nu_-3000_crpa_active_cnp_fock_ref_fullchi0_hartreefix_lk24_lg9_iter200_from60}"
STATE="${STATE:-${RUN_DIR}/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz}"
SCF_ONLY_DIR="${SCF_ONLY_DIR:-${RUN_DIR}/path_bands/scf_line_only_20260523}"
STEM="theta_105_nu_-3000_init_vp_seed_001_lk24_lg9"
SCF_TSV="${SCF_ONLY_DIR}/${STEM}_hf_scf_path.tsv"
SCF_SUMMARY="${SCF_ONLY_DIR}/${STEM}_hf_scf_path_summary.json"
BANDWIDTH_JSON="${SCF_ONLY_DIR}/${STEM}_scf_bandwidth_report.json"

echo "[job] run_dir=${RUN_DIR}"
echo "[job] state=${STATE}"
echo "[job] scf_only_dir=${SCF_ONLY_DIR}"

python scripts/plot_tbg_hf_scf_grid_path_from_state.py \
  --state "${STATE}" \
  --output-dir "${SCF_ONLY_DIR}" \
  --path-kind gamma-m-k-gamma-kprime \
  --points-per-segment 120

python scripts/summarize_tbg_hf_scf_path_tsv.py \
  --tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${SCF_SUMMARY}"

python scripts/report_tbg_hf_scf_bandwidth.py \
  --state "${STATE}" \
  --scf-tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${BANDWIDTH_JSON}"

cp -a "${SCF_TSV}" "${RUN_DIR}/hf_band_nu_-3_crpa.csv"
cp -a "${SCF_ONLY_DIR}/${STEM}_scf_grid_band_plot.png" "${RUN_DIR}/hf_band_nu_-3_crpa.png"

echo "[job] scf_plot=${SCF_ONLY_DIR}/${STEM}_scf_grid_band_plot.png"
echo "[job] scf_summary=${SCF_SUMMARY}"
echo "[job] bandwidth_report=${BANDWIDTH_JSON}"
echo "[job] complete"
