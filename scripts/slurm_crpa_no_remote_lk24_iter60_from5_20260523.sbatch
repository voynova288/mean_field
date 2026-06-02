#!/bin/bash
# Continue the lk24 no-remote cRPA-HF diagnostic from the 5-step SCF state.
# This writes SCF-grid-only line data; no reconstructed/off-grid path bands.

#SBATCH -J crpa_norem60
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o logs/crpa_no_remote_lk24_iter60_from5_20260523_%j.out
#SBATCH -e logs/crpa_no_remote_lk24_iter60_from5_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_no_remote_iter60_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export MEAN_FIELD_CRPA_SPLIT_MODE="no_remote_projector"

CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
SOURCE_STATE="${SOURCE_STATE:-${REPO_ROOT}/results/TBG_HF_cRPA/rejected_smoke_20260523/crpa_gap_smoke_20260523_fockshellfix_lk24/theta_105_nu_-3000_crpa_no_remote_projector_lk24_lg9_iter5/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_no_remote_lk24_iter60_from5}"
RUN_TAG="${RUN_TAG:-crpa_no_remote_projector_lk24_lg9_iter60_from5}"
MAX_ITER="${MAX_ITER:-60}"

if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing HF-compatible cRPA artifact under ${CRPA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_STATE}" ]]; then
  echo "Missing source state ${SOURCE_STATE}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] source_state=${SOURCE_STATE}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] max_iter=${MAX_ITER}"

python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu -3 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --lk 24 \
  --lg 9 \
  --overlap-lg 9 \
  --w0 79.7 \
  --w1 97.4 \
  --vf 2135.4 \
  --epsilon-r 4 \
  --tanh-argument-scale-a 162.60162601626016 \
  --zero-limit finite \
  --max-iter "${MAX_ITER}" \
  --points-per-segment 120 \
  --path-kind gamma-m-k-gamma-kprime \
  --summary-mode both \
  --init vp:1 \
  --initial-state "${SOURCE_STATE}" \
  --crpa-dir "${CRPA_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear

RUN_DIR="${OUTPUT_ROOT}/theta_105_nu_-3000_${RUN_TAG}"
STEM="theta_105_nu_-3000_init_vp_seed_001_lk24_lg9"
STATE="${RUN_DIR}/states/${STEM}.npz"
SCF_TSV="${RUN_DIR}/path_bands/${STEM}_hf_scf_path.tsv"
BANDWIDTH_JSON="${RUN_DIR}/path_bands/${STEM}_scf_bandwidth_report.json"

python scripts/report_tbg_hf_scf_bandwidth.py \
  --state "${STATE}" \
  --scf-tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${BANDWIDTH_JSON}"

echo "[job] run_dir=${RUN_DIR}"
echo "[job] scf_plot=${RUN_DIR}/path_bands/${STEM}_scf_grid_band_plot.png"
echo "[job] scf_tsv=${SCF_TSV}"
echo "[job] bandwidth_report=${BANDWIDTH_JSON}"
echo "[job] complete"
