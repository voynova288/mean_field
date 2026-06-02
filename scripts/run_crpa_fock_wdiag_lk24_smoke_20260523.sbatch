#!/bin/bash
# Full-shell short smoke for the Fock diag(screened_v) cRPA diagnostic.

#SBATCH -J mf_crpa_wdiag
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o logs/crpa_fock_wdiag_lk24_smoke_20260523_%j.out
#SBATCH -e logs/crpa_fock_wdiag_lk24_smoke_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_fock_wdiag_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export MEAN_FIELD_CRPA_SPLIT_MODE="${MEAN_FIELD_CRPA_SPLIT_MODE:-remote_projector}"
export MEAN_FIELD_CRPA_REMOTE_BARE_SCALE="${MEAN_FIELD_CRPA_REMOTE_BARE_SCALE:-1.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523}"
SOURCE_CRPA_DIR="${SOURCE_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
WDIG_CRPA_DIR="${WDIG_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_fock_wdiag_hfcompatible_20260523}"
MAX_ITER="${MAX_ITER:-5}"
INIT_TAG="${INIT_TAG:-vp:1}"

if [[ ! -f "${SOURCE_CRPA_DIR}/crpa_params.json" || ! -f "${SOURCE_CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing source HF-compatible cRPA artifact under ${SOURCE_CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] source_crpa_dir=${SOURCE_CRPA_DIR}"
echo "[job] wdiag_crpa_dir=${WDIG_CRPA_DIR}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] remote_bare_scale=${MEAN_FIELD_CRPA_REMOTE_BARE_SCALE}"

if [[ "${FORCE_REBUILD_WDIG:-0}" == "1" || ! -f "${WDIG_CRPA_DIR}/diagnostic_fock_wdiag_report.json" ]]; then
  echo "[artifact] build diag(screened_v) Fock artifact"
  python scripts/make_crpa_fock_wdiag_artifact.py \
    --input "${SOURCE_CRPA_DIR}" \
    --output "${WDIG_CRPA_DIR}" \
    --overwrite
else
  echo "[artifact] reusing ${WDIG_CRPA_DIR}"
fi

scale_tag="$(printf '%s' "${MEAN_FIELD_CRPA_REMOTE_BARE_SCALE}" | tr '.-' 'pm')"
run_tag="crpa_fock_wdiag_rscale_${scale_tag}_lk24_lg9_iter${MAX_ITER}"

echo "[case] start tag=${run_tag}"
python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu -3 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${run_tag}" \
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
  --init "${INIT_TAG}" \
  --crpa-dir "${WDIG_CRPA_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear
echo "[case] done tag=${run_tag}"

echo "[job] complete"
