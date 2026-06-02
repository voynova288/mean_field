#!/bin/bash
# Full-shell short smoke for the cRPA remote/reference bare-term scale.

#SBATCH -J mf_crpa_rscale
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o logs/crpa_remote_scale_lk24_smoke_20260523_%j.out
#SBATCH -e logs/crpa_remote_scale_lk24_smoke_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_remote_scale_lk24_${SLURM_JOB_ID:-manual}}"
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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
REMOTE_SCALES="${REMOTE_SCALES:-0.25 0.5 0.75}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-crpa_remote_scale}"

if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing HF-compatible cRPA artifact under ${CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] remote_scales=${REMOTE_SCALES}"

for scale in ${REMOTE_SCALES}; do
  export MEAN_FIELD_CRPA_REMOTE_BARE_SCALE="${scale}"
  scale_tag="$(printf '%s' "${scale}" | tr '.-' 'pm')"
  run_tag="${RUN_TAG_PREFIX}_${scale_tag}_lk24_lg9_iter5"

  echo "[case] start scale=${scale} tag=${run_tag}"
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
    --max-iter 5 \
    --points-per-segment 120 \
    --path-kind gamma-m-k-gamma-kprime \
    --summary-mode both \
    --init vp:1 \
    --crpa-dir "${CRPA_DIR}" \
    --fock-interpolation matrix_diagonal \
    --path-fock-interpolation linear
  echo "[case] done scale=${scale} tag=${run_tag}"
done

echo "[job] complete"
