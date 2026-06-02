#!/bin/bash
# Diagnostic-only zero-fill BM/HF run using the Zhang paper-reference cRPA
# artifact at the same lg/lk shell as the cRPA bundle.  This is a short
# early-gate run, not a converged production output.

#SBATCH -J crpa_zfill_lk24
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o logs/crpa_zero_fill_paperref_remote_lk24_lg9_iter5_20260524_%j.out
#SBATCH -e logs/crpa_zero_fill_paperref_remote_lk24_lg9_iter5_20260524_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_zfill_lk24_${SLURM_JOB_ID:-manual}}"
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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260524_zero_fill_paperref_remote_lk24_lg9_iter5}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_paper_ref_fixed_20260519_merged}"
RUN_TAG="${RUN_TAG:-crpa_zero_fill_paperref_remote_lk24_lg9_iter5}"

if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing paper-reference cRPA artifact under ${CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] run_tag=${RUN_TAG}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] boundary=zero_fill"
echo "[job] note=short early-gate diagnostic, no reconstructed path"

python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu -3 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --lk 24 \
  --lg 9 \
  --overlap-lg 9 \
  --zero-fill-g-grid \
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
  --allow-incompatible-crpa \
  --diagnostic-only \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear \
  --write-scf-path

RUN_DIR="${OUTPUT_ROOT}/theta_105_nu_-3000_${RUN_TAG}"
STATE="${RUN_DIR}/states/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9.npz"
SCF_TSV="${RUN_DIR}/path_bands/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_hf_scf_path.tsv"
REPORT="${RUN_DIR}/path_bands/theta_105_nu_-3000_init_vp_seed_001_lk24_lg9_scf_bandwidth_report.json"

python scripts/report_tbg_hf_scf_bandwidth.py \
  --state "${STATE}" \
  --scf-tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${REPORT}"

echo "[job] bandwidth_report=${REPORT}"
echo "[job] complete"
