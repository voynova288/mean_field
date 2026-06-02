#!/bin/bash
# Small-system gate for active-CNP Fock reference with Hartree[D] and Fock[P].

#SBATCH -J crpa_acnp_hd_sm
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/crpa_acnp_hd_small_20260523_%j.out
#SBATCH -e logs/crpa_acnp_hd_small_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_acnp_hd_small_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

export MEAN_FIELD_CRPA_SPLIT_MODE="${MEAN_FIELD_CRPA_SPLIT_MODE:-active_cnp_fock_reference_hartree_delta_projector}"
export MEAN_FIELD_CRPA_REMOTE_BARE_SCALE="${MEAN_FIELD_CRPA_REMOTE_BARE_SCALE:-1.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523_active_cnp_hd_small_fullchi0}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
RUN_TAG="${RUN_TAG:-crpa_active_cnp_fock_ref_hd_fp_lk6_lg3_iter5}"

if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing HF-compatible cRPA artifact under ${CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] run_tag=${RUN_TAG}"

python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu -3 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --lk 6 \
  --lg 3 \
  --overlap-lg 3 \
  --w0 79.7 \
  --w1 97.4 \
  --vf 2135.4 \
  --epsilon-r 4 \
  --tanh-argument-scale-a 162.60162601626016 \
  --zero-limit finite \
  --max-iter 5 \
  --points-per-segment 40 \
  --path-kind gamma-m-k-gamma-kprime \
  --summary-mode both \
  --init vp:1 \
  --crpa-dir "${CRPA_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear \
  --write-scf-path

RUN_DIR="${OUTPUT_ROOT}/theta_105_nu_-3000_${RUN_TAG}"
PATH_DIR="${RUN_DIR}/path_bands"
SCF_TSV="${PATH_DIR}/theta_105_nu_-3000_init_vp_seed_001_lk6_lg3_hf_scf_path.tsv"
SUMMARY_JSON="${PATH_DIR}/theta_105_nu_-3000_init_vp_seed_001_lk6_lg3_hf_scf_path_summary.json"

python scripts/summarize_tbg_hf_scf_path_tsv.py \
  --tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${SUMMARY_JSON}"

echo "[job] scf_summary=${SUMMARY_JSON}"
echo "[job] complete"
