#!/bin/bash
# Run nu=-3 Fig. 4 HF+cRPA bands with an HF-compatible epsilon=4 artifact.
# The Zhang paper-reference dielectric bundle remains a validation reference;
# production HF must use periodic_g_grid=true / form_factor_mode=k_periodic_zero_fill.

#SBATCH -J mf_crpahf_m3_fixedeps
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 3-00:00:00
#SBATCH -o logs/custom_b0_hf_lk24_crpa_nu_m3_fixedeps_%j.out
#SBATCH -e logs/custom_b0_hf_lk24_crpa_nu_m3_fixedeps_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_hf_crpa_fixedeps_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

CRPA_BUNDLE="${REPO_ROOT}/results/TBG_HF_cRPA/crpa_dielectric_fixed_20260519"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
OUTPUT_ROOT="${REPO_ROOT}/results/TBG_HF_cRPA/hf_crpa_fig4_eps4_runs"
RUN_TAG="${RUN_TAG:-fig4_eps4_crpa_hfcompatible_20260522_lk24_q11_gamma_m_k_gamma_kprime}"

if [[ ! -d "${CRPA_BUNDLE}" ]]; then
  echo "Missing cRPA bundle: ${CRPA_BUNDLE}" >&2
  exit 2
fi
if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing merged cRPA artifact files under: ${CRPA_DIR}" >&2
  exit 2
fi
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] numba_threads=${NUMBA_NUM_THREADS:-}"
echo "[job] crpa_bundle=${CRPA_BUNDLE}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] run_tag=${RUN_TAG}"
echo "[job] target_epsilon_r=4"
echo "[job] note=production HF requires HF-compatible cRPA; the paper-reference dielectric bundle is not passed to HF"
echo "[job] note=using bm:1 start to avoid importing the Fig. 5/no-cRPA epsilon=10 initial density"

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
  --max-iter 3000 \
  --points-per-segment 120 \
  --path-kind gamma-m-k-gamma-kprime \
  --write-scf-path \
  --init bm:1 \
  --summary-mode both \
  --crpa-dir "${CRPA_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear
