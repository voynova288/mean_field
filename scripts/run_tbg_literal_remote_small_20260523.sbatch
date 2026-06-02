#!/bin/bash
# Small-system diagnostic for the literal remote-band bare background.

#SBATCH -J mf_lit_remote_sm
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/literal_remote_small_20260523_%j.out
#SBATCH -e logs/literal_remote_small_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_literal_remote_small_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523/literal_remote_small_lk6_lg3_iter5_variants}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
DYNAMIC="${DYNAMIC:-both}"
CRPA_CHI0_SCALE="${CRPA_CHI0_SCALE:-1.0}"
INCLUDE_REMOTE_RENORMALIZED_BASIS="${INCLUDE_REMOTE_RENORMALIZED_BASIS:-0}"
REMOTE_BAND_WINDOW="${REMOTE_BAND_WINDOW:-0}"

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] output_dir=${OUTPUT_DIR}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] dynamic=${DYNAMIC}"
echo "[job] crpa_chi0_scale=${CRPA_CHI0_SCALE}"
echo "[job] include_remote_renormalized_basis=${INCLUDE_REMOTE_RENORMALIZED_BASIS}"
echo "[job] remote_band_window=${REMOTE_BAND_WINDOW}"

extra_args=()
if [[ "${INCLUDE_REMOTE_RENORMALIZED_BASIS}" == "1" ]]; then
  extra_args+=(--include-remote-renormalized-basis)
fi

python scripts/diagnose_tbg_literal_remote_bare_small.py \
  --output-dir "${OUTPUT_DIR}" \
  --crpa-dir "${CRPA_DIR}" \
  --dynamic "${DYNAMIC}" \
  --crpa-chi0-scale "${CRPA_CHI0_SCALE}" \
  --lk 6 \
  --lg 3 \
  --overlap-lg 3 \
  --max-iter 5 \
  --theta-deg 1.05 \
  --nu -3 \
  --w0 79.7 \
  --w1 97.4 \
  --vf 2135.4 \
  --epsilon-r 4 \
  --tanh-argument-scale-a 162.60162601626016 \
  --finite-zero-limit \
  --fock-interpolation matrix_diagonal \
  --init vp \
  --seed 1 \
  --remote-band-window "${REMOTE_BAND_WINDOW}" \
  "${extra_args[@]}"

echo "[job] complete"
