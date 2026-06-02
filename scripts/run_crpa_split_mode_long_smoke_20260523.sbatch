#!/bin/bash
# Longer small-shell cRPA split-mode smoke for candidate modes.

#SBATCH -J mf_crpa_split100
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o logs/crpa_split_mode_long_smoke_20260523_%j.out
#SBATCH -e logs/crpa_split_mode_long_smoke_20260523_%j.err

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
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
CASE_CPUS="${CASE_CPUS:-32}"

if [[ ! -f "${CRPA_DIR}/crpa_params.json" || ! -f "${CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing HF-compatible cRPA artifact under ${CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] case_cpus=${CASE_CPUS}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] crpa_dir=${CRPA_DIR}"

run_case() {
  local mode="$1"
  local run_tag="crpa_split_mode_${mode}_vp_lk12_lg3_iter100"
  local case_log="${REPO_ROOT}/logs/crpa_split_mode_long_${mode}_${SLURM_JOB_ID:-manual}.out"
  local case_mpl="/tmp/mplconfig_crpa_split_mode_long_${mode}_${SLURM_JOB_ID:-manual}"

  (
    export MEAN_FIELD_CRPA_SPLIT_MODE="${mode}"
    export OMP_NUM_THREADS="${CASE_CPUS}"
    export OPENBLAS_NUM_THREADS="${CASE_CPUS}"
    export MKL_NUM_THREADS="${CASE_CPUS}"
    export BLIS_NUM_THREADS="${CASE_CPUS}"
    export NUMEXPR_NUM_THREADS="${CASE_CPUS}"
    export NUMBA_NUM_THREADS="${CASE_CPUS}"
    export MPLCONFIGDIR="${case_mpl}"
    mkdir -p "${MPLCONFIGDIR}"

    echo "[case] split_mode=${mode}"
    echo "[case] threads=${CASE_CPUS}"
    python scripts/mean_field_tools.py run_custom_b0_hf_case \
      --theta-deg 1.05 \
      --nu -3 \
      --output-root "${OUTPUT_ROOT}" \
      --run-tag "${run_tag}" \
      --lk 12 \
      --lg 3 \
      --overlap-lg 3 \
      --w0 79.7 \
      --w1 97.4 \
      --vf 2135.4 \
      --epsilon-r 4 \
      --tanh-argument-scale-a 162.60162601626016 \
      --zero-limit finite \
      --max-iter 100 \
      --points-per-segment 40 \
      --path-kind gamma-m-k-gamma-kprime \
      --summary-mode both \
      --init vp:1 \
      --crpa-dir "${CRPA_DIR}" \
      --fock-interpolation matrix_diagonal \
      --path-fock-interpolation linear
  ) > "${case_log}" 2>&1
}

pids=()
run_case "no_remote_projector" &
pids+=("$!")
run_case "no_remote_delta" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[job] case logs:"
ls -1 logs/crpa_split_mode_long_*_"${SLURM_JOB_ID:-manual}".out
echo "[job] complete status=${status}"
exit "${status}"
