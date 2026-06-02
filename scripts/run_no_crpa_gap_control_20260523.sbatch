#!/bin/bash
# Small-shell no-cRPA control for the nu=-3 direct-gap/bandwidth smoke gate.

#SBATCH -J mf_nocrpa_gap_ctl
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -o logs/no_crpa_gap_control_20260523_%j.out
#SBATCH -e logs/no_crpa_gap_control_20260523_%j.err

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

OUTPUT_ROOT="${NO_CRPA_OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_gap_smoke_20260523}"
CASE_CPUS="${CASE_CPUS:-32}"

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] case_cpus=${CASE_CPUS}"
echo "[job] output_root=${OUTPUT_ROOT}"

run_case() {
  local label="$1"
  local epsilon_r="$2"
  local run_tag="no_crpa_gap_control_${label}_lk12_lg3_iter5"
  local case_log="${REPO_ROOT}/logs/no_crpa_gap_control_${label}_${SLURM_JOB_ID:-manual}.out"
  local case_mpl="/tmp/mplconfig_no_crpa_gap_control_${label}_${SLURM_JOB_ID:-manual}"

  (
    export OMP_NUM_THREADS="${CASE_CPUS}"
    export OPENBLAS_NUM_THREADS="${CASE_CPUS}"
    export MKL_NUM_THREADS="${CASE_CPUS}"
    export BLIS_NUM_THREADS="${CASE_CPUS}"
    export NUMEXPR_NUM_THREADS="${CASE_CPUS}"
    export NUMBA_NUM_THREADS="${CASE_CPUS}"
    export MPLCONFIGDIR="${case_mpl}"
    mkdir -p "${MPLCONFIGDIR}"

    echo "[case] label=${label}"
    echo "[case] epsilon_r=${epsilon_r}"
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
      --epsilon-r "${epsilon_r}" \
      --tanh-argument-scale-a 162.60162601626016 \
      --zero-limit finite \
      --max-iter 5 \
      --points-per-segment 40 \
      --path-kind gamma-m-k-gamma-kprime \
      --summary-mode both \
      --init bm:1
  ) > "${case_log}" 2>&1
}

pids=()
run_case "eps10_bm" "10" &
pids+=("$!")
run_case "eps4_bm" "4" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[job] case logs:"
ls -1 logs/no_crpa_gap_control_*_"${SLURM_JOB_ID:-manual}".out
echo "[job] complete status=${status}"
exit "${status}"
