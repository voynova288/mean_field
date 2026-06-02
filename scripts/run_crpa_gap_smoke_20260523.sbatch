#!/bin/bash
# Small-shell HF+cRPA diagnostic for the nu=-3 direct-gap/bandwidth gate.

#SBATCH -J mf_crpa_gap_smoke
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 00:50:00
#SBATCH -o logs/crpa_gap_smoke_20260523_%j.out
#SBATCH -e logs/crpa_gap_smoke_20260523_%j.err

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
HF_CRPA_DIR="${HF_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged}"
PAPER_CRPA_DIR="${PAPER_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_paper_ref_fixed_20260519_merged}"
CASE_CPUS="${CASE_CPUS:-16}"

if [[ ! -f "${HF_CRPA_DIR}/crpa_params.json" || ! -f "${HF_CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing HF-compatible cRPA artifact under ${HF_CRPA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${PAPER_CRPA_DIR}/crpa_params.json" || ! -f "${PAPER_CRPA_DIR}/screened_coulomb.npz" ]]; then
  echo "Missing paper-reference cRPA artifact under ${PAPER_CRPA_DIR}" >&2
  exit 2
fi

echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] case_cpus=${CASE_CPUS}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] hf_crpa_dir=${HF_CRPA_DIR}"
echo "[job] paper_crpa_dir=${PAPER_CRPA_DIR}"

run_case() {
  local label="$1"
  local crpa_dir="$2"
  local init_spec="$3"
  local allow_incompatible="$4"
  local run_tag="crpa_gap_smoke_${label}_lk12_lg3_iter5"
  local case_log="${REPO_ROOT}/logs/crpa_gap_smoke_${label}_${SLURM_JOB_ID:-manual}.out"
  local case_mpl="/tmp/mplconfig_crpa_gap_smoke_${label}_${SLURM_JOB_ID:-manual}"

  (
    export OMP_NUM_THREADS="${CASE_CPUS}"
    export OPENBLAS_NUM_THREADS="${CASE_CPUS}"
    export MKL_NUM_THREADS="${CASE_CPUS}"
    export BLIS_NUM_THREADS="${CASE_CPUS}"
    export NUMEXPR_NUM_THREADS="${CASE_CPUS}"
    export NUMBA_NUM_THREADS="${CASE_CPUS}"
    export MPLCONFIGDIR="${case_mpl}"
    mkdir -p "${MPLCONFIGDIR}"

    args=(
      python scripts/mean_field_tools.py run_custom_b0_hf_case
      --theta-deg 1.05
      --nu -3
      --output-root "${OUTPUT_ROOT}"
      --run-tag "${run_tag}"
      --lk 12
      --lg 3
      --overlap-lg 3
      --w0 79.7
      --w1 97.4
      --vf 2135.4
      --epsilon-r 4
      --tanh-argument-scale-a 162.60162601626016
      --zero-limit finite
      --max-iter 5
      --points-per-segment 40
      --path-kind gamma-m-k-gamma-kprime
      --summary-mode both
      --init "${init_spec}"
      --crpa-dir "${crpa_dir}"
      --fock-interpolation matrix_diagonal
      --path-fock-interpolation linear
    )
    if [[ "${allow_incompatible}" == "1" ]]; then
      args+=(--allow-incompatible-crpa --diagnostic-only)
    fi

    echo "[case] label=${label}"
    echo "[case] init=${init_spec}"
    echo "[case] crpa_dir=${crpa_dir}"
    echo "[case] allow_incompatible=${allow_incompatible}"
    echo "[case] threads=${CASE_CPUS}"
    "${args[@]}"
  ) > "${case_log}" 2>&1
}

pids=()
run_case "hfcompat_vp" "${HF_CRPA_DIR}" "vp:1" "0" &
pids+=("$!")
run_case "hfcompat_educated" "${HF_CRPA_DIR}" "educated:1" "0" &
pids+=("$!")
run_case "paperref_vp" "${PAPER_CRPA_DIR}" "vp:1" "1" &
pids+=("$!")
run_case "paperref_educated" "${PAPER_CRPA_DIR}" "educated:1" "1" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[job] case logs:"
ls -1 logs/crpa_gap_smoke_*_"${SLURM_JOB_ID:-manual}".out
echo "[job] complete status=${status}"
exit "${status}"
