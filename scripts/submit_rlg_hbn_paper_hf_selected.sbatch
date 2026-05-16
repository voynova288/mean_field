#!/bin/bash
# Run selected RnG/hBN paper HF task ids inside one Slurm allocation.
#
# This is a recovery wrapper for partially completed packed-array runs. It keeps
# independent selected tasks packed in one node allocation without rerunning an
# entire packed array group.
#
# Usage:
#   sbatch scripts/submit_rlg_hbn_paper_hf_selected.sbatch \
#     <fig5|fig6> <output-root> <task-id-csv> [max_iter] [threads_per_task]
#
# Example:
#   sbatch scripts/submit_rlg_hbn_paper_hf_selected.sbatch \
#     fig5 results/RnG_hBN/rlg_hbn_fig5_packed_20260509_181024 9,11 80 32

#SBATCH -J rlg_hbn_hf_selected
#SBATCH -p regular256
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH -t 3-00:00:00
#SBATCH -o logs/rlg_hbn_hf_selected_%j.out
#SBATCH -e logs/rlg_hbn_hf_selected_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 <fig5|fig6> <output-root> <task-id-csv> [max_iter] [threads_per_task]" >&2
  exit 2
fi

PAPER_TARGET="$1"
OUTPUT_ROOT="$2"
TASK_ID_CSV="$3"
MAX_ITER="${4:-80}"
THREADS_PER_TASK="${5:-32}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

case "${PAPER_TARGET}" in
  fig5)
    XI_VALUES=(1)
    V_VALUES=(40 48 56 64)
    ;;
  fig6)
    XI_VALUES=(0 1)
    V_VALUES=(64)
    ;;
  *)
    echo "Unsupported paper target: ${PAPER_TARGET}" >&2
    exit 2
    ;;
esac

INIT_MODES=(flavor bm perturbed)
SEEDS=(1)
TASKS_PER_PANEL=$((${#INIT_MODES[@]} * ${#SEEDS[@]}))
TASK_COUNT=$((${#XI_VALUES[@]} * ${#V_VALUES[@]} * TASKS_PER_PANEL))
RESERVED_CPUS="${SLURM_CPUS_PER_TASK:-1}"

IFS=',' read -r -a SELECTED_TASK_IDS <<< "${TASK_ID_CSV}"
if (( ${#SELECTED_TASK_IDS[@]} < 1 )); then
  echo "No task ids provided in ${TASK_ID_CSV}" >&2
  exit 2
fi

REQUESTED_CPUS=$((${#SELECTED_TASK_IDS[@]} * THREADS_PER_TASK))
if (( REQUESTED_CPUS > RESERVED_CPUS )); then
  echo "selected_task_count * threads_per_task = ${REQUESTED_CPUS}, but Slurm reserved ${RESERVED_CPUS} CPUs" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}/tasks"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export MEAN_FIELD_RLG_HBN_USE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS="${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS:-16}"

python - <<'PY'
import numba
print(f"[numba] version={numba.__version__}", flush=True)
PY

echo "[job] paper_target=${PAPER_TARGET}"
echo "[job] selected_task_ids=${TASK_ID_CSV}"
echo "[job] task_count=${TASK_COUNT}"
echo "[job] threads_per_task=${THREADS_PER_TASK}"
echo "[job] reserved_cpus=${RESERVED_CPUS}"
echo "[job] requested_cpus=${REQUESTED_CPUS}"
echo "[job] remote_chunk_bands=${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"

pids=()
task_labels=()

for raw_task_id in "${SELECTED_TASK_IDS[@]}"; do
  TASK_ID="$(echo "${raw_task_id}" | tr -d '[:space:]')"
  if [[ -z "${TASK_ID}" ]]; then
    continue
  fi
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid task id ${TASK_ID}" >&2
    exit 2
  fi
  if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
    echo "Task id ${TASK_ID} outside valid range [0, $((TASK_COUNT - 1))]" >&2
    exit 2
  fi

  PANEL_INDEX=$((TASK_ID / TASKS_PER_PANEL))
  WITHIN_PANEL=$((TASK_ID % TASKS_PER_PANEL))
  INIT_INDEX=$((WITHIN_PANEL / ${#SEEDS[@]}))
  SEED_INDEX=$((WITHIN_PANEL % ${#SEEDS[@]}))
  XI_INDEX=$((PANEL_INDEX / ${#V_VALUES[@]}))
  V_INDEX=$((PANEL_INDEX % ${#V_VALUES[@]}))

  XI="${XI_VALUES[$XI_INDEX]}"
  V_MEV="${V_VALUES[$V_INDEX]}"
  INIT_MODE="${INIT_MODES[$INIT_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  PANEL="xi${XI}_V$(printf "%03d" "${V_MEV}")meV"
  TASK_DIR="${OUTPUT_ROOT}/tasks/task_${TASK_ID}_${PANEL}_${INIT_MODE}_seed${SEED}"
  PANEL_DIR="${TASK_DIR}/${PANEL}"
  mkdir -p "${TASK_DIR}"

  if [[ -s "${PANEL_DIR}/hf_convergence.json" && -s "${PANEL_DIR}/hf_ground_state.npz" ]]; then
    echo "[skip] task_id=${TASK_ID} panel=${PANEL} init=${INIT_MODE} seed=${SEED} already complete"
    continue
  fi

  echo "[launch] task_id=${TASK_ID}/${TASK_COUNT} panel=${PANEL} init=${INIT_MODE} seed=${SEED} dir=${TASK_DIR}"
  (
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_hbn_selected_${SLURM_JOB_ID:-manual}_${TASK_ID}}"
    mkdir -p "${MPLCONFIGDIR}"
    export OMP_NUM_THREADS="${THREADS_PER_TASK}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_TASK}"
    export MKL_NUM_THREADS="${THREADS_PER_TASK}"
    export BLIS_NUM_THREADS="${THREADS_PER_TASK}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_TASK}"
    export NUMBA_NUM_THREADS="${THREADS_PER_TASK}"
    echo "[resume] start task_id=${TASK_ID} panel=${PANEL} init=${INIT_MODE} seed=${SEED} threads=${THREADS_PER_TASK} hostname=$(hostname)"
    python -m mean_field.devtools.run_rlg_hbn_paper_hf \
      --paper-target "${PAPER_TARGET}" \
      --output-dir "${TASK_DIR}" \
      --xi-values "${XI}" \
      --v-values-mev "${V_MEV}" \
      --init-modes "${INIT_MODE}" \
      --seeds "${SEED}" \
      --max-iter "${MAX_ITER}"
    echo "[resume] done task_id=${TASK_ID}"
  ) >> "${TASK_DIR}/task_stdout.log" 2>> "${TASK_DIR}/task_stderr.log" &
  pids+=("$!")
  task_labels+=("${TASK_ID}:${PANEL}:${INIT_MODE}:seed${SEED}")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  label="${task_labels[$idx]}"
  if wait "${pid}"; then
    echo "[wait] success ${label}"
  else
    rc=$?
    echo "[wait] failure ${label} rc=${rc}" >&2
    status=1
  fi
done

if (( ${#pids[@]} == 0 )); then
  echo "[done] all selected tasks were already complete"
fi

exit "${status}"
