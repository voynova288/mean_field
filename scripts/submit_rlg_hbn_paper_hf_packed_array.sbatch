#!/bin/bash
# Packed RnG/hBN paper HF array launcher.
#
# Each Slurm array index owns a small group of independent paper-HF tasks and
# runs them concurrently on one node. This keeps CPU allocations saturated even
# when one numba process does not scale linearly to all reserved cores.

#SBATCH -J rlg_hbn_hf_pack
#SBATCH -p regular256
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/rlg_hbn_hf_pack_%A_%a.out
#SBATCH -e logs/rlg_hbn_hf_pack_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=... scripts/submit_rlg_hbn_paper_hf_packed_array.sbatch <fig5|fig6> <output-root> [max_iter] [tasks_per_node] [threads_per_task] [cache-root]" >&2
  exit 2
fi

PAPER_TARGET="$1"
OUTPUT_ROOT="$2"
MAX_ITER="${3:-80}"
TASKS_PER_NODE="${4:-2}"
THREADS_PER_TASK="${5:-32}"
CACHE_ROOT="${6:-${OUTPUT_ROOT}/cache}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"
SCREENING_SOLVER="${SCREENING_SOLVER:-grid}"
SCREENING_U_MIN_MEV="${SCREENING_U_MIN_MEV:--100}"
SCREENING_U_MAX_MEV="${SCREENING_U_MAX_MEV:-200}"
SCREENING_U_GRID_POINTS="${SCREENING_U_GRID_POINTS:-121}"
SCREENING_MESH_SIZE="${SCREENING_MESH_SIZE:-}"
PRECISION="${PRECISION:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
BETA="${BETA:-1.0}"
ODA_STALL_THRESHOLD="${ODA_STALL_THRESHOLD:-1e-3}"

if (( TASKS_PER_NODE < 1 )); then
  echo "tasks_per_node must be >= 1, got ${TASKS_PER_NODE}" >&2
  exit 2
fi
if (( THREADS_PER_TASK < 1 )); then
  echo "threads_per_task must be >= 1, got ${THREADS_PER_TASK}" >&2
  exit 2
fi
RESERVED_CPUS="${SLURM_CPUS_PER_TASK:-1}"
REQUESTED_CPUS=$((TASKS_PER_NODE * THREADS_PER_TASK))
if (( REQUESTED_CPUS > RESERVED_CPUS )); then
  if [[ "${ALLOW_OVERSUBSCRIBE:-0}" != "1" ]]; then
    echo "tasks_per_node * threads_per_task = ${REQUESTED_CPUS}, but Slurm reserved ${RESERVED_CPUS} CPUs" >&2
    echo "Set ALLOW_OVERSUBSCRIBE=1 only after a live utilization probe shows underuse." >&2
    exit 2
  fi
  echo "[warn] overpacking threads: requested_threads=${REQUESTED_CPUS} reserved_cpus=${RESERVED_CPUS}" >&2
fi

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
    DEFAULT_SCREENING_MESH_SIZE=12
    INIT_SEED_SPECS=(
      flavor:1
      bm:1
      perturbed:1
    )
    ;;
  fig6)
    XI_VALUES=(0 1)
    V_VALUES=(64)
    DEFAULT_SCREENING_MESH_SIZE=18
    INIT_SEED_SPECS=(
      flavor:1 flavor:2
      bm:1
      perturbed:1 perturbed:2 perturbed:3 perturbed:4
      random:1 random:2 random:3 random:4
    )
    ;;
  *)
    echo "Unsupported paper target: ${PAPER_TARGET}" >&2
    exit 2
    ;;
esac
SCREENING_MESH_SIZE="${SCREENING_MESH_SIZE:-${DEFAULT_SCREENING_MESH_SIZE}}"
if [[ -z "${PRECISION}" ]]; then
  if [[ "${PAPER_TARGET}" == "fig6" ]]; then
    PRECISION="1e-4"
  else
    PRECISION="1e-6"
  fi
fi

PANEL_COUNT=$((${#XI_VALUES[@]} * ${#V_VALUES[@]}))
TASKS_PER_PANEL=${#INIT_SEED_SPECS[@]}
TASK_COUNT=$((PANEL_COUNT * TASKS_PER_PANEL))
GROUP_ID="${SLURM_ARRAY_TASK_ID:-0}"
FIRST_TASK_ID=$((GROUP_ID * TASKS_PER_NODE))
if (( FIRST_TASK_ID >= TASK_COUNT )); then
  echo "Array group ${GROUP_ID} outside valid range for ${TASK_COUNT} tasks at ${TASKS_PER_NODE} tasks/node" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}/tasks" "${CACHE_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export MEAN_FIELD_RLG_HBN_USE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS="${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS:-16}"
export MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE="${MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE:-0}"

python - <<'PY'
import numba
print(f"[numba] version={numba.__version__}", flush=True)
PY

echo "[job] paper_target=${PAPER_TARGET}"
echo "[job] group_id=${GROUP_ID}"
echo "[job] task_count=${TASK_COUNT}"
echo "[job] tasks_per_node=${TASKS_PER_NODE}"
echo "[job] threads_per_task=${THREADS_PER_TASK}"
echo "[job] precision=${PRECISION}"
echo "[job] checkpoint_interval=${CHECKPOINT_INTERVAL}"
echo "[job] reserved_cpus=${RESERVED_CPUS}"
echo "[job] requested_cpus=${REQUESTED_CPUS}"
echo "[job] remote_chunk_bands=${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS}"
echo "[job] mmap_overlap_cache=${MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] cache_root=${CACHE_ROOT}"
echo "[job] cache_policy=${CACHE_POLICY}"
echo "[job] screening_solver=${SCREENING_SOLVER}"
echo "[job] screening_mesh_size=${SCREENING_MESH_SIZE}"
echo "[job] screening_grid=${SCREENING_U_MIN_MEV}:${SCREENING_U_MAX_MEV}:${SCREENING_U_GRID_POINTS}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"

pids=()
task_labels=()

for ((slot = 0; slot < TASKS_PER_NODE; slot++)); do
  TASK_ID=$((FIRST_TASK_ID + slot))
  if (( TASK_ID >= TASK_COUNT )); then
    continue
  fi

  PANEL_INDEX=$((TASK_ID / TASKS_PER_PANEL))
  WITHIN_PANEL=$((TASK_ID % TASKS_PER_PANEL))
  XI_INDEX=$((PANEL_INDEX / ${#V_VALUES[@]}))
  V_INDEX=$((PANEL_INDEX % ${#V_VALUES[@]}))

  XI="${XI_VALUES[$XI_INDEX]}"
  V_MEV="${V_VALUES[$V_INDEX]}"
  INIT_SEED_SPEC="${INIT_SEED_SPECS[$WITHIN_PANEL]}"
  INIT_MODE="${INIT_SEED_SPEC%%:*}"
  SEED="${INIT_SEED_SPEC##*:}"
  PANEL="xi${XI}_V$(printf "%03d" "${V_MEV}")meV"
  TASK_DIR="${OUTPUT_ROOT}/tasks/task_${TASK_ID}_${PANEL}_${INIT_MODE}_seed${SEED}"
  mkdir -p "${TASK_DIR}"

  echo "[launch] slot=${slot} task_id=${TASK_ID}/${TASK_COUNT} panel=${PANEL} init=${INIT_MODE} seed=${SEED} dir=${TASK_DIR}"
  (
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_hbn_${SLURM_JOB_ID:-manual}_${GROUP_ID}_${TASK_ID}}"
    mkdir -p "${MPLCONFIGDIR}"
    export OMP_NUM_THREADS="${THREADS_PER_TASK}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_TASK}"
    export MKL_NUM_THREADS="${THREADS_PER_TASK}"
    export BLIS_NUM_THREADS="${THREADS_PER_TASK}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_TASK}"
    export NUMBA_NUM_THREADS="${THREADS_PER_TASK}"
    echo "[task] start task_id=${TASK_ID} panel=${PANEL} init=${INIT_MODE} seed=${SEED} threads=${THREADS_PER_TASK} hostname=$(hostname)"
    python -m mean_field.devtools.run_rlg_hbn_paper_hf \
      --paper-target "${PAPER_TARGET}" \
      --output-dir "${TASK_DIR}" \
      --cache-dir "${CACHE_ROOT}" \
      --cache-policy "${CACHE_POLICY}" \
      --screening-solver "${SCREENING_SOLVER}" \
      --screening-u-min-mev "${SCREENING_U_MIN_MEV}" \
      --screening-u-max-mev "${SCREENING_U_MAX_MEV}" \
      --screening-u-grid-points "${SCREENING_U_GRID_POINTS}" \
      --screening-mesh-size "${SCREENING_MESH_SIZE}" \
      --skip-screening-check \
      --xi-values "${XI}" \
      --v-values-mev "${V_MEV}" \
      --init-modes "${INIT_MODE}" \
      --seeds "${SEED}" \
      --max-iter "${MAX_ITER}" \
      --precision "${PRECISION}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --beta "${BETA}" \
      --oda-stall-threshold "${ODA_STALL_THRESHOLD}"
    echo "[task] done task_id=${TASK_ID}"
  ) > "${TASK_DIR}/task_stdout.log" 2> "${TASK_DIR}/task_stderr.log" &
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

exit "${status}"
