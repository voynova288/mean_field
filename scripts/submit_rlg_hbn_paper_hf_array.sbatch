#!/bin/bash
# Submit with:
#   sbatch --array=0-11%6 scripts/submit_rlg_hbn_paper_hf_array.sbatch fig5 results/RnG_hBN/rlg_hbn_fig5_parallel
#   sbatch --array=0-25%8 scripts/submit_rlg_hbn_paper_hf_array.sbatch fig6 results/RnG_hBN/rlg_hbn_fig6_parallel 120 results/RnG_hBN/cache_fig6

#SBATCH -J rlg_hbn_hf
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -o logs/rlg_hbn_hf_%A_%a.out
#SBATCH -e logs/rlg_hbn_hf_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=... scripts/submit_rlg_hbn_paper_hf_array.sbatch <fig5|fig6> <output-root> [max_iter] [cache-root]" >&2
  exit 2
fi

PAPER_TARGET="$1"
OUTPUT_ROOT="$2"
MAX_ITER="${3:-80}"
CACHE_ROOT="${4:-${OUTPUT_ROOT}/cache}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"
SCREENING_SOLVER="${SCREENING_SOLVER:-grid}"
SCREENING_U_MIN_MEV="${SCREENING_U_MIN_MEV:--100}"
SCREENING_U_MAX_MEV="${SCREENING_U_MAX_MEV:-200}"
SCREENING_U_GRID_POINTS="${SCREENING_U_GRID_POINTS:-121}"
SCREENING_MESH_SIZE="${SCREENING_MESH_SIZE:-}"
TASKS_PER_ALLOCATION="${TASKS_PER_ALLOCATION:-1}"
PRECISION="${PRECISION:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
BETA="${BETA:-1.0}"
ODA_STALL_THRESHOLD="${ODA_STALL_THRESHOLD:-1e-3}"

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
    ;;
  fig6)
    XI_VALUES=(0 1)
    V_VALUES=(64)
    DEFAULT_SCREENING_MESH_SIZE=18
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
if [[ "${PAPER_TARGET}" == "fig6" ]]; then
  INIT_SEED_SPECS=(
    flavor:1 flavor:2
    bm:1
    perturbed:1 perturbed:2 perturbed:3 perturbed:4
    random:1 random:2 random:3 random:4
  )
else
  INIT_SEED_SPECS=(flavor:1 bm:1 perturbed:1)
fi

PANEL_COUNT=$((${#XI_VALUES[@]} * ${#V_VALUES[@]}))
TASKS_PER_PANEL=${#INIT_SEED_SPECS[@]}
TASK_COUNT=$((PANEL_COUNT * TASKS_PER_PANEL))
if (( TASKS_PER_ALLOCATION < 1 )); then
  echo "TASKS_PER_ALLOCATION must be >= 1, got ${TASKS_PER_ALLOCATION}" >&2
  exit 2
fi
if (( TASKS_PER_ALLOCATION > 1 )); then
  ALLOCATION_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  START_TASK=$((ALLOCATION_TASK_ID * TASKS_PER_ALLOCATION))
  END_TASK=$((START_TASK + TASKS_PER_ALLOCATION - 1))
  if (( END_TASK >= TASK_COUNT )); then
    END_TASK=$((TASK_COUNT - 1))
  fi
  if (( START_TASK >= TASK_COUNT )); then
    echo "[pack] allocation_task_id=${ALLOCATION_TASK_ID} has no work for ${PAPER_TARGET}"
    exit 0
  fi
  THREADS_PER_PROCESS="${THREADS_PER_PROCESS:-$(( ${SLURM_CPUS_PER_TASK:-1} / TASKS_PER_ALLOCATION ))}"
  if (( THREADS_PER_PROCESS < 1 )); then
    echo "THREADS_PER_PROCESS must be >= 1, got ${THREADS_PER_PROCESS}" >&2
    exit 2
  fi
  echo "[pack] allocation_task_id=${ALLOCATION_TASK_ID}"
  echo "[pack] task_range=${START_TASK}-${END_TASK}"
  echo "[pack] tasks_per_allocation=${TASKS_PER_ALLOCATION}"
  echo "[pack] allocation_cpus=${SLURM_CPUS_PER_TASK:-}"
  echo "[pack] threads_per_process=${THREADS_PER_PROCESS}"

  mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}/tasks" "${CACHE_ROOT}"
  PACK_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
  declare -a PIDS=()
  declare -a CHILD_IDS=()
  for CHILD_TASK_ID in $(seq "${START_TASK}" "${END_TASK}"); do
    CHILD_OUT="${REPO_ROOT}/logs/rlg_fig6_hfinit_${PACK_TAG}_task${CHILD_TASK_ID}.out"
    CHILD_ERR="${REPO_ROOT}/logs/rlg_fig6_hfinit_${PACK_TAG}_task${CHILD_TASK_ID}.err"
    echo "[pack] launch task=${CHILD_TASK_ID} out=${CHILD_OUT}"
    (
      export TASKS_PER_ALLOCATION=1
      export SLURM_ARRAY_TASK_ID="${CHILD_TASK_ID}"
      export SLURM_CPUS_PER_TASK="${THREADS_PER_PROCESS}"
      bash "${BASH_SOURCE[0]}" "${PAPER_TARGET}" "${OUTPUT_ROOT}" "${MAX_ITER}" "${CACHE_ROOT}"
    ) >"${CHILD_OUT}" 2>"${CHILD_ERR}" &
    PIDS+=("$!")
    CHILD_IDS+=("${CHILD_TASK_ID}")
  done

  FAILED=0
  for IDX in "${!PIDS[@]}"; do
    if wait "${PIDS[$IDX]}"; then
      echo "[pack] done task=${CHILD_IDS[$IDX]}"
    else
      STATUS=$?
      echo "[pack] failed task=${CHILD_IDS[$IDX]} status=${STATUS}" >&2
      FAILED=1
    fi
  done
  exit "${FAILED}"
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Array task ${TASK_ID} outside valid range 0..$((TASK_COUNT - 1)) for ${PAPER_TARGET}" >&2
  exit 2
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

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}/tasks" "${CACHE_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_hbn_${SLURM_JOB_ID:-manual}_${TASK_ID}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
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

TASK_DIR="${OUTPUT_ROOT}/tasks/task_${TASK_ID}_${PANEL}_${INIT_MODE}_seed${SEED}"
echo "[job] paper_target=${PAPER_TARGET}"
echo "[job] task_id=${TASK_ID}/${TASK_COUNT}"
echo "[job] panel=${PANEL}"
echo "[job] init_mode=${INIT_MODE}"
echo "[job] seed=${SEED}"
echo "[job] output_dir=${TASK_DIR}"
echo "[job] cache_root=${CACHE_ROOT}"
echo "[job] cache_policy=${CACHE_POLICY}"
echo "[job] screening_solver=${SCREENING_SOLVER}"
echo "[job] screening_mesh_size=${SCREENING_MESH_SIZE}"
echo "[job] screening_grid=${SCREENING_U_MIN_MEV}:${SCREENING_U_MAX_MEV}:${SCREENING_U_GRID_POINTS}"
echo "[job] precision=${PRECISION}"
echo "[job] checkpoint_interval=${CHECKPOINT_INTERVAL}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] numba_threads=${NUMBA_NUM_THREADS:-}"
echo "[job] remote_chunk_bands=${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS}"
echo "[job] mmap_overlap_cache=${MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE}"

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
