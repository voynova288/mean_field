#!/bin/bash
# Submit with:
#   sbatch --array=0-11%6 scripts/submit_rlg_hbn_paper_hf_array.sbatch fig5 results/RnG_hBN/rlg_hbn_fig5_parallel
#   sbatch --array=0-5%6 scripts/submit_rlg_hbn_paper_hf_array.sbatch fig6 results/RnG_hBN/rlg_hbn_fig6_parallel

#SBATCH -J rlg_hbn_hf
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=90G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/rlg_hbn_hf_%A_%a.out
#SBATCH -e logs/rlg_hbn_hf_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=... scripts/submit_rlg_hbn_paper_hf_array.sbatch <fig5|fig6> <output-root> [max_iter]" >&2
  exit 2
fi

PAPER_TARGET="$1"
OUTPUT_ROOT="$2"
MAX_ITER="${3:-80}"

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

PANEL_COUNT=$((${#XI_VALUES[@]} * ${#V_VALUES[@]}))
TASKS_PER_PANEL=$((${#INIT_MODES[@]} * ${#SEEDS[@]}))
TASK_COUNT=$((PANEL_COUNT * TASKS_PER_PANEL))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Array task ${TASK_ID} outside valid range 0..$((TASK_COUNT - 1)) for ${PAPER_TARGET}" >&2
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

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}/tasks"

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
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] numba_threads=${NUMBA_NUM_THREADS:-}"

python -m mean_field.devtools.run_rlg_hbn_paper_hf \
  --paper-target "${PAPER_TARGET}" \
  --output-dir "${TASK_DIR}" \
  --xi-values "${XI}" \
  --v-values-mev "${V_MEV}" \
  --init-modes "${INIT_MODE}" \
  --seeds "${SEED}" \
  --max-iter "${MAX_ITER}"
