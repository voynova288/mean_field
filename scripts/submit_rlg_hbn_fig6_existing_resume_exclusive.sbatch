#!/bin/bash
# Resume one existing RLG/hBN Fig. 6 HF initialization per exclusive node.
#
# This wrapper preserves the old task directory layout from
# fig6_hf_bands_parallel_20260516_161014 while skipping duplicate deterministic
# BM seeds. Each Slurm array element runs exactly one (xi, init_mode, seed) spec
# and resumes from runs/<init>_seed<seed>/checkpoints/hf_checkpoint_latest.npz
# when present.
#
# Usage:
#   sbatch --array=0-25%8 scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch \
#     <output-root> <cache-root> [max_iter]

#SBATCH -J rlg_fig6_resume
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -o logs/rlg_fig6_resume_%A_%a.out
#SBATCH -e logs/rlg_fig6_resume_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=0-25%N $0 <output-root> <cache-root> [max_iter]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
CACHE_ROOT="$2"
MAX_ITER="${3:-120}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"
SCREENING_SOLVER="${SCREENING_SOLVER:-grid}"
SCREENING_U_MIN_MEV="${SCREENING_U_MIN_MEV:--100}"
SCREENING_U_MAX_MEV="${SCREENING_U_MAX_MEV:-200}"
SCREENING_U_GRID_POINTS="${SCREENING_U_GRID_POINTS:-121}"
SCREENING_MESH_SIZE="${SCREENING_MESH_SIZE:-18}"
PRECISION="${PRECISION:-1e-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
BETA="${BETA:-1.0}"
ODA_STALL_THRESHOLD="${ODA_STALL_THRESHOLD:-1e-3}"

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

# Old 32-task Fig. 6 layout, with duplicate deterministic BM seeds removed.
OLD_TASK_IDS=(
  0 1 2 3 4
  8 9 10 11 12 13 14 15
  16 17 18 19 20
  24 25 26 27 28 29 30 31
)
SPEC_COUNT=${#OLD_TASK_IDS[@]}
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= SPEC_COUNT )); then
  echo "Array index ${ARRAY_INDEX} outside valid range [0, $((SPEC_COUNT - 1))]" >&2
  exit 2
fi

TASK_ID="${OLD_TASK_IDS[$ARRAY_INDEX]}"
XI_VALUES=(0 1)
V_VALUES=(64)
INIT_MODES=(flavor bm perturbed random)
SEEDS=(1 2 3 4)
TASKS_PER_PANEL=16

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

if [[ "${INIT_MODE}" == "flavor" && "${SEED}" -gt 2 ]]; then
  echo "[skip] flavor seed ${SEED} is no longer part of the Fig. 6 production set; keeping only flavor:1,2"
  exit 0
fi

if [[ "${INIT_MODE}" == "bm" && "${SEED}" != "1" ]]; then
  echo "Refusing duplicate deterministic BM task_id=${TASK_ID} init=${INIT_MODE} seed=${SEED}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${TASK_DIR}" "${CACHE_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_hbn_fig6_resume_${SLURM_JOB_ID:-manual}_${ARRAY_INDEX}}"
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

echo "[job] resume_existing_fig6=true"
echo "[job] array_index=${ARRAY_INDEX}/${SPEC_COUNT}"
echo "[job] old_task_id=${TASK_ID}"
echo "[job] panel=${PANEL}"
echo "[job] init_mode=${INIT_MODE}"
echo "[job] seed=${SEED}"
echo "[job] run_specs=${INIT_MODE}:${SEED}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] task_dir=${TASK_DIR}"
echo "[job] panel_dir=${PANEL_DIR}"
echo "[job] cache_root=${CACHE_ROOT}"
echo "[job] cache_policy=${CACHE_POLICY}"
echo "[job] max_iter=${MAX_ITER}"
echo "[job] precision=${PRECISION}"
echo "[job] checkpoint_interval=${CHECKPOINT_INTERVAL}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] numba_threads=${NUMBA_NUM_THREADS:-}"
echo "[job] remote_chunk_bands=${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS}"
echo "[job] mmap_overlap_cache=${MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE}"

python -m mean_field.devtools.run_rlg_hbn_paper_hf \
  --paper-target fig6 \
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
  --run-specs "${INIT_MODE}:${SEED}" \
  --max-iter "${MAX_ITER}" \
  --precision "${PRECISION}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --beta "${BETA}" \
  --oda-stall-threshold "${ODA_STALL_THRESHOLD}"
