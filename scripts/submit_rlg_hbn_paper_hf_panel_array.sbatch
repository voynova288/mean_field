#!/bin/bash
# Run one RnG/hBN paper HF panel per Slurm array task.
#
# Compared with the packed init-mode launcher, this shares the expensive
# screened-basis, overlap, and remote-background construction across all init
# modes for the same panel. The Python runner writes per-iteration checkpoints
# under each panel's runs/<init>_seed<seed>/checkpoints directory.
#
# Usage:
#   sbatch --array=0-1%1 scripts/submit_rlg_hbn_paper_hf_panel_array.sbatch \
#     fig6 <output-root> [max_iter] [threads_per_task]

#SBATCH -J rlg_hbn_hf_panel
#SBATCH -p long
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -o logs/rlg_hbn_hf_panel_%A_%a.out
#SBATCH -e logs/rlg_hbn_hf_panel_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=... $0 <fig5|fig6> <output-root> [max_iter] [threads_per_task]" >&2
  exit 2
fi

PAPER_TARGET="$1"
OUTPUT_ROOT="$2"
MAX_ITER="${3:-160}"
THREADS_PER_TASK="${4:-${SLURM_CPUS_PER_TASK:-64}}"

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

PANEL_COUNT=$((${#XI_VALUES[@]} * ${#V_VALUES[@]}))
PANEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( PANEL_INDEX < 0 || PANEL_INDEX >= PANEL_COUNT )); then
  echo "Panel index ${PANEL_INDEX} outside valid range [0, $((PANEL_COUNT - 1))]" >&2
  exit 2
fi

XI_INDEX=$((PANEL_INDEX / ${#V_VALUES[@]}))
V_INDEX=$((PANEL_INDEX % ${#V_VALUES[@]}))
XI="${XI_VALUES[$XI_INDEX]}"
V_MEV="${V_VALUES[$V_INDEX]}"
PANEL="xi${XI}_V$(printf "%03d" "${V_MEV}")meV"
TASK_DIR="${OUTPUT_ROOT}/tasks/panel_${PANEL_INDEX}_${PANEL}"

mkdir -p "${REPO_ROOT}/logs" "${TASK_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_hbn_panel_${SLURM_JOB_ID:-manual}_${PANEL_INDEX}}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export OMP_NUM_THREADS="${THREADS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_TASK}"
export MKL_NUM_THREADS="${THREADS_PER_TASK}"
export BLIS_NUM_THREADS="${THREADS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_TASK}"
export NUMBA_NUM_THREADS="${THREADS_PER_TASK}"
export MEAN_FIELD_RLG_HBN_USE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS="${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS:-16}"
mkdir -p "${MPLCONFIGDIR}"

python - <<'PY'
import numba
print(f"[numba] version={numba.__version__}", flush=True)
PY

echo "[job] paper_target=${PAPER_TARGET}"
echo "[job] panel_index=${PANEL_INDEX}/${PANEL_COUNT}"
echo "[job] panel=${PANEL}"
echo "[job] max_iter=${MAX_ITER}"
echo "[job] threads_per_task=${THREADS_PER_TASK}"
echo "[job] slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] remote_chunk_bands=${MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] task_dir=${TASK_DIR}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"

python -m mean_field.devtools.run_rlg_hbn_paper_hf \
  --paper-target "${PAPER_TARGET}" \
  --output-dir "${TASK_DIR}" \
  --xi-values "${XI}" \
  --v-values-mev "${V_MEV}" \
  --init-modes flavor,bm,perturbed \
  --seeds 1 \
  --max-iter "${MAX_ITER}"
