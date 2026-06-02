#!/bin/bash
# Small SCF-grid HF gate using a no-alias HF-compatible cRPA artifact.

#SBATCH -J crpa_alias_hf
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o logs/crpa_alias_hf_small_20260524_%j.out
#SBATCH -e logs/crpa_alias_hf_small_20260524_%j.err

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_alias_hf_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

export MEAN_FIELD_CRPA_SPLIT_MODE="${MEAN_FIELD_CRPA_SPLIT_MODE:-active_cnp_fock_reference_projector}"
export MEAN_FIELD_CRPA_REMOTE_BARE_SCALE="${MEAN_FIELD_CRPA_REMOTE_BARE_SCALE:-1.0}"

CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_alias_gate_20260524/hf_periodic_lk6_lg7_q5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_alias_gate_20260524/hf_small_lk6_lg3_q5}"
LK="${LK:-6}"
LG="${LG:-3}"
OVERLAP_LG="${OVERLAP_LG:-${LG}}"
MAX_ITER="${MAX_ITER:-5}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-40}"
INIT_SPEC="${INIT_SPEC:-vp:1}"
INIT_MODE="${INIT_SPEC%%:*}"
if [[ "${INIT_SPEC}" == *:* ]]; then
  INIT_SEED="${INIT_SPEC#*:}"
else
  INIT_SEED="1"
fi
printf -v INIT_SEED_PADDED "%03d" "${INIT_SEED}"
INITIAL_STATE="${INITIAL_STATE:-}"
INITIAL_STATE_RESAMPLE="${INITIAL_STATE_RESAMPLE:-none}"
RUN_TAG="${RUN_TAG:-crpa_alias_legal_${MEAN_FIELD_CRPA_SPLIT_MODE}_lk${LK}_lg${LG}_iter${MAX_ITER}}"

echo "[job] hostname=$(hostname)"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] split_mode=${MEAN_FIELD_CRPA_SPLIT_MODE}"
echo "[job] crpa_dir=${CRPA_DIR}"
echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] run_tag=${RUN_TAG}"
echo "[job] lk=${LK} lg=${LG} overlap_lg=${OVERLAP_LG} max_iter=${MAX_ITER}"
echo "[job] init=${INIT_SPEC} initial_state=${INITIAL_STATE}"

CMD=(python scripts/mean_field_tools.py run_custom_b0_hf_case
  --theta-deg 1.05 \
  --nu -3 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --lk "${LK}" \
  --lg "${LG}" \
  --overlap-lg "${OVERLAP_LG}" \
  --w0 79.7 \
  --w1 97.4 \
  --vf 2135.4 \
  --epsilon-r 4 \
  --tanh-argument-scale-a 162.60162601626016 \
  --zero-limit finite \
  --max-iter "${MAX_ITER}" \
  --points-per-segment "${POINTS_PER_SEGMENT}" \
  --path-kind gamma-m-k-gamma-kprime \
  --summary-mode both \
  --init "${INIT_SPEC}" \
  --crpa-dir "${CRPA_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear \
  --write-scf-path)

if [[ -n "${INITIAL_STATE}" ]]; then
  CMD+=(--initial-state "${INITIAL_STATE}" --initial-state-resample "${INITIAL_STATE_RESAMPLE}")
fi

"${CMD[@]}"

RUN_DIR="${OUTPUT_ROOT}/theta_105_nu_-3000_${RUN_TAG}"
PATH_DIR="${RUN_DIR}/path_bands"
SCF_TSV="${PATH_DIR}/theta_105_nu_-3000_init_${INIT_MODE}_seed_${INIT_SEED_PADDED}_lk${LK}_lg${LG}_hf_scf_path.tsv"
SUMMARY_JSON="${PATH_DIR}/theta_105_nu_-3000_init_${INIT_MODE}_seed_${INIT_SEED_PADDED}_lk${LK}_lg${LG}_hf_scf_path_summary.json"

python scripts/summarize_tbg_hf_scf_path_tsv.py \
  --tsv "${SCF_TSV}" \
  --nu -3 \
  --output-json "${SUMMARY_JSON}"

echo "[job] scf_summary=${SUMMARY_JSON}"
cat "${SUMMARY_JSON}"
echo "[job] scf_plot=${PATH_DIR}/theta_105_nu_-3000_init_${INIT_MODE}_seed_${INIT_SEED_PADDED}_lk${LK}_lg${LG}_scf_grid_band_plot.png"
echo "[job] complete"
