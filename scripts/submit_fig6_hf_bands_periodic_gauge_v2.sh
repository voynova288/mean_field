#!/bin/bash
# Submit a fresh RLG/hBN Fig. 6 chain after the periodic-gauge relabel fix.
#
# This intentionally uses a new output root and a new cache root.  The first
# two array tasks warm the xi=0 and xi=1 panel caches before the full array is
# released, avoiding concurrent writes to the same fresh basis/overlap cache.

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs results/RnG_hBN

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPO_ROOT}/results/RnG_hBN/fig6_hf_bands_periodic_gauge_v2_${STAMP}}"
CACHE="${CACHE:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_periodic_gauge_v2_${STAMP}}"
PARTITION="${PARTITION:-regular6430}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
WALLTIME="${WALLTIME:-7-00:00:00}"
MAIN_ARRAY_THROTTLE="${MAIN_ARRAY_THROTTLE:-5}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
MAX_ITER="${MAX_ITER:-120}"
CACHE_POLICY="${CACHE_POLICY:-reuse}"
SCREENING_SOLVER="${SCREENING_SOLVER:-grid}"
SCREENING_U_MIN_MEV="${SCREENING_U_MIN_MEV:--100}"
SCREENING_U_MAX_MEV="${SCREENING_U_MAX_MEV:-200}"
SCREENING_U_GRID_POINTS="${SCREENING_U_GRID_POINTS:-121}"
SCREENING_MESH_SIZE="${SCREENING_MESH_SIZE:-18}"
PRECISION="${PRECISION:-1e-6}"
BETA="${BETA:-1.0}"
ODA_STALL_THRESHOLD="${ODA_STALL_THRESHOLD:-1e-3}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-48}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
DPI="${DPI:-180}"

RESUME_TASK_COUNT=26
RESUME_ARRAY_MAX=$((RESUME_TASK_COUNT - 1))
MANIFEST="${OUT}/periodic_gauge_v2_jobs.tsv"

mkdir -p "${OUT}" "${CACHE}"

SBATCH_EXCLUDE_ARGS=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SBATCH_EXCLUDE_ARGS=(--exclude="${EXCLUDE_NODES}")
fi

EXPORT_COMMON="ALL,MEAN_FIELD_RLG_HBN_USE_NUMBA=1,MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1,MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS=16"
EXPORT_HF="${EXPORT_COMMON},CACHE_POLICY=${CACHE_POLICY},SCREENING_SOLVER=${SCREENING_SOLVER},SCREENING_U_MIN_MEV=${SCREENING_U_MIN_MEV},SCREENING_U_MAX_MEV=${SCREENING_U_MAX_MEV},SCREENING_U_GRID_POINTS=${SCREENING_U_GRID_POINTS},SCREENING_MESH_SIZE=${SCREENING_MESH_SIZE},PRECISION=${PRECISION},BETA=${BETA},ODA_STALL_THRESHOLD=${ODA_STALL_THRESHOLD}"

{
  echo -e "stage\tjob_id\tdependency\toutput\tcommand"
} > "${MANIFEST}"

echo "[submit] out=${OUT}"
echo "[submit] cache=${CACHE}"
echo "[submit] partition=${PARTITION} cpus=${CPUS_PER_TASK} exclusive mem=0"
if [[ -n "${EXCLUDE_NODES}" ]]; then
  echo "[submit] exclude=${EXCLUDE_NODES}"
fi
echo "[submit] warmup xi0 array=0"
echo "[submit] warmup xi1 array=13"
echo "[submit] main array=0-${RESUME_ARRAY_MAX}%${MAIN_ARRAY_THROTTLE}"

WARMUP_XI0_JOB="${WARMUP_XI0_JOB:-}"
if [[ -z "${WARMUP_XI0_JOB}" ]]; then
  WARMUP_XI0_JOB="$(
    sbatch --parsable \
      -J rlg_fig6_v2_warm_xi0 \
      -p "${PARTITION}" \
      "${SBATCH_EXCLUDE_ARGS[@]}" \
      -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive --mem=0 \
      -t "${WALLTIME}" \
      -o logs/rlg_fig6_v2_warm_xi0_%A_%a.out \
      -e logs/rlg_fig6_v2_warm_xi0_%A_%a.err \
      --array="0" \
      --export="${EXPORT_HF}" \
      scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch \
      "${OUT}" "${CACHE}" "${MAX_ITER}"
  )"
  WARMUP_XI0_JOB="${WARMUP_XI0_JOB%%;*}"
else
  echo "[submit] reusing warmup_xi0=${WARMUP_XI0_JOB}"
fi
echo -e "warmup_xi0\t${WARMUP_XI0_JOB}\t\t${OUT}\txi0 cache warmup task" >> "${MANIFEST}"

WARMUP_XI1_JOB="${WARMUP_XI1_JOB:-}"
if [[ -z "${WARMUP_XI1_JOB}" ]]; then
  WARMUP_XI1_JOB="$(
    sbatch --parsable \
      -J rlg_fig6_v2_warm_xi1 \
      -p "${PARTITION}" \
      "${SBATCH_EXCLUDE_ARGS[@]}" \
      -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive --mem=0 \
      -t "${WALLTIME}" \
      -o logs/rlg_fig6_v2_warm_xi1_%A_%a.out \
      -e logs/rlg_fig6_v2_warm_xi1_%A_%a.err \
      --array="13" \
      --export="${EXPORT_HF}" \
      scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch \
      "${OUT}" "${CACHE}" "${MAX_ITER}"
  )"
  WARMUP_XI1_JOB="${WARMUP_XI1_JOB%%;*}"
else
  echo "[submit] reusing warmup_xi1=${WARMUP_XI1_JOB}"
fi
echo -e "warmup_xi1\t${WARMUP_XI1_JOB}\t\t${OUT}\txi1 cache warmup task" >> "${MANIFEST}"

WARMUP_DEPENDENCY="afterok:${WARMUP_XI0_JOB}:${WARMUP_XI1_JOB}"

MAIN_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_v2_hf \
    -p "${PARTITION}" \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_v2_hf_%A_%a.out \
    -e logs/rlg_fig6_v2_hf_%A_%a.err \
    --dependency="${WARMUP_DEPENDENCY}" \
    --array="0-${RESUME_ARRAY_MAX}%${MAIN_ARRAY_THROTTLE}" \
    --export="${EXPORT_HF}" \
    scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch \
    "${OUT}" "${CACHE}" "${MAX_ITER}"
)"
MAIN_JOB="${MAIN_JOB%%;*}"
echo -e "hf_array\t${MAIN_JOB}\t${WARMUP_DEPENDENCY}\t${OUT}\t${RESUME_TASK_COUNT} existing-layout init tasks, v2 cache" >> "${MANIFEST}"

MERGE_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_v2_merge \
    -p "${PARTITION}" \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t 04:00:00 \
    -o logs/rlg_fig6_v2_merge_%j.out \
    -e logs/rlg_fig6_v2_merge_%j.err \
    --dependency="afterok:${MAIN_JOB}" \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.merge_rlg_hbn_parallel_hf \
      --source-root "${OUT}" \
      --paper-target fig6
)"
MERGE_JOB="${MERGE_JOB%%;*}"
echo -e "merge\t${MERGE_JOB}\tafterok:${MAIN_JOB}\t${OUT}\tmerge_rlg_hbn_parallel_hf" >> "${MANIFEST}"

PLOT_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_v2_plot \
    -p "${PARTITION}" \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_v2_plot_%j.out \
    -e logs/rlg_fig6_v2_plot_%j.err \
    --dependency="afterok:${MERGE_JOB}" \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.plot_rlg_hbn_paper_hf_bands \
      --source-dir "${OUT}" \
      --paper-target fig6 \
      --cache-dir "${CACHE}" \
      --cache-policy "${CACHE_POLICY}" \
      --points-per-segment "${POINTS_PER_SEGMENT}" \
      --chunk-size "${CHUNK_SIZE}" \
      --dpi "${DPI}"
)"
PLOT_JOB="${PLOT_JOB%%;*}"
echo -e "plot\t${PLOT_JOB}\tafterok:${MERGE_JOB}\t${OUT}\tplot_rlg_hbn_paper_hf_bands" >> "${MANIFEST}"

VERIFY_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_v2_verify \
    -p "${PARTITION}" \
    "${SBATCH_EXCLUDE_ARGS[@]}" \
    -N 1 --ntasks=1 --cpus-per-task=4 --mem=8G \
    -t 00:30:00 \
    -o logs/rlg_fig6_v2_verify_%j.out \
    -e logs/rlg_fig6_v2_verify_%j.err \
    --dependency="afterok:${PLOT_JOB}" \
    scripts/run_rlg_hbn_fig6_verify.sbatch "${OUT}"
)"
VERIFY_JOB="${VERIFY_JOB%%;*}"
echo -e "verify\t${VERIFY_JOB}\tafterok:${PLOT_JOB}\t${OUT}\tverify_rlg_hbn_fig6_outputs" >> "${MANIFEST}"

echo "[submitted] warmup_xi0=${WARMUP_XI0_JOB}"
echo "[submitted] warmup_xi1=${WARMUP_XI1_JOB}"
echo "[submitted] hf_array=${MAIN_JOB}"
echo "[submitted] merge=${MERGE_JOB}"
echo "[submitted] plot=${PLOT_JOB}"
echo "[submitted] verify=${VERIFY_JOB}"
echo "[submitted] manifest=${MANIFEST}"
cat "${MANIFEST}"
