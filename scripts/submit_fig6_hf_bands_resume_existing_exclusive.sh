#!/bin/bash
# Submit a recovery chain for an existing RLG/hBN Fig. 6 output tree.
#
# One HF initialization runs per exclusive CPU node, using the old task
# directory layout and skipping duplicate deterministic BM seeds.
#
# Submit from login002:
#   ssh login002 'cd /data/home/ziyuzhu/Mean_Field && bash scripts/submit_fig6_hf_bands_resume_existing_exclusive.sh'

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs results/RnG_hBN

OUT="${OUT:-${REPO_ROOT}/results/RnG_hBN/fig6_hf_bands_parallel_20260516_161014}"
CACHE="${CACHE:-${REPO_ROOT}/results/RnG_hBN/cache_fig6}"
PARTITION="${PARTITION:-regular6430}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
WALLTIME="${WALLTIME:-7-00:00:00}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-8}"
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
MANIFEST="${OUT}/resume_existing_exclusive_jobs.tsv"

mkdir -p "${OUT}" "${CACHE}"

EXPORT_COMMON="ALL,MEAN_FIELD_RLG_HBN_USE_NUMBA=1,MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1,MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS=16"
EXPORT_HF="${EXPORT_COMMON},CACHE_POLICY=${CACHE_POLICY},SCREENING_SOLVER=${SCREENING_SOLVER},SCREENING_U_MIN_MEV=${SCREENING_U_MIN_MEV},SCREENING_U_MAX_MEV=${SCREENING_U_MAX_MEV},SCREENING_U_GRID_POINTS=${SCREENING_U_GRID_POINTS},SCREENING_MESH_SIZE=${SCREENING_MESH_SIZE},PRECISION=${PRECISION},BETA=${BETA},ODA_STALL_THRESHOLD=${ODA_STALL_THRESHOLD}"

{
  echo -e "stage\tjob_id\tdependency\toutput\tcommand"
} > "${MANIFEST}"

echo "[submit] out=${OUT}"
echo "[submit] cache=${CACHE}"
echo "[submit] partition=${PARTITION} cpus=${CPUS_PER_TASK} exclusive mem=0"
echo "[submit] resume_task_count=${RESUME_TASK_COUNT} array=0-${RESUME_ARRAY_MAX}%${ARRAY_THROTTLE}"

RESUME_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_resume \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_resume_%A_%a.out \
    -e logs/rlg_fig6_resume_%A_%a.err \
    --array="0-${RESUME_ARRAY_MAX}%${ARRAY_THROTTLE}" \
    --export="${EXPORT_HF}" \
    scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch \
    "${OUT}" "${CACHE}" "${MAX_ITER}"
)"
RESUME_JOB="${RESUME_JOB%%;*}"
echo -e "resume_hf\t${RESUME_JOB}\t\t${OUT}\t${RESUME_TASK_COUNT} existing init tasks, 1 exclusive node/init" >> "${MANIFEST}"

MERGE_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_merge \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t 04:00:00 \
    -o logs/rlg_fig6_merge_%j.out \
    -e logs/rlg_fig6_merge_%j.err \
    --dependency="afterok:${RESUME_JOB}" \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.merge_rlg_hbn_parallel_hf \
      --source-root "${OUT}" \
      --paper-target fig6
)"
MERGE_JOB="${MERGE_JOB%%;*}"
echo -e "merge\t${MERGE_JOB}\tafterok:${RESUME_JOB}\t${OUT}\tmerge_rlg_hbn_parallel_hf" >> "${MANIFEST}"

PLOT_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_plot \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_plot_%j.out \
    -e logs/rlg_fig6_plot_%j.err \
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
    -J rlg_fig6_verify \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task=4 --mem=8G \
    -t 00:30:00 \
    -o logs/rlg_fig6_verify_%j.out \
    -e logs/rlg_fig6_verify_%j.err \
    --dependency="afterok:${PLOT_JOB}" \
    scripts/run_rlg_hbn_fig6_verify.sbatch "${OUT}"
)"
VERIFY_JOB="${VERIFY_JOB%%;*}"
echo -e "verify\t${VERIFY_JOB}\tafterok:${PLOT_JOB}\t${OUT}\tverify_rlg_hbn_fig6_outputs" >> "${MANIFEST}"

echo "[submitted] resume_hf=${RESUME_JOB} (0-${RESUME_ARRAY_MAX}%${ARRAY_THROTTLE}, exclusive node/init)"
echo "[submitted] merge=${MERGE_JOB}"
echo "[submitted] plot=${PLOT_JOB}"
echo "[submitted] verify=${VERIFY_JOB}"
echo "[submitted] manifest=${MANIFEST}"
echo "[monitor] squeue -j ${RESUME_JOB},${MERGE_JOB},${PLOT_JOB},${VERIFY_JOB}"
