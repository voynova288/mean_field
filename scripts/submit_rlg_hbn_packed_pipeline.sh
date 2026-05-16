#!/bin/bash
# Submit RnG/hBN paper HF jobs with packed CPU-saturating array tasks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs results/RnG_hBN

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
MAX_ITER="${MAX_ITER:-80}"
TASKS_PER_NODE_FIG5="${TASKS_PER_NODE_FIG5:-6}"
THREADS_PER_TASK_FIG5="${THREADS_PER_TASK_FIG5:-16}"
TASKS_PER_NODE_FIG6="${TASKS_PER_NODE_FIG6:-3}"
THREADS_PER_TASK_FIG6="${THREADS_PER_TASK_FIG6:-24}"
ARRAY_LIMIT_FIG5="${ARRAY_LIMIT_FIG5:-2}"
ARRAY_LIMIT_FIG6="${ARRAY_LIMIT_FIG6:-2}"

FIG5_TASKS=12
FIG6_TASKS=6
FIG5_LAST_GROUP=$(((FIG5_TASKS + TASKS_PER_NODE_FIG5 - 1) / TASKS_PER_NODE_FIG5 - 1))
FIG6_LAST_GROUP=$(((FIG6_TASKS + TASKS_PER_NODE_FIG6 - 1) / TASKS_PER_NODE_FIG6 - 1))

FIG5_ROOT="${REPO_ROOT}/results/RnG_hBN/rlg_hbn_fig5_packed_${STAMP}"
FIG6_ROOT="${REPO_ROOT}/results/RnG_hBN/rlg_hbn_fig6_packed_${STAMP}"
mkdir -p "${FIG5_ROOT}" "${FIG6_ROOT}"

FIG5_ARRAY=$(sbatch --parsable \
  -J rlg_fig5_hf_packed \
  -p regular256 \
  -c 64 \
  --mem=180G \
  -t 2-00:00:00 \
  --export=ALL,ALLOW_OVERSUBSCRIBE=1 \
  --array=0-"${FIG5_LAST_GROUP}"%"${ARRAY_LIMIT_FIG5}" \
  -o "logs/rlg_fig5_hf_packed_%A_%a.out" \
  -e "logs/rlg_fig5_hf_packed_%A_%a.err" \
  scripts/submit_rlg_hbn_paper_hf_packed_array.sbatch fig5 "${FIG5_ROOT}" "${MAX_ITER}" "${TASKS_PER_NODE_FIG5}" "${THREADS_PER_TASK_FIG5}")

FIG6_ARRAY=$(sbatch --parsable \
  -J rlg_fig6_hf_packed \
  -p regular256 \
  -c 64 \
  --mem=240G \
  -t 2-00:00:00 \
  --export=ALL,ALLOW_OVERSUBSCRIBE=1 \
  --array=0-"${FIG6_LAST_GROUP}"%"${ARRAY_LIMIT_FIG6}" \
  -o "logs/rlg_fig6_hf_packed_%A_%a.out" \
  -e "logs/rlg_fig6_hf_packed_%A_%a.err" \
  scripts/submit_rlg_hbn_paper_hf_packed_array.sbatch fig6 "${FIG6_ROOT}" "${MAX_ITER}" "${TASKS_PER_NODE_FIG6}" "${THREADS_PER_TASK_FIG6}")

FIG5_MERGE=$(sbatch --parsable \
  -J rlg_fig5_merge \
  -p test \
  -c 4 \
  --mem=16G \
  -t 00:30:00 \
  --dependency=afterok:"${FIG5_ARRAY}" \
  -o "logs/rlg_fig5_merge_%j.out" \
  -e "logs/rlg_fig5_merge_%j.err" \
  scripts/submit_mean_field.sbatch \
  python scripts/mean_field_tools.py merge_rlg_hbn_parallel_hf \
  --source-root "${FIG5_ROOT}" \
  --paper-target fig5)

FIG6_MERGE=$(sbatch --parsable \
  -J rlg_fig6_merge \
  -p test \
  -c 4 \
  --mem=16G \
  -t 00:30:00 \
  --dependency=afterok:"${FIG6_ARRAY}" \
  -o "logs/rlg_fig6_merge_%j.out" \
  -e "logs/rlg_fig6_merge_%j.err" \
  scripts/submit_mean_field.sbatch \
  python scripts/mean_field_tools.py merge_rlg_hbn_parallel_hf \
  --source-root "${FIG6_ROOT}" \
  --paper-target fig6)

FIG5_BANDS=$(sbatch --parsable \
  -J rlg_fig5_bands_packed \
  -p regular256 \
  -c 64 \
  --mem=80G \
  -t 1-00:00:00 \
  --dependency=afterok:"${FIG5_MERGE}" \
  -o "logs/rlg_fig5_bands_packed_%j.out" \
  -e "logs/rlg_fig5_bands_packed_%j.err" \
  scripts/submit_mean_field.sbatch \
  python -m mean_field.devtools.plot_rlg_hbn_paper_hf_bands \
  --source-dir "${FIG5_ROOT}" \
  --paper-target fig5)

FIG6_BANDS=$(sbatch --parsable \
  -J rlg_fig6_bands_packed \
  -p regular128 \
  -c 64 \
  --mem=125G \
  -t 1-00:00:00 \
  --dependency=afterok:"${FIG6_MERGE}" \
  -o "logs/rlg_fig6_bands_packed_%j.out" \
  -e "logs/rlg_fig6_bands_packed_%j.err" \
  scripts/submit_mean_field.sbatch \
  python -m mean_field.devtools.plot_rlg_hbn_paper_hf_bands \
  --source-dir "${FIG6_ROOT}" \
  --paper-target fig6)

MANIFEST="${REPO_ROOT}/results/RnG_hBN/rlg_hbn_packed_submission_${STAMP}.tsv"
{
  printf "target\tstage\tjob_id\troot\ttasks_per_node\tthreads_per_task\n"
  printf "fig5\thf_array\t%s\t%s\t%s\t%s\n" "${FIG5_ARRAY}" "${FIG5_ROOT}" "${TASKS_PER_NODE_FIG5}" "${THREADS_PER_TASK_FIG5}"
  printf "fig5\tmerge\t%s\t%s\t%s\t%s\n" "${FIG5_MERGE}" "${FIG5_ROOT}" "${TASKS_PER_NODE_FIG5}" "${THREADS_PER_TASK_FIG5}"
  printf "fig5\tbands\t%s\t%s\t%s\t%s\n" "${FIG5_BANDS}" "${FIG5_ROOT}" "${TASKS_PER_NODE_FIG5}" "${THREADS_PER_TASK_FIG5}"
  printf "fig6\thf_array\t%s\t%s\t%s\t%s\n" "${FIG6_ARRAY}" "${FIG6_ROOT}" "${TASKS_PER_NODE_FIG6}" "${THREADS_PER_TASK_FIG6}"
  printf "fig6\tmerge\t%s\t%s\t%s\t%s\n" "${FIG6_MERGE}" "${FIG6_ROOT}" "${TASKS_PER_NODE_FIG6}" "${THREADS_PER_TASK_FIG6}"
  printf "fig6\tbands\t%s\t%s\t%s\t%s\n" "${FIG6_BANDS}" "${FIG6_ROOT}" "${TASKS_PER_NODE_FIG6}" "${THREADS_PER_TASK_FIG6}"
} > "${MANIFEST}"

echo "[submitted] manifest=${MANIFEST}"
cat "${MANIFEST}"
