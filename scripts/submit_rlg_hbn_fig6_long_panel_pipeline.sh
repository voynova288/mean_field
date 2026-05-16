#!/bin/bash
# Submit a long-walltime Fig. 6 RnG/hBN HF chain with one panel per array task.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs results/RnG_hBN

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
MAX_ITER="${MAX_ITER:-160}"
THREADS_PER_TASK="${THREADS_PER_TASK:-64}"
ARRAY_LIMIT_FIG6="${ARRAY_LIMIT_FIG6:-1}"

FIG6_ROOT="${REPO_ROOT}/results/RnG_hBN/rlg_hbn_fig6_long_panel_${STAMP}"
mkdir -p "${FIG6_ROOT}"

FIG6_ARRAY=$(sbatch --parsable \
  -J rlg_fig6_hf_panel \
  -p long \
  -c 64 \
  --exclusive \
  --mem=0 \
  -t 7-00:00:00 \
  --array=0-1%"${ARRAY_LIMIT_FIG6}" \
  -o "logs/rlg_fig6_hf_panel_%A_%a.out" \
  -e "logs/rlg_fig6_hf_panel_%A_%a.err" \
  scripts/submit_rlg_hbn_paper_hf_panel_array.sbatch \
  fig6 "${FIG6_ROOT}" "${MAX_ITER}" "${THREADS_PER_TASK}")

FIG6_MERGE=$(sbatch --parsable \
  -J rlg_fig6_merge_panel \
  -p test \
  -c 4 \
  --mem=16G \
  -t 00:30:00 \
  --dependency=afterok:"${FIG6_ARRAY}" \
  -o "logs/rlg_fig6_merge_panel_%j.out" \
  -e "logs/rlg_fig6_merge_panel_%j.err" \
  scripts/submit_mean_field.sbatch \
  python scripts/mean_field_tools.py merge_rlg_hbn_parallel_hf \
  --source-root "${FIG6_ROOT}" \
  --paper-target fig6)

FIG6_BANDS=$(sbatch --parsable \
  -J rlg_fig6_bands_panel \
  -p long \
  -c 64 \
  --exclusive \
  --mem=0 \
  -t 7-00:00:00 \
  --dependency=afterok:"${FIG6_MERGE}" \
  -o "logs/rlg_fig6_bands_panel_%j.out" \
  -e "logs/rlg_fig6_bands_panel_%j.err" \
  scripts/submit_mean_field.sbatch \
  python -m mean_field.devtools.plot_rlg_hbn_paper_hf_bands \
  --source-dir "${FIG6_ROOT}" \
  --paper-target fig6)

MANIFEST="${REPO_ROOT}/results/RnG_hBN/rlg_hbn_fig6_long_panel_submission_${STAMP}.tsv"
{
  printf "target\tstage\tjob_id\troot\tarray_limit\tthreads_per_task\tmax_iter\n"
  printf "fig6\thf_panel_array\t%s\t%s\t%s\t%s\t%s\n" "${FIG6_ARRAY}" "${FIG6_ROOT}" "${ARRAY_LIMIT_FIG6}" "${THREADS_PER_TASK}" "${MAX_ITER}"
  printf "fig6\tmerge\t%s\t%s\t%s\t%s\t%s\n" "${FIG6_MERGE}" "${FIG6_ROOT}" "${ARRAY_LIMIT_FIG6}" "${THREADS_PER_TASK}" "${MAX_ITER}"
  printf "fig6\tbands\t%s\t%s\t%s\t%s\t%s\n" "${FIG6_BANDS}" "${FIG6_ROOT}" "${ARRAY_LIMIT_FIG6}" "${THREADS_PER_TASK}" "${MAX_ITER}"
} > "${MANIFEST}"

echo "[submitted] manifest=${MANIFEST}"
cat "${MANIFEST}"
