#!/bin/bash
# Submit cached RLG/hBN Fig. 6 HF bands as a dependency chain:
#   prereq screening checkpoint -> cache warmup -> 32 HF init tasks -> merge -> plot -> verify
#
# The HF array uses one task per (xi, init_mode, seed):
#   xi in {0,1}, init_mode in {flavor,bm,perturbed,random}, seed in {1,2,3,4}.
# Submit from login002:
#   ssh login002 'cd /data/home/ziyuzhu/Mean_Field && bash scripts/submit_fig6_hf_bands_cached_init_parallel.sh'
#
# Useful overrides:
#   ARRAY_THROTTLE=8 PARTITION=regular6430 MAX_ITER=120 bash scripts/submit_fig6_hf_bands_cached_init_parallel.sh
#   OUT=/path/to/out CACHE=/path/to/cache bash scripts/submit_fig6_hf_bands_cached_init_parallel.sh

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs results/RnG_hBN

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPO_ROOT}/results/RnG_hBN/fig6_hf_bands_parallel_${STAMP}}"
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
PRECISION="${PRECISION:-1e-6}"
BETA="${BETA:-1.0}"
ODA_STALL_THRESHOLD="${ODA_STALL_THRESHOLD:-1e-3}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-48}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
DPI="${DPI:-180}"

mkdir -p "${OUT}" "${CACHE}"

EXPORT_COMMON="ALL,MEAN_FIELD_RLG_HBN_USE_NUMBA=1,MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1,MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS=16"
EXPORT_HF="${EXPORT_COMMON},CACHE_POLICY=${CACHE_POLICY},SCREENING_SOLVER=${SCREENING_SOLVER},SCREENING_U_MIN_MEV=${SCREENING_U_MIN_MEV},SCREENING_U_MAX_MEV=${SCREENING_U_MAX_MEV},SCREENING_U_GRID_POINTS=${SCREENING_U_GRID_POINTS},PRECISION=${PRECISION},BETA=${BETA},ODA_STALL_THRESHOLD=${ODA_STALL_THRESHOLD}"
MANIFEST="${OUT}/submission_jobs.tsv"

{
  echo -e "stage\tjob_id\tdependency\toutput\tcommand"
} > "${MANIFEST}"

echo "[submit] out=${OUT}"
echo "[submit] cache=${CACHE}"
echo "[submit] partition=${PARTITION} cpus=${CPUS_PER_TASK} array_throttle=${ARRAY_THROTTLE}"

PREREQ_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_prereq \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_prereq_%j.out \
    -e logs/rlg_fig6_prereq_%j.err \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.validate_rlg_hbn_fig6_prereqs \
      --cache-dir "${CACHE}" \
      --cache-policy "${CACHE_POLICY}" \
      --screening-solver "${SCREENING_SOLVER}" \
      --screening-u-min-mev "${SCREENING_U_MIN_MEV}" \
      --screening-u-max-mev "${SCREENING_U_MAX_MEV}" \
      --screening-u-grid-points "${SCREENING_U_GRID_POINTS}" \
      --tolerance-mev 3.0
)"
PREREQ_JOB="${PREREQ_JOB%%;*}"
echo -e "prereq\t${PREREQ_JOB}\t\t${OUT}\tvalidate_rlg_hbn_fig6_prereqs" >> "${MANIFEST}"

WARMUP_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_cache \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_cache_%j.out \
    -e logs/rlg_fig6_cache_%j.err \
    --dependency="afterok:${PREREQ_JOB}" \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.warm_rlg_hbn_fig6_cache \
      --output-dir "${OUT}" \
      --cache-dir "${CACHE}" \
      --cache-policy "${CACHE_POLICY}" \
      --screening-solver "${SCREENING_SOLVER}" \
      --screening-u-min-mev "${SCREENING_U_MIN_MEV}" \
      --screening-u-max-mev "${SCREENING_U_MAX_MEV}" \
      --screening-u-grid-points "${SCREENING_U_GRID_POINTS}"
)"
WARMUP_JOB="${WARMUP_JOB%%;*}"
echo -e "warmup\t${WARMUP_JOB}\tafterok:${PREREQ_JOB}\t${OUT}\twarm_rlg_hbn_fig6_cache" >> "${MANIFEST}"

HF_ARRAY_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_hfinit \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t "${WALLTIME}" \
    -o logs/rlg_fig6_hfinit_%A_%a.out \
    -e logs/rlg_fig6_hfinit_%A_%a.err \
    --array="0-31%${ARRAY_THROTTLE}" \
    --dependency="afterok:${WARMUP_JOB}" \
    --export="${EXPORT_HF}" \
    scripts/submit_rlg_hbn_paper_hf_array.sbatch \
    fig6 "${OUT}" "${MAX_ITER}" "${CACHE}"
)"
HF_ARRAY_JOB="${HF_ARRAY_JOB%%;*}"
echo -e "hf_array\t${HF_ARRAY_JOB}\tafterok:${WARMUP_JOB}\t${OUT}\t32 init tasks" >> "${MANIFEST}"

MERGE_JOB="$(
  sbatch --parsable \
    -J rlg_fig6_merge \
    -p "${PARTITION}" \
    -N 1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem=0 \
    -t 04:00:00 \
    -o logs/rlg_fig6_merge_%j.out \
    -e logs/rlg_fig6_merge_%j.err \
    --dependency="afterok:${HF_ARRAY_JOB}" \
    --export="${EXPORT_COMMON}" \
    scripts/submit_mean_field.sbatch \
    python -m mean_field.devtools.merge_rlg_hbn_parallel_hf \
      --source-root "${OUT}" \
      --paper-target fig6
)"
MERGE_JOB="${MERGE_JOB%%;*}"
echo -e "merge\t${MERGE_JOB}\tafterok:${HF_ARRAY_JOB}\t${OUT}\tmerge_rlg_hbn_parallel_hf" >> "${MANIFEST}"

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

echo "[submitted] prereq=${PREREQ_JOB}"
echo "[submitted] warmup=${WARMUP_JOB}"
echo "[submitted] hf_array=${HF_ARRAY_JOB} (0-31%${ARRAY_THROTTLE})"
echo "[submitted] merge=${MERGE_JOB}"
echo "[submitted] plot=${PLOT_JOB}"
echo "[submitted] verify=${VERIFY_JOB}"
echo "[submitted] manifest=${MANIFEST}"
echo "[monitor] squeue -j ${PREREQ_JOB},${WARMUP_JOB},${HF_ARRAY_JOB},${MERGE_JOB},${PLOT_JOB},${VERIFY_JOB}"
echo "[monitor] tail -f logs/rlg_fig6_hfinit_${HF_ARRAY_JOB}_0.out"
