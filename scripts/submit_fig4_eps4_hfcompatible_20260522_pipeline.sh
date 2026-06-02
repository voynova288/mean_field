#!/bin/bash
# Submit the corrected Fig. 4 epsilon=4 HF-compatible cRPA -> validation -> nu=-3 band chain.

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs results/TBG_HF_cRPA/hf_crpa_fig4_eps4_runs

ACCOUNT="${ACCOUNT:-hmt03}"
TAG="${CRPA_CONVENTION_TAG:-hfcompatible_fig4_20260522_epsbn4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA}"
CHUNK_ROOT="${OUTPUT_ROOT}/crpa_lk24_lg9_q11_${TAG}_chunks"
MERGED_DIR="${OUTPUT_ROOT}/crpa_lk24_lg9_q11_${TAG}_merged"
BM_SOLUTION="${CHUNK_ROOT}/bm_lk24_lg9_${TAG}_cache"
CRPA_MANIFEST="${OUTPUT_ROOT}/submission_jobs_crpa_lk24_lg9_q11_${TAG}.tsv"
VALIDATION_DIR="${OUTPUT_ROOT}/crpa_validation_20260522_fig4_hfcompatible_epsbn4"
HF_OUTPUT_ROOT="${OUTPUT_ROOT}/hf_crpa_fig4_eps4_runs"
HF_RUN_TAG="fig4_eps4_crpa_hfcompatible_20260522_lk24_q11_gamma_m_k_gamma_kprime"
HANDOFF="${HF_OUTPUT_ROOT}/fig4_eps4_hfcompatible_20260522_handoff.md"
SUBMISSION_MANIFEST="${HF_OUTPUT_ROOT}/fig4_eps4_hfcompatible_20260522_submission.tsv"

echo "[submit] tag=${TAG}"
echo "[submit] account=${ACCOUNT}"
echo "[submit] chunk_root=${CHUNK_ROOT}"
echo "[submit] merged_dir=${MERGED_DIR}"
echo "[submit] validation_dir=${VALIDATION_DIR}"
echo "[submit] hf_output_root=${HF_OUTPUT_ROOT}"

submit_output="$(
  ACCOUNT="${ACCOUNT}" \
  CRPA_HF_COMPATIBLE=1 \
  CRPA_CONVENTION_TAG="${TAG}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CHUNK_ROOT="${CHUNK_ROOT}" \
  MERGED_DIR="${MERGED_DIR}" \
  BM_SOLUTION="${BM_SOLUTION}" \
  CRPA_MANIFEST="${CRPA_MANIFEST}" \
  PARTITION="${CRPA_PARTITION:-regular}" \
  MERGE_PARTITION="${CRPA_MERGE_PARTITION:-regular}" \
  BM_CPUS="${CRPA_CPUS:-56}" \
  CHUNK_CPUS="${CRPA_CPUS:-56}" \
  MERGE_CPUS="${CRPA_CPUS:-56}" \
  BM_MEM="${CRPA_MEM:-0}" \
  CHUNK_MEM="${CRPA_MEM:-0}" \
  MERGE_MEM="${CRPA_MEM:-0}" \
  BM_TIME="${CRPA_BM_TIME:-06:00:00}" \
  CHUNK_TIME="${CRPA_CHUNK_TIME:-12:00:00}" \
  MERGE_TIME="${CRPA_MERGE_TIME:-01:00:00}" \
  CHUNKS_PER_NODE="${CHUNKS_PER_NODE:-2}" \
  ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}" \
  BM_EXCLUDE="${BM_EXCLUDE:-}" \
  CHUNK_EXCLUDE="${CHUNK_EXCLUDE:-}" \
  bash scripts/submit_tbg_crpa_fig4_pipeline.sh submit-crpa
)"
printf "%s\n" "${submit_output}"

bm_job="$(awk -F= '/submitted bm_job=/{print $2}' <<<"${submit_output}" | tail -1)"
chunk_job="$(awk -F= '/submitted array_job=/{print $2}' <<<"${submit_output}" | tail -1)"
merge_job="$(awk -F= '/submitted merge_job=/{print $2}' <<<"${submit_output}" | tail -1)"
if [[ -z "${merge_job}" ]]; then
  echo "Could not parse merge job from submit output." >&2
  exit 2
fi

validation_job="$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    --dependency="afterok:${merge_job}" \
    -p test \
    -N 1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    -t 00:30:00 \
    -J "crpa_fig4_hfcompat_val_20260522" \
    -o "logs/crpa_fig4_hfcompat_val_20260522_%j.out" \
    -e "logs/crpa_fig4_hfcompat_val_20260522_%j.err" \
    scripts/submit_mean_field.sbatch \
    python scripts/mean_field_tools.py validate_tbg_crpa_artifact \
      --crpa-dir "${MERGED_DIR}" \
      --output-dir "${VALIDATION_DIR}" \
      --require-hf-compatible \
      --overlap-lg 9
)"

band_job="$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    --dependency="afterok:${validation_job}" \
    --export=ALL,CRPA_DIR="${MERGED_DIR}",RUN_TAG="${HF_RUN_TAG}" \
    scripts/run_hf_band_nu_m3_crpa_dielectric_fixed_20260520.sbatch
)"

{
  printf "stage\tjob_id\tdependency\tpath\n"
  printf "bm\t%s\tnone\t%s\n" "${bm_job}" "${BM_SOLUTION}"
  printf "chunks\t%s\tafterok:%s\t%s\n" "${chunk_job}" "${bm_job}" "${CHUNK_ROOT}"
  printf "merge\t%s\tafterok:%s\t%s\n" "${merge_job}" "${chunk_job}" "${MERGED_DIR}"
  printf "validation\t%s\tafterok:%s\t%s\n" "${validation_job}" "${merge_job}" "${VALIDATION_DIR}"
  printf "band\t%s\tafterok:%s\t%s\n" "${band_job}" "${validation_job}" "${HF_OUTPUT_ROOT}"
} > "${SUBMISSION_MANIFEST}"

cat > "${HANDOFF}" <<EOF
# Fig. 4 epsilon=4 HF-compatible cRPA nu=-3 rerun handoff, 2026-05-22

## Purpose

Run the corrected workflow required by \`plan/crpa工作文档.md\`: do not pass the
Zhang paper-reference \`zhang_zero_fill/periodic_g_grid=false\` artifact to HF.
Generate a fresh HF-compatible cRPA artifact and run the \`nu=-3\` band only
after the HF-compatible validation gate succeeds.

## Convention

- cRPA tag: \`${TAG}\`
- cRPA artifact: \`${MERGED_DIR}\`
- required convention: \`periodic_g_grid=true\`, \`form_factor_mode=k_periodic_zero_fill\`, \`occupation_mode=cnp_index\`
- q table: \`lk=24\`, \`lg=9\`, \`q_lg=11\`
- HF overlap: \`overlap_lg=9\`
- HF Fock lookup: \`matrix_diagonal\`
- paper-reference dielectric bundle kept only as validation context:
  \`${OUTPUT_ROOT}/crpa_dielectric_fixed_20260519\`

## Jobs

- BM cache: \`${bm_job}\`
- cRPA chunk array: \`${chunk_job}\`
- merge: \`${merge_job}\`
- HF-compatible validation: \`${validation_job}\`
- nu=-3 band: \`${band_job}\`

## Paths

- cRPA manifest: \`${CRPA_MANIFEST}\`
- submission manifest: \`${SUBMISSION_MANIFEST}\`
- validation output: \`${VALIDATION_DIR}\`
- HF output root: \`${HF_OUTPUT_ROOT}\`
- expected band run directory:
  \`${HF_OUTPUT_ROOT}/theta_105_nu_-3000_${HF_RUN_TAG}\`

## Acceptance

- validation job exits \`0:0\`
- validation report status is \`pass\`
- band job exits \`0:0\`
- band summary has \`converged=true\`
- \`q_lookup_failures=0\`
- \`q_lookup_fallbacks=0\`
- path band TSV/PNG files exist under \`path_bands/\`
EOF

echo "submitted bm_job=${bm_job}"
echo "submitted chunk_job=${chunk_job}"
echo "submitted merge_job=${merge_job}"
echo "submitted validation_job=${validation_job}"
echo "submitted band_job=${band_job}"
echo "submission_manifest=${SUBMISSION_MANIFEST}"
echo "handoff=${HANDOFF}"
