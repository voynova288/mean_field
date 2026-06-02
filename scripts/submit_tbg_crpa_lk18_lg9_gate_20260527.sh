#!/bin/bash
# Submit the medium-size TBG HF+cRPA gate requested on 2026-05-27:
# HF k mesh lk=18 with HF reciprocal/overlap shell lg=9.
#
# For exact HF-compatible matrix-diagonal lookup, overlap_lg=9 requires
# q_lg=11.  Production cRPA now uses periodic k wrapping with finite-G
# zero-fill form factors, so crpa_lg=9 is the intended matching shell.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
cd "${REPO_ROOT}"
mkdir -p logs

ACCOUNT="${ACCOUNT:-hmt03}"

CRPA_OUTPUT_ROOT="${CRPA_OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk18_lg9_q11_kperiodic_zerofill_20260530}"
HF_OUTPUT_ROOT="${HF_OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA/hf_crpa_lk18_lg9_q11_kperiodic_zerofill_20260530}"
CRPA_MANIFEST="${CRPA_MANIFEST:-${CRPA_OUTPUT_ROOT}/submission_jobs_crpa_lk18_lg9_q11_kperiodic_zerofill_20260530.tsv}"
HANDOFF="${HANDOFF:-${HF_OUTPUT_ROOT}/submission_lk18_lg9_q11_kperiodic_zerofill_20260530.tsv}"

CRPA_LK="${CRPA_LK:-18}"
CRPA_LG="${CRPA_LG:-9}"
CRPA_Q_LG="${CRPA_Q_LG:-11}"

HF_LK="${HF_LK:-18}"
HF_LG="${HF_LG:-9}"
HF_OVERLAP_LG="${HF_OVERLAP_LG:-9}"
MAX_ITER="${MAX_ITER:-300}"
INIT_SPEC="${INIT_SPEC:-vp:1}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-40}"
RUN_TAG="${RUN_TAG:-lk18_lg9_q11_iter${MAX_ITER}_vp_w1_97p4}"

CRPA_PARTITION="${CRPA_PARTITION:-regular128}"
BM_CPUS="${BM_CPUS:-64}"
BM_MEM="${BM_MEM:-0}"
BM_TIME="${BM_TIME:-3-00:00:00}"
CHUNK_CPUS="${CHUNK_CPUS:-64}"
CHUNK_MEM="${CHUNK_MEM:-0}"
CHUNK_TIME="${CHUNK_TIME:-3-00:00:00}"
MERGE_CPUS="${MERGE_CPUS:-64}"
MERGE_MEM="${MERGE_MEM:-0}"
MERGE_TIME="${MERGE_TIME:-03:00:00}"
CHUNK_COUNT="${CHUNK_COUNT:-144}"
CHUNKS_PER_NODE="${CHUNKS_PER_NODE:-2}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"

HF_PARTITION="${HF_PARTITION:-regular128}"
HF_CPUS="${HF_CPUS:-64}"
HF_MEM="${HF_MEM:-0}"
HF_TIME="${HF_TIME:-3-00:00:00}"

mkdir -p "${CRPA_OUTPUT_ROOT}" "${HF_OUTPUT_ROOT}"

echo "[lk18-lg9-gate] submitting legal cRPA artifact"
echo "[lk18-lg9-gate] crpa_output_root=${CRPA_OUTPUT_ROOT}"
echo "[lk18-lg9-gate] hf_output_root=${HF_OUTPUT_ROOT}"
echo "[lk18-lg9-gate] crpa lk=${CRPA_LK} lg=${CRPA_LG} q_lg=${CRPA_Q_LG}"
echo "[lk18-lg9-gate] hf lk=${HF_LK} lg=${HF_LG} overlap_lg=${HF_OVERLAP_LG} max_iter=${MAX_ITER}"

ACCOUNT="${ACCOUNT}" \
OUTPUT_ROOT="${CRPA_OUTPUT_ROOT}" \
LK="${CRPA_LK}" \
LG="${CRPA_LG}" \
Q_LG="${CRPA_Q_LG}" \
CRPA_CONVENTION_TAG="hf_kperiodic_zerofill" \
CRPA_HF_COMPATIBLE=1 \
CHUNK_COUNT="${CHUNK_COUNT}" \
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY}" \
CHUNKS_PER_NODE="${CHUNKS_PER_NODE}" \
PARTITION="${CRPA_PARTITION}" \
MERGE_PARTITION="${CRPA_PARTITION}" \
BM_CPUS="${BM_CPUS}" \
BM_MEM="${BM_MEM}" \
BM_TIME="${BM_TIME}" \
CHUNK_CPUS="${CHUNK_CPUS}" \
CHUNK_MEM="${CHUNK_MEM}" \
CHUNK_TIME="${CHUNK_TIME}" \
MERGE_CPUS="${MERGE_CPUS}" \
MERGE_MEM="${MERGE_MEM}" \
MERGE_TIME="${MERGE_TIME}" \
CRPA_MANIFEST="${CRPA_MANIFEST}" \
BM_EXCLUDE="" \
CHUNK_EXCLUDE="" \
bash scripts/submit_tbg_crpa_fig4_pipeline.sh submit-crpa

MERGE_JOB="$(awk -F '\t' '$1 == "merge" {print $2}' "${CRPA_MANIFEST}")"
CRPA_DIR="$(awk -F '\t' '$1 == "merge" {print $8}' "${CRPA_MANIFEST}")"
if [[ -z "${MERGE_JOB}" || -z "${CRPA_DIR}" ]]; then
  echo "Failed to read merge job or cRPA directory from ${CRPA_MANIFEST}" >&2
  exit 1
fi

echo "[lk18-lg9-gate] submitting dependent HF job afterok:${MERGE_JOB}"
HF_JOB="$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    -p "${HF_PARTITION}" \
    -N 1 \
    --ntasks=1 \
    --cpus-per-task="${HF_CPUS}" \
    --mem="${HF_MEM}" \
    --exclusive \
    -t "${HF_TIME}" \
    --dependency="afterok:${MERGE_JOB}" \
    -J "mf_crpa_lk18_lg9" \
    -o "logs/hf_crpa_lk18_lg9_%j.out" \
    -e "logs/hf_crpa_lk18_lg9_%j.err" \
    --export=ALL,LD_LIBRARY_PATH=,CRPA_DIR="${CRPA_DIR}",OUTPUT_ROOT="${HF_OUTPUT_ROOT}",LK="${HF_LK}",LG="${HF_LG}",OVERLAP_LG="${HF_OVERLAP_LG}",MAX_ITER="${MAX_ITER}",POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT}",INIT_SPEC="${INIT_SPEC}",RUN_TAG="${RUN_TAG}",MEAN_FIELD_CRPA_SPLIT_MODE=active_cnp_fock_reference_projector,MEAN_FIELD_CRPA_REMOTE_BARE_SCALE=1.0 \
    scripts/slurm_crpa_alias_hf_small_20260524.sbatch
)"

{
  printf "stage\tjob_id\tdependency\tpartition\tcpus\tmem\ttime\tpath_or_note\n"
  awk -F '\t' 'NR > 1 {print}' "${CRPA_MANIFEST}"
  printf "hf\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n" "${HF_JOB}" "${MERGE_JOB}" "${HF_PARTITION}" "${HF_CPUS}" "${HF_MEM}" "${HF_TIME}" "${HF_OUTPUT_ROOT}"
  printf "hf_config\t%s\tnone\t%s\t%s\t%s\t%s\tlk=%s lg=%s overlap_lg=%s max_iter=%s init=%s crpa_dir=%s run_tag=%s\n" \
    "${HF_JOB}" "${HF_PARTITION}" "-" "-" "-" "${HF_LK}" "${HF_LG}" "${HF_OVERLAP_LG}" "${MAX_ITER}" "${INIT_SPEC}" "${CRPA_DIR}" "${RUN_TAG}"
} > "${HANDOFF}"

echo "submitted_crpa_manifest=${CRPA_MANIFEST}"
echo "submitted_handoff=${HANDOFF}"
echo "submitted_merge_job=${MERGE_JOB}"
echo "submitted_hf_job=${HF_JOB}"
