#!/bin/bash
# Check or submit only the missing Zhang Appendix Fig. 4 cRPA chunks.
#
# Default mode is check-only:
#   bash scripts/submit_tbg_crpa_fig4_missing_chunks.sh
#
# Submit mode must be chosen explicitly, preferably from login002 after the
# upstream array has drained:
#   bash scripts/submit_tbg_crpa_fig4_missing_chunks.sh submit
#
# To also submit the replacement merge as an afterok dependency of the
# supplement array:
#   bash scripts/submit_tbg_crpa_fig4_missing_chunks.sh submit-with-merge

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA}"
LK="${LK:-24}"
LG="${LG:-9}"
Q_LG="${Q_LG:-11}"
CHUNK_COUNT="${CHUNK_COUNT:-144}"
CHUNK_ROOT="${CHUNK_ROOT:-${OUTPUT_ROOT}/crpa_lk${LK}_lg${LG}_q${Q_LG}_zhang_appendix_fig4_chunks}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_ROOT}/crpa_lk${LK}_lg${LG}_q${Q_LG}_zhang_appendix_fig4_merged}"
BM_SOLUTION="${BM_SOLUTION:-${CHUNK_ROOT}/bm_lk${LK}_lg${LG}_zhang_appendix_fig4_cache}"

PARTITION="${PARTITION:-regular6430}"
CHUNK_CPUS="${CHUNK_CPUS:-64}"
CHUNK_MEM="${CHUNK_MEM:-0}"
CHUNK_TIME="${CHUNK_TIME:-12:00:00}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-3}"
CHUNK_EXCLUDE="${CHUNK_EXCLUDE:-node023,node024}"
UPSTREAM_JOB_ID="${UPSTREAM_JOB_ID:-91471}"
ALLOW_ACTIVE="${ALLOW_ACTIVE:-0}"
MERGE_PARTITION="${MERGE_PARTITION:-regular6430}"
MERGE_CPUS="${MERGE_CPUS:-64}"
MERGE_MEM="${MERGE_MEM:-0}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"

MODE="${1:-check}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MISSING_FILE="${MISSING_FILE:-${CHUNK_ROOT}/missing_chunks_${RUN_ID}.txt}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/submission_jobs_crpa_lk${LK}_lg${LG}_q${Q_LG}_zhang_appendix_fig4_supplement_${RUN_ID}.tsv}"

required_files=(
  screened_coulomb.npz
  effective_epsilon.npz
  crpa_params.json
  validation_report.md
)

usage() {
  sed -n '1,12p' "$0" >&2
  echo "Modes: check, submit, submit-with-merge, submit-merge" >&2
}

repo_cd() {
  mkdir -p "${REPO_ROOT}/logs" "${CHUNK_ROOT}" "${OUTPUT_ROOT}"
  cd "${REPO_ROOT}"
}

write_missing_file() {
  local i file d complete
  : > "${MISSING_FILE}"
  for ((i = 0; i < CHUNK_COUNT; i++)); do
    d="${CHUNK_ROOT}/chunk_${i}"
    complete=1
    for file in "${required_files[@]}"; do
      if [[ ! -s "${d}/${file}" ]]; then
        complete=0
        break
      fi
    done
    if [[ "${complete}" -eq 0 ]]; then
      printf "%s\n" "${i}" >> "${MISSING_FILE}"
    fi
  done
}

missing_count() {
  wc -l < "${MISSING_FILE}" | tr -d ' '
}

array_spec_from_missing() {
  paste -sd, "${MISSING_FILE}"
}

print_check_summary() {
  local count
  count="$(missing_count)"
  echo "[fig4-crpa-supplement] chunk_root=${CHUNK_ROOT}"
  echo "[fig4-crpa-supplement] merged_dir=${MERGED_DIR}"
  echo "[fig4-crpa-supplement] bm_solution=${BM_SOLUTION}"
  echo "[fig4-crpa-supplement] complete_chunks=$((CHUNK_COUNT - count))/${CHUNK_COUNT}"
  echo "[fig4-crpa-supplement] missing_chunks=${count}"
  echo "[fig4-crpa-supplement] missing_file=${MISSING_FILE}"
  if [[ "${count}" -gt 0 ]]; then
    echo "[fig4-crpa-supplement] missing_indices=$(array_spec_from_missing)"
  fi
}

guard_upstream_drained() {
  if [[ -z "${UPSTREAM_JOB_ID}" || "${ALLOW_ACTIVE}" == "1" || "${ALLOW_ACTIVE}" == "true" ]]; then
    return 0
  fi
  if ! command -v squeue >/dev/null 2>&1; then
    echo "[fig4-crpa-supplement] squeue not found; run submit mode from login002 or set ALLOW_ACTIVE=1." >&2
    exit 2
  fi
  if squeue -h -j "${UPSTREAM_JOB_ID}" | grep -q .; then
    echo "[fig4-crpa-supplement] upstream job ${UPSTREAM_JOB_ID} is still active; wait for it to drain or set ALLOW_ACTIVE=1." >&2
    exit 2
  fi
}

submit_missing() {
  local count array_spec job_id
  count="$(missing_count)"
  if [[ "${count}" -eq 0 ]]; then
    echo "[fig4-crpa-supplement] no missing chunks; nothing to submit."
    return 0
  fi
  array_spec="$(array_spec_from_missing)%${ARRAY_CONCURRENCY}"

  local sbatch_args=(
    --parsable
    --array="${array_spec}"
    -p "${PARTITION}"
    -N 1
    --ntasks=1
    -c "${CHUNK_CPUS}"
    --mem="${CHUNK_MEM}"
    --exclusive
    --export=ALL,LD_LIBRARY_PATH=
    -t "${CHUNK_TIME}"
    -J "crpa_zfg4_sup_lk${LK}_q${Q_LG}"
    -o "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_supplement_%A_%a.out"
    -e "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_supplement_%A_%a.err"
  )
  if [[ -n "${CHUNK_EXCLUDE}" ]]; then
    sbatch_args+=(--exclude="${CHUNK_EXCLUDE}")
  fi

  job_id="$(
    sbatch "${sbatch_args[@]}" \
      scripts/submit_tbg_crpa_chunk_array.sbatch \
      "${BM_SOLUTION}" \
      "${CHUNK_ROOT}" \
      "${Q_LG}" \
      "${CHUNK_COUNT}"
  )"

  {
    printf "stage\tjob_id\tpartition\tcpus\tmem\ttime\tarray_spec\texclude\tmissing_file\tpath\n"
    printf "chunks_supplement\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_id}" "${PARTITION}" "${CHUNK_CPUS}" "${CHUNK_MEM}" "${CHUNK_TIME}" \
      "${array_spec}" "${CHUNK_EXCLUDE:-none}" "${MISSING_FILE}" "${CHUNK_ROOT}"
  } > "${MANIFEST}"

  echo "submitted supplement_job=${job_id}"
  echo "submission_manifest=${MANIFEST}"
  SUPPLEMENT_JOB_ID="${job_id}"
}

submit_merge() {
  local count dependency_arg job_id i
  count="$(missing_count)"
  if [[ "${count}" -ne 0 && -z "${MERGE_DEPENDENCY:-}" ]]; then
    echo "[fig4-crpa-supplement] refusing merge: ${count} chunks are still missing and MERGE_DEPENDENCY is empty." >&2
    exit 2
  fi

  local chunk_args=()
  for ((i = 0; i < CHUNK_COUNT; i++)); do
    chunk_args+=(--chunk "${CHUNK_ROOT}/chunk_${i}")
  done

  local sbatch_args=(
    --parsable
    -p "${MERGE_PARTITION}"
    -N 1
    --ntasks=1
    -c "${MERGE_CPUS}"
    --mem="${MERGE_MEM}"
    --exclusive
    --export=ALL,LD_LIBRARY_PATH=
    -t "${MERGE_TIME}"
    -J "crpa_zfg4_lk${LK}_q${Q_LG}_merge"
    -o "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_merge_supplement_%j.out"
    -e "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_merge_supplement_%j.err"
  )
  if [[ -n "${MERGE_DEPENDENCY:-}" ]]; then
    sbatch_args+=(--dependency="${MERGE_DEPENDENCY}")
    dependency_arg="${MERGE_DEPENDENCY}"
  else
    dependency_arg="none"
  fi

  job_id="$(
    sbatch "${sbatch_args[@]}" \
      scripts/submit_mean_field.sbatch \
      python scripts/mean_field_tools.py merge_tbg_crpa_chunks \
        --output-dir "${MERGED_DIR}" \
        "${chunk_args[@]}"
  )"

  {
    printf "stage\tjob_id\tdependency\tpartition\tcpus\tmem\ttime\tpath\n"
    printf "merge_supplement\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_id}" "${dependency_arg}" "${MERGE_PARTITION}" "${MERGE_CPUS}" "${MERGE_MEM}" "${MERGE_TIME}" "${MERGED_DIR}"
  } >> "${MANIFEST}"

  echo "submitted merge_job=${job_id}"
  echo "submission_manifest=${MANIFEST}"
}

repo_cd
write_missing_file

case "${MODE}" in
  check)
    print_check_summary
    ;;
  submit)
    print_check_summary
    guard_upstream_drained
    submit_missing
    ;;
  submit-with-merge)
    print_check_summary
    guard_upstream_drained
    submit_missing
    if [[ -n "${SUPPLEMENT_JOB_ID:-}" ]]; then
      MERGE_DEPENDENCY="afterok:${SUPPLEMENT_JOB_ID}" submit_merge
    else
      submit_merge
    fi
    ;;
  submit-merge)
    print_check_summary
    guard_upstream_drained
    submit_merge
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac
