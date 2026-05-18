#!/bin/bash
# Submit or run the Zhang Appendix Fig. 4 cRPA-HF production workflow.
#
# Operator modes:
#   submit-all          submit cRPA lk24 table, then dependent HF+cRPA array
#   submit-crpa         submit only cRPA lk24 BM/chunk/merge chain
#   submit-hf           submit only HF+cRPA array; set DEPENDENCY=afterok:<merge_job>
#
# Internal Slurm modes:
#   run-hf-array-task   run one array task selected from FILLINGS_CSV
#   run-hf-filling      run selected cRPA-HF rows for one filling

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA}"
HF_OUTPUT_ROOT="${HF_OUTPUT_ROOT:-${OUTPUT_ROOT}/hf_crpa_production_runs}"
SELECTED_MANIFEST="${SELECTED_MANIFEST:-${REPO_ROOT}/results/TBG_HF/custom_b0_hf_targeted_runs/selected_eps10_appendix_fig4_all7_20260509.tsv}"

LK="${LK:-24}"
LG="${LG:-9}"
Q_LG="${Q_LG:-11}"
CHUNK_COUNT="${CHUNK_COUNT:-144}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
CRPA_HF_COMPATIBLE="${CRPA_HF_COMPATIBLE:-1}"
if [[ "${CRPA_HF_COMPATIBLE}" == "1" || "${CRPA_HF_COMPATIBLE}" == "true" ]]; then
  CRPA_CONVENTION_TAG="${CRPA_CONVENTION_TAG:-hf_compatible}"
  CRPA_EXTRA_ARGS=(--hf-compatible)
else
  CRPA_CONVENTION_TAG="${CRPA_CONVENTION_TAG:-zhang_appendix_fig4}"
  CRPA_EXTRA_ARGS=()
fi
CHUNK_ROOT="${CHUNK_ROOT:-${OUTPUT_ROOT}/crpa_lk${LK}_lg${LG}_q${Q_LG}_${CRPA_CONVENTION_TAG}_chunks}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_ROOT}/crpa_lk${LK}_lg${LG}_q${Q_LG}_${CRPA_CONVENTION_TAG}_merged}"
BM_SOLUTION="${BM_SOLUTION:-${CHUNK_ROOT}/bm_lk${LK}_lg${LG}_${CRPA_CONVENTION_TAG}_cache}"
CRPA_MANIFEST="${CRPA_MANIFEST:-${OUTPUT_ROOT}/submission_jobs_crpa_lk${LK}_lg${LG}_q${Q_LG}_${CRPA_CONVENTION_TAG}_20260509.tsv}"

CRPA_DIR="${CRPA_DIR:-${MERGED_DIR}}"
CRPA_RUN_TAG_SUFFIX="${CRPA_RUN_TAG_SUFFIX:-crpa_${CRPA_CONVENTION_TAG}_lk${LK}_q${Q_LG}_20260509_gamma_m_k_gamma_kprime}"
MAX_ITER="${MAX_ITER:-3000}"
TARGET_LK="${TARGET_LK:-24}"
OVERLAP_LG="${OVERLAP_LG:-9}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-120}"
PATH_KIND="${PATH_KIND:-gamma-m-k-gamma-kprime}"
INITIAL_STATE_RESAMPLE="${INITIAL_STATE_RESAMPLE:-bilinear}"
FOCK_INTERPOLATION="${FOCK_INTERPOLATION:-matrix_diagonal}"
HF_MANIFEST="${HF_MANIFEST:-${HF_OUTPUT_ROOT}/submission_jobs_crpa_lk24_selected_${CRPA_RUN_TAG_SUFFIX}.tsv}"

PARTITION="${PARTITION:-regular6430}"
BM_CPUS="${BM_CPUS:-64}"
BM_MEM="${BM_MEM:-0}"
BM_TIME="${BM_TIME:-06:00:00}"
CHUNK_CPUS="${CHUNK_CPUS:-64}"
CHUNK_MEM="${CHUNK_MEM:-0}"
CHUNK_TIME="${CHUNK_TIME:-12:00:00}"
CHUNKS_PER_NODE="${CHUNKS_PER_NODE:-1}"
MERGE_PARTITION="${MERGE_PARTITION:-regular6430}"
MERGE_CPUS="${MERGE_CPUS:-64}"
MERGE_MEM="${MERGE_MEM:-0}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"
HF_CPUS="${HF_CPUS:-64}"
HF_MEM="${HF_MEM:-0}"
HF_TIME="${HF_TIME:-3-00:00:00}"
HF_EXCLUSIVE="${HF_EXCLUSIVE:-1}"
HF_ARRAY_CONCURRENCY="${HF_ARRAY_CONCURRENCY:-2}"
FILLINGS_CSV_OVERRIDE="${FILLINGS_CSV_OVERRIDE:-}"
DEPENDENCY="${DEPENDENCY:-}"
CRPA_DEPENDENCY="${CRPA_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-0}"
BM_NODELIST="${BM_NODELIST:-}"
BM_EXCLUDE="${BM_EXCLUDE:-node023,node024}"
CHUNK_EXCLUDE="${CHUNK_EXCLUDE:-node023,node024}"

usage() {
  sed -n '1,16p' "$0" >&2
}

repo_cd() {
  mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}" "${HF_OUTPUT_ROOT}" "${CHUNK_ROOT}"
  cd "${REPO_ROOT}"
}

submit_crpa() {
  repo_cd
  echo "[fig4-crpa] lk=${LK} lg=${LG} q_lg=${Q_LG}"
  echo "[fig4-crpa] chunk_count=${CHUNK_COUNT} concurrency=${ARRAY_CONCURRENCY}"
  echo "[fig4-crpa] hf_compatible=${CRPA_HF_COMPATIBLE}"
  echo "[fig4-crpa] convention_tag=${CRPA_CONVENTION_TAG}"
  echo "[fig4-crpa] bm_solution=${BM_SOLUTION}"
  echo "[fig4-crpa] chunk_root=${CHUNK_ROOT}"
  echo "[fig4-crpa] merged_dir=${MERGED_DIR}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "dry_run=true"
    SUBMITTED_MERGE_JOB=""
    return 0
  fi

  local bm_job array_job merge_job
  local bm_place_args=()
  if [[ -n "${CRPA_DEPENDENCY}" ]]; then
    bm_place_args+=(--dependency="${CRPA_DEPENDENCY}")
  fi
  if [[ -n "${BM_NODELIST}" ]]; then
    bm_place_args+=(--nodelist="${BM_NODELIST}")
  fi
  if [[ -n "${BM_EXCLUDE}" ]]; then
    bm_place_args+=(--exclude="${BM_EXCLUDE}")
  fi
  bm_job="$(
    sbatch --parsable \
      -p "${PARTITION}" \
      -N 1 \
      --ntasks=1 \
      -c "${BM_CPUS}" \
      --mem="${BM_MEM}" \
      --exclusive \
      --export=ALL,LD_LIBRARY_PATH= \
      "${bm_place_args[@]}" \
      -t "${BM_TIME}" \
      -J "crpa_zfg4_bm_lk${LK}_lg${LG}" \
      -o "logs/crpa_zhang_fig4_bm_lk${LK}_lg${LG}_%j.out" \
      -e "logs/crpa_zhang_fig4_bm_lk${LK}_lg${LG}_%j.err" \
      scripts/submit_mean_field.sbatch \
      python scripts/mean_field_tools.py prepare_tbg_crpa_bm \
        --theta-deg 1.05 \
        --vf 2135.4 \
        --w0 79.7 \
        --w1 97.4 \
        --lk "${LK}" \
        --lg "${LG}" \
        "${CRPA_EXTRA_ARGS[@]}" \
        --output-path "${BM_SOLUTION}"
  )"

  local chunk_array_count chunk_array_last chunk_script chunk_extra_args=()
  if [[ "${CHUNKS_PER_NODE}" -gt 1 ]]; then
    chunk_array_count=$(((CHUNK_COUNT + CHUNKS_PER_NODE - 1) / CHUNKS_PER_NODE))
    chunk_script="scripts/submit_tbg_crpa_packed_chunk_array.sbatch"
    chunk_extra_args=("${CHUNKS_PER_NODE}")
  else
    chunk_array_count="${CHUNK_COUNT}"
    chunk_script="scripts/submit_tbg_crpa_chunk_array.sbatch"
  fi
  chunk_array_last=$((chunk_array_count - 1))

  array_job="$(
    sbatch --parsable \
      --dependency="afterok:${bm_job}" \
      --array="0-${chunk_array_last}%${ARRAY_CONCURRENCY}" \
      -p "${PARTITION}" \
      -N 1 \
      --ntasks=1 \
      -c "${CHUNK_CPUS}" \
      --mem="${CHUNK_MEM}" \
      --exclusive \
      --export=ALL,LD_LIBRARY_PATH= \
      --exclude="${CHUNK_EXCLUDE}" \
      -t "${CHUNK_TIME}" \
      -J "crpa_zfg4_lk${LK}_q${Q_LG}" \
      -o "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_chunk_%A_%a.out" \
      -e "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_chunk_%A_%a.err" \
      "${chunk_script}" \
        "${BM_SOLUTION}" \
        "${CHUNK_ROOT}" \
        "${Q_LG}" \
        "${CHUNK_COUNT}" \
        "${chunk_extra_args[@]}" \
        "${CRPA_EXTRA_ARGS[@]}"
  )"

  local chunk_args=()
  for ((i = 0; i < CHUNK_COUNT; i++)); do
    chunk_args+=(--chunk "${CHUNK_ROOT}/chunk_${i}")
  done

  merge_job="$(
    sbatch --parsable \
      --dependency="afterok:${array_job}" \
      -p "${MERGE_PARTITION}" \
      -N 1 \
      --ntasks=1 \
      -c "${MERGE_CPUS}" \
      --mem="${MERGE_MEM}" \
      --exclusive \
      --export=ALL,LD_LIBRARY_PATH= \
      -t "${MERGE_TIME}" \
      -J "crpa_zfg4_lk${LK}_q${Q_LG}_merge" \
      -o "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_merge_%j.out" \
      -e "logs/crpa_zhang_fig4_lk${LK}_lg${LG}_q${Q_LG}_merge_%j.err" \
      scripts/submit_mean_field.sbatch \
      python scripts/mean_field_tools.py merge_tbg_crpa_chunks \
        --output-dir "${MERGED_DIR}" \
        "${chunk_args[@]}"
  )"

  {
    printf "stage\tjob_id\tdependency\tpartition\tcpus\tmem\ttime\tpath\n"
    printf "bm\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${bm_job}" "${CRPA_DEPENDENCY:-none}" "${PARTITION}" "${BM_CPUS}" "${BM_MEM}" "${BM_TIME}" "${BM_SOLUTION}"
    printf "chunks\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n" "${array_job}" "${bm_job}" "${PARTITION}" "${CHUNK_CPUS}" "${CHUNK_MEM}" "${CHUNK_TIME}" "${CHUNK_ROOT} chunks_per_node=${CHUNKS_PER_NODE}"
    printf "merge\t%s\tafterok:%s\t%s\t%s\t%s\t%s\t%s\n" "${merge_job}" "${array_job}" "${MERGE_PARTITION}" "${MERGE_CPUS}" "${MERGE_MEM}" "${MERGE_TIME}" "${MERGED_DIR}"
    printf "convention\t%s\tnone\t%s\t%s\t%s\t%s\t%s\n" "${CRPA_CONVENTION_TAG}" "${PARTITION}" "-" "-" "-" "hf_compatible=${CRPA_HF_COMPATIBLE}"
    if [[ -n "${CRPA_DEPENDENCY}" ]]; then
      printf "pre_gate\t%s\tnone\t%s\t%s\t%s\t%s\t%s\n" "${CRPA_DEPENDENCY}" "${PARTITION}" "-" "-" "-" "crpa_dependency"
    fi
  } > "${CRPA_MANIFEST}"

  echo "submitted bm_job=${bm_job}"
  echo "submitted array_job=${array_job}"
  echo "submitted merge_job=${merge_job}"
  echo "submission_manifest=${CRPA_MANIFEST}"
  SUBMITTED_MERGE_JOB="${merge_job}"
}

fillings_csv_from_manifest() {
  if [[ -n "${FILLINGS_CSV_OVERRIDE}" ]]; then
    printf "%s\n" "${FILLINGS_CSV_OVERRIDE}"
  else
    awk -F '\t' 'NR > 1 && !seen[$2]++ {print $2}' "${SELECTED_MANIFEST}" | sort -n | paste -sd, -
  fi
}

submit_hf() {
  repo_cd
  if [[ ! -f "${SELECTED_MANIFEST}" ]]; then
    echo "selected manifest not found: ${SELECTED_MANIFEST}" >&2
    exit 1
  fi

  local fillings_csv array_count array_spec
  fillings_csv="$(fillings_csv_from_manifest)"
  if [[ -z "${fillings_csv}" ]]; then
    echo "No fillings found in ${SELECTED_MANIFEST}" >&2
    exit 1
  fi
  array_count="$(awk -F, '{print NF}' <<<"${fillings_csv}")"
  array_spec="0-$((array_count - 1))%${HF_ARRAY_CONCURRENCY}"

  echo "[fig4-hf] selected_manifest=${SELECTED_MANIFEST}"
  echo "[fig4-hf] crpa_dir=${CRPA_DIR}"
  echo "[fig4-hf] fillings=${fillings_csv}"
  echo "[fig4-hf] array_spec=${array_spec}"
  echo "[fig4-hf] dependency=${DEPENDENCY:-none}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "dry_run=true"
    return 0
  fi

  local sbatch_args=(
    --parsable
    --array="${array_spec}"
    -p "${PARTITION}"
    -N 1
    --ntasks=1
    --cpus-per-task="${HF_CPUS}"
    --mem="${HF_MEM}"
    -t "${HF_TIME}"
    -J "mf_crpahf_lk24"
    -o "logs/custom_b0_hf_lk24_crpa_%A_%a.out"
    -e "logs/custom_b0_hf_lk24_crpa_%A_%a.err"
    --export=ALL,LD_LIBRARY_PATH=,HF_OUTPUT_ROOT="${HF_OUTPUT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",SELECTED_MANIFEST="${SELECTED_MANIFEST}",CRPA_DIR="${CRPA_DIR}",CRPA_RUN_TAG_SUFFIX="${CRPA_RUN_TAG_SUFFIX}",MAX_ITER="${MAX_ITER}",TARGET_LK="${TARGET_LK}",OVERLAP_LG="${OVERLAP_LG}",POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT}",PATH_KIND="${PATH_KIND}",INITIAL_STATE_RESAMPLE="${INITIAL_STATE_RESAMPLE}",FOCK_INTERPOLATION="${FOCK_INTERPOLATION}",FILLINGS_CSV="${fillings_csv//,/:}"
  )
  if [[ "${HF_EXCLUSIVE}" == "1" || "${HF_EXCLUSIVE}" == "true" ]]; then
    sbatch_args+=(--exclusive)
  fi
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=(--dependency="${DEPENDENCY}")
  fi

  local job_id
  job_id="$(
    sbatch "${sbatch_args[@]}" \
      scripts/submit_mean_field.sbatch \
      bash scripts/submit_tbg_crpa_fig4_pipeline.sh run-hf-array-task
  )"

  {
    printf "job_id\tarray_spec\tpartition\tmax_iter\ttarget_lk\toverlap_lg\tpath_kind\thf_output_root\tselected_manifest\tcrpa_dir\trun_tag_suffix\tfillings\tdependency\texclusive\n"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_id}" "${array_spec}" "${PARTITION}" "${MAX_ITER}" "${TARGET_LK}" "${OVERLAP_LG}" "${PATH_KIND}" \
      "${HF_OUTPUT_ROOT}" "${SELECTED_MANIFEST}" "${CRPA_DIR}" "${CRPA_RUN_TAG_SUFFIX}" "${fillings_csv}" "${DEPENDENCY:-none}" "${HF_EXCLUSIVE}"
  } > "${HF_MANIFEST}"

  echo "submitted hf_job=${job_id}"
  echo "submission_manifest=${HF_MANIFEST}"
}

run_hf_array_task() {
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${FILLINGS_CSV:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID and FILLINGS_CSV are required." >&2
    exit 2
  fi
  local normalized_fillings="${FILLINGS_CSV//:/,}"
  IFS=',' read -r -a fillings <<< "${normalized_fillings}"
  if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#fillings[@]}" ]]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} out of range for ${normalized_fillings}" >&2
    exit 2
  fi
  bash scripts/submit_tbg_crpa_fig4_pipeline.sh run-hf-filling "${SELECTED_MANIFEST}" "${fillings[${SLURM_ARRAY_TASK_ID}]}"
}

run_hf_filling() {
  if [[ $# -ne 2 ]]; then
    echo "Usage: $0 run-hf-filling <selected-manifest.tsv> <nu>" >&2
    exit 2
  fi
  local selected_manifest="$1"
  local target_nu="$2"
  local matched=0

  while IFS=$'\t' read -r theta_deg nu init_mode seed initial_state source_lk target_lk run_tag epsilon_r w1 source_energy source_converged source_summary source_iterations source_final_error; do
    if [[ "${theta_deg}" == "theta_deg" || "${nu}" != "${target_nu}" ]]; then
      continue
    fi
    matched=$((matched + 1))
    local crpa_run_tag="${run_tag}_${CRPA_RUN_TAG_SUFFIX}"
    echo "[fig4-hf] start index=${matched} nu=${nu} init=${init_mode}:${seed} run_tag=${crpa_run_tag}"
    echo "[fig4-hf] initial_state=${initial_state}"
    python scripts/mean_field_tools.py run_custom_b0_hf_case \
      --theta-deg "${theta_deg}" \
      --nu "${nu}" \
      --output-root "${HF_OUTPUT_ROOT}" \
      --run-tag "${crpa_run_tag}" \
      --lk "${target_lk:-${TARGET_LK}}" \
      --lg 9 \
      --overlap-lg "${OVERLAP_LG}" \
      --w0 79.7 \
      --w1 "${w1}" \
      --vf 2135.4 \
      --epsilon-r "${epsilon_r}" \
      --tanh-argument-scale-a 162.60162601626016 \
      --zero-limit finite \
      --max-iter "${MAX_ITER}" \
      --points-per-segment "${POINTS_PER_SEGMENT}" \
      --path-kind "${PATH_KIND}" \
      --write-scf-path \
      --init "${init_mode}:${seed}" \
      --initial-state "${initial_state}" \
      --initial-state-resample "${INITIAL_STATE_RESAMPLE}" \
      --summary-mode parts \
      --crpa-dir "${CRPA_DIR}" \
      --fock-interpolation "${FOCK_INTERPOLATION}"
  done < "${selected_manifest}"

  if [[ "${matched}" -eq 0 ]]; then
    echo "No selected rows found for nu=${target_nu} in ${selected_manifest}" >&2
    exit 1
  fi

  awk -F '\t' -v target_nu="${target_nu}" 'NR > 1 && $2 == target_nu && !seen[$1 "\t" $2 "\t" $8]++ {print $1 "\t" $2 "\t" $8}' "${selected_manifest}" |
  while IFS=$'\t' read -r theta_deg nu run_tag; do
    python scripts/mean_field_tools.py run_custom_b0_hf_case \
      --theta-deg "${theta_deg}" \
      --nu "${nu}" \
      --output-root "${HF_OUTPUT_ROOT}" \
      --run-tag "${run_tag}_${CRPA_RUN_TAG_SUFFIX}" \
      --combine-summary-only
  done
}

mode="${1:-submit-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${mode}" in
  submit-all)
    submit_crpa
    if [[ -n "${SUBMITTED_MERGE_JOB:-}" ]]; then
      DEPENDENCY="afterok:${SUBMITTED_MERGE_JOB}" submit_hf
    fi
    ;;
  submit-crpa)
    submit_crpa
    ;;
  submit-hf)
    submit_hf
    ;;
  run-hf-array-task)
    run_hf_array_task
    ;;
  run-hf-filling)
    run_hf_filling "$@"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage
    exit 2
    ;;
esac
