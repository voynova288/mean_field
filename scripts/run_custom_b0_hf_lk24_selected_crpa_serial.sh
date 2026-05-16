#!/bin/bash
# Run selected custom B0 HF refinements for one filling with cRPA-screened interaction.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <selected-manifest.tsv> <nu>" >&2
  exit 2
fi

SELECTED_MANIFEST="$1"
TARGET_NU="$2"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/hf_crpa_production_runs}"
CRPA_DIR="${CRPA_DIR:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hf_compatible_cnpindex_packed_20260515_merged}"
CRPA_RUN_TAG_SUFFIX="${CRPA_RUN_TAG_SUFFIX:-crpa_hfcompat_cnp_packed_lk24_q11_20260515_gamma_m_k_gamma_kprime}"
MAX_ITER="${MAX_ITER:-3000}"
TARGET_LK="${TARGET_LK:-24}"
OVERLAP_LG="${OVERLAP_LG:-9}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-120}"
PATH_KIND="${PATH_KIND:-gamma-m-k-gamma-kprime}"
INITIAL_STATE_RESAMPLE="${INITIAL_STATE_RESAMPLE:-bilinear}"
FOCK_INTERPOLATION="${FOCK_INTERPOLATION:-matrix_diagonal}"

if [[ ! -f "${SELECTED_MANIFEST}" ]]; then
  echo "selected manifest not found: ${SELECTED_MANIFEST}" >&2
  exit 1
fi

if [[ ! -d "${CRPA_DIR}" ]]; then
  echo "cRPA directory not found yet: ${CRPA_DIR}" >&2
  exit 1
fi

echo "[selected-crpa-lk24] target_nu=${TARGET_NU}"
echo "[selected-crpa-lk24] selected_manifest=${SELECTED_MANIFEST}"
echo "[selected-crpa-lk24] output_root=${OUTPUT_ROOT}"
echo "[selected-crpa-lk24] crpa_dir=${CRPA_DIR}"
echo "[selected-crpa-lk24] max_iter=${MAX_ITER}"
echo "[selected-crpa-lk24] target_lk=${TARGET_LK}"
echo "[selected-crpa-lk24] overlap_lg=${OVERLAP_LG}"
echo "[selected-crpa-lk24] path_kind=${PATH_KIND}"
echo "[selected-crpa-lk24] initial_state_resample=${INITIAL_STATE_RESAMPLE}"
echo "[selected-crpa-lk24] fock_interpolation=${FOCK_INTERPOLATION}"

matched=0
while IFS=$'\t' read -r theta_deg nu init_mode seed initial_state source_lk target_lk run_tag epsilon_r w1 source_energy source_converged source_summary source_iterations source_final_error; do
  if [[ "${theta_deg}" == "theta_deg" ]]; then
    continue
  fi
  if [[ "${nu}" != "${TARGET_NU}" ]]; then
    continue
  fi
  matched=$((matched + 1))
  crpa_run_tag="${run_tag}_${CRPA_RUN_TAG_SUFFIX}"

  echo "[selected-crpa-lk24] start index=${matched} theta_deg=${theta_deg} nu=${nu} init=${init_mode}:${seed}"
  echo "[selected-crpa-lk24] source_energy=${source_energy} source_converged=${source_converged}"
  echo "[selected-crpa-lk24] source_iterations=${source_iterations:-NA} source_final_error=${source_final_error:-NA}"
  echo "[selected-crpa-lk24] source_summary=${source_summary}"
  echo "[selected-crpa-lk24] source_lk=${source_lk} target_lk=${target_lk}"
  echo "[selected-crpa-lk24] initial_state=${initial_state}"
  echo "[selected-crpa-lk24] run_tag=${crpa_run_tag}"

  python scripts/mean_field_tools.py run_custom_b0_hf_case \
    --theta-deg "${theta_deg}" \
    --nu "${nu}" \
    --output-root "${OUTPUT_ROOT}" \
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
done < "${SELECTED_MANIFEST}"

if [[ "${matched}" -eq 0 ]]; then
  echo "No selected cRPA tasks found for nu=${TARGET_NU} in ${SELECTED_MANIFEST}" >&2
  exit 1
fi

awk -F '\t' -v target_nu="${TARGET_NU}" 'NR > 1 && $2 == target_nu && !seen[$1 "\t" $2 "\t" $8]++ {print $1 "\t" $2 "\t" $8}' "${SELECTED_MANIFEST}" |
while IFS=$'\t' read -r theta_deg nu run_tag; do
  crpa_run_tag="${run_tag}_${CRPA_RUN_TAG_SUFFIX}"
  echo "[selected-crpa-lk24] combine theta_deg=${theta_deg} nu=${nu} run_tag=${crpa_run_tag}"
  python scripts/mean_field_tools.py run_custom_b0_hf_case \
    --theta-deg "${theta_deg}" \
    --nu "${nu}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-tag "${crpa_run_tag}" \
    --combine-summary-only
done

echo "[selected-crpa-lk24] completed_serial_tasks=${matched}"
