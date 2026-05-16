#!/bin/bash
# Slurm array wrapper for selected cRPA-HF filling jobs.

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must run inside a Slurm array job." >&2
  exit 2
fi

if [[ -z "${SELECTED_MANIFEST:-}" ]]; then
  echo "SELECTED_MANIFEST is required." >&2
  exit 2
fi

if [[ -z "${FILLINGS_CSV:-}" ]]; then
  echo "FILLINGS_CSV is required." >&2
  exit 2
fi

IFS=',' read -r -a fillings <<< "${FILLINGS_CSV}"
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#fillings[@]}" ]]; then
  echo "Array task ${SLURM_ARRAY_TASK_ID} out of range for fillings ${FILLINGS_CSV}" >&2
  exit 2
fi

nu="${fillings[${SLURM_ARRAY_TASK_ID}]}"
echo "[selected-crpa-array] task_id=${SLURM_ARRAY_TASK_ID} nu=${nu} fillings=${FILLINGS_CSV}"
bash scripts/run_custom_b0_hf_lk24_selected_crpa_serial.sh "${SELECTED_MANIFEST}" "${nu}"
