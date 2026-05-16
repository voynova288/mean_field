#!/bin/bash
# Run one filling of the no-cRPA Wang/Zhang bare HF framework band-plot job.

set -euo pipefail

FILLINGS_CSV="${FILLINGS_CSV:--3,-2,-1,0,1,2,3}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/data/home/ziyuzhu/Mean_Field/benchmarks/Liu_reproduce_ref}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/hf_framework_bands/liu_ref_lk24_20260516_wang_zhang_bare_converged}"
MAX_ITER="${MAX_ITER:-3000}"
PRECISION="${PRECISION:-1e-5}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-120}"
THRESHOLD="${THRESHOLD:-1e-7}"
USE_NUMBA="${USE_NUMBA:-auto}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required." >&2
  exit 2
fi

normalized_fillings="${FILLINGS_CSV//:/,}"
IFS=',' read -r -a fillings <<< "${normalized_fillings}"
if [[ "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${#fillings[@]}" ]]; then
  echo "Array task ${SLURM_ARRAY_TASK_ID} out of range for ${normalized_fillings}" >&2
  exit 2
fi

nu="${fillings[${SLURM_ARRAY_TASK_ID}]}"
echo "[bare-hf-bands] task=${SLURM_ARRAY_TASK_ID} nu=${nu}"
echo "[bare-hf-bands] reference_root=${REFERENCE_ROOT}"
echo "[bare-hf-bands] output_dir=${OUTPUT_DIR}"

python scripts/mean_field_tools.py run_bare_hf_framework_band_plots_against_liu_ref \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --nu "${nu}" \
  --max-iter "${MAX_ITER}" \
  --precision "${PRECISION}" \
  --points-per-segment "${POINTS_PER_SEGMENT}" \
  --threshold "${THRESHOLD}" \
  --use-numba "${USE_NUMBA}"
