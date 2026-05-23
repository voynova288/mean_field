#!/bin/bash
# Merge the completed HF-compatible Fig. 4 cRPA chunks and write diagnostics.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/TBG_HF_cRPA}"
TAG="${CRPA_CONVENTION_TAG:-hfcompatible_fig4_20260522_epsbn4}"
CHUNK_COUNT="${CHUNK_COUNT:-144}"
CHUNK_ROOT="${CHUNK_ROOT:-${OUTPUT_ROOT}/crpa_lk24_lg9_q11_${TAG}_chunks}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_ROOT}/crpa_lk24_lg9_q11_${TAG}_merged}"

cd "${REPO_ROOT}"

chunk_args=()
for ((i = 0; i < CHUNK_COUNT; i++)); do
  chunk_dir="${CHUNK_ROOT}/chunk_${i}"
  if [[ ! -f "${chunk_dir}/crpa_params.json" ]]; then
    echo "Missing chunk artifact: ${chunk_dir}" >&2
    exit 2
  fi
  chunk_args+=(--chunk "${chunk_dir}")
done

echo "[merge] chunk_root=${CHUNK_ROOT}"
echo "[merge] merged_dir=${MERGED_DIR}"
echo "[merge] chunk_count=${CHUNK_COUNT}"

python scripts/mean_field_tools.py merge_tbg_crpa_chunks \
  --output-dir "${MERGED_DIR}" \
  "${chunk_args[@]}"
