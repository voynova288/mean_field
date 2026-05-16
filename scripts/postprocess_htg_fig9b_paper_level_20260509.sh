#!/bin/bash
set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
OUTPUT_DIR="${1:-${REPO_ROOT}/results/HTG/htg_fig9b_bandwidth_scan_8x10_paper_level_20260509_001}"
PREFIX="${2:-fig9b_conduction_bandwidth_scan_8x10_paper_level}"

cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_htg9b_paper_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

TSV="${OUTPUT_DIR}/${PREFIX}.tsv"
if [[ ! -s "${TSV}" ]]; then
  echo "Missing scan TSV: ${TSV}" >&2
  exit 2
fi

python3 scripts/plot_htg_fig9b_bandwidth_from_tsv.py "${TSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}" \
  --require-fig9b-8x10

python3 scripts/validate_htg_fig9b_reproduction.py \
  --scan-dir "${OUTPUT_DIR}" \
  --scan-prefix "${PREFIX}"

python3 scripts/analyze_htg_fig9b_seed_convergence.py \
  "${OUTPUT_DIR}/${PREFIX}.tsv" \
  "${OUTPUT_DIR}/${PREFIX}_run_details.tsv" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}_seed_convergence"

find "${OUTPUT_DIR}" -maxdepth 1 -type f \
  \( -name "${PREFIX}*.png" -o -name "${PREFIX}*.pdf" -o -name "${PREFIX}_seed_convergence.*" -o -name "grid_metadata.json" -o -name "fig9b_reproduction_validation.md" \) \
  -printf "%f %s bytes\n" | sort
