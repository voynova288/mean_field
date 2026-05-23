#!/bin/bash
# Short startup gate for the Zhang Supplementary Fig. 4 HF+cRPA runner.

set -euo pipefail

CRPA_DIR="${CRPA_DIR:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_merged}"
CRPA_PHYSICS_REFERENCE_DIR="${CRPA_PHYSICS_REFERENCE_DIR:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/hf_crpa_validation_smoke_20260510}"

python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu 0 \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "fig4_crpa_matrixdiag_smoke" \
  --lk 24 \
  --lg 9 \
  --overlap-lg 9 \
  --w0 79.7 \
  --w1 97.4 \
  --vf 2135.4 \
  --epsilon-r 10 \
  --tanh-argument-scale-a 162.60162601626016 \
  --zero-limit finite \
  --max-iter 1 \
  --points-per-segment 8 \
  --path-kind gamma-m-k-gamma-kprime \
  --init bm:1 \
  --summary-mode root \
  --crpa-dir "${CRPA_DIR}" \
  --allow-incompatible-crpa \
  --diagnostic-only \
  --crpa-physics-reference-dir "${CRPA_PHYSICS_REFERENCE_DIR}" \
  --fock-interpolation matrix_diagonal \
  --path-fock-interpolation linear
