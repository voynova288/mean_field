#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-results/HTG/htg_fig9b_bandwidth_scan_8x10_multiinit_final_error_20260509_001}"
shift || true

# Explicit calculation centers for Kwan Fig. 9b. Do not infer these from major
# tick labels, and do not use plotted cell edges as calculation points.
thetas=(1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95)
waas=(40 47.5 55 60 65 70 75 80 85 90)
cases=()
for waa in "${waas[@]}"; do
  for theta in "${thetas[@]}"; do
    cases+=("${theta}:${waa}:theta${theta}_wAA${waa}")
  done
done

IFS=,
cases_arg="${cases[*]}"
unset IFS

python3 scripts/mean_field_tools.py run_htg_fig9b_exact_anchor_scan \
  --output-dir "${OUTPUT_DIR}" \
  --cases "${cases_arg}" \
  --artifact-prefix fig9b_conduction_bandwidth_scan_8x10_final_error \
  --report-title "HTG Fig. 9b 8x10 Multi-Init Final-Error Scan" \
  --init-modes d3b,d3a,bm,fi,flavor,vp,sp,chern,perturbed,random \
  "$@"
