#!/bin/bash
# Focused gate for the cRPA/HF convention split repaired on 2026-05-19.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_hf_logic_gate_20260519}"
ZHANG_CRPA_DIR="${ZHANG_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_recheck_20260516_merged}"
HF_CRPA_DIR="${HF_CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_hfperiodic_qshiftfix_20260518_merged}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_hf_logic_gate_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

echo "[crpa-hf-logic-gate] repo_root=${REPO_ROOT}"
echo "[crpa-hf-logic-gate] out_dir=${OUT_DIR}"
echo "[crpa-hf-logic-gate] zhang_crpa_dir=${ZHANG_CRPA_DIR}"
echo "[crpa-hf-logic-gate] hf_crpa_dir=${HF_CRPA_DIR}"

echo "[crpa-hf-logic-gate] pytest start"
python -m pytest -q tests/test_crpa_core.py | tee "${OUT_DIR}/test_crpa_core.log"

echo "[crpa-hf-logic-gate] validate Zhang paper-reference artifact"
python -m mean_field.devtools.validate_tbg_crpa_artifact \
  --crpa-dir "${ZHANG_CRPA_DIR}" \
  --output-dir "${OUT_DIR}/artifact_validation"

echo "[crpa-hf-logic-gate] validate HF-compatible artifact metadata/algebra"
python -m mean_field.devtools.validate_tbg_crpa_artifact \
  --crpa-dir "${HF_CRPA_DIR}" \
  --require-hf-compatible \
  --output-dir "${OUT_DIR}/artifact_validation"

echo "[crpa-hf-logic-gate] representative Zhang Fig. 1(e) points"
python - "${ZHANG_CRPA_DIR}" "${OUT_DIR}/fig1e_representative_points.csv" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from mean_field.crpa import load_crpa_result
from mean_field.crpa.diagnostics import representative_fig1e_window_curve
from mean_field.crpa.validation import DEFAULT_FIG1E_PAPER_POINTS, compare_fig1e_window_to_paper_points

crpa_dir = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
result = load_crpa_result(crpa_dir)
xs, ys, _counts = representative_fig1e_window_curve(result)
comparison = compare_fig1e_window_to_paper_points(result)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["q_nm_inv", "computed_eps_times_bn", "paper_eps_times_bn", "diff"])
    for q_value, reference_value in DEFAULT_FIG1E_PAPER_POINTS:
        computed = float(np.interp(float(q_value), xs, ys))
        writer.writerow([f"{q_value:.12g}", f"{computed:.12g}", f"{reference_value:.12g}", f"{computed - reference_value:.12g}"])

print(f"[fig1e-points] csv={out_csv}")
print(
    "[fig1e-points] "
    f"rmse={float(comparison['fig1e_paper_rmse']):.12g} "
    f"max_abs={float(comparison['fig1e_paper_max_abs']):.12g} "
    f"mean_abs={float(comparison['fig1e_paper_mean_abs']):.12g}"
)
for q_value, reference_value in DEFAULT_FIG1E_PAPER_POINTS:
    computed = float(np.interp(float(q_value), xs, ys))
    print(
        "[fig1e-points] "
        f"q={q_value:.6g} computed={computed:.12g} paper={reference_value:.12g} diff={computed - reference_value:.12g}"
    )
PY

echo "[crpa-hf-logic-gate] done"
