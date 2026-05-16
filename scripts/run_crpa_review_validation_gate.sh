#!/bin/bash
# Focused cRPA review gate: unit tests plus diagnostics from an existing artifact.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/ziyuzhu/Mean_Field}"
CRPA_DIR="${CRPA_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_merged}"

cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_review_gate_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

echo "[crpa-review-gate] repo_root=${REPO_ROOT}"
echo "[crpa-review-gate] crpa_dir=${CRPA_DIR}"
echo "[crpa-review-gate] pytest start"
python -m pytest -q tests/test_crpa_core.py

echo "[crpa-review-gate] diagnostics start"
python scripts/mean_field_tools.py diagnose_tbg_crpa_epsilon --crpa-dir "${CRPA_DIR}"
echo "[crpa-review-gate] done"
