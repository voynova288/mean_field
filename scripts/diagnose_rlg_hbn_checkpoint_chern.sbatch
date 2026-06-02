#!/bin/bash
# Diagnose one RLG/hBN HF checkpoint with exact SCF-mesh path bands and sewn
# projected-basis Chern numbers.  This intentionally reads an existing
# checkpoint; it does not rebuild or rerun HF.
#
# Usage:
#   sbatch scripts/diagnose_rlg_hbn_checkpoint_chern.sbatch \
#     <checkpoint-npz> <paper-hf-config-json> <cache-manifest-json> <panel-name> <output-dir>

#SBATCH -J rlg_ckpt_chern
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/rlg_ckpt_chern_%j.out
#SBATCH -e logs/rlg_ckpt_chern_%j.err

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: sbatch $0 <checkpoint-npz> <paper-hf-config-json> <cache-manifest-json> <panel-name> <output-dir>" >&2
  exit 2
fi

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

CHECKPOINT="$(realpath "$1")"
CONFIG="$(realpath "$2")"
CACHE_MANIFEST="$(realpath "$3")"
PANEL="$4"
OUT_DIR="$(realpath -m "$5")"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix}"

mkdir -p logs "${OUT_DIR}/${PANEL}"
test -f "${CHECKPOINT}"
test -f "${CONFIG}"
test -f "${CACHE_MANIFEST}"

cp -f "${CONFIG}" "${OUT_DIR}/paper_hf_config.json"
ln -sfn "${CHECKPOINT}" "${OUT_DIR}/${PANEL}/hf_ground_state.npz"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_ckpt_chern_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

python - "${CACHE_MANIFEST}" "${OUT_DIR}/${PANEL}/hf_convergence.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = payload.get("entries", [])
basis = next((entry for entry in entries if entry.get("kind") == "basis"), None)
overlap = next((entry for entry in entries if entry.get("kind") == "overlap"), None)
screening = next((entry for entry in entries if entry.get("kind") == "screening"), None)
if basis is None:
    raise SystemExit(f"{manifest_path} has no basis entry")
output_path.write_text(
    json.dumps(
        {
            "basis_cache_key": str(basis["key"]),
            "overlap_cache_key": "" if overlap is None else str(overlap["key"]),
            "screening_cache_key": "" if screening is None else str(screening["key"]),
            "diagnostic_source": str(manifest_path),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "[job] checkpoint=${CHECKPOINT}"
echo "[job] config=${CONFIG}"
echo "[job] cache_manifest=${CACHE_MANIFEST}"
echo "[job] panel=${PANEL}"
echo "[job] out_dir=${OUT_DIR}"
echo "[job] cache_dir=${CACHE_DIR}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

PATH_KIND="${PATH_KIND:-paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma}"

python scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
  --checkpoint "${OUT_DIR}/${PANEL}/hf_ground_state.npz" \
  --config "${OUT_DIR}/paper_hf_config.json" \
  --panel "${PANEL}" \
  --output-dir "${OUT_DIR}/${PANEL}/scf_exact_path" \
  --path-kind "${PATH_KIND}" \
  --ylim-mev=-120,120 \
  --dpi 180

python scripts/compute_rlg_hbn_hf_chern.py \
  --source-dir "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --spin-index 0 \
  --eta-index 0 \
  --compare-paper-fig6

test -f "${OUT_DIR}/${PANEL}/scf_exact_path/current_checkpoint_scf_exact_${PATH_KIND}_bands.npz"
test -f "${OUT_DIR}/${PANEL}/hf_chern_numbers.json"
test -f "${OUT_DIR}/hf_chern_summary.json"
test -f "${OUT_DIR}/hf_chern_report.md"

echo "[done] bands_npz=${OUT_DIR}/${PANEL}/scf_exact_path/current_checkpoint_scf_exact_${PATH_KIND}_bands.npz"
echo "[done] chern_summary=${OUT_DIR}/hf_chern_summary.json"
