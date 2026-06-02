#!/bin/bash
# Plot exact SCF-mesh HF band snapshots and compute middle-flat-band Chern
# numbers for the completed xi=0 and xi=1 RLG/hBN Fig. 6 source states.
#
# Usage:
#   sbatch scripts/plot_completed_rlg_fig6_bands_chern.sbatch <source-root> <output-dir>

#SBATCH -J rlg_fig6_done_scf
#SBATCH --account=hmt03
#SBATCH -p regular6430
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/rlg_fig6_done_plot_%j.out
#SBATCH -e logs/rlg_fig6_done_plot_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <source-root> <output-dir>" >&2
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

SOURCE_ROOT="$(realpath "$1")"
OUT_DIR="$(realpath -m "$2")"
XI0_TASK="${SOURCE_ROOT}/tasks/task_0_xi0_V064meV_flavor_seed1"
XI1_TASK="${SOURCE_ROOT}/tasks/task_16_xi1_V064meV_flavor_seed1"
XI0_PANEL="${XI0_TASK}/xi0_V064meV"
XI1_PANEL="${XI1_TASK}/xi1_V064meV"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix}"

mkdir -p logs "${OUT_DIR}/xi0_V064meV" "${OUT_DIR}/xi1_V064meV"

for required in \
  "${XI0_TASK}/paper_hf_config.json" \
  "${XI0_PANEL}/hf_ground_state.npz" \
  "${XI0_PANEL}/hf_convergence.json" \
  "${XI1_PANEL}/hf_ground_state.npz" \
  "${XI1_PANEL}/hf_convergence.json"
do
  test -f "${required}"
done

cp -f "${XI0_TASK}/paper_hf_config.json" "${OUT_DIR}/paper_hf_config.json"
ln -sfn "${XI0_PANEL}/hf_ground_state.npz" "${OUT_DIR}/xi0_V064meV/hf_ground_state.npz"
ln -sfn "${XI0_PANEL}/hf_convergence.json" "${OUT_DIR}/xi0_V064meV/hf_convergence.json"
ln -sfn "${XI1_PANEL}/hf_ground_state.npz" "${OUT_DIR}/xi1_V064meV/hf_ground_state.npz"
ln -sfn "${XI1_PANEL}/hf_convergence.json" "${OUT_DIR}/xi1_V064meV/hf_convergence.json"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_fig6_done_plot_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

echo "[job] source_root=${SOURCE_ROOT}"
echo "[job] out_dir=${OUT_DIR}"
echo "[job] cache_dir=${CACHE_DIR}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

PATH_KIND="${PATH_KIND:-paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma}"

python scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
  --checkpoint "${OUT_DIR}/xi0_V064meV/hf_ground_state.npz" \
  --config "${OUT_DIR}/paper_hf_config.json" \
  --panel xi0_V064meV \
  --output-dir "${OUT_DIR}/xi0_V064meV/scf_exact_path" \
  --path-kind "${PATH_KIND}" \
  --ylim-mev=-90,90 \
  --dpi 180

python scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
  --checkpoint "${OUT_DIR}/xi1_V064meV/hf_ground_state.npz" \
  --config "${OUT_DIR}/paper_hf_config.json" \
  --panel xi1_V064meV \
  --output-dir "${OUT_DIR}/xi1_V064meV/scf_exact_path" \
  --path-kind "${PATH_KIND}" \
  --ylim-mev=-90,90 \
  --dpi 180

python - "${OUT_DIR}" "${PATH_KIND}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

out_dir = Path(sys.argv[1])
path_kind = sys.argv[2]
rows = []
for panel in ("xi0_V064meV", "xi1_V064meV"):
    png = out_dir / panel / "scf_exact_path" / f"current_checkpoint_scf_exact_{path_kind}_bands.png"
    if not png.exists():
        raise FileNotFoundError(png)
    rows.append((panel, png))

fig, axes = plt.subplots(1, len(rows), figsize=(8.8, 3.6), constrained_layout=True)
if len(rows) == 1:
    axes = [axes]
for ax, (panel, png) in zip(axes, rows, strict=True):
    ax.imshow(mpimg.imread(png))
    ax.set_title(panel, fontsize=9)
    ax.axis("off")
fig.savefig(out_dir / "paper_fig6_hf_bands_scf_exact_contact_sheet.png", dpi=180)
plt.close(fig)
PY

python scripts/compute_rlg_hbn_hf_chern.py \
  --source-dir "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --spin-index 0 \
  --eta-index 0 \
  --compare-paper-fig6

test -f "${OUT_DIR}/paper_fig6_hf_bands_scf_exact_contact_sheet.png"
test -f "${OUT_DIR}/xi0_V064meV/scf_exact_path/current_checkpoint_scf_exact_${PATH_KIND}_bands.npz"
test -f "${OUT_DIR}/xi1_V064meV/scf_exact_path/current_checkpoint_scf_exact_${PATH_KIND}_bands.npz"
test -f "${OUT_DIR}/hf_chern_summary.json"
test -f "${OUT_DIR}/hf_chern_report.md"

echo "[done] bands_png=${OUT_DIR}/paper_fig6_hf_bands_scf_exact_contact_sheet.png"
echo "[done] chern_summary=${OUT_DIR}/hf_chern_summary.json"
echo "[done] chern_report=${OUT_DIR}/hf_chern_report.md"
