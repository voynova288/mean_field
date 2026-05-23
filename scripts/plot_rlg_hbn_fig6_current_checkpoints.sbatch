#!/bin/bash
# Plot current exact-SCF-path band snapshots for all checkpoints under an
# existing RLG/hBN Fig. 6 output root.
#
# Usage:
#   sbatch scripts/plot_rlg_hbn_fig6_current_checkpoints.sbatch <output-root> [snapshot-name]

#SBATCH -J rlg_fig6_ckpt_bands
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/rlg_fig6_ckpt_bands_%j.out
#SBATCH -e logs/rlg_fig6_ckpt_bands_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <output-root> [snapshot-name]" >&2
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

OUTPUT_ROOT="$1"
SNAPSHOT_NAME="${2:-current_checkpoint_scf_exact_$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_ROOT="${OUTPUT_ROOT}/checkpoint_band_snapshots/${SNAPSHOT_NAME}"
PATH_KIND="${PATH_KIND:-paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma}"
YLIM_MEV="${YLIM_MEV:--120,120}"
DPI="${DPI:-180}"

mkdir -p "${REPO_ROOT}/logs" "${SNAPSHOT_ROOT}/status"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_fig6_ckpt_bands_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

# Each worker is an independent 1-thread plot process.  The Slurm CPU request
# is consumed by running many checkpoints concurrently rather than by
# oversubscribing BLAS inside each small plot.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

WORKERS="${PLOT_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "[job] output_root=${OUTPUT_ROOT}"
echo "[job] snapshot_root=${SNAPSHOT_ROOT}"
echo "[job] path_kind=${PATH_KIND}"
echo "[job] ylim_mev=${YLIM_MEV}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] workers=${WORKERS}"

mapfile -t CHECKPOINT_JSONS < <(
  find "${OUTPUT_ROOT}/tasks" \
    -path '*/runs/*/checkpoints/hf_checkpoint_latest.json' \
    -type f | sort -V
)

if [[ "${#CHECKPOINT_JSONS[@]}" -eq 0 ]]; then
  echo "[error] no hf_checkpoint_latest.json files found under ${OUTPUT_ROOT}/tasks" >&2
  exit 1
fi
echo "[job] checkpoint_count=${#CHECKPOINT_JSONS[@]}"

plot_one() {
  local checkpoint_json="$1"

  local checkpoint_dir run_dir runs_dir panel_dir task_dir
  checkpoint_dir="$(dirname "${checkpoint_json}")"
  run_dir="$(dirname "${checkpoint_dir}")"
  runs_dir="$(dirname "${run_dir}")"
  panel_dir="$(dirname "${runs_dir}")"
  task_dir="$(dirname "${panel_dir}")"

  local task_name panel run_name config checkpoint_npz iteration init_mode seed
  task_name="$(basename "${task_dir}")"
  panel="$(basename "${panel_dir}")"
  run_name="$(basename "${run_dir}")"
  config="${task_dir}/paper_hf_config.json"

  if [[ ! -f "${config}" ]]; then
    echo "[skip] ${task_name}/${run_name}: missing ${config}" >&2
    return 0
  fi

  checkpoint_npz="$(
    python - "${checkpoint_json}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("checkpoint_npz", ""))
PY
  )"
  iteration="$(
    python - "${checkpoint_json}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("iteration", "unknown"))
PY
  )"
  init_mode="$(
    python - "${checkpoint_json}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("init_mode", "unknown"))
PY
  )"
  seed="$(
    python - "${checkpoint_json}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("seed", "unknown"))
PY
  )"

  if [[ -z "${checkpoint_npz}" || ! -f "${checkpoint_npz}" ]]; then
    echo "[skip] ${task_name}/${run_name}: missing checkpoint npz ${checkpoint_npz}" >&2
    return 0
  fi

  local out_dir input_dir local_npz local_json local_config status_file safe_iter
  safe_iter="${iteration//[^A-Za-z0-9_.-]/_}"
  out_dir="${SNAPSHOT_ROOT}/${task_name}/iter${safe_iter}_${run_name}"
  input_dir="${out_dir}/inputs"
  status_file="${SNAPSHOT_ROOT}/status/${task_name}_${run_name}.tsv"
  mkdir -p "${input_dir}" "${out_dir}"

  local_npz="${input_dir}/hf_checkpoint_latest.npz"
  local_json="${input_dir}/hf_checkpoint_latest.json"
  local_config="${input_dir}/paper_hf_config.json"
  cp -f "${checkpoint_npz}" "${local_npz}"
  cp -f "${checkpoint_json}" "${local_json}"
  cp -f "${config}" "${local_config}"

  python scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
    --checkpoint "${local_npz}" \
    --config "${local_config}" \
    --panel "${panel}" \
    --output-dir "${out_dir}" \
    --path-kind "${PATH_KIND}" \
    "--ylim-mev=${YLIM_MEV}" \
    --dpi "${DPI}"

  local png
  png="${out_dir}/current_checkpoint_scf_exact_${PATH_KIND}_bands.png"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task_name}" "${panel}" "${run_name}" "${init_mode}" "${seed}" "${iteration}" "${png}" \
    > "${status_file}"
}
export -f plot_one
export SNAPSHOT_ROOT PATH_KIND YLIM_MEV DPI

printf "%s\n" "${CHECKPOINT_JSONS[@]}" | xargs -r -n 1 -P "${WORKERS}" bash -c 'plot_one "$0"'

MANIFEST="${SNAPSHOT_ROOT}/checkpoint_band_plots.tsv"
{
  echo -e "task\tpanel\trun\tinit_mode\tseed\titeration\tpng"
  find "${SNAPSHOT_ROOT}/status" -type f -name '*.tsv' -print0 | sort -zV | xargs -0r cat
} > "${MANIFEST}"

python - "${MANIFEST}" "${SNAPSHOT_ROOT}/checkpoint_band_plots_contact_sheet.png" <<'PY'
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

manifest = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
    if not line.strip():
        continue
    task, panel, run, init_mode, seed, iteration, png = line.split("\t")
    if Path(png).exists():
        rows.append((task, panel, run, iteration, Path(png)))

if not rows:
    raise SystemExit("no plotted PNG files found for contact sheet")

cols = 3
fig_rows = math.ceil(len(rows) / cols)
fig, axes = plt.subplots(fig_rows, cols, figsize=(cols * 4.4, fig_rows * 3.7), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")

for ax, (task, panel, run, iteration, png) in zip(axes.ravel(), rows, strict=False):
    image = mpimg.imread(png)
    ax.imshow(image)
    ax.set_title(f"{task}\n{panel} {run} iter {iteration}", fontsize=8)
    ax.axis("off")

fig.tight_layout()
fig.savefig(output, dpi=180)
plt.close(fig)
print(f"[done] contact_sheet={output}", flush=True)
PY

echo "[done] manifest=${MANIFEST}"
echo "[done] snapshot_root=${SNAPSHOT_ROOT}"
