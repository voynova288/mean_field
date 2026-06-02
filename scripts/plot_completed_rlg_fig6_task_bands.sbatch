#!/bin/bash
# Plot exact-SCF-path band figures for every completed RLG/hBN Fig. 6 task
# under a source task root. A completed task is one with both
# <panel>/hf_ground_state.npz and <panel>/hf_convergence.json.
#
# Usage:
#   sbatch scripts/plot_completed_rlg_fig6_task_bands.sbatch <source-root> <output-dir>

#SBATCH -J rlg_done_bands
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/rlg_done_bands_%j.out
#SBATCH -e logs/rlg_done_bands_%j.err

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
PATH_KIND="${PATH_KIND:-paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma}"
Y_LIM="${Y_LIM:--90,90}"
DPI="${DPI:-180}"

mkdir -p logs "${OUT_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_done_bands_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

MANIFEST="${OUT_DIR}/completed_task_band_manifest.tsv"
printf 'task\tpanel\tinit_mode\tseed\titerations\tfinal_error\tfinal_energy_mev\texit_reason\tpng\tpdf\n' > "${MANIFEST}"

mapfile -t TASK_LINES < <(python - "${SOURCE_ROOT}" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
rows = []
for task in sorted(root.glob('tasks/task_*')):
    match = re.match(r'task_(\d+)_(xi[-0-9]+_V\d+meV)_(.+)_seed(\d+)$', task.name)
    if match is None:
        continue
    task_id = int(match.group(1))
    panel = match.group(2)
    init_mode = match.group(3)
    seed = int(match.group(4))
    panel_dir = task / panel
    if not (task / 'paper_hf_config.json').is_file():
        continue
    if not (panel_dir / 'hf_ground_state.npz').is_file():
        continue
    if not (panel_dir / 'hf_convergence.json').is_file():
        continue
    conv = json.loads((panel_dir / 'hf_convergence.json').read_text(encoding='utf-8'))
    best = conv.get('best') or {}
    rows.append((task_id, task.name, panel, init_mode, seed, best.get('iterations'), best.get('final_error'), best.get('final_energy_mev'), best.get('exit_reason')))
for row in sorted(rows):
    print('\t'.join('' if value is None else str(value) for value in row))
PY
)

if (( ${#TASK_LINES[@]} == 0 )); then
  echo "No completed tasks found under ${SOURCE_ROOT}/tasks" >&2
  exit 1
fi

echo "[job] source_root=${SOURCE_ROOT}"
echo "[job] out_dir=${OUT_DIR}"
echo "[job] completed_count=${#TASK_LINES[@]}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

for line in "${TASK_LINES[@]}"; do
  IFS=$'\t' read -r task_id task_name panel init_mode seed iterations final_error final_energy exit_reason <<< "${line}"
  TASK_DIR="${SOURCE_ROOT}/tasks/${task_name}"
  PANEL_DIR="${TASK_DIR}/${panel}"
  TASK_OUT="${OUT_DIR}/${task_name}/${panel}/scf_exact_path"
  mkdir -p "${TASK_OUT}"
  echo "[plot] task=${task_name} panel=${panel} iter=${iterations} err=${final_error}"
  python scripts/plot_rlg_hbn_checkpoint_scf_exact_path.py \
    --checkpoint "${PANEL_DIR}/hf_ground_state.npz" \
    --config "${TASK_DIR}/paper_hf_config.json" \
    --panel "${panel}" \
    --output-dir "${TASK_OUT}" \
    --path-kind "${PATH_KIND}" \
    --ylim-mev="${Y_LIM}" \
    --dpi "${DPI}"
  PNG="${TASK_OUT}/current_checkpoint_scf_exact_${PATH_KIND}_bands.png"
  PDF="${TASK_OUT}/current_checkpoint_scf_exact_${PATH_KIND}_bands.pdf"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task_name}" "${panel}" "${init_mode}" "${seed}" "${iterations}" "${final_error}" "${final_energy}" "${exit_reason}" "${PNG}" "${PDF}" >> "${MANIFEST}"
done

python - "${OUT_DIR}" "${MANIFEST}" <<'PY'
from __future__ import annotations
from pathlib import Path
import csv
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

out_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open(encoding='utf-8'), delimiter='\t'))
if not rows:
    raise SystemExit('empty manifest')
cols = min(3, len(rows))
rows_n = math.ceil(len(rows) / cols)
fig, axes = plt.subplots(rows_n, cols, figsize=(4.7 * cols, 3.8 * rows_n), constrained_layout=True)
if rows_n == 1 and cols == 1:
    axes_grid = [[axes]]
elif rows_n == 1:
    axes_grid = [list(axes)]
elif cols == 1:
    axes_grid = [[ax] for ax in axes]
else:
    axes_grid = axes.tolist()
flat_axes = [ax for row in axes_grid for ax in row]
for ax in flat_axes[len(rows):]:
    ax.axis('off')
for ax, row in zip(flat_axes, rows, strict=False):
    png = Path(row['png'])
    ax.imshow(mpimg.imread(png))
    err_text = row['final_error']
    try:
        err_text = f"{float(err_text):.2e}"
    except Exception:
        pass
    ax.set_title(f"{row['task']}\n{row['panel']} {row['init_mode']}:{row['seed']} err={err_text}", fontsize=8)
    ax.axis('off')
contact = out_dir / 'completed_task_bands_contact_sheet.png'
fig.savefig(contact, dpi=180)
plt.close(fig)

md = out_dir / 'completed_task_band_report.md'
lines = [
    '# Completed RLG/hBN Fig. 6 task band plots',
    '',
    f'- Manifest: `{manifest}`',
    f'- Contact sheet: `{contact}`',
    '',
    '| task | panel | init | seed | iter | final error | exit | png |',
    '| --- | --- | --- | ---: | ---: | ---: | --- | --- |',
]
for row in rows:
    lines.append(
        f"| {row['task']} | {row['panel']} | {row['init_mode']} | {row['seed']} | {row['iterations']} | {row['final_error']} | {row['exit_reason']} | `{row['png']}` |"
    )
md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"[done] contact_sheet={contact}")
print(f"[done] manifest={manifest}")
print(f"[done] report={md}")
PY
