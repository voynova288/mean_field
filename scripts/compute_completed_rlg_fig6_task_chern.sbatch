#!/bin/bash
# Compute/aggregate Chern numbers for every completed RLG/hBN Fig. 6 task
# under a source root. Completed means the task panel has both
# hf_ground_state.npz and hf_convergence.json.
#
# Usage:
#   sbatch scripts/compute_completed_rlg_fig6_task_chern.sbatch <source-root> <output-dir> [cache-dir]

#SBATCH -J rlg_done_chern
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/rlg_done_chern_%j.out
#SBATCH -e logs/rlg_done_chern_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <source-root> <output-dir> [cache-dir]" >&2
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
CACHE_DIR="$(realpath -m "${3:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix}")"
mkdir -p logs "${OUT_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_done_chern_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

mapfile -t TASK_DIRS < <(python - "${SOURCE_ROOT}" <<'PY'
from __future__ import annotations
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
rows = []
for task in root.glob('tasks/task_*'):
    match = re.match(r'task_(\d+)_(xi[-0-9]+_V\d+meV)_(.+)_seed(\d+)$', task.name)
    if match is None:
        continue
    panel = match.group(2)
    panel_dir = task / panel
    if (task / 'paper_hf_config.json').is_file() and (panel_dir / 'hf_ground_state.npz').is_file() and (panel_dir / 'hf_convergence.json').is_file():
        rows.append((int(match.group(1)), task))
for _, task in sorted(rows):
    print(task)
PY
)

if (( ${#TASK_DIRS[@]} == 0 )); then
  echo "No completed tasks found under ${SOURCE_ROOT}/tasks" >&2
  exit 1
fi

echo "[job] source_root=${SOURCE_ROOT}"
echo "[job] out_dir=${OUT_DIR}"
echo "[job] cache_dir=${CACHE_DIR}"
echo "[job] completed_count=${#TASK_DIRS[@]}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

for task_dir in "${TASK_DIRS[@]}"; do
  echo "[chern-task] ${task_dir}"
  python scripts/compute_rlg_hbn_hf_chern.py \
    --source-dir "${task_dir}" \
    --cache-dir "${CACHE_DIR}" \
    --spin-index 0 \
    --eta-index 0 \
    --compare-paper-fig6
 done

python - "${OUT_DIR}" "${TASK_DIRS[@]}" <<'PY'
from __future__ import annotations
from pathlib import Path
import csv
import json
import re
import sys
out_dir = Path(sys.argv[1])
task_dirs = [Path(value) for value in sys.argv[2:]]
rows = []
for task in task_dirs:
    match = re.match(r'task_(\d+)_(xi[-0-9]+_V\d+meV)_(.+)_seed(\d+)$', task.name)
    summary = json.loads((task / 'hf_chern_summary.json').read_text(encoding='utf-8'))
    for panel in summary['panels']:
        occupied = panel['chern']['occupied_conduction']
        comparison = panel.get('paper_fig6_comparison') or {}
        conv = json.loads((task / panel['panel'] / 'hf_convergence.json').read_text(encoding='utf-8'))
        best = conv.get('best') or {}
        rows.append({
            'task_id': int(match.group(1)) if match else -1,
            'task': task.name,
            'panel': panel['panel'],
            'init_mode': match.group(3) if match else '',
            'seed': int(match.group(4)) if match else -1,
            'iterations': best.get('iterations'),
            'final_error': best.get('final_error'),
            'final_energy_mev': best.get('final_energy_mev'),
            'exit_reason': best.get('exit_reason'),
            'chern': occupied['chern_number'],
            'rounded': occupied['rounded_chern_number'],
            'abs_rounded': occupied['absolute_rounded_chern_number'],
            'paper_abs': comparison.get('paper_expected_abs_chern'),
            'matches_paper': comparison.get('matches_paper_abs_chern'),
            'max_flux_over_pi': occupied.get('max_abs_berry_flux_over_pi'),
            'chern_summary': str(task / 'hf_chern_summary.json'),
        })
rows.sort(key=lambda row: row['task_id'])
manifest = out_dir / 'completed_task_chern_manifest.tsv'
with manifest.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

md = out_dir / 'completed_task_chern_report.md'
lines = [
    '# Completed RLG/hBN Fig. 6 task Chern numbers',
    '',
    f'- Manifest: `{manifest}`',
    '',
    '| task | panel | init | seed | iter | err | C | rounded | |C| | paper | match |',
    '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
]
for row in rows:
    lines.append(
        f"| {row['task']} | {row['panel']} | {row['init_mode']} | {row['seed']} | {row['iterations']} | {row['final_error']} | {float(row['chern']):.8f} | {row['rounded']} | {row['abs_rounded']} | {row['paper_abs']} | {row['matches_paper']} |"
    )
md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"[done] manifest={manifest}")
print(f"[done] report={md}")
PY
