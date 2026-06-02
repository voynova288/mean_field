#!/bin/bash
#SBATCH -J tmbg_pol_s1a_scan
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/tmbg_polshyn_s1a_delta_scan_%j.out
#SBATCH -e logs/tmbg_polshyn_s1a_delta_scan_%j.err

set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tmbg_polshyn_s1a_scan_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

ROOT="${ROOT:-${REPO_ROOT}/results/TMBG_Polshyn2021_figS1/s1a_delta_scan_${SLURM_JOB_ID:-manual}}"
export ROOT
mkdir -p "${ROOT}"

echo "[job] hostname=$(hostname) job_id=${SLURM_JOB_ID:-manual} root=${ROOT} blg_stacking=${BLG_STACKING:-BA}"
for DELTA in ${DELTA_LIST:--0.080 -0.060 -0.040 -0.033 -0.020 0.000 0.020}; do
  TAG="delta_$(printf '%+04.0f' "$(python - <<PY
print(float('${DELTA}')*1000)
PY
)")mev"
  OUT_DIR="${ROOT}/${TAG}"
  echo "[scan] DELTA=${DELTA} OUT_DIR=${OUT_DIR}"
  python scripts/mean_field_tools.py run_tmbg_polshyn_figs1_abc \
    --output-dir "${OUT_DIR}" \
    --theta-deg "${THETA_DEG:-1.29}" \
    --parameter-set "${PARAMETER_SET:-polshyn2020}" \
    --interlayer-potential "${DELTA}" \
    --blg-stacking "${BLG_STACKING:-BA}" \
    --n-shells "${N_SHELLS:-5}" \
    --auto-target-role "${AUTO_TARGET_ROLE:-upper-flat}" \
    --s1a-points-per-segment "${S1A_POINTS_PER_SEGMENT:-80}" \
    --s1a-bands-per-side "${S1A_BANDS_PER_SIDE:-5}" \
    --panels a
 done

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import os
root = Path(os.environ['ROOT'])
items = []
for summary in sorted(root.glob('delta_*mev/polshyn_figS1_summary.json')):
    png = summary.parent / 'polshyn_figS1a_noninteracting_bands.png'
    if png.exists():
        items.append((summary.parent.name, png))
if items:
    thumbs = []
    for label, path in items:
        img = Image.open(path).convert('RGB')
        img.thumbnail((390, 270))
        canvas = Image.new('RGB', (400, 300), 'white')
        ImageDraw.Draw(canvas).text((8, 8), label, fill=(0, 0, 0))
        canvas.paste(img, (5, 28))
        thumbs.append(canvas)
    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * 400, rows * 300), 'white')
    for idx, img in enumerate(thumbs):
        sheet.paste(img, ((idx % cols) * 400, (idx // cols) * 300))
    out = root / 's1a_delta_scan_contact.png'
    sheet.save(out)
    print(f'[scan] contact_sheet={out}', flush=True)
PY
