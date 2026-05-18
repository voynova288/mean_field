#!/bin/bash
# Cached one-allocation RLG/hBN Fig. 6 HF band reproduction.
#
# Submit from login002:
#   cd /data/home/ziyuzhu/Mean_Field
#   sbatch scripts/run_fig6_hf_bands_cached.slurm
#
# Main log:
#   tail -f logs/rlg_fig6_cached_<jobid>.out
#
# Panel progress:
#   tail -f results/RnG_hBN/fig6_hf_bands_<jobid>_*/xi0_V064meV/hf_progress_events.jsonl
#   tail -f results/RnG_hBN/fig6_hf_bands_<jobid>_*/xi1_V064meV/hf_progress_events.jsonl
#
# If HF is interrupted, do not delete OUT. Re-run run_rlg_hbn_paper_hf with the
# same --output-dir and --cache-dir. Completed run archives are skipped and
# latest checkpoints are resumed. Screening, basis, and overlap caches should hit.
#
# If plotting is interrupted, re-run plot_rlg_hbn_paper_hf_bands with the same
# --source-dir and --cache-dir. The path_bands cache should make the retry fast.

#SBATCH --job-name=rlg_fig6_cached
#SBATCH --output=logs/rlg_fig6_cached_%j.out
#SBATCH --error=logs/rlg_fig6_cached_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=long
#SBATCH --time=7-00:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/mean_field" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs
mkdir -p results/RnG_hBN

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_rlg_fig6_cached_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

# Force numba; fail fast if unavailable.
export MEAN_FIELD_RLG_HBN_USE_NUMBA=1
export MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# Use the full requested CPU allocation and avoid hidden BLAS oversubscription.
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/results/RnG_hBN/fig6_hf_bands_${SLURM_JOB_ID:-manual}_${RUN_STAMP}"
CACHE="$PWD/results/RnG_hBN/cache_fig6"

mkdir -p "$OUT"
mkdir -p "$CACHE"

echo "[info] host=$(hostname)"
echo "[info] out=$OUT"
echo "[info] cache=$CACHE"
echo "[info] python=$(which python)"
echo "[info] slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[info] numba_threads=${NUMBA_NUM_THREADS}"
python - <<'PY'
import sys
print("[info] sys.executable=", sys.executable)
import numba
print("[info] numba=", numba.__version__)
import mean_field
print("[info] mean_field import OK")
PY

echo "[stage] run tests"
python -m pytest \
  tests/test_rlg_hbn_fig6_projection.py \
  tests/test_rlg_hbn_cache_roundtrip.py \
  -q

echo "[stage] prereq screening checkpoint"
python -m mean_field.devtools.validate_rlg_hbn_fig6_prereqs \
  --cache-dir "$CACHE" \
  --cache-policy reuse \
  --screening-solver grid \
  --screening-u-min-mev -100 \
  --screening-u-max-mev 200 \
  --screening-u-grid-points 121 \
  --tolerance-mev 3.0 | tee "$OUT/prereq_screening.log"

echo "[stage] dry run"
python -m mean_field.devtools.run_rlg_hbn_paper_hf \
  --paper-target fig6 \
  --output-dir "$OUT" \
  --cache-dir "$CACHE" \
  --cache-policy reuse \
  --screening-solver grid \
  --screening-u-min-mev -100 \
  --screening-u-max-mev 200 \
  --screening-u-grid-points 121 \
  --init-modes flavor,bm,perturbed,random \
  --seeds 1,2,3,4 \
  --max-iter 120 \
  --precision 1e-6 \
  --beta 1.0 \
  --oda-stall-threshold 1e-3 \
  --dry-run | tee "$OUT/dry_run.log"

echo "[stage] run HF source states"
python -m mean_field.devtools.run_rlg_hbn_paper_hf \
  --paper-target fig6 \
  --output-dir "$OUT" \
  --cache-dir "$CACHE" \
  --cache-policy reuse \
  --screening-solver grid \
  --screening-u-min-mev -100 \
  --screening-u-max-mev 200 \
  --screening-u-grid-points 121 \
  --init-modes flavor,bm,perturbed,random \
  --seeds 1,2,3,4 \
  --max-iter 120 \
  --precision 1e-6 \
  --beta 1.0 \
  --oda-stall-threshold 1e-3 | tee "$OUT/run_hf.log"

echo "[stage] plot path bands"
python -m mean_field.devtools.plot_rlg_hbn_paper_hf_bands \
  --source-dir "$OUT" \
  --paper-target fig6 \
  --cache-dir "$CACHE" \
  --cache-policy reuse \
  --points-per-segment 48 \
  --chunk-size 8 \
  --dpi 180 | tee "$OUT/plot_bands.log"

echo "[stage] verify outputs"
test -f "$OUT/paper_fig6_hf_bands.png"
test -f "$OUT/paper_fig6_hf_bands.pdf"
test -f "$OUT/hf_band_plot_summary.json"
test -f "$OUT/cache_manifest.json"
test -f "$OUT/xi0_V064meV/hf_ground_state.npz"
test -f "$OUT/xi1_V064meV/hf_ground_state.npz"
test -f "$OUT/xi0_V064meV/hf_bands_path.npz"
test -f "$OUT/xi1_V064meV/hf_bands_path.npz"
test -f "$OUT/xi0_V064meV/hf_convergence.json"
test -f "$OUT/xi1_V064meV/hf_convergence.json"

echo "[done] Fig. 6 band outputs:"
echo "$OUT/paper_fig6_hf_bands.png"
echo "$OUT/paper_fig6_hf_bands.pdf"
echo "[done] cache:"
echo "$CACHE"
