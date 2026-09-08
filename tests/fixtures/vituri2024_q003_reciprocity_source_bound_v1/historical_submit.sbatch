#!/bin/bash
#SBATCH --job-name=vituri_q003_reciprocity_v24
#SBATCH --account=hmt03
#SBATCH --partition=regular256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/data/home/ziyuzhu/.runs/Mean_Field_6dcc724_vituri_fig2_q003_whole_inventory_reciprocity_comparison_v24_20260907/logs/slurm_%j.out
#SBATCH --error=/data/home/ziyuzhu/.runs/Mean_Field_6dcc724_vituri_fig2_q003_whole_inventory_reciprocity_comparison_v24_20260907/logs/slurm_%j.err

set -euo pipefail
ROOT=/data/home/ziyuzhu/.runs/Mean_Field_6dcc724_vituri_fig2_q003_whole_inventory_reciprocity_comparison_v24_20260907
PYTHON=/data/home/ziyuzhu/miniconda3/bin/python
FAILURE_STAGE=wrapper_or_preflight_failure
publish_wrapper_failure() {
  code=$?
  trap - EXIT
  if [[ $code -ne 0 ]]; then "$PYTHON" -S "$ROOT/bin/publish_wrapper_failure.py" "$ROOT" "$FAILURE_STAGE" "$code" || true; fi
  exit "$code"
}
trap publish_wrapper_failure EXIT
: "${EXPECTED_PREPARATION_SHA256:?missing detached preparation SHA-256}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "hmt03" ]]
[[ "${SLURM_JOB_PARTITION:-}" == "regular256" ]]
[[ "${SLURM_NNODES:-}" == "1" ]]
[[ "${SLURM_NTASKS:-}" == "1" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == "64" ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^64(\(x1\))?$ ]]
export RUNROOT="$ROOT"
export PYTHONPATH="$ROOT/source/src:/data/home/ziyuzhu/miniconda3/lib/python3.13/site-packages"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C TZ=UTC MALLOC_ARENA_MAX=2
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK} VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export SLURM_EXPORT_ENV=ALL
cd "$ROOT"
printf 'host=%s\ndate=%s\njob=%s\npartition=%s\ncpus=%s\nsource=%s\n' "$(/usr/bin/hostname)" "$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURM_CPUS_PER_TASK" "$(<provenance/SOURCE_COMMIT.txt)"
/usr/bin/taskset -pc $$
"$PYTHON" -S "$ROOT/bin/run_qualifier.py" --launch-preflight
FAILURE_STAGE=srun_failure
/usr/bin/srun --export=ALL --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 "$PYTHON" -S "$ROOT/bin/run_qualifier.py"
FAILURE_STAGE=finalizer_failure
"$PYTHON" -S "$ROOT/bin/finalize_after_srun.py"
