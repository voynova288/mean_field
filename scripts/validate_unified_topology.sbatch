#!/bin/bash
#SBATCH -J topo_framework_val
#SBATCH -p test
#SBATCH --account=hmt03
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o logs/topo_framework_val_%j.out
#SBATCH -e logs/topo_framework_val_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/analysis/topology" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
fi
cd "${REPO_ROOT}"
mkdir -p logs results/topology_framework_validation_20260528

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

printf '[job] host=%s\n' "$(hostname)"
printf '[job] date=%s\n' "$(date --iso-8601=seconds)"
printf '[job] repo=%s\n' "${REPO_ROOT}"
printf '[job] slurm_job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf '[job] partition=%s cpus=%s\n' "${SLURM_JOB_PARTITION:-}" "${SLURM_CPUS_PER_TASK:-}"
printf '[job] python=%s\n' "$(command -v python)"
printf '[job] PYTHONPATH=%s\n' "${PYTHONPATH}"

python -m compileall -q src tests

python -m pytest \
  tests/test_analysis_topology.py \
  tests/test_system_topology_wrappers.py \
  tests/test_tmbg_topology.py \
  tests/test_htg_mean_field_adapter.py \
  tests/test_tmbg_plot.py \
  tests/test_rlg_hbn_model.py \
  -q

python -m analysis.topology.validate_existing_results \
  --root "${REPO_ROOT}" \
  --output-dir "${REPO_ROOT}/results/topology_framework_validation_20260528"

printf '[job] completed=%s\n' "$(date --iso-8601=seconds)"
