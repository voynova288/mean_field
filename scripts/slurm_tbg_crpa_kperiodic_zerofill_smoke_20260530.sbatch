#!/bin/bash
# Small cRPA dielectric gate for the 2026-05-30 finite-G form-factor fix.

#SBATCH -J crpa_kzf_smoke
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/crpa_kzf_smoke_%j.out
#SBATCH -e logs/crpa_kzf_smoke_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_kzf_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

LK="${LK:-6}"
LG="${LG:-5}"
Q_LG="${Q_LG:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_kperiodic_zerofill_smoke_20260530/lk${LK}_lg${LG}_q${Q_LG}}"

echo "[job] command=run_tbg_crpa_epsilon lk=${LK} lg=${LG} q_lg=${Q_LG} output=${OUTPUT_DIR}"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

python scripts/mean_field_tools.py run_tbg_crpa_epsilon \
  --theta-deg 1.05 \
  --vf 2135.4 \
  --w0 79.7 \
  --w1 97.4 \
  --lk "${LK}" \
  --lg "${LG}" \
  --q-lg "${Q_LG}" \
  --epsilon-bn 4.0 \
  --ds-angstrom 400.0 \
  --eta-mev 1.0 \
  --hf-compatible \
  --form-factor-mode k_periodic_zero_fill \
  --output-dir "${OUTPUT_DIR}"

