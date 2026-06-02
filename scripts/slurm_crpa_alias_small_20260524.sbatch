#!/bin/bash
# Small HF-compatible cRPA no-alias gate for the Q+wrap form-factor convention.

#SBATCH -J crpa_alias_small
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o logs/crpa_alias_small_20260524_%j.out
#SBATCH -e logs/crpa_alias_small_20260524_%j.err

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_alias_small_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

LK="${LK:-4}"
LG="${LG:-7}"
Q_LG="${Q_LG:-5}"
OUT="${OUT:-${REPO_ROOT}/results/TBG_HF_cRPA/crpa_alias_gate_20260524/hf_periodic_lk${LK}_lg${LG}_q${Q_LG}}"

echo "[job] hostname=$(hostname)"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "[job] output=${OUT}"
echo "[job] run=hf_periodic no-alias small cRPA lk=${LK} lg=${LG} q_lg=${Q_LG}"

/data/home/ziyuzhu/miniconda3/bin/python -m mean_field.devtools.run_tbg_crpa_epsilon \
  --theta-deg 1.05 \
  --lk "${LK}" \
  --lg "${LG}" \
  --q-lg "${Q_LG}" \
  --epsilon-bn 4.0 \
  --ds-angstrom 400.0 \
  --eta-mev 1.0 \
  --hf-compatible \
  --output-dir "${OUT}"

echo "[job] diagnostic_summary=${OUT}/crpa_epsilon_diagnostics_summary.md"
sed -n '1,90p' "${OUT}/crpa_epsilon_diagnostics_summary.md"
echo "[job] artifacts"
ls -lh "${OUT}/epsilon_vs_q.pdf" "${OUT}/epsilon_by_Q_shell.png" "${OUT}/crpa_epsilon_radial_bin_stats_0_1p2.csv"
