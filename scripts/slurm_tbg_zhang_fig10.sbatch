#!/bin/bash
#SBATCH -J tbg_zfig10
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH -o logs/tbg_zhang_fig10_%j.out
#SBATCH -e logs/tbg_zhang_fig10_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tbg_zhang_fig10_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/TBG_Zhang2022_fig10/zhang_fig10_lg11_k12_eps7_ds400_hf_random_seed1_primitivev0_${SLURM_JOB_ID:-manual}}"

cat <<EOF
[job] command=python scripts/mean_field_tools.py run_tbg_zhang_fig10 --output-dir ${OUT_DIR}
[job] hostname=$(hostname)
[job] date=$(date -Is)
[job] repo=${REPO_ROOT}
[job] job_id=${SLURM_JOB_ID:-manual}
[job] partition=${SLURM_JOB_PARTITION:-}
[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}
[job] mem_per_node=${SLURM_MEM_PER_NODE:-}
[job] OMP_NUM_THREADS=${OMP_NUM_THREADS}
[job] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}
[job] MKL_NUM_THREADS=${MKL_NUM_THREADS}
[job] output_dir=${OUT_DIR}
EOF

python scripts/mean_field_tools.py run_tbg_zhang_fig10 \
  --output-dir "${OUT_DIR}" \
  --theta-deg 1.05 \
  --vf 2135.4 \
  --w0 79.7 \
  --w1 97.4 \
  --epsilon-r 7 \
  --ds-angstrom 400 \
  --coulomb-area primitive \
  --lg 11 \
  --kmesh 12 \
  --interaction-lg 11 \
  --path-points-per-segment 40 \
  --max-iter 80 \
  --mixing 0.5 \
  --precision 1e-6 \
  --scf-interaction hartree_fock \
  --init random \
  --seed 1 \
  --hartree-density-reference occupation \
  --energy-reference max_selected \
  --plot-sector partial \
  --ylim-ev=-0.4,0.1
