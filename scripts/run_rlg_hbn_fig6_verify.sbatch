#!/bin/bash
# Lightweight Slurm wrapper for verifying cached RLG/hBN Fig. 6 output files.

#SBATCH -J rlg_fig6_verify
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/rlg_fig6_verify_%j.out
#SBATCH -e logs/rlg_fig6_verify_%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch scripts/run_rlg_hbn_fig6_verify.sbatch <output-dir>" >&2
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

mkdir -p logs

echo "[job] verify output_dir=$1"
echo "[job] hostname=$(hostname)"
echo "[job] partition=${SLURM_JOB_PARTITION:-}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

bash scripts/verify_rlg_hbn_fig6_outputs.sh "$1"
