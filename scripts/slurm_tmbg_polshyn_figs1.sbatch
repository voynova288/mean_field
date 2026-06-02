#!/bin/bash
#SBATCH -J tmbg_pol_s1abc
#SBATCH --account=hmt03
#SBATCH -p regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o logs/tmbg_polshyn_figs1_%j.out
#SBATCH -e logs/tmbg_polshyn_figs1_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tmbg_polshyn_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/TMBG_Polshyn2021_figS1/polshyn_figS1abc_theta129_D40meV_k9_g2_seed1_${SLURM_JOB_ID:-manual}}"
SKIP_RECONSTRUCTED_LINE="${SKIP_RECONSTRUCTED_LINE:-1}"

echo "[job] hostname=$(hostname) job_id=${SLURM_JOB_ID:-manual} cpus=${SLURM_CPUS_PER_TASK:-} mem=${SLURM_MEM_PER_NODE:-} output_dir=${OUT_DIR}"
echo "[job] date=$(date --iso-8601=seconds) repo=${REPO_ROOT}"

python scripts/mean_field_tools.py run_tmbg_polshyn_figs1_abc \
  --output-dir "${OUT_DIR}" \
  --theta-deg "${THETA_DEG:-1.29}" \
  --parameter-set "${PARAMETER_SET:-polshyn2020}" \
  --interlayer-potential "${INTERLAYER_POTENTIAL:--0.033}" \
  --staggered-potential "${STAGGERED_POTENTIAL:-0.0}" \
  --blg-stacking "${BLG_STACKING:-BA}" \
  --n-shells "${N_SHELLS:-5}" \
  --target-band-index "${TARGET_BAND_INDEX:-auto}" \
  --auto-target-role "${AUTO_TARGET_ROLE:-upper-flat}" \
  --s1a-points-per-segment "${S1A_POINTS_PER_SEGMENT:-80}" \
  --s1a-bands-per-side "${S1A_BANDS_PER_SIDE:-5}" \
  --panels "${PANELS:-abc}" \
  --kmesh "${KMESH:-9}" \
  --line-points "${LINE_POINTS:-49}" \
  --g-shells "${G_SHELLS:-2}" \
  --epsilon-r "${EPSILON_R:-20}" \
  --gate-distance-nm "${GATE_DISTANCE_NM:-120}" \
  --coulomb-area "${COULOMB_AREA:-supercell}" \
  --hf-engine "${HF_ENGINE:-wang}" \
  --hartree-scale "${HARTREE_SCALE:-1.0}" \
  --fock-scale "${FOCK_SCALE:-1.0}" \
  ${ZERO_HARTREE_Q0:+--zero-hartree-q0} \
  --max-iter "${MAX_ITER:-80}" \
  --mixing "${MIXING:-0.5}" \
  --oda-stall-threshold "${ODA_STALL_THRESHOLD:-1e-4}" \
  --h0-subtraction "${H0_SUBTRACTION:-active-reference}" \
  --p0-reference "${P0_REFERENCE:-decoupled-layers}" \
  ${SKIP_RECONSTRUCTED_LINE:+--skip-reconstructed-line} \
  --precision "${PRECISION:-1e-6}" \
  --init "${INIT_MODE:-cdw}" \
  --seed "${SEED:-1}"
