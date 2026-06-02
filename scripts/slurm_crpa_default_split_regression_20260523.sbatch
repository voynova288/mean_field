#!/bin/bash
# Focused regression checks for the production cRPA-HF split-mode default.

#SBATCH -J crpa_split_reg
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/crpa_default_split_regression_20260523_%j.out
#SBATCH -e logs/crpa_default_split_regression_20260523_%j.err

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_crpa_split_reg_${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

python -m py_compile \
  src/mean_field/crpa/hf_interface.py \
  src/mean_field/devtools/run_custom_b0_hf_case.py \
  tests/test_crpa_core.py

python -m pytest -q \
  tests/test_crpa_core.py::test_crpa_split_mode_default_is_active_cnp_fock_reference \
  tests/test_crpa_core.py::test_active_cnp_fock_reference_hartree_delta_split_mode \
  tests/test_crpa_core.py::test_remote_fock_plus_active_cnp_fock_reference_split_mode \
  tests/test_crpa_core.py::test_crpa_hartree_full_matrix_uses_input_shift_as_first_index \
  tests/test_crpa_core.py::test_crpa_hartree_delta_fock_projector_components_split_densities \
  tests/test_crpa_core.py::test_remote_reference_eq19_direct_terms_match_half_reference_delta \
  tests/test_crpa_core.py::test_crpa_split_with_identity_screening_matches_bare_delta_hf
