#!/bin/bash
# Focused regressions for the active-CNP Fock-reference Hartree[D]/Fock[P] diagnostic.

#SBATCH -J crpa_acnp_hd_tests
#SBATCH --account=hmt03
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o logs/crpa_acnp_hd_tests_%j.out
#SBATCH -e logs/crpa_acnp_hd_tests_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/data/home/ziyuzhu/Mean_Field}"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[job] active-CNP Hartree[D]/Fock[P] regression"
echo "[job] hostname=$(hostname)"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

/data/home/ziyuzhu/miniconda3/bin/python -m pytest -q \
  tests/test_crpa_core.py::test_crpa_hartree_delta_fock_projector_components_split_densities \
  tests/test_crpa_core.py::test_active_cnp_fock_reference_hartree_delta_split_mode \
  tests/test_crpa_core.py::test_crpa_target_components_from_densities_split_hartree_fock \
  tests/test_crpa_core.py::test_crpa_hartree_full_matrix_uses_input_shift_as_first_index \
  tests/test_crpa_core.py::test_crpa_path_hartree_full_matrix_uses_input_shift_as_first_index
