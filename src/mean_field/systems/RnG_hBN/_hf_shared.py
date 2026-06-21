from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
import os

import numpy as np

from ...core.lattice import KPath
from ...core.hf import (
    average_reference_density as core_average_reference_density,
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    HartreeFockStepResult,
    apply_random_projector_rotation,
    random_unitary_from_hermitian,
    ComponentGroup,
    component_group_indices,
    compute_hf_energy,
    compute_oda_parameter,
    ProjectedWavefunctionBasis,
    calculate_projected_overlap_between,
    compute_density_overlap_trace_from_diagonal,
    find_chemical_potential,
    occupied_state_mask,
    run_hartree_fock_problem,
    shift_wavefunction_grid,
)
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, valence_band_count
from .interaction import RLGhBNInteractionParams, VALID_INTERACTION_SCHEMES, layer_coulomb_matrix_mev_nm2
from .lattice import RLGhBNLattice, build_moire_k_grid
from .model import RLGhBNModel
from .screening import (
    ScreenedInterlayerPotentialResult,
    moire_cell_area_nm2,
    solve_screened_interlayer_potential,
    solve_screened_interlayer_potential_grid,
)

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - exercised on systems without numba.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False
else:
    _NUMBA_AVAILABLE = True


VALLEY_SEQUENCE = (1, -1)
RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION = "centered_cell_reciprocal_relabel_pad1_v2"
RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING = 1
RLG_HBN_FORM_FACTOR_CONVENTION_VERSION = "physical_q_plus_g_valley_signed_raw_shift_v2"


def rlg_hbn_layer_component_groups(layer_count: int) -> tuple[ComponentGroup, ...]:
    """Return RnG/hBN layer groups in the local sublattice-resolved basis.

    The core HF layer only knows named local-component subsets.  RnG/hBN owns
    the physical convention that each layer contributes the two local sublattice
    components ``[2*layer, 2*layer+1]`` inside every reciprocal-grid cell.
    """

    resolved_layer_count = int(layer_count)
    if resolved_layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    return tuple(
        ComponentGroup(f"layer_{layer}", np.asarray([2 * layer, 2 * layer + 1], dtype=int))
        for layer in range(resolved_layer_count)
    )


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _rlg_hbn_require_numba() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA", default=False)


def _rlg_hbn_use_numba() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_USE_NUMBA", default=True)


def _rlg_hbn_zero_literal_q0_fock() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False)


def _maybe_zero_literal_q0_fock_kernel(shift: tuple[int, int], fock_kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(fock_kernel)
    if not _rlg_hbn_zero_literal_q0_fock() or (int(shift[0]), int(shift[1])) != (0, 0):
        return kernel
    if kernel.ndim != 4:
        raise ValueError(f"Expected fock kernel shape (nk_target, nk_source, layer, layer), got {kernel.shape}")
    adjusted = np.array(kernel, copy=True)
    n_diag = min(adjusted.shape[0], adjusted.shape[1])
    for ik in range(n_diag):
        adjusted[ik, ik, :, :] = 0.0
    return adjusted


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _contract_layer_fock_term_numba_kernel(
        left_overlap: np.ndarray,
        density_delta: np.ndarray,
        coeff_matrix: np.ndarray,
        right_overlap: np.ndarray,
    ) -> np.ndarray:
        nt_target = left_overlap.shape[0]
        nk_target = left_overlap.shape[1]
        nt_source = left_overlap.shape[2]
        nk_source = left_overlap.shape[3]
        out = np.empty((nt_target, nt_target, nk_target), dtype=np.complex128)
        output_pairs = nt_target * nt_target
        total_terms = nk_target * output_pairs
        for linear_index in prange(total_terms):
            ik_target = linear_index // output_pairs
            pair_index = linear_index - ik_target * output_pairs
            a = pair_index // nt_target
            b = pair_index - a * nt_target
            total = 0.0 + 0.0j
            for ik_source in range(nk_source):
                coeff = coeff_matrix[ik_target, ik_source]
                if coeff == 0.0:
                    continue
                source_total = 0.0 + 0.0j
                for d in range(nt_source):
                    left_density = 0.0 + 0.0j
                    for c in range(nt_source):
                        left_density += (
                            left_overlap[a, ik_target, c, ik_source]
                            * density_delta[d, c, ik_source]
                        )
                    source_total += left_density * np.conj(right_overlap[b, ik_target, d, ik_source])
                total += coeff * source_total
            out[a, b, ik_target] = total
        return out

else:

    def _contract_layer_fock_term_numba_kernel(
        left_overlap: np.ndarray,
        density_delta: np.ndarray,
        coeff_matrix: np.ndarray,
        right_overlap: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available for RnG/hBN Fock contraction")

__all__ = [name for name in globals() if not name.startswith('__')]
