from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import HamiltonianGaugeData, hamiltonian_gauge_data
from analysis.shift_current import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    PairTransitionKernel,
    ShiftCurrentComponent,
    ShiftCurrentConvention,
    ShiftCurrentTensors,
    component_from_any,
    component_kernel_from_gauge_pair,
    precompute_shift_current_tensors,
)

from .hamiltonian import build_hamiltonian, build_hamiltonian_d2hdk2, build_hamiltonian_dhdk
from .lattice import TDBGLattice
from .model import TDBGModel
from .params import TDBGParameters

JOYA_GAMMA_CENTERED_FRAC_SHIFT: tuple[float, float] = (-1.0 / 6.0, -1.0 / 6.0)


def joya_gamma_centered_frac_shift() -> tuple[float, float]:
    """Fractional mBZ-cell shift used for Joya-2025 Gamma-centered maps.

    In the local TDBG convention ``k=0`` is the moire ``kappa_-`` point and
    ``Gamma_M = (g_m1 + g_m2)/3``.  Shifting a Monkhorst-Pack parallelogram by
    ``(-1/6, -1/6)`` recenters the sampled cell on ``Gamma_M`` without changing
    its area.
    """

    return JOYA_GAMMA_CENTERED_FRAC_SHIFT


def joya_gamma_centered_k_grid(
    lattice: TDBGLattice,
    mesh_size: int,
    *,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gamma-centered Joya-2025 mBZ sampling grid.

    This intentionally keeps the shifted fractional coordinates unwrapped.
    Plane-wave continuum Hamiltonians are not necessarily represented in a
    strictly periodic gauge across the parallelogram boundary, so wrapping
    ``-1/6`` to ``5/6`` is not an innocuous bookkeeping change for diagnostics.
    """

    n = int(mesh_size)
    if n <= 0:
        raise ValueError(f"Expected a positive mesh_size, got {mesh_size}")
    shift_1, shift_2 = JOYA_GAMMA_CENTERED_FRAC_SHIFT
    if endpoint:
        frac_1 = np.linspace(0.0, 1.0, n, dtype=float) + shift_1
        frac_2 = np.linspace(0.0, 1.0, n, dtype=float) + shift_2
    else:
        frac_1 = np.arange(n, dtype=float) / float(n) + shift_1
        frac_2 = np.arange(n, dtype=float) / float(n) + shift_2
    frac_i, frac_j = np.meshgrid(frac_1, frac_2, indexing="ij")
    frac_grid = np.stack([frac_i, frac_j], axis=-1)
    kvec = frac_i * complex(lattice.g_m1) + frac_j * complex(lattice.g_m2)
    return frac_grid, np.asarray(kvec, dtype=np.complex128)


def mirror_x_tensor_component_sign(component: ShiftCurrentComponent | tuple[int, int, int] | str) -> int:
    """Sign acquired by a rank-3 tensor component under ``x -> -x``."""

    comp = component_from_any(component)
    return -1 if sum(1 for axis in comp.as_tuple if int(axis) == 0) % 2 else 1


def valley_mirror_x_tensor_component_sign(
    component: ShiftCurrentComponent | tuple[int, int, int] | str,
    *,
    valley: int,
    mirror_valley: int = -1,
) -> int:
    """Joya finite-cutoff valley-axis sign for summing local K± tensors.

    The local K- coordinate convention is related to the common physical axes by
    an x mirror.  K+ components are returned unchanged; K- components get the
    rank-3 mirror sign.  This is a system-convention helper, not a generic
    response formula.
    """

    return mirror_x_tensor_component_sign(component) if int(valley) == int(mirror_valley) else 1


def transform_valley_component_to_physical_axes(
    value: np.ndarray | float | complex,
    component: ShiftCurrentComponent | tuple[int, int, int] | str,
    *,
    valley: int,
    mirror_valley: int = -1,
) -> np.ndarray | float | complex:
    """Apply the Joya K- mirror-x tensor sign to a local component value."""

    sign = valley_mirror_x_tensor_component_sign(component, valley=valley, mirror_valley=mirror_valley)
    return value * sign


@dataclass(frozen=True)
class TDBGShiftCurrentPoint:
    """TDBG one-k point data prepared for generic shift-current APIs."""

    k_tilde: complex
    valley: int
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    dhdk: np.ndarray
    d2hdk: np.ndarray
    gauge_data: HamiltonianGaugeData


def finite_difference_dhdk(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int | None = None,
    step_nm_inv: float = 1.0e-6,
) -> np.ndarray:
    """Return finite-difference ``dH/d(kx,ky)`` for TDBG in eV nm.

    ``k_tilde`` and the moire reciprocal vectors are represented as complex
    Cartesian momenta in ``nm^-1``.  The returned array has shape
    ``(2, matrix_dim, matrix_dim)`` and is suitable for
    ``analysis.shift_current`` / ``response_derivative_gauge``.
    """

    step = float(step_nm_inv)
    if step <= 0.0:
        raise ValueError(f"step_nm_inv must be positive, got {step_nm_inv}")
    resolved_valley = int(params.valley if valley is None else valley)
    k0 = complex(k_tilde)
    hx_p = build_hamiltonian(k0 + step, lattice, params, valley=resolved_valley)
    hx_m = build_hamiltonian(k0 - step, lattice, params, valley=resolved_valley)
    hy_p = build_hamiltonian(k0 + 1.0j * step, lattice, params, valley=resolved_valley)
    hy_m = build_hamiltonian(k0 - 1.0j * step, lattice, params, valley=resolved_valley)
    return np.stack(((hx_p - hx_m) / (2.0 * step), (hy_p - hy_m) / (2.0 * step)), axis=0)


def zero_second_derivative(lattice: TDBGLattice) -> np.ndarray:
    """Return the TDBG continuum-model ``d²H/dk²`` tensor (zero)."""

    return build_hamiltonian_d2hdk2(lattice)


def shift_current_point_data(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int | None = None,
    fd_step_nm_inv: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-10,
) -> TDBGShiftCurrentPoint:
    """Diagonalize TDBG and prepare Hamiltonian-gauge response data.

    The response path uses analytic lab-frame ``dH/dk`` by default.  The
    ``fd_step_nm_inv`` argument is retained for backward-compatible callers of
    this adapter; use :func:`finite_difference_dhdk` directly for derivative
    validation.
    """

    _ = fd_step_nm_inv
    resolved_valley = int(params.valley if valley is None else valley)
    h0 = build_hamiltonian(complex(k_tilde), lattice, params, valley=resolved_valley)
    evals, evecs = eigh(h0)
    dh = build_hamiltonian_dhdk(lattice, params, valley=resolved_valley)
    d2h = build_hamiltonian_d2hdk2(lattice)
    gauge = hamiltonian_gauge_data(evals, evecs, dh, denominator_cutoff=float(denominator_cutoff_ev), d2hdk=d2h)
    return TDBGShiftCurrentPoint(
        k_tilde=complex(k_tilde),
        valley=resolved_valley,
        energies_ev=np.asarray(evals, dtype=float),
        eigenvectors=np.asarray(evecs, dtype=np.complex128),
        dhdk=dh,
        d2hdk=d2h,
        gauge_data=gauge,
    )


def shift_current_tensors_at_k(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int | None = None,
    fd_step_nm_inv: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-10,
    principal_value_eta_ev: float | None = None,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
) -> ShiftCurrentTensors:
    """Return full generic shift-current tensors for a tiny TDBG k point.

    This is intended for tests/small diagnostics.  Large cutoffs should use the
    selected-pair helper below to avoid constructing the full generalized-
    derivative tensor.
    """

    point = shift_current_point_data(
        k_tilde,
        lattice,
        params,
        valley=valley,
        fd_step_nm_inv=fd_step_nm_inv,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    return precompute_shift_current_tensors(
        point.energies_ev,
        point.eigenvectors,
        point.dhdk,
        mu_ev=mu_ev,
        temperature_k=temperature_k,
        denominator_cutoff_ev=denominator_cutoff_ev,
        d2hdk=point.d2hdk,
        principal_value_eta_ev=principal_value_eta_ev,
    )


def component_kernel_at_k(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | tuple[int, int, int] | str,
    *,
    valley: int | None = None,
    fd_step_nm_inv: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-10,
    principal_value_eta_ev: float | None = None,
    convention: ShiftCurrentConvention = JOYA_EQ7_GEOMETRIC_CONVENTION,
) -> PairTransitionKernel:
    """Return a selected-pair TDBG kernel using the generic shift-current API."""

    point = shift_current_point_data(
        k_tilde,
        lattice,
        params,
        valley=valley,
        fd_step_nm_inv=fd_step_nm_inv,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    return component_kernel_from_gauge_pair(
        point.gauge_data.velocity_h,
        point.gauge_data.energies,
        point.gauge_data.berry_connection,
        int(initial_band),
        int(final_band),
        component,
        denominator_cutoff_ev=denominator_cutoff_ev,
        second_velocity_h=point.gauge_data.second_velocity_h,
        principal_value_eta_ev=principal_value_eta_ev,
        convention=convention,
    )


def model_shift_current_point_data(
    model: TDBGModel,
    k_tilde: complex,
    *,
    valley: int | None = None,
    fd_step_nm_inv: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-10,
) -> TDBGShiftCurrentPoint:
    return shift_current_point_data(
        k_tilde,
        model.lattice,
        model.params,
        valley=valley,
        fd_step_nm_inv=fd_step_nm_inv,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
