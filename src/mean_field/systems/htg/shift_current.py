"""hTG shift-current system wrapper around the common response API.

This module preserves the legacy hTG response surface while delegating response
formula work to ``analysis.shift_current`` and gauge-safe derivatives to
``analysis.response_derivative_gauge``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair as _gauge_generalized_derivative_pair,
    berry_connection_pair as _gauge_berry_connection_pair,
)
from analysis.shift_current import (
    Component,
    HTG_LEGACY_CONVENTION,
    KB_EV_PER_K,
    SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    ShiftCurrentTensors as _GenericShiftCurrentTensors,
    add_transitions_to_integral as _generic_add_transitions_to_integral,
    axis_index as _axis_index,
    berry_connection_matrix,
    component_label as _component_label,
    component_transition_weight as _generic_component_transition_weight,
    component_transition_weight_from_gauge_pair,
    fermi_occupation,
    generalized_derivative_from_velocity,
    lorentzian_delta,
    parse_component as _parse_component,
    positive_transition_terms as _generic_positive_transition_terms,
    precompute_shift_current_tensors,
    second_derivative_matrices,
    spectra_from_transition_table as _generic_spectra_from_transition_table,
    velocity_matrices,
)


Axis = int
AXIS_LABELS = {"x": 0, "y": 1, 0: 0, 1: 1}


def axis_index(axis: str | int) -> int:
    value = _axis_index(axis)
    if value not in (0, 1):
        raise ValueError(f"Unsupported hTG axis {axis!r}; expected x/y or 0/1")
    return int(value)


def parse_component(text: str) -> Component:
    """Parse labels such as 'x;yy' or 'yyy'."""

    comp = _parse_component(text).as_tuple
    if any(axis not in (0, 1) for axis in comp):
        raise ValueError(f"hTG shift-current components are two-dimensional, got {text!r}")
    return comp


def component_label(component: Component) -> str:
    return _component_label(component, style="semicolon")


@dataclass(frozen=True)
class GeneralizedDerivativeResult:
    values: np.ndarray  # values[deriv_a, conn_b, n, m] = r_{nm;a}^b
    skipped_small_denominators: int


def berry_connection_from_D(D: np.ndarray, energies_ev: np.ndarray, *, denominator_cutoff_ev: float) -> np.ndarray:
    """Gauge-free interband Berry connection r_nm^a = -i D_nm^a/(E_n-E_m)."""

    return berry_connection_matrix(D, energies_ev, denominator_cutoff_ev=denominator_cutoff_ev)


def generalized_derivative_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> GeneralizedDerivativeResult:
    """Gauge-free generalized derivative in the common WannierBerri convention.

    Legacy hTG wrapper: formula work is delegated to `analysis.shift_current`
    and ultimately `analysis.response_derivative_gauge`.
    """

    result = generalized_derivative_from_velocity(
        D,
        energies_ev,
        denominator_cutoff_ev=float(denominator_cutoff_ev),
        second_velocity_h=W,
    )
    return GeneralizedDerivativeResult(values=result.values, skipped_small_denominators=result.skipped_small_denominators)


@dataclass(frozen=True)
class PairGeneralizedDerivativeResult:
    values: np.ndarray  # values[deriv_a, conn_b] = r_{nm;a}^b for one selected pair
    skipped_small_denominators: int


def generalized_derivative_pair_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    n: int,
    m: int,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> PairGeneralizedDerivativeResult:
    """Selected-pair generalized derivative using the common formula layer."""

    result = _gauge_generalized_derivative_pair(
        D,
        energies_ev,
        int(n),
        int(m),
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=W,
    )
    return PairGeneralizedDerivativeResult(
        values=result.values,
        skipped_small_denominators=result.skipped_small_denominators,
    )


def berry_connection_pair_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    n: int,
    m: int,
    *,
    denominator_cutoff_ev: float,
) -> np.ndarray:
    """Return r_nm^a for one selected pair as a two-component vector."""

    return _gauge_berry_connection_pair(
        D,
        energies_ev,
        int(n),
        int(m),
        denominator_cutoff=float(denominator_cutoff_ev),
    )


def component_transition_weight_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    occupations: np.ndarray,
    n: int,
    m: int,
    component: Component,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> tuple[complex, int]:
    """Return a selected-pair transition weight and skipped-denominator count."""

    r = berry_connection_matrix(D, energies_ev, denominator_cutoff_ev=denominator_cutoff_ev)
    result = component_transition_weight_from_gauge_pair(
        D,
        energies_ev,
        r,
        occupations,
        int(n),
        int(m),
        component,
        denominator_cutoff_ev=float(denominator_cutoff_ev),
        second_velocity_h=W,
        convention=HTG_LEGACY_CONVENTION,
    )
    return complex(result.weight), int(result.skipped_small_denominators)


@dataclass(frozen=True)
class ResponseTensors:
    energies_ev: np.ndarray
    occupations: np.ndarray
    D: np.ndarray
    r: np.ndarray
    r_covariant: np.ndarray
    skipped_small_denominators: int


def precompute_response_tensors(
    energies_ev: np.ndarray,
    evecs: np.ndarray,
    dhdk: tuple[np.ndarray, np.ndarray] | np.ndarray,
    *,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
    d2hdk: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | np.ndarray | None = None,
) -> ResponseTensors:
    tensors = precompute_shift_current_tensors(
        energies_ev,
        evecs,
        dhdk,
        mu_ev=mu_ev,
        temperature_k=temperature_k,
        denominator_cutoff_ev=denominator_cutoff_ev,
        d2hdk=d2hdk,
    )
    return ResponseTensors(
        energies_ev=tensors.energies_ev,
        occupations=tensors.occupations,
        D=tensors.velocity_h,
        r=tensors.berry_connection,
        r_covariant=tensors.berry_connection_gen_derivative,
        skipped_small_denominators=tensors.skipped_small_denominators,
    )


def component_transition_weight(tensors: ResponseTensors, n: int, m: int, component: Component) -> complex:
    """Return legacy hTG transition weight via the common formula layer."""

    common = _GenericShiftCurrentTensors(
        energies_ev=np.asarray(tensors.energies_ev, dtype=float),
        occupations=np.asarray(tensors.occupations, dtype=float),
        velocity_h=np.asarray(tensors.D, dtype=np.complex128),
        berry_connection=np.asarray(tensors.r, dtype=np.complex128),
        berry_connection_gen_derivative=np.asarray(tensors.r_covariant, dtype=np.complex128),
        skipped_small_denominators=int(tensors.skipped_small_denominators),
    )
    return _generic_component_transition_weight(
        common,
        int(n),
        int(m),
        component,
        convention=HTG_LEGACY_CONVENTION,
    )


def positive_transition_terms(
    tensors: ResponseTensors,
    component: Component,
    *,
    min_transition_ev: float = 0.0,
    min_abs_occupation_diff: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect positive-energy interband transition energies and component weights."""

    common = _GenericShiftCurrentTensors(
        energies_ev=np.asarray(tensors.energies_ev, dtype=float),
        occupations=np.asarray(tensors.occupations, dtype=float),
        velocity_h=np.asarray(tensors.D, dtype=np.complex128),
        berry_connection=np.asarray(tensors.r, dtype=np.complex128),
        berry_connection_gen_derivative=np.asarray(tensors.r_covariant, dtype=np.complex128),
        skipped_small_denominators=int(tensors.skipped_small_denominators),
    )
    return _generic_positive_transition_terms(
        common,
        component,
        min_transition_ev=min_transition_ev,
        min_abs_occupation_diff=min_abs_occupation_diff,
        convention=HTG_LEGACY_CONVENTION,
    )


def add_transitions_to_integral(
    integral: np.ndarray,
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
) -> None:
    """Accumulate one k-point's transitions into the BZ integral in-place."""

    _generic_add_transitions_to_integral(
        integral,
        photon_energies_ev,
        transition_energies_ev,
        transition_weights,
        k_weight_nm_inv_sq=k_weight_nm_inv_sq,
        eta_ev=eta_ev,
        include_bz_factor=True,
        convention=HTG_LEGACY_CONVENTION,
    )


def sigma_from_integral(integral_nm_per_ev: np.ndarray) -> np.ndarray:
    """Convert the accumulated complex integral to microampere*nm/V^2."""

    return np.real(-1.0j * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 * np.asarray(integral_nm_per_ev))


def spectra_from_transition_table(
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
) -> dict[str, np.ndarray]:
    """Build spectra for several named components from transition arrays."""

    return _generic_spectra_from_transition_table(
        photon_energies_ev,
        transition_table,
        k_weight_nm_inv_sq=k_weight_nm_inv_sq,
        eta_ev=eta_ev,
        prefactor=SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
        prefactor_phase=-1.0j,
        include_bz_factor=True,
        convention=HTG_LEGACY_CONVENTION,
    )

__all__ = [
    "SHIFT_CURRENT_PREFAC_UA_NM_PER_V2",
    "KB_EV_PER_K",
    "Component",
    "Axis",
    "AXIS_LABELS",
    "berry_connection_matrix",
    "fermi_occupation",
    "lorentzian_delta",
    "second_derivative_matrices",
    "velocity_matrices",
    "axis_index",
    "parse_component",
    "component_label",
    "GeneralizedDerivativeResult",
    "berry_connection_from_D",
    "generalized_derivative_from_D",
    "PairGeneralizedDerivativeResult",
    "generalized_derivative_pair_from_D",
    "berry_connection_pair_from_D",
    "component_transition_weight_from_D",
    "ResponseTensors",
    "precompute_response_tensors",
    "component_transition_weight",
    "positive_transition_terms",
    "add_transitions_to_integral",
    "sigma_from_integral",
    "spectra_from_transition_table",
]
