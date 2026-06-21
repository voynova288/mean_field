from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .components import (
    Axis,
    Component,
    ShiftCurrentComponent,
    axis_index,
    component_from_any,
    component_label,
    parse_component,
)
from .conventions import (
    E_CHARGE_C,
    HBAR_EV_S,
    HBAR_J_S,
    HTG_LEGACY_CONVENTION,
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    KB_EV_PER_K,
    OpticalSymmetrization,
    SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    ShiftCurrentConvention,
    WANNIERBERRI_INTERNAL_IMN_CONVENTION,
)
from .heatmap import (
    accumulate_fermi_omega_heatmap,
    add_transitions_to_integral,
    conductivity_from_integral,
    fermi_window_indices,
    lorentzian_delta,
    spectra_from_transition_table,
)
from .occupations import fermi_occupation

from .gauge import (
    GeneralizedDerivativeData,
    HamiltonianGaugeData,
    PairGeneralizedDerivativeData,
    berry_connection_generalized_derivative,
    berry_connection_generalized_derivative_pair,
    energy_difference_inverse,
    hamiltonian_gauge_data,
    matrix_in_eigenbasis,
)



def _operator_array(operators: Sequence[np.ndarray] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(operators, dtype=np.complex128)
    if arr.ndim < 3 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"{name} must have shape (...,basis,basis), got {arr.shape}")
    return arr


def velocity_matrices(eigenvectors: np.ndarray, dhdk: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Return ``<u_n|partial_a H|u_m>`` with shape ``(ndim,nb,nb)``."""

    raw = _operator_array(dhdk, name="dhdk")
    if raw.ndim != 3:
        raise ValueError(f"dhdk must have shape (ndim,basis,basis), got {raw.shape}")
    return matrix_in_eigenbasis(eigenvectors, raw)


def second_derivative_matrices(eigenvectors: np.ndarray, d2hdk: Sequence[Sequence[np.ndarray]] | np.ndarray) -> np.ndarray:
    """Return ``<u_n|partial_a partial_b H|u_m>`` as ``(ndim,ndim,nb,nb)``."""

    raw = _operator_array(d2hdk, name="d2hdk")
    if raw.ndim != 4 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"d2hdk must have shape (ndim,ndim,basis,basis), got {raw.shape}")
    return matrix_in_eigenbasis(eigenvectors, raw)


def berry_connection_matrix(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
) -> np.ndarray:
    """Return interband Berry connection ``A^a_nm = -i V^a_nm/(E_n-E_m)``.

    This is exactly the ``external_terms=False`` WannierBerri Hamiltonian-gauge
    convention ``A_H = i D_H`` with ``D_H = -V/(E_n-E_m)``.
    """

    V = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies_ev, dtype=float)
    if V.ndim != 3 or V.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V.shape} for nb={e.size}")
    inv = energy_difference_inverse(e, cutoff=float(denominator_cutoff_ev))
    return 1.0j * (-V * inv[None, :, :])


def generalized_derivative_from_velocity(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
) -> GeneralizedDerivativeData:
    """Generic full-band generalized derivative of the Berry connection."""

    return berry_connection_generalized_derivative(
        velocity_h,
        energies_ev,
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=second_velocity_h,
        principal_value_eta=principal_value_eta_ev,
    )


@dataclass(frozen=True)
class ShiftCurrentTensors:
    """Reusable one-k-point ingredients for shift-current calculations."""

    energies_ev: np.ndarray
    occupations: np.ndarray
    velocity_h: np.ndarray
    berry_connection: np.ndarray
    berry_connection_gen_derivative: np.ndarray
    skipped_small_denominators: int
    second_velocity_h: np.ndarray | None = None
    gauge_data: HamiltonianGaugeData | None = None

    @property
    def D(self) -> np.ndarray:
        return self.velocity_h

    @property
    def r(self) -> np.ndarray:
        return self.berry_connection

    @property
    def r_covariant(self) -> np.ndarray:
        return self.berry_connection_gen_derivative




def precompute_shift_current_tensors(
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: Sequence[np.ndarray] | np.ndarray,
    *,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
    d2hdk: Sequence[Sequence[np.ndarray]] | np.ndarray | None = None,
    external_connection: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
) -> ShiftCurrentTensors:
    """Build full one-k-point shift-current tensors from a system adapter.

    A system only provides eigenpairs and Hamiltonian derivatives.  The
    Hamiltonian-gauge rotation and generalized derivative are delegated to
    ``analysis.optical_response.gauge``.
    """

    if external_connection is not None:
        raise NotImplementedError(
            "external_connection is not supported in analysis.shift_current yet: "
            "the generalized derivative of external position/Berry terms must be implemented first. "
            "Use analysis.optical_response.gauge directly for audited external_terms work."
        )
    raw_dh = _operator_array(dhdk, name="dhdk")
    raw_d2 = None if d2hdk is None else _operator_array(d2hdk, name="d2hdk")
    gauge = hamiltonian_gauge_data(
        energies_ev,
        eigenvectors,
        raw_dh,
        denominator_cutoff=float(denominator_cutoff_ev),
        d2hdk=raw_d2,
        external_connection=external_connection,
    )
    gd = generalized_derivative_from_velocity(
        gauge.velocity_h,
        gauge.energies,
        denominator_cutoff_ev=float(denominator_cutoff_ev),
        second_velocity_h=gauge.second_velocity_h,
        principal_value_eta_ev=principal_value_eta_ev,
    )
    return ShiftCurrentTensors(
        energies_ev=gauge.energies,
        occupations=fermi_occupation(gauge.energies, mu_ev=mu_ev, temperature_k=temperature_k),
        velocity_h=gauge.velocity_h,
        berry_connection=gauge.berry_connection,
        berry_connection_gen_derivative=gd.values,
        skipped_small_denominators=gd.skipped_small_denominators,
        second_velocity_h=gauge.second_velocity_h,
        gauge_data=gauge,
    )


def _validate_optical_symmetrization(value: OpticalSymmetrization) -> OpticalSymmetrization:
    if value not in ("none", "sum", "average"):
        raise ValueError(f"optical_symmetrization must be 'none', 'sum', or 'average', got {value!r}")
    return value


def component_amplitude_from_pair(
    berry_connection: np.ndarray,
    pair_gen_derivative: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> complex:
    """Return the complex geometric product for one direct transition.

    For ``sigma^a_{bc}``, the ordered product is
    ``A^b_mn (A^c_nm)_;a`` for transition ``n -> m``.  Optical-axis
    symmetrization and geometric sign may be supplied explicitly or through a
    named convention such as ``JOYA_EQ7_GEOMETRIC_CONVENTION``.
    """

    comp = component_from_any(component)
    if convention is not None:
        optical_symmetrization = convention.optical_symmetrization
    sym = _validate_optical_symmetrization(optical_symmetrization)
    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(pair_gen_derivative, dtype=np.complex128)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"berry_connection must have shape (ndim,nb,nb), got {A.shape}")
    if G.shape != (A.shape[0], A.shape[0]):
        raise ValueError(f"pair_gen_derivative has shape {G.shape}, expected {(A.shape[0], A.shape[0])}")
    n = int(initial_band)
    m = int(final_band)
    a, b, c = comp.as_tuple
    ordered = A[b, m, n] * G[a, c]
    if sym == "none":
        total = ordered
    else:
        swapped = A[c, m, n] * G[a, b]
        total = ordered + swapped
        if sym == "average":
            total *= 0.5
    if convention is not None:
        total *= float(convention.geometric_sign)
    return complex(total)


def component_kernel_from_pair(
    berry_connection: np.ndarray,
    pair_gen_derivative: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> float:
    """Return the real gauge-invariant kernel ``Im[component_amplitude]``."""

    return float(
        np.imag(
            component_amplitude_from_pair(
                berry_connection,
                pair_gen_derivative,
                initial_band=initial_band,
                final_band=final_band,
                component=component,
                optical_symmetrization=optical_symmetrization,
                convention=convention,
            )
        )
    )


@dataclass(frozen=True)
class PairTransitionKernel:
    """Selected-pair geometric kernel using the full virtual/intermediate sum."""

    amplitude: complex
    kernel: float
    generalized_derivative: PairGeneralizedDerivativeData

    @property
    def skipped_small_denominators(self) -> int:
        return int(self.generalized_derivative.skipped_small_denominators)


@dataclass(frozen=True)
class PairTransitionWeight:
    """Occupation-weighted selected-pair result using the full virtual sum."""

    weight: complex
    kernel: float
    generalized_derivative: PairGeneralizedDerivativeData

    @property
    def skipped_small_denominators(self) -> int:
        return int(self.generalized_derivative.skipped_small_denominators)


def component_kernel_from_gauge_pair(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    berry_connection: np.ndarray,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> PairTransitionKernel:
    """Compute one raw geometric transition kernel from Hamiltonian-gauge data.

    This is the selected-pair/full-virtual-band path used by Joya-style
    heatmaps: it avoids constructing the full generalized-derivative tensor but
    still sums intermediate states over the full supplied basis.
    """

    n = int(initial_band)
    m = int(final_band)
    pair = berry_connection_generalized_derivative_pair(
        velocity_h,
        energies_ev,
        n,
        m,
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=second_velocity_h,
        principal_value_eta=principal_value_eta_ev,
    )
    amp = component_amplitude_from_pair(
        berry_connection,
        pair.values,
        initial_band=n,
        final_band=m,
        component=component,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    return PairTransitionKernel(amplitude=amp, kernel=float(np.imag(amp)), generalized_derivative=pair)


def component_transition_weight_from_gauge_pair(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    berry_connection: np.ndarray,
    occupations: np.ndarray,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> PairTransitionWeight:
    """Compute one occupation-weighted transition without a full GD tensor."""

    n = int(initial_band)
    m = int(final_band)
    raw = component_kernel_from_gauge_pair(
        velocity_h,
        energies_ev,
        berry_connection,
        n,
        m,
        component,
        denominator_cutoff_ev=denominator_cutoff_ev,
        second_velocity_h=second_velocity_h,
        principal_value_eta_ev=principal_value_eta_ev,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    fnm = float(np.asarray(occupations, dtype=float)[n] - np.asarray(occupations, dtype=float)[m])
    weight = complex(fnm * raw.amplitude)
    return PairTransitionWeight(weight=weight, kernel=float(np.imag(weight)), generalized_derivative=raw.generalized_derivative)


def component_transition_weight(
    tensors: ShiftCurrentTensors,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> complex:
    """Return occupation-weighted full-tensor transition amplitude."""

    n = int(initial_band)
    m = int(final_band)
    G = np.asarray(tensors.berry_connection_gen_derivative, dtype=np.complex128)
    if G.ndim != 4 or G.shape[2:] != (tensors.energies_ev.size, tensors.energies_ev.size):
        raise ValueError(f"berry_connection_gen_derivative must have shape (ndim,ndim,nb,nb), got {G.shape}")
    amp = component_amplitude_from_pair(
        tensors.berry_connection,
        G[:, :, n, m],
        initial_band=n,
        final_band=m,
        component=component,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    fnm = float(tensors.occupations[n] - tensors.occupations[m])
    return complex(fnm * amp)


def positive_transition_pairs(
    energies_ev: np.ndarray,
    occupations: np.ndarray | None = None,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
) -> list[tuple[int, int, float, float]]:
    """Return ``(n,m,E_m-E_n,f_n-f_m)`` for positive direct transitions."""

    energies = np.asarray(energies_ev, dtype=float)
    occ = None if occupations is None else np.asarray(occupations, dtype=float)
    if occ is not None and occ.shape != energies.shape:
        raise ValueError(f"occupations has shape {occ.shape}, expected {energies.shape}")
    bands = list(range(energies.size)) if selected_bands is None else [int(i) for i in selected_bands]
    out: list[tuple[int, int, float, float]] = []
    for n in bands:
        for m in bands:
            if n == m:
                continue
            transition = float(energies[m] - energies[n])
            if transition <= float(min_transition_ev):
                continue
            if max_transition_ev is not None and transition > float(max_transition_ev):
                continue
            fnm = 1.0 if occ is None else float(occ[n] - occ[m])
            if occ is not None and abs(fnm) < float(min_abs_occupation_diff):
                continue
            out.append((n, m, transition, fnm))
    return out


def positive_transition_terms(
    tensors: ShiftCurrentTensors,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect transition energies and complex weights for one k point."""

    transitions: list[float] = []
    weights: list[complex] = []
    for n, m, transition_ev, _fnm in positive_transition_pairs(
        tensors.energies_ev,
        tensors.occupations,
        selected_bands=selected_bands,
        min_transition_ev=min_transition_ev,
        max_transition_ev=max_transition_ev,
        min_abs_occupation_diff=min_abs_occupation_diff,
    ):
        weight = component_transition_weight(
            tensors,
            n,
            m,
            component,
            optical_symmetrization=optical_symmetrization,
            convention=convention,
        )
        if np.isfinite(weight.real) and np.isfinite(weight.imag):
            transitions.append(float(transition_ev))
            weights.append(complex(weight))
    return np.asarray(transitions, dtype=float), np.asarray(weights, dtype=np.complex128)




__all__ = [
    "Axis",
    "Component",
    "E_CHARGE_C",
    "HBAR_EV_S",
    "HBAR_J_S",
    "HTG_LEGACY_CONVENTION",
    "JOYA_EQ7_GEOMETRIC_CONVENTION",
    "KB_EV_PER_K",
    "OpticalSymmetrization",
    "PairTransitionKernel",
    "PairTransitionWeight",
    "SHIFT_CURRENT_PREFAC_UA_NM_PER_V2",
    "ShiftCurrentComponent",
    "ShiftCurrentConvention",
    "ShiftCurrentTensors",
    "WANNIERBERRI_INTERNAL_IMN_CONVENTION",
    "accumulate_fermi_omega_heatmap",
    "add_transitions_to_integral",
    "axis_index",
    "berry_connection_matrix",
    "component_amplitude_from_pair",
    "component_from_any",
    "component_kernel_from_gauge_pair",
    "component_kernel_from_pair",
    "component_label",
    "component_transition_weight",
    "component_transition_weight_from_gauge_pair",
    "conductivity_from_integral",
    "fermi_occupation",
    "fermi_window_indices",
    "generalized_derivative_from_velocity",
    "lorentzian_delta",
    "parse_component",
    "positive_transition_pairs",
    "positive_transition_terms",
    "precompute_shift_current_tensors",
    "second_derivative_matrices",
    "spectra_from_transition_table",
    "velocity_matrices",
]
