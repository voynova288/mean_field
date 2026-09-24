from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence, cast
import math

import numpy as np

from analysis.injection_current import (
    InjectionCurrentTensors,
    accumulate_injection_spectrum,
    cpge_part,
    injection_spectra_from_transition_table,
    injection_transition_weight,
    linear_injection_part,
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)
from analysis.shift_current import (
    SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    OpticalSymmetrization,
    ShiftCurrentComponent,
    ShiftCurrentConvention,
    ShiftCurrentTensors,
    add_transitions_to_integral,
    component_transition_weight,
    conductivity_from_integral,
    positive_transition_terms as positive_shift_transition_terms,
    precompute_shift_current_tensors,
    spectra_from_transition_table as shift_spectra_from_transition_table,
)

OpticalTransitionResponseKind = Literal["shift_current", "injection_current"]
OpticalKPointResponseKind = Literal[
    "linear_conductivity",
    "kerr_faraday",
    "second_harmonic_generation",
    "third_harmonic_generation",
]
OpticalResponseKind = Literal[
    "shift_current",
    "injection_current",
    "linear_conductivity",
    "kerr_faraday",
    "second_harmonic_generation",
    "third_harmonic_generation",
]
OpticalTransitionResponseKindLike = Literal[
    "shift",
    "shift_current",
    "injection",
    "injection_current",
    "cpge",
]
OpticalKPointResponseKindLike = Literal[
    "linear",
    "linear_conductivity",
    "conductivity",
    "kerr",
    "faraday",
    "kerr_faraday",
    "shg",
    "second_harmonic",
    "second_harmonic_generation",
    "thg",
    "third_harmonic",
    "third_harmonic_generation",
]
OpticalResponseKindLike = Literal[
    "shift",
    "shift_current",
    "injection",
    "injection_current",
    "cpge",
    "linear",
    "linear_conductivity",
    "conductivity",
    "kerr",
    "faraday",
    "kerr_faraday",
    "shg",
    "second_harmonic",
    "second_harmonic_generation",
    "thg",
    "third_harmonic",
    "third_harmonic_generation",
]


def canonical_response_kind(kind: OpticalResponseKindLike | str) -> OpticalResponseKind:
    """Normalize every user-facing optical response alias.

    The common package has two transition-table families and four full-k-point
    tensor families. ``"cpge"`` aliases injection current. ``"kerr"`` and
    ``"faraday"`` both return the paired Kerr/Faraday response.
    """

    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"shift", "shift_current"}:
        return "shift_current"
    if normalized in {"injection", "injection_current", "cpge"}:
        return "injection_current"
    if normalized in {"linear", "linear_conductivity", "conductivity"}:
        return "linear_conductivity"
    if normalized in {"kerr", "faraday", "kerr_faraday"}:
        return "kerr_faraday"
    if normalized in {"shg", "second_harmonic", "second_harmonic_generation"}:
        return "second_harmonic_generation"
    if normalized in {"thg", "third_harmonic", "third_harmonic_generation"}:
        return "third_harmonic_generation"
    raise ValueError(
        f"Unsupported optical response kind {kind!r}; expected a shift, "
        "injection/CPGE, linear-conductivity, Kerr/Faraday, SHG, or THG alias."
    )


def _require_transition_response_kind(
    kind: OpticalResponseKindLike | str,
    *,
    helper: str,
) -> OpticalTransitionResponseKind:
    resolved = canonical_response_kind(kind)
    if resolved not in {"shift_current", "injection_current"}:
        raise ValueError(
            f"{helper} supports only transition-table response kinds "
            "'shift_current' and 'injection_current'; use "
            "optical_response_from_kpoint_data for linear conductivity, "
            "Kerr/Faraday, SHG, or THG."
        )
    return cast(OpticalTransitionResponseKind, resolved)


@dataclass(frozen=True)
class OpticalResponseTensors:
    """Generic one-k-point optical tensors.

    This is a thin, explicit union over the two transition-table response
    families. System adapters should build this object from energies,
    eigenvectors, and Hamiltonian derivatives, then route all transition-table
    and spectrum construction through the generic helpers below.
    """

    kind: OpticalTransitionResponseKind
    shift_current: ShiftCurrentTensors | None = None
    injection_current: InjectionCurrentTensors | None = None

    @property
    def tensors(self) -> ShiftCurrentTensors | InjectionCurrentTensors:
        if self.kind == "shift_current":
            if self.shift_current is None:
                raise ValueError("shift_current tensors are missing")
            return self.shift_current
        if self.injection_current is None:
            raise ValueError("injection_current tensors are missing")
        return self.injection_current

    @property
    def energies_ev(self) -> np.ndarray:
        return self.tensors.energies_ev

    @property
    def occupations(self) -> np.ndarray:
        return self.tensors.occupations

    @property
    def velocity_h(self) -> np.ndarray:
        return self.tensors.velocity_h

    @property
    def berry_connection(self) -> np.ndarray:
        return self.tensors.berry_connection

    def require_shift_current(self) -> ShiftCurrentTensors:
        if self.kind != "shift_current" or self.shift_current is None:
            raise ValueError(f"Optical tensors have kind={self.kind!r}, not 'shift_current'")
        return self.shift_current

    def require_injection_current(self) -> InjectionCurrentTensors:
        if self.kind != "injection_current" or self.injection_current is None:
            raise ValueError(f"Optical tensors have kind={self.kind!r}, not 'injection_current'")
        return self.injection_current


OpticalTensors = OpticalResponseTensors


def precompute_optical_tensors(
    kind: OpticalTransitionResponseKindLike | str,
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
) -> OpticalResponseTensors:
    """Build one-k-point tensors for a supported optical response family.

    Parameters are deliberately the system-adapter contract: energies,
    eigenvectors, and Hamiltonian derivatives.  Momentum units and final SI
    prefactors remain caller/workflow responsibilities.

    ``d2hdk`` and ``principal_value_eta_ev`` are used only by shift current;
    ``d2hdk`` is accepted but ignored for injection current so callers may pass
    a shared derivative bundle without branching.  ``external_connection`` is
    not accepted here because external-position generalized derivatives need a
    separate audited implementation.
    """

    resolved = _require_transition_response_kind(
        kind,
        helper="precompute_optical_tensors",
    )
    if resolved == "shift_current":
        return OpticalResponseTensors(
            kind="shift_current",
            shift_current=precompute_shift_current_tensors(
                energies_ev,
                eigenvectors,
                dhdk,
                mu_ev=mu_ev,
                temperature_k=temperature_k,
                denominator_cutoff_ev=denominator_cutoff_ev,
                d2hdk=d2hdk,
                external_connection=external_connection,
                principal_value_eta_ev=principal_value_eta_ev,
            ),
        )
    if external_connection is not None:
        raise NotImplementedError(
            "external_connection is not supported for injection_current in analysis.optical; "
            "derive and validate external Berry-connection terms first."
        )
    if principal_value_eta_ev is not None:
        raise ValueError("principal_value_eta_ev is a shift-current regularizer and is not used for injection_current")
    return OpticalResponseTensors(
        kind="injection_current",
        injection_current=precompute_injection_current_tensors(
            energies_ev,
            eigenvectors,
            dhdk,
            mu_ev=mu_ev,
            temperature_k=temperature_k,
            denominator_cutoff_ev=denominator_cutoff_ev,
        ),
    )


def optical_transition_weight(
    tensors: OpticalResponseTensors,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    optical_symmetrization: OpticalSymmetrization = "sum",
    shift_convention: ShiftCurrentConvention | None = None,
) -> complex:
    """Return one occupation-weighted transition amplitude for any optical kind."""

    if tensors.kind == "shift_current":
        return component_transition_weight(
            tensors.require_shift_current(),
            initial_band,
            final_band,
            component,
            optical_symmetrization=optical_symmetrization,
            convention=shift_convention,
        )
    return injection_transition_weight(tensors.require_injection_current(), initial_band, final_band, component)


def positive_optical_transition_terms(
    tensors: OpticalResponseTensors,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
    optical_symmetrization: OpticalSymmetrization = "sum",
    shift_convention: ShiftCurrentConvention | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive transition energies and weights for any optical kind."""

    if tensors.kind == "shift_current":
        return positive_shift_transition_terms(
            tensors.require_shift_current(),
            component,
            selected_bands=selected_bands,
            min_transition_ev=min_transition_ev,
            max_transition_ev=max_transition_ev,
            min_abs_occupation_diff=min_abs_occupation_diff,
            optical_symmetrization=optical_symmetrization,
            convention=shift_convention,
        )
    return positive_injection_transition_terms(
        tensors.require_injection_current(),
        component,
        selected_bands=selected_bands,
        min_transition_ev=min_transition_ev,
        max_transition_ev=max_transition_ev,
        min_abs_occupation_diff=min_abs_occupation_diff,
    )


def accumulate_optical_spectrum(
    kind: OpticalTransitionResponseKindLike | str,
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight: float,
    eta_ev: float,
    include_bz_factor: bool = True,
    normalized_lorentzian: bool = True,
    shift_prefactor: float = SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    shift_prefactor_phase: complex = -1.0j,
    injection_prefactor: complex = -2.0 * math.pi,
    relaxation_time: float = 1.0,
    shift_convention: ShiftCurrentConvention | None = None,
) -> np.ndarray:
    """Accumulate a spectrum from transition tables for any optical kind.

    Shift-current output is real after applying ``shift_prefactor_phase`` and
    ``shift_prefactor``.  Injection-current output remains complex; use
    :func:`linear_injection_part` and :func:`cpge_part` for linear and circular
    channels.
    """

    resolved = _require_transition_response_kind(
        kind,
        helper="accumulate_optical_spectrum",
    )
    if resolved == "shift_current":
        integral = np.zeros_like(np.asarray(photon_energies_ev, dtype=float), dtype=np.complex128)
        add_transitions_to_integral(
            integral,
            photon_energies_ev,
            transition_energies_ev,
            transition_weights,
            k_weight_nm_inv_sq=float(k_weight),
            eta_ev=float(eta_ev),
            normalized_lorentzian=bool(normalized_lorentzian),
            include_bz_factor=bool(include_bz_factor),
            convention=shift_convention,
        )
        return conductivity_from_integral(integral, prefactor=shift_prefactor, phase=shift_prefactor_phase)
    return accumulate_injection_spectrum(
        photon_energies_ev,
        transition_energies_ev,
        transition_weights,
        k_weight=float(k_weight),
        eta_ev=float(eta_ev),
        include_bz_factor=bool(include_bz_factor),
        normalized_lorentzian=bool(normalized_lorentzian),
        prefactor=injection_prefactor,
        relaxation_time=float(relaxation_time),
    )


def optical_spectra_from_transition_table(
    kind: OpticalTransitionResponseKindLike | str,
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight: float,
    eta_ev: float,
    include_bz_factor: bool = True,
    normalized_lorentzian: bool = True,
    shift_prefactor: float = SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    shift_prefactor_phase: complex = -1.0j,
    injection_prefactor: complex = -2.0 * math.pi,
    relaxation_time: float = 1.0,
    shift_convention: ShiftCurrentConvention | None = None,
) -> dict[str, np.ndarray]:
    """Build spectra for a named transition table through the generic API."""

    resolved = _require_transition_response_kind(
        kind,
        helper="optical_spectra_from_transition_table",
    )
    if resolved == "shift_current":
        return shift_spectra_from_transition_table(
            photon_energies_ev,
            transition_table,
            k_weight_nm_inv_sq=float(k_weight),
            eta_ev=float(eta_ev),
            prefactor=shift_prefactor,
            prefactor_phase=shift_prefactor_phase,
            normalized_lorentzian=normalized_lorentzian,
            include_bz_factor=include_bz_factor,
            convention=shift_convention,
        )
    return injection_spectra_from_transition_table(
        photon_energies_ev,
        transition_table,
        k_weight=float(k_weight),
        eta_ev=float(eta_ev),
        include_bz_factor=include_bz_factor,
        normalized_lorentzian=normalized_lorentzian,
        prefactor=injection_prefactor,
        relaxation_time=relaxation_time,
    )


__all__ = [
    "OpticalKPointResponseKind",
    "OpticalKPointResponseKindLike",
    "OpticalResponseKind",
    "OpticalResponseKindLike",
    "OpticalTransitionResponseKind",
    "OpticalTransitionResponseKindLike",
    "OpticalResponseTensors",
    "OpticalTensors",
    "accumulate_optical_spectrum",
    "canonical_response_kind",
    "cpge_part",
    "linear_injection_part",
    "optical_spectra_from_transition_table",
    "optical_transition_weight",
    "positive_optical_transition_terms",
    "precompute_optical_tensors",
]
