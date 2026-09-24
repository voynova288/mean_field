from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, cast, overload

import numpy as np

from analysis.optical.core import OpticalResponseKindLike, canonical_response_kind
from analysis.optical.harmonic import (
    IndependentInputAdiabaticSwitching,
    PassosFiniteBandVelocityGauge,
    harmonic_response_from_kpoint_data,
)
from analysis.optical.kmesh import (
    OpticalKPointData,
    linear_conductivity_from_kpoint_data,
)
from analysis.optical.linear import LinearConductivityResult
from analysis.optical.passos import PassosHarmonicConductivityResult, PassosKPointData
from analysis.optical.magneto import (
    ComplexPolarization,
    MagnetoOpticalAmplitudes,
    VACUUM_IMPEDANCE_OHM,
    faraday_kerr_from_sheet_conductivity,
    faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed,
    faraday_kerr_from_sheet_conductivity_on_substrate,
    unwrap_polarization_angle,
)


SpectralRefractiveIndex = complex | float | np.ndarray
HarmonicResponseKindLike = Literal[
    "shg",
    "second_harmonic",
    "second-harmonic",
    "second_harmonic_generation",
    "second-harmonic-generation",
    "thg",
    "third_harmonic",
    "third-harmonic",
    "third_harmonic_generation",
    "third-harmonic-generation",
]


@dataclass(frozen=True, kw_only=True)
class NormalIncidenceSameSideGeometry:
    """Physical same-incidence-side sheet reflection and transmission.

    The incident wave approaches from medium 1, transmission enters medium 2,
    and reflection returns into medium 1. Refractive indices may be scalar or
    pointwise arrays on the response frequency grid.
    """

    incident_refractive_index: SpectralRefractiveIndex = 1.0
    transmitted_refractive_index: SpectralRefractiveIndex = 1.0
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM
    kind: Literal["normal_incidence_same_side"] = field(
        default="normal_incidence_same_side",
        init=False,
    )

    def __post_init__(self) -> None:
        n1 = np.array(self.incident_refractive_index, dtype=np.complex128, copy=True)
        n2 = np.array(self.transmitted_refractive_index, dtype=np.complex128, copy=True)
        if np.any(~np.isfinite(n1)) or np.any(~np.isfinite(n2)):
            raise ValueError("geometry refractive indices must contain only finite values")
        impedance = float(self.vacuum_impedance_ohm)
        if not np.isfinite(impedance) or impedance <= 0.0:
            raise ValueError("geometry vacuum_impedance_ohm must be finite and positive")
        n1.setflags(write=False)
        n2.setflags(write=False)
        object.__setattr__(self, "incident_refractive_index", n1)
        object.__setattr__(self, "transmitted_refractive_index", n2)
        object.__setattr__(self, "vacuum_impedance_ohm", impedance)


@dataclass(frozen=True, kw_only=True)
class Okada2016MixedPulseGeometry:
    """Okada vacuum-to-substrate Faraday and substrate-side Kerr geometry."""

    substrate_refractive_index: SpectralRefractiveIndex
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM
    kind: Literal["okada_2016_mixed_pulse_substrate"] = field(
        default="okada_2016_mixed_pulse_substrate",
        init=False,
    )

    def __post_init__(self) -> None:
        substrate = np.array(
            self.substrate_refractive_index,
            dtype=np.complex128,
            copy=True,
        )
        if np.any(~np.isfinite(substrate)):
            raise ValueError("substrate_refractive_index must contain only finite values")
        impedance = float(self.vacuum_impedance_ohm)
        if not np.isfinite(impedance) or impedance <= 0.0:
            raise ValueError("geometry vacuum_impedance_ohm must be finite and positive")
        substrate.setflags(write=False)
        object.__setattr__(self, "substrate_refractive_index", substrate)
        object.__setattr__(self, "vacuum_impedance_ohm", impedance)


KerrFaradayGeometry = NormalIncidenceSameSideGeometry | Okada2016MixedPulseGeometry


@dataclass(frozen=True)
class KerrFaradayResponse:
    """Generic normal-incidence response from a 2D sheet conductivity.

    ``linear`` stores the underlying optical sheet conductivity. Kerr/Faraday
    front doors require it to be in Siemens. ``angles`` stores the exact Jones
    amplitudes, polarization masks, geometry, and boundary-scattering metadata.
    """

    photon_energies_ev: np.ndarray
    linear: LinearConductivityResult
    angles: MagnetoOpticalAmplitudes

    @property
    def conductivity(self) -> np.ndarray:
        """Sheet conductivity tensor in Siemens, shape ``(n_omega,2,2)``."""

        return self.linear.conductivity

    @property
    def skipped_small_denominators(self) -> int:
        return int(self.linear.skipped_small_denominators)

    @property
    def faraday_angle_rad(self) -> np.ndarray:
        return self.angles.faraday_angle_rad

    @property
    def kerr_angle_rad(self) -> np.ndarray:
        return self.angles.kerr_angle_rad

    @property
    def faraday_polarization(self) -> ComplexPolarization:
        return self.angles.faraday_polarization

    @property
    def kerr_polarization(self) -> ComplexPolarization:
        return self.angles.kerr_polarization

    @property
    def faraday_ellipticity(self) -> np.ndarray:
        return self.angles.faraday_ellipticity

    @property
    def kerr_ellipticity(self) -> np.ndarray:
        return self.angles.kerr_ellipticity

    @property
    def geometry(self) -> str:
        return self.angles.geometry

    @property
    def tensor_convention(self) -> str:
        return self.angles.tensor_convention

    def unwrapped_faraday_angle_rad(
        self,
        *,
        axis: int = -1,
        valid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unwrap Faraday orientation modulo pi on contiguous valid segments."""

        mask = self.faraday_polarization.orientation_defined
        if valid is not None:
            mask = mask & np.asarray(valid, dtype=bool)
        return unwrap_polarization_angle(
            self.faraday_angle_rad,
            axis=axis,
            valid=mask,
        )

    def unwrapped_kerr_angle_rad(
        self,
        *,
        axis: int = -1,
        valid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unwrap Kerr orientation modulo pi on contiguous valid segments."""

        mask = self.kerr_polarization.orientation_defined
        if valid is not None:
            mask = mask & np.asarray(valid, dtype=bool)
        return unwrap_polarization_angle(
            self.kerr_angle_rad,
            axis=axis,
            valid=mask,
        )


def _validated_linear_sheet_result(
    linear: LinearConductivityResult,
) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(linear.photon_energies_ev, dtype=float)
    conductivity = np.asarray(linear.conductivity, dtype=np.complex128)
    if omega.ndim != 1 or np.any(~np.isfinite(omega)):
        raise ValueError(
            "linear.photon_energies_ev must be a finite one-dimensional array"
        )
    expected_shape = (omega.size, 2, 2)
    if conductivity.shape != expected_shape:
        raise ValueError(
            "Kerr/Faraday requires linear.conductivity shape "
            f"{expected_shape}, got {conductivity.shape}"
        )
    if np.any(~np.isfinite(conductivity)):
        raise ValueError("linear.conductivity must contain only finite values")
    return omega, conductivity


def _validated_photon_energy_grid(photon_energies_ev: np.ndarray) -> np.ndarray:
    omega = np.asarray(photon_energies_ev, dtype=float)
    if omega.ndim != 1 or np.any(~np.isfinite(omega)):
        raise ValueError("photon_energies_ev must be a finite one-dimensional array")
    return omega


def _validated_spectral_medium(
    value: complex | float | np.ndarray,
    omega: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    medium = np.asarray(value, dtype=np.complex128)
    if medium.shape not in ((), omega.shape):
        raise ValueError(
            f"{name} must be scalar or have the pointwise frequency shape "
            f"{omega.shape}, got {medium.shape}"
        )
    if np.any(~np.isfinite(medium)):
        raise ValueError(f"{name} must contain only finite values")
    return medium


def _response(
    linear: LinearConductivityResult,
    angles: MagnetoOpticalAmplitudes,
) -> KerrFaradayResponse:
    omega, _ = _validated_linear_sheet_result(linear)
    if np.shape(angles.faraday_angle_rad) != omega.shape:
        raise ValueError(
            "medium broadcasting must preserve the one-dimensional frequency "
            f"shape {omega.shape}, got {np.shape(angles.faraday_angle_rad)}"
        )
    return KerrFaradayResponse(
        photon_energies_ev=omega,
        linear=linear,
        angles=angles,
    )


def kerr_faraday_from_linear_conductivity(
    linear: LinearConductivityResult,
    *,
    incident_refractive_index: complex | float | np.ndarray = 1.0,
    transmitted_refractive_index: complex | float | np.ndarray = 1.0,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """Convert SI sheet conductivity using a generic two-medium interface.

    Reflection and transmission share incidence from medium 1. Defaults give a
    free-standing sheet in vacuum. Paper-specific mixed-pulse geometries remain
    explicit named helpers.
    """

    omega, conductivity = _validated_linear_sheet_result(linear)
    n1 = _validated_spectral_medium(
        incident_refractive_index, omega, name="incident_refractive_index"
    )
    n2 = _validated_spectral_medium(
        transmitted_refractive_index, omega, name="transmitted_refractive_index"
    )
    angles = faraday_kerr_from_sheet_conductivity(
        conductivity,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    return _response(linear, angles)


def kerr_faraday_from_linear_conductivity_liu_dai_2020_printed(
    linear: LinearConductivityResult,
    *,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """Convert conductivity with the transverse sign printed by Liu-Dai 2020."""

    _, conductivity = _validated_linear_sheet_result(linear)
    angles = faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed(
        conductivity,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    return _response(linear, angles)


def kerr_faraday_from_linear_conductivity_on_substrate(
    linear: LinearConductivityResult,
    *,
    substrate_refractive_index: complex | float | np.ndarray,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """Convert SI sheet conductivity in the Okada mixed-pulse geometry."""

    omega, conductivity = _validated_linear_sheet_result(linear)
    substrate = _validated_spectral_medium(
        substrate_refractive_index, omega, name="substrate_refractive_index"
    )
    angles = faraday_kerr_from_sheet_conductivity_on_substrate(
        conductivity,
        substrate_refractive_index=substrate,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    return _response(linear, angles)


def kerr_faraday_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    prefactor: complex,
    eta_ev: float = 1.0e-3,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    incident_refractive_index: complex | float | np.ndarray = 1.0,
    transmitted_refractive_index: complex | float | np.ndarray = 1.0,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """K-point payloads -> SI sheet conductivity -> generic sheet optics.

    ``prefactor`` is required and must convert the adapter's velocity and
    reciprocal-space units into 2D sheet conductivity in Siemens.
    """

    linear = linear_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        eta_ev=eta_ev,
        prefactor=prefactor,
        include_bz_factor=include_bz_factor,
        denominator_cutoff_ev=denominator_cutoff_ev,
        selected_bands=selected_bands,
    )
    return kerr_faraday_from_linear_conductivity(
        linear,
        incident_refractive_index=incident_refractive_index,
        transmitted_refractive_index=transmitted_refractive_index,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )


def kerr_faraday_from_kpoint_data_liu_dai_2020_printed(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    prefactor: complex,
    eta_ev: float = 1.0e-3,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """K-point front door for Liu-Dai's explicitly printed transverse sign."""

    linear = linear_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        eta_ev=eta_ev,
        prefactor=prefactor,
        include_bz_factor=include_bz_factor,
        denominator_cutoff_ev=denominator_cutoff_ev,
        selected_bands=selected_bands,
    )
    return kerr_faraday_from_linear_conductivity_liu_dai_2020_printed(
        linear,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )


def kerr_faraday_from_kpoint_data_on_substrate(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    substrate_refractive_index: complex | float | np.ndarray,
    prefactor: complex,
    eta_ev: float = 1.0e-3,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> KerrFaradayResponse:
    """K-point payloads -> SI sheet conductivity -> Okada pulse geometry."""

    linear = linear_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        eta_ev=eta_ev,
        prefactor=prefactor,
        include_bz_factor=include_bz_factor,
        denominator_cutoff_ev=denominator_cutoff_ev,
        selected_bands=selected_bands,
    )
    return kerr_faraday_from_linear_conductivity_on_substrate(
        linear,
        substrate_refractive_index=substrate_refractive_index,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )


OpticalKPointResponse = (
    LinearConductivityResult
    | KerrFaradayResponse
    | PassosHarmonicConductivityResult
)


@overload
def optical_response_from_kpoint_data(
    kind: Literal[
        "linear",
        "linear_conductivity",
        "linear-conductivity",
        "conductivity",
    ],
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    si_prefactor: complex,
    eta_ev: float = 1.0e-3,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    geometry: None = None,
) -> LinearConductivityResult: ...


@overload
def optical_response_from_kpoint_data(
    kind: Literal[
        "kerr",
        "faraday",
        "kerr_faraday",
        "kerr-faraday",
    ],
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    si_prefactor: complex,
    eta_ev: float = 1.0e-3,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    geometry: KerrFaradayGeometry | None = None,
) -> KerrFaradayResponse: ...


@overload
def optical_response_from_kpoint_data(
    kind: HarmonicResponseKindLike,
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    adiabatic_gamma_ev: float,
    formulation: None = None,
    si_prefactor: None = None,
    eta_ev: None = None,
    include_bz_factor: None = None,
    denominator_cutoff_ev: None = None,
    selected_bands: None = None,
    geometry: None = None,
) -> PassosHarmonicConductivityResult: ...


@overload
def optical_response_from_kpoint_data(
    kind: HarmonicResponseKindLike,
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    formulation: PassosFiniteBandVelocityGauge,
    adiabatic_gamma_ev: None = None,
    si_prefactor: None = None,
    eta_ev: None = None,
    include_bz_factor: None = None,
    denominator_cutoff_ev: None = None,
    selected_bands: None = None,
    geometry: None = None,
) -> PassosHarmonicConductivityResult: ...


@overload
def optical_response_from_kpoint_data(
    kind: str,
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData] | Iterable[PassosKPointData],
    *,
    si_prefactor: complex | None = None,
    eta_ev: float | None = None,
    include_bz_factor: bool | None = None,
    denominator_cutoff_ev: float | None = None,
    selected_bands: Iterable[int] | None = None,
    geometry: KerrFaradayGeometry | None = None,
    adiabatic_gamma_ev: float | None = None,
    formulation: PassosFiniteBandVelocityGauge | None = None,
) -> OpticalKPointResponse: ...


def optical_response_from_kpoint_data(
    kind: OpticalResponseKindLike | str,
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData] | Iterable[PassosKPointData],
    *,
    si_prefactor: complex | None = None,
    eta_ev: float | None = None,
    include_bz_factor: bool | None = None,
    denominator_cutoff_ev: float | None = None,
    selected_bands: Iterable[int] | None = None,
    geometry: KerrFaradayGeometry | None = None,
    adiabatic_gamma_ev: float | None = None,
    formulation: PassosFiniteBandVelocityGauge | None = None,
) -> OpticalKPointResponse:
    """Common full-k-point front door for linear, Kerr, SHG, and THG.

    Linear/Kerr payloads require an explicit sheet-SI ``si_prefactor``. Passos
    SHG/THG payloads instead carry a complete derivative tower and exact BZ/unit
    contract, so their SI conversion is derived internally and an external
    prefactor or selected-band window is rejected. Generic Kerr always uses the
    physical ``j=sigma E`` Maxwell convention; Liu-Dai printed-sign
    compatibility remains available only through its named API.
    """

    omega = _validated_photon_energy_grid(photon_energies_ev)
    resolved = canonical_response_kind(kind)
    if resolved in {"shift_current", "injection_current"}:
        raise ValueError(
            "optical_response_from_kpoint_data supports linear conductivity, "
            "Kerr/Faraday, SHG, and THG; use the transition-table generic API "
            f"for {resolved!r}."
        )
    if resolved in {
        "second_harmonic_generation",
        "third_harmonic_generation",
    }:
        if omega.size == 0 or np.any(omega <= 0.0):
            raise ValueError("SHG/THG photon_energies_ev must be nonempty and positive")
        if si_prefactor is not None:
            raise ValueError("SHG/THG derives its SI prefactor from the Passos payload")
        if eta_ev is not None:
            raise ValueError("SHG/THG uses adiabatic_gamma_ev, not eta_ev")
        if include_bz_factor is not None:
            raise ValueError("SHG/THG fixes the full-BZ (2*pi)^-d factor internally")
        if denominator_cutoff_ev is not None:
            raise ValueError("SHG/THG has no denominator_cutoff_ev policy")
        if selected_bands is not None:
            raise ValueError("SHG/THG forbids selected_bands; use the complete finite model")
        if geometry is not None:
            raise ValueError("geometry is valid only for a Kerr/Faraday response")
        if formulation is not None and adiabatic_gamma_ev is not None:
            raise ValueError(
                "SHG/THG accepts either formulation or legacy "
                "adiabatic_gamma_ev, not both"
            )
        if formulation is None:
            if adiabatic_gamma_ev is None:
                raise TypeError("SHG/THG requires formulation or adiabatic_gamma_ev")
            if isinstance(adiabatic_gamma_ev, (bool, np.bool_)):
                raise TypeError("adiabatic_gamma_ev must be a real scalar, not bool")
            gamma = float(adiabatic_gamma_ev)
            if not np.isfinite(gamma) or gamma <= 0.0:
                raise ValueError("adiabatic_gamma_ev must be finite and positive")
            formulation = PassosFiniteBandVelocityGauge(
                switching=IndependentInputAdiabaticSwitching(gamma_ev=gamma)
            )
        elif not isinstance(formulation, PassosFiniteBandVelocityGauge):
            raise TypeError("formulation must be PassosFiniteBandVelocityGauge")
        return harmonic_response_from_kpoint_data(
            resolved,
            omega,
            cast(Iterable[PassosKPointData], kpoints),
            formulation=formulation,
        )

    if formulation is not None:
        raise ValueError("formulation is valid only for SHG/THG")
    if adiabatic_gamma_ev is not None:
        raise ValueError("adiabatic_gamma_ev is valid only for SHG/THG")
    if si_prefactor is None:
        raise TypeError("si_prefactor must be a finite complex scalar")
    try:
        prefactor = complex(si_prefactor)
    except (TypeError, ValueError) as error:
        raise TypeError("si_prefactor must be a finite complex scalar") from error
    if not np.isfinite(prefactor.real) or not np.isfinite(prefactor.imag):
        raise ValueError("si_prefactor must be a finite complex scalar")
    eta = 1.0e-3 if eta_ev is None else float(eta_ev)
    cutoff = 1.0e-12 if denominator_cutoff_ev is None else float(
        denominator_cutoff_ev
    )
    use_bz_factor = True if include_bz_factor is None else bool(include_bz_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta_ev must be finite and positive")
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("denominator_cutoff_ev must be finite and non-negative")
    if resolved == "linear_conductivity":
        if geometry is not None:
            raise ValueError("geometry is valid only for a Kerr/Faraday response")
        return linear_conductivity_from_kpoint_data(
            omega,
            cast(Iterable[OpticalKPointData], kpoints),
            eta_ev=eta,
            prefactor=prefactor,
            include_bz_factor=use_bz_factor,
            denominator_cutoff_ev=cutoff,
            selected_bands=selected_bands,
        )

    selected_geometry: KerrFaradayGeometry
    if geometry is None:
        selected_geometry = NormalIncidenceSameSideGeometry()
    elif isinstance(
        geometry,
        (NormalIncidenceSameSideGeometry, Okada2016MixedPulseGeometry),
    ):
        selected_geometry = geometry
    else:
        raise TypeError(
            "geometry must be NormalIncidenceSameSideGeometry, "
            "Okada2016MixedPulseGeometry, or None"
        )

    if isinstance(selected_geometry, NormalIncidenceSameSideGeometry):
        n1 = _validated_spectral_medium(
            selected_geometry.incident_refractive_index,
            omega,
            name="geometry.incident_refractive_index",
        )
        n2 = _validated_spectral_medium(
            selected_geometry.transmitted_refractive_index,
            omega,
            name="geometry.transmitted_refractive_index",
        )
        return kerr_faraday_from_kpoint_data(
            omega,
            cast(Iterable[OpticalKPointData], kpoints),
            prefactor=prefactor,
            eta_ev=eta,
            include_bz_factor=use_bz_factor,
            denominator_cutoff_ev=cutoff,
            selected_bands=selected_bands,
            incident_refractive_index=n1,
            transmitted_refractive_index=n2,
            vacuum_impedance_ohm=selected_geometry.vacuum_impedance_ohm,
        )
    substrate = _validated_spectral_medium(
        selected_geometry.substrate_refractive_index,
        omega,
        name="geometry.substrate_refractive_index",
    )
    return kerr_faraday_from_kpoint_data_on_substrate(
        omega,
        cast(Iterable[OpticalKPointData], kpoints),
        substrate_refractive_index=substrate,
        prefactor=prefactor,
        eta_ev=eta,
        include_bz_factor=use_bz_factor,
        denominator_cutoff_ev=cutoff,
        selected_bands=selected_bands,
        vacuum_impedance_ohm=selected_geometry.vacuum_impedance_ohm,
    )


__all__ = [
    "KerrFaradayGeometry",
    "KerrFaradayResponse",
    "NormalIncidenceSameSideGeometry",
    "Okada2016MixedPulseGeometry",
    "OpticalKPointResponse",
    "SpectralRefractiveIndex",
    "kerr_faraday_from_kpoint_data",
    "kerr_faraday_from_kpoint_data_liu_dai_2020_printed",
    "kerr_faraday_from_kpoint_data_on_substrate",
    "kerr_faraday_from_linear_conductivity",
    "kerr_faraday_from_linear_conductivity_liu_dai_2020_printed",
    "kerr_faraday_from_linear_conductivity_on_substrate",
    "optical_response_from_kpoint_data",
]
