from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# CODATA values used for quasi-2D sheet optics.
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
VACUUM_PERMEABILITY_H_PER_M = 1.25663706212e-6
VACUUM_IMPEDANCE_OHM = float(
    np.sqrt(VACUUM_PERMEABILITY_H_PER_M / VACUUM_PERMITTIVITY_F_PER_M)
)
VACUUM_ADMITTANCE_S = 1.0 / VACUUM_IMPEDANCE_OHM

# Public sign contract for complex fields and reflected-beam reporting.
PHASOR_CONVENTION = "exp(-i omega t)"
SAMPLE_AXES_CONVENTION = "fixed right-handed sample axes (x,y,+z)"
REFLECTED_FIELD_CONVENTION = "Jones components remain in fixed sample x/y axes"


def _readonly_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ComplexPolarization:
    """Principal polarization orientation and Okada ellipticity.

    ``tan(complex_angle_rad)=E_y/E_x`` modulo pi. The orientation is the
    principal Stokes branch in ``[-pi/2, pi/2]``. Okada's signed ellipticity is
    ``tanh(Im(complex_angle_rad))``.

    Complex amplitudes use ``exp(-i omega t)``. A nonzero circular field has a
    defined ellipticity ``+/-1`` but no defined ellipse orientation. A zero
    Jones field has no defined polarization.
    ``polarization_defined`` and ``orientation_defined`` expose both cases.
    """

    complex_angle_rad: np.ndarray
    rotation_angle_rad: np.ndarray
    ellipticity: np.ndarray
    jones_ratio: np.ndarray
    polarization_defined: np.ndarray
    orientation_defined: np.ndarray


@dataclass(frozen=True)
class NormalIncidenceSheetScattering:
    """Jones scattering matrices for one zero-thickness conducting sheet.

    The interface is between scalar, nonmagnetic media with refractive indices
    ``incident_refractive_index`` and ``transmitted_refractive_index``. Final
    matrix axes are ``(output polarization, input polarization)`` in fixed
    sample Cartesian ``(x,y)`` axes. The local solve points from medium 1 to
    medium 2; ``incident_propagation_direction_in_sample`` records whether
    that direction is sample ``+z`` or ``-z``. Reflected components remain in
    fixed sample axes rather than being reoriented for an observer looking
    along the reflected beam. Complex fields
    use ``exp(-i omega t)``. The conductivity contract is
    ``j_a=sum_b sigma_ab E_b`` and sheet conductivity is in Siemens.
    """

    transmission_matrix: np.ndarray
    reflection_matrix: np.ndarray
    incident_refractive_index: np.ndarray
    transmitted_refractive_index: np.ndarray
    vacuum_impedance_ohm: float
    incident_propagation_direction_in_sample: int = 1

    def __post_init__(self) -> None:
        transmission = _readonly_array(
            self.transmission_matrix, dtype=np.complex128
        )
        reflection = _readonly_array(
            self.reflection_matrix, dtype=np.complex128
        )
        if transmission.ndim < 2 or transmission.shape[-2:] != (2, 2):
            raise ValueError("transmission_matrix must have final shape (2,2)")
        if reflection.shape != transmission.shape:
            raise ValueError(
                "reflection_matrix must have the same shape as transmission_matrix"
            )
        batch_shape = transmission.shape[:-2]
        n1 = _readonly_array(
            np.broadcast_to(self.incident_refractive_index, batch_shape),
            dtype=np.complex128,
        )
        n2 = _readonly_array(
            np.broadcast_to(self.transmitted_refractive_index, batch_shape),
            dtype=np.complex128,
        )
        identity = np.broadcast_to(
            np.eye(2, dtype=np.complex128), transmission.shape
        )
        if not np.allclose(reflection, transmission - identity, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError("reflection_matrix must equal transmission_matrix-I")
        object.__setattr__(self, "transmission_matrix", transmission)
        object.__setattr__(self, "reflection_matrix", reflection)
        object.__setattr__(self, "incident_refractive_index", n1)
        object.__setattr__(self, "transmitted_refractive_index", n2)
        object.__setattr__(
            self,
            "vacuum_impedance_ohm",
            _validated_vacuum_impedance(self.vacuum_impedance_ohm),
        )
        direction = int(self.incident_propagation_direction_in_sample)
        if direction not in (-1, 1):
            raise ValueError(
                "incident_propagation_direction_in_sample must be +1 or -1"
            )
        object.__setattr__(
            self, "incident_propagation_direction_in_sample", direction
        )

    @staticmethod
    def _apply(matrix: np.ndarray, incident_jones: np.ndarray) -> np.ndarray:
        field = np.asarray(incident_jones, dtype=np.complex128)
        if field.shape[-1:] != (2,):
            raise ValueError(
                "incident_jones must have final shape (2,), "
                f"got {field.shape}"
            )
        try:
            batch_shape = np.broadcast_shapes(matrix.shape[:-2], field.shape[:-1])
        except ValueError as error:
            raise ValueError(
                "incident_jones leading shape is not broadcast-compatible with "
                f"the scattering batch shape {matrix.shape[:-2]}"
            ) from error
        matrix_b = np.broadcast_to(matrix, batch_shape + (2, 2))
        field_b = np.broadcast_to(field, batch_shape + (2,))
        return np.einsum("...ab,...b->...a", matrix_b, field_b, optimize=True)

    def transmitted_field(self, incident_jones: np.ndarray) -> np.ndarray:
        """Apply the transmission Jones matrix to an arbitrary incident field."""

        return self._apply(self.transmission_matrix, incident_jones)

    def reflected_field(self, incident_jones: np.ndarray) -> np.ndarray:
        """Apply the reflection Jones matrix to an arbitrary incident field."""

        return self._apply(self.reflection_matrix, incident_jones)

    @property
    def transmitted_x_incident(self) -> np.ndarray:
        """Transmitted Jones field for unit x-polarized incidence."""

        return self.transmission_matrix[..., :, 0]

    @property
    def reflected_x_incident(self) -> np.ndarray:
        """Reflected Jones field for unit x-polarized incidence."""

        return self.reflection_matrix[..., :, 0]


@dataclass(frozen=True)
class MagnetoOpticalAmplitudes:
    """Normal-incidence x-input amplitudes and principal rotations.

    ``transmission_scattering`` and ``reflection_scattering`` identify the
    exact boundary solves used to obtain each field. They differ for the Okada
    mixed-pulse geometry. ``geometry`` and ``tensor_convention`` make saved
    results self-describing without introducing paper-specific plotting data.
    """

    reflected_x: np.ndarray
    reflected_y: np.ndarray
    transmitted_x: np.ndarray
    transmitted_y: np.ndarray
    faraday_angle_rad: np.ndarray
    kerr_angle_rad: np.ndarray
    geometry: str = "normal_incidence_same_side"
    tensor_convention: str = "physical_j_equals_sigma_e"
    transmission_scattering: NormalIncidenceSheetScattering | None = None
    reflection_scattering: NormalIncidenceSheetScattering | None = None

    def __post_init__(self) -> None:
        field_names = (
            "reflected_x",
            "reflected_y",
            "transmitted_x",
            "transmitted_y",
        )
        fields = {
            name: _readonly_array(getattr(self, name), dtype=np.complex128)
            for name in field_names
        }
        angle_names = ("faraday_angle_rad", "kerr_angle_rad")
        angles = {
            name: _readonly_array(getattr(self, name), dtype=float)
            for name in angle_names
        }
        if len({array.shape for array in fields.values()}) != 1:
            raise ValueError("all reflected/transmitted field components must share a shape")
        field_shape = fields["reflected_x"].shape
        if any(array.shape != field_shape for array in angles.values()):
            raise ValueError("stored Kerr/Faraday angles must match the field shape")
        derived_faraday = complex_polarization_from_jones(
            fields["transmitted_x"], fields["transmitted_y"], zero_policy="nan"
        ).rotation_angle_rad
        derived_kerr = complex_polarization_from_jones(
            fields["reflected_x"], fields["reflected_y"], zero_policy="nan"
        ).rotation_angle_rad
        for name, stored, derived in (
            ("faraday", angles["faraday_angle_rad"], derived_faraday),
            ("kerr", angles["kerr_angle_rad"], derived_kerr),
        ):
            if not np.array_equal(np.isfinite(stored), np.isfinite(derived)):
                raise ValueError(f"stored {name} angle validity disagrees with Jones fields")
            finite = np.isfinite(stored)
            if np.any(finite) and not np.allclose(
                np.exp(2.0j * stored[finite]),
                np.exp(2.0j * derived[finite]),
                rtol=1.0e-13,
                atol=1.0e-13,
            ):
                raise ValueError(f"stored {name} angle disagrees with Jones fields modulo pi")
        for name, array in {**fields, **angles}.items():
            object.__setattr__(self, name, array)
        if not self.geometry:
            raise ValueError("geometry metadata must not be empty")
        if not self.tensor_convention:
            raise ValueError("tensor_convention metadata must not be empty")

    @property
    def faraday_polarization(self) -> ComplexPolarization:
        """Exact transmitted polarization, including ellipticity and masks."""

        return complex_polarization_from_jones(
            self.transmitted_x,
            self.transmitted_y,
            zero_policy="nan",
        )

    @property
    def kerr_polarization(self) -> ComplexPolarization:
        """Exact reflected polarization, including ellipticity and masks."""

        return complex_polarization_from_jones(
            self.reflected_x,
            self.reflected_y,
            zero_policy="nan",
        )

    @property
    def faraday_ellipticity(self) -> np.ndarray:
        return self.faraday_polarization.ellipticity

    @property
    def kerr_ellipticity(self) -> np.ndarray:
        return self.kerr_polarization.ellipticity


@dataclass(frozen=True)
class SmallConductivityAngles:
    """Small-sheet-conductivity approximation to Kerr/Faraday rotations."""

    faraday_angle_rad: np.ndarray
    kerr_angle_rad: np.ndarray


def _validated_vacuum_impedance(vacuum_impedance_ohm: float) -> float:
    value = float(vacuum_impedance_ohm)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "vacuum_impedance_ohm must be finite and positive, "
            f"got {vacuum_impedance_ohm}"
        )
    return value


def _as_sheet_conductivity_2x2(sheet_conductivity: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sheet_conductivity, dtype=np.complex128)
    if sigma.ndim < 2 or sigma.shape[-2:] != (2, 2):
        raise ValueError(
            "sheet_conductivity must have final shape (2,2), "
            f"got {sigma.shape}"
        )
    if np.any(~np.isfinite(sigma)):
        raise ValueError("sheet_conductivity must contain only finite values")
    return sigma


def _broadcast_sheet_and_media(
    sheet_conductivity: np.ndarray,
    incident_refractive_index: complex | float | np.ndarray,
    transmitted_refractive_index: complex | float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = _as_sheet_conductivity_2x2(sheet_conductivity)
    n1 = np.asarray(incident_refractive_index, dtype=np.complex128)
    n2 = np.asarray(transmitted_refractive_index, dtype=np.complex128)
    if np.any(~np.isfinite(n1)) or np.any(~np.isfinite(n2)):
        raise ValueError("refractive indices must contain only finite values")
    try:
        batch_shape = np.broadcast_shapes(sigma.shape[:-2], n1.shape, n2.shape)
    except ValueError as error:
        raise ValueError(
            "sheet_conductivity and refractive-index leading shapes are not "
            f"broadcast-compatible: {sigma.shape[:-2]}, {n1.shape}, {n2.shape}"
        ) from error
    return (
        np.broadcast_to(sigma, batch_shape + (2, 2)),
        np.broadcast_to(n1, batch_shape),
        np.broadcast_to(n2, batch_shape),
    )


def sheet_scattering_at_normal_incidence(
    sheet_conductivity: np.ndarray,
    *,
    incident_refractive_index: complex | float | np.ndarray = 1.0,
    transmitted_refractive_index: complex | float | np.ndarray = 1.0,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
    incident_propagation_direction_in_sample: Literal[-1, 1] = 1,
) -> NormalIncidenceSheetScattering:
    r"""Return exact normal-incidence Jones reflection/transmission matrices.

    For medium admittances ``Y_i=n_i/Z0``, continuity of tangential electric
    field and the sheet-current boundary condition give

    .. math::

        [(Y_1+Y_2)I+\sigma]E_t = 2Y_1E_i,
        \qquad E_r=E_t-E_i.

    Therefore ``T=2Y1*[(Y1+Y2)I+sigma]^-1`` and ``R=T-I``. The solve is batched
    over every broadcast-compatible leading conductivity/medium axis.
    """

    z0 = _validated_vacuum_impedance(vacuum_impedance_ohm)
    sigma, n1, n2 = _broadcast_sheet_and_media(
        sheet_conductivity,
        incident_refractive_index,
        transmitted_refractive_index,
    )
    y1 = n1 / z0
    y2 = n2 / z0
    identity = np.broadcast_to(np.eye(2, dtype=np.complex128), sigma.shape)
    boundary_matrix = (y1 + y2)[..., None, None] * identity + sigma
    rhs = (2.0 * y1)[..., None, None] * identity
    try:
        transmission = np.linalg.solve(boundary_matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "normal-incidence sheet boundary matrix is singular"
        ) from error
    reflection = transmission - identity
    return NormalIncidenceSheetScattering(
        transmission_matrix=transmission,
        reflection_matrix=reflection,
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
        vacuum_impedance_ohm=z0,
        incident_propagation_direction_in_sample=(
            incident_propagation_direction_in_sample
        ),
    )


def complex_polarization_from_jones(
    field_x: np.ndarray | complex | float,
    field_y: np.ndarray | complex | float,
    *,
    zero_policy: Literal["raise", "nan"] = "raise",
) -> ComplexPolarization:
    """Recover principal rotation and signed ellipticity from Jones fields.

    Stokes quantities are evaluated after pointwise amplitude normalization, so
    the result is invariant under common complex scale and remains stable for
    extremely large or small finite fields. ``zero_policy='nan'`` preserves
    undefined spectral points for later segmented angle unwrapping.
    """

    if zero_policy not in ("raise", "nan"):
        raise ValueError("zero_policy must be 'raise' or 'nan'")
    ex, ey = np.broadcast_arrays(
        np.asarray(field_x, dtype=np.complex128),
        np.asarray(field_y, dtype=np.complex128),
    )
    if np.any(~np.isfinite(ex)) or np.any(~np.isfinite(ey)):
        raise ValueError("Jones fields must contain only finite values")
    scale = np.maximum(np.abs(ex), np.abs(ey))
    polarization_defined = scale > 0.0
    if zero_policy == "raise" and np.any(~polarization_defined):
        raise ValueError("the zero Jones vector has no defined polarization")
    safe_scale = np.where(polarization_defined, scale, 1.0)
    ex_normalized = ex / safe_scale
    ey_normalized = ey / safe_scale

    intensity = np.abs(ex_normalized) ** 2 + np.abs(ey_normalized) ** 2
    cross = np.conjugate(ex_normalized) * ey_normalized
    stokes_1 = np.abs(ex_normalized) ** 2 - np.abs(ey_normalized) ** 2
    stokes_2 = 2.0 * np.real(cross)
    stokes_3 = 2.0 * np.imag(cross)
    linear_magnitude = np.hypot(stokes_1, stokes_2)

    rotation = 0.5 * np.arctan2(stokes_2, stokes_1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ellipticity = stokes_3 / (intensity + linear_magnitude)
    ellipticity = np.clip(ellipticity, -1.0, 1.0)
    circular = polarization_defined & (
        linear_magnitude <= 64.0 * np.finfo(float).eps * intensity
    )
    orientation_defined = polarization_defined & ~circular
    rotation = np.where(orientation_defined, rotation, np.nan)
    ellipticity = np.where(polarization_defined, ellipticity, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ey / ex
        imaginary_angle = np.arctanh(ellipticity)
    angle = np.empty(rotation.shape, dtype=np.complex128)
    angle.real = rotation
    angle.imag = imaginary_angle
    return ComplexPolarization(
        complex_angle_rad=angle,
        rotation_angle_rad=rotation,
        ellipticity=ellipticity,
        jones_ratio=ratio,
        polarization_defined=polarization_defined,
        orientation_defined=orientation_defined,
    )


def unwrap_polarization_angle(
    principal_angle_rad: np.ndarray,
    *,
    axis: int = -1,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Unwrap a polarization orientation modulo pi along one ordered axis.

    Each contiguous valid segment is unwrapped independently as
    ``0.5*np.unwrap(2*theta)``. Undefined gaps remain ``nan`` and are never used
    to infer a branch connection. As for every discrete unwrap, adjacent valid
    samples must differ physically by less than ``pi/2``.
    """

    angle = np.asarray(principal_angle_rad, dtype=float)
    if angle.ndim == 0:
        is_valid = bool(np.isfinite(angle))
        if valid is not None:
            is_valid = is_valid and bool(np.asarray(valid, dtype=bool))
        return np.asarray(angle if is_valid else np.nan)
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += angle.ndim
    if normalized_axis < 0 or normalized_axis >= angle.ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for an array with {angle.ndim} dimensions"
        )
    if valid is None:
        valid_mask = np.isfinite(angle)
    else:
        try:
            valid_mask = np.broadcast_to(np.asarray(valid, dtype=bool), angle.shape)
        except ValueError as error:
            raise ValueError(
                f"valid mask shape is not broadcast-compatible with {angle.shape}"
            ) from error
        valid_mask = valid_mask & np.isfinite(angle)

    moved_angle = np.moveaxis(angle, normalized_axis, -1)
    moved_valid = np.moveaxis(valid_mask, normalized_axis, -1)
    output = np.full(moved_angle.shape, np.nan, dtype=float)
    for outer_index in np.ndindex(moved_angle.shape[:-1]):
        row = moved_angle[outer_index]
        row_valid = moved_valid[outer_index]
        index = 0
        while index < row.size:
            if not row_valid[index]:
                index += 1
                continue
            stop = index + 1
            while stop < row.size and row_valid[stop]:
                stop += 1
            output[outer_index + (slice(index, stop),)] = 0.5 * np.unwrap(
                2.0 * row[index:stop]
            )
            index = stop
    return np.moveaxis(output, -1, normalized_axis)


def _amplitudes_from_scatterings(
    transmission_scattering: NormalIncidenceSheetScattering,
    reflection_scattering: NormalIncidenceSheetScattering,
    *,
    geometry: str,
    tensor_convention: str = "physical_j_equals_sigma_e",
) -> MagnetoOpticalAmplitudes:
    transmitted = transmission_scattering.transmitted_x_incident
    reflected = reflection_scattering.reflected_x_incident
    faraday = complex_polarization_from_jones(
        transmitted[..., 0], transmitted[..., 1], zero_policy="nan"
    )
    kerr = complex_polarization_from_jones(
        reflected[..., 0], reflected[..., 1], zero_policy="nan"
    )
    return MagnetoOpticalAmplitudes(
        reflected_x=reflected[..., 0],
        reflected_y=reflected[..., 1],
        transmitted_x=transmitted[..., 0],
        transmitted_y=transmitted[..., 1],
        faraday_angle_rad=faraday.rotation_angle_rad,
        kerr_angle_rad=kerr.rotation_angle_rad,
        geometry=geometry,
        tensor_convention=tensor_convention,
        transmission_scattering=transmission_scattering,
        reflection_scattering=reflection_scattering,
    )


def faraday_kerr_from_sheet_conductivity(
    sheet_conductivity: np.ndarray,
    *,
    incident_refractive_index: complex | float | np.ndarray = 1.0,
    transmitted_refractive_index: complex | float | np.ndarray = 1.0,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> MagnetoOpticalAmplitudes:
    """Return exact same-incidence-side Kerr/Faraday response of a 2D sheet.

    Defaults reproduce a free-standing sheet in vacuum. More generally the
    incident wave approaches the sheet from medium 1, the transmitted wave
    exits into medium 2, and the reflected wave returns into medium 1.
    """

    scattering = sheet_scattering_at_normal_incidence(
        sheet_conductivity,
        incident_refractive_index=incident_refractive_index,
        transmitted_refractive_index=transmitted_refractive_index,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    return _amplitudes_from_scatterings(
        scattering,
        scattering,
        geometry="normal_incidence_same_side",
    )


def faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed(
    sheet_conductivity: np.ndarray,
    *,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> MagnetoOpticalAmplitudes:
    """Return the transverse sign printed in Liu-Dai 2020 Eqs. (21)-(23).

    This compatibility helper negates only the physical off-diagonal tensor
    entries before the vacuum Maxwell solve. It must not be used as a generic
    convention switch.
    """

    sigma = _as_sheet_conductivity_2x2(sheet_conductivity).copy()
    sigma[..., 0, 1] *= -1.0
    sigma[..., 1, 0] *= -1.0
    scattering = sheet_scattering_at_normal_incidence(
        sigma,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    return _amplitudes_from_scatterings(
        scattering,
        scattering,
        geometry="liu_dai_2020_printed_vacuum",
        tensor_convention="liu_dai_2020_printed_transverse_sign",
    )


def faraday_kerr_from_sheet_conductivity_on_substrate(
    sheet_conductivity: np.ndarray,
    *,
    substrate_refractive_index: complex | float | np.ndarray,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> MagnetoOpticalAmplitudes:
    """Return the Okada mixed-pulse substrate Kerr/Faraday geometry.

    Faraday transmission is vacuum to substrate. The delayed Kerr pulse is
    incident from the substrate and reflected back into it. This differs from
    ordinary vacuum-side reflection from a supported sheet.
    """

    transmission_scattering = sheet_scattering_at_normal_incidence(
        sheet_conductivity,
        incident_refractive_index=1.0,
        transmitted_refractive_index=substrate_refractive_index,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
    )
    reflection_scattering = sheet_scattering_at_normal_incidence(
        sheet_conductivity,
        incident_refractive_index=substrate_refractive_index,
        transmitted_refractive_index=1.0,
        vacuum_impedance_ohm=vacuum_impedance_ohm,
        incident_propagation_direction_in_sample=-1,
    )
    return _amplitudes_from_scatterings(
        transmission_scattering,
        reflection_scattering,
        geometry="okada_2016_mixed_pulse_substrate",
    )


def sheet_conductivity_from_scalar_normalized_transmission(
    normalized_field_transmission: np.ndarray | complex | float,
    *,
    incident_refractive_index: complex | float = 1.0,
    transmitted_refractive_index: complex | float,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> np.ndarray:
    """Invert a complex scalar field transmission normalized by the bare interface."""

    z0 = _validated_vacuum_impedance(vacuum_impedance_ohm)
    transmission = np.asarray(normalized_field_transmission, dtype=np.complex128)
    admittance_sum = (
        np.asarray(incident_refractive_index, dtype=np.complex128)
        + np.asarray(transmitted_refractive_index, dtype=np.complex128)
    ) / z0
    with np.errstate(divide="ignore", invalid="ignore"):
        return admittance_sum * (1.0 / transmission - 1.0)


def sheet_conductivity_from_normalized_transmission_matrix(
    normalized_field_transmission: np.ndarray,
    *,
    incident_refractive_index: complex | float = 1.0,
    transmitted_refractive_index: complex | float,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> np.ndarray:
    """Invert a complex 2x2 normalized Jones field-transmission matrix."""

    z0 = _validated_vacuum_impedance(vacuum_impedance_ohm)
    transmission = np.asarray(normalized_field_transmission, dtype=np.complex128)
    if transmission.ndim < 2 or transmission.shape[-2:] != (2, 2):
        raise ValueError(
            "normalized_field_transmission must have final shape (2,2), "
            f"got {transmission.shape}"
        )
    n1 = np.asarray(incident_refractive_index, dtype=np.complex128)
    n2 = np.asarray(transmitted_refractive_index, dtype=np.complex128)
    try:
        batch_shape = np.broadcast_shapes(transmission.shape[:-2], n1.shape, n2.shape)
    except ValueError as error:
        raise ValueError(
            "normalized transmission and refractive-index shapes are not "
            "broadcast-compatible"
        ) from error
    transmission_b = np.broadcast_to(transmission, batch_shape + (2, 2))
    admittance_sum = np.broadcast_to((n1 + n2) / z0, batch_shape)
    identity = np.broadcast_to(
        np.eye(2, dtype=np.complex128), batch_shape + (2, 2)
    )
    inverse = np.linalg.solve(transmission_b, identity)
    return admittance_sum[..., None, None] * (inverse - identity)


def faraday_kerr_small_conductivity(
    sheet_conductivity: np.ndarray,
    *,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> SmallConductivityAngles:
    """Physical weak-sheet approximation for a free-standing vacuum sheet."""

    sigma = _as_sheet_conductivity_2x2(sheet_conductivity)
    y2 = 2.0 / _validated_vacuum_impedance(vacuum_impedance_ohm)
    sxx = sigma[..., 0, 0]
    syx = sigma[..., 1, 0]
    faraday = np.real(np.arctan(-syx / y2))
    with np.errstate(divide="ignore", invalid="ignore"):
        kerr = np.real(np.arctan(syx / sxx))
    return SmallConductivityAngles(
        faraday_angle_rad=faraday,
        kerr_angle_rad=kerr,
    )


__all__ = [
    "ComplexPolarization",
    "MagnetoOpticalAmplitudes",
    "NormalIncidenceSheetScattering",
    "PHASOR_CONVENTION",
    "REFLECTED_FIELD_CONVENTION",
    "SAMPLE_AXES_CONVENTION",
    "SmallConductivityAngles",
    "VACUUM_ADMITTANCE_S",
    "VACUUM_IMPEDANCE_OHM",
    "VACUUM_PERMEABILITY_H_PER_M",
    "VACUUM_PERMITTIVITY_F_PER_M",
    "complex_polarization_from_jones",
    "faraday_kerr_from_sheet_conductivity",
    "faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed",
    "faraday_kerr_from_sheet_conductivity_on_substrate",
    "faraday_kerr_small_conductivity",
    "sheet_conductivity_from_normalized_transmission_matrix",
    "sheet_conductivity_from_scalar_normalized_transmission",
    "sheet_scattering_at_normal_incidence",
    "unwrap_polarization_angle",
]
