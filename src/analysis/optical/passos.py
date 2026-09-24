from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement, permutations, product
from math import factorial
import operator
from typing import Iterable, Literal, Sequence

import numpy as np

from analysis.optical.linear import E2_OVER_HBAR_S
from analysis.optical.magneto import VACUUM_PERMITTIVITY_F_PER_M
from analysis.response_derivative_gauge import matrix_in_eigenbasis
from analysis.shift_current import HBAR_EV_S, fermi_occupation


DerivativeProvenance = Literal[
    "fixed_basis_complete",
    "externally_certified_covariant",
]
OccupationKind = Literal["fermi_dirac", "stationary_custom"]


def _integer_index(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _kahan_add(
    total: np.ndarray,
    compensation: np.ndarray,
    value: np.ndarray,
) -> None:
    corrected = value - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


@dataclass(frozen=True, kw_only=True, eq=False)
class PassosPrimitiveBZGrid:
    """Midpoint quadrature of one primitive full BZ with optional refinement.

    ``reciprocal_vectors`` contains primitive reciprocal vectors as rows in a
    Cartesian frame. Selected base cells may be replaced exactly by equal
    refined subcells; replacement preserves the full primitive-cell measure.
    Derivative axes use the same Cartesian frame, and ``(2*pi)^-d`` is not
    included in point weights.
    """

    reciprocal_vectors: np.ndarray
    mesh_shape: tuple[int, ...]
    coordinate_length_unit_m: float
    grid_label: str
    refined_cell_indices: tuple[int, ...] = ()
    refinement_factor: int = 1
    _unrefined_cell_indices: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        vectors = np.array(self.reciprocal_vectors, dtype=float, copy=True)
        shape = tuple(
            _integer_index(value, name="mesh_shape entry")
            for value in self.mesh_shape
        )
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("mesh_shape entries must be positive")
        dimension = len(shape)
        if vectors.shape != (dimension, dimension):
            raise ValueError(
                f"reciprocal_vectors must have shape {(dimension, dimension)}"
            )
        if np.any(~np.isfinite(vectors)):
            raise ValueError("reciprocal_vectors must be finite")
        determinant = float(abs(np.linalg.det(vectors)))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise ValueError("reciprocal_vectors must span a nonzero primitive cell")
        length = float(self.coordinate_length_unit_m)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("coordinate_length_unit_m must be finite and positive")
        label = str(self.grid_label).strip()
        if not label:
            raise ValueError("grid_label must not be empty")
        factor = _integer_index(self.refinement_factor, name="refinement_factor")
        if factor <= 0:
            raise ValueError("refinement_factor must be positive")
        base_count = int(np.prod(shape, dtype=np.int64))
        refined = tuple(
            sorted(
                _integer_index(value, name="refined_cell_indices entry")
                for value in self.refined_cell_indices
            )
        )
        if len(set(refined)) != len(refined):
            raise ValueError("refined_cell_indices must be unique")
        if any(value < 0 or value >= base_count for value in refined):
            raise ValueError("refined_cell_indices contains an out-of-range cell")
        if refined and factor == 1:
            raise ValueError(
                "nonempty refined_cell_indices requires refinement_factor > 1"
            )
        refined_set = set(refined)
        unrefined = tuple(
            index for index in range(base_count) if index not in refined_set
        )
        vectors.setflags(write=False)
        object.__setattr__(self, "reciprocal_vectors", vectors)
        object.__setattr__(self, "mesh_shape", shape)
        object.__setattr__(self, "coordinate_length_unit_m", length)
        object.__setattr__(self, "grid_label", label)
        object.__setattr__(self, "refinement_factor", factor)
        object.__setattr__(self, "refined_cell_indices", refined)
        object.__setattr__(self, "_unrefined_cell_indices", unrefined)

    @property
    def integration_dimension(self) -> int:
        return len(self.mesh_shape)

    @property
    def base_number_of_cells(self) -> int:
        return int(np.prod(self.mesh_shape, dtype=np.int64))

    @property
    def refined_subcells_per_cell(self) -> int:
        return self.refinement_factor**self.integration_dimension

    @property
    def number_of_points(self) -> int:
        return len(self._unrefined_cell_indices) + (
            len(self.refined_cell_indices) * self.refined_subcells_per_cell
        )

    @property
    def full_bz_measure(self) -> float:
        return float(abs(np.linalg.det(self.reciprocal_vectors)))

    @property
    def base_cell_weight(self) -> float:
        return self.full_bz_measure / self.base_number_of_cells

    @property
    def point_weight(self) -> float:
        if self.refined_cell_indices:
            raise ValueError("refined grids have index-dependent point weights")
        return self.base_cell_weight

    def _point_descriptor(self, grid_index: int) -> tuple[int, int | None]:
        index = _integer_index(grid_index, name="grid_index")
        if index < 0 or index >= self.number_of_points:
            raise IndexError(
                f"grid_index {index} is outside [0,{self.number_of_points})"
            )
        unrefined_count = len(self._unrefined_cell_indices)
        if index < unrefined_count:
            return self._unrefined_cell_indices[index], None
        refined_offset = index - unrefined_count
        refined_position, subcell_index = divmod(
            refined_offset,
            self.refined_subcells_per_cell,
        )
        return self.refined_cell_indices[refined_position], subcell_index

    def point_weight_for_index(self, grid_index: int) -> float:
        _, subcell_index = self._point_descriptor(grid_index)
        if subcell_index is None:
            return self.base_cell_weight
        return self.base_cell_weight / self.refined_subcells_per_cell

    def cartesian_point(self, grid_index: int) -> np.ndarray:
        """Return one base/refined midpoint in reciprocal coordinate units."""

        base_index, subcell_index = self._point_descriptor(grid_index)
        base_multi = np.asarray(
            np.unravel_index(base_index, self.mesh_shape),
            dtype=float,
        )
        if subcell_index is None:
            fractional_offset = np.full(self.integration_dimension, 0.5)
        else:
            subcell_multi = np.asarray(
                np.unravel_index(
                    subcell_index,
                    (self.refinement_factor,) * self.integration_dimension,
                ),
                dtype=float,
            )
            fractional_offset = (
                subcell_multi + 0.5
            ) / self.refinement_factor
        fractional = (
            base_multi + fractional_offset
        ) / np.asarray(self.mesh_shape, dtype=float)
        point = fractional @ self.reciprocal_vectors
        point.setflags(write=False)
        return point

    def compatible_with(self, other: object) -> bool:
        return (
            isinstance(other, PassosPrimitiveBZGrid)
            and self.mesh_shape == other.mesh_shape
            and self.coordinate_length_unit_m == other.coordinate_length_unit_m
            and self.grid_label == other.grid_label
            and self.refined_cell_indices == other.refined_cell_indices
            and self.refinement_factor == other.refinement_factor
            and np.array_equal(self.reciprocal_vectors, other.reciprocal_vectors)
        )


@dataclass(frozen=True, kw_only=True)
class PassosModelCertificate:
    """Immutable declaration binding a derivative tower to one finite model."""

    model_id: str
    source_id: str
    basis_dimension: int
    maximum_derivative_order: int
    response_axis_labels: tuple[str, ...]
    derivative_provenance: DerivativeProvenance

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        source_id = str(self.source_id).strip()
        basis_dimension = _integer_index(
            self.basis_dimension,
            name="basis_dimension",
        )
        maximum_order = _integer_index(
            self.maximum_derivative_order,
            name="maximum_derivative_order",
        )
        if not model_id or not source_id:
            raise ValueError("model_id and source_id must not be empty")
        if basis_dimension <= 0 or maximum_order <= 0:
            raise ValueError("basis_dimension and maximum_derivative_order must be positive")
        axis_labels = tuple(str(label).strip() for label in self.response_axis_labels)
        if not axis_labels or any(not label for label in axis_labels):
            raise ValueError("response_axis_labels must contain nonempty labels")
        if len(set(axis_labels)) != len(axis_labels):
            raise ValueError("response_axis_labels must be unique")
        if self.derivative_provenance not in {
            "fixed_basis_complete",
            "externally_certified_covariant",
        }:
            raise ValueError("unsupported derivative_provenance")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "basis_dimension", basis_dimension)
        object.__setattr__(self, "maximum_derivative_order", maximum_order)
        object.__setattr__(self, "response_axis_labels", axis_labels)

    @property
    def basis_connection_contract(self) -> Literal[
        "zero_connection_k_independent_orthonormal",
        "externally_supplied_covariant_tower",
    ]:
        if self.derivative_provenance == "fixed_basis_complete":
            return "zero_connection_k_independent_orthonormal"
        return "externally_supplied_covariant_tower"


@dataclass(frozen=True, kw_only=True)
class PassosOccupationSpec:
    """Common stationary occupation provenance for a complete k grid."""

    kind: OccupationKind
    chemical_potential_ev: float = 0.0
    temperature_k: float = 0.0
    state_id: str = ""

    def __post_init__(self) -> None:
        chemical_potential = float(self.chemical_potential_ev)
        temperature = float(self.temperature_k)
        if not np.isfinite(chemical_potential):
            raise ValueError("chemical_potential_ev must be finite")
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature_k must be finite and non-negative")
        state_id = str(self.state_id).strip()
        if self.kind == "fermi_dirac":
            if state_id:
                raise ValueError("fermi_dirac occupation must not set state_id")
        elif self.kind == "stationary_custom":
            if not state_id:
                raise ValueError("stationary_custom occupation requires state_id")
        else:
            raise ValueError("unsupported occupation kind")
        object.__setattr__(self, "chemical_potential_ev", chemical_potential)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "state_id", state_id)

    @classmethod
    def fermi_dirac(
        cls,
        *,
        chemical_potential_ev: float,
        temperature_k: float,
    ) -> PassosOccupationSpec:
        return cls(
            kind="fermi_dirac",
            chemical_potential_ev=chemical_potential_ev,
            temperature_k=temperature_k,
        )

    @classmethod
    def stationary_custom(cls, *, state_id: str) -> PassosOccupationSpec:
        return cls(kind="stationary_custom", state_id=state_id)

    def values_for_energies(self, energies_ev: np.ndarray) -> np.ndarray:
        if self.kind != "fermi_dirac":
            raise ValueError("custom stationary occupations must be supplied explicitly")
        return fermi_occupation(
            np.asarray(energies_ev, dtype=float),
            mu_ev=self.chemical_potential_ev,
            temperature_k=self.temperature_k,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class PassosKPointData:
    """One certified complete-model point for Passos Eq. (13) recursion."""

    energies_ev: np.ndarray
    occupations: np.ndarray
    covariant_derivatives_ev: tuple[np.ndarray, ...]
    grid: PassosPrimitiveBZGrid
    grid_index: int
    model_certificate: PassosModelCertificate
    occupation_spec: PassosOccupationSpec
    degeneracy_tolerance_ev: float = 1.0e-10

    def __post_init__(self) -> None:
        energies = np.array(self.energies_ev, dtype=float, copy=True)
        occupations = np.array(self.occupations, dtype=float, copy=True)
        if energies.ndim != 1 or energies.size == 0 or np.any(~np.isfinite(energies)):
            raise ValueError("energies_ev must be a nonempty finite 1D array")
        if occupations.shape != energies.shape or np.any(~np.isfinite(occupations)):
            raise ValueError("occupations must be finite and match energies_ev")
        if np.any((occupations < 0.0) | (occupations > 1.0)):
            raise ValueError("occupations must lie in [0,1]")
        if not isinstance(self.grid, PassosPrimitiveBZGrid):
            raise TypeError("grid must be PassosPrimitiveBZGrid")
        grid_index = _integer_index(self.grid_index, name="grid_index")
        self.grid.cartesian_point(grid_index)
        if not isinstance(self.model_certificate, PassosModelCertificate):
            raise TypeError("model_certificate must be PassosModelCertificate")
        if energies.size != self.model_certificate.basis_dimension:
            raise ValueError(
                "payload band count must equal the certified complete basis dimension"
            )
        if not isinstance(self.occupation_spec, PassosOccupationSpec):
            raise TypeError("occupation_spec must be PassosOccupationSpec")
        if self.occupation_spec.kind == "fermi_dirac":
            expected_occupations = self.occupation_spec.values_for_energies(energies)
            if not np.allclose(
                occupations,
                expected_occupations,
                rtol=0.0,
                atol=1.0e-13,
            ):
                raise ValueError("occupations disagree with the common Fermi-Dirac spec")
        degeneracy_tolerance = float(self.degeneracy_tolerance_ev)
        if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
            raise ValueError("degeneracy_tolerance_ev must be finite and non-negative")
        degenerate = np.abs(energies[:, None] - energies[None, :]) <= degeneracy_tolerance
        if np.any(
            degenerate
            & (np.abs(occupations[:, None] - occupations[None, :]) > 1.0e-12)
        ):
            raise ValueError(
                "equilibrium occupations must be equal inside degenerate subspaces"
            )
        derivatives = _validated_derivative_tower(
            self.covariant_derivatives_ev,
            number_of_bands=energies.size,
        )
        if len(derivatives) != self.model_certificate.maximum_derivative_order:
            raise ValueError("derivative tower does not match the model certificate")
        if derivatives[0].shape[0] != len(
            self.model_certificate.response_axis_labels
        ):
            raise ValueError("derivative axes do not match the model certificate")
        energies.setflags(write=False)
        occupations.setflags(write=False)
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "covariant_derivatives_ev", derivatives)
        object.__setattr__(self, "grid_index", grid_index)
        object.__setattr__(self, "degeneracy_tolerance_ev", degeneracy_tolerance)

    @property
    def number_of_axes(self) -> int:
        return int(self.covariant_derivatives_ev[0].shape[0])

    @property
    def number_of_bands(self) -> int:
        return int(self.energies_ev.size)

    @property
    def maximum_order(self) -> int:
        return len(self.covariant_derivatives_ev)

    @property
    def weight(self) -> float:
        return self.grid.point_weight_for_index(self.grid_index)

    @property
    def k_coordinate(self) -> np.ndarray:
        return self.grid.cartesian_point(self.grid_index)


@dataclass(frozen=True, kw_only=True, eq=False)
class PassosHarmonicConductivityResult:
    """Declared complete-model, exact-full-grid symmetric conductivity."""

    order: int
    photon_energies_ev: np.ndarray
    conductivity: np.ndarray
    eq32_sector_conductivity: np.ndarray
    absolute_contribution_sum: np.ndarray
    adiabatic_gamma_ev: float
    grid: PassosPrimitiveBZGrid
    model_certificate: PassosModelCertificate
    occupation_spec: PassosOccupationSpec
    kpoint_count: int
    intrinsic_permutation_symmetrized: Literal[True] = True

    def __post_init__(self) -> None:
        order = _integer_index(self.order, name="order")
        omega = np.array(self.photon_energies_ev, dtype=float, copy=True)
        conductivity = np.array(self.conductivity, dtype=np.complex128, copy=True)
        sectors = np.array(self.eq32_sector_conductivity, dtype=np.complex128, copy=True)
        absolute_sum = np.array(self.absolute_contribution_sum, dtype=float, copy=True)
        if order <= 0:
            raise ValueError("order must be positive")
        if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
            raise ValueError("photon_energies_ev must be a nonempty finite 1D array")
        if np.any(omega <= 0.0):
            raise ValueError("photon_energies_ev must be strictly positive")
        if not isinstance(self.grid, PassosPrimitiveBZGrid):
            raise TypeError("grid must be PassosPrimitiveBZGrid")
        if not isinstance(self.model_certificate, PassosModelCertificate):
            raise TypeError("model_certificate must be PassosModelCertificate")
        if not isinstance(self.occupation_spec, PassosOccupationSpec):
            raise TypeError("occupation_spec must be PassosOccupationSpec")
        if self.model_certificate.maximum_derivative_order < order + 1:
            raise ValueError("model certificate lacks the derivative order required")
        naxis = len(self.model_certificate.response_axis_labels)
        expected_shape = (omega.size,) + (naxis,) * (order + 1)
        if conductivity.shape != expected_shape:
            raise ValueError("conductivity shape is incompatible with response order")
        if sectors.shape != (order + 1,) + conductivity.shape:
            raise ValueError("eq32_sector_conductivity has the wrong shape")
        if absolute_sum.shape != conductivity.shape or np.any(~np.isfinite(absolute_sum)):
            raise ValueError("absolute_contribution_sum must be finite and match conductivity")
        if np.any(~np.isfinite(conductivity)) or np.any(~np.isfinite(sectors)):
            raise ValueError("conductivity and Eq.32 sectors must be finite")
        sector_difference = np.abs(np.sum(sectors, axis=0) - conductivity)
        sector_tolerance = 2.0e-12 * np.maximum(absolute_sum, 1.0e-300)
        if np.any(sector_difference > sector_tolerance):
            raise ValueError("Eq.32 sectors do not sum to conductivity")
        triangle_tolerance = 1.0e-12 * np.maximum(absolute_sum, 1.0e-300)
        if np.any(absolute_sum + triangle_tolerance < np.abs(conductivity)):
            raise ValueError("absolute_contribution_sum cannot be smaller than |conductivity|")
        if isinstance(self.adiabatic_gamma_ev, (bool, np.bool_)):
            raise TypeError("adiabatic_gamma_ev must be a real scalar, not bool")
        gamma = float(self.adiabatic_gamma_ev)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("adiabatic_gamma_ev must be finite and positive")
        count = _integer_index(self.kpoint_count, name="kpoint_count")
        if count != self.grid.number_of_points:
            raise ValueError("kpoint_count must equal the certified full-BZ grid size")
        if self.intrinsic_permutation_symmetrized is not True:
            raise ValueError("physical Passos results must be permutation symmetrized")
        omega.setflags(write=False)
        conductivity.setflags(write=False)
        sectors.setflags(write=False)
        absolute_sum.setflags(write=False)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "photon_energies_ev", omega)
        object.__setattr__(self, "conductivity", conductivity)
        object.__setattr__(self, "eq32_sector_conductivity", sectors)
        object.__setattr__(self, "absolute_contribution_sum", absolute_sum)
        object.__setattr__(self, "adiabatic_gamma_ev", gamma)
        object.__setattr__(self, "kpoint_count", count)

    @property
    def response_name(self) -> str:
        if self.order == 2:
            return "second_harmonic_generation"
        if self.order == 3:
            return "third_harmonic_generation"
        return f"order_{self.order}_harmonic_generation"

    @property
    def gauge_kind(self) -> Literal["passos_finite_band_velocity_gauge"]:
        return "passos_finite_band_velocity_gauge"

    @property
    def causal_prescription_kind(self) -> Literal["independent_input_adiabatic"]:
        return "independent_input_adiabatic"

    @property
    def basis_connection_contract(self) -> str:
        return self.model_certificate.basis_connection_contract

    @property
    def cancellation_ratio(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.absolute_contribution_sum > 0.0,
                np.abs(self.conductivity) / self.absolute_contribution_sum,
                0.0,
            )

    @property
    def numerical_resolution_limited(self) -> np.ndarray:
        return self.cancellation_ratio < 256.0 * np.finfo(float).eps

    @property
    def si_unit(self) -> str:
        if self.grid.integration_dimension == 2:
            if self.order == 1:
                return "S"
            if self.order == 2:
                return "A m / V^2"
            if self.order == 3:
                return "A m^2 / V^3"
        return (
            "SI coefficient derived from q^(n+1)/hbar with coordinate-length "
            f"power {self.order + 1 - self.grid.integration_dimension}"
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class PassosSHGSusceptibilityResult:
    """Sheet SHG susceptibility and optional effective-bulk conversion."""

    photon_energies_ev: np.ndarray
    sheet_susceptibility_m2_per_v: np.ndarray
    effective_thickness_m: float | None = None
    effective_bulk_susceptibility_m_per_v: np.ndarray | None = None

    def __post_init__(self) -> None:
        photon = np.array(self.photon_energies_ev, dtype=float, copy=True)
        sheet = np.array(
            self.sheet_susceptibility_m2_per_v,
            dtype=np.complex128,
            copy=True,
        )
        if photon.ndim != 1 or photon.size == 0 or np.any(photon <= 0.0):
            raise ValueError("photon_energies_ev must be a nonempty positive 1D array")
        if sheet.shape[0] != photon.size or np.any(~np.isfinite(sheet)):
            raise ValueError("sheet susceptibility must be finite on the photon grid")
        thickness = self.effective_thickness_m
        bulk = self.effective_bulk_susceptibility_m_per_v
        if thickness is None:
            if bulk is not None:
                raise ValueError("effective bulk susceptibility requires a thickness")
            bulk_array = None
        else:
            thickness = float(thickness)
            if not np.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("effective_thickness_m must be finite and positive")
            if bulk is None:
                raise ValueError("effective thickness requires bulk susceptibility")
            bulk_array = np.array(bulk, dtype=np.complex128, copy=True)
            if bulk_array.shape != sheet.shape or np.any(~np.isfinite(bulk_array)):
                raise ValueError("effective bulk susceptibility must match sheet shape")
            bulk_array.setflags(write=False)
        photon.setflags(write=False)
        sheet.setflags(write=False)
        object.__setattr__(self, "photon_energies_ev", photon)
        object.__setattr__(self, "sheet_susceptibility_m2_per_v", sheet)
        object.__setattr__(self, "effective_thickness_m", thickness)
        object.__setattr__(self, "effective_bulk_susceptibility_m_per_v", bulk_array)


def _validated_derivative_tower(
    derivatives: Sequence[np.ndarray],
    *,
    number_of_bands: int,
) -> tuple[np.ndarray, ...]:
    tower = tuple(np.array(value, dtype=np.complex128, copy=True) for value in derivatives)
    if not tower:
        raise ValueError("at least the first covariant Hamiltonian derivative is required")
    nb = int(number_of_bands)
    naxis = tower[0].shape[0] if tower[0].ndim >= 3 else 0
    if naxis <= 0:
        raise ValueError("first derivative must have shape (naxis,nb,nb)")
    for order, derivative in enumerate(tower, start=1):
        expected = (naxis,) * order + (nb, nb)
        if derivative.shape != expected:
            raise ValueError(
                f"order-{order} derivative has shape {derivative.shape}, expected {expected}"
            )
        if np.any(~np.isfinite(derivative)):
            raise ValueError(f"order-{order} derivative contains nonfinite values")
        matrices = derivative.reshape((-1, nb, nb))
        if not np.allclose(
            matrices,
            np.swapaxes(matrices.conjugate(), -1, -2),
            rtol=1.0e-10,
            atol=1.0e-11,
        ):
            raise ValueError(
                f"order-{order} covariant Hamiltonian derivatives must be Hermitian"
            )
        derivative.setflags(write=False)
    return tower


def passos_kpoint_data_from_fixed_basis_derivatives(
    basis_hamiltonian_ev: np.ndarray,
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    basis_derivatives_ev: Sequence[np.ndarray],
    *,
    grid: PassosPrimitiveBZGrid,
    grid_index: int,
    model_certificate: PassosModelCertificate,
    occupation_spec: PassosOccupationSpec,
    fixed_basis_connection_is_zero: Literal[True],
    occupations: np.ndarray | None = None,
    degeneracy_tolerance_ev: float = 1.0e-10,
) -> PassosKPointData:
    """Build a point from a complete k-independent orthonormal zero-connection basis."""

    if not isinstance(grid, PassosPrimitiveBZGrid):
        raise TypeError("grid must be PassosPrimitiveBZGrid")
    if not isinstance(model_certificate, PassosModelCertificate):
        raise TypeError("model_certificate must be PassosModelCertificate")
    if not isinstance(occupation_spec, PassosOccupationSpec):
        raise TypeError("occupation_spec must be PassosOccupationSpec")
    if fixed_basis_connection_is_zero is not True:
        raise ValueError(
            "fixed-basis construction requires an explicit zero basis connection; "
            "supply a separately derived covariant tower for moving/embedded bases"
        )
    hamiltonian = np.asarray(basis_hamiltonian_ev, dtype=np.complex128)
    energies = np.asarray(energies_ev, dtype=float)
    eigenvectors_array = np.asarray(eigenvectors, dtype=np.complex128)
    if model_certificate.derivative_provenance != "fixed_basis_complete":
        raise ValueError("fixed-basis constructor requires fixed_basis_complete certificate")
    if hamiltonian.shape != (model_certificate.basis_dimension,) * 2:
        raise ValueError("basis_hamiltonian_ev shape disagrees with model certificate")
    if not np.allclose(hamiltonian, hamiltonian.conjugate().T, rtol=1e-11, atol=1e-12):
        raise ValueError("basis_hamiltonian_ev must be Hermitian")
    if eigenvectors_array.shape != hamiltonian.shape or energies.size != hamiltonian.shape[0]:
        raise ValueError("eigensystem must be square and complete")
    identity = np.eye(energies.size, dtype=np.complex128)
    if not np.allclose(
        eigenvectors_array.conjugate().T @ eigenvectors_array,
        identity,
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise ValueError("eigenvectors must be unitary")
    diagonalized = eigenvectors_array.conjugate().T @ hamiltonian @ eigenvectors_array
    if not np.allclose(
        diagonalized,
        np.diag(energies),
        rtol=1.0e-10,
        atol=1.0e-11,
    ):
        raise ValueError("eigenpairs do not diagonalize basis_hamiltonian_ev")
    raw_derivatives = tuple(np.asarray(value, dtype=np.complex128) for value in basis_derivatives_ev)
    if len(raw_derivatives) != model_certificate.maximum_derivative_order:
        raise ValueError("basis derivative tower disagrees with model certificate")
    naxis = raw_derivatives[0].shape[0] if raw_derivatives[0].ndim >= 3 else 0
    for derivative_order, derivative in enumerate(raw_derivatives, start=1):
        expected = (naxis,) * derivative_order + hamiltonian.shape
        if derivative.shape != expected:
            raise ValueError(
                f"order-{derivative_order} basis derivative has shape "
                f"{derivative.shape}, expected {expected}"
            )
    covariant_derivatives = tuple(
        matrix_in_eigenbasis(eigenvectors_array, derivative)
        for derivative in raw_derivatives
    )
    if occupation_spec.kind == "fermi_dirac":
        occupation_array = occupation_spec.values_for_energies(energies)
        if occupations is not None and not np.allclose(
            occupations,
            occupation_array,
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError("supplied occupations disagree with occupation_spec")
    else:
        if occupations is None:
            raise ValueError("stationary_custom occupation requires explicit values")
        occupation_array = np.asarray(occupations, dtype=float)
    return PassosKPointData(
        energies_ev=energies,
        occupations=occupation_array,
        covariant_derivatives_ev=covariant_derivatives,
        grid=grid,
        grid_index=grid_index,
        model_certificate=model_certificate,
        occupation_spec=occupation_spec,
        degeneracy_tolerance_ev=degeneracy_tolerance_ev,
    )


def _commutator_with_frequency(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.ndim == 2:
        return left @ right - right @ left
    return np.einsum("ij,wjk->wik", left, right, optimize=True) - np.einsum(
        "wij,jk->wik", right, left, optimize=True
    )


def _passos_harmonic_kernel_evaluation(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    occupations: np.ndarray,
    covariant_derivatives_ev: Sequence[np.ndarray],
    *,
    order: int,
    adiabatic_gamma_ev: float,
    intrinsic_permutation: bool,
    degeneracy_tolerance_ev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    harmonic_order = _integer_index(order, name="order")
    if harmonic_order <= 0:
        raise ValueError("order must be positive")
    omega = np.asarray(photon_energies_ev, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("photon_energies_ev must be a nonempty finite 1D array")
    if np.any(omega <= 0.0):
        raise ValueError("photon_energies_ev must be strictly positive")
    if isinstance(adiabatic_gamma_ev, (bool, np.bool_)):
        raise TypeError("adiabatic_gamma_ev must be a real scalar, not bool")
    gamma = float(adiabatic_gamma_ev)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("adiabatic_gamma_ev must be finite and positive")
    degeneracy_tolerance = float(degeneracy_tolerance_ev)
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance_ev must be finite and non-negative")
    energies = np.asarray(energies_ev, dtype=float)
    occupation = np.asarray(occupations, dtype=float)
    if (
        energies.ndim != 1
        or occupation.shape != energies.shape
        or np.any(~np.isfinite(energies))
        or np.any(~np.isfinite(occupation))
    ):
        raise ValueError("energies_ev and occupations must be matching finite 1D arrays")
    if np.any((occupation < 0.0) | (occupation > 1.0)):
        raise ValueError("occupations must lie in [0,1]")
    degenerate = np.abs(energies[:, None] - energies[None, :]) <= degeneracy_tolerance
    if np.any(
        degenerate
        & (np.abs(occupation[:, None] - occupation[None, :]) > 1.0e-12)
    ):
        raise ValueError("occupations must be equal inside degenerate subspaces")
    derivatives = _validated_derivative_tower(
        covariant_derivatives_ev,
        number_of_bands=energies.size,
    )
    if len(derivatives) < harmonic_order + 1:
        raise ValueError(
            f"order-{harmonic_order} response requires derivative orders 1 "
            f"through {harmonic_order + 1}"
        )
    nb = energies.size
    naxis = derivatives[0].shape[0]
    rho0 = np.diag(occupation.astype(np.complex128))
    energy_difference = energies[:, None] - energies[None, :]
    z = omega.astype(np.complex128) + 1.0j * gamma
    density_cache: dict[tuple[int, ...], np.ndarray] = {(): rho0}

    def density(axes: tuple[int, ...]) -> np.ndarray:
        cached = density_cache.get(axes)
        if cached is not None:
            return cached
        response_order = len(axes)
        numerator = np.zeros((omega.size, nb, nb), dtype=np.complex128)
        compensation = np.zeros_like(numerator)
        for prefix_length in range(1, response_order + 1):
            vertex = derivatives[prefix_length - 1][axes[:prefix_length]]
            tail = density(axes[prefix_length:])
            commutator = _commutator_with_frequency(vertex, tail)
            if commutator.ndim == 2:
                commutator = np.broadcast_to(commutator, (omega.size, nb, nb))
            _kahan_add(
                numerator,
                compensation,
                commutator / factorial(prefix_length),
            )
        denominator = response_order * z[:, None, None] - energy_difference[None]
        value = numerator / denominator
        density_cache[axes] = value
        return value

    ordered_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def ordered_sectors(output_axis: int, input_axes: tuple[int, ...]) -> np.ndarray:
        key = (output_axis, input_axes)
        cached = ordered_cache.get(key)
        if cached is not None:
            return cached
        sectors = np.empty(
            (harmonic_order + 1, omega.size),
            dtype=np.complex128,
        )
        for prefix_length in range(harmonic_order + 1):
            vertex_axes = (output_axis,) + input_axes[:prefix_length]
            vertex = derivatives[prefix_length][vertex_axes]
            tail = density(input_axes[prefix_length:])
            if tail.ndim == 2:
                trace = np.einsum("ij,ji->", vertex, tail, optimize=True)
            else:
                trace = np.einsum("ij,wji->w", vertex, tail, optimize=True)
            sectors[prefix_length] = (
                trace / factorial(prefix_length) / z**harmonic_order
            )
        ordered_cache[key] = sectors
        return sectors

    response_shape = (omega.size,) + (naxis,) * (harmonic_order + 1)
    sector_result = np.zeros(
        (harmonic_order + 1,) + response_shape,
        dtype=np.complex128,
    )
    branch_absolute_result = np.zeros(response_shape, dtype=float)
    if intrinsic_permutation:
        for output_axis in range(naxis):
            for canonical_axes in combinations_with_replacement(
                range(naxis),
                harmonic_order,
            ):
                permutation_list = list(permutations(canonical_axes))
                permutation_sum = np.zeros(
                    (harmonic_order + 1, omega.size),
                    dtype=np.complex128,
                )
                permutation_compensation = np.zeros_like(permutation_sum)
                permutation_absolute_sum = np.zeros(
                    (harmonic_order + 1, omega.size),
                    dtype=float,
                )
                permutation_absolute_compensation = np.zeros_like(
                    permutation_absolute_sum
                )
                for input_axes in permutation_list:
                    ordered = ordered_sectors(output_axis, tuple(input_axes))
                    _kahan_add(
                        permutation_sum,
                        permutation_compensation,
                        ordered,
                    )
                    _kahan_add(
                        permutation_absolute_sum,
                        permutation_absolute_compensation,
                        np.abs(ordered),
                    )
                component_sectors = permutation_sum / factorial(harmonic_order)
                component_branch_absolute = np.sum(
                    permutation_absolute_sum,
                    axis=0,
                ) / factorial(harmonic_order)
                for input_axes in set(permutation_list):
                    component_index = (slice(None), output_axis) + tuple(input_axes)
                    sector_result[(slice(None),) + component_index] = component_sectors
                    branch_absolute_result[component_index] = component_branch_absolute
    else:
        for output_axis in range(naxis):
            for input_axes in product(range(naxis), repeat=harmonic_order):
                component_sectors = ordered_sectors(output_axis, tuple(input_axes))
                component_index = (slice(None), output_axis) + tuple(input_axes)
                sector_result[(slice(None),) + component_index] = component_sectors
                branch_absolute_result[component_index] = np.sum(
                    np.abs(component_sectors),
                    axis=0,
                )
    return (
        np.sum(sector_result, axis=0),
        sector_result,
        branch_absolute_result,
    )


def passos_harmonic_kernel(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    occupations: np.ndarray,
    covariant_derivatives_ev: Sequence[np.ndarray],
    *,
    order: int,
    adiabatic_gamma_ev: float,
    intrinsic_permutation: bool = True,
    degeneracy_tolerance_ev: float = 1.0e-10,
) -> np.ndarray:
    r"""Reduced one-k Passos Eq. (31)-(32) mathematical kernel.

    ``G^(r)=hbar^r h^(r)`` and ``R^(r)=hbar^r rho^(r)`` are used. The function
    excludes BZ measure, model certification, and SI conversion; physical
    accumulation always uses intrinsic permutation symmetry.
    """

    values, _, _ = _passos_harmonic_kernel_evaluation(
        photon_energies_ev,
        energies_ev,
        occupations,
        covariant_derivatives_ev,
        order=order,
        adiabatic_gamma_ev=adiabatic_gamma_ev,
        intrinsic_permutation=intrinsic_permutation,
        degeneracy_tolerance_ev=degeneracy_tolerance_ev,
    )
    return values


def passos_si_prefactor(
    order: int,
    *,
    integration_dimension: int,
    length_unit_m: float,
) -> complex:
    """Return the SI conversion for eV/Cartesian-coordinate Eq. (32) data."""

    harmonic_order = _integer_index(order, name="order")
    dimension = _integer_index(integration_dimension, name="integration_dimension")
    length = float(length_unit_m)
    if harmonic_order <= 0 or dimension <= 0:
        raise ValueError("order and integration_dimension must be positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length_unit_m must be finite and positive")
    phase = (1.0j**harmonic_order) * ((-1.0) ** (harmonic_order + 1))
    return phase * E2_OVER_HBAR_S * length ** (
        harmonic_order + 1 - dimension
    )


def passos_harmonic_conductivity_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    order: int,
    adiabatic_gamma_ev: float,
) -> PassosHarmonicConductivityResult:
    """Accumulate a declared complete model over an increasing exact full-BZ grid."""

    harmonic_order = _integer_index(order, name="order")
    if harmonic_order <= 0:
        raise ValueError("order must be positive")
    omega = np.asarray(photon_energies_ev, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("photon_energies_ev must be a nonempty finite 1D array")
    if np.any(omega <= 0.0):
        raise ValueError("photon_energies_ev must be strictly positive")
    if isinstance(adiabatic_gamma_ev, (bool, np.bool_)):
        raise TypeError("adiabatic_gamma_ev must be a real scalar, not bool")
    gamma = float(adiabatic_gamma_ev)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("adiabatic_gamma_ev must be finite and positive")
    iterator = iter(kpoints)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("At least one Passos k-point is required") from error
    if not isinstance(first, PassosKPointData):
        raise TypeError("kpoints must contain PassosKPointData")
    reference_grid = first.grid
    reference_certificate = first.model_certificate
    reference_occupation = first.occupation_spec
    reference_axes = first.number_of_axes
    prefactor = passos_si_prefactor(
        harmonic_order,
        integration_dimension=reference_grid.integration_dimension,
        length_unit_m=reference_grid.coordinate_length_unit_m,
    )
    total: np.ndarray | None = None
    total_compensation: np.ndarray | None = None
    sector_total: np.ndarray | None = None
    sector_compensation: np.ndarray | None = None
    absolute_sum: np.ndarray | None = None
    absolute_compensation: np.ndarray | None = None
    seen_indices: set[int] = set()
    last_grid_index = -1

    def add_point(point: PassosKPointData) -> None:
        nonlocal total, total_compensation, sector_total, sector_compensation
        nonlocal absolute_sum, absolute_compensation, last_grid_index
        if not isinstance(point, PassosKPointData):
            raise TypeError("kpoints must contain PassosKPointData")
        if (
            not point.grid.compatible_with(reference_grid)
            or point.model_certificate != reference_certificate
            or point.occupation_spec != reference_occupation
            or point.number_of_axes != reference_axes
        ):
            raise ValueError("all Passos k-points must share grid/model/state metadata")
        if point.grid_index in seen_indices:
            raise ValueError(f"duplicate Passos grid index {point.grid_index}")
        if point.grid_index < last_grid_index:
            raise ValueError(
                "Passos k-points must be supplied in increasing grid_index order "
                "for deterministic compensated summation"
            )
        seen_indices.add(point.grid_index)
        last_grid_index = point.grid_index
        kernel, sectors, branch_absolute = _passos_harmonic_kernel_evaluation(
            omega,
            point.energies_ev,
            point.occupations,
            point.covariant_derivatives_ev,
            order=harmonic_order,
            adiabatic_gamma_ev=gamma,
            intrinsic_permutation=True,
            degeneracy_tolerance_ev=point.degeneracy_tolerance_ev,
        )
        factor = point.weight / (2.0 * np.pi) ** reference_grid.integration_dimension
        contribution = factor * kernel
        sector_contribution = factor * sectors
        if total is None:
            total = np.zeros_like(contribution)
            total_compensation = np.zeros_like(contribution)
            sector_total = np.zeros_like(sector_contribution)
            sector_compensation = np.zeros_like(sector_contribution)
            absolute_sum = np.zeros(contribution.shape, dtype=float)
            absolute_compensation = np.zeros_like(absolute_sum)
        assert total_compensation is not None
        assert sector_total is not None and sector_compensation is not None
        assert absolute_sum is not None and absolute_compensation is not None
        _kahan_add(total, total_compensation, contribution)
        _kahan_add(sector_total, sector_compensation, sector_contribution)
        _kahan_add(
            absolute_sum,
            absolute_compensation,
            abs(factor) * branch_absolute,
        )

    add_point(first)
    for point in iterator:
        add_point(point)
    assert total is not None and sector_total is not None and absolute_sum is not None
    expected_indices = set(range(reference_grid.number_of_points))
    if seen_indices != expected_indices:
        missing_count = len(expected_indices - seen_indices)
        raise ValueError(
            "Passos input does not cover the certified full BZ exactly once; "
            f"missing {missing_count} grid points"
        )
    sector_sum = np.zeros_like(total)
    sector_sum_compensation = np.zeros_like(total)
    for sector in sector_total:
        _kahan_add(sector_sum, sector_sum_compensation, sector)
    total = sector_sum
    total *= prefactor
    sector_total *= prefactor
    absolute_sum *= abs(prefactor)
    return PassosHarmonicConductivityResult(
        order=harmonic_order,
        photon_energies_ev=omega,
        conductivity=total,
        eq32_sector_conductivity=sector_total,
        absolute_contribution_sum=absolute_sum,
        adiabatic_gamma_ev=gamma,
        grid=reference_grid,
        model_certificate=reference_certificate,
        occupation_spec=reference_occupation,
        kpoint_count=len(seen_indices),
    )


def passos_shg_susceptibility(
    response: PassosHarmonicConductivityResult,
    *,
    effective_thickness_m: float | None = None,
    epsilon0_f_per_m: float = VACUUM_PERMITTIVITY_F_PER_M,
) -> PassosSHGSusceptibilityResult:
    r"""Convert 2D SHG sheet conductivity to sheet susceptibility.

    With the package convention ``exp(-i*omega*t)``, the sheet polarization
    obeys ``J_s(2w)=-i*2w*P_s(2w)`` and therefore
    ``chi_s^(2)=i*sigma_s^(2)/(2*epsilon0*w)`` in m^2/V. A bulk-like m/V value
    is returned only when an explicit effective thickness is supplied.
    """

    if not isinstance(response, PassosHarmonicConductivityResult):
        raise TypeError("response must be PassosHarmonicConductivityResult")
    if response.order != 2:
        raise ValueError("SHG susceptibility conversion requires an order-2 response")
    if response.grid.integration_dimension != 2:
        raise ValueError("sheet susceptibility conversion requires a 2D BZ response")
    epsilon0 = float(epsilon0_f_per_m)
    if not np.isfinite(epsilon0) or epsilon0 <= 0.0:
        raise ValueError("epsilon0_f_per_m must be finite and positive")
    angular_frequency = response.photon_energies_ev / HBAR_EV_S
    reshape = (angular_frequency.size,) + (1,) * (
        response.conductivity.ndim - 1
    )
    sheet = (
        1.0j
        * response.conductivity
        / (2.0 * epsilon0 * angular_frequency.reshape(reshape))
    )
    if effective_thickness_m is None:
        bulk = None
        thickness = None
    else:
        thickness = float(effective_thickness_m)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("effective_thickness_m must be finite and positive")
        bulk = sheet / thickness
    return PassosSHGSusceptibilityResult(
        photon_energies_ev=response.photon_energies_ev,
        sheet_susceptibility_m2_per_v=sheet,
        effective_thickness_m=thickness,
        effective_bulk_susceptibility_m_per_v=bulk,
    )


def passos_shg_conductivity_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    adiabatic_gamma_ev: float,
) -> PassosHarmonicConductivityResult:
    return passos_harmonic_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        order=2,
        adiabatic_gamma_ev=adiabatic_gamma_ev,
    )


def passos_thg_conductivity_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    adiabatic_gamma_ev: float,
) -> PassosHarmonicConductivityResult:
    return passos_harmonic_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        order=3,
        adiabatic_gamma_ev=adiabatic_gamma_ev,
    )


__all__ = [
    "DerivativeProvenance",
    "OccupationKind",
    "PassosHarmonicConductivityResult",
    "PassosKPointData",
    "PassosSHGSusceptibilityResult",
    "PassosModelCertificate",
    "PassosOccupationSpec",
    "PassosPrimitiveBZGrid",
    "passos_harmonic_conductivity_from_kpoint_data",
    "passos_harmonic_kernel",
    "passos_kpoint_data_from_fixed_basis_derivatives",
    "passos_shg_conductivity_from_kpoint_data",
    "passos_shg_susceptibility",
    "passos_si_prefactor",
    "passos_thg_conductivity_from_kpoint_data",
]
