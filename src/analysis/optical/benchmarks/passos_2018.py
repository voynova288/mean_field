from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Iterator

import numpy as np

from analysis.optical.linear import E2_OVER_HBAR_S
from analysis.optical.passos import (
    PassosKPointData,
    PassosModelCertificate,
    PassosOccupationSpec,
    PassosPrimitiveBZGrid,
    passos_kpoint_data_from_fixed_basis_derivatives,
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class Passos2018GrapheneParameters:
    """Nearest-neighbour graphene parameters used in Passos et al. Fig. 3."""

    hopping_ev: float = 1.0
    chemical_potential_ratio: float = 0.1
    lattice_constant_m: float = 0.246e-9
    temperature_k: float = 0.0
    spin_copies: int = 2

    def __post_init__(self) -> None:
        hopping = float(self.hopping_ev)
        mu_ratio = float(self.chemical_potential_ratio)
        lattice = float(self.lattice_constant_m)
        temperature = float(self.temperature_k)
        copies = _positive_integer(self.spin_copies, name="spin_copies")
        if not np.isfinite(hopping) or hopping <= 0.0:
            raise ValueError("hopping_ev must be finite and positive")
        if not np.isfinite(mu_ratio):
            raise ValueError("chemical_potential_ratio must be finite")
        if not np.isfinite(lattice) or lattice <= 0.0:
            raise ValueError("lattice_constant_m must be finite and positive")
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature_k must be finite and non-negative")
        object.__setattr__(self, "hopping_ev", hopping)
        object.__setattr__(self, "chemical_potential_ratio", mu_ratio)
        object.__setattr__(self, "lattice_constant_m", lattice)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "spin_copies", copies)

    @property
    def chemical_potential_ev(self) -> float:
        return self.chemical_potential_ratio * self.hopping_ev


@dataclass(frozen=True)
class RiceMeleParameters:
    """Inversion-broken two-band chain for an independent SHG check."""

    intracell_hopping_ev: float = 1.0
    intercell_hopping_ev: float = 0.7
    staggered_potential_ev: float = 0.4
    lattice_constant_m: float = 1.0e-9
    chemical_potential_ev: float = 0.0
    temperature_k: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.intracell_hopping_ev,
            self.intercell_hopping_ev,
            self.staggered_potential_ev,
            self.lattice_constant_m,
            self.chemical_potential_ev,
            self.temperature_k,
        )
        if any(not np.isfinite(float(value)) for value in values):
            raise ValueError("Rice-Mele parameters must be finite")
        if self.lattice_constant_m <= 0.0:
            raise ValueError("lattice_constant_m must be positive")
        if self.temperature_k < 0.0:
            raise ValueError("temperature_k must be non-negative")


def graphene_direct_lattice_vectors() -> np.ndarray:
    """Rows are the dimensionless vectors a1/a and a2/a of paper Eq. (44)."""

    return np.array(
        [
            [0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
        ],
        dtype=float,
    )


def graphene_reciprocal_vectors() -> np.ndarray:
    direct = graphene_direct_lattice_vectors()
    return 2.0 * np.pi * np.linalg.inv(direct).T


def passos2018_graphene_grid(
    mesh_shape: tuple[int, int],
    *,
    lattice_constant_m: float,
) -> PassosPrimitiveBZGrid:
    if len(mesh_shape) != 2:
        raise ValueError("graphene mesh_shape must have two entries")
    return PassosPrimitiveBZGrid(
        reciprocal_vectors=graphene_reciprocal_vectors(),
        mesh_shape=mesh_shape,
        coordinate_length_unit_m=lattice_constant_m,
        grid_label=f"passos2018_graphene_midpoint_{mesh_shape[0]}x{mesh_shape[1]}",
    )


def passos2018_graphene_refined_grid(
    base_mesh_size: int,
    *,
    patch_half_width_cells: int,
    refinement_factor: int,
    lattice_constant_m: float,
) -> PassosPrimitiveBZGrid:
    """Refine exact base cells around both Dirac valleys without overlap.

    The base size must be divisible by three so the two valleys at fractional
    reciprocal coordinates ``(1/3,2/3)`` and ``(2/3,1/3)`` lie on base-cell
    boundaries. Each square patch contains ``(2*patch_half_width_cells)^2``
    complete base cells and replaces each by equal refined subcells.
    """

    size = _positive_integer(base_mesh_size, name="base_mesh_size")
    half_width = _positive_integer(
        patch_half_width_cells,
        name="patch_half_width_cells",
    )
    factor = _positive_integer(refinement_factor, name="refinement_factor")
    if size % 3 != 0:
        raise ValueError("base_mesh_size must be divisible by three")
    if 4 * half_width >= size // 3:
        raise ValueError("Dirac refinement patches are too wide or overlapping")
    refined_cells: set[int] = set()
    for center in ((size // 3, 2 * size // 3), (2 * size // 3, size // 3)):
        for first in range(center[0] - half_width, center[0] + half_width):
            for second in range(center[1] - half_width, center[1] + half_width):
                refined_cells.add(
                    int(np.ravel_multi_index((first, second), (size, size)))
                )
    return PassosPrimitiveBZGrid(
        reciprocal_vectors=graphene_reciprocal_vectors(),
        mesh_shape=(size, size),
        coordinate_length_unit_m=lattice_constant_m,
        grid_label=(
            f"passos2018_graphene_refined_base{size}_half{half_width}_r{factor}"
        ),
        refined_cell_indices=tuple(sorted(refined_cells)),
        refinement_factor=factor,
    )


def _validate_graphene_grid(
    grid: PassosPrimitiveBZGrid,
    parameters: Passos2018GrapheneParameters,
) -> None:
    if grid.integration_dimension != 2:
        raise ValueError("Passos graphene requires a two-dimensional BZ grid")
    if not np.allclose(
        grid.reciprocal_vectors,
        graphene_reciprocal_vectors(),
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ValueError("grid reciprocal vectors do not match Passos graphene")
    if grid.coordinate_length_unit_m != parameters.lattice_constant_m:
        raise ValueError("grid length unit does not match graphene lattice constant")


def _graphene_two_band_hamiltonian_and_x_derivatives(
    k_cartesian: np.ndarray,
    *,
    hopping_ev: float,
    maximum_order: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    maximum_order_value = _positive_integer(maximum_order, name="maximum_order")
    direct = graphene_direct_lattice_vectors()
    k = np.asarray(k_cartesian, dtype=float)
    if k.shape != (2,) or np.any(~np.isfinite(k)):
        raise ValueError("graphene k_cartesian must be a finite 2-vector")
    phases = np.exp(-1.0j * (direct @ k))
    phi = 1.0 + np.sum(phases)
    hamiltonian = float(hopping_ev) * np.array(
        [[0.0, phi], [phi.conjugate(), 0.0]],
        dtype=np.complex128,
    )
    derivatives: list[np.ndarray] = []
    for order in range(1, maximum_order_value + 1):
        dphi = np.sum(((-1.0j * direct[:, 0]) ** order) * phases)
        derivative = float(hopping_ev) * np.array(
            [[0.0, dphi], [dphi.conjugate(), 0.0]],
            dtype=np.complex128,
        )
        derivatives.append(derivative.reshape((1,) * order + (2, 2)))
    return hamiltonian, tuple(derivatives)


def passos2018_graphene_hamiltonian_and_derivatives(
    k_cartesian: np.ndarray,
    *,
    parameters: Passos2018GrapheneParameters,
    maximum_order: int = 4,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return the complete spin-explicit model and x-derivative tower."""

    maximum_order_value = _positive_integer(maximum_order, name="maximum_order")
    h2, derivatives2 = _graphene_two_band_hamiltonian_and_x_derivatives(
        k_cartesian,
        hopping_ev=parameters.hopping_ev,
        maximum_order=maximum_order_value,
    )
    copies = _positive_integer(parameters.spin_copies, name="spin_copies")
    identity = np.eye(copies, dtype=np.complex128)
    hamiltonian = np.kron(identity, h2)
    derivatives = tuple(
        np.kron(identity, derivative.reshape(2, 2)).reshape(
            (1,) * order + hamiltonian.shape
        )
        for order, derivative in enumerate(derivatives2, start=1)
    )
    return hamiltonian, derivatives


def iter_passos2018_graphene_kpoints(
    grid: PassosPrimitiveBZGrid,
    *,
    parameters: Passos2018GrapheneParameters,
    maximum_order: int = 4,
) -> Iterator[PassosKPointData]:
    """Yield the declared complete primitive-BZ model for the THG benchmark."""

    _validate_graphene_grid(grid, parameters)
    copies = _positive_integer(parameters.spin_copies, name="spin_copies")
    maximum_order_value = _positive_integer(maximum_order, name="maximum_order")
    certificate = PassosModelCertificate(
        model_id="passos2018_nearest_neighbor_graphene",
        source_id=(
            "Passos2018_Eqs43-50_fixed_basis_derivatives_"
            f"t={parameters.hopping_ev}_spin_copies={copies}"
        ),
        basis_dimension=2 * copies,
        maximum_derivative_order=maximum_order_value,
        response_axis_labels=("x",),
        derivative_provenance="fixed_basis_complete",
    )
    occupation = PassosOccupationSpec.fermi_dirac(
        chemical_potential_ev=parameters.chemical_potential_ev,
        temperature_k=parameters.temperature_k,
    )
    for grid_index in range(grid.number_of_points):
        hamiltonian, derivatives = passos2018_graphene_hamiltonian_and_derivatives(
            grid.cartesian_point(grid_index),
            parameters=parameters,
            maximum_order=maximum_order_value,
        )
        energies, eigenvectors = np.linalg.eigh(hamiltonian)
        yield passos_kpoint_data_from_fixed_basis_derivatives(
            hamiltonian,
            energies,
            eigenvectors,
            derivatives,
            grid=grid,
            grid_index=grid_index,
            model_certificate=certificate,
            occupation_spec=occupation,
            fixed_basis_connection_is_zero=True,
        )


def passos2018_sigma0_si(parameters: Passos2018GrapheneParameters) -> float:
    r"""Paper normalization 3 q^4 a^2 t^2/(16 pi hbar mu^4).

    The paper comparison contract uses two explicit spin copies. The full BZ
    already includes both valleys; no extra valley multiplier is applied.
    """

    if parameters.spin_copies != 2:
        raise ValueError("the paper sigma0 contract requires spin_copies=2")
    mu = parameters.chemical_potential_ev
    if mu == 0.0:
        raise ValueError("chemical potential must be nonzero for sigma0")
    return (
        3.0
        * E2_OVER_HBAR_S
        * parameters.lattice_constant_m**2
        * parameters.hopping_ev**2
        / (16.0 * np.pi * mu**4)
    )


def rice_mele_grid(
    mesh_size: int,
    *,
    lattice_constant_m: float,
) -> PassosPrimitiveBZGrid:
    size = _positive_integer(mesh_size, name="mesh_size")
    return PassosPrimitiveBZGrid(
        reciprocal_vectors=np.array([[2.0 * np.pi]], dtype=float),
        mesh_shape=(size,),
        coordinate_length_unit_m=lattice_constant_m,
        grid_label=f"rice_mele_midpoint_{size}",
    )


def _validate_rice_mele_grid(
    grid: PassosPrimitiveBZGrid,
    parameters: RiceMeleParameters,
) -> None:
    expected_reciprocal = np.array([[2.0 * np.pi]], dtype=float)
    if grid.integration_dimension != 1 or not np.allclose(
        grid.reciprocal_vectors,
        expected_reciprocal,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ValueError("Rice-Mele requires the one-dimensional primitive BZ")
    if grid.coordinate_length_unit_m != parameters.lattice_constant_m:
        raise ValueError("grid length unit does not match Rice-Mele lattice constant")


def rice_mele_hamiltonian_and_derivatives(
    k_dimensionless: float,
    *,
    parameters: RiceMeleParameters,
    maximum_order: int = 3,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    maximum_order_value = _positive_integer(maximum_order, name="maximum_order")
    t1 = float(parameters.intracell_hopping_ev)
    t2 = float(parameters.intercell_hopping_ev)
    delta = float(parameters.staggered_potential_ev)
    k = float(k_dimensionless)
    if not np.isfinite(k):
        raise ValueError("k_dimensionless must be finite")
    dx = t1 + t2 * np.cos(k)
    dy = t2 * np.sin(k)
    hamiltonian = np.array(
        [[delta, dx - 1.0j * dy], [dx + 1.0j * dy, -delta]],
        dtype=np.complex128,
    )
    derivatives: list[np.ndarray] = []
    for order in range(1, maximum_order_value + 1):
        derivative_dx = t2 * np.cos(k + order * np.pi / 2.0)
        derivative_dy = t2 * np.sin(k + order * np.pi / 2.0)
        derivative = np.array(
            [
                [0.0, derivative_dx - 1.0j * derivative_dy],
                [derivative_dx + 1.0j * derivative_dy, 0.0],
            ],
            dtype=np.complex128,
        )
        derivatives.append(derivative.reshape((1,) * order + (2, 2)))
    return hamiltonian, tuple(derivatives)


def iter_rice_mele_kpoints(
    grid: PassosPrimitiveBZGrid,
    *,
    parameters: RiceMeleParameters,
    maximum_order: int = 3,
) -> Iterator[PassosKPointData]:
    _validate_rice_mele_grid(grid, parameters)
    maximum_order_value = _positive_integer(maximum_order, name="maximum_order")
    certificate = PassosModelCertificate(
        model_id="rice_mele_complete_two_band",
        source_id=(
            "RiceMele_fixed_basis_"
            f"t1={parameters.intracell_hopping_ev}_"
            f"t2={parameters.intercell_hopping_ev}_"
            f"delta={parameters.staggered_potential_ev}"
        ),
        basis_dimension=2,
        maximum_derivative_order=maximum_order_value,
        response_axis_labels=("x",),
        derivative_provenance="fixed_basis_complete",
    )
    occupation = PassosOccupationSpec.fermi_dirac(
        chemical_potential_ev=parameters.chemical_potential_ev,
        temperature_k=parameters.temperature_k,
    )
    for grid_index in range(grid.number_of_points):
        k_value = float(grid.cartesian_point(grid_index)[0])
        hamiltonian, derivatives = rice_mele_hamiltonian_and_derivatives(
            k_value,
            parameters=parameters,
            maximum_order=maximum_order_value,
        )
        energies, eigenvectors = np.linalg.eigh(hamiltonian)
        yield passos_kpoint_data_from_fixed_basis_derivatives(
            hamiltonian,
            energies,
            eigenvectors,
            derivatives,
            grid=grid,
            grid_index=grid_index,
            model_certificate=certificate,
            occupation_spec=occupation,
            fixed_basis_connection_is_zero=True,
        )


__all__ = [
    "Passos2018GrapheneParameters",
    "RiceMeleParameters",
    "graphene_direct_lattice_vectors",
    "graphene_reciprocal_vectors",
    "iter_passos2018_graphene_kpoints",
    "iter_rice_mele_kpoints",
    "passos2018_graphene_grid",
    "passos2018_graphene_refined_grid",
    "passos2018_graphene_hamiltonian_and_derivatives",
    "passos2018_sigma0_si",
    "rice_mele_grid",
    "rice_mele_hamiltonian_and_derivatives",
]
