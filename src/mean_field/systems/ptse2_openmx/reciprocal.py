"""Reciprocal PAO projection in the OpenMX super gauge.

This module follows the convention frozen in
``results/ptse2_openmx_screened_hf/FORM_FACTOR_DERIVATION.md``.
The full PAO projection matrix is never materialized by the active-state path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from .source import (
    ANGSTROM_TO_BOHR,
    OpenMXBasisSpec,
    OpenMXFourierSourceManifest,
    OpenMXSource,
)


@dataclass(frozen=True)
class OpenMXRadialTransformTable:
    """OpenMX ``.ftpao`` radial Hankel-transform table."""

    source_id: str
    artifact_sha256: str
    species_order: tuple[str, ...]
    basis_specs: tuple[OpenMXBasisSpec, ...]
    momentum_grid_bohr_inv: np.ndarray
    values: dict[tuple[str, int, int], np.ndarray]
    stored_values_per_function: int

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        basis_specs: Iterable[OpenMXBasisSpec],
        *,
        source_id: str,
        expected_sha256: str,
        energy_cutoff_ry: float = 3600.0,
        momentum_grid_size: int = 900,
        padding_values: int = 2,
    ) -> "OpenMXRadialTransformTable":
        specs = tuple(basis_specs)
        if energy_cutoff_ry <= 0.0:
            raise ValueError("energy_cutoff_ry must be positive")
        if momentum_grid_size < 4:
            raise ValueError("momentum_grid_size must be at least four")
        if padding_values < 0:
            raise ValueError("padding_values cannot be negative")

        path_object = Path(path)
        actual_sha256 = _sha256_file(path_object)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f".ftpao hash mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        stored = momentum_grid_size + padding_values
        expected_functions = sum(spec.radial_function_count for spec in specs)
        raw = np.fromfile(path_object, dtype=np.float64)
        expected_values = expected_functions * stored
        if raw.size != expected_values:
            raise ValueError(
                f"unexpected .ftpao size: found {raw.size} doubles, "
                f"expected {expected_values}"
            )

        momentum_grid = (
            np.arange(momentum_grid_size, dtype=np.float64)
            * np.sqrt(energy_cutoff_ry)
            / momentum_grid_size
        )
        values: dict[tuple[str, int, int], np.ndarray] = {}
        offset = 0
        for spec in specs:
            for l, count in enumerate(spec.radial_multiplicities):
                for radial in range(count):
                    block = raw[offset : offset + stored]
                    values[(spec.species, l, radial)] = np.array(
                        block[:momentum_grid_size], copy=True
                    )
                    offset += stored
        if offset != raw.size:
            raise RuntimeError("internal .ftpao parsing offset mismatch")

        return cls(
            source_id=source_id,
            artifact_sha256=actual_sha256,
            species_order=tuple(spec.species for spec in specs),
            basis_specs=specs,
            momentum_grid_bohr_inv=momentum_grid,
            values=values,
            stored_values_per_function=stored,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: OpenMXFourierSourceManifest,
        source: OpenMXSource,
    ) -> "OpenMXRadialTransformTable":
        manifest.validate_source_basis(source)
        return cls.from_file(
            manifest.radial_transform_path,
            manifest.basis_specs,
            source_id=manifest.source_id,
            expected_sha256=manifest.radial_transform_sha256,
            energy_cutoff_ry=manifest.energy_cutoff_ry,
            momentum_grid_size=manifest.momentum_grid_size,
            padding_values=manifest.padding_values_per_function,
        )

    @property
    def maximum_momentum_bohr_inv(self) -> float:
        return float(self.momentum_grid_bohr_inv[-1])

    def evaluate(
        self,
        species: str,
        l: int,
        radial: int,
        momentum_bohr_inv: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate one transform with OpenMX's ``RF_BesselF`` interpolation."""

        try:
            table = self.values[(species, l, radial)]
        except KeyError as exc:
            raise KeyError(
                f"missing radial transform for {(species, l, radial)!r}"
            ) from exc
        return _openmx_cubic_interpolate(
            self.momentum_grid_bohr_inv,
            table,
            momentum_bohr_inv,
        )


@dataclass(frozen=True)
class ReciprocalGrid:
    """Finite three-dimensional integer reciprocal inventory."""

    integer_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.integer_indices)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("integer_indices must have shape (n_g, 3)")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("integer_indices must be integer-valued")
        normalized = np.asarray(indices, dtype=np.int64)
        if normalized.size:
            unique = np.unique(normalized, axis=0)
            if unique.shape[0] != normalized.shape[0]:
                raise ValueError("integer reciprocal indices must be unique")
        object.__setattr__(self, "integer_indices", normalized)

    def cartesian_bohr_inv(self, source: OpenMXSource) -> np.ndarray:
        return self.integer_indices @ source.reciprocal_bohr


@dataclass(frozen=True)
class ActiveReciprocalStates:
    """Active spinor coefficients U[G, spin, band] at one k point."""

    source_id: str
    reciprocal_bohr: np.ndarray
    k_fractional: np.ndarray
    grid: ReciprocalGrid
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        reciprocal = np.asarray(self.reciprocal_bohr, dtype=np.float64)
        k = np.asarray(self.k_fractional, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if reciprocal.shape != (3, 3):
            raise ValueError("reciprocal_bohr must have shape (3, 3)")
        if k.shape != (3,):
            raise ValueError("k_fractional must have shape (3,)")
        if coefficients.ndim != 3:
            raise ValueError("coefficients must have shape (n_g, n_spin, n_band)")
        if coefficients.shape[0] != self.grid.integer_indices.shape[0]:
            raise ValueError("coefficient and reciprocal-grid lengths differ")
        object.__setattr__(self, "reciprocal_bohr", reciprocal)
        object.__setattr__(self, "k_fractional", k)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def band_count(self) -> int:
        return int(self.coefficients.shape[2])

    @property
    def spin_count(self) -> int:
        return int(self.coefficients.shape[1])

    def gram(self) -> np.ndarray:
        flattened = self.coefficients.reshape(-1, self.band_count)
        return flattened.conj().T @ flattened


def generate_kinetic_reciprocal_grid(
    source: OpenMXSource,
    k_fractional: Iterable[float],
    cutoff_ry: float,
) -> ReciprocalGrid:
    """Generate all integer G with ``|k+G|^2 <= cutoff_ry`` in atomic units."""

    k = np.asarray(tuple(k_fractional), dtype=np.float64)
    if k.shape != (3,):
        raise ValueError("k_fractional must contain three components")
    if cutoff_ry < 0.0:
        raise ValueError("cutoff_ry cannot be negative")

    pmax = float(np.sqrt(cutoff_ry))
    direct_lengths = np.linalg.norm(source.lattice_bohr, axis=1)
    bounds = np.ceil(pmax * direct_lengths / (2.0 * np.pi) + np.abs(k)).astype(int)
    reciprocal = source.reciprocal_bohr
    k_cart = k @ reciprocal
    accepted: list[np.ndarray] = []

    n2_values = np.arange(-bounds[1], bounds[1] + 1, dtype=np.int64)
    n3_values = np.arange(-bounds[2], bounds[2] + 1, dtype=np.int64)
    n2_mesh, n3_mesh = np.meshgrid(n2_values, n3_values, indexing="ij")
    tail = np.column_stack([n2_mesh.ravel(), n3_mesh.ravel()])
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, cutoff_ry)

    for n1 in range(-bounds[0], bounds[0] + 1):
        indices = np.empty((tail.shape[0], 3), dtype=np.int64)
        indices[:, 0] = n1
        indices[:, 1:] = tail
        momenta = k_cart + indices @ reciprocal
        keep = np.einsum("gi,gi->g", momenta, momenta) <= cutoff_ry + tolerance
        if np.any(keep):
            accepted.append(indices[keep])

    if not accepted:
        return ReciprocalGrid(np.empty((0, 3), dtype=np.int64))
    return ReciprocalGrid(np.concatenate(accepted, axis=0))


def evaluate_atom_pao_projection(
    source: OpenMXSource,
    radial_table: OpenMXRadialTransformTable,
    species: str,
    position_bohr: Iterable[float],
    k_fractional: Iterable[float],
    integer_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate B[G, local scalar PAO] for one atom and a supplied G batch."""

    if radial_table.basis_specs != source.basis_specs:
        raise ValueError(".ftpao basis specification does not match the OpenMX input")
    if species not in source.basis_by_species:
        raise KeyError(f"unknown species {species!r}")
    position = np.asarray(tuple(position_bohr), dtype=np.float64)
    k = np.asarray(tuple(k_fractional), dtype=np.float64)
    indices = np.asarray(integer_indices)
    if position.shape != (3,) or k.shape != (3,):
        raise ValueError("position and k_fractional must contain three components")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("integer_indices must have shape (n_g, 3)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("integer_indices must use an integer dtype")

    reciprocal = source.reciprocal_bohr
    g_cart = np.asarray(indices, dtype=np.int64) @ reciprocal
    momentum = k @ reciprocal + g_cart
    momentum_norm = np.linalg.norm(momentum, axis=1)
    center_phase = np.exp(-1j * (g_cart @ position))
    channels = source.basis_by_species[species].channels()
    projection = np.empty((indices.shape[0], len(channels)), dtype=np.complex128)
    harmonic_cache: dict[int, np.ndarray] = {}
    radial_cache: dict[tuple[int, int], np.ndarray] = {}
    for channel_index, (l, radial, harmonic) in enumerate(channels):
        if l not in harmonic_cache:
            harmonic_cache[l] = openmx_real_spherical_harmonics(l, momentum)
        radial_key = (l, radial)
        if radial_key not in radial_cache:
            radial_cache[radial_key] = radial_table.evaluate(
                species, l, radial, momentum_norm
            )
        projection[:, channel_index] = (
            center_phase
            * 4.0
            * np.pi
            * ((-1j) ** l)
            * harmonic_cache[l][:, harmonic]
            * radial_cache[radial_key]
            / np.sqrt(source.volume_bohr3)
        )
    return projection


def build_active_reciprocal_states(
    source: OpenMXSource,
    radial_table: OpenMXRadialTransformTable,
    k_fractional: Iterable[float],
    grid: ReciprocalGrid,
    band_coefficients: np.ndarray,
    *,
    batch_size: int = 4096,
) -> ActiveReciprocalStates:
    """Build U directly without storing the full reciprocal PAO matrix B.

    ``band_coefficients`` must be in original OpenMX atom/PAO order and in the
    super gauge used by :func:`assemble_super_gauge_matrix`.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    k = np.asarray(tuple(k_fractional), dtype=np.float64)
    if k.shape != (3,):
        raise ValueError("k_fractional must contain three components")
    coefficients = np.asarray(band_coefficients, dtype=np.complex128)
    if coefficients.ndim != 2:
        raise ValueError("band_coefficients must have shape (n_spinor_pao, n_band)")
    if coefficients.shape[0] != source.spinor_pao_count:
        raise ValueError(
            f"expected {source.spinor_pao_count} PAO rows, got {coefficients.shape[0]}"
        )
    if source.spin_blocks != 2:
        raise ValueError("the initial PtSe2 reciprocal path requires NC two-spin-block input")
    if radial_table.basis_specs != source.basis_specs:
        raise ValueError(".ftpao basis specification does not match the OpenMX input")

    reciprocal = source.reciprocal_bohr
    g_cart_all = grid.cartesian_bohr_inv(source)
    k_cart = k @ reciprocal
    scalar_map = source.scalar_pao_map()
    scalar_count = source.scalar_pao_count
    band_count = coefficients.shape[1]
    result = np.zeros(
        (grid.integer_indices.shape[0], source.spin_blocks, band_count),
        dtype=np.complex128,
    )
    basis_by_species = source.basis_by_species
    species_plans: list[
        tuple[str, np.ndarray, tuple[tuple[int, int, int], ...], np.ndarray]
    ] = []
    for species in radial_table.species_order:
        atom_indices = [atom.index for atom in source.atoms if atom.species == species]
        positions = (
            np.asarray(
                [source.atoms[index].position_angstrom for index in atom_indices],
                dtype=np.float64,
            )
            * ANGSTROM_TO_BOHR
        )
        channels = basis_by_species[species].channels()
        channel_lookup = {channel: position for position, channel in enumerate(channels)}
        scalar_indices = np.full((len(atom_indices), len(channels)), -1, dtype=np.int64)
        atom_row = {atom_index: row for row, atom_index in enumerate(atom_indices)}
        for orbital in scalar_map:
            if orbital.species != species:
                continue
            scalar_indices[
                atom_row[orbital.atom_index],
                channel_lookup[(orbital.l, orbital.radial, orbital.harmonic)],
            ] = orbital.index
        if np.any(scalar_indices < 0):
            raise RuntimeError(f"incomplete PAO channel map for species {species!r}")
        species_coefficients = np.empty(
            (len(atom_indices), len(channels), source.spin_blocks, band_count),
            dtype=np.complex128,
        )
        for spin in range(source.spin_blocks):
            species_coefficients[:, :, spin, :] = coefficients[
                scalar_indices + spin * scalar_count, :
            ]
        species_plans.append((species, positions, channels, species_coefficients))

    for start in range(0, grid.integer_indices.shape[0], batch_size):
        stop = min(start + batch_size, grid.integer_indices.shape[0])
        g_cart = g_cart_all[start:stop]
        momentum = k_cart + g_cart
        momentum_norm = np.linalg.norm(momentum, axis=1)
        batch_result = np.zeros(
            (stop - start, source.spin_blocks, band_count), dtype=np.complex128
        )

        for species, positions, channels, species_coefficients in species_plans:
            center_phase = np.exp(-1j * (g_cart @ positions.T))
            structure_factors = center_phase @ species_coefficients.reshape(
                positions.shape[0], -1
            )
            structure_factors = structure_factors.reshape(
                stop - start,
                len(channels),
                source.spin_blocks,
                band_count,
            )

            channel_amplitudes = np.empty(
                (stop - start, len(channels)), dtype=np.complex128
            )
            harmonic_cache: dict[int, np.ndarray] = {}
            radial_cache: dict[tuple[int, int], np.ndarray] = {}
            for channel_index, (l, radial, harmonic) in enumerate(channels):
                if l not in harmonic_cache:
                    harmonic_cache[l] = openmx_real_spherical_harmonics(l, momentum)
                radial_key = (l, radial)
                if radial_key not in radial_cache:
                    radial_cache[radial_key] = radial_table.evaluate(
                        species, l, radial, momentum_norm
                    )
                channel_amplitudes[:, channel_index] = (
                    4.0
                    * np.pi
                    * ((-1j) ** l)
                    * harmonic_cache[l][:, harmonic]
                    * radial_cache[radial_key]
                    / np.sqrt(source.volume_bohr3)
                )

            batch_result += np.einsum(
                "gc,gcsn->gsn",
                channel_amplitudes,
                structure_factors,
                optimize=True,
            )

        result[start:stop] = batch_result

    return ActiveReciprocalStates(
        source_id=f"{radial_table.source_id}:{source.input_sha256}",
        reciprocal_bohr=source.reciprocal_bohr,
        k_fractional=k,
        grid=grid,
        coefficients=result,
    )


def openmx_real_spherical_harmonics(l: int, vectors: np.ndarray) -> np.ndarray:
    """Evaluate OpenMX real harmonics for l=0,1,2 in exact local ordering."""

    vectors_array = np.asarray(vectors, dtype=np.float64)
    if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
        raise ValueError("vectors must have shape (n, 3)")
    if l not in {0, 1, 2}:
        raise ValueError("initial PtSe2 basis supports only l=0,1,2")
    count = vectors_array.shape[0]
    result = np.zeros((count, 2 * l + 1), dtype=np.float64)
    norm = np.linalg.norm(vectors_array, axis=1)
    nonzero = norm > 0.0

    if l == 0:
        result[:, 0] = 0.282094791773878
        return result

    unit = np.zeros_like(vectors_array)
    unit[nonzero] = vectors_array[nonzero] / norm[nonzero, None]
    x, y, z = unit.T
    if l == 1:
        result[:, 0] = 0.48860251190292 * x
        result[:, 1] = 0.48860251190292 * y
        result[:, 2] = 0.48860251190292 * z
        return result

    result[:, 0] = 0.94617469575756 * z * z - 0.31539156525252
    result[:, 1] = 0.54627421529604 * (x * x - y * y)
    result[:, 2] = 1.09254843059208 * x * y
    result[:, 3] = 1.09254843059208 * x * z
    result[:, 4] = 1.09254843059208 * y * z
    result[~nonzero] = 0.0
    return result


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _openmx_cubic_interpolate(
    grid: np.ndarray,
    values: np.ndarray,
    points: np.ndarray | float,
) -> np.ndarray:
    """Vectorized port of OpenMX ``RF_BesselF.c``."""

    grid_array = np.asarray(grid, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    if grid_array.ndim != 1 or value_array.shape != grid_array.shape:
        raise ValueError("grid and values must be one-dimensional and equally sized")
    if grid_array.size < 4 or np.any(np.diff(grid_array) <= 0.0):
        raise ValueError("grid must be strictly increasing with at least four points")
    if np.any(point_array < 0.0):
        raise ValueError("radial momentum cannot be negative")

    flat = point_array.reshape(-1)
    output = np.zeros_like(flat)
    valid = flat <= grid_array[-1]
    if not np.any(valid):
        return output.reshape(point_array.shape)

    x = flat[valid]
    m = np.searchsorted(grid_array, x, side="left")
    m = np.clip(m, 2, grid_array.size - 1)

    h1 = grid_array[m - 1] - grid_array[m - 2]
    h2 = grid_array[m] - grid_array[m - 1]
    f1 = value_array[m - 2]
    f2 = value_array[m - 1]
    f3 = value_array[m]

    at_last = m == grid_array.size - 1
    h3 = np.empty_like(h2)
    f4 = np.empty_like(f3)
    h3[at_last] = -(h1[at_last] + h2[at_last])
    f4[at_last] = f1[at_last]
    interior = ~at_last
    h3[interior] = grid_array[m[interior] + 1] - grid_array[m[interior]]
    f4[interior] = value_array[m[interior] + 1]

    g1 = ((f3 - f2) * h1 / h2 + (f2 - f1) * h2 / h1) / (h1 + h2)
    g2 = ((f4 - f3) * h2 / h3 + (f3 - f2) * h3 / h2) / (h2 + h3)
    y1 = (x - grid_array[m - 1]) / h2
    y2 = (x - grid_array[m]) / h2
    interpolated = y2 * y2 * (
        3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2
    ) + y1 * y1 * (
        3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1
    )
    output[valid] = interpolated
    return output.reshape(point_array.shape)
