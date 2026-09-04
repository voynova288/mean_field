"""Source-bound sector inventory for full selected-spin spiral stability.

This module is the bookkeeping precursor to a full momentum-off-diagonal
orbital Hessian.  It partitions every occupied-to-virtual coordinate of the
selected-spin global-rank Grassmann tangent by

``d = integer_label(particle) - integer_label(hole)``

and by the conserved valley charge

``chi = valley(particle) - valley(hole)``.

The finite square is never wrapped.  A sector ``(d, chi)`` is conjugate to
``(-d, -chi)``, but their dimensions are retained independently: a finite
source occupation pattern need not make them equal.  The inventory contains
all global-rank occupation-transfer directions, including the earlier
``d=(0,0)`` local-rank-preserving subset.

This file deliberately implements no off-k interaction response and grants no
Hessian or eigensolver authority.  In particular, integer-label conservation
is the convention of the scalable translational HF source; parity with the
older literal-float quartet mask in ``vituri2024_tdhf_full_functional`` is not
asserted.  That distinction must be resolved and independently qualified by a
later response adapter.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
import math
from typing import Final, Iterator

import numpy as np

from .vituri2024_hf_fft import Vituri2024TranslationalHFFFTFunctional
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_spiral_stability import (
    Vituri2024HFSpiralStabilityPreparation,
)

Array = np.ndarray

VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION: Final[str] = (
    "vituri2024_hf_spiral_full_selected_spin_sector_inventory.v1"
)
VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY: Final[str] = (
    "source_bound_selected_spin_global_rank_transition_inventory_only_not_"
    "off_k_functional_hessian_eigensolver_local_stability_or_paper_authority"
)
VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT: Final[str] = (
    "centered_square_integer_labels_particle_minus_hole_signed_displacement_"
    "no_wrap_no_carry_no_interpolation"
)
VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES: Final[tuple[int, int, int]] = (
    -2,
    0,
    2,
)

_INVENTORY_FACTORY_TOKEN = object()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _readonly_exact(value: object, dtype: np.dtype, shape: tuple[int, ...]) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or value.flags.writeable
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(
            f"expected finite read-only exact {dtype} array with shape {shape}"
        )
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _validate_centered_square_labels(labels: object) -> tuple[Array, int]:
    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.dtype(np.int64)
        or labels.ndim != 2
        or labels.shape[1:] != (2,)
    ):
        raise ValueError("integer mesh labels must be an exact int64 (Nk,2) array")
    nk = int(labels.shape[0])
    size = math.isqrt(nk)
    if size * size != nk or size < 1 or size % 2 != 1:
        raise ValueError("integer mesh labels must form a complete odd square")
    half = size // 2
    expected = np.asarray(
        [
            (ix, iy)
            for iy in range(-half, half + 1)
            for ix in range(-half, half + 1)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(labels, expected):
        raise ValueError(
            "integer mesh labels must use complete centered iy-outer/ix-inner order"
        )
    return _readonly_exact(labels, np.dtype(np.int64), (nk, 2)), size


def _exact_ordered_pair_counts(
    labels: Array,
    occupations: Array,
    *,
    mesh_size: int,
    particle_chunk_size: int = 128,
) -> Array:
    """Count all ordered selected-valley particle-hole transitions exactly.

    The accumulation uses integer differences and ``np.bincount`` only.  No
    floating FFT/correlation or tolerance enters the dimension inventory.
    """

    width = 2 * mesh_size - 1
    offset = mesh_size - 1
    counts = np.zeros((2, 2, width, width), dtype=np.int64)
    for particle_valley in range(2):
        particle_labels = labels[~occupations[particle_valley]]
        for hole_valley in range(2):
            hole_labels = labels[occupations[hole_valley]]
            for start in range(0, len(particle_labels), particle_chunk_size):
                particle = particle_labels[start : start + particle_chunk_size]
                difference = particle[:, None, :] - hole_labels[None, :, :]
                bins = (
                    (difference[..., 1] + offset) * width
                    + difference[..., 0]
                    + offset
                )
                counts[particle_valley, hole_valley] += np.bincount(
                    bins.reshape(-1), minlength=width * width
                ).reshape(width, width)
    counts.setflags(write=False)
    return counts


def _charge_counts(ordered_pair_counts: Array, valleys: tuple[int, int]) -> Array:
    width = ordered_pair_counts.shape[-1]
    result = np.zeros((3, width, width), dtype=np.int64)
    charge_to_index = {
        charge: index
        for index, charge in enumerate(VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES)
    }
    for particle_valley, particle_charge in enumerate(valleys):
        for hole_valley, hole_charge in enumerate(valleys):
            charge = particle_charge - hole_charge
            result[charge_to_index[charge]] += ordered_pair_counts[
                particle_valley, hole_valley
            ]
    result.setflags(write=False)
    return result


def _nonempty_conjugate_orbit_count(
    charge_counts: Array, *, mesh_size: int
) -> int:
    offset = mesh_size - 1
    count = 0
    for displacement_y in range(-offset, offset + 1):
        for displacement_x in range(-offset, offset + 1):
            for charge_index, charge in enumerate(
                VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES
            ):
                key_tuple = (displacement_x, displacement_y, charge)
                partner_tuple = (-displacement_x, -displacement_y, -charge)
                if partner_tuple < key_tuple:
                    continue
                partner_charge_index = VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES.index(
                    -charge
                )
                dimension = int(
                    charge_counts[
                        charge_index,
                        displacement_y + offset,
                        displacement_x + offset,
                    ]
                )
                partner_dimension = int(
                    charge_counts[
                        partner_charge_index,
                        -displacement_y + offset,
                        -displacement_x + offset,
                    ]
                )
                if dimension or partner_dimension:
                    count += 1
    return count


@dataclass(frozen=True, slots=True, order=True)
class Vituri2024HFSpiralFullSectorKey:
    """One signed momentum-displacement and valley-charge sector."""

    displacement_x: int
    displacement_y: int
    valley_charge: int

    def __post_init__(self) -> None:
        for name in ("displacement_x", "displacement_y", "valley_charge"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"sector {name} must be an exact int")
        if self.valley_charge not in VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES:
            raise ValueError("sector valley charge must be one of -2, 0, +2")

    @property
    def conjugate(self) -> "Vituri2024HFSpiralFullSectorKey":
        return Vituri2024HFSpiralFullSectorKey(
            -self.displacement_x,
            -self.displacement_y,
            -self.valley_charge,
        )


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralFullSectorOrbit:
    """One unreduced ``key <-> conjugate(key)`` real-Hessian orbit."""

    first: Vituri2024HFSpiralFullSectorKey
    second: Vituri2024HFSpiralFullSectorKey
    first_complex_dimension: int
    second_complex_dimension: int

    def __post_init__(self) -> None:
        if self.first.conjugate != self.second or self.second.conjugate != self.first:
            raise ValueError("sector orbit keys are not exact conjugates")
        if self.second < self.first:
            raise ValueError("sector orbit keys are not in canonical order")
        for name in ("first_complex_dimension", "second_complex_dimension"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("sector orbit dimensions must be nonnegative ints")
        if self.first == self.second and (
            self.first_complex_dimension != self.second_complex_dimension
        ):
            raise ValueError("self-conjugate sector dimension was duplicated inconsistently")

    @property
    def self_conjugate(self) -> bool:
        return self.first == self.second

    @property
    def complex_dimension(self) -> int:
        if self.self_conjugate:
            return self.first_complex_dimension
        return self.first_complex_dimension + self.second_complex_dimension

    @property
    def real_dimension(self) -> int:
        return 2 * self.complex_dimension


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralFullSectorInventory:
    """Complete source-bound selected-spin global-rank transition inventory."""

    _factory_token: InitVar[object]
    restricted_preparation: Vituri2024HFSpiralStabilityPreparation
    integer_mesh_labels: Array
    selected_occupations: Array
    selected_valleys: tuple[int, int]
    ordered_pair_complex_dimensions: Array
    charge_sector_complex_dimensions: Array
    mesh_size: int
    selected_occupied_count: int
    selected_virtual_count: int
    zero_displacement_complex_dimension: int
    momentum_off_diagonal_complex_dimension: int
    complex_dimension: int
    real_dimension: int
    nonempty_sector_count: int
    nonempty_conjugate_orbit_count: int
    inventory_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION,
        init=False,
    )
    authority: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY,
        init=False,
    )
    momentum_contract: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT,
        init=False,
    )
    valley_charges: tuple[int, int, int] = field(
        default=VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES,
        init=False,
    )
    candidate_only: bool = field(default=True, init=False)
    spectator_frozen_to_identity: bool = field(default=True, init=False)
    global_selected_rank_complete: bool = field(default=True, init=False)
    no_wrap: bool = field(default=True, init=False)
    integer_label_partitioning_convention: bool = field(default=True, init=False)
    integer_label_interaction_conservation_established: bool = field(
        default=False, init=False
    )
    conjugate_sector_dimensions_assumed_equal: bool = field(default=False, init=False)
    off_k_response_implemented: bool = field(default=False, init=False)
    literal_float_full_functional_parity_established: bool = field(
        default=False, init=False
    )
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _INVENTORY_FACTORY_TOKEN:
            raise TypeError("full sector inventory is factory-only")
        object.__setattr__(self, "inventory_fingerprint", self._expected_fingerprint())
        self.validate_live_state()

    @property
    def nk(self) -> int:
        return self.mesh_size * self.mesh_size

    @property
    def displacement_width(self) -> int:
        return 2 * self.mesh_size - 1

    @property
    def displacement_offset(self) -> int:
        return self.mesh_size - 1

    @property
    def total_sector_count(self) -> int:
        return 3 * self.displacement_width * self.displacement_width

    @property
    def zero_dimension_sector_count(self) -> int:
        return self.total_sector_count - self.nonempty_sector_count

    @property
    def restricted_complex_dimension(self) -> int:
        return self.restricted_preparation.hessian.complex_dimension

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.inventory_fingerprint

    def _key_indices(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> tuple[int, int, int]:
        if type(key) is not Vituri2024HFSpiralFullSectorKey:
            raise TypeError("sector key must have exact Vituri type")
        offset = self.displacement_offset
        x_index = key.displacement_x + offset
        y_index = key.displacement_y + offset
        if not (
            0 <= x_index < self.displacement_width
            and 0 <= y_index < self.displacement_width
        ):
            raise KeyError("sector displacement lies outside the finite no-wrap range")
        charge_index = self.valley_charges.index(key.valley_charge)
        return charge_index, y_index, x_index

    def sector_complex_dimension(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> int:
        charge, y_index, x_index = self._key_indices(key)
        return int(self.charge_sector_complex_dimensions[charge, y_index, x_index])

    def ordered_valley_pair_complex_dimension(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        *,
        particle_valley: int,
        hole_valley: int,
    ) -> int:
        if type(particle_valley) is not int or type(hole_valley) is not int:
            raise TypeError("ordered valley labels must be exact ints")
        if particle_valley not in self.selected_valleys or hole_valley not in self.selected_valleys:
            raise KeyError("ordered valley pair is outside the selected-spin valleys")
        if particle_valley - hole_valley != key.valley_charge:
            return 0
        _charge, y_index, x_index = self._key_indices(key)
        particle_index = self.selected_valleys.index(particle_valley)
        hole_index = self.selected_valleys.index(hole_valley)
        return int(
            self.ordered_pair_complex_dimensions[
                particle_index, hole_index, y_index, x_index
            ]
        )

    def iter_sector_keys(
        self, *, include_zero_dimension: bool = True
    ) -> Iterator[Vituri2024HFSpiralFullSectorKey]:
        offset = self.displacement_offset
        for displacement_y in range(-offset, offset + 1):
            for displacement_x in range(-offset, offset + 1):
                for charge in self.valley_charges:
                    key = Vituri2024HFSpiralFullSectorKey(
                        displacement_x, displacement_y, charge
                    )
                    if include_zero_dimension or self.sector_complex_dimension(key) > 0:
                        yield key

    def iter_conjugate_orbits(
        self, *, include_zero_dimension: bool = False
    ) -> Iterator[Vituri2024HFSpiralFullSectorOrbit]:
        for key in self.iter_sector_keys(include_zero_dimension=True):
            partner = key.conjugate
            if partner < key:
                continue
            first_dimension = self.sector_complex_dimension(key)
            second_dimension = self.sector_complex_dimension(partner)
            if (
                not include_zero_dimension
                and first_dimension == 0
                and second_dimension == 0
            ):
                continue
            yield Vituri2024HFSpiralFullSectorOrbit(
                key,
                partner,
                first_dimension,
                second_dimension,
            )

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "authority": self.authority,
                "momentum_contract": self.momentum_contract,
                "restricted_preparation": self.restricted_preparation.receipt.fingerprint,
                "prepared": self.restricted_preparation.prepared.fingerprint,
                "density_native": self.restricted_preparation.receipt.density_native_sha256,
                "fft_plan": self.restricted_preparation.prepared.functional.fft_plan.fingerprint,
                "integer_mesh_labels": _array_sha256(self.integer_mesh_labels),
                "selected_occupations": _array_sha256(self.selected_occupations),
                "selected_valleys": self.selected_valleys,
                "ordered_pair_dimensions": _array_sha256(
                    self.ordered_pair_complex_dimensions
                ),
                "charge_sector_dimensions": _array_sha256(
                    self.charge_sector_complex_dimensions
                ),
                "mesh_size": self.mesh_size,
                "selected_occupied_count": self.selected_occupied_count,
                "selected_virtual_count": self.selected_virtual_count,
                "zero_displacement_complex_dimension": (
                    self.zero_displacement_complex_dimension
                ),
                "momentum_off_diagonal_complex_dimension": (
                    self.momentum_off_diagonal_complex_dimension
                ),
                "complex_dimension": self.complex_dimension,
                "real_dimension": self.real_dimension,
                "nonempty_sector_count": self.nonempty_sector_count,
                "nonempty_conjugate_orbit_count": (
                    self.nonempty_conjugate_orbit_count
                ),
                "flags": (
                    self.candidate_only,
                    self.spectator_frozen_to_identity,
                    self.global_selected_rank_complete,
                    self.no_wrap,
                    self.integer_label_partitioning_convention,
                    self.integer_label_interaction_conservation_established,
                    self.conjugate_sector_dimensions_assumed_equal,
                    self.off_k_response_implemented,
                    self.literal_float_full_functional_parity_established,
                    self.reciprocity_established,
                    self.hermitian_eigensolver_authorized,
                    self.full_local_stability_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        if type(self.restricted_preparation) is not Vituri2024HFSpiralStabilityPreparation:
            raise TypeError("inventory requires the exact restricted preparation")
        self.restricted_preparation.validate_live_state()
        functional = self.restricted_preparation.prepared.functional
        if type(functional) is not Vituri2024TranslationalHFFFTFunctional:
            raise TypeError("full no-wrap sector inventory requires the exact FFT preparation")
        labels, mesh_size = _validate_centered_square_labels(self.integer_mesh_labels)
        nk = mesh_size * mesh_size
        occupations = _readonly_exact(
            self.selected_occupations,
            np.dtype(np.bool_),
            (2, nk),
        )
        if not np.array_equal(labels, functional.integer_mesh_labels):
            raise ValueError("inventory integer labels drifted from the FFT source")
        selected_indices = self.restricted_preparation.receipt.selected_flavor_indices
        expected_valleys = tuple(
            INTERNAL_FLAVOR_ORDER[index][0] for index in selected_indices
        )
        if expected_valleys != self.selected_valleys or set(expected_valleys) != {-1, 1}:
            raise ValueError("inventory selected-valley labels drifted")
        projectors = self.restricted_preparation.selected_projectors_conventional
        expected_occupations = np.stack(
            (projectors[0, 0, :].real, projectors[1, 1, :].real), axis=0
        ).astype(np.bool_)
        if not np.array_equal(occupations, expected_occupations):
            raise ValueError("inventory occupations drifted from the source projector")
        width = 2 * mesh_size - 1
        ordered = _readonly_exact(
            self.ordered_pair_complex_dimensions,
            np.dtype(np.int64),
            (2, 2, width, width),
        )
        by_charge = _readonly_exact(
            self.charge_sector_complex_dimensions,
            np.dtype(np.int64),
            (3, width, width),
        )
        if np.any(ordered < 0) or np.any(by_charge < 0):
            raise ValueError("sector dimensions must be nonnegative")
        rebuilt_ordered = _exact_ordered_pair_counts(
            labels,
            occupations,
            mesh_size=mesh_size,
        )
        if not np.array_equal(ordered, rebuilt_ordered):
            raise ValueError("ordered sector inventory drifted from source occupations")
        rebuilt_charge = _charge_counts(ordered, self.selected_valleys)
        if not np.array_equal(by_charge, rebuilt_charge):
            raise ValueError("charge-sector aggregation drifted")
        offset = mesh_size - 1
        for y_index, displacement_y in enumerate(range(-offset, offset + 1)):
            for x_index, displacement_x in enumerate(range(-offset, offset + 1)):
                support = (mesh_size - abs(displacement_x)) * (
                    mesh_size - abs(displacement_y)
                )
                if np.any(ordered[:, :, y_index, x_index] > support):
                    raise ValueError("sector dimension exceeds no-wrap pair support")
        occupied = int(np.count_nonzero(occupations))
        virtual = 2 * nk - occupied
        complex_dimension = occupied * virtual
        zero_displacement = int(np.sum(by_charge[:, offset, offset]))
        off_diagonal = complex_dimension - zero_displacement
        nonempty = int(np.count_nonzero(by_charge))
        orbit_count = sum(1 for _ in self.iter_conjugate_orbits())
        orbit_complex_dimension = sum(
            orbit.complex_dimension for orbit in self.iter_conjugate_orbits()
        )
        expected_complex = (
            self.restricted_preparation.prepared.selected_rank
            * (2 * nk - self.restricted_preparation.prepared.selected_rank)
        )
        locked = (
            self.api_version
            == VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION,
            self.authority == VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY,
            self.momentum_contract
            == VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT,
            self.valley_charges == VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES,
            self.mesh_size == mesh_size,
            self.selected_occupied_count == occupied,
            self.selected_virtual_count == virtual,
            self.zero_displacement_complex_dimension == zero_displacement,
            self.momentum_off_diagonal_complex_dimension == off_diagonal,
            self.complex_dimension == complex_dimension == expected_complex,
            self.real_dimension == 2 * complex_dimension,
            self.nonempty_sector_count == nonempty,
            self.nonempty_conjugate_orbit_count == orbit_count,
            int(np.sum(ordered)) == complex_dimension,
            int(np.sum(by_charge)) == complex_dimension,
            orbit_complex_dimension == complex_dimension,
            zero_displacement == self.restricted_complex_dimension,
            self.candidate_only is True,
            self.spectator_frozen_to_identity is True,
            self.global_selected_rank_complete is True,
            self.no_wrap is True,
            self.integer_label_partitioning_convention is True,
            self.integer_label_interaction_conservation_established is False,
            self.conjugate_sector_dimensions_assumed_equal is False,
            self.off_k_response_implemented is False,
            self.literal_float_full_functional_parity_established is False,
            self.reciprocity_established is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("full sector inventory invariant or authority drifted")
        if hasattr(self, "inventory_fingerprint") and (
            self.inventory_fingerprint != self._expected_fingerprint()
        ):
            raise ValueError("full sector inventory fingerprint drifted")


def build_vituri2024_hf_spiral_full_sector_inventory(
    restricted_preparation: Vituri2024HFSpiralStabilityPreparation,
) -> Vituri2024HFSpiralFullSectorInventory:
    """Build the complete exact-count inventory for one normal FFT endpoint."""

    if type(restricted_preparation) is not Vituri2024HFSpiralStabilityPreparation:
        raise TypeError("inventory requires the exact restricted preparation")
    restricted_preparation.validate_live_state()
    functional = restricted_preparation.prepared.functional
    if type(functional) is not Vituri2024TranslationalHFFFTFunctional:
        raise TypeError("full no-wrap sector inventory requires the exact FFT preparation")
    labels, mesh_size = _validate_centered_square_labels(
        functional.integer_mesh_labels
    )
    projectors = restricted_preparation.selected_projectors_conventional
    occupations_mutable = np.stack(
        (projectors[0, 0, :].real, projectors[1, 1, :].real), axis=0
    )
    if not np.all((occupations_mutable == 0.0) | (occupations_mutable == 1.0)):
        raise ValueError("full sector inventory requires exact normal occupations")
    occupation_bits = occupations_mutable.astype(np.bool_)
    occupation_bits.setflags(write=False)
    occupations = _readonly_exact(
        occupation_bits,
        np.dtype(np.bool_),
        (2, restricted_preparation.prepared.nk),
    )
    selected_valleys = tuple(
        INTERNAL_FLAVOR_ORDER[index][0]
        for index in restricted_preparation.receipt.selected_flavor_indices
    )
    if selected_valleys != (-1, 1):
        raise ValueError("selected-spin valley order must be exactly (-1,+1)")
    ordered_counts = _exact_ordered_pair_counts(
        labels,
        occupations,
        mesh_size=mesh_size,
    )
    ordered = _readonly_exact(
        ordered_counts,
        np.dtype(np.int64),
        ordered_counts.shape,
    )
    charge_counts = _charge_counts(ordered, selected_valleys)
    by_charge = _readonly_exact(
        charge_counts,
        np.dtype(np.int64),
        charge_counts.shape,
    )
    occupied = int(np.count_nonzero(occupations))
    virtual = 2 * restricted_preparation.prepared.nk - occupied
    complex_dimension = occupied * virtual
    offset = mesh_size - 1
    zero_displacement = int(np.sum(by_charge[:, offset, offset]))
    nonempty = int(np.count_nonzero(by_charge))

    orbit_count = _nonempty_conjugate_orbit_count(
        by_charge, mesh_size=mesh_size
    )
    return Vituri2024HFSpiralFullSectorInventory(
        _factory_token=_INVENTORY_FACTORY_TOKEN,
        restricted_preparation=restricted_preparation,
        integer_mesh_labels=labels,
        selected_occupations=occupations,
        selected_valleys=selected_valleys,  # type: ignore[arg-type]
        ordered_pair_complex_dimensions=ordered,
        charge_sector_complex_dimensions=by_charge,
        mesh_size=mesh_size,
        selected_occupied_count=occupied,
        selected_virtual_count=virtual,
        zero_displacement_complex_dimension=zero_displacement,
        momentum_off_diagonal_complex_dimension=(
            complex_dimension - zero_displacement
        ),
        complex_dimension=complex_dimension,
        real_dimension=2 * complex_dimension,
        nonempty_sector_count=nonempty,
        nonempty_conjugate_orbit_count=orbit_count,
    )


__all__ = [
    "VITURI2024_HF_SPIRAL_FULL_SECTOR_CHARGES",
    "VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_API_VERSION",
    "VITURI2024_HF_SPIRAL_FULL_SECTOR_INVENTORY_AUTHORITY",
    "VITURI2024_HF_SPIRAL_FULL_SECTOR_MOMENTUM_CONTRACT",
    "Vituri2024HFSpiralFullSectorInventory",
    "Vituri2024HFSpiralFullSectorKey",
    "Vituri2024HFSpiralFullSectorOrbit",
    "build_vituri2024_hf_spiral_full_sector_inventory",
]
