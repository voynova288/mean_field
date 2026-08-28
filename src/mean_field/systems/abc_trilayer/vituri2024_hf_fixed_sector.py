"""Exhaustive fixed-sector homogeneous Hartree--Fock for Vituri 2024.

This system-layer adapter implements the finite-volume half-metal sector used
by the sealed job461276 discriminator.  It fixes each flavor rank, represents
exact ``h0`` shells by one common uniform initializer, and exhaustively replays
every simultaneous per-flavor coordinate branch from that initializer through
:func:`mean_field.core.hf.run_hartree_fock_problem`.

The dense functional remains the oracle and the FFT functional is only an
algebraically equivalent backend.  No reference counterterm is added: in
particular there is no ``R=I`` subtraction.  Every in-process result is a
candidate receipt with no independent authority or local Hessian stability;
sealed independent finite-volume authority remains external job461276.  The
adapter establishes neither an author cutoff nor UV convergence, unrestricted
stability, TDHF readiness, production authority, or paper reproduction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from itertools import combinations, product
import json
import math
from numbers import Real
from typing import Final, Literal, TypeAlias

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)

from .vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_scf import (
    Vituri2024HFState,
    Vituri2024PreparedHomogeneousHF,
    make_vituri2024_hf_state,
)

Array = np.ndarray
BoundaryKind = Literal["unique", "exact", "positive_subtolerance"]
EndpointOutcome = Literal["stationary", "normal_endpoint_gate_rejection"]

VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "branch_choice_in_final_map_rejection",
        "branch_choice_not_applied_rejection",
        "branch_frontier_in_final_map_rejection",
        "diagonal_coherence_rejection",
        "fresh_final_fock_boundary_rejection",
        "fresh_final_fock_common_mu_rejection",
        "h0_positive_subtolerance_splitting_rejection",
        "positive_subtolerance_splitting_rejection",
    }
)

VITURI2024_FIXED_SECTOR_HF_API_VERSION: Final[str] = (
    "vituri2024_homogeneous_half_metal_fixed_sector.v1"
)
VITURI2024_FIXED_SECTOR_RESULT_SCHEMA_VERSION: Final[str] = (
    "vituri2024_homogeneous_half_metal_fixed_sector_result.v1"
)
VITURI2024_FIXED_SECTOR_AUTHORITY: Final[str] = (
    "in_process_candidate_receipt_only_external_job461276_retains_independent_"
    "finite_volume_fixed_sector_authority"
)
VITURI2024_FIXED_SECTOR_INITIALIZER_MODE: Final[str] = (
    "uniform_exact_shell_half_metal"
)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _positive_real(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _strict_float(value: object, label: str, *, finite: bool = True) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be a float")
    if finite and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validate_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _strict_tuple(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _readonly(value: object, dtype: np.dtype | None = None) -> Array:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _relative_density_norm(updated: object, previous: object) -> float:
    updated_array = np.asarray(updated)
    numerator = float(np.linalg.norm(np.asarray(previous) - updated_array))
    denominator = float(np.linalg.norm(updated_array))
    if denominator < 1.0e-15:
        return 0.0 if numerator < 1.0e-15 else float("inf")
    return numerator / denominator


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _array_exact(left: object, right: object) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes(order="C") == b.tobytes(order="C")
    )


def _energy_from_engine_inputs(
    interaction_h: Array, h0: Array, native_density: Array
) -> float:
    one_body = np.einsum("abk,abk->", h0, native_density, optimize=False)
    interaction = 0.5 * np.einsum(
        "abk,abk->", interaction_h, native_density, optimize=False
    )
    total = complex(one_body + interaction)
    if abs(total.imag) > 5.0e-11 * max(1.0, abs(total), abs(one_body), abs(interaction)):
        raise ValueError("Vituri fixed-sector engine energy is materially complex")
    return float(total.real)


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorPolicy:
    """Numerical gates and finite branch caps; no physical cutoff is defaulted."""

    selected_hole_spin: Literal[-1, 1] = 1
    boundary_floor_ev: float = 1.0e-12
    boundary_roundoff_multiplier: float = 64.0
    max_iter: int = 50
    oda_stall_threshold: float = 1.0e-8
    max_oda_lambda: float = 1.0
    hamiltonian_hermiticity_tolerance_ev: float = 1.0e-10
    diagonal_coherence_tolerance_ev: float = 1.0e-10
    final_raw_norm_tolerance: float = 1.0e-10
    commutator_tolerance_ev: float = 1.0e-8
    idempotency_tolerance: float = 1.0e-8
    population_tolerance: float = 1.0e-8
    fresh_fock_tolerance_ev: float = 1.0e-10
    energy_parity_relative_tolerance: float = 1.0e-10
    maximum_generation: int = 16
    maximum_choices_per_trigger: int = 4096
    maximum_replayed_paths: int = 512
    maximum_endpoints: int = 64
    initializer_mode: str = VITURI2024_FIXED_SECTOR_INITIALIZER_MODE
    initializer_seed: int = 0
    convergence_rule: str = "raw"
    author_cutoff_identified: bool = False
    uv_plateau_established: bool = False
    unrestricted_ground_state_established: bool = False
    local_hessian_stability_established: bool = False
    full_paper_reproduction_verified: bool = False
    tdhf_authority: bool = False
    production_authority: bool = False
    visual_match_promotes_authority: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        selected_hole_spin = _strict_int(self.selected_hole_spin, "selected_hole_spin")
        if selected_hole_spin not in (-1, 1):
            raise ValueError("selected_hole_spin must be -1 or +1")
        object.__setattr__(self, "selected_hole_spin", selected_hole_spin)
        for name in (
            "boundary_floor_ev",
            "boundary_roundoff_multiplier",
            "oda_stall_threshold",
            "max_oda_lambda",
            "hamiltonian_hermiticity_tolerance_ev",
            "diagonal_coherence_tolerance_ev",
            "final_raw_norm_tolerance",
            "commutator_tolerance_ev",
            "idempotency_tolerance",
            "population_tolerance",
            "fresh_fock_tolerance_ev",
            "energy_parity_relative_tolerance",
        ):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.max_oda_lambda > 1.0:
            raise ValueError("max_oda_lambda must not exceed one")
        for name in (
            "max_iter",
            "maximum_generation",
            "maximum_choices_per_trigger",
            "maximum_replayed_paths",
            "maximum_endpoints",
        ):
            value = _strict_int(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.initializer_mode != VITURI2024_FIXED_SECTOR_INITIALIZER_MODE:
            raise ValueError("fixed-sector initializer mode is locked")
        if self.initializer_seed != 0 or self.convergence_rule != "raw":
            raise ValueError("fixed-sector seed/convergence semantics are locked")
        if tuple(INTERNAL_FLAVOR_ORDER) != ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            raise RuntimeError("Vituri internal flavor order drifted")
        authority_names = (
            "author_cutoff_identified",
            "uv_plateau_established",
            "unrestricted_ground_state_established",
            "local_hessian_stability_established",
            "full_paper_reproduction_verified",
            "tdhf_authority",
            "production_authority",
            "visual_match_promotes_authority",
        )
        for name in authority_names:
            _strict_bool(getattr(self, name), name)
        if any(getattr(self, name) for name in authority_names):
            raise ValueError("fixed-sector policy authority was inflated")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        }
        payload["api_version"] = VITURI2024_FIXED_SECTOR_HF_API_VERSION
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    @property
    def full_flavors(self) -> tuple[int, int]:
        return (0, 2) if self.selected_hole_spin == 1 else (1, 3)

    @property
    def partial_flavors(self) -> tuple[int, int]:
        return (1, 3) if self.selected_hole_spin == 1 else (0, 2)

    def electron_counts(self, nk: int, holes_per_valley: int) -> tuple[int, int, int, int]:
        count = nk - holes_per_valley
        return tuple(nk if flavor in self.full_flavors else count for flavor in range(4))  # type: ignore[return-value]

    def validate_live_state(self) -> None:
        expected = replace(self)
        if expected.fingerprint != self.fingerprint:
            raise ValueError("fixed-sector policy fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBoundary:
    flavor: int
    electron_count: int
    kind: BoundaryKind
    lower_ev: float
    upper_ev: float
    gap_ev: float
    effective_tolerance_ev: float
    strictly_below_indices: tuple[int, ...] = ()
    shell_indices: tuple[int, ...] = ()
    selected_rank: int = 0
    occupied_indices: tuple[int, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        flavor = _strict_int(self.flavor, "flavor")
        count = _strict_int(self.electron_count, "electron_count")
        if flavor not in range(4) or count < 1:
            raise ValueError("invalid fixed-sector boundary flavor/count")
        if self.kind not in ("unique", "exact", "positive_subtolerance"):
            raise ValueError("invalid fixed-sector boundary kind")
        values = (self.lower_ev, self.upper_ev, self.gap_ev, self.effective_tolerance_ev)
        for value, label in zip(
            values,
            ("lower_ev", "upper_ev", "gap_ev", "effective_tolerance_ev"),
            strict=True,
        ):
            _strict_float(value, label)
        for name in (
            "strictly_below_indices",
            "shell_indices",
            "occupied_indices",
        ):
            _strict_tuple(getattr(self, name), name)
        if self.effective_tolerance_ev <= 0.0 or self.gap_ev < 0.0:
            raise ValueError("fixed-sector boundary tolerance/gap is invalid")
        if self.upper_ev - self.lower_ev != self.gap_ev:
            raise ValueError("fixed-sector boundary gap does not match its endpoints")
        below = tuple(_strict_int(x, "strictly_below index") for x in self.strictly_below_indices)
        shell = tuple(_strict_int(x, "shell index") for x in self.shell_indices)
        occupied = tuple(_strict_int(x, "occupied index") for x in self.occupied_indices)
        rank = _strict_int(self.selected_rank, "selected_rank")
        if any(tuple(sorted(set(items))) != items for items in (below, shell, occupied)):
            raise ValueError("boundary indices must be unique and sorted")
        if self.kind == "unique":
            valid = self.gap_ev > self.effective_tolerance_ev and len(occupied) == count
            valid = valid and not shell and rank == 0
        elif self.kind == "exact":
            valid = self.gap_ev == 0.0 and 0 < rank < len(shell)
            valid = valid and len(below) + rank == count and not occupied
        else:
            valid = 0.0 < self.gap_ev <= self.effective_tolerance_ev
            valid = valid and not below and not shell and not occupied and rank == 0
        if not valid:
            raise ValueError("fixed-sector boundary classification is inconsistent")
        object.__setattr__(self, "flavor", flavor)
        object.__setattr__(self, "electron_count", count)
        object.__setattr__(self, "strictly_below_indices", below)
        object.__setattr__(self, "shell_indices", shell)
        object.__setattr__(self, "occupied_indices", occupied)
        object.__setattr__(self, "selected_rank", rank)
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(self) -> None:
        if replace(self).fingerprint != self.fingerprint:
            raise ValueError("fixed-sector boundary fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorInitializer:
    density_native: Array
    density_sha256: str
    h0_sha256: str
    electron_counts_by_flavor: tuple[int, int, int, int]
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...]
    mirror_symmetric: bool
    prepared_fingerprint: str
    policy_fingerprint: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        density = _readonly(self.density_native, np.dtype(np.complex128))
        if density.ndim != 3 or density.shape[:2] != (4, 4) or not np.all(np.isfinite(density)):
            raise ValueError("fixed-sector initializer density is invalid")
        for name in (
            "density_sha256",
            "h0_sha256",
            "prepared_fingerprint",
            "policy_fingerprint",
        ):
            _validate_sha256(getattr(self, name), name)
        if _array_sha256(density) != self.density_sha256:
            raise ValueError("fixed-sector initializer density hash mismatch")
        _strict_bool(self.mirror_symmetric, "mirror_symmetric")
        if not self.mirror_symmetric:
            raise ValueError("fixed-sector common initializer must be mirror symmetric")
        _strict_tuple(self.boundaries, "initializer boundaries")
        _strict_tuple(self.electron_counts_by_flavor, "initializer electron counts")
        if any(type(boundary) is not Vituri2024FixedSectorBoundary for boundary in self.boundaries):
            raise TypeError("initializer boundaries must be typed")
        if tuple(boundary.flavor for boundary in self.boundaries) not in ((1, 3), (0, 2)):
            raise ValueError("fixed-sector initializer partial-flavor order drifted")
        for boundary in self.boundaries:
            boundary.validate_live_state()
            if boundary.kind == "positive_subtolerance":
                raise ValueError("rejected boundary cannot enter an initializer")
        counts = tuple(_strict_int(x, "initializer electron count") for x in self.electron_counts_by_flavor)
        if len(counts) != 4:
            raise ValueError("initializer requires four flavor counts")
        object.__setattr__(self, "density_native", density)
        object.__setattr__(self, "electron_counts_by_flavor", counts)
        payload = {
            "api_version": VITURI2024_FIXED_SECTOR_HF_API_VERSION,
            "density_sha256": self.density_sha256,
            "h0_sha256": self.h0_sha256,
            "electron_counts_by_flavor": counts,
            "boundary_fingerprints": [item.fingerprint for item in self.boundaries],
            "mirror_symmetric": self.mirror_symmetric,
            "prepared_fingerprint": self.prepared_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(self) -> None:
        if _array_sha256(self.density_native) != self.density_sha256:
            raise ValueError("fixed-sector initializer array mutated")
        expected = replace(self)
        if expected.fingerprint != self.fingerprint:
            raise ValueError("fixed-sector initializer fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBranchTrigger:
    generation: int
    exact_fock_sha256: str
    previous_density_sha256: str
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...]
    canonical_choice_count: int
    canonical_order: str = (
        "partial_flavor_order; lexicographic_combinations; Cartesian_product"
    )
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        generation = _strict_int(self.generation, "generation")
        count = _strict_int(self.canonical_choice_count, "canonical_choice_count")
        if generation < 0 or count < 1:
            raise ValueError("invalid branch trigger generation/count")
        _validate_sha256(self.exact_fock_sha256, "exact_fock_sha256")
        _validate_sha256(self.previous_density_sha256, "previous_density_sha256")
        _strict_tuple(self.boundaries, "branch trigger boundaries")
        if type(self.canonical_order) is not str:
            raise TypeError("branch trigger canonical_order must be a string")
        if not self.boundaries or any(type(item) is not Vituri2024FixedSectorBoundary or item.kind != "exact" for item in self.boundaries):
            raise ValueError("branch trigger requires typed exact boundaries")
        flavors = tuple(item.flavor for item in self.boundaries)
        if flavors != tuple(sorted(flavors)) or len(set(flavors)) != len(flavors):
            raise ValueError("branch trigger flavor order is not canonical")
        for boundary in self.boundaries:
            boundary.validate_live_state()
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "canonical_choice_count", count)
        payload = {
            "generation": generation,
            "exact_fock_sha256": self.exact_fock_sha256,
            "previous_density_sha256": self.previous_density_sha256,
            "boundary_fingerprints": [item.fingerprint for item in self.boundaries],
            "canonical_choice_count": count,
            "canonical_order": self.canonical_order,
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(self) -> None:
        if replace(self).fingerprint != self.fingerprint:
            raise ValueError("fixed-sector branch trigger fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBranchChoice:
    """One canonical member of the aggregate simultaneous shell product."""

    trigger: Vituri2024FixedSectorBranchTrigger
    canonical_choice_index: int
    canonical_choice_count: int
    selected_momentum_indices_by_flavor: tuple[tuple[int, tuple[int, ...]], ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.trigger) is not Vituri2024FixedSectorBranchTrigger:
            raise TypeError("branch choice trigger must be typed")
        self.trigger.validate_live_state()
        index = _strict_int(self.canonical_choice_index, "canonical_choice_index")
        count = _strict_int(self.canonical_choice_count, "canonical_choice_count")
        if count != self.trigger.canonical_choice_count or not 0 <= index < count:
            raise ValueError("branch choice index/count does not match trigger")
        _strict_tuple(
            self.selected_momentum_indices_by_flavor,
            "selected_momentum_indices_by_flavor",
        )
        expected_flavors = tuple(item.flavor for item in self.trigger.boundaries)
        selected_flavors: list[int] = []
        canonical_selected: list[tuple[int, tuple[int, ...]]] = []
        for item, boundary in zip(
            self.selected_momentum_indices_by_flavor,
            self.trigger.boundaries,
            strict=True,
        ):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("branch flavor selection must be a pair")
            flavor = _strict_int(item[0], "branch flavor")
            _strict_tuple(item[1], "selected momentum indices")
            selected = tuple(
                _strict_int(value, "selected momentum index") for value in item[1]
            )
            if selected != tuple(sorted(set(selected))):
                raise ValueError("selected momentum indices must be sorted and unique")
            if flavor != boundary.flavor:
                raise ValueError("branch choice flavor binding drifted")
            if len(selected) != boundary.selected_rank or not set(selected).issubset(
                boundary.shell_indices
            ):
                raise ValueError("branch choice is outside its exact shell")
            selected_flavors.append(flavor)
            canonical_selected.append((flavor, selected))
        if tuple(selected_flavors) != expected_flavors:
            raise ValueError("branch choice flavor order drifted")
        aggregate_count = 1
        expected_index = 0
        for boundary, (_, selected) in zip(
            self.trigger.boundaries, canonical_selected, strict=True
        ):
            positions = tuple(boundary.shell_indices.index(value) for value in selected)
            leaf_index = 0
            previous = -1
            for selected_offset, position in enumerate(positions):
                for skipped in range(previous + 1, position):
                    leaf_index += math.comb(
                        len(boundary.shell_indices) - skipped - 1,
                        len(selected) - selected_offset - 1,
                    )
                previous = position
            leaf_count = math.comb(
                len(boundary.shell_indices), boundary.selected_rank
            )
            expected_index = expected_index * leaf_count + leaf_index
            aggregate_count *= leaf_count
        if aggregate_count != count or expected_index != index:
            raise ValueError("branch choice index does not match canonical Cartesian order")
        canonical_tuple = tuple(canonical_selected)
        object.__setattr__(self, "canonical_choice_index", index)
        object.__setattr__(self, "canonical_choice_count", count)
        object.__setattr__(self, "selected_momentum_indices_by_flavor", canonical_tuple)
        payload = {
            "trigger_fingerprint": self.trigger.fingerprint,
            "canonical_choice_index": index,
            "canonical_choice_count": count,
            "selected_momentum_indices_by_flavor": canonical_tuple,
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(self) -> None:
        if replace(self).fingerprint != self.fingerprint:
            raise ValueError("fixed-sector branch choice fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBranchPath:
    choices: tuple[Vituri2024FixedSectorBranchChoice, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.choices) is not tuple:
            raise TypeError("fixed-sector branch path must be a tuple")
        for generation, choice in enumerate(self.choices):
            if type(choice) is not Vituri2024FixedSectorBranchChoice:
                raise TypeError("fixed-sector path entries must be typed choices")
            choice.validate_live_state()
            if choice.trigger.generation != generation:
                raise ValueError("fixed-sector path generations are not ordered")
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint([choice.fingerprint for choice in self.choices]),
        )

    @property
    def path_id(self) -> str:
        if not self.choices:
            return "root"
        indices = "_".join(str(item.canonical_choice_index) for item in self.choices)
        return f"g{indices}_{self.fingerprint[:12]}"

    def validate_live_state(self) -> None:
        if replace(self).fingerprint != self.fingerprint:
            raise ValueError("fixed-sector branch path fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorFreshMap:
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...]
    common_mu_lower_ev: float
    common_mu_upper_ev: float
    common_mu_width_ev: float
    fresh_hamiltonian_sha256: str
    fresh_raw_density_sha256: str

    def __post_init__(self) -> None:
        _strict_tuple(self.boundaries, "fresh-map boundaries")
        if not self.boundaries:
            raise ValueError("fresh map requires partial-flavor boundaries")
        for boundary in self.boundaries:
            if type(boundary) is not Vituri2024FixedSectorBoundary:
                raise TypeError("fresh-map boundary must be typed")
            boundary.validate_live_state()
            if boundary.kind != "unique":
                raise ValueError("fresh map requires unique boundaries")
        for name in ("common_mu_lower_ev", "common_mu_upper_ev", "common_mu_width_ev"):
            _strict_float(getattr(self, name), name)
        if self.common_mu_upper_ev - self.common_mu_lower_ev != self.common_mu_width_ev:
            raise ValueError("fresh-map chemical-potential width drifted")
        if self.common_mu_width_ev <= 0.0:
            raise ValueError("fresh-map chemical-potential interval must be positive")
        _validate_sha256(self.fresh_hamiltonian_sha256, "fresh_hamiltonian_sha256")
        _validate_sha256(self.fresh_raw_density_sha256, "fresh_raw_density_sha256")

    def validate_live_state(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorEndpointMetrics:
    final_raw_norm: float
    engine_reported_final_raw_norm: float
    projector_defect: float
    raw_projector_defect: float
    commutator_residual_ev: float
    hamiltonian_offdiagonal_residual_ev: float
    density_offdiagonal_residual: float
    electron_populations_by_flavor: tuple[float, float, float, float]
    raw_electron_populations_by_flavor: tuple[float, float, float, float]
    population_residual: float
    raw_coordinate_occupations_binary_exact: bool
    raw_populations_exact: bool
    fresh_fock_recompute_residual_ev: float
    fresh_raw_equals_engine_final_raw_exact_bytes: bool
    energy_from_fresh_f_ev: float
    independent_energy_ev: float
    engine_energy_ev: float
    energy_e_f_residual_ev: float

    def __post_init__(self) -> None:
        ungated_energy_fields = {
            "energy_from_fresh_f_ev", "independent_energy_ev", "engine_energy_ev"
        }
        for name in (
            "final_raw_norm",
            "engine_reported_final_raw_norm",
            "projector_defect",
            "raw_projector_defect",
            "commutator_residual_ev",
            "hamiltonian_offdiagonal_residual_ev",
            "density_offdiagonal_residual",
            "population_residual",
            "fresh_fock_recompute_residual_ev",
            "energy_from_fresh_f_ev",
            "independent_energy_ev",
            "engine_energy_ev",
            "energy_e_f_residual_ev",
        ):
            value = _strict_float(getattr(self, name), name)
            if name not in ungated_energy_fields and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "electron_populations_by_flavor",
            "raw_electron_populations_by_flavor",
        ):
            values = _strict_tuple(getattr(self, name), name)
            if len(values) != 4:
                raise ValueError(f"{name} must contain four flavors")
            for value in values:
                _strict_float(value, f"{name} entry")
        for name in (
            "raw_coordinate_occupations_binary_exact",
            "raw_populations_exact",
            "fresh_raw_equals_engine_final_raw_exact_bytes",
        ):
            _strict_bool(getattr(self, name), name)

    def validate_live_state(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorEndpoint:
    path: Vituri2024FixedSectorBranchPath
    outcome: EndpointOutcome
    stationary: bool
    converged: bool
    exit_reason: str
    iterations: int
    consumed_choice_fingerprints: tuple[str, ...]
    iter_energy: Array
    iter_err: Array
    iter_oda: Array
    final_density: Array
    engine_final_raw_density: Array
    fresh_raw_density: Array
    fresh_hamiltonian: Array
    fresh_energies: Array
    engine_energies: Array
    metrics: Vituri2024FixedSectorEndpointMetrics
    fresh_map: Vituri2024FixedSectorFreshMap
    final_density_sha256: str
    fresh_raw_density_sha256: str
    engine_final_raw_density_sha256: str
    final_hamiltonian_sha256: str
    final_energies_sha256: str
    exhaustive_closure: bool = False

    def __post_init__(self) -> None:
        if type(self.path) is not Vituri2024FixedSectorBranchPath:
            raise TypeError("endpoint path must be typed")
        self.path.validate_live_state()
        _strict_bool(self.exhaustive_closure, "exhaustive_closure")
        if self.exhaustive_closure:
            raise ValueError("one fixed-sector path cannot claim exhaustive closure")
        _strict_bool(self.stationary, "stationary")
        _strict_bool(self.converged, "converged")
        if self.outcome not in ("stationary", "normal_endpoint_gate_rejection"):
            raise ValueError("endpoint outcome is not registered")
        if self.stationary != (self.outcome == "stationary"):
            raise ValueError("endpoint outcome/stationarity mismatch")
        if type(self.exit_reason) is not str or not self.exit_reason:
            raise TypeError("endpoint exit_reason must be a nonempty string")
        if self.converged != (self.exit_reason == "converged"):
            raise ValueError("endpoint convergence/exit-reason mismatch")
        iterations = _strict_int(self.iterations, "iterations")
        if iterations < 1:
            raise ValueError("endpoint iterations must be positive")
        _strict_tuple(self.consumed_choice_fingerprints, "consumed choice fingerprints")
        expected_consumed = tuple(item.fingerprint for item in self.path.choices)
        if self.consumed_choice_fingerprints != expected_consumed:
            raise ValueError("endpoint consumed path fingerprint inventory drifted")
        if type(self.metrics) is not Vituri2024FixedSectorEndpointMetrics:
            raise TypeError("endpoint metrics must be typed")
        if type(self.fresh_map) is not Vituri2024FixedSectorFreshMap:
            raise TypeError("endpoint fresh map must be typed")
        self.metrics.validate_live_state()
        self.fresh_map.validate_live_state()
        for name in (
            "iter_energy", "iter_err", "iter_oda", "final_density",
            "engine_final_raw_density", "fresh_raw_density", "fresh_hamiltonian",
            "fresh_energies", "engine_energies",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        histories = (self.iter_energy, self.iter_err, self.iter_oda)
        if any(array.ndim != 1 or array.size != iterations or not np.all(np.isfinite(array)) for array in histories):
            raise ValueError("endpoint iteration histories are inconsistent")
        density_shape = self.final_density.shape
        density_arrays = (
            self.final_density, self.engine_final_raw_density,
            self.fresh_raw_density, self.fresh_hamiltonian,
        )
        if (
            self.final_density.ndim != 3
            or density_shape[:2] != (4, 4)
            or any(array.shape != density_shape for array in density_arrays)
            or not all(np.all(np.isfinite(array)) for array in density_arrays)
        ):
            raise ValueError("endpoint density/Hamiltonian arrays are inconsistent")
        energy_shape = (4, density_shape[2])
        if (
            self.fresh_energies.shape != energy_shape
            or self.engine_energies.shape != energy_shape
            or not np.all(np.isfinite(self.fresh_energies))
            or not np.all(np.isfinite(self.engine_energies))
        ):
            raise ValueError("endpoint energy arrays are inconsistent")
        checks = (
            (self.final_density, self.final_density_sha256),
            (self.fresh_raw_density, self.fresh_raw_density_sha256),
            (self.engine_final_raw_density, self.engine_final_raw_density_sha256),
            (self.fresh_hamiltonian, self.final_hamiltonian_sha256),
            (self.fresh_energies, self.final_energies_sha256),
        )
        for value, expected_hash in checks:
            _validate_sha256(expected_hash, "endpoint array SHA256")
            if _array_sha256(value) != expected_hash:
                raise ValueError("endpoint array hash mismatch")
        if self.fresh_map.fresh_hamiltonian_sha256 != self.final_hamiltonian_sha256:
            raise ValueError("endpoint fresh-map Hamiltonian hash mismatch")
        if self.fresh_map.fresh_raw_density_sha256 != self.fresh_raw_density_sha256:
            raise ValueError("endpoint fresh-map raw-density hash mismatch")
        if self.metrics.final_raw_norm != _max_abs(self.fresh_raw_density - self.final_density):
            raise ValueError("endpoint fresh raw norm metric drifted")
        expected_engine_norm = _relative_density_norm(
            self.engine_final_raw_density, self.final_density
        )
        if self.metrics.engine_reported_final_raw_norm != expected_engine_norm:
            raise ValueError("endpoint engine raw norm metric drifted")
        exact_raw = _array_exact(self.fresh_raw_density, self.engine_final_raw_density)
        if self.metrics.fresh_raw_equals_engine_final_raw_exact_bytes is not exact_raw:
            raise ValueError("endpoint fresh/engine raw equality metric drifted")
        object.__setattr__(self, "iterations", iterations)

    def validate_live_state(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorScientificRejection:
    path: Vituri2024FixedSectorBranchPath
    classification: str
    stage: str
    message: str
    consumed_choice_fingerprints: tuple[str, ...]
    pending_choice_fingerprint: str | None
    evidence_arrays: tuple[tuple[str, Array], ...]
    evidence_hashes: tuple[tuple[str, str], ...] = field(init=False)
    exhaustive_closure: bool = False
    stationary: bool = False

    def __post_init__(self) -> None:
        if type(self.path) is not Vituri2024FixedSectorBranchPath:
            raise TypeError("scientific rejection path must be typed")
        self.path.validate_live_state()
        if self.classification not in VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS:
            raise ValueError("scientific rejection classification is not registered")
        if type(self.stage) is not str or not self.stage or type(self.message) is not str or not self.message:
            raise TypeError("scientific rejection stage/message must be nonempty strings")
        _strict_bool(self.exhaustive_closure, "exhaustive_closure")
        _strict_bool(self.stationary, "stationary")
        if self.exhaustive_closure or self.stationary:
            raise ValueError("scientific rejection cannot claim closure/stationarity")
        _strict_tuple(self.consumed_choice_fingerprints, "consumed choice fingerprints")
        expected = tuple(
            item.fingerprint
            for item in self.path.choices[: len(self.consumed_choice_fingerprints)]
        )
        if self.consumed_choice_fingerprints != expected:
            raise ValueError("scientific rejection consumed path prefix drifted")
        if self.pending_choice_fingerprint is not None:
            _validate_sha256(self.pending_choice_fingerprint, "pending_choice_fingerprint")
            generation = len(self.consumed_choice_fingerprints)
            if generation >= len(self.path.choices) or self.path.choices[generation].fingerprint != self.pending_choice_fingerprint:
                raise ValueError("scientific rejection pending choice drifted")
        _strict_tuple(self.evidence_arrays, "scientific rejection evidence arrays")
        frozen: list[tuple[str, Array]] = []
        hashes: list[tuple[str, str]] = []
        names: list[str] = []
        for item in self.evidence_arrays:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or not item[0]:
                raise TypeError("scientific rejection evidence entry is malformed")
            name = item[0]
            array = _readonly(item[1])
            if not np.all(np.isfinite(array)):
                raise ValueError("scientific rejection evidence must be finite")
            names.append(name)
            frozen.append((name, array))
            hashes.append((name, _array_sha256(array)))
        if names != sorted(set(names)):
            raise ValueError("scientific rejection evidence names must be sorted and unique")
        object.__setattr__(self, "evidence_arrays", tuple(frozen))
        object.__setattr__(self, "evidence_hashes", tuple(hashes))

    def validate_live_state(self) -> None:
        expected_hashes = tuple(
            (name, _array_sha256(array)) for name, array in self.evidence_arrays
        )
        if expected_hashes != self.evidence_hashes:
            raise ValueError("scientific rejection evidence hash drifted")
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorInitializationRejection:
    classification: Literal["h0_positive_subtolerance_splitting_rejection"]
    stage: Literal["common_initializer"]
    message: str
    prepared_fingerprint: str
    policy_fingerprint: str
    evidence_arrays: tuple[tuple[str, Array], ...]
    evidence_hashes: tuple[tuple[str, str], ...] = field(init=False)
    in_process_candidate_only: bool = True
    independent_finite_volume_fixed_sector_full_scf_discriminator: bool = False
    local_hessian_stability_established: bool = False

    def __post_init__(self) -> None:
        if (
            self.classification != "h0_positive_subtolerance_splitting_rejection"
            or self.classification not in VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS
        ):
            raise ValueError("initializer rejection classification is not registered")
        if self.stage != "common_initializer" or type(self.message) is not str or not self.message:
            raise ValueError("initializer rejection stage/message drifted")
        _validate_sha256(self.prepared_fingerprint, "prepared_fingerprint")
        _validate_sha256(self.policy_fingerprint, "policy_fingerprint")
        for name in (
            "in_process_candidate_only",
            "independent_finite_volume_fixed_sector_full_scf_discriminator",
            "local_hessian_stability_established",
        ):
            _strict_bool(getattr(self, name), name)
        if (
            not self.in_process_candidate_only
            or self.independent_finite_volume_fixed_sector_full_scf_discriminator
            or self.local_hessian_stability_established
        ):
            raise ValueError("initializer rejection authority was inflated")
        _strict_tuple(self.evidence_arrays, "initializer rejection evidence arrays")
        frozen: list[tuple[str, Array]] = []
        hashes: list[tuple[str, str]] = []
        names: list[str] = []
        for item in self.evidence_arrays:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or not item[0]:
                raise TypeError("initializer rejection evidence entry is malformed")
            array = _readonly(item[1])
            if not np.all(np.isfinite(array)):
                raise ValueError("initializer rejection evidence must be finite")
            names.append(item[0])
            frozen.append((item[0], array))
            hashes.append((item[0], _array_sha256(array)))
        if names != sorted(set(names)):
            raise ValueError("initializer rejection evidence names must be sorted and unique")
        object.__setattr__(self, "evidence_arrays", tuple(frozen))
        object.__setattr__(self, "evidence_hashes", tuple(hashes))

    def validate_live_state(self) -> None:
        if tuple((name, _array_sha256(array)) for name, array in self.evidence_arrays) != self.evidence_hashes:
            raise ValueError("initializer rejection evidence hash drifted")
        self.__post_init__()

    def metadata_dict(self) -> dict[str, object]:
        return {
            "schema": VITURI2024_FIXED_SECTOR_RESULT_SCHEMA_VERSION,
            "outcome": "initializer_scientific_rejection",
            "classification": self.classification,
            "stage": self.stage,
            "message": self.message,
            "prepared_fingerprint": self.prepared_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "evidence_hashes": dict(self.evidence_hashes),
            "in_process_candidate_only": True,
            "independent_finite_volume_fixed_sector_full_scf_discriminator": False,
            "local_hessian_stability_established": False,
        }

    def array_payload(self) -> dict[str, Array]:
        return {
            f"initializer_rejection_{name}": array
            for name, array in self.evidence_arrays
        }


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBranchFrontier:
    path: Vituri2024FixedSectorBranchPath
    trigger: Vituri2024FixedSectorBranchTrigger
    choices: tuple[Vituri2024FixedSectorBranchChoice, ...]
    exhaustive_closure: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.path) is not Vituri2024FixedSectorBranchPath
            or type(self.trigger) is not Vituri2024FixedSectorBranchTrigger
        ):
            raise TypeError("branch frontier path/trigger must be typed")
        self.path.validate_live_state()
        self.trigger.validate_live_state()
        _strict_tuple(self.choices, "branch frontier choices")
        if not self.choices or any(
            type(choice) is not Vituri2024FixedSectorBranchChoice
            for choice in self.choices
        ):
            raise TypeError("branch frontier choices must be typed and nonempty")
        for index, choice in enumerate(self.choices):
            choice.validate_live_state()
            if (
                choice.trigger != self.trigger
                or choice.canonical_choice_index != index
                or choice.canonical_choice_count != len(self.choices)
            ):
                raise ValueError("branch frontier canonical inventory drifted")
        if self.trigger.generation != len(self.path.choices):
            raise ValueError("branch frontier generation/path mismatch")
        _strict_bool(self.exhaustive_closure, "exhaustive_closure")
        if self.exhaustive_closure:
            raise ValueError("branch frontier cannot claim exhaustive closure")


Vituri2024FixedSectorPathOutcome: TypeAlias = (
    Vituri2024FixedSectorEndpoint
    | Vituri2024FixedSectorScientificRejection
    | Vituri2024FixedSectorBranchFrontier
)


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorBFSNode:
    path: Vituri2024FixedSectorBranchPath
    outcome: str
    child_path_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.path) is not Vituri2024FixedSectorBranchPath:
            raise TypeError("BFS node path must be typed")
        self.path.validate_live_state()
        if type(self.outcome) is not str or not self.outcome:
            raise TypeError("BFS node outcome must be a nonempty string")
        _strict_tuple(self.child_path_ids, "BFS child path IDs")
        if any(type(value) is not str or not value for value in self.child_path_ids):
            raise TypeError("BFS child path IDs must be strings")
        if len(set(self.child_path_ids)) != len(self.child_path_ids):
            raise ValueError("BFS child path IDs must be unique")

    def validate_live_state(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorStationaryGroup:
    density_sha256: str
    hamiltonian_sha256: str
    energies_sha256: str
    path_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("density_sha256", "hamiltonian_sha256", "energies_sha256"):
            _validate_sha256(getattr(self, name), name)
        _strict_tuple(self.path_ids, "stationary group path IDs")
        if not self.path_ids or any(type(value) is not str or not value for value in self.path_ids):
            raise ValueError("stationary group path inventory is invalid")
        if len(set(self.path_ids)) != len(self.path_ids):
            raise ValueError("stationary group path IDs must be unique")

    def validate_live_state(self) -> None:
        self.__post_init__()


def _derive_stationary_groups(
    endpoints: tuple[Vituri2024FixedSectorEndpoint, ...],
) -> tuple[Vituri2024FixedSectorStationaryGroup, ...]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for endpoint in endpoints:
        if endpoint.stationary:
            key = (
                endpoint.final_density_sha256,
                endpoint.final_hamiltonian_sha256,
                endpoint.final_energies_sha256,
            )
            grouped.setdefault(key, []).append(endpoint.path.path_id)
    return tuple(
        Vituri2024FixedSectorStationaryGroup(*key, tuple(path_ids))
        for key, path_ids in sorted(grouped.items())
    )


def _endpoint_satisfies_stationary_gates(
    endpoint: Vituri2024FixedSectorEndpoint,
    policy: Vituri2024FixedSectorPolicy,
) -> bool:
    metrics = endpoint.metrics
    scale = max(
        1.0,
        abs(metrics.energy_from_fresh_f_ev),
        abs(metrics.independent_energy_ev),
        abs(metrics.engine_energy_ev),
    )
    return bool(
        endpoint.converged
        and metrics.final_raw_norm <= policy.final_raw_norm_tolerance
        and metrics.engine_reported_final_raw_norm <= policy.final_raw_norm_tolerance
        and metrics.projector_defect <= policy.idempotency_tolerance
        and metrics.raw_projector_defect <= policy.idempotency_tolerance
        and metrics.commutator_residual_ev <= policy.commutator_tolerance_ev
        and metrics.hamiltonian_offdiagonal_residual_ev <= policy.diagonal_coherence_tolerance_ev
        and metrics.density_offdiagonal_residual <= policy.diagonal_coherence_tolerance_ev
        and metrics.population_residual <= policy.population_tolerance
        and metrics.raw_coordinate_occupations_binary_exact
        and metrics.raw_populations_exact
        and metrics.fresh_raw_equals_engine_final_raw_exact_bytes
        and metrics.fresh_fock_recompute_residual_ev <= policy.fresh_fock_tolerance_ev
        and metrics.energy_e_f_residual_ev <= policy.energy_parity_relative_tolerance * scale
        and all(boundary.gap_ev > 0.0 for boundary in endpoint.fresh_map.boundaries)
        and endpoint.fresh_map.common_mu_width_ev > 0.0
    )


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorSearchResult:
    policy: Vituri2024FixedSectorPolicy
    initializer: Vituri2024FixedSectorInitializer
    prepared_fingerprint: str
    nodes: tuple[Vituri2024FixedSectorBFSNode, ...]
    endpoints: tuple[Vituri2024FixedSectorEndpoint, ...]
    rejections: tuple[Vituri2024FixedSectorScientificRejection, ...]
    stationary_groups: tuple[Vituri2024FixedSectorStationaryGroup, ...]
    replayed_path_count: int
    endpoint_count: int
    branch_tree_exhausted: bool
    unconsumed_frontier_count: int
    all_normal_endpoints_stationary: bool
    exact_stationary_endpoint_array_coalescence: bool
    representative_endpoint: Vituri2024FixedSectorEndpoint | None
    authority: str = VITURI2024_FIXED_SECTOR_AUTHORITY
    in_process_candidate_only: bool = True
    independent_finite_volume_fixed_sector_full_scf_discriminator: bool = False
    local_hessian_stability_established: bool = False
    author_cutoff_identified: bool = False
    uv_plateau_established: bool = False
    unrestricted_ground_state_established: bool = False
    full_paper_reproduction_verified: bool = False
    tdhf_authority: bool = False
    production_authority: bool = False
    visual_match_promotes_authority: bool = False

    def __post_init__(self) -> None:
        self.validate_live_state()

    def validate_live_state(self) -> None:
        if (
            type(self.policy) is not Vituri2024FixedSectorPolicy
            or type(self.initializer) is not Vituri2024FixedSectorInitializer
        ):
            raise TypeError("result policy/initializer must be typed")
        self.policy.validate_live_state()
        self.initializer.validate_live_state()
        _validate_sha256(self.prepared_fingerprint, "prepared_fingerprint")
        if (
            self.prepared_fingerprint != self.initializer.prepared_fingerprint
            or self.policy.fingerprint != self.initializer.policy_fingerprint
        ):
            raise ValueError("result prepared/initializer/policy binding mismatch")
        for name, item_type in (
            ("nodes", Vituri2024FixedSectorBFSNode),
            ("endpoints", Vituri2024FixedSectorEndpoint),
            ("rejections", Vituri2024FixedSectorScientificRejection),
            ("stationary_groups", Vituri2024FixedSectorStationaryGroup),
        ):
            values = _strict_tuple(getattr(self, name), f"result {name}")
            if any(type(value) is not item_type for value in values):
                raise TypeError(f"result {name} contain wrong-typed entries")
            for value in values:
                value.validate_live_state()
        if (
            not _strict_bool(self.branch_tree_exhausted, "branch_tree_exhausted")
            or _strict_int(self.unconsumed_frontier_count, "unconsumed_frontier_count") != 0
        ):
            raise ValueError("a fixed-sector result cannot expose incomplete BFS closure")
        if _strict_int(self.replayed_path_count, "replayed_path_count") != len(self.nodes):
            raise ValueError("fixed-sector replayed-path count mismatch")
        if _strict_int(self.endpoint_count, "endpoint_count") != len(self.endpoints) + len(self.rejections):
            raise ValueError("fixed-sector endpoint count mismatch")
        node_by_path = {node.path.path_id: node for node in self.nodes}
        if (
            len(node_by_path) != len(self.nodes)
            or not self.nodes
            or self.nodes[0].path.path_id != "root"
        ):
            raise ValueError("BFS node path inventory is incomplete or duplicated")
        terminal_outcomes: dict[str, str] = {}
        for endpoint in self.endpoints:
            if endpoint.stationary is not _endpoint_satisfies_stationary_gates(endpoint, self.policy):
                raise ValueError("endpoint stationarity does not derive from policy gates")
            if endpoint.path.path_id in terminal_outcomes:
                raise ValueError("duplicate terminal endpoint path")
            terminal_outcomes[endpoint.path.path_id] = endpoint.outcome
        for rejection in self.rejections:
            if rejection.path.path_id in terminal_outcomes:
                raise ValueError("duplicate terminal rejection path")
            terminal_outcomes[rejection.path.path_id] = rejection.classification
        for node in self.nodes:
            if node.outcome == "expanded_exact_frontier":
                if not node.child_path_ids:
                    raise ValueError("expanded BFS node lacks children")
                for child_id in node.child_path_ids:
                    child = node_by_path.get(child_id)
                    if child is None or child.path.choices[:-1] != node.path.choices:
                        raise ValueError("BFS child path inventory drifted")
            elif node.child_path_ids or terminal_outcomes.get(node.path.path_id) != node.outcome:
                raise ValueError("BFS terminal node inventory drifted")
        expected_terminal_paths = {
            node.path.path_id
            for node in self.nodes
            if node.outcome != "expanded_exact_frontier"
        }
        if set(terminal_outcomes) != expected_terminal_paths:
            raise ValueError("BFS endpoint/path/node inventory drifted")
        expected_all_stationary = all(item.stationary for item in self.endpoints)
        if (
            _strict_bool(
                self.all_normal_endpoints_stationary,
                "all_normal_endpoints_stationary",
            )
            is not expected_all_stationary
        ):
            raise ValueError("all-normal-endpoints stationarity verdict drifted")
        expected_groups = _derive_stationary_groups(self.endpoints)
        if self.stationary_groups != expected_groups:
            raise ValueError("stationary group derivation drifted")
        stationary = tuple(item for item in self.endpoints if item.stationary)
        coalesced = bool(stationary) and len(expected_groups) == 1
        if (
            _strict_bool(
                self.exact_stationary_endpoint_array_coalescence,
                "exact_stationary_endpoint_array_coalescence",
            )
            is not coalesced
        ):
            raise ValueError("stationary coalescence verdict mismatch")
        expected_representative = stationary[0] if coalesced else None
        if self.representative_endpoint is not expected_representative:
            raise ValueError(
                "representative must be the first fully coalesced stationary endpoint"
            )
        if self.authority != VITURI2024_FIXED_SECTOR_AUTHORITY:
            raise ValueError("fixed-sector result authority label drifted")
        authority_bools = (
            self.independent_finite_volume_fixed_sector_full_scf_discriminator,
            self.local_hessian_stability_established,
            self.author_cutoff_identified,
            self.uv_plateau_established,
            self.unrestricted_ground_state_established,
            self.full_paper_reproduction_verified,
            self.tdhf_authority,
            self.production_authority,
            self.visual_match_promotes_authority,
        )
        _strict_bool(self.in_process_candidate_only, "in_process_candidate_only")
        for index, value in enumerate(authority_bools):
            _strict_bool(value, f"authority flag {index}")
        if not self.in_process_candidate_only or any(authority_bools):
            raise ValueError("fixed-sector result authority was inflated")

    def metadata_dict(self) -> dict[str, object]:
        """Return candidate-only metadata without independent postflight claims."""

        return {
            "schema": VITURI2024_FIXED_SECTOR_RESULT_SCHEMA_VERSION,
            "authority": self.authority,
            "prepared_fingerprint": self.prepared_fingerprint,
            "policy_fingerprint": self.policy.fingerprint,
            "initializer_fingerprint": self.initializer.fingerprint,
            "initializer_density_sha256": self.initializer.density_sha256,
            "selected_hole_spin": self.policy.selected_hole_spin,
            "sealed_job461276_reference_selected_hole_spin": 1,
            "sealed_job461276_fixture_parity_applicable": self.policy.selected_hole_spin == 1,
            "generic_symmetry_related_unsealed_candidate": self.policy.selected_hole_spin == -1,
            "replayed_path_count": self.replayed_path_count,
            "endpoint_count": self.endpoint_count,
            "stationary_endpoint_count": sum(item.stationary for item in self.endpoints),
            "scientific_rejection_count": len(self.rejections),
            "branch_tree_exhausted": self.branch_tree_exhausted,
            "unconsumed_frontier_count": self.unconsumed_frontier_count,
            "all_normal_endpoints_stationary": self.all_normal_endpoints_stationary,
            "exact_stationary_endpoint_array_coalescence": self.exact_stationary_endpoint_array_coalescence,
            "representative_path_id": None if self.representative_endpoint is None else self.representative_endpoint.path.path_id,
            "stationary_groups": [asdict(group) for group in self.stationary_groups],
            "authority_flags": {
                "in_process_candidate_only": self.in_process_candidate_only,
                "independent_finite_volume_fixed_sector_full_scf_discriminator": self.independent_finite_volume_fixed_sector_full_scf_discriminator,
                "local_hessian_stability_established": self.local_hessian_stability_established,
                "author_cutoff_identified": self.author_cutoff_identified,
                "uv_plateau_established": self.uv_plateau_established,
                "unrestricted_ground_state_established": self.unrestricted_ground_state_established,
                "full_paper_reproduction_verified": self.full_paper_reproduction_verified,
                "tdhf_authority": self.tdhf_authority,
                "production_authority": self.production_authority,
                "visual_match_promotes_authority": self.visual_match_promotes_authority,
            },
        }

    def array_payload(self) -> dict[str, Array]:
        """Return all retained immutable arrays, including rejection evidence."""

        payload = {"common_initializer_density": self.initializer.density_native}
        for index, endpoint in enumerate(self.endpoints):
            prefix = f"endpoint_{index:04d}"
            for name in (
                "iter_energy", "iter_err", "iter_oda", "final_density",
                "engine_final_raw_density", "fresh_raw_density",
                "fresh_hamiltonian", "fresh_energies", "engine_energies",
            ):
                payload[f"{prefix}_{name}"] = getattr(endpoint, name)
        for index, rejection in enumerate(self.rejections):
            for name, array in rejection.evidence_arrays:
                payload[f"rejection_{index:04d}_{name}"] = array
        return payload


Vituri2024FixedSectorBFSOutcome: TypeAlias = (
    Vituri2024FixedSectorSearchResult | Vituri2024FixedSectorInitializationRejection
)


class _ScientificTerminal(RuntimeError):
    def __init__(
        self,
        classification: str,
        stage: str,
        message: str,
        evidence: dict[str, Array] | None = None,
    ) -> None:
        if classification not in VITURI2024_FIXED_SECTOR_TERMINAL_CLASSIFICATIONS:
            raise ValueError("private scientific terminal classification is not registered")
        if type(stage) is not str or not stage or type(message) is not str or not message:
            raise TypeError("private scientific terminal stage/message must be nonempty")
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.evidence = {} if evidence is None else {
            name: _readonly(value) for name, value in evidence.items()
        }


class _BranchFrontierSignal(RuntimeError):
    def __init__(self, frontier: Vituri2024FixedSectorBranchFrontier):
        super().__init__("exact fixed-sector branch frontier")
        self.frontier = frontier


def _validate_prepared_policy(
    prepared: Vituri2024PreparedHomogeneousHF,
    policy: Vituri2024FixedSectorPolicy,
) -> None:
    if type(prepared) is not Vituri2024PreparedHomogeneousHF:
        raise TypeError("prepared must be Vituri2024PreparedHomogeneousHF")
    if type(policy) is not Vituri2024FixedSectorPolicy:
        raise TypeError("policy must be Vituri2024FixedSectorPolicy")
    prepared.validate_live_state()
    policy.validate_live_state()


def _boundary_partition(values: Array, count: int) -> tuple[float, float, float]:
    if not 0 < count < values.size:
        raise ValueError("partial-flavor electron count lies outside the mesh")
    selected = np.partition(np.asarray(values, dtype=np.float64), (count - 1, count))
    lower = float(selected[count - 1])
    upper = float(selected[count])
    return lower, upper, upper - lower


def _analyze_fixed_sector_boundary_unchecked(
    prepared: Vituri2024PreparedHomogeneousHF,
    hamiltonian: Array,
    *,
    flavor: int,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorBoundary:
    flavor = _strict_int(flavor, "flavor")
    if flavor not in policy.partial_flavors:
        raise ValueError("boundary analysis is defined only for partial flavors")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    expected = (4, 4, prepared.spec.nk)
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise ValueError("fixed-sector Hamiltonian shape/finiteness failed")
    values = np.asarray(matrix[flavor, flavor].real, dtype=np.float64)
    count = policy.electron_counts(prepared.spec.nk, prepared.spec.holes_per_valley)[flavor]
    lower, upper, gap = _boundary_partition(values, count)
    tolerance = float(
        max(
            policy.boundary_floor_ev,
            policy.boundary_roundoff_multiplier
            * np.finfo(np.float64).eps
            * _max_abs(matrix),
        )
    )
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError("fixed-sector boundary gap is negative or nonfinite")
    if gap > tolerance:
        occupied = tuple(int(x) for x in np.flatnonzero(values <= lower))
        return Vituri2024FixedSectorBoundary(
            flavor=flavor,
            electron_count=count,
            kind="unique",
            lower_ev=lower,
            upper_ev=upper,
            gap_ev=gap,
            effective_tolerance_ev=tolerance,
            occupied_indices=occupied,
        )
    if 0.0 < gap <= tolerance:
        return Vituri2024FixedSectorBoundary(
            flavor=flavor,
            electron_count=count,
            kind="positive_subtolerance",
            lower_ev=lower,
            upper_ev=upper,
            gap_ev=gap,
            effective_tolerance_ev=tolerance,
        )
    if gap != 0.0:
        raise ValueError("positive subtolerance classification escaped")
    below = tuple(int(x) for x in np.flatnonzero(values < lower))
    shell = tuple(int(x) for x in np.flatnonzero(values == lower))
    rank = count - len(below)
    return Vituri2024FixedSectorBoundary(
        flavor=flavor,
        electron_count=count,
        kind="exact",
        lower_ev=lower,
        upper_ev=upper,
        gap_ev=gap,
        effective_tolerance_ev=tolerance,
        strictly_below_indices=below,
        shell_indices=shell,
        selected_rank=rank,
    )


def analyze_vituri2024_fixed_sector_boundary(
    prepared: Vituri2024PreparedHomogeneousHF,
    hamiltonian: Array,
    *,
    flavor: int,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorBoundary:
    """Classify one partial-flavor boundary without stable-sort tie breaking."""

    _validate_prepared_policy(prepared, policy)
    return _analyze_fixed_sector_boundary_unchecked(
        prepared, hamiltonian, flavor=flavor, policy=policy
    )


def _mirror_indices(labels: Array) -> Array:
    lookup = {(int(ix), int(iy)): index for index, (ix, iy) in enumerate(labels)}
    try:
        mirror = np.asarray(
            [lookup[(int(ix), -int(iy))] for ix, iy in labels], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("Vituri Cartesian labels lack mirror closure") from error
    if not np.array_equal(mirror[mirror], np.arange(labels.shape[0])):
        raise ValueError("Vituri Cartesian mirror is not an involution")
    return mirror


def _build_vituri2024_fixed_sector_initializer(
    prepared: Vituri2024PreparedHomogeneousHF,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorInitializer:
    """Build the common mirror-symmetric exact-shell mixed initializer."""

    _validate_prepared_policy(prepared, policy)
    counts = policy.electron_counts(prepared.spec.nk, prepared.spec.holes_per_valley)
    conventional = np.zeros((4, 4, prepared.spec.nk), dtype=np.complex128)
    for flavor in policy.full_flavors:
        conventional[flavor, flavor, :] = 1.0
    boundaries: list[Vituri2024FixedSectorBoundary] = []
    for flavor in policy.partial_flavors:
        boundary = _analyze_fixed_sector_boundary_unchecked(
            prepared, prepared.h0_native, flavor=flavor, policy=policy
        )
        if boundary.kind == "positive_subtolerance":
            raise _ScientificTerminal(
                "h0_positive_subtolerance_splitting_rejection",
                "common_initializer",
                f"flavor {flavor} h0 boundary has positive subtolerance splitting",
                {"h0": np.asarray(prepared.h0_native).copy()},
            )
        if boundary.kind == "unique":
            conventional[flavor, flavor, list(boundary.occupied_indices)] = 1.0
        else:
            conventional[flavor, flavor, list(boundary.strictly_below_indices)] = 1.0
            fraction = boundary.selected_rank / len(boundary.shell_indices)
            conventional[flavor, flavor, list(boundary.shell_indices)] = fraction
        boundaries.append(boundary)
    populations = tuple(float(np.real(conventional[f, f]).sum()) for f in range(4))
    if any(abs(value - target) > 1.0e-12 for value, target in zip(populations, counts, strict=True)):
        raise ValueError("fixed-sector common initializer population mismatch")
    mirror = _mirror_indices(prepared.integer_mesh_labels)
    mirror_symmetric = _max_abs(conventional - conventional[:, :, mirror]) <= 1.0e-15
    if not mirror_symmetric:
        raise ValueError("fixed-sector common initializer is not mirror symmetric")
    native = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    native = _readonly(native, np.dtype(np.complex128))
    return Vituri2024FixedSectorInitializer(
        density_native=native,
        density_sha256=_array_sha256(native),
        h0_sha256=_array_sha256(prepared.h0_native),
        electron_counts_by_flavor=counts,
        boundaries=tuple(boundaries),
        mirror_symmetric=True,
        prepared_fingerprint=prepared.fingerprint,
        policy_fingerprint=policy.fingerprint,
    )


def build_vituri2024_fixed_sector_initializer(
    prepared: Vituri2024PreparedHomogeneousHF,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorInitializer | Vituri2024FixedSectorInitializationRejection:
    """Build the initializer or return immutable typed h0 rejection evidence."""

    _validate_prepared_policy(prepared, policy)
    try:
        return _build_vituri2024_fixed_sector_initializer(prepared, policy)
    except _ScientificTerminal as error:
        if error.classification != "h0_positive_subtolerance_splitting_rejection":
            raise
        return Vituri2024FixedSectorInitializationRejection(
            classification="h0_positive_subtolerance_splitting_rejection",
            stage="common_initializer",
            message=str(error),
            prepared_fingerprint=prepared.fingerprint,
            policy_fingerprint=policy.fingerprint,
            evidence_arrays=tuple(sorted(error.evidence.items())),
        )


def _enumerate_fixed_sector_branch_choices_unchecked(
    prepared: Vituri2024PreparedHomogeneousHF,
    hamiltonian: Array,
    previous_density_native: Array,
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...],
    *,
    generation: int,
    policy: Vituri2024FixedSectorPolicy,
) -> tuple[Vituri2024FixedSectorBranchChoice, ...]:
    generation = _strict_int(generation, "generation")
    if generation < 0 or generation >= policy.maximum_generation:
        raise RuntimeError("maximum fixed-sector branch generation reached")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    previous = np.asarray(previous_density_native, dtype=np.complex128)
    if matrix.shape != (4, 4, prepared.spec.nk) or previous.shape != matrix.shape:
        raise ValueError("fixed-sector branch trigger array shape mismatch")
    _strict_tuple(boundaries, "fixed-sector boundaries")
    if any(type(item) is not Vituri2024FixedSectorBoundary for item in boundaries):
        raise TypeError("fixed-sector boundaries must be typed")
    exact = tuple(item for item in boundaries if item.kind == "exact")
    if not exact:
        raise ValueError("branch enumeration requires at least one exact boundary")
    expected_exact_flavors = tuple(
        flavor
        for flavor in policy.partial_flavors
        if any(boundary.flavor == flavor and boundary.kind == "exact" for boundary in boundaries)
    )
    if tuple(item.flavor for item in exact) != expected_exact_flavors:
        raise ValueError("exact boundaries are not in partial-flavor order")
    fock_hash = _array_sha256(matrix)
    previous_hash = _array_sha256(previous)
    per_flavor_counts = tuple(
        math.comb(len(boundary.shell_indices), boundary.selected_rank)
        for boundary in exact
    )
    aggregate_count = math.prod(per_flavor_counts)
    if aggregate_count > policy.maximum_choices_per_trigger:
        raise RuntimeError("fixed-sector Cartesian branch choice cap exceeded")
    # Materialize only after the declared aggregate Cartesian-product cap passes.
    leaves = tuple(
        tuple(combinations(boundary.shell_indices, boundary.selected_rank))
        for boundary in exact
    )
    if tuple(len(items) for items in leaves) != per_flavor_counts:
        raise RuntimeError("canonical combination inventory drifted")
    trigger = Vituri2024FixedSectorBranchTrigger(
        generation=generation,
        exact_fock_sha256=fock_hash,
        previous_density_sha256=previous_hash,
        boundaries=exact,
        canonical_choice_count=aggregate_count,
    )
    result: list[Vituri2024FixedSectorBranchChoice] = []
    for index, selected in enumerate(product(*leaves)):
        selected_by_flavor = tuple(
            (boundary.flavor, tuple(indices))
            for boundary, indices in zip(exact, selected, strict=True)
        )
        result.append(
            Vituri2024FixedSectorBranchChoice(
                trigger=trigger,
                canonical_choice_index=index,
                canonical_choice_count=aggregate_count,
                selected_momentum_indices_by_flavor=selected_by_flavor,
            )
        )
    if len(result) != aggregate_count:
        raise RuntimeError("aggregate Cartesian branch inventory drifted")
    return tuple(result)


def enumerate_vituri2024_fixed_sector_branch_choices(
    prepared: Vituri2024PreparedHomogeneousHF,
    hamiltonian: Array,
    previous_density_native: Array,
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...],
    *,
    generation: int,
    policy: Vituri2024FixedSectorPolicy,
) -> tuple[Vituri2024FixedSectorBranchChoice, ...]:
    """Enumerate the simultaneous per-flavor Cartesian branch product."""

    _validate_prepared_policy(prepared, policy)
    return _enumerate_fixed_sector_branch_choices_unchecked(
        prepared,
        hamiltonian,
        previous_density_native,
        boundaries,
        generation=generation,
        policy=policy,
    )


def _validate_hamiltonian(
    prepared: Vituri2024PreparedHomogeneousHF,
    matrix: Array,
    policy: Vituri2024FixedSectorPolicy,
) -> None:
    if matrix.shape != (4, 4, prepared.spec.nk) or not np.all(np.isfinite(matrix)):
        raise ValueError("fixed-sector Hamiltonian shape/finiteness failed")
    hermiticity = _max_abs(matrix - matrix.swapaxes(0, 1).conj())
    if hermiticity > policy.hamiltonian_hermiticity_tolerance_ev:
        raise ValueError("fixed-sector Hamiltonian Hermiticity failed")
    off = matrix.copy()
    for flavor in range(4):
        off[flavor, flavor, :] = 0.0
    residual = _max_abs(off)
    if residual > policy.diagonal_coherence_tolerance_ev:
        raise _ScientificTerminal(
            "diagonal_coherence_rejection",
            "density_update",
            f"off-diagonal Fock residual {residual} exceeds fixed-sector gate",
            {"hamiltonian": matrix.copy()},
        )


def _raw_density_from_boundaries(
    prepared: Vituri2024PreparedHomogeneousHF,
    policy: Vituri2024FixedSectorPolicy,
    boundaries: tuple[Vituri2024FixedSectorBoundary, ...],
    choice: Vituri2024FixedSectorBranchChoice | None,
) -> tuple[Array, float, float]:
    matrix = np.zeros((4, 4, prepared.spec.nk), dtype=np.complex128)
    for flavor in policy.full_flavors:
        matrix[flavor, flavor, :] = 1.0
    choice_map = {} if choice is None else dict(choice.selected_momentum_indices_by_flavor)
    intervals: list[tuple[float, float]] = []
    for boundary in boundaries:
        if boundary.kind == "positive_subtolerance":
            raise RuntimeError("rejected boundary reached raw projector builder")
        if boundary.kind == "unique":
            occupied = boundary.occupied_indices
        else:
            if choice is None or boundary.flavor not in choice_map:
                raise RuntimeError("exact fixed-sector boundary lacks branch choice")
            selected = choice_map[boundary.flavor]
            if len(selected) != boundary.selected_rank or not set(selected).issubset(boundary.shell_indices):
                raise ValueError("fixed-sector branch selected invalid shell coordinates")
            occupied = boundary.strictly_below_indices + selected
        if len(occupied) != boundary.electron_count or len(set(occupied)) != len(occupied):
            raise ValueError("fixed-sector coordinate branch changed rank")
        matrix[boundary.flavor, boundary.flavor, list(occupied)] = 1.0
        intervals.append((boundary.lower_ev, boundary.upper_ev))
    native = np.asarray(
        vituri2024_conventional_k_diagonal_to_native_density(matrix),
        dtype=np.complex128,
    )
    return native, max(item[0] for item in intervals), min(item[1] for item in intervals)


def _fresh_unique_map(
    prepared: Vituri2024PreparedHomogeneousHF,
    hamiltonian: Array,
    policy: Vituri2024FixedSectorPolicy,
) -> tuple[Array, Vituri2024FixedSectorFreshMap]:
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    _validate_hamiltonian(prepared, matrix, policy)
    boundaries = tuple(
        _analyze_fixed_sector_boundary_unchecked(
            prepared, matrix, flavor=flavor, policy=policy
        )
        for flavor in policy.partial_flavors
    )
    if any(item.kind != "unique" for item in boundaries):
        raise _ScientificTerminal(
            "fresh_final_fock_boundary_rejection",
            "exact_final_recomputation",
            "fresh final fixed-sector boundary is not unique",
            {"fresh_hamiltonian": matrix.copy()},
        )
    raw, lower, upper = _raw_density_from_boundaries(prepared, policy, boundaries, None)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise _ScientificTerminal(
            "fresh_final_fock_common_mu_rejection",
            "exact_final_recomputation",
            "fresh final partial flavors have no positive common chemical-potential interval",
            {"fresh_hamiltonian": matrix.copy(), "fresh_raw_density": raw.copy()},
        )
    fresh_map = Vituri2024FixedSectorFreshMap(
        boundaries=boundaries,
        common_mu_lower_ev=lower,
        common_mu_upper_ev=upper,
        common_mu_width_ev=upper - lower,
        fresh_hamiltonian_sha256=_array_sha256(matrix),
        fresh_raw_density_sha256=_array_sha256(raw),
    )
    return raw, fresh_map


def _stationary_metrics(
    prepared: Vituri2024PreparedHomogeneousHF,
    policy: Vituri2024FixedSectorPolicy,
    density_native: Array,
    raw_native: Array,
    engine_raw_native: Array,
    hamiltonian: Array,
    *,
    engine_reported_final_raw_norm: float,
    fresh_fock_residual: float,
    energy_from_fresh_f: float,
    independent_energy: float,
    engine_energy: float,
) -> Vituri2024FixedSectorEndpointMetrics:
    density = vituri2024_native_density_to_conventional_k_diagonal(density_native)
    raw = vituri2024_native_density_to_conventional_k_diagonal(raw_native)
    d2 = np.einsum("abk,bck->ack", density, density, optimize=True)
    r2 = np.einsum("abk,bck->ack", raw, raw, optimize=True)
    comm = np.einsum("abk,bck->ack", hamiltonian, density, optimize=True)
    comm -= np.einsum("abk,bck->ack", density, hamiltonian, optimize=True)
    off_h = np.asarray(hamiltonian).copy()
    off_d = np.asarray(density).copy()
    for flavor in range(4):
        off_h[flavor, flavor, :] = 0.0
        off_d[flavor, flavor, :] = 0.0
    expected = policy.electron_counts(prepared.spec.nk, prepared.spec.holes_per_valley)
    populations = tuple(float(np.real(density[f, f]).sum()) for f in range(4))
    raw_populations = tuple(float(np.real(raw[f, f]).sum()) for f in range(4))
    raw_diag = np.real(np.diagonal(raw, axis1=0, axis2=1)).T
    return Vituri2024FixedSectorEndpointMetrics(
        final_raw_norm=_max_abs(raw_native - density_native),
        engine_reported_final_raw_norm=float(engine_reported_final_raw_norm),
        projector_defect=_max_abs(d2 - density),
        raw_projector_defect=_max_abs(r2 - raw),
        commutator_residual_ev=_max_abs(comm),
        hamiltonian_offdiagonal_residual_ev=_max_abs(off_h),
        density_offdiagonal_residual=_max_abs(off_d),
        electron_populations_by_flavor=populations,
        raw_electron_populations_by_flavor=raw_populations,
        population_residual=max(abs(x - y) for x, y in zip(populations, expected, strict=True)),
        raw_coordinate_occupations_binary_exact=bool(np.all((raw_diag == 0.0) | (raw_diag == 1.0))),
        raw_populations_exact=all(x == float(y) for x, y in zip(raw_populations, expected, strict=True)),
        fresh_fock_recompute_residual_ev=fresh_fock_residual,
        fresh_raw_equals_engine_final_raw_exact_bytes=_array_exact(
            raw_native, engine_raw_native
        ),
        energy_from_fresh_f_ev=energy_from_fresh_f,
        independent_energy_ev=independent_energy,
        engine_energy_ev=engine_energy,
        energy_e_f_residual_ev=max(
            abs(energy_from_fresh_f - independent_energy),
            abs(energy_from_fresh_f - engine_energy),
        ),
    )


def _scientific_record(
    path: Vituri2024FixedSectorBranchPath,
    error: _ScientificTerminal,
    consumed: list[Vituri2024FixedSectorBranchChoice],
    pending: dict[str, object] | None,
) -> Vituri2024FixedSectorScientificRejection:
    consumed_fingerprints = tuple(item.fingerprint for item in consumed)
    expected = tuple(item.fingerprint for item in path.choices[: len(consumed)])
    if consumed_fingerprints != expected:
        raise RuntimeError("scientific rejection consumed path prefix mismatch")
    pending_choice = None if pending is None else pending.get("choice")
    pending_fingerprint = None
    if pending_choice is not None:
        if type(pending_choice) is not Vituri2024FixedSectorBranchChoice:
            raise RuntimeError("scientific rejection pending choice is malformed")
        generation = len(consumed)
        if generation >= len(path.choices) or path.choices[generation] != pending_choice:
            raise RuntimeError("scientific rejection pending path choice mismatch")
        pending_fingerprint = pending_choice.fingerprint
    return Vituri2024FixedSectorScientificRejection(
        path=path,
        classification=error.classification,
        stage=error.stage,
        message=str(error),
        consumed_choice_fingerprints=consumed_fingerprints,
        pending_choice_fingerprint=pending_fingerprint,
        evidence_arrays=tuple(sorted(error.evidence.items())),
    )


def run_vituri2024_fixed_sector_path(
    prepared: Vituri2024PreparedHomogeneousHF,
    initializer: Vituri2024FixedSectorInitializer,
    path: Vituri2024FixedSectorBranchPath,
    *,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorPathOutcome:
    """Replay one ordered path.

    This low-level result always has ``exhaustive_closure=False`` when it is a
    frontier or rejection.  A caller requiring closure must use
    :func:`run_vituri2024_fixed_sector_bfs`.
    """

    _validate_prepared_policy(prepared, policy)
    if type(initializer) is not Vituri2024FixedSectorInitializer:
        raise TypeError("initializer must be Vituri2024FixedSectorInitializer")
    if type(path) is not Vituri2024FixedSectorBranchPath:
        raise TypeError("path must be Vituri2024FixedSectorBranchPath")
    initializer.validate_live_state()
    path.validate_live_state()
    if initializer.prepared_fingerprint != prepared.fingerprint or initializer.policy_fingerprint != policy.fingerprint:
        raise ValueError("fixed-sector initializer preparation/policy binding mismatch")

    rebuilt = _build_vituri2024_fixed_sector_initializer(prepared, policy)
    if (
        not _array_exact(initializer.density_native, rebuilt.density_native)
        or initializer.density_sha256 != rebuilt.density_sha256
        or initializer.h0_sha256 != rebuilt.h0_sha256
        or initializer.h0_sha256 != _array_sha256(prepared.h0_native)
        or initializer.electron_counts_by_flavor != rebuilt.electron_counts_by_flavor
        or initializer.boundaries != rebuilt.boundaries
        or initializer.mirror_symmetric is not rebuilt.mirror_symmetric
        or initializer.prepared_fingerprint != rebuilt.prepared_fingerprint
        or initializer.policy_fingerprint != rebuilt.policy_fingerprint
        or initializer.fingerprint != rebuilt.fingerprint
    ):
        raise ValueError("fixed-sector initializer does not exactly match independent rebuild")

    state = make_vituri2024_hf_state(prepared)
    interaction_action = prepared.functional.make_validated_interaction_action()
    consumed: list[Vituri2024FixedSectorBranchChoice] = []
    pending: dict[str, object] | None = None
    final_updates: list[DensityUpdateResult] = []

    def initialize(target: Vituri2024HFState, *, init_mode: str, seed: int) -> None:
        if target is not state or init_mode != policy.initializer_mode or seed != policy.initializer_seed:
            raise RuntimeError("fixed-sector common initializer call binding drifted")
        target.density[:, :, :] = initializer.density_native
        target.hamiltonian[:, :, :] = target.h0
        target.energies[:, :] = np.nan
        target.mu = float("nan")
        target.diagnostics.clear()
        target.diagnostics["fixed_sector_initializer_hash_bound"] = 1.0

    def engine_would_be_in_final_map() -> bool:
        if not state.diagnostics.get("fixed_sector_last_iteration"):
            return False
        return bool(
            state.diagnostics.get("fixed_sector_last_raw_norm", math.inf) <= state.precision
            or state.diagnostics.get("fixed_sector_last_oda_lambda", 1.0) < policy.oda_stall_threshold
            or int(state.diagnostics["fixed_sector_last_iteration"]) >= policy.max_iter
        )

    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        nonlocal pending
        if pending is not None:
            raise RuntimeError("fixed-sector density-builder receipt was not consumed")
        matrix = np.asarray(hamiltonian, dtype=np.complex128)
        _validate_hamiltonian(prepared, matrix, policy)
        previous = np.asarray(state.density, dtype=np.complex128).copy()
        boundaries = tuple(
            _analyze_fixed_sector_boundary_unchecked(
                prepared, matrix, flavor=flavor, policy=policy
            )
            for flavor in policy.partial_flavors
        )
        if any(item.kind == "positive_subtolerance" for item in boundaries):
            raise _ScientificTerminal(
                "positive_subtolerance_splitting_rejection",
                "density_update",
                "positive partial-flavor splitting lies below the effective tolerance",
                {"density": previous, "hamiltonian": matrix.copy()},
            )
        exact = tuple(item for item in boundaries if item.kind == "exact")
        generation = len(consumed)
        choice: Vituri2024FixedSectorBranchChoice | None = None
        if exact:
            choices = _enumerate_fixed_sector_branch_choices_unchecked(
                prepared,
                matrix,
                previous,
                boundaries,
                generation=generation,
                policy=policy,
            )
            if generation >= len(path.choices):
                if engine_would_be_in_final_map():
                    raise _ScientificTerminal(
                        "branch_frontier_in_final_map_rejection",
                        "final_density_recomputation",
                        "an unresolved branch frontier appeared in the final map",
                        {"density": previous, "hamiltonian": matrix.copy()},
                    )
                trigger = choices[0].trigger
                raise _BranchFrontierSignal(
                    Vituri2024FixedSectorBranchFrontier(path, trigger, choices)
                )
            choice = path.choices[generation]
            index = choice.canonical_choice_index
            if (
                choice.trigger.fingerprint != choices[0].trigger.fingerprint
                or index >= len(choices)
                or choice != choices[index]
            ):
                raise ValueError("fixed-sector ordered path trigger/choice mismatch")
        raw, common_lower, common_upper = _raw_density_from_boundaries(
            prepared, policy, boundaries, choice
        )
        energies = np.asarray(
            np.real(np.diagonal(matrix, axis1=0, axis2=1)).T,
            dtype=np.float64,
        )
        update = DensityUpdateResult(
            density=raw,
            energies=energies,
            mu=0.5 * (common_lower + common_upper),
            observables={
                "common_mu_lower_ev": common_lower,
                "common_mu_upper_ev": common_upper,
                "common_mu_width_ev": common_upper - common_lower,
            },
        )
        pending = {
            "update_id": id(update),
            "raw_hash": _array_sha256(raw),
            "fock_hash": _array_sha256(matrix),
            "previous_hash": _array_sha256(previous),
            "choice": choice,
        }
        return update

    def step_callback(target: Vituri2024HFState, step: object) -> None:
        nonlocal pending
        if target is not state:
            raise RuntimeError("fixed-sector step callback target identity drifted")
        if pending is None or pending["update_id"] != id(step.density_update):
            raise RuntimeError("fixed-sector step callback lost builder receipt")
        if pending["raw_hash"] != _array_sha256(step.density_update.density):
            raise RuntimeError("fixed-sector raw-density binding failed")
        if pending["fock_hash"] != _array_sha256(step.total_hamiltonian):
            raise RuntimeError("fixed-sector exact-Fock binding failed")
        if pending["previous_hash"] != _array_sha256(step.previous_density):
            raise RuntimeError("fixed-sector previous-density binding failed")
        expected = step.oda_lambda * step.density_update.density + (1.0 - step.oda_lambda) * step.previous_density
        if not np.array_equal(expected, step.mixed_density) or not np.array_equal(target.density, step.mixed_density):
            raise RuntimeError("fixed-sector generic ODA applied-state binding failed")
        choice = pending["choice"]
        if choice is not None:
            if step.oda_lambda <= 0.0 or np.array_equal(step.previous_density, step.mixed_density):
                raise _ScientificTerminal(
                    "branch_choice_not_applied_rejection",
                    "step_callback",
                    "an exact branch choice did not change the applied density",
                    {"raw_density": step.density_update.density.copy()},
                )
            consumed.append(choice)
        target.diagnostics["fixed_sector_last_iteration"] = float(step.iteration)
        target.diagnostics["fixed_sector_last_raw_norm"] = float(step.norm_raw)
        target.diagnostics["fixed_sector_last_oda_lambda"] = float(step.oda_lambda)
        pending = None

    def final_callback(target: Vituri2024HFState, update: DensityUpdateResult) -> None:
        nonlocal pending
        if target is not state or pending is None or pending["update_id"] != id(update):
            raise RuntimeError("fixed-sector final callback lost builder receipt")
        if pending["choice"] is not None:
            raise _ScientificTerminal(
                "branch_choice_in_final_map_rejection",
                "final_density_recomputation",
                "a registered branch choice was consumed only by the final map",
                {"final_raw_density": update.density.copy()},
            )
        final_updates.append(update)
        pending = None

    problem = HartreeFockProblem(
        initializer=initialize,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_action,
            density_builder=density_builder,
            energy_functional=_energy_from_engine_inputs,
            oda_delta_interaction_builder=interaction_action,
            step_callback=step_callback,
            final_state_callback=final_callback,
            convergence_rule="raw",
        ),
    )
    try:
        run = run_hartree_fock_problem(
            state,
            problem,
            init_mode=policy.initializer_mode,
            seed=policy.initializer_seed,
            max_iter=policy.max_iter,
            oda_stall_threshold=policy.oda_stall_threshold,
            max_oda_lambda=policy.max_oda_lambda,
        )
    except _BranchFrontierSignal as signal:
        return signal.frontier
    except _ScientificTerminal as error:
        return _scientific_record(path, error, consumed, pending)

    if pending is not None or len(final_updates) != 1 or len(consumed) != len(path.choices):
        raise RuntimeError("fixed-sector endpoint did not consume exactly its ordered path")
    engine_final_raw = np.asarray(final_updates[0].density, dtype=np.complex128)
    fresh_h = np.asarray(prepared.functional.fock(run.state.density), dtype=np.complex128)
    fresh_residual = _max_abs(fresh_h - run.state.hamiltonian)
    try:
        fresh_raw, fresh_map = _fresh_unique_map(prepared, fresh_h, policy)
    except _ScientificTerminal as error:
        error.evidence.setdefault("engine_final_raw_density", engine_final_raw.copy())
        return _scientific_record(path, error, consumed, pending)
    fresh_energies = np.asarray(
        np.real(np.diagonal(fresh_h, axis1=0, axis2=1)).T,
        dtype=np.float64,
    )
    independent_energy = float(prepared.functional.energy(run.state.density))
    engine_energy = float(run.state.diagnostics["hf_energy"])
    energy_from_fresh_f = float(
        np.real(
            np.einsum("abk,abk->", prepared.h0_native, run.state.density, optimize=False)
            + 0.5
            * np.einsum(
                "abk,abk->",
                fresh_h - prepared.h0_native,
                run.state.density,
                optimize=False,
            )
        )
    )
    metrics = _stationary_metrics(
        prepared,
        policy,
        np.asarray(run.state.density),
        fresh_raw,
        engine_final_raw,
        fresh_h,
        engine_reported_final_raw_norm=float(run.state.diagnostics["final_raw_norm"]),
        fresh_fock_residual=fresh_residual,
        energy_from_fresh_f=energy_from_fresh_f,
        independent_energy=independent_energy,
        engine_energy=engine_energy,
    )
    scale = max(1.0, abs(energy_from_fresh_f), abs(independent_energy), abs(engine_energy))
    stationary = bool(
        run.converged
        and metrics.final_raw_norm <= policy.final_raw_norm_tolerance
        and metrics.engine_reported_final_raw_norm <= policy.final_raw_norm_tolerance
        and metrics.projector_defect <= policy.idempotency_tolerance
        and metrics.raw_projector_defect <= policy.idempotency_tolerance
        and metrics.commutator_residual_ev <= policy.commutator_tolerance_ev
        and metrics.hamiltonian_offdiagonal_residual_ev <= policy.diagonal_coherence_tolerance_ev
        and metrics.density_offdiagonal_residual <= policy.diagonal_coherence_tolerance_ev
        and metrics.population_residual <= policy.population_tolerance
        and metrics.raw_coordinate_occupations_binary_exact
        and metrics.raw_populations_exact
        and metrics.fresh_raw_equals_engine_final_raw_exact_bytes
        and metrics.fresh_fock_recompute_residual_ev <= policy.fresh_fock_tolerance_ev
        and metrics.energy_e_f_residual_ev <= policy.energy_parity_relative_tolerance * scale
        and all(boundary.gap_ev > 0.0 for boundary in fresh_map.boundaries)
        and fresh_map.common_mu_width_ev > 0.0
    )
    return Vituri2024FixedSectorEndpoint(
        path=path,
        outcome="stationary" if stationary else "normal_endpoint_gate_rejection",
        stationary=stationary,
        converged=run.converged,
        exit_reason=run.exit_reason,
        iterations=run.iterations,
        consumed_choice_fingerprints=tuple(item.fingerprint for item in consumed),
        iter_energy=run.iter_energy,
        iter_err=run.iter_err,
        iter_oda=run.iter_oda,
        final_density=np.asarray(run.state.density),
        engine_final_raw_density=engine_final_raw,
        fresh_raw_density=fresh_raw,
        fresh_hamiltonian=fresh_h,
        fresh_energies=fresh_energies,
        engine_energies=np.asarray(run.state.energies),
        metrics=metrics,
        fresh_map=fresh_map,
        final_density_sha256=_array_sha256(run.state.density),
        fresh_raw_density_sha256=_array_sha256(fresh_raw),
        engine_final_raw_density_sha256=_array_sha256(engine_final_raw),
        final_hamiltonian_sha256=_array_sha256(fresh_h),
        final_energies_sha256=_array_sha256(fresh_energies),
    )


def run_vituri2024_fixed_sector_bfs(
    prepared: Vituri2024PreparedHomogeneousHF,
    *,
    policy: Vituri2024FixedSectorPolicy,
) -> Vituri2024FixedSectorBFSOutcome:
    """Return a closed in-process candidate receipt or typed init rejection.

    Independent finite-volume authority remains external sealed job461276.
    """

    _validate_prepared_policy(prepared, policy)
    initializer_outcome = build_vituri2024_fixed_sector_initializer(prepared, policy)
    if isinstance(
        initializer_outcome, Vituri2024FixedSectorInitializationRejection
    ):
        return initializer_outcome
    initializer = initializer_outcome
    root = Vituri2024FixedSectorBranchPath()
    queue: deque[Vituri2024FixedSectorBranchPath] = deque((root,))
    queued = {root.fingerprint}
    nodes: list[Vituri2024FixedSectorBFSNode] = []
    endpoints: list[Vituri2024FixedSectorEndpoint] = []
    rejections: list[Vituri2024FixedSectorScientificRejection] = []
    while queue:
        if len(nodes) >= policy.maximum_replayed_paths:
            raise RuntimeError("fixed-sector replay cap reached with unresolved frontier")
        path = queue.popleft()
        outcome = run_vituri2024_fixed_sector_path(
            prepared, initializer, path, policy=policy
        )
        if isinstance(outcome, Vituri2024FixedSectorBranchFrontier):
            children: list[str] = []
            for choice in outcome.choices:
                child = Vituri2024FixedSectorBranchPath(path.choices + (choice,))
                if child.fingerprint in queued:
                    raise RuntimeError("duplicate fixed-sector ordered branch path")
                queued.add(child.fingerprint)
                queue.append(child)
                children.append(child.path_id)
            if len(nodes) + 1 + len(queue) > policy.maximum_replayed_paths:
                raise RuntimeError("fixed-sector frontier expansion exceeds replay cap")
            nodes.append(
                Vituri2024FixedSectorBFSNode(
                    path, "expanded_exact_frontier", tuple(children)
                )
            )
            continue
        if len(endpoints) + len(rejections) >= policy.maximum_endpoints:
            raise RuntimeError("fixed-sector endpoint cap reached before closure")
        if isinstance(outcome, Vituri2024FixedSectorScientificRejection):
            rejections.append(outcome)
            nodes.append(Vituri2024FixedSectorBFSNode(path, outcome.classification))
        else:
            endpoints.append(outcome)
            nodes.append(Vituri2024FixedSectorBFSNode(path, outcome.outcome))
    endpoint_tuple = tuple(endpoints)
    groups = _derive_stationary_groups(endpoint_tuple)
    stationary = tuple(item for item in endpoint_tuple if item.stationary)
    coalesced = bool(stationary) and len(groups) == 1
    result = Vituri2024FixedSectorSearchResult(
        policy=policy,
        initializer=initializer,
        prepared_fingerprint=prepared.fingerprint,
        nodes=tuple(nodes),
        endpoints=endpoint_tuple,
        rejections=tuple(rejections),
        stationary_groups=groups,
        replayed_path_count=len(nodes),
        endpoint_count=len(endpoints) + len(rejections),
        branch_tree_exhausted=True,
        unconsumed_frontier_count=0,
        all_normal_endpoints_stationary=all(item.stationary for item in endpoints),
        exact_stationary_endpoint_array_coalescence=coalesced,
        representative_endpoint=stationary[0] if coalesced else None,
        in_process_candidate_only=True,
        independent_finite_volume_fixed_sector_full_scf_discriminator=False,
        local_hessian_stability_established=False,
    )
    # Caps are checked again before a qualified candidate can leave this process.
    if (
        result.replayed_path_count > policy.maximum_replayed_paths
        or result.endpoint_count > policy.maximum_endpoints
    ):
        raise RuntimeError("fixed-sector retained-evidence cap exceeded")
    return result


__all__ = [
    "Vituri2024FixedSectorBFSOutcome",
    "Vituri2024FixedSectorInitializationRejection",
    "Vituri2024FixedSectorPolicy",
    "Vituri2024FixedSectorSearchResult",
    "run_vituri2024_fixed_sector_bfs",
]
