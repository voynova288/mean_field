"""Exact-shell closure for finite-q Vituri normal spiral comparators.

This module closes only the translation-preserving, valley-incoherent normal
coordinate sector at one prepared finite-q spiral problem.  The selected spin
has one global rank across both valleys; the opposite spin is exactly full.
Every unresolved exact Fock shell is expanded in canonical coordinate order
and every descendant is replayed from one common normal initializer.

The SCF map is always executed by :func:`run_hartree_fock_problem`.  This
adapter declares exact full steps as a separate fixed-point discriminator.
For the eight job-468711 normal rejections, an external faithful diagnostic
observed only unit ODA steps before each exact trigger; production use must bind
that evidence separately.  This generic API does not claim ODA-trajectory
parity for arbitrary prepared problems or after a branch is exposed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ...core.hf.engine import DensityUpdateResult
from ...core.hf.problem import HartreeFockKernel, HartreeFockProblem, run_hartree_fock_problem
from .vituri2024_hf import (
    vituri2024_conventional_k_diagonal_to_native_density,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_spiral import (
    Vituri2024PreparedHFSpiral,
    Vituri2024SpiralOccupationBoundaryError,
    diagnose_vituri2024_hf_spiral_stationarity,
    make_vituri2024_hf_spiral_problem,
    make_vituri2024_hf_spiral_state,
    make_vituri2024_spiral_initial_density,
)

Array = NDArray[np.generic]
BoundaryKind = Literal["unique", "exact", "positive_subtolerance"]
VITURI2024_SPIRAL_NORMAL_CLOSURE_API_VERSION = "1"
VITURI2024_SPIRAL_NORMAL_CLOSURE_AUTHORITY = (
    "candidate finite-domain translation-preserving valley-incoherent normal "
    "global-rank coordinate-sector exact-shell closure"
)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _positive_real(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _readonly(value: object, dtype: np.dtype | None = None) -> Array:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError("evidence array must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _array_sha256(value: object) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array), initial=0.0))


def _selected_flavors(prepared: Vituri2024PreparedHFSpiral) -> tuple[int, int]:
    selected = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == prepared.choice.selected_spin
    )
    if len(selected) != 2:
        raise RuntimeError("Vituri selected-spin flavor inventory drifted")
    return selected  # type: ignore[return-value]


def _spectator_flavors(prepared: Vituri2024PreparedHFSpiral) -> tuple[int, int]:
    spectators = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == -prepared.choice.selected_spin
    )
    if len(spectators) != 2:
        raise RuntimeError("Vituri spectator-spin flavor inventory drifted")
    return spectators  # type: ignore[return-value]


def _energy_from_engine_inputs(
    interaction_h_native: Array, h0_native: Array, density_native: Array
) -> float:
    one_body = np.einsum(
        "abk,abk->", h0_native, density_native, optimize=False
    )
    interaction = 0.5 * np.einsum(
        "abk,abk->", interaction_h_native, density_native, optimize=False
    )
    total = complex(one_body + interaction)
    scale = max(1.0, abs(total), abs(one_body), abs(interaction))
    if abs(total.imag) > 5.0e-11 * scale:
        raise ValueError("spiral normal closure energy is materially complex")
    return float(total.real)


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalClosurePolicy:
    boundary_floor_ev: float = 1.0e-12
    boundary_roundoff_multiplier: float = 64.0
    hamiltonian_hermiticity_tolerance_ev: float = 1.0e-10
    final_raw_norm_tolerance: float = 1.0e-8
    commutator_tolerance_ev: float = 1.0e-8
    idempotency_tolerance: float = 1.0e-8
    population_tolerance: float = 1.0e-8
    energy_parity_relative_tolerance: float = 1.0e-10
    max_iter: int = 300
    maximum_generation: int = 16
    maximum_choices_per_trigger: int = 4096
    maximum_replayed_paths: int = 512
    maximum_terminals: int = 128
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "boundary_floor_ev",
            "boundary_roundoff_multiplier",
            "hamiltonian_hermiticity_tolerance_ev",
            "final_raw_norm_tolerance",
            "commutator_tolerance_ev",
            "idempotency_tolerance",
            "population_tolerance",
            "energy_parity_relative_tolerance",
        ):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        for name in (
            "max_iter",
            "maximum_generation",
            "maximum_choices_per_trigger",
            "maximum_replayed_paths",
            "maximum_terminals",
        ):
            value = _strict_int(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        }
        payload["api_version"] = VITURI2024_SPIRAL_NORMAL_CLOSURE_API_VERSION
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(self) -> None:
        if replace(self).fingerprint != self.fingerprint:
            raise ValueError("spiral normal closure policy fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBoundary:
    kind: BoundaryKind
    selected_rank: int
    occupied_max_ev: float
    empty_min_ev: float
    gap_ev: float
    effective_tolerance_ev: float
    strictly_below_flat_indices: tuple[int, ...] = ()
    shell_flat_indices: tuple[int, ...] = ()
    shell_selected_rank: int = 0
    occupied_flat_indices: tuple[int, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        rank = _strict_int(self.selected_rank, "selected_rank")
        shell_rank = _strict_int(self.shell_selected_rank, "shell_selected_rank")
        if self.kind not in ("unique", "exact", "positive_subtolerance"):
            raise ValueError("invalid normal boundary kind")
        for value, label in (
            (self.occupied_max_ev, "occupied_max_ev"),
            (self.empty_min_ev, "empty_min_ev"),
            (self.gap_ev, "gap_ev"),
            (self.effective_tolerance_ev, "effective_tolerance_ev"),
        ):
            if not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if self.empty_min_ev - self.occupied_max_ev != self.gap_ev:
            raise ValueError("normal boundary endpoint/gap mismatch")
        if self.gap_ev < 0.0 or self.effective_tolerance_ev <= 0.0:
            raise ValueError("normal boundary gap/tolerance invalid")
        below = tuple(_strict_int(x, "below index") for x in self.strictly_below_flat_indices)
        shell = tuple(_strict_int(x, "shell index") for x in self.shell_flat_indices)
        occupied = tuple(_strict_int(x, "occupied index") for x in self.occupied_flat_indices)
        if any(tuple(sorted(set(items))) != items for items in (below, shell, occupied)):
            raise ValueError("normal boundary indices must be sorted and unique")
        if self.kind == "unique":
            valid = self.gap_ev > self.effective_tolerance_ev
            valid = valid and len(occupied) == rank and not below and not shell and shell_rank == 0
        elif self.kind == "exact":
            valid = self.gap_ev == 0.0 and 0 < shell_rank < len(shell)
            valid = valid and len(below) + shell_rank == rank and not occupied
        else:
            valid = 0.0 < self.gap_ev <= self.effective_tolerance_ev
            valid = valid and not below and not shell and not occupied and shell_rank == 0
        if not valid:
            raise ValueError("normal boundary classification is inconsistent")
        object.__setattr__(self, "selected_rank", rank)
        object.__setattr__(self, "shell_selected_rank", shell_rank)
        object.__setattr__(self, "strictly_below_flat_indices", below)
        object.__setattr__(self, "shell_flat_indices", shell)
        object.__setattr__(self, "occupied_flat_indices", occupied)
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalInitializer:
    density_native: Array
    density_sha256: str
    h0_sha256: str
    prepared_fingerprint: str
    policy_fingerprint: str
    boundary: Vituri2024SpiralNormalBoundary
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        density = _readonly(self.density_native, np.dtype(np.complex128))
        if density.ndim != 3 or density.shape[:2] != (4, 4):
            raise ValueError("normal initializer density shape mismatch")
        if self.density_sha256 != _array_sha256(density):
            raise ValueError("normal initializer density hash mismatch")
        object.__setattr__(self, "density_native", density)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "density_sha256": self.density_sha256,
                    "h0_sha256": self.h0_sha256,
                    "prepared_fingerprint": self.prepared_fingerprint,
                    "policy_fingerprint": self.policy_fingerprint,
                    "boundary_fingerprint": self.boundary.fingerprint,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBranchTrigger:
    generation: int
    exact_fock_sha256: str
    previous_density_sha256: str
    boundary: Vituri2024SpiralNormalBoundary
    shell_previous_populations: tuple[float, ...]
    canonical_choice_count: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        generation = _strict_int(self.generation, "generation")
        count = _strict_int(self.canonical_choice_count, "canonical_choice_count")
        if generation < 0 or self.boundary.kind != "exact":
            raise ValueError("branch trigger generation/boundary invalid")
        expected = math.comb(
            len(self.boundary.shell_flat_indices), self.boundary.shell_selected_rank
        )
        if count != expected or count < 2:
            raise ValueError("branch trigger choice count invalid")
        populations = tuple(float(x) for x in self.shell_previous_populations)
        if len(populations) != len(self.boundary.shell_flat_indices) or not all(
            math.isfinite(x) for x in populations
        ):
            raise ValueError("branch trigger overlap inventory invalid")
        # Populations are retained as diagnostics, not as a selector.  The
        # declared closure exhausts every coordinate projector in an exact Fock
        # shell even when maximum-overlap continuation would prefer one subset.
        # This is intentionally the same no-postselection branch universe used
        # by the established Vituri fixed-sector closure.
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "canonical_choice_count", count)
        object.__setattr__(self, "shell_previous_populations", populations)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "generation": generation,
                    "exact_fock_sha256": self.exact_fock_sha256,
                    "previous_density_sha256": self.previous_density_sha256,
                    "boundary_fingerprint": self.boundary.fingerprint,
                    "shell_previous_populations": populations,
                    "canonical_choice_count": count,
                    "canonical_order": "selected_valley_slot_times_Nk_plus_k_then_itertools_combinations",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBranchChoice:
    trigger: Vituri2024SpiralNormalBranchTrigger
    canonical_choice_index: int
    selected_shell_flat_indices: tuple[int, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        index = _strict_int(self.canonical_choice_index, "canonical_choice_index")
        selected = tuple(_strict_int(x, "selected shell index") for x in self.selected_shell_flat_indices)
        inventory = tuple(
            itertools.combinations(
                self.trigger.boundary.shell_flat_indices,
                self.trigger.boundary.shell_selected_rank,
            )
        )
        if index < 0 or index >= len(inventory) or selected != inventory[index]:
            raise ValueError("normal branch choice is outside canonical inventory")
        object.__setattr__(self, "canonical_choice_index", index)
        object.__setattr__(self, "selected_shell_flat_indices", selected)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "trigger_fingerprint": self.trigger.fingerprint,
                    "canonical_choice_index": index,
                    "selected_shell_flat_indices": selected,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBranchPath:
    choices: tuple[Vituri2024SpiralNormalBranchChoice, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.choices) is not tuple:
            raise TypeError("normal branch path choices must be a tuple")
        for generation, choice in enumerate(self.choices):
            if type(choice) is not Vituri2024SpiralNormalBranchChoice:
                raise TypeError("normal branch path entries must be typed")
            if choice.trigger.generation != generation:
                raise ValueError("normal branch path generations are not ordered")
        object.__setattr__(self, "fingerprint", _fingerprint([x.fingerprint for x in self.choices]))

    @property
    def path_id(self) -> str:
        if not self.choices:
            return "root"
        indices = "_".join(str(x.canonical_choice_index) for x in self.choices)
        return f"g{indices}_{self.fingerprint[:12]}"


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBranchFrontier:
    path: Vituri2024SpiralNormalBranchPath
    trigger: Vituri2024SpiralNormalBranchTrigger
    choices: tuple[Vituri2024SpiralNormalBranchChoice, ...]


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalScientificRejection:
    path: Vituri2024SpiralNormalBranchPath
    classification: str
    stage: str
    message: str
    consumed_choice_fingerprints: tuple[str, ...]
    applied_lambdas: tuple[float, ...] = ()
    evidence_arrays: tuple[tuple[str, Array], ...] = ()

    def __post_init__(self) -> None:
        lambdas = tuple(float(value) for value in self.applied_lambdas)
        if not all(math.isfinite(value) for value in lambdas):
            raise ValueError("normal rejection lambda receipts must be finite")
        arrays = tuple((name, _readonly(value)) for name, value in self.evidence_arrays)
        object.__setattr__(self, "applied_lambdas", lambdas)
        object.__setattr__(self, "evidence_arrays", arrays)


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalEndpointMetrics:
    selected_rank_residual: float
    spectator_full_residual: float
    normal_density_offdiagonal_residual: float
    binary_coordinate_occupation_residual: float
    idempotency_residual: float
    hermiticity_residual: float
    commutator_residual_ev: float
    final_raw_norm: float
    engine_final_raw_parity_residual: float
    energy_parity_residual_ev: float
    fresh_boundary_gap_ev: float
    all_full_step_lambdas_exact: bool


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalEndpoint:
    path: Vituri2024SpiralNormalBranchPath
    outcome: Literal["stationary", "normal_endpoint_gate_rejection"]
    stationary: bool
    converged: bool
    exit_reason: str
    consumed_choice_fingerprints: tuple[str, ...]
    iter_energy: Array
    iter_err: Array
    iter_oda: Array
    final_density: Array
    fresh_raw_density: Array
    fresh_hamiltonian: Array
    energy_ev: float
    metrics: Vituri2024SpiralNormalEndpointMetrics
    final_density_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "iter_energy",
            "iter_err",
            "iter_oda",
            "final_density",
            "fresh_raw_density",
            "fresh_hamiltonian",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.final_density_sha256 != _array_sha256(self.final_density):
            raise ValueError("normal endpoint final-density hash mismatch")
        if self.stationary != (self.outcome == "stationary"):
            raise ValueError("normal endpoint stationarity/outcome mismatch")


NormalPathOutcome: TypeAlias = (
    Vituri2024SpiralNormalBranchFrontier
    | Vituri2024SpiralNormalScientificRejection
    | Vituri2024SpiralNormalEndpoint
)


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalBFSNode:
    path: Vituri2024SpiralNormalBranchPath
    outcome: str
    child_path_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalStationaryGroup:
    final_density_sha256: str
    path_ids: tuple[str, ...]
    energy_min_ev: float
    energy_max_ev: float


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalClosureResult:
    policy: Vituri2024SpiralNormalClosurePolicy
    initializer: Vituri2024SpiralNormalInitializer
    prepared_fingerprint: str
    nodes: tuple[Vituri2024SpiralNormalBFSNode, ...]
    endpoints: tuple[Vituri2024SpiralNormalEndpoint, ...]
    rejections: tuple[Vituri2024SpiralNormalScientificRejection, ...]
    stationary_groups: tuple[Vituri2024SpiralNormalStationaryGroup, ...]
    branch_tree_exhausted: bool
    deterministic_terminal_replay_verified: bool
    all_normal_endpoints_stationary: bool
    all_applied_steps_full_step: bool
    candidate_finite_domain_only: bool = True
    same_q_energy_comparison_authorized: bool = False
    uv_authority: bool = False
    unrestricted_ground_state_authority: bool = False
    local_hessian_authority: bool = False
    fig2_reproduction_authority: bool = False
    tdhf_authority: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not self.branch_tree_exhausted or not self.deterministic_terminal_replay_verified:
            raise ValueError("normal closure result requires exhaustive deterministic replay")
        if not self.candidate_finite_domain_only:
            raise ValueError("normal closure candidate-only authority was removed")
        if any(
            (
                self.same_q_energy_comparison_authorized,
                self.uv_authority,
                self.unrestricted_ground_state_authority,
                self.local_hessian_authority,
                self.fig2_reproduction_authority,
                self.tdhf_authority,
                self.production_authority,
            )
        ):
            raise ValueError("normal closure authority was inflated")


class _ScientificTerminal(RuntimeError):
    def __init__(self, classification: str, stage: str, message: str, evidence: dict[str, Array] | None = None):
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.evidence = {} if evidence is None else evidence


class _FrontierSignal(RuntimeError):
    def __init__(self, frontier: Vituri2024SpiralNormalBranchFrontier):
        super().__init__("normal exact-shell branch frontier")
        self.frontier = frontier


def analyze_vituri2024_spiral_normal_boundary(
    prepared: Vituri2024PreparedHFSpiral,
    hamiltonian: Array,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> Vituri2024SpiralNormalBoundary:
    """Classify the global selected-spin coordinate boundary without sorting."""

    prepared.validate_live_state()
    policy.validate_live_state()
    matrix = np.asarray(hamiltonian)
    if matrix.dtype != np.dtype(np.complex128) or matrix.shape != (4, 4, prepared.nk):
        raise ValueError("normal closure Hamiltonian must be complex128 (4,4,Nk)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("normal closure Hamiltonian must be finite")
    hermiticity = _max_abs(matrix - matrix.swapaxes(0, 1).conj())
    if hermiticity > policy.hamiltonian_hermiticity_tolerance_ev:
        raise _ScientificTerminal(
            "hamiltonian_hermiticity_rejection",
            "boundary_analysis",
            "normal closure Hamiltonian is materially non-Hermitian",
            {"hamiltonian": matrix.copy()},
        )
    selected = _selected_flavors(prepared)
    spectators = _spectator_flavors(prepared)
    offdiag = _max_abs(matrix[selected[0], selected[1], :])
    cross = _max_abs(
        matrix[np.ix_(selected, spectators, np.arange(prepared.nk, dtype=np.int64))]
    )
    if offdiag != 0.0:
        raise _ScientificTerminal(
            "normal_coordinate_fock_violation",
            "boundary_analysis",
            "selected-valley Fock block is not exactly diagonal in the normal coordinate basis",
            {"hamiltonian": matrix.copy()},
        )
    if cross > prepared.choice.spin_block_tolerance_ev:
        raise _ScientificTerminal(
            "spin_block_fock_rejection",
            "boundary_analysis",
            "selected/spectator Fock coupling exceeds the spiral contract",
            {"hamiltonian": matrix.copy()},
        )
    flat = np.stack([matrix[f, f, :].real for f in selected]).reshape(-1, order="C")
    rank = prepared.selected_rank
    partitioned = np.partition(flat, (rank - 1, rank))
    lower = float(partitioned[rank - 1])
    upper = float(partitioned[rank])
    gap = upper - lower
    tolerance = max(
        policy.boundary_floor_ev,
        policy.boundary_roundoff_multiplier
        * np.finfo(np.float64).eps
        * max(1.0, _max_abs(matrix)),
    )
    if gap > tolerance:
        occupied = tuple(int(x) for x in np.flatnonzero(flat <= lower))
        return Vituri2024SpiralNormalBoundary(
            "unique", rank, lower, upper, gap, tolerance, occupied_flat_indices=occupied
        )
    if 0.0 < gap <= tolerance:
        return Vituri2024SpiralNormalBoundary(
            "positive_subtolerance", rank, lower, upper, gap, tolerance
        )
    if gap != 0.0:
        raise RuntimeError("normal boundary escaped exhaustive classification")
    below = tuple(int(x) for x in np.flatnonzero(flat < lower))
    shell = tuple(int(x) for x in np.flatnonzero(flat == lower))
    shell_rank = rank - len(below)
    return Vituri2024SpiralNormalBoundary(
        "exact",
        rank,
        lower,
        upper,
        gap,
        tolerance,
        strictly_below_flat_indices=below,
        shell_flat_indices=shell,
        shell_selected_rank=shell_rank,
    )


def enumerate_vituri2024_spiral_normal_branch_choices(
    prepared: Vituri2024PreparedHFSpiral,
    hamiltonian: Array,
    previous_density_native: Array,
    boundary: Vituri2024SpiralNormalBoundary,
    *,
    generation: int,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> tuple[Vituri2024SpiralNormalBranchChoice, ...]:
    """Enumerate every canonical coordinate projector in one exact shell."""

    if boundary.kind != "exact":
        raise ValueError("normal branch enumeration requires an exact boundary")
    generation = _strict_int(generation, "generation")
    if generation >= policy.maximum_generation:
        raise RuntimeError("maximum normal branch generation reached")
    previous = np.asarray(previous_density_native)
    if previous.dtype != np.dtype(np.complex128) or previous.shape != (4, 4, prepared.nk):
        raise ValueError("previous normal density shape/dtype mismatch")
    conventional = vituri2024_native_density_to_conventional_k_diagonal(previous)
    selected = _selected_flavors(prepared)
    populations: list[float] = []
    for flat_index in boundary.shell_flat_indices:
        valley_slot, momentum = divmod(flat_index, prepared.nk)
        populations.append(float(conventional[selected[valley_slot], selected[valley_slot], momentum].real))
    count = math.comb(len(boundary.shell_flat_indices), boundary.shell_selected_rank)
    if count > policy.maximum_choices_per_trigger:
        raise RuntimeError("normal exact-shell choice cap exceeded")
    trigger = Vituri2024SpiralNormalBranchTrigger(
        generation=generation,
        exact_fock_sha256=_array_sha256(hamiltonian),
        previous_density_sha256=_array_sha256(previous),
        boundary=boundary,
        shell_previous_populations=tuple(populations),
        canonical_choice_count=count,
    )
    inventory = tuple(
        itertools.combinations(boundary.shell_flat_indices, boundary.shell_selected_rank)
    )
    return tuple(
        Vituri2024SpiralNormalBranchChoice(trigger, index, selected_shell)
        for index, selected_shell in enumerate(inventory)
    )


def _density_from_boundary(
    prepared: Vituri2024PreparedHFSpiral,
    boundary: Vituri2024SpiralNormalBoundary,
    choice: Vituri2024SpiralNormalBranchChoice | None,
) -> Array:
    conventional = np.zeros((4, 4, prepared.nk), dtype=np.complex128)
    for flavor in _spectator_flavors(prepared):
        conventional[flavor, flavor, :] = 1.0
    if boundary.kind == "unique":
        occupied = boundary.occupied_flat_indices
    elif boundary.kind == "exact":
        if choice is None:
            raise RuntimeError("exact normal boundary lacks a branch choice")
        occupied = boundary.strictly_below_flat_indices + choice.selected_shell_flat_indices
    else:
        raise RuntimeError("positive-subtolerance boundary reached projector builder")
    if len(occupied) != prepared.selected_rank or len(set(occupied)) != len(occupied):
        raise RuntimeError("normal coordinate projector changed the selected global rank")
    selected = _selected_flavors(prepared)
    for flat_index in occupied:
        valley_slot, momentum = divmod(flat_index, prepared.nk)
        conventional[selected[valley_slot], selected[valley_slot], momentum] = 1.0
    return np.asarray(
        vituri2024_conventional_k_diagonal_to_native_density(conventional),
        dtype=np.complex128,
    )


def build_vituri2024_spiral_normal_initializer(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> Vituri2024SpiralNormalInitializer:
    """Build one open-gap common normal initializer without sorting."""

    if _max_abs(prepared.functional.normal_order_reference_conventional) != 0.0:
        raise ValueError(
            "spiral normal closure currently requires the exact zero normal-order reference"
        )
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, np.asarray(prepared.h0_native), policy=policy
    )
    if boundary.kind != "unique":
        raise _ScientificTerminal(
            "h0_normal_boundary_rejection",
            "common_initializer",
            "common normal h0 boundary is not uniquely open",
            {"h0": np.asarray(prepared.h0_native).copy()},
        )
    independent = _density_from_boundary(prepared, boundary, None)
    existing = make_vituri2024_spiral_initial_density(prepared, init_mode="normal")
    if not np.array_equal(independent, existing):
        raise RuntimeError("independent threshold initializer disagrees with spiral normal initializer")
    return Vituri2024SpiralNormalInitializer(
        density_native=independent,
        density_sha256=_array_sha256(independent),
        h0_sha256=_array_sha256(prepared.h0_native),
        prepared_fingerprint=prepared.fingerprint,
        policy_fingerprint=policy.fingerprint,
        boundary=boundary,
    )


def _run_path(
    prepared: Vituri2024PreparedHFSpiral,
    initializer: Vituri2024SpiralNormalInitializer,
    path: Vituri2024SpiralNormalBranchPath,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> NormalPathOutcome:
    rebuilt = build_vituri2024_spiral_normal_initializer(prepared, policy=policy)
    if (
        initializer.fingerprint != rebuilt.fingerprint
        or not np.array_equal(initializer.density_native, rebuilt.density_native)
        or initializer.prepared_fingerprint != prepared.fingerprint
        or initializer.policy_fingerprint != policy.fingerprint
    ):
        raise ValueError("normal common initializer independent rebuild mismatch")
    state = make_vituri2024_hf_spiral_state(prepared)
    base_problem = make_vituri2024_hf_spiral_problem(
        prepared,
        initial_density_native=initializer.density_native,
        mixing_mode="full_step",
    )
    consumed: list[Vituri2024SpiralNormalBranchChoice] = []
    pending: dict[str, object] | None = None
    final_raw: list[Array] = []
    lambda_receipts: list[float] = []

    def would_be_final_map() -> bool:
        iteration = int(state.diagnostics.get("normal_closure_last_iteration", 0))
        raw = float(state.diagnostics.get("normal_closure_last_raw_norm", math.inf))
        return iteration > 0 and (raw <= state.precision or iteration >= policy.max_iter)

    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        nonlocal pending
        if pending is not None:
            raise RuntimeError("normal closure density-builder receipt was not consumed")
        matrix = np.asarray(hamiltonian, dtype=np.complex128)
        previous = np.asarray(state.density, dtype=np.complex128).copy()
        try:
            boundary = analyze_vituri2024_spiral_normal_boundary(
                prepared, matrix, policy=policy
            )
        except _ScientificTerminal:
            raise
        if boundary.kind == "positive_subtolerance":
            raise _ScientificTerminal(
                "positive_subtolerance_splitting_rejection",
                "density_update",
                "positive selected-spin splitting lies below the effective tolerance",
                {"density": previous, "hamiltonian": matrix.copy()},
            )
        choice: Vituri2024SpiralNormalBranchChoice | None = None
        if boundary.kind == "exact":
            choices = enumerate_vituri2024_spiral_normal_branch_choices(
                prepared,
                matrix,
                previous,
                boundary,
                generation=len(consumed),
                policy=policy,
            )
            if len(consumed) >= len(path.choices):
                if would_be_final_map():
                    raise _ScientificTerminal(
                        "branch_frontier_in_final_map_rejection",
                        "final_density_recomputation",
                        "an unresolved normal branch frontier appeared only in the final map",
                        {"density": previous, "hamiltonian": matrix.copy()},
                    )
                raise _FrontierSignal(
                    Vituri2024SpiralNormalBranchFrontier(path, choices[0].trigger, choices)
                )
            choice = path.choices[len(consumed)]
            index = choice.canonical_choice_index
            if (
                index >= len(choices)
                or choice.trigger.fingerprint != choices[0].trigger.fingerprint
                or choice != choices[index]
            ):
                raise ValueError("normal ordered path trigger/choice mismatch")
            if would_be_final_map():
                raise _ScientificTerminal(
                    "branch_choice_in_final_map_rejection",
                    "final_density_recomputation",
                    "a registered normal branch choice would be consumed only by the final map",
                    {"density": previous, "hamiltonian": matrix.copy()},
                )
        raw = _density_from_boundary(prepared, boundary, choice)
        energies = np.asarray(
            np.real(np.diagonal(matrix, axis1=0, axis2=1)).T,
            dtype=np.float64,
        )
        update = DensityUpdateResult(
            density=raw,
            energies=energies,
            mu=0.5 * (boundary.occupied_max_ev + boundary.empty_min_ev),
            observables={
                "selected_boundary_gap_ev": boundary.gap_ev,
                "normal_exact_shell_dimension": float(len(boundary.shell_flat_indices)),
                "normal_exact_shell_rank": float(boundary.shell_selected_rank),
            },
        )
        pending = {
            "update_id": id(update),
            "raw_sha256": _array_sha256(raw),
            "fock_sha256": _array_sha256(matrix),
            "previous_sha256": _array_sha256(previous),
            "choice": choice,
        }
        return update

    def step_callback(target: object, step: object) -> None:
        nonlocal pending
        if target is not state or pending is None:
            raise RuntimeError("normal closure step callback lost its builder receipt")
        if pending["update_id"] != id(step.density_update):
            raise RuntimeError("normal closure update identity binding failed")
        if pending["raw_sha256"] != _array_sha256(step.density_update.density):
            raise RuntimeError("normal closure raw-density binding failed")
        if pending["fock_sha256"] != _array_sha256(step.total_hamiltonian):
            raise RuntimeError("normal closure exact-Fock binding failed")
        if pending["previous_sha256"] != _array_sha256(step.previous_density):
            raise RuntimeError("normal closure previous-density binding failed")
        if step.oda_lambda != 1.0:
            raise RuntimeError("normal closure full-step policy returned a non-unit lambda")
        if not np.array_equal(step.mixed_density, step.density_update.density):
            raise RuntimeError("normal closure full step did not install the raw projector exactly")
        choice = pending["choice"]
        if choice is not None:
            if np.array_equal(step.previous_density, step.mixed_density):
                lambda_receipts.append(float(step.oda_lambda))
                raise _ScientificTerminal(
                    "branch_choice_not_applied_rejection",
                    "step_callback",
                    "an exact normal branch choice did not change the applied density",
                    {"raw_density": np.asarray(step.density_update.density).copy()},
                )
            consumed.append(choice)
        lambda_receipts.append(float(step.oda_lambda))
        state.diagnostics["normal_closure_last_iteration"] = float(step.iteration)
        state.diagnostics["normal_closure_last_raw_norm"] = float(step.norm_raw)
        pending = None

    def final_state_callback(target: object, update: DensityUpdateResult) -> None:
        nonlocal pending
        if target is not state or pending is None or pending["update_id"] != id(update):
            raise RuntimeError("normal closure final-map receipt binding failed")
        if pending["choice"] is not None:
            raise RuntimeError("normal closure final map attempted to consume a branch choice")
        final_raw.append(np.asarray(update.density, dtype=np.complex128).copy())
        pending = None

    kernel = replace(
        base_problem.kernel,
        density_builder=density_builder,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
    )
    problem = HartreeFockProblem(initializer=base_problem.initializer, kernel=kernel)
    try:
        run = run_hartree_fock_problem(
            state,
            problem,
            init_mode="provided_density",
            seed=0,
            max_iter=policy.max_iter,
            oda_stall_threshold=1.0e-12,
            max_oda_lambda=1.0,
        )
    except _FrontierSignal as signal:
        return signal.frontier
    except _ScientificTerminal as error:
        return Vituri2024SpiralNormalScientificRejection(
            path=path,
            classification=error.classification,
            stage=error.stage,
            message=str(error),
            consumed_choice_fingerprints=tuple(x.fingerprint for x in consumed),
            applied_lambdas=tuple(lambda_receipts),
            evidence_arrays=tuple(sorted(error.evidence.items())),
        )
    if pending is not None or len(final_raw) != 1:
        raise RuntimeError("normal closure final-map accounting failed")
    if len(consumed) != len(path.choices):
        return Vituri2024SpiralNormalScientificRejection(
            path=path,
            classification="unconsumed_branch_path_rejection",
            stage="post_run",
            message="the registered normal branch path was not fully consumed",
            consumed_choice_fingerprints=tuple(x.fingerprint for x in consumed),
            applied_lambdas=tuple(lambda_receipts),
        )
    try:
        stationarity, fresh_map = diagnose_vituri2024_hf_spiral_stationarity(prepared, run)
    except Vituri2024SpiralOccupationBoundaryError as error:
        return Vituri2024SpiralNormalScientificRejection(
            path=path,
            classification="fresh_stationarity_rejection",
            stage="fresh_final_map",
            message=f"fresh normal stationarity recomputation failed: {type(error).__name__}: {error}",
            consumed_choice_fingerprints=tuple(x.fingerprint for x in consumed),
            applied_lambdas=tuple(lambda_receipts),
            evidence_arrays=(("final_density", np.asarray(run.state.density).copy()),),
        )
    density = np.asarray(run.state.density, dtype=np.complex128)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(density)
    selected = _selected_flavors(prepared)
    spectators = _spectator_flavors(prepared)
    selected_block = conventional[
        np.ix_(selected, selected, np.arange(prepared.nk, dtype=np.int64))
    ]
    spectator_block = conventional[
        np.ix_(spectators, spectators, np.arange(prepared.nk, dtype=np.int64))
    ]
    diag = np.concatenate(
        [selected_block[index, index, :].real for index in range(2)]
    )
    normal_offdiag = _max_abs(selected_block[0, 1, :])
    binary_residual = _max_abs(diag - np.rint(diag))
    rank_residual = abs(float(np.sum(diag)) - prepared.selected_rank)
    spectator_residual = _max_abs(
        spectator_block - np.eye(2, dtype=np.complex128)[:, :, None]
    )
    idempotency = 0.0
    commutator = 0.0
    fresh_h = np.asarray(prepared.functional.fock(density), dtype=np.complex128)
    for momentum in range(prepared.nk):
        projector = conventional[:, :, momentum]
        fock = fresh_h[:, :, momentum]
        idempotency = max(idempotency, _max_abs(projector @ projector - projector))
        commutator = max(commutator, _max_abs(fock @ projector - projector @ fock))
    hermiticity = _max_abs(conventional - conventional.swapaxes(0, 1).conj())
    fresh_raw_density = np.asarray(fresh_map.density_native)
    final_raw_norm = _max_abs(fresh_raw_density - density)
    engine_final_raw_parity = _max_abs(final_raw[0] - fresh_raw_density)
    engine_energy = float(run.state.diagnostics["hf_energy"])
    functional_energy = float(prepared.functional.energy(density))
    interaction_native = np.asarray(
        base_problem.kernel.interaction_builder(density), dtype=np.complex128
    )
    reconstructed_energy = _energy_from_engine_inputs(
        interaction_native, np.asarray(state.h0), density
    )
    energy_residual = max(
        abs(engine_energy - functional_energy),
        abs(engine_energy - reconstructed_energy),
        abs(functional_energy - reconstructed_energy),
    )
    energy_scale = max(
        1.0, abs(engine_energy), abs(functional_energy), abs(reconstructed_energy)
    )
    all_lambdas = bool(lambda_receipts) and all(x == 1.0 for x in lambda_receipts)
    metrics = Vituri2024SpiralNormalEndpointMetrics(
        selected_rank_residual=rank_residual,
        spectator_full_residual=spectator_residual,
        normal_density_offdiagonal_residual=normal_offdiag,
        binary_coordinate_occupation_residual=binary_residual,
        idempotency_residual=idempotency,
        hermiticity_residual=hermiticity,
        commutator_residual_ev=commutator,
        final_raw_norm=final_raw_norm,
        engine_final_raw_parity_residual=engine_final_raw_parity,
        energy_parity_residual_ev=energy_residual,
        fresh_boundary_gap_ev=fresh_map.diagnostics.boundary.gap_ev,
        all_full_step_lambdas_exact=all_lambdas,
    )
    stationary = bool(
        run.converged
        and stationarity.stationary
        and rank_residual <= policy.population_tolerance
        and spectator_residual <= policy.population_tolerance
        and normal_offdiag == 0.0
        and binary_residual == 0.0
        and idempotency <= policy.idempotency_tolerance
        and hermiticity <= policy.idempotency_tolerance
        and commutator <= policy.commutator_tolerance_ev
        and final_raw_norm <= policy.final_raw_norm_tolerance
        and engine_final_raw_parity == 0.0
        and energy_residual <= policy.energy_parity_relative_tolerance * energy_scale
        and fresh_map.diagnostics.boundary.gap_ev > fresh_map.diagnostics.boundary.effective_floor_ev
        and all_lambdas
    )
    return Vituri2024SpiralNormalEndpoint(
        path=path,
        outcome="stationary" if stationary else "normal_endpoint_gate_rejection",
        stationary=stationary,
        converged=run.converged,
        exit_reason=run.exit_reason,
        consumed_choice_fingerprints=tuple(x.fingerprint for x in consumed),
        iter_energy=np.asarray(run.iter_energy),
        iter_err=np.asarray(run.iter_err),
        iter_oda=np.asarray(run.iter_oda),
        final_density=density,
        fresh_raw_density=np.asarray(fresh_map.density_native),
        fresh_hamiltonian=fresh_h,
        energy_ev=engine_energy,
        metrics=metrics,
        final_density_sha256=_array_sha256(density),
    )


def _outcome_digest(outcome: NormalPathOutcome) -> str:
    if isinstance(outcome, Vituri2024SpiralNormalEndpoint):
        payload = {
            "kind": "endpoint",
            "path": outcome.path.fingerprint,
            "outcome": outcome.outcome,
            "stationary": outcome.stationary,
            "converged": outcome.converged,
            "exit_reason": outcome.exit_reason,
            "consumed": outcome.consumed_choice_fingerprints,
            "iter_energy": _array_sha256(outcome.iter_energy),
            "iter_err": _array_sha256(outcome.iter_err),
            "iter_oda": _array_sha256(outcome.iter_oda),
            "final_density": outcome.final_density_sha256,
            "fresh_raw": _array_sha256(outcome.fresh_raw_density),
            "fresh_h": _array_sha256(outcome.fresh_hamiltonian),
            "energy_ev": outcome.energy_ev,
            "metrics": tuple(
                (name, getattr(outcome.metrics, name))
                for name in outcome.metrics.__dataclass_fields__
            ),
        }
    elif isinstance(outcome, Vituri2024SpiralNormalScientificRejection):
        payload = {
            "kind": "rejection",
            "path": outcome.path.fingerprint,
            "classification": outcome.classification,
            "stage": outcome.stage,
            "message": outcome.message,
            "consumed": outcome.consumed_choice_fingerprints,
            "applied_lambdas": outcome.applied_lambdas,
            "evidence": [(name, _array_sha256(value)) for name, value in outcome.evidence_arrays],
        }
    else:
        payload = {
            "kind": "frontier",
            "path": outcome.path.fingerprint,
            "trigger": outcome.trigger.fingerprint,
            "choices": [x.fingerprint for x in outcome.choices],
        }
    return _fingerprint(payload)


def _stationary_groups(
    endpoints: tuple[Vituri2024SpiralNormalEndpoint, ...],
) -> tuple[Vituri2024SpiralNormalStationaryGroup, ...]:
    grouped: dict[str, list[Vituri2024SpiralNormalEndpoint]] = {}
    for endpoint in endpoints:
        if endpoint.stationary:
            grouped.setdefault(endpoint.final_density_sha256, []).append(endpoint)
    return tuple(
        Vituri2024SpiralNormalStationaryGroup(
            final_density_sha256=digest,
            path_ids=tuple(item.path.path_id for item in members),
            energy_min_ev=min(item.energy_ev for item in members),
            energy_max_ev=max(item.energy_ev for item in members),
        )
        for digest, members in sorted(grouped.items())
    )


def run_vituri2024_spiral_normal_exact_shell_closure(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy | None = None,
) -> Vituri2024SpiralNormalClosureResult:
    """Exhaust and byte-replay the declared normal coordinate branch tree."""

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be Vituri2024PreparedHFSpiral")
    active_policy = Vituri2024SpiralNormalClosurePolicy() if policy is None else policy
    active_policy.validate_live_state()
    initializer = build_vituri2024_spiral_normal_initializer(
        prepared, policy=active_policy
    )
    root = Vituri2024SpiralNormalBranchPath()
    queue: deque[Vituri2024SpiralNormalBranchPath] = deque((root,))
    queued = {root.fingerprint}
    nodes: list[Vituri2024SpiralNormalBFSNode] = []
    endpoints: list[Vituri2024SpiralNormalEndpoint] = []
    rejections: list[Vituri2024SpiralNormalScientificRejection] = []
    terminal_outcomes: list[NormalPathOutcome] = []
    while queue:
        if len(nodes) >= active_policy.maximum_replayed_paths:
            raise RuntimeError("normal closure replay cap reached with unresolved frontier")
        path = queue.popleft()
        outcome = _run_path(prepared, initializer, path, policy=active_policy)
        if isinstance(outcome, Vituri2024SpiralNormalBranchFrontier):
            children: list[str] = []
            for choice in outcome.choices:
                child = Vituri2024SpiralNormalBranchPath(path.choices + (choice,))
                if child.fingerprint in queued:
                    raise RuntimeError("duplicate normal ordered branch path")
                queued.add(child.fingerprint)
                queue.append(child)
                children.append(child.path_id)
            if len(nodes) + 1 + len(queue) > active_policy.maximum_replayed_paths:
                raise RuntimeError("normal frontier expansion exceeds replay cap")
            nodes.append(Vituri2024SpiralNormalBFSNode(path, "expanded_exact_frontier", tuple(children)))
            continue
        if len(terminal_outcomes) >= active_policy.maximum_terminals:
            raise RuntimeError("normal closure terminal cap reached before closure")
        terminal_outcomes.append(outcome)
        if isinstance(outcome, Vituri2024SpiralNormalEndpoint):
            endpoints.append(outcome)
            nodes.append(Vituri2024SpiralNormalBFSNode(path, outcome.outcome))
        else:
            rejections.append(outcome)
            nodes.append(Vituri2024SpiralNormalBFSNode(path, outcome.classification))
    # Every terminal path is recomputed from the independently rebuilt common
    # initializer.  No parent endpoint or intermediate density is reused.
    for original in terminal_outcomes:
        replay = _run_path(prepared, initializer, original.path, policy=active_policy)
        if isinstance(replay, Vituri2024SpiralNormalBranchFrontier):
            raise RuntimeError("closed normal terminal replay reopened a frontier")
        if _outcome_digest(replay) != _outcome_digest(original):
            raise RuntimeError("normal terminal deterministic replay mismatch")
    endpoint_tuple = tuple(endpoints)
    all_terminal_lambdas = tuple(
        value
        for outcome in terminal_outcomes
        for value in (
            tuple(float(x) for x in outcome.iter_oda)
            if isinstance(outcome, Vituri2024SpiralNormalEndpoint)
            else outcome.applied_lambdas
        )
    )
    return Vituri2024SpiralNormalClosureResult(
        policy=active_policy,
        initializer=initializer,
        prepared_fingerprint=prepared.fingerprint,
        nodes=tuple(nodes),
        endpoints=endpoint_tuple,
        rejections=tuple(rejections),
        stationary_groups=_stationary_groups(endpoint_tuple),
        branch_tree_exhausted=True,
        deterministic_terminal_replay_verified=True,
        all_normal_endpoints_stationary=bool(endpoint_tuple) and all(x.stationary for x in endpoint_tuple),
        all_applied_steps_full_step=bool(all_terminal_lambdas)
        and all(value == 1.0 for value in all_terminal_lambdas),
    )


__all__ = [
    "VITURI2024_SPIRAL_NORMAL_CLOSURE_API_VERSION",
    "VITURI2024_SPIRAL_NORMAL_CLOSURE_AUTHORITY",
    "Vituri2024SpiralNormalBoundary",
    "Vituri2024SpiralNormalBranchChoice",
    "Vituri2024SpiralNormalBranchPath",
    "Vituri2024SpiralNormalBranchTrigger",
    "Vituri2024SpiralNormalClosurePolicy",
    "Vituri2024SpiralNormalClosureResult",
    "analyze_vituri2024_spiral_normal_boundary",
    "build_vituri2024_spiral_normal_initializer",
    "enumerate_vituri2024_spiral_normal_branch_choices",
    "run_vituri2024_spiral_normal_exact_shell_closure",
]
