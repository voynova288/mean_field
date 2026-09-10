"""Exact-shell closure for finite-q Vituri normal spiral comparators.

This module closes only the translation-preserving, valley-incoherent normal
coordinate sector at one prepared finite-q spiral problem.  The selected spin
has one global rank across both valleys; the opposite spin is exactly full.
Every unresolved exact Fock shell is expanded in canonical coordinate order
and every descendant is replayed from one common normal initializer. An
optional same-physics dense float64 boundary oracle may replace only an FFT
positive-subtolerance boundary by an independently exact dense shell, after
same-density FFT recomputation, full-Fock, boundary-gap, scalar-energy, gauge,
mesh, reference, and interaction lineage gates pass. The old FFT rejection is
never relabeled and no tolerance is loosened.

The SCF map is always executed by :func:`run_hartree_fock_problem`.  This
adapter declares exact full steps as a separate fixed-point discriminator.
For the eight job-468711 normal rejections, an external faithful diagnostic
observed only unit ODA steps before each exact trigger; production use must bind
that evidence separately.  This generic API does not claim ODA-trajectory
parity for arbitrary prepared problems or after a branch is exposed.  Its
branch-tree exhaustion is conditional on the declared root; it neither
exhausts a global source inventory nor excludes prior source-group
postselection.
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
from .vituri2024_hf_spiral_density_embedding import (
    validate_vituri2024_spiral_density_embedding_receipt,
)
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
VITURI2024_SPIRAL_NORMAL_DENSE_BOUNDARY_ORACLE_API_VERSION = "1"
VITURI2024_SPIRAL_NORMAL_CLOSURE_AUTHORITY = (
    "candidate finite-domain translation-preserving valley-incoherent normal "
    "global-rank coordinate-sector exact-shell closure"
)
VITURI2024_SPIRAL_NORMAL_SOURCE_LINEAGE_API_VERSION = "2"


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
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape)
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
class Vituri2024SpiralNormalDenseBoundaryOracleReceipt:
    pair_fingerprint: str
    density_sha256: str
    applied_fft_fock_sha256: str
    dense_oracle_fock_sha256: str
    applied_boundary: Vituri2024SpiralNormalBoundary
    dense_boundary: Vituri2024SpiralNormalBoundary
    applied_fft_recompute_exact: bool
    dense_fft_fock_max_abs_ev: float
    dense_fft_boundary_gap_abs_difference_ev: float
    dense_fft_energy_abs_ev: float
    fock_parity_gate_ev: float
    energy_parity_gate_ev: float
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.pair_fingerprint or not self.density_sha256:
            raise ValueError("dense boundary oracle pair/density binding is empty")
        if (
            not self.applied_fft_fock_sha256
            or not self.dense_oracle_fock_sha256
            or self.applied_boundary.kind != "positive_subtolerance"
            or self.dense_boundary.kind != "exact"
            or self.applied_fft_recompute_exact is not True
        ):
            raise ValueError("dense boundary oracle map/classification binding is invalid")
        values = {
            "dense_fft_fock_max_abs_ev": self.dense_fft_fock_max_abs_ev,
            "dense_fft_boundary_gap_abs_difference_ev": self.dense_fft_boundary_gap_abs_difference_ev,
            "dense_fft_energy_abs_ev": self.dense_fft_energy_abs_ev,
            "fock_parity_gate_ev": self.fock_parity_gate_ev,
            "energy_parity_gate_ev": self.energy_parity_gate_ev,
        }
        for label, value in values.items():
            if not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if self.fock_parity_gate_ev <= 0.0 or self.energy_parity_gate_ev <= 0.0:
            raise ValueError("dense boundary oracle parity gates must be positive")
        if (
            self.dense_fft_fock_max_abs_ev > self.fock_parity_gate_ev
            or self.dense_fft_boundary_gap_abs_difference_ev > self.fock_parity_gate_ev
            or self.dense_fft_energy_abs_ev > self.energy_parity_gate_ev
        ):
            raise ValueError("dense boundary oracle parity gate failed")
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "dense_boundary_oracle_api_version": (
                        VITURI2024_SPIRAL_NORMAL_DENSE_BOUNDARY_ORACLE_API_VERSION
                    ),
                    "pair_fingerprint": self.pair_fingerprint,
                    "density_sha256": self.density_sha256,
                    "applied_fft_fock_sha256": self.applied_fft_fock_sha256,
                    "dense_oracle_fock_sha256": self.dense_oracle_fock_sha256,
                    "applied_boundary_fingerprint": self.applied_boundary.fingerprint,
                    "dense_boundary_fingerprint": self.dense_boundary.fingerprint,
                    "applied_fft_recompute_exact": self.applied_fft_recompute_exact,
                    **values,
                }
            ),
        )

@dataclass(frozen=True, slots=True)
class Vituri2024SpiralNormalInitializer:
    density_native: Array
    density_sha256: str
    h0_sha256: str
    prepared_fingerprint: str
    policy_fingerprint: str
    boundary: Vituri2024SpiralNormalBoundary
    h0_boundary_role: Literal[
        "unique_aufbau",
        "declared_common_seed_exact_shell",
        "validated_embedding_receipt",
    ]
    selected_occupied_flat_indices: tuple[int, ...]
    embedding_receipt_fingerprint: str | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        density = _readonly(self.density_native, np.dtype(np.complex128))
        if density.ndim != 3 or density.shape[:2] != (4, 4):
            raise ValueError("normal initializer density shape mismatch")
        if self.density_sha256 != _array_sha256(density):
            raise ValueError("normal initializer density hash mismatch")
        occupied = tuple(
            _strict_int(value, "initializer occupied index")
            for value in self.selected_occupied_flat_indices
        )
        if (
            tuple(sorted(set(occupied))) != occupied
            or len(occupied) != self.boundary.selected_rank
        ):
            raise ValueError("normal initializer occupied inventory is invalid")
        if self.h0_boundary_role == "unique_aufbau":
            valid = self.boundary.kind == "unique" and occupied == self.boundary.occupied_flat_indices
        elif self.h0_boundary_role == "declared_common_seed_exact_shell":
            selected_shell = set(occupied) & set(self.boundary.shell_flat_indices)
            valid = (
                self.boundary.kind == "exact"
                and set(self.boundary.strictly_below_flat_indices).issubset(occupied)
                and set(occupied).issubset(
                    set(self.boundary.strictly_below_flat_indices)
                    | set(self.boundary.shell_flat_indices)
                )
                and len(selected_shell) == self.boundary.shell_selected_rank
            )
        elif self.h0_boundary_role == "validated_embedding_receipt":
            # This coordinate projector is intentionally independent of the h0
            # boundary, but may only enter through a live validated embedding.
            valid = (
                len(occupied) == self.boundary.selected_rank
                and type(self.embedding_receipt_fingerprint) is str
                and len(self.embedding_receipt_fingerprint) == 64
            )
        else:
            raise ValueError("normal initializer h0 boundary role is invalid")
        if not valid:
            raise ValueError("normal initializer boundary/occupation receipt is inconsistent")
        if (
            self.h0_boundary_role != "validated_embedding_receipt"
            and self.embedding_receipt_fingerprint is not None
        ):
            raise ValueError("legacy normal initializer cannot bind an embedding receipt")
        object.__setattr__(self, "density_native", density)
        object.__setattr__(self, "selected_occupied_flat_indices", occupied)
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
                    "h0_boundary_role": self.h0_boundary_role,
                    "selected_occupied_flat_indices": occupied,
                    # Preserve the legacy initializer payload byte-for-byte:
                    # this key did not exist before supplied embeddings.
                    **(
                        {}
                        if self.embedding_receipt_fingerprint is None
                        else {
                            "embedding_receipt_fingerprint": (
                                self.embedding_receipt_fingerprint
                            )
                        }
                    ),
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
    dense_boundary_oracle_receipt: Vituri2024SpiralNormalDenseBoundaryOracleReceipt | None = None
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
        receipt = self.dense_boundary_oracle_receipt
        if receipt is not None:
            if type(receipt) is not Vituri2024SpiralNormalDenseBoundaryOracleReceipt:
                raise TypeError("branch trigger dense boundary oracle receipt must be typed")
            if (
                self.exact_fock_sha256 != receipt.dense_oracle_fock_sha256
                or self.previous_density_sha256 != receipt.density_sha256
                or self.boundary.fingerprint != receipt.dense_boundary.fingerprint
            ):
                raise ValueError("branch trigger dense boundary oracle binding mismatch")
        # Populations are retained as diagnostics, not as a selector.  The
        # declared closure exhausts every coordinate projector in an exact Fock
        # shell even when maximum-overlap continuation would prefer one subset.
        # This is the same within-root exact-shell branch universe used by the
        # established Vituri fixed-sector closure; it makes no claim about
        # source-group selection before this root was supplied.
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
                    **(
                        {}
                        if receipt is None
                        else {
                            "dense_boundary_oracle_receipt_fingerprint": receipt.fingerprint
                        }
                    ),
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

    def __post_init__(self) -> None:
        if (
            type(self.final_density_sha256) is not str
            or len(self.final_density_sha256) != 64
            or not self.path_ids
            or any(type(path_id) is not str or not path_id for path_id in self.path_ids)
            or not math.isfinite(self.energy_min_ev)
            or not math.isfinite(self.energy_max_ev)
            or self.energy_min_ev > self.energy_max_ev
        ):
            raise ValueError("normal stationary group evidence is invalid")


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
    dense_boundary_oracle_prepared_fingerprint: str | None = None
    dense_boundary_oracle_pair_fingerprint: str | None = None
    dense_boundary_oracle_receipt_fingerprints: tuple[str, ...] = ()
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
        receipts = tuple(self.dense_boundary_oracle_receipt_fingerprints)
        if tuple(sorted(set(receipts))) != receipts:
            raise ValueError("dense boundary oracle receipt inventory must be sorted and unique")
        oracle_bound = (
            self.dense_boundary_oracle_prepared_fingerprint is not None
            and self.dense_boundary_oracle_pair_fingerprint is not None
        )
        if oracle_bound != bool(
            self.dense_boundary_oracle_prepared_fingerprint
            or self.dense_boundary_oracle_pair_fingerprint
        ) or (receipts and not oracle_bound):
            raise ValueError("normal closure dense boundary oracle result binding is inconsistent")
        object.__setattr__(self, "dense_boundary_oracle_receipt_fingerprints", receipts)
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

    @property
    def exhaustion_scope(self) -> Literal[
        "conditional_on_common_h0_root", "conditional_on_supplied_root"
    ]:
        return (
            "conditional_on_supplied_root"
            if self.initializer.embedding_receipt_fingerprint is not None
            else "conditional_on_common_h0_root"
        )

    @property
    def global_source_inventory_exhausted(self) -> Literal[False]:
        return False

    @property
    def source_postselection_excluded(self) -> Literal[False]:
        return False


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


def _dense_boundary_oracle_pair_fingerprint(
    prepared: Vituri2024PreparedHFSpiral,
    dense_prepared: Vituri2024PreparedHFSpiral,
) -> str:
    """Validate one same-physics FFT/dense preparation pair."""

    for value, label in ((prepared, "prepared"), (dense_prepared, "dense_prepared")):
        if type(value) is not Vituri2024PreparedHFSpiral:
            raise TypeError(f"{label} must be Vituri2024PreparedHFSpiral")
        value.validate_live_state()
    if prepared.backend_kind != "fft" or dense_prepared.backend_kind != "dense":
        raise ValueError("dense boundary oracle requires primary FFT and oracle dense backends")
    if (
        prepared.choice.fingerprint != dense_prepared.choice.fingerprint
        or prepared.holes_per_valley != dense_prepared.holes_per_valley
        or prepared.precision != dense_prepared.precision
        or prepared.nk != dense_prepared.nk
    ):
        raise ValueError("dense boundary oracle preparation scalar/choice mismatch")
    arrays = (
        (prepared.ordered_mesh, dense_prepared.ordered_mesh, "ordered_mesh"),
        (
            prepared.shifted_momenta_by_valley,
            dense_prepared.shifted_momenta_by_valley,
            "shifted_momenta_by_valley",
        ),
        (
            prepared.active_band_states,
            dense_prepared.active_band_states,
            "active_band_states",
        ),
        (
            prepared.active_band_energies_by_valley,
            dense_prepared.active_band_energies_by_valley,
            "active_band_energies_by_valley",
        ),
        (prepared.h0_native, dense_prepared.h0_native, "h0_native"),
        (
            prepared.functional.normal_order_reference_conventional,
            dense_prepared.functional.normal_order_reference_conventional,
            "normal_order_reference_conventional",
        ),
    )
    for left, right, label in arrays:
        if not np.array_equal(left, right):
            raise ValueError(f"dense boundary oracle preparation array mismatch: {label}")
    left_gauge = prepared.gauge_receipt
    right_gauge = dense_prepared.gauge_receipt
    if (left_gauge is None) != (right_gauge is None) or (
        left_gauge is not None
        and right_gauge is not None
        and left_gauge.fingerprint != right_gauge.fingerprint
    ):
        raise ValueError("dense boundary oracle gauge receipt mismatch")
    for attribute in (
        "normal_order_reference_fingerprint",
        "interaction_fingerprint",
    ):
        if getattr(prepared.functional, attribute) != getattr(
            dense_prepared.functional, attribute
        ):
            raise ValueError(f"dense boundary oracle functional {attribute} mismatch")
    if (
        prepared.functional.mesh_receipt.fingerprint
        != dense_prepared.functional.mesh_receipt.fingerprint
        or prepared.functional.q0_choice.fingerprint
        != dense_prepared.functional.q0_choice.fingerprint
    ):
        raise ValueError("dense boundary oracle mesh/q0 physical-map mismatch")
    return _fingerprint(
        {
            "dense_boundary_oracle_api_version": (
                VITURI2024_SPIRAL_NORMAL_DENSE_BOUNDARY_ORACLE_API_VERSION
            ),
            "primary_prepared_fingerprint": prepared.fingerprint,
            "dense_prepared_fingerprint": dense_prepared.fingerprint,
            "choice_fingerprint": prepared.choice.fingerprint,
            "array_sha256": {
                label: _array_sha256(left) for left, _right, label in arrays
            },
            "gauge_fingerprint": None if left_gauge is None else left_gauge.fingerprint,
            "mesh_receipt_fingerprint": prepared.functional.mesh_receipt.fingerprint,
            "interaction_fingerprint": prepared.functional.interaction_fingerprint,
            "normal_order_reference_fingerprint": (
                prepared.functional.normal_order_reference_fingerprint
            ),
            "q0_choice_fingerprint": prepared.functional.q0_choice.fingerprint,
            "backend_roles": (prepared.backend_kind, dense_prepared.backend_kind),
        }
    )


def _build_dense_boundary_oracle_receipt(
    *,
    pair_fingerprint: str,
    density: Array,
    applied_fft_fock: Array,
    primary_recomputed_fock: Array,
    dense_oracle_fock: Array,
    applied_boundary: Vituri2024SpiralNormalBoundary,
    dense_boundary: Vituri2024SpiralNormalBoundary,
    primary_energy_ev: float,
    dense_energy_ev: float,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> Vituri2024SpiralNormalDenseBoundaryOracleReceipt:
    """Build a fail-closed same-density dense-oracle qualification receipt."""

    if applied_boundary.kind != "positive_subtolerance":
        raise ValueError("dense boundary oracle requires an applied positive-subtolerance map")
    if dense_boundary.kind != "exact":
        raise _ScientificTerminal(
            "dense_boundary_oracle_not_exact_rejection",
            "dense_boundary_oracle",
            "independent dense Fock does not have an exact selected-spin boundary",
            {
                "density": np.asarray(density).copy(),
                "applied_fft_fock": np.asarray(applied_fft_fock).copy(),
                "dense_oracle_fock": np.asarray(dense_oracle_fock).copy(),
            },
        )
    if not np.array_equal(applied_fft_fock, primary_recomputed_fock):
        raise _ScientificTerminal(
            "dense_boundary_oracle_applied_map_mismatch_rejection",
            "dense_boundary_oracle",
            "applied FFT Hamiltonian differs from the same-density FFT recomputation",
            {
                "density": np.asarray(density).copy(),
                "applied_fft_fock": np.asarray(applied_fft_fock).copy(),
                "primary_recomputed_fock": np.asarray(primary_recomputed_fock).copy(),
            },
        )
    fock_residual = _max_abs(
        np.asarray(dense_oracle_fock) - np.asarray(applied_fft_fock)
    )
    gap_residual = abs(dense_boundary.gap_ev - applied_boundary.gap_ev)
    energy_residual = abs(float(dense_energy_ev) - float(primary_energy_ev))
    fock_gate = max(
        applied_boundary.effective_tolerance_ev,
        dense_boundary.effective_tolerance_ev,
    )
    energy_scale = max(1.0, abs(float(primary_energy_ev)), abs(float(dense_energy_ev)))
    energy_gate = policy.energy_parity_relative_tolerance * energy_scale
    if fock_residual > fock_gate or gap_residual > fock_gate or energy_residual > energy_gate:
        raise _ScientificTerminal(
            "dense_boundary_oracle_parity_rejection",
            "dense_boundary_oracle",
            "dense and FFT same-density maps exceed the bound parity gates",
            {
                "density": np.asarray(density).copy(),
                "applied_fft_fock": np.asarray(applied_fft_fock).copy(),
                "dense_oracle_fock": np.asarray(dense_oracle_fock).copy(),
            },
        )
    return Vituri2024SpiralNormalDenseBoundaryOracleReceipt(
        pair_fingerprint=pair_fingerprint,
        density_sha256=_array_sha256(density),
        applied_fft_fock_sha256=_array_sha256(applied_fft_fock),
        dense_oracle_fock_sha256=_array_sha256(dense_oracle_fock),
        applied_boundary=applied_boundary,
        dense_boundary=dense_boundary,
        applied_fft_recompute_exact=True,
        dense_fft_fock_max_abs_ev=fock_residual,
        dense_fft_boundary_gap_abs_difference_ev=gap_residual,
        dense_fft_energy_abs_ev=energy_residual,
        fock_parity_gate_ev=fock_gate,
        energy_parity_gate_ev=energy_gate,
    )


def certify_vituri2024_spiral_normal_dense_boundary_oracle(
    prepared: Vituri2024PreparedHFSpiral,
    dense_prepared: Vituri2024PreparedHFSpiral,
    density_native: Array,
    applied_fft_hamiltonian: Array,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
) -> tuple[
    Array,
    Vituri2024SpiralNormalBoundary,
    Vituri2024SpiralNormalDenseBoundaryOracleReceipt,
]:
    """Certify one FFT positive-subtolerance map against dense float64 Fock.

    The old FFT rejection is not relabeled.  This function independently
    rebuilds both same-density maps and returns the exact dense map only after
    stored/applied FFT identity plus Fock, gap, and scalar-energy parity pass.
    """

    pair_fingerprint = _dense_boundary_oracle_pair_fingerprint(
        prepared, dense_prepared
    )
    policy.validate_live_state()
    density = np.asarray(density_native)
    applied = np.asarray(applied_fft_hamiltonian)
    expected_shape = (4, 4, prepared.nk)
    if (
        density.dtype != np.dtype(np.complex128)
        or density.shape != expected_shape
        or applied.dtype != np.dtype(np.complex128)
        or applied.shape != expected_shape
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(applied))
    ):
        raise ValueError("dense boundary oracle density/applied-map shape or dtype mismatch")
    applied_boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, applied, policy=policy
    )
    if applied_boundary.kind != "positive_subtolerance":
        raise ValueError("dense boundary oracle may only consume positive-subtolerance FFT maps")
    primary_recomputed = np.asarray(
        prepared.functional.fock(density), dtype=np.complex128
    )
    dense_fock = np.asarray(
        dense_prepared.functional.fock(density), dtype=np.complex128
    )
    dense_boundary = analyze_vituri2024_spiral_normal_boundary(
        dense_prepared, dense_fock, policy=policy
    )
    receipt = _build_dense_boundary_oracle_receipt(
        pair_fingerprint=pair_fingerprint,
        density=density,
        applied_fft_fock=applied,
        primary_recomputed_fock=primary_recomputed,
        dense_oracle_fock=dense_fock,
        applied_boundary=applied_boundary,
        dense_boundary=dense_boundary,
        primary_energy_ev=float(prepared.functional.energy(density)),
        dense_energy_ev=float(dense_prepared.functional.energy(density)),
        policy=policy,
    )
    return _readonly(dense_fock, np.dtype(np.complex128)), dense_boundary, receipt


def enumerate_vituri2024_spiral_normal_branch_choices(
    prepared: Vituri2024PreparedHFSpiral,
    hamiltonian: Array,
    previous_density_native: Array,
    boundary: Vituri2024SpiralNormalBoundary,
    *,
    generation: int,
    policy: Vituri2024SpiralNormalClosurePolicy,
    dense_boundary_oracle_receipt: (
        Vituri2024SpiralNormalDenseBoundaryOracleReceipt | None
    ) = None,
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
        dense_boundary_oracle_receipt=dense_boundary_oracle_receipt,
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


def _bind_supplied_normal_density(
    prepared: Vituri2024PreparedHFSpiral,
    density_native: object,
) -> tuple[Array, tuple[int, ...]]:
    """Privately bind one exact normal coordinate projector at target rank."""

    density = np.asarray(density_native)
    if (
        density.dtype != np.dtype(np.complex128)
        or density.shape != (4, 4, prepared.nk)
        or not np.all(np.isfinite(density))
    ):
        raise ValueError(
            "supplied normal density must be finite complex128 with shape (4,4,Nk)"
        )
    bound = _readonly(density, np.dtype(np.complex128))
    conventional = vituri2024_native_density_to_conventional_k_diagonal(bound)
    diagonal = np.zeros_like(conventional)
    indices = np.arange(4, dtype=np.int64)
    diagonal[indices, indices, :] = conventional[indices, indices, :]
    if not np.array_equal(conventional, diagonal):
        raise ValueError("supplied normal density must be exactly flavor diagonal")
    populations = conventional[indices, indices, :].real
    if (
        not np.array_equal(conventional[indices, indices, :].imag, np.zeros((4, prepared.nk)))
        or not np.array_equal(populations, np.rint(populations))
        or np.any((populations < 0.0) | (populations > 1.0))
    ):
        raise ValueError("supplied normal density must have exact binary populations")
    spectators = _spectator_flavors(prepared)
    if not np.array_equal(
        conventional[np.asarray(spectators, dtype=np.int64), np.asarray(spectators, dtype=np.int64), :],
        np.ones((2, prepared.nk), dtype=np.complex128),
    ):
        raise ValueError("supplied normal density must keep the spectator spin exactly full")
    selected = _selected_flavors(prepared)
    selected_diagonal = np.concatenate(
        [conventional[flavor, flavor, :].real for flavor in selected]
    )
    occupied = tuple(int(value) for value in np.flatnonzero(selected_diagonal == 1.0))
    if len(occupied) != prepared.selected_rank:
        raise ValueError("supplied normal density changed the selected-spin global rank")
    return bound, occupied


def build_vituri2024_spiral_normal_initializer(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
    density_embedding_receipt: object | None = None,
) -> Vituri2024SpiralNormalInitializer:
    """Bind the common-h0 seed or a factory-validated embedded projector.

    A target-domain density cannot be supplied bare.  The optional root must
    arrive through a typed embedding receipt whose complete live preparation,
    density hash, scalar, label-map, and index-partition contract is rechecked.
    Exhaustion from such a root is conditional and says nothing about whether
    a source group was postselected before embedding.
    """

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be Vituri2024PreparedHFSpiral")
    prepared.validate_live_state()
    policy.validate_live_state()
    if _max_abs(prepared.functional.normal_order_reference_conventional) != 0.0:
        raise ValueError(
            "spiral normal closure currently requires the exact zero normal-order reference"
        )
    boundary = analyze_vituri2024_spiral_normal_boundary(
        prepared, np.asarray(prepared.h0_native), policy=policy
    )
    if density_embedding_receipt is not None:
        density_embedding_receipt = (
            validate_vituri2024_spiral_density_embedding_receipt(
                density_embedding_receipt
            )
        )
        if (
            density_embedding_receipt.target_prepared_fingerprint != prepared.fingerprint
            or density_embedding_receipt.target_prepared.fingerprint != prepared.fingerprint
            or density_embedding_receipt.target_density_sha256
            != _array_sha256(density_embedding_receipt.embedded_density_native)
        ):
            raise ValueError("embedding receipt does not hash-match target prepared/density")
        supplied, occupied = _bind_supplied_normal_density(
            prepared, density_embedding_receipt.embedded_density_native
        )
        return Vituri2024SpiralNormalInitializer(
            density_native=supplied,
            density_sha256=_array_sha256(supplied),
            h0_sha256=_array_sha256(prepared.h0_native),
            prepared_fingerprint=prepared.fingerprint,
            policy_fingerprint=policy.fingerprint,
            boundary=boundary,
            h0_boundary_role="validated_embedding_receipt",
            selected_occupied_flat_indices=occupied,
            embedding_receipt_fingerprint=density_embedding_receipt.fingerprint,
        )
    if boundary.kind == "positive_subtolerance":
        raise _ScientificTerminal(
            "h0_normal_positive_subtolerance_rejection",
            "common_initializer",
            "common normal h0 boundary has a positive sub-tolerance splitting",
            {"h0": np.asarray(prepared.h0_native).copy()},
        )
    existing = make_vituri2024_spiral_initial_density(prepared, init_mode="normal")
    conventional = vituri2024_native_density_to_conventional_k_diagonal(existing)
    selected = _selected_flavors(prepared)
    selected_diagonal = np.concatenate(
        [conventional[flavor, flavor, :].real for flavor in selected]
    )
    if _max_abs(selected_diagonal - np.rint(selected_diagonal)) != 0.0:
        raise RuntimeError("declared common normal initializer is not an exact coordinate projector")
    occupied = tuple(int(value) for value in np.flatnonzero(selected_diagonal == 1.0))
    h0_flat = np.concatenate(
        [np.asarray(prepared.h0_native[flavor, flavor, :].real) for flavor in selected]
    )
    canonical_flat_indices = np.arange(2 * prepared.nk, dtype=np.int64)
    # Independently reproduce the source scout's declared descending-energy,
    # canonical-C-order seed rule. The secondary integer key is explicit;
    # this is an initializer lineage receipt, not a physical shell selector.
    descending_seed_order = np.lexsort((canonical_flat_indices, -h0_flat))
    hole_count = 2 * prepared.holes_per_valley
    expected_occupied = tuple(
        int(value)
        for value in np.flatnonzero(
            ~np.isin(canonical_flat_indices, descending_seed_order[:hole_count])
        )
    )
    if occupied != expected_occupied:
        raise RuntimeError("declared common normal initializer canonical seed inventory drifted")
    if boundary.kind == "unique":
        independent = _density_from_boundary(prepared, boundary, None)
        if not np.array_equal(independent, existing):
            raise RuntimeError("independent threshold initializer disagrees with spiral normal initializer")
        role: Literal["unique_aufbau", "declared_common_seed_exact_shell"] = "unique_aufbau"
    else:
        role = "declared_common_seed_exact_shell"
    return Vituri2024SpiralNormalInitializer(
        density_native=existing,
        density_sha256=_array_sha256(existing),
        h0_sha256=_array_sha256(prepared.h0_native),
        prepared_fingerprint=prepared.fingerprint,
        policy_fingerprint=policy.fingerprint,
        boundary=boundary,
        h0_boundary_role=role,
        selected_occupied_flat_indices=occupied,
    )


def _run_path(
    prepared: Vituri2024PreparedHFSpiral,
    initializer: Vituri2024SpiralNormalInitializer,
    path: Vituri2024SpiralNormalBranchPath,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy,
    density_embedding_receipt: object | None,
    dense_boundary_oracle_prepared: Vituri2024PreparedHFSpiral | None,
    dense_boundary_oracle_cache: dict[
        str,
        tuple[
            Array,
            Vituri2024SpiralNormalBoundary,
            Vituri2024SpiralNormalDenseBoundaryOracleReceipt,
        ],
    ],
    dense_boundary_oracle_receipt_fingerprints: list[str],
) -> NormalPathOutcome:
    rebuilt = build_vituri2024_spiral_normal_initializer(
        prepared,
        policy=policy,
        density_embedding_receipt=density_embedding_receipt,
    )
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
        branch_matrix = matrix
        oracle_receipt: Vituri2024SpiralNormalDenseBoundaryOracleReceipt | None = None
        if boundary.kind == "positive_subtolerance":
            if dense_boundary_oracle_prepared is None:
                raise _ScientificTerminal(
                    "positive_subtolerance_splitting_rejection",
                    "density_update",
                    "positive selected-spin splitting lies below the effective tolerance",
                    {"density": previous, "hamiltonian": matrix.copy()},
                )
            density_sha256 = _array_sha256(previous)
            cached = dense_boundary_oracle_cache.get(density_sha256)
            if cached is None:
                cached = certify_vituri2024_spiral_normal_dense_boundary_oracle(
                    prepared,
                    dense_boundary_oracle_prepared,
                    previous,
                    matrix,
                    policy=policy,
                )
                dense_boundary_oracle_cache[density_sha256] = cached
            else:
                cached_fock, cached_boundary, cached_receipt = cached
                if (
                    cached_receipt.density_sha256 != density_sha256
                    or cached_receipt.applied_fft_fock_sha256 != _array_sha256(matrix)
                    or cached_receipt.dense_oracle_fock_sha256
                    != _array_sha256(cached_fock)
                    or cached_receipt.dense_boundary.fingerprint
                    != cached_boundary.fingerprint
                ):
                    raise RuntimeError("dense boundary oracle cache binding mismatch")
            branch_matrix, boundary, oracle_receipt = cached
            dense_boundary_oracle_receipt_fingerprints.append(
                oracle_receipt.fingerprint
            )
        terminal_evidence = {
            "density": previous,
            "hamiltonian": matrix.copy(),
        }
        if oracle_receipt is not None:
            terminal_evidence["dense_oracle_hamiltonian"] = np.asarray(
                branch_matrix
            ).copy()
        choice: Vituri2024SpiralNormalBranchChoice | None = None
        if boundary.kind == "exact":
            choices = enumerate_vituri2024_spiral_normal_branch_choices(
                prepared,
                branch_matrix,
                previous,
                boundary,
                generation=len(consumed),
                policy=policy,
                dense_boundary_oracle_receipt=oracle_receipt,
            )
            if len(consumed) >= len(path.choices):
                if would_be_final_map():
                    raise _ScientificTerminal(
                        "branch_frontier_in_final_map_rejection",
                        "final_density_recomputation",
                        "an unresolved normal branch frontier appeared only in the final map",
                        terminal_evidence,
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
                    terminal_evidence,
                )
        raw = _density_from_boundary(prepared, boundary, choice)
        energies = np.asarray(
            np.real(np.diagonal(branch_matrix, axis1=0, axis2=1)).T,
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
                "dense_boundary_oracle_used": float(oracle_receipt is not None),
            },
        )
        pending = {
            "update_id": id(update),
            "raw_sha256": _array_sha256(raw),
            "applied_fock_sha256": _array_sha256(matrix),
            "boundary_fock_sha256": _array_sha256(branch_matrix),
            "oracle_receipt_fingerprint": (
                None if oracle_receipt is None else oracle_receipt.fingerprint
            ),
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
        if pending["applied_fock_sha256"] != _array_sha256(step.total_hamiltonian):
            raise RuntimeError("normal closure applied FFT-Fock binding failed")
        if pending["previous_sha256"] != _array_sha256(step.previous_density):
            raise RuntimeError("normal closure previous-density binding failed")
        if step.oda_lambda != 1.0:
            raise RuntimeError("normal closure full-step policy returned a non-unit lambda")
        if not np.array_equal(step.mixed_density, step.density_update.density):
            raise RuntimeError("normal closure full step did not install the raw projector exactly")
        choice = pending["choice"]
        if choice is not None:
            receipt = choice.trigger.dense_boundary_oracle_receipt
            if (
                pending["boundary_fock_sha256"] != choice.trigger.exact_fock_sha256
                or pending["oracle_receipt_fingerprint"]
                != (None if receipt is None else receipt.fingerprint)
            ):
                raise RuntimeError("normal closure branch boundary-Fock binding failed")
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


def _strict_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _strict_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise TypeError(f"{label} must be an exact tuple of nonempty strings")
    return value


def _stationary_group_fingerprint(
    group: Vituri2024SpiralNormalStationaryGroup,
) -> str:
    if type(group) is not Vituri2024SpiralNormalStationaryGroup:
        raise TypeError("source stationary group must be exact typed evidence")
    return _fingerprint(
        {
            "final_density_sha256": group.final_density_sha256,
            "path_ids": group.path_ids,
            "energy_min_ev": group.energy_min_ev,
            "energy_max_ev": group.energy_max_ev,
        }
    )


def _source_closure_lineage_snapshot(
    result: Vituri2024SpiralNormalClosureResult,
) -> dict[str, object]:
    """Build the exact scalar/tuple digest schema for a source closure."""

    if type(result) is not Vituri2024SpiralNormalClosureResult:
        raise TypeError("source_closure_result must be typed")
    expected_result_schema = (
        "policy",
        "initializer",
        "prepared_fingerprint",
        "nodes",
        "endpoints",
        "rejections",
        "stationary_groups",
        "branch_tree_exhausted",
        "deterministic_terminal_replay_verified",
        "all_normal_endpoints_stationary",
        "all_applied_steps_full_step",
        "dense_boundary_oracle_prepared_fingerprint",
        "dense_boundary_oracle_pair_fingerprint",
        "dense_boundary_oracle_receipt_fingerprints",
        "candidate_finite_domain_only",
        "same_q_energy_comparison_authorized",
        "uv_authority",
        "unrestricted_ground_state_authority",
        "local_hessian_authority",
        "fig2_reproduction_authority",
        "tdhf_authority",
        "production_authority",
    )
    if tuple(result.__dataclass_fields__) != expected_result_schema:
        raise TypeError("source closure result schema drifted")
    result.policy.validate_live_state()
    _strict_sha256(result.prepared_fingerprint, "source prepared fingerprint")
    if result.initializer.prepared_fingerprint != result.prepared_fingerprint:
        raise ValueError("source initializer/prepared fingerprint mismatch")
    initializer_payload = {
        "density_sha256": result.initializer.density_sha256,
        "h0_sha256": result.initializer.h0_sha256,
        "prepared_fingerprint": result.initializer.prepared_fingerprint,
        "policy_fingerprint": result.initializer.policy_fingerprint,
        "boundary_fingerprint": result.initializer.boundary.fingerprint,
        "h0_boundary_role": result.initializer.h0_boundary_role,
        "selected_occupied_flat_indices": (
            result.initializer.selected_occupied_flat_indices
        ),
        **(
            {}
            if result.initializer.embedding_receipt_fingerprint is None
            else {
                "embedding_receipt_fingerprint": (
                    result.initializer.embedding_receipt_fingerprint
                )
            }
        ),
    }
    if (
        result.initializer.policy_fingerprint != result.policy.fingerprint
        or result.initializer.density_sha256
        != _array_sha256(result.initializer.density_native)
        or result.initializer.fingerprint != _fingerprint(initializer_payload)
    ):
        raise ValueError("source initializer live fingerprint/hash drifted")
    for name in (
        "nodes",
        "endpoints",
        "rejections",
        "stationary_groups",
        "dense_boundary_oracle_receipt_fingerprints",
    ):
        if type(getattr(result, name)) is not tuple:
            raise TypeError(f"source closure {name} must be an exact tuple")
    for name in (
        "branch_tree_exhausted",
        "deterministic_terminal_replay_verified",
        "all_normal_endpoints_stationary",
        "all_applied_steps_full_step",
        "candidate_finite_domain_only",
        "same_q_energy_comparison_authorized",
        "uv_authority",
        "unrestricted_ground_state_authority",
        "local_hessian_authority",
        "fig2_reproduction_authority",
        "tdhf_authority",
        "production_authority",
    ):
        if type(getattr(result, name)) is not bool:
            raise TypeError(f"source closure {name} must be an exact bool")
    if (
        result.branch_tree_exhausted is not True
        or result.deterministic_terminal_replay_verified is not True
        or result.global_source_inventory_exhausted is not False
        or result.source_postselection_excluded is not False
        or result.candidate_finite_domain_only is not True
        or any(
            (
                result.same_q_energy_comparison_authorized,
                result.uv_authority,
                result.unrestricted_ground_state_authority,
                result.local_hessian_authority,
                result.fig2_reproduction_authority,
                result.tdhf_authority,
                result.production_authority,
            )
        )
    ):
        raise ValueError("source closure authority/scope is not lineage-qualified")
    if any(type(node) is not Vituri2024SpiralNormalBFSNode for node in result.nodes):
        raise TypeError("source closure node inventory is not exact typed evidence")
    if any(type(item) is not Vituri2024SpiralNormalEndpoint for item in result.endpoints):
        raise TypeError("source closure endpoint inventory is not exact typed evidence")
    if any(
        type(item) is not Vituri2024SpiralNormalScientificRejection
        for item in result.rejections
    ):
        raise TypeError("source closure rejection inventory is not exact typed evidence")
    if any(
        type(item) is not Vituri2024SpiralNormalStationaryGroup
        for item in result.stationary_groups
    ):
        raise TypeError("source closure stationary-group inventory is not exact typed evidence")
    expected_density_shape = result.initializer.density_native.shape
    for index, endpoint in enumerate(result.endpoints):
        density = endpoint.final_density
        label = f"source endpoint[{index}] final density"
        if type(density) is not np.ndarray:
            raise TypeError(f"{label} must be an exact ndarray")
        if density.dtype != np.dtype(np.complex128):
            raise TypeError(f"{label} must have dtype complex128")
        if (
            density.ndim != 3
            or density.shape[:2] != (4, 4)
            or density.shape != expected_density_shape
        ):
            raise ValueError(f"{label} shape mismatch")
        if not np.all(np.isfinite(density)):
            raise ValueError(f"{label} must be finite")
        _strict_sha256(endpoint.final_density_sha256, f"{label} SHA256")
        if _array_sha256(density) != endpoint.final_density_sha256:
            raise ValueError(f"{label} live hash mismatch")
    recomputed_groups = _stationary_groups(result.endpoints)
    if recomputed_groups != result.stationary_groups:
        raise ValueError("source closure stationary-group inventory drifted")
    expected_all_stationary = bool(result.endpoints) and all(
        endpoint.stationary for endpoint in result.endpoints
    )
    if result.all_normal_endpoints_stationary != expected_all_stationary:
        raise ValueError("source closure aggregate stationarity flag drifted")
    dense_receipts = _strict_string_tuple(
        result.dense_boundary_oracle_receipt_fingerprints,
        "dense boundary oracle receipt fingerprints",
    )
    if tuple(sorted(set(dense_receipts))) != dense_receipts:
        raise ValueError("source closure dense-oracle inventory is not sorted/unique")
    for name in (
        "dense_boundary_oracle_prepared_fingerprint",
        "dense_boundary_oracle_pair_fingerprint",
    ):
        value = getattr(result, name)
        if value is not None:
            _strict_sha256(value, name)
    oracle_bound = (
        result.dense_boundary_oracle_prepared_fingerprint is not None
        and result.dense_boundary_oracle_pair_fingerprint is not None
    )
    if oracle_bound != bool(
        result.dense_boundary_oracle_prepared_fingerprint
        or result.dense_boundary_oracle_pair_fingerprint
    ) or (dense_receipts and not oracle_bound):
        raise ValueError("source closure dense-oracle binding is inconsistent")

    node_digests = tuple(
        _fingerprint(
            {
                "path_id": node.path.path_id,
                "path_fingerprint": node.path.fingerprint,
                "outcome": node.outcome,
                "child_path_ids": node.child_path_ids,
            }
        )
        for node in result.nodes
    )
    endpoint_digests = tuple(_outcome_digest(item) for item in result.endpoints)
    rejection_digests = tuple(_outcome_digest(item) for item in result.rejections)
    group_fingerprints = tuple(
        _stationary_group_fingerprint(group) for group in result.stationary_groups
    )
    payload = {
        "api_version": VITURI2024_SPIRAL_NORMAL_SOURCE_LINEAGE_API_VERSION,
        "result_schema": expected_result_schema,
        "prepared_fingerprint": result.prepared_fingerprint,
        "policy_schema_and_values": tuple(
            (name, getattr(result.policy, name))
            for name in result.policy.__dataclass_fields__
        ),
        "initializer_payload": tuple(initializer_payload.items()),
        "initializer_fingerprint": result.initializer.fingerprint,
        "node_path_ids": tuple(node.path.path_id for node in result.nodes),
        "node_digests": node_digests,
        "endpoint_path_ids": tuple(item.path.path_id for item in result.endpoints),
        "endpoint_outcome_digests": endpoint_digests,
        "rejection_path_ids": tuple(item.path.path_id for item in result.rejections),
        "rejection_outcome_digests": rejection_digests,
        "stationary_group_fingerprints": group_fingerprints,
        "stationary_group_path_ids": tuple(
            group.path_ids for group in result.stationary_groups
        ),
        "branch_tree_exhausted": result.branch_tree_exhausted,
        "exhaustion_scope": result.exhaustion_scope,
        "global_source_inventory_exhausted": (
            result.global_source_inventory_exhausted
        ),
        "source_postselection_excluded": result.source_postselection_excluded,
        "deterministic_terminal_replay_verified": (
            result.deterministic_terminal_replay_verified
        ),
        "all_normal_endpoints_stationary": result.all_normal_endpoints_stationary,
        "all_applied_steps_full_step": result.all_applied_steps_full_step,
        "dense_boundary_oracle_prepared_fingerprint": (
            result.dense_boundary_oracle_prepared_fingerprint
        ),
        "dense_boundary_oracle_pair_fingerprint": (
            result.dense_boundary_oracle_pair_fingerprint
        ),
        "dense_boundary_oracle_receipt_fingerprints": dense_receipts,
        "candidate_finite_domain_only": result.candidate_finite_domain_only,
        "same_q_energy_comparison_authorized": (
            result.same_q_energy_comparison_authorized
        ),
        "uv_authority": result.uv_authority,
        "unrestricted_ground_state_authority": (
            result.unrestricted_ground_state_authority
        ),
        "local_hessian_authority": result.local_hessian_authority,
        "fig2_reproduction_authority": result.fig2_reproduction_authority,
        "tdhf_authority": result.tdhf_authority,
        "production_authority": result.production_authority,
    }
    return {
        "digest": _fingerprint(payload),
        "node_digests": node_digests,
        "endpoint_digests": endpoint_digests,
        "rejection_digests": rejection_digests,
        "group_fingerprints": group_fingerprints,
    }


def _source_closure_lineage_digest(
    result: Vituri2024SpiralNormalClosureResult,
) -> str:
    return str(_source_closure_lineage_snapshot(result)["digest"])


@dataclass(frozen=True, slots=True, init=False)
class _Vituri2024SpiralNormalSourceLineageReceipt:
    """Private, detached, array-free source-group lineage factory product."""

    source_closure_digest: str
    source_prepared_fingerprint: str
    source_policy_fingerprint: str
    source_initializer_fingerprint: str
    source_initializer_density_sha256: str
    source_group_index: int
    source_group_fingerprint: str
    source_group_final_density_sha256: str
    source_group_endpoint_path_ids: tuple[str, ...]
    all_source_endpoint_path_ids: tuple[str, ...]
    source_node_digests: tuple[str, ...]
    source_endpoint_outcome_digests: tuple[str, ...]
    source_rejection_outcome_digests: tuple[str, ...]
    source_stationary_group_fingerprints: tuple[str, ...]
    source_group_endpoint_multiplicity: int
    all_source_endpoint_multiplicity: int
    stationary_group_multiplicity: int
    stationary_endpoint_multiplicity: int
    source_rejection_multiplicity: int
    exhaustion_scope: Literal[
        "conditional_on_common_h0_root", "conditional_on_supplied_root"
    ]
    branch_tree_exhausted: Literal[True]
    deterministic_terminal_replay_verified: Literal[True]
    all_normal_endpoints_stationary: bool
    all_applied_steps_full_step: bool
    dense_boundary_oracle_prepared_fingerprint: str | None
    dense_boundary_oracle_pair_fingerprint: str | None
    dense_boundary_oracle_receipt_fingerprints: tuple[str, ...]
    candidate_finite_domain_only: Literal[True]
    same_q_energy_comparison_authorized: Literal[False]
    uv_authority: Literal[False]
    unrestricted_ground_state_authority: Literal[False]
    local_hessian_authority: Literal[False]
    fig2_reproduction_authority: Literal[False]
    tdhf_authority: Literal[False]
    production_authority: Literal[False]
    global_source_inventory_exhausted: Literal[False]
    source_group_selected: Literal[True]
    source_postselection_excluded: Literal[False]
    fingerprint: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("normal source-lineage receipts are factory-only")

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_SPIRAL_NORMAL_SOURCE_LINEAGE_API_VERSION,
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "fingerprint"
                },
            }
        )

    def _validate_detached_state(self) -> None:
        expected_schema = (
            "source_closure_digest",
            "source_prepared_fingerprint",
            "source_policy_fingerprint",
            "source_initializer_fingerprint",
            "source_initializer_density_sha256",
            "source_group_index",
            "source_group_fingerprint",
            "source_group_final_density_sha256",
            "source_group_endpoint_path_ids",
            "all_source_endpoint_path_ids",
            "source_node_digests",
            "source_endpoint_outcome_digests",
            "source_rejection_outcome_digests",
            "source_stationary_group_fingerprints",
            "source_group_endpoint_multiplicity",
            "all_source_endpoint_multiplicity",
            "stationary_group_multiplicity",
            "stationary_endpoint_multiplicity",
            "source_rejection_multiplicity",
            "exhaustion_scope",
            "branch_tree_exhausted",
            "deterministic_terminal_replay_verified",
            "all_normal_endpoints_stationary",
            "all_applied_steps_full_step",
            "dense_boundary_oracle_prepared_fingerprint",
            "dense_boundary_oracle_pair_fingerprint",
            "dense_boundary_oracle_receipt_fingerprints",
            "candidate_finite_domain_only",
            "same_q_energy_comparison_authorized",
            "uv_authority",
            "unrestricted_ground_state_authority",
            "local_hessian_authority",
            "fig2_reproduction_authority",
            "tdhf_authority",
            "production_authority",
            "global_source_inventory_exhausted",
            "source_group_selected",
            "source_postselection_excluded",
            "fingerprint",
        )
        if tuple(self.__dataclass_fields__) != expected_schema:
            raise TypeError("source-lineage receipt schema drifted")
        for name in (
            "source_closure_digest",
            "source_prepared_fingerprint",
            "source_policy_fingerprint",
            "source_initializer_fingerprint",
            "source_initializer_density_sha256",
            "source_group_fingerprint",
            "source_group_final_density_sha256",
            "fingerprint",
        ):
            _strict_sha256(getattr(self, name), name)
        for name in (
            "source_node_digests",
            "source_endpoint_outcome_digests",
            "source_rejection_outcome_digests",
            "source_stationary_group_fingerprints",
            "dense_boundary_oracle_receipt_fingerprints",
        ):
            values = _strict_string_tuple(getattr(self, name), name)
            for index, value in enumerate(values):
                _strict_sha256(value, f"{name}[{index}]")
        group_paths = _strict_string_tuple(
            self.source_group_endpoint_path_ids,
            "source_group_endpoint_path_ids",
        )
        all_paths = _strict_string_tuple(
            self.all_source_endpoint_path_ids, "all_source_endpoint_path_ids"
        )
        if not group_paths:
            raise ValueError("source-group path inventory must be nonempty")
        for name in (
            "source_group_index",
            "source_group_endpoint_multiplicity",
            "all_source_endpoint_multiplicity",
            "stationary_group_multiplicity",
            "stationary_endpoint_multiplicity",
            "source_rejection_multiplicity",
        ):
            value = _strict_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.source_group_index >= self.stationary_group_multiplicity
            or self.source_group_fingerprint
            != self.source_stationary_group_fingerprints[self.source_group_index]
            or not set(group_paths).issubset(all_paths)
            or self.source_group_endpoint_multiplicity != len(group_paths)
            or self.all_source_endpoint_multiplicity != len(all_paths)
            or self.all_source_endpoint_multiplicity
            != len(self.source_endpoint_outcome_digests)
            or self.stationary_group_multiplicity
            != len(self.source_stationary_group_fingerprints)
            or self.stationary_endpoint_multiplicity < len(group_paths)
            or self.stationary_endpoint_multiplicity > len(all_paths)
            or (
                self.all_normal_endpoints_stationary
                and self.stationary_endpoint_multiplicity != len(all_paths)
            )
            or self.source_rejection_multiplicity
            != len(self.source_rejection_outcome_digests)
        ):
            raise ValueError("source-lineage multiplicity schema is inconsistent")
        for name in (
            "branch_tree_exhausted",
            "deterministic_terminal_replay_verified",
            "all_normal_endpoints_stationary",
            "all_applied_steps_full_step",
            "candidate_finite_domain_only",
            "same_q_energy_comparison_authorized",
            "uv_authority",
            "unrestricted_ground_state_authority",
            "local_hessian_authority",
            "fig2_reproduction_authority",
            "tdhf_authority",
            "production_authority",
            "global_source_inventory_exhausted",
            "source_group_selected",
            "source_postselection_excluded",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if (
            type(self.exhaustion_scope) is not str
            or self.exhaustion_scope
            not in ("conditional_on_common_h0_root", "conditional_on_supplied_root")
            or self.branch_tree_exhausted is not True
            or self.deterministic_terminal_replay_verified is not True
            or self.candidate_finite_domain_only is not True
            or self.global_source_inventory_exhausted is not False
            or self.source_group_selected is not True
            or self.source_postselection_excluded is not False
            or any(
                (
                    self.same_q_energy_comparison_authorized,
                    self.uv_authority,
                    self.unrestricted_ground_state_authority,
                    self.local_hessian_authority,
                    self.fig2_reproduction_authority,
                    self.tdhf_authority,
                    self.production_authority,
                )
            )
        ):
            raise ValueError("source-lineage authority/scope flags are invalid")
        for name in (
            "dense_boundary_oracle_prepared_fingerprint",
            "dense_boundary_oracle_pair_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_sha256(value, name)
        oracle_bound = (
            self.dense_boundary_oracle_prepared_fingerprint is not None
            and self.dense_boundary_oracle_pair_fingerprint is not None
        )
        if oracle_bound != bool(
            self.dense_boundary_oracle_prepared_fingerprint
            or self.dense_boundary_oracle_pair_fingerprint
        ) or (self.dense_boundary_oracle_receipt_fingerprints and not oracle_bound):
            raise ValueError("source-lineage dense-oracle binding is inconsistent")
        allowed_scalars = (bytes, str, int, float, bool, type(None))

        def detached(value: object) -> bool:
            return type(value) in allowed_scalars or (
                type(value) is tuple and all(detached(item) for item in value)
            )

        if any(
            not detached(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        ):
            raise TypeError("source-lineage receipt must remain bytes/scalars/tuples only")
        if self._current_fingerprint() != self.fingerprint:
            raise ValueError("source-lineage receipt fingerprint drifted")




def _source_lineage_fields(
    source_closure_result: Vituri2024SpiralNormalClosureResult,
    source_group_index: int,
    snapshot: dict[str, object],
) -> dict[str, object]:
    group = source_closure_result.stationary_groups[source_group_index]
    return {
        "source_closure_digest": snapshot["digest"],
        "source_prepared_fingerprint": source_closure_result.prepared_fingerprint,
        "source_policy_fingerprint": source_closure_result.policy.fingerprint,
        "source_initializer_fingerprint": source_closure_result.initializer.fingerprint,
        "source_initializer_density_sha256": source_closure_result.initializer.density_sha256,
        "source_group_index": source_group_index,
        "source_group_fingerprint": _stationary_group_fingerprint(group),
        "source_group_final_density_sha256": group.final_density_sha256,
        "source_group_endpoint_path_ids": group.path_ids,
        "all_source_endpoint_path_ids": tuple(
            endpoint.path.path_id for endpoint in source_closure_result.endpoints
        ),
        "source_node_digests": snapshot["node_digests"],
        "source_endpoint_outcome_digests": snapshot["endpoint_digests"],
        "source_rejection_outcome_digests": snapshot["rejection_digests"],
        "source_stationary_group_fingerprints": snapshot["group_fingerprints"],
        "source_group_endpoint_multiplicity": len(group.path_ids),
        "all_source_endpoint_multiplicity": len(source_closure_result.endpoints),
        "stationary_group_multiplicity": len(source_closure_result.stationary_groups),
        "stationary_endpoint_multiplicity": sum(
            len(item.path_ids) for item in source_closure_result.stationary_groups
        ),
        "source_rejection_multiplicity": len(source_closure_result.rejections),
        "exhaustion_scope": source_closure_result.exhaustion_scope,
        "branch_tree_exhausted": source_closure_result.branch_tree_exhausted,
        "deterministic_terminal_replay_verified": source_closure_result.deterministic_terminal_replay_verified,
        "all_normal_endpoints_stationary": source_closure_result.all_normal_endpoints_stationary,
        "all_applied_steps_full_step": source_closure_result.all_applied_steps_full_step,
        "dense_boundary_oracle_prepared_fingerprint": source_closure_result.dense_boundary_oracle_prepared_fingerprint,
        "dense_boundary_oracle_pair_fingerprint": source_closure_result.dense_boundary_oracle_pair_fingerprint,
        "dense_boundary_oracle_receipt_fingerprints": source_closure_result.dense_boundary_oracle_receipt_fingerprints,
        "candidate_finite_domain_only": source_closure_result.candidate_finite_domain_only,
        "same_q_energy_comparison_authorized": source_closure_result.same_q_energy_comparison_authorized,
        "uv_authority": source_closure_result.uv_authority,
        "unrestricted_ground_state_authority": source_closure_result.unrestricted_ground_state_authority,
        "local_hessian_authority": source_closure_result.local_hessian_authority,
        "fig2_reproduction_authority": source_closure_result.fig2_reproduction_authority,
        "tdhf_authority": source_closure_result.tdhf_authority,
        "production_authority": source_closure_result.production_authority,
        "global_source_inventory_exhausted": False,
        "source_group_selected": True,
        "source_postselection_excluded": False,
    }


def _make_source_lineage_authority_api(run_implementation):
    """Share strong result/product identity registration only inside this closure."""

    issued_identities: dict[int, tuple[object, str]] = {}
    complete_inventory_qualification_by_id: dict[int, tuple[str, str]] = {}

    def require_registered_result(
        result: object,
    ) -> tuple[Vituri2024SpiralNormalClosureResult, dict[str, object]]:
        if type(result) is not Vituri2024SpiralNormalClosureResult:
            raise TypeError("source_closure_result must be typed")
        issued = issued_identities.get(id(result))
        if issued is None or issued[0] is not result:
            raise TypeError(
                "source closure result identity is not registered by the public runner"
            )
        snapshot = _source_closure_lineage_snapshot(result)
        if issued[1] != snapshot["digest"]:
            raise ValueError("source closure result changed after runner registration")
        return result, snapshot

    def construct(**fields: object) -> _Vituri2024SpiralNormalSourceLineageReceipt:
        expected_fields = tuple(
            name
            for name in _Vituri2024SpiralNormalSourceLineageReceipt.__dataclass_fields__
            if name != "fingerprint"
        )
        if set(fields) != set(expected_fields):
            raise TypeError("source-lineage receipt factory field schema drifted")
        product = object.__new__(_Vituri2024SpiralNormalSourceLineageReceipt)
        for name in expected_fields:
            object.__setattr__(product, name, fields[name])
        object.__setattr__(product, "fingerprint", product._current_fingerprint())
        product._validate_detached_state()
        issued_state_fingerprint = product._current_fingerprint()
        if issued_state_fingerprint != product.fingerprint:
            raise RuntimeError("source-lineage receipt issuance fingerprint drifted")
        issued_identities[id(product)] = (product, issued_state_fingerprint)
        return product

    def validate(
        product: object, *, require_complete_inventory: bool = False
    ) -> _Vituri2024SpiralNormalSourceLineageReceipt:
        if type(require_complete_inventory) is not bool:
            raise TypeError("require_complete_inventory must be an exact bool")
        if type(product) is not _Vituri2024SpiralNormalSourceLineageReceipt:
            raise TypeError("source-lineage receipt must be a factory product")
        issued = issued_identities.get(id(product))
        if issued is None or issued[0] is not product:
            raise TypeError("source-lineage receipt identity is not registered")
        product._validate_detached_state()
        if product._current_fingerprint() != issued[1]:
            raise ValueError("source-lineage receipt changed after factory issuance")
        if require_complete_inventory and complete_inventory_qualification_by_id.get(
            id(product)
        ) != (issued[1], product.source_closure_digest):
            raise ValueError("source-lineage receipt lacks complete-inventory validation")
        return product

    def run_vituri2024_spiral_normal_exact_shell_closure(
        prepared: Vituri2024PreparedHFSpiral,
        *,
        policy: Vituri2024SpiralNormalClosurePolicy | None = None,
        dense_boundary_oracle_prepared: Vituri2024PreparedHFSpiral | None = None,
        density_embedding_receipt: object | None = None,
    ) -> Vituri2024SpiralNormalClosureResult:
        """Run, live-validate, and identity-register one closure result."""

        result = run_implementation(
            prepared,
            policy=policy,
            dense_boundary_oracle_prepared=dense_boundary_oracle_prepared,
            density_embedding_receipt=density_embedding_receipt,
        )
        snapshot = _source_closure_lineage_snapshot(result)
        identity = id(result)
        if identity in issued_identities:
            raise RuntimeError("source closure result identity was already registered")
        issued_identities[identity] = (result, str(snapshot["digest"]))
        return result

    def make(
        source_closure_result: Vituri2024SpiralNormalClosureResult,
        source_group: Vituri2024SpiralNormalStationaryGroup,
    ) -> _Vituri2024SpiralNormalSourceLineageReceipt:
        source_closure_result, snapshot = require_registered_result(
            source_closure_result
        )
        if type(source_group) is not Vituri2024SpiralNormalStationaryGroup:
            raise TypeError("source_group must be typed")
        matches = tuple(
            index
            for index, candidate in enumerate(source_closure_result.stationary_groups)
            if candidate is source_group
        )
        if len(matches) != 1:
            raise ValueError("source group must be the exact member of its closure result")
        return validate(
            construct(
                **_source_lineage_fields(
                    source_closure_result, matches[0], snapshot
                )
            )
        )

    def validate_complete(
        source_closure_result: Vituri2024SpiralNormalClosureResult,
        receipts: tuple[object, ...],
    ) -> tuple[_Vituri2024SpiralNormalSourceLineageReceipt, ...]:
        source_closure_result, snapshot = require_registered_result(
            source_closure_result
        )
        if type(receipts) is not tuple:
            raise TypeError("source-group lineage inventory must be a tuple")
        expected_indices = tuple(range(len(source_closure_result.stationary_groups)))
        if not expected_indices:
            raise ValueError("source closure has no stationary group to launch")
        validated = tuple(validate(receipt) for receipt in receipts)
        ordered = tuple(sorted(validated, key=lambda item: item.source_group_index))
        if tuple(item.source_group_index for item in ordered) != expected_indices:
            raise ValueError("source-group lineage inventory is incomplete or duplicated")
        for index, receipt in enumerate(ordered):
            expected = _source_lineage_fields(source_closure_result, index, snapshot)
            actual = {
                name: getattr(receipt, name)
                for name in receipt.__dataclass_fields__
                if name != "fingerprint"
            }
            if actual != expected:
                raise ValueError(
                    "source-group lineage belongs to a foreign or tampered closure"
                )
        for receipt in ordered:
            issued = issued_identities[id(receipt)]
            complete_inventory_qualification_by_id[id(receipt)] = (
                issued[1],
                str(snapshot["digest"]),
            )
        return ordered

    exported_runner_name = "run_vituri2024_spiral_normal_exact_shell_closure"
    run_vituri2024_spiral_normal_exact_shell_closure.__module__ = __name__
    run_vituri2024_spiral_normal_exact_shell_closure.__name__ = exported_runner_name
    run_vituri2024_spiral_normal_exact_shell_closure.__qualname__ = exported_runner_name

    return (
        run_vituri2024_spiral_normal_exact_shell_closure,
        make,
        validate,
        validate_complete,
    )







def _run_vituri2024_spiral_normal_exact_shell_closure_implementation(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    policy: Vituri2024SpiralNormalClosurePolicy | None = None,
    dense_boundary_oracle_prepared: Vituri2024PreparedHFSpiral | None = None,
    density_embedding_receipt: object | None = None,
) -> Vituri2024SpiralNormalClosureResult:
    """Exhaust and byte-replay branches conditional on one declared root.

    A supplied root is accepted only as a typed, live-validated embedding
    receipt.  This closure does not establish a global source inventory and
    does not claim that source-group postselection was absent.
    """

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be Vituri2024PreparedHFSpiral")
    active_policy = Vituri2024SpiralNormalClosurePolicy() if policy is None else policy
    active_policy.validate_live_state()
    oracle_pair_fingerprint = (
        None
        if dense_boundary_oracle_prepared is None
        else _dense_boundary_oracle_pair_fingerprint(
            prepared, dense_boundary_oracle_prepared
        )
    )
    initializer = build_vituri2024_spiral_normal_initializer(
        prepared,
        policy=active_policy,
        density_embedding_receipt=density_embedding_receipt,
    )
    root = Vituri2024SpiralNormalBranchPath()
    queue: deque[Vituri2024SpiralNormalBranchPath] = deque((root,))
    queued = {root.fingerprint}
    nodes: list[Vituri2024SpiralNormalBFSNode] = []
    endpoints: list[Vituri2024SpiralNormalEndpoint] = []
    rejections: list[Vituri2024SpiralNormalScientificRejection] = []
    terminal_outcomes: list[NormalPathOutcome] = []
    dense_boundary_oracle_cache: dict[
        str,
        tuple[
            Array,
            Vituri2024SpiralNormalBoundary,
            Vituri2024SpiralNormalDenseBoundaryOracleReceipt,
        ],
    ] = {}
    dense_boundary_oracle_receipts: list[str] = []
    while queue:
        if len(nodes) >= active_policy.maximum_replayed_paths:
            raise RuntimeError("normal closure replay cap reached with unresolved frontier")
        path = queue.popleft()
        outcome = _run_path(
            prepared,
            initializer,
            path,
            policy=active_policy,
            density_embedding_receipt=density_embedding_receipt,
            dense_boundary_oracle_prepared=dense_boundary_oracle_prepared,
            dense_boundary_oracle_cache=dense_boundary_oracle_cache,
            dense_boundary_oracle_receipt_fingerprints=dense_boundary_oracle_receipts,
        )
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
    # initializer. No parent endpoint or intermediate density is reused. Dense
    # oracle maps are recomputed into a separate phase cache, not inherited
    # from the branch-tree pass.
    replay_oracle_cache: dict[
        str,
        tuple[
            Array,
            Vituri2024SpiralNormalBoundary,
            Vituri2024SpiralNormalDenseBoundaryOracleReceipt,
        ],
    ] = {}
    replay_oracle_receipts: list[str] = []
    for original in terminal_outcomes:
        replay = _run_path(
            prepared,
            initializer,
            original.path,
            policy=active_policy,
            density_embedding_receipt=density_embedding_receipt,
            dense_boundary_oracle_prepared=dense_boundary_oracle_prepared,
            dense_boundary_oracle_cache=replay_oracle_cache,
            dense_boundary_oracle_receipt_fingerprints=replay_oracle_receipts,
        )
        if isinstance(replay, Vituri2024SpiralNormalBranchFrontier):
            raise RuntimeError("closed normal terminal replay reopened a frontier")
        if _outcome_digest(replay) != _outcome_digest(original):
            raise RuntimeError("normal terminal deterministic replay mismatch")
    unique_oracle_receipts = tuple(sorted(set(dense_boundary_oracle_receipts)))
    if tuple(sorted(set(replay_oracle_receipts))) != unique_oracle_receipts:
        raise RuntimeError("normal dense boundary oracle deterministic replay mismatch")
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
        dense_boundary_oracle_prepared_fingerprint=(
            None
            if dense_boundary_oracle_prepared is None
            else dense_boundary_oracle_prepared.fingerprint
        ),
        dense_boundary_oracle_pair_fingerprint=oracle_pair_fingerprint,
        dense_boundary_oracle_receipt_fingerprints=unique_oracle_receipts,
    )


(
    run_vituri2024_spiral_normal_exact_shell_closure,
    make_vituri2024_spiral_normal_source_lineage_receipt,
    validate_vituri2024_spiral_normal_source_lineage_receipt,
    validate_complete_vituri2024_spiral_normal_source_group_lineages,
) = _make_source_lineage_authority_api(
    _run_vituri2024_spiral_normal_exact_shell_closure_implementation
)

__all__ = [
    "VITURI2024_SPIRAL_NORMAL_CLOSURE_API_VERSION",
    "VITURI2024_SPIRAL_NORMAL_CLOSURE_AUTHORITY",
    "VITURI2024_SPIRAL_NORMAL_DENSE_BOUNDARY_ORACLE_API_VERSION",
    "VITURI2024_SPIRAL_NORMAL_SOURCE_LINEAGE_API_VERSION",
    "Vituri2024SpiralNormalBoundary",
    "Vituri2024SpiralNormalBranchChoice",
    "Vituri2024SpiralNormalBranchPath",
    "Vituri2024SpiralNormalBranchTrigger",
    "Vituri2024SpiralNormalClosurePolicy",
    "Vituri2024SpiralNormalClosureResult",
    "Vituri2024SpiralNormalDenseBoundaryOracleReceipt",
    "analyze_vituri2024_spiral_normal_boundary",
    "certify_vituri2024_spiral_normal_dense_boundary_oracle",
    "build_vituri2024_spiral_normal_initializer",
    "enumerate_vituri2024_spiral_normal_branch_choices",
    "make_vituri2024_spiral_normal_source_lineage_receipt",
    "run_vituri2024_spiral_normal_exact_shell_closure",
    "validate_complete_vituri2024_spiral_normal_source_group_lineages",
    "validate_vituri2024_spiral_normal_source_lineage_receipt",
]
