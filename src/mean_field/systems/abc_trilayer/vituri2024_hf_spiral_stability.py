"""Thin Vituri adapter for the first restricted spiral-stability lane.

Only selected-spin, k-diagonal, local-rank-preserving orbital rotations are
retained.  The opposite-spin spectator is fixed to the identity.  This is a
candidate action, not a full local-stability test: it omits occupation
transfers between k blocks, does not establish weighted dF reciprocity, and
does not authorize a Hermitian eigensolver.

The generic candidate uses conventional projectors.  Vituri stores
``rho_ab=<c_a^dagger c_b>``, so every selected conventional tangent is embedded
in the full four-flavor conventional layout and transposed through the public
native-density converter before calling the same live functional at the exact
endpoint anchor.  The scalar diagnostic follows the same embedding, with the
spectator block exactly equal to the identity.  All block weights are exactly
one, matching the raw-total, unnormalized k sum returned by the functional.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
from typing import Final

import numpy as np

from mean_field.core.hf import (
    FivePointCurvatureCheck,
    ZeroTemperatureRaggedOrbitalHessian,
    build_zero_temperature_ragged_orbital_hessian,
)

from .vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    vituri2024_conventional_k_diagonal_to_native_density,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from .vituri2024_hf_fft import Vituri2024TranslationalHFFFTFunctional
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_spiral import Vituri2024PreparedHFSpiral

Array = np.ndarray
SelectedEnergyCallback = Callable[[Array], float]

VITURI2024_HF_SPIRAL_STABILITY_API_VERSION: Final[str] = (
    "vituri2024_hf_spiral_restricted_stability.v2"
)
VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY: Final[str] = (
    "local_rank_preserving_k_diagonal_restricted_hessian_candidate_not_full_local_stability"
)
VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION: Final[str] = (
    "raw_total_energy_unweighted_k_sum_all_one_block_weights_no_extra_normalization"
)
VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS: Final[tuple[float, float]] = (
    4.0e-3,
    2.0e-3,
)
VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED: Final[int] = 20240917
VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER: Final[float] = 64.0

_STABILITY_PREPARATION_FACTORY_TOKEN = object()


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


def _checked_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _readonly_complex128(value: object, shape: tuple[int, ...], label: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.complex128)
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must be finite complex128 shape {shape}")
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.complex128).reshape(
        shape
    )
    result.setflags(write=False)
    return result


def _selected_and_spectator_indices(
    prepared: Vituri2024PreparedHFSpiral,
) -> tuple[tuple[int, int], tuple[int, int]]:
    selected = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == prepared.choice.selected_spin
    )
    spectator = tuple(
        index
        for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin == -prepared.choice.selected_spin
    )
    if len(selected) != 2 or len(spectator) != 2:
        raise RuntimeError("Vituri internal flavor order no longer has two spins per lane")
    return selected, spectator  # type: ignore[return-value]


def _canonical_selected_basis(
    occupations: Array, local_ranks: tuple[int, ...]
) -> Array:
    nk = len(local_ranks)
    basis = np.empty((2, 2, nk), dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    for momentum, rank in enumerate(local_ranks):
        basis[:, :, momentum] = (
            swap
            if rank == 1 and occupations[momentum, 0].real == 0.0
            else identity
        )
    return _readonly_complex128(basis, (2, 2, nk), "selected orbital basis")


def _callback_embedding_inventory_fingerprint(
    *,
    selected_indices: tuple[int, int],
    spectator_indices: tuple[int, int],
    nk: int,
    density_native_sha256: str,
    backend_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "api_version": VITURI2024_HF_SPIRAL_STABILITY_API_VERSION,
            "selected_indices": selected_indices,
            "spectator_indices": spectator_indices,
            "nk": nk,
            "anchor_native_sha256": density_native_sha256,
            "backend_fingerprint": backend_fingerprint,
            "selected_layout": "conventional_(2,2,Nk)",
            "full_layout": "conventional_(4,4,Nk)",
            "native_conversion": (
                "vituri2024_conventional_k_diagonal_to_native_density"
            ),
            "spectator_embedding": "exact_identity_each_k",
            "cross_blocks": "exact_zero",
            "index_policy": "immutable_tuples_rebuilt_np_ix_per_call",
        }
    )


def _backend_fingerprint(
    functional: Vituri2024TranslationalHFFunctional
    | Vituri2024TranslationalHFFFTFunctional,
    backend_kind: str,
) -> str:
    payload: dict[str, object] = {
        "backend_kind": backend_kind,
        "backend_type": type(functional).__name__,
        "functional_fingerprint": functional.fingerprint,
        "implementation_fingerprint": functional.implementation_fingerprint,
    }
    if type(functional) is Vituri2024TranslationalHFFFTFunctional:
        payload["fft_plan_fingerprint"] = functional.fft_plan.fingerprint
    return _fingerprint(payload)


@dataclass(frozen=True, slots=True)
class _RestrictedFockDerivativeCallback:
    functional: Vituri2024TranslationalHFFunctional | Vituri2024TranslationalHFFFTFunctional
    anchor_native: Array
    selected_indices: tuple[int, int]
    nk: int
    embedding_inventory_fingerprint: str

    def __call__(self, selected_direction: Array) -> Array:
        direction = _readonly_complex128(
            selected_direction,
            (2, 2, self.nk),
            "selected conventional direction",
        )
        if not np.array_equal(direction, direction.swapaxes(0, 1).conj()):
            raise ValueError("selected conventional direction must be exactly Hermitian")
        selected = np.asarray(self.selected_indices, dtype=np.int64)
        momenta = np.arange(self.nk, dtype=np.int64)
        full_direction = np.zeros((4, 4, self.nk), dtype=np.complex128)
        full_direction[np.ix_(selected, selected, momenta)] = direction
        native_direction = vituri2024_conventional_k_diagonal_to_native_density(
            full_direction
        )
        response = self.functional.fock_derivative(
            self.anchor_native, native_direction
        )
        if (
            type(response) is not np.ndarray
            or response.dtype != np.dtype(np.complex128)
            or response.shape != (4, 4, self.nk)
            or not np.all(np.isfinite(response))
        ):
            raise ValueError("functional dF returned an invalid conventional response")
        return np.array(
            response[np.ix_(selected, selected, momenta)], copy=True
        )


@dataclass(frozen=True, slots=True)
class _ExactUnitaryEnergyCallback:
    functional: Vituri2024TranslationalHFFunctional | Vituri2024TranslationalHFFFTFunctional
    selected_indices: tuple[int, int]
    spectator_indices: tuple[int, int]
    nk: int
    embedding_inventory_fingerprint: str

    def __call__(self, selected_density: Array) -> float:
        selected_clean = _readonly_complex128(
            selected_density,
            (2, 2, self.nk),
            "selected conventional projector",
        )
        selected = np.asarray(self.selected_indices, dtype=np.int64)
        spectators = np.asarray(self.spectator_indices, dtype=np.int64)
        momenta = np.arange(self.nk, dtype=np.int64)
        spectator_identity = np.repeat(
            np.eye(2, dtype=np.complex128)[:, :, None], self.nk, axis=2
        )
        full_projector = np.zeros((4, 4, self.nk), dtype=np.complex128)
        full_projector[np.ix_(selected, selected, momenta)] = selected_clean
        full_projector[np.ix_(spectators, spectators, momenta)] = spectator_identity
        native = vituri2024_conventional_k_diagonal_to_native_density(full_projector)
        return self.functional.energy(native)


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralStabilityCurvatureStep:
    """One diagnostic-only E/F/dF exact-unitary curvature comparison."""

    direction_index: int
    direction_sha256: str
    step: float
    generic_check: FivePointCurvatureCheck
    offset_roundoff_bound_ev: float
    factor_two_wrong_residual_ev: float
    nk_wrong_normalization_residual_ev: float
    diagnostic_only: bool = field(default=True, init=False)
    exact_unitary_e_f_df_composition: bool = field(default=True, init=False)
    clears_offset_roundoff_bound: bool = field(default=True, init=False)
    factor_two_canary_rejected: bool = field(default=True, init=False)
    nk_normalization_canary_rejected: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _checked_sha256(self.direction_sha256, "curvature direction_sha256")
        if type(self.direction_index) is not int or self.direction_index < 0:
            raise ValueError("curvature direction index must be a nonnegative integer")
        if type(self.generic_check) is not FivePointCurvatureCheck:
            raise TypeError("curvature step requires an exact generic five-point check")
        if self.step != self.generic_check.step or self.generic_check.diagnostic_only is not True:
            raise ValueError("curvature step/generic diagnostic binding drifted")
        values = (
            self.offset_roundoff_bound_ev,
            self.factor_two_wrong_residual_ev,
            self.nk_wrong_normalization_residual_ev,
        )
        if any(type(value) is not float or not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("curvature diagnostic residuals must be finite nonnegative floats")
        comparison_floor = (
            self.generic_check.curvature_tolerance + self.offset_roundoff_bound_ev
        )
        locked = (
            self.generic_check.passed is True,
            min(
                abs(self.generic_check.predicted_curvature),
                abs(self.generic_check.finite_difference_curvature),
            )
            > self.offset_roundoff_bound_ev,
            self.factor_two_wrong_residual_ev > comparison_floor,
            self.nk_wrong_normalization_residual_ev > comparison_floor,
            self.diagnostic_only is True,
            self.exact_unitary_e_f_df_composition is True,
            self.clears_offset_roundoff_bound is True,
            self.factor_two_canary_rejected is True,
            self.nk_normalization_canary_rejected is True,
        )
        if not all(locked):
            raise ValueError("curvature diagnostic or normalization canary did not clear")


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralStabilityCurvatureDiagnostic:
    """Deterministic diagnostic suite; never scalar-Hessian authority."""

    seed: int
    direction_count: int
    steps: tuple[float, ...]
    evidence: tuple[Vituri2024HFSpiralStabilityCurvatureStep, ...]
    diagnostic_only: bool = field(default=True, init=False)
    scalar_hessian_authority_established: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or type(self.direction_count) is not int:
            raise TypeError("curvature seed and direction_count must be exact integers")
        if self.direction_count < 1 or len(self.steps) < 2:
            raise ValueError("curvature diagnostic requires directions and at least two steps")
        if (
            len(self.evidence) != self.direction_count * len(self.steps)
            or any(type(item) is not Vituri2024HFSpiralStabilityCurvatureStep for item in self.evidence)
            or self.diagnostic_only is not True
            or self.scalar_hessian_authority_established is not False
        ):
            raise ValueError("curvature diagnostic inventory or authority drifted")
        expected = tuple(
            (direction_index, step)
            for direction_index in range(self.direction_count)
            for step in self.steps
        )
        actual = tuple((item.direction_index, item.step) for item in self.evidence)
        if actual != expected:
            raise ValueError("curvature direction/step ordering drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralStabilityReceipt:
    """Hash-bound scope receipt for one restricted candidate preparation."""

    density_native_sha256: str
    fresh_hamiltonian_conventional_sha256: str
    prepared_fingerprint: str
    functional_fingerprint: str
    choice_fingerprint: str
    gauge_receipt_fingerprint: str | None
    interaction_fingerprint: str
    normal_order_reference_fingerprint: str
    normal_order_reference_native_sha256: str
    backend_kind: str
    backend_type: str
    backend_fingerprint: str
    callback_embedding_inventory_fingerprint: str
    selected_flavor_indices: tuple[int, int]
    spectator_flavor_indices: tuple[int, int]
    selected_global_rank: int
    local_ranks: tuple[int, ...]
    local_rank_inventory_sha256: str
    local_rank_counts_0_1_2: tuple[int, int, int]
    raw_total_energy_ev: float
    normalization: str = field(
        default=VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION, init=False
    )
    authority: str = field(
        default=VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    local_rank_preserving_only: bool = field(default=True, init=False)
    k_diagonal_only: bool = field(default=True, init=False)
    spectator_frozen_to_identity: bool = field(default=True, init=False)
    exact_unitary_scalar_diagnostic_only: bool = field(default=True, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    occupation_transfer_stability_established: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name in (
            "density_native_sha256",
            "fresh_hamiltonian_conventional_sha256",
            "prepared_fingerprint",
            "functional_fingerprint",
            "choice_fingerprint",
            "interaction_fingerprint",
            "normal_order_reference_fingerprint",
            "normal_order_reference_native_sha256",
            "backend_fingerprint",
            "callback_embedding_inventory_fingerprint",
            "local_rank_inventory_sha256",
        ):
            _checked_sha256(getattr(self, name), name)
        if self.gauge_receipt_fingerprint is not None:
            _checked_sha256(self.gauge_receipt_fingerprint, "gauge_receipt_fingerprint")
        if self.backend_kind not in ("dense", "fft"):
            raise ValueError("backend_kind must be dense or fft")
        expected_type = (
            "Vituri2024TranslationalHFFunctional"
            if self.backend_kind == "dense"
            else "Vituri2024TranslationalHFFFTFunctional"
        )
        if self.backend_type != expected_type:
            raise ValueError("backend type/kind binding drifted")
        if (
            any(type(index) is not int for index in self.selected_flavor_indices)
            or any(type(index) is not int for index in self.spectator_flavor_indices)
        ):
            raise TypeError("selected and spectator indices must be exact integers")
        if set(self.selected_flavor_indices) & set(self.spectator_flavor_indices):
            raise ValueError("selected and spectator flavor inventories overlap")
        if tuple(sorted(self.selected_flavor_indices + self.spectator_flavor_indices)) != tuple(
            range(len(INTERNAL_FLAVOR_ORDER))
        ):
            raise ValueError("selected and spectator flavor inventories are incomplete")
        if any(type(rank) is not int or rank not in (0, 1, 2) for rank in self.local_ranks):
            raise ValueError("local ranks must be exact integers in 0/1/2")
        expected_counts = tuple(self.local_ranks.count(rank) for rank in (0, 1, 2))
        expected_rank_digest = _array_sha256(
            np.asarray(self.local_ranks, dtype=np.int64)
        )
        if (
            self.local_rank_counts_0_1_2 != expected_counts
            or self.local_rank_inventory_sha256 != expected_rank_digest
        ):
            raise ValueError("local-rank inventory/counts drifted")
        if self.selected_global_rank != sum(self.local_ranks):
            raise ValueError("selected global/local rank binding drifted")
        if not np.isfinite(self.raw_total_energy_ev):
            raise ValueError("raw total energy must be finite")
        locked = (
            self.normalization == VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION,
            self.authority == VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY,
            self.candidate_only is True,
            self.local_rank_preserving_only is True,
            self.k_diagonal_only is True,
            self.spectator_frozen_to_identity is True,
            self.exact_unitary_scalar_diagnostic_only is True,
            self.reciprocity_established is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.occupation_transfer_stability_established is False,
        )
        if not all(locked):
            raise ValueError("restricted stability authority was inflated")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_HF_SPIRAL_STABILITY_API_VERSION,
                "density": self.density_native_sha256,
                "fock": self.fresh_hamiltonian_conventional_sha256,
                "prepared": self.prepared_fingerprint,
                "functional": self.functional_fingerprint,
                "choice": self.choice_fingerprint,
                "gauge": self.gauge_receipt_fingerprint,
                "interaction": self.interaction_fingerprint,
                "normal_reference": self.normal_order_reference_fingerprint,
                "normal_reference_array": self.normal_order_reference_native_sha256,
                "backend": self.backend_fingerprint,
                "callback_embedding_inventory": (
                    self.callback_embedding_inventory_fingerprint
                ),
                "selected": self.selected_flavor_indices,
                "spectator": self.spectator_flavor_indices,
                "global_rank": self.selected_global_rank,
                "local_rank_inventory": self.local_rank_inventory_sha256,
                "local_rank_counts": self.local_rank_counts_0_1_2,
                "raw_total_energy_ev": self.raw_total_energy_ev.hex(),
                "normalization": self.normalization,
                "authority": self.authority,
                "flags": (
                    self.candidate_only,
                    self.local_rank_preserving_only,
                    self.k_diagonal_only,
                    self.spectator_frozen_to_identity,
                    self.exact_unitary_scalar_diagnostic_only,
                    self.reciprocity_established,
                    self.hermitian_eigensolver_authorized,
                    self.full_local_stability_established,
                    self.occupation_transfer_stability_established,
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralStabilityPreparation:
    """Factory-only selected-spin candidate bound to one live endpoint."""

    _factory_token: InitVar[object]
    prepared: Vituri2024PreparedHFSpiral
    density_native: Array
    fresh_hamiltonian_conventional: Array
    selected_projectors_conventional: Array
    selected_hamiltonians_conventional: Array
    selected_orbital_basis_conventional: Array
    hessian: ZeroTemperatureRaggedOrbitalHessian
    fock_derivative_callback: Callable[[Array], Array] = field(
        repr=False, compare=False
    )
    exact_unitary_energy_callback: SelectedEnergyCallback = field(
        repr=False, compare=False
    )
    receipt: Vituri2024HFSpiralStabilityReceipt
    authority: str = field(
        default=VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _STABILITY_PREPARATION_FACTORY_TOKEN:
            raise TypeError("spiral stability preparation is factory-only")
        self.validate_live_state()

    @property
    def energy_callback(self) -> SelectedEnergyCallback:
        """Compatibility spelling for the exact-unitary diagnostic callback."""

        return self.exact_unitary_energy_callback

    @property
    def local_ranks(self) -> tuple[int, ...]:
        return self.receipt.local_ranks

    def diagnose_exact_unitary_curvature(
        self,
        *,
        seed: int = VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED,
        direction_count: int = 2,
        steps: tuple[float, ...] = VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS,
        curvature_atol: float = 2.0e-8,
        curvature_rtol: float = 2.0e-7,
        stationarity_atol: float = 2.0e-8,
    ) -> Vituri2024HFSpiralStabilityCurvatureDiagnostic:
        """Compose exact-unitary E with the generic F/dF Hessian diagnostic.

        This finite inventory is diagnostic only.  Each complex tangent is
        represented by deterministic real coordinates and normalized before
        evaluation.  The five-point energy-offset roundoff bound and explicit
        missing-factor-two and erroneous-``/Nk`` canaries must all clear.
        """

        self.validate_live_state()
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("curvature seed must be an integer")
        if (
            isinstance(direction_count, (bool, np.bool_))
            or not isinstance(direction_count, (int, np.integer))
            or direction_count < 1
        ):
            raise ValueError("direction_count must be a positive integer")
        clean_steps = tuple(float(step) for step in steps)
        if len(clean_steps) < 2 or any(
            not np.isfinite(step) or step <= 0.0 for step in clean_steps
        ):
            raise ValueError("at least two finite positive curvature steps are required")
        if len(set(clean_steps)) != len(clean_steps):
            raise ValueError("curvature steps must be distinct")
        if self.hessian.real_dimension == 0:
            raise ValueError("zero-dimensional candidate has no curvature directions")

        rng = np.random.default_rng(int(seed))
        evidence: list[Vituri2024HFSpiralStabilityCurvatureStep] = []
        eps = np.finfo(np.float64).eps
        for direction_index in range(int(direction_count)):
            direction = rng.standard_normal(self.hessian.real_dimension)
            direction /= np.linalg.norm(direction)
            direction_sha256 = _array_sha256(direction)
            for step in clean_steps:
                check = self.hessian.check_five_point_curvature(
                    direction,
                    self.exact_unitary_energy_callback,
                    step=step,
                    curvature_atol=curvature_atol,
                    curvature_rtol=curvature_rtol,
                    stationarity_atol=stationarity_atol,
                )
                em2, em1, e0, ep1, ep2 = (
                    check.energies_minus_2h_to_plus_2h
                )
                weighted_offset = (
                    abs(em2)
                    + 16.0 * abs(em1)
                    + 30.0 * abs(e0)
                    + 16.0 * abs(ep1)
                    + abs(ep2)
                )
                offset_bound = float(
                    VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER
                    * eps
                    * weighted_offset
                    / (12.0 * step * step)
                )
                predicted = check.predicted_curvature
                finite_difference = check.finite_difference_curvature
                evidence.append(
                    Vituri2024HFSpiralStabilityCurvatureStep(
                        direction_index=direction_index,
                        direction_sha256=direction_sha256,
                        step=step,
                        generic_check=check,
                        offset_roundoff_bound_ev=offset_bound,
                        factor_two_wrong_residual_ev=float(
                            abs(finite_difference - 0.5 * predicted)
                        ),
                        nk_wrong_normalization_residual_ev=float(
                            abs(finite_difference - predicted / self.prepared.nk)
                        ),
                    )
                )
        return Vituri2024HFSpiralStabilityCurvatureDiagnostic(
            seed=int(seed),
            direction_count=int(direction_count),
            steps=clean_steps,
            evidence=tuple(evidence),
        )

    def validate_live_state(self) -> None:
        if type(self.prepared) is not Vituri2024PreparedHFSpiral:
            raise TypeError("stability preparation requires exact Vituri spiral preparation")
        self.prepared.validate_live_state()
        if type(self.receipt) is not Vituri2024HFSpiralStabilityReceipt:
            raise TypeError("stability receipt type drifted")
        functional = self.prepared.functional
        if type(functional) not in (
            Vituri2024TranslationalHFFunctional,
            Vituri2024TranslationalHFFFTFunctional,
        ):
            raise TypeError("restricted stability functional type drifted")
        expected_backend = _backend_fingerprint(functional, self.prepared.backend_kind)
        expected_gauge = (
            None
            if self.prepared.gauge_receipt is None
            else self.prepared.gauge_receipt.fingerprint
        )
        selected_indices, spectator_indices = _selected_and_spectator_indices(
            self.prepared
        )
        expected_inventory = _callback_embedding_inventory_fingerprint(
            selected_indices=selected_indices,
            spectator_indices=spectator_indices,
            nk=self.prepared.nk,
            density_native_sha256=self.receipt.density_native_sha256,
            backend_fingerprint=expected_backend,
        )
        if (
            self.receipt.prepared_fingerprint != self.prepared.fingerprint
            or self.receipt.functional_fingerprint != functional.fingerprint
            or self.receipt.choice_fingerprint != self.prepared.choice.fingerprint
            or self.receipt.gauge_receipt_fingerprint != expected_gauge
            or self.receipt.interaction_fingerprint != functional.interaction_fingerprint
            or self.receipt.normal_order_reference_fingerprint
            != functional.normal_order_reference_fingerprint
            or self.receipt.normal_order_reference_native_sha256
            != _array_sha256(functional.normal_order_reference_native)
            or self.receipt.backend_kind != self.prepared.backend_kind
            or self.receipt.backend_type != type(functional).__name__
            or self.receipt.backend_fingerprint != expected_backend
            or self.receipt.callback_embedding_inventory_fingerprint
            != expected_inventory
            or self.receipt.selected_flavor_indices != selected_indices
            or self.receipt.spectator_flavor_indices != spectator_indices
        ):
            raise ValueError("restricted stability prepared/functional binding drifted")

        nk = self.prepared.nk
        shape = (len(INTERNAL_FLAVOR_ORDER),) * 2 + (nk,)
        for array, expected_shape, label in (
            (self.density_native, shape, "density_native"),
            (
                self.fresh_hamiltonian_conventional,
                shape,
                "fresh_hamiltonian_conventional",
            ),
            (
                self.selected_projectors_conventional,
                (2, 2, nk),
                "selected_projectors_conventional",
            ),
            (
                self.selected_hamiltonians_conventional,
                (2, 2, nk),
                "selected_hamiltonians_conventional",
            ),
            (
                self.selected_orbital_basis_conventional,
                (2, 2, nk),
                "selected_orbital_basis_conventional",
            ),
        ):
            if (
                type(array) is not np.ndarray
                or array.dtype != np.dtype(np.complex128)
                or array.shape != expected_shape
                or array.flags.writeable
                or not array.flags.c_contiguous
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"restricted stability {label} live array drifted")
        if (
            _array_sha256(self.density_native) != self.receipt.density_native_sha256
            or _array_sha256(self.fresh_hamiltonian_conventional)
            != self.receipt.fresh_hamiltonian_conventional_sha256
        ):
            raise ValueError("restricted stability endpoint arrays drifted")

        fresh = functional.fock(self.density_native)
        if (
            type(fresh) is not np.ndarray
            or fresh.dtype != np.dtype(np.complex128)
            or fresh.shape != shape
            or not np.all(np.isfinite(fresh))
            or not np.array_equal(fresh, self.fresh_hamiltonian_conventional)
            or _array_sha256(fresh)
            != self.receipt.fresh_hamiltonian_conventional_sha256
        ):
            raise ValueError("restricted stability exact fresh Fock/hash drifted")

        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            self.density_native
        )
        if not np.array_equal(
            vituri2024_conventional_k_diagonal_to_native_density(conventional),
            self.density_native,
        ):
            raise ValueError("restricted stability density orientation drifted")
        selected = np.asarray(selected_indices, dtype=np.int64)
        spectators = np.asarray(spectator_indices, dtype=np.int64)
        momenta = np.arange(nk, dtype=np.int64)
        selected_projectors = conventional[np.ix_(selected, selected, momenta)]
        spectator_projectors = conventional[np.ix_(spectators, spectators, momenta)]
        selected_to_spectator = conventional[np.ix_(selected, spectators, momenta)]
        spectator_to_selected = conventional[np.ix_(spectators, selected, momenta)]
        spectator_identity = np.repeat(
            np.eye(2, dtype=np.complex128)[:, :, None], nk, axis=2
        )
        if (
            not np.array_equal(spectator_projectors, spectator_identity)
            or np.count_nonzero(selected_to_spectator)
            or np.count_nonzero(spectator_to_selected)
            or not np.array_equal(
                selected_projectors, self.selected_projectors_conventional
            )
        ):
            raise ValueError("restricted stability selected/spectator slices drifted")
        if np.count_nonzero(selected_projectors[0, 1, :]) or np.count_nonzero(
            selected_projectors[1, 0, :]
        ):
            raise ValueError("restricted stability selected projector is not normal")
        occupations = np.stack(
            (selected_projectors[0, 0, :], selected_projectors[1, 1, :]), axis=1
        )
        if np.count_nonzero(occupations.imag) or not np.all(
            (occupations.real == 0.0) | (occupations.real == 1.0)
        ):
            raise ValueError("restricted stability selected occupations drifted")
        local_ranks = tuple(int(value) for value in np.sum(occupations.real, axis=1))
        local_rank_array = np.asarray(local_ranks, dtype=np.int64)
        rank_counts = tuple(local_ranks.count(rank) for rank in (0, 1, 2))
        canonical_basis = _canonical_selected_basis(occupations, local_ranks)
        selected_hamiltonians = fresh[np.ix_(selected, selected, momenta)]
        if (
            local_ranks != self.receipt.local_ranks
            or _array_sha256(local_rank_array)
            != self.receipt.local_rank_inventory_sha256
            or rank_counts != self.receipt.local_rank_counts_0_1_2
            or sum(local_ranks) != self.receipt.selected_global_rank
            or self.receipt.selected_global_rank != self.prepared.selected_rank
            or not np.array_equal(
                canonical_basis, self.selected_orbital_basis_conventional
            )
            or not np.array_equal(
                selected_hamiltonians, self.selected_hamiltonians_conventional
            )
        ):
            raise ValueError("restricted stability rank/basis/Fock slices drifted")

        if (
            type(self.fock_derivative_callback) is not _RestrictedFockDerivativeCallback
            or type(self.exact_unitary_energy_callback) is not _ExactUnitaryEnergyCallback
        ):
            raise TypeError("restricted stability callback type drifted")
        derivative = self.fock_derivative_callback
        energy = self.exact_unitary_energy_callback
        if (
            derivative.functional is not functional
            or energy.functional is not functional
            or derivative.selected_indices != selected_indices
            or energy.selected_indices != selected_indices
            or energy.spectator_indices != spectator_indices
            or derivative.nk != nk
            or energy.nk != nk
            or derivative.embedding_inventory_fingerprint != expected_inventory
            or energy.embedding_inventory_fingerprint != expected_inventory
            or derivative.anchor_native.flags.writeable
            or not np.array_equal(derivative.anchor_native, self.density_native)
            or _array_sha256(derivative.anchor_native)
            != self.receipt.density_native_sha256
        ):
            raise ValueError("restricted stability callback embedding drifted")

        expected_complex_dimension = sum(rank * (2 - rank) for rank in local_ranks)
        if (
            type(self.hessian) is not ZeroTemperatureRaggedOrbitalHessian
            or not np.array_equal(
                self.hessian.projectors, self.selected_projectors_conventional
            )
            or not np.array_equal(
                self.hessian.hamiltonians, self.selected_hamiltonians_conventional
            )
            or not np.array_equal(
                self.hessian.orbital_basis,
                self.selected_orbital_basis_conventional,
            )
            or self.hessian.occupied_counts != local_ranks
            or self.hessian.fock_derivative is not derivative
            or not np.array_equal(self.hessian.block_weights, np.ones(nk))
            or self.hessian.n != 2
            or self.hessian.nblock != nk
            or self.hessian.complex_dimension != expected_complex_dimension
            or self.hessian.real_dimension != 2 * expected_complex_dimension
            or tuple(layout.block for layout in self.hessian.layouts)
            != tuple(range(nk))
            or tuple(layout.occupied_count for layout in self.hessian.layouts)
            != local_ranks
        ):
            raise ValueError("restricted stability generic-Hessian relation drifted")

        raw_energy = functional.energy(self.density_native)
        if (
            type(raw_energy) is not float
            or raw_energy != self.receipt.raw_total_energy_ev
            or energy(self.selected_projectors_conventional) != raw_energy
        ):
            raise ValueError("restricted stability exact endpoint energy drifted")
        locked = (
            self.authority == VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY,
            self.candidate_only is True,
            self.reciprocity_established is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
        )
        if not all(locked):
            raise ValueError("restricted stability preparation authority was inflated")


def prepare_vituri2024_hf_spiral_stability(
    prepared: Vituri2024PreparedHFSpiral,
    density_native: Array,
    fresh_hamiltonian_conventional: Array,
    *,
    expected_density_native_sha256: str | None = None,
    expected_fresh_hamiltonian_conventional_sha256: str | None = None,
) -> Vituri2024HFSpiralStabilityPreparation:
    """Build the first restricted, normal, k-diagonal stability candidate.

    Optional hashes bind caller-owned endpoint arrays before any copy is made.
    The supplied Hamiltonian must additionally be exactly equal to a fresh
    ``functional.fock(density_native)`` evaluation; hash agreement alone is
    never accepted as a substitute for array equality.
    """

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be exact Vituri2024PreparedHFSpiral")
    prepared.validate_live_state()
    functional = prepared.functional
    if type(functional) not in (
        Vituri2024TranslationalHFFunctional,
        Vituri2024TranslationalHFFFTFunctional,
    ):
        raise TypeError("restricted stability requires an exact dense or FFT backend")

    nk = prepared.nk
    density = _readonly_complex128(
        density_native, (len(INTERNAL_FLAVOR_ORDER),) * 2 + (nk,), "density_native"
    )
    supplied_fock = _readonly_complex128(
        fresh_hamiltonian_conventional,
        (len(INTERNAL_FLAVOR_ORDER),) * 2 + (nk,),
        "fresh_hamiltonian_conventional",
    )
    density_hash = _array_sha256(density)
    supplied_fock_hash = _array_sha256(supplied_fock)
    if expected_density_native_sha256 is not None and (
        _checked_sha256(
            expected_density_native_sha256, "expected_density_native_sha256"
        )
        != density_hash
    ):
        raise ValueError("density_native supplied hash mismatch")
    if expected_fresh_hamiltonian_conventional_sha256 is not None and (
        _checked_sha256(
            expected_fresh_hamiltonian_conventional_sha256,
            "expected_fresh_hamiltonian_conventional_sha256",
        )
        != supplied_fock_hash
    ):
        raise ValueError("fresh Hamiltonian supplied hash mismatch")

    fresh = functional.fock(density)
    if (
        type(fresh) is not np.ndarray
        or fresh.dtype != np.dtype(np.complex128)
        or fresh.shape != supplied_fock.shape
        or not np.all(np.isfinite(fresh))
    ):
        raise ValueError("functional.fock returned an invalid conventional Hamiltonian")
    if not np.array_equal(supplied_fock, fresh):
        raise ValueError(
            "fresh_hamiltonian_conventional is stale: it must exactly equal "
            "functional.fock(density_native)"
        )
    if _array_sha256(fresh) != supplied_fock_hash:
        raise RuntimeError("exact fresh-Fock equality did not preserve its hash binding")

    conventional = vituri2024_native_density_to_conventional_k_diagonal(density)
    if not np.array_equal(
        vituri2024_conventional_k_diagonal_to_native_density(conventional), density
    ):
        raise ValueError("native/conventional endpoint density orientation did not roundtrip")
    selected_indices, spectator_indices = _selected_and_spectator_indices(prepared)
    selected = np.asarray(selected_indices, dtype=np.int64)
    spectators = np.asarray(spectator_indices, dtype=np.int64)
    momenta = np.arange(nk, dtype=np.int64)
    selected_projectors = conventional[np.ix_(selected, selected, momenta)]
    spectator_projectors = conventional[np.ix_(spectators, spectators, momenta)]
    selected_to_spectator = conventional[np.ix_(selected, spectators, momenta)]
    spectator_to_selected = conventional[np.ix_(spectators, selected, momenta)]
    spectator_identity = np.repeat(
        np.eye(2, dtype=np.complex128)[:, :, None], nk, axis=2
    )
    if not np.array_equal(spectator_projectors, spectator_identity):
        raise ValueError("spectator density block must be exactly full at every k")
    if np.count_nonzero(selected_to_spectator) or np.count_nonzero(
        spectator_to_selected
    ):
        raise ValueError("selected-spectator density blocks must be exactly zero")
    if np.count_nonzero(selected_projectors[0, 1, :]) or np.count_nonzero(
        selected_projectors[1, 0, :]
    ):
        raise ValueError("selected normal projector must be coordinate diagonal")
    occupations = np.stack(
        (selected_projectors[0, 0, :], selected_projectors[1, 1, :]), axis=1
    )
    if np.count_nonzero(occupations.imag) or not np.all(
        (occupations.real == 0.0) | (occupations.real == 1.0)
    ):
        raise ValueError("selected normal occupations must be exact 0/1 values")
    local_ranks = tuple(int(value) for value in np.sum(occupations.real, axis=1))
    if sum(local_ranks) != prepared.selected_rank:
        raise ValueError("selected global rank does not equal prepared.selected_rank")

    basis = _canonical_selected_basis(occupations, local_ranks)
    selected_projectors = _readonly_complex128(
        np.asarray(selected_projectors, dtype=np.complex128),
        (2, 2, nk),
        "selected projectors",
    )
    selected_hamiltonians = _readonly_complex128(
        np.asarray(
            supplied_fock[np.ix_(selected, selected, momenta)],
            dtype=np.complex128,
        ),
        (2, 2, nk),
        "selected Hamiltonians",
    )
    backend_fingerprint = _backend_fingerprint(functional, prepared.backend_kind)
    embedding_inventory = _callback_embedding_inventory_fingerprint(
        selected_indices=selected_indices,
        spectator_indices=spectator_indices,
        nk=nk,
        density_native_sha256=density_hash,
        backend_fingerprint=backend_fingerprint,
    )
    restricted_fock_derivative = _RestrictedFockDerivativeCallback(
        functional=functional,
        anchor_native=density,
        selected_indices=selected_indices,
        nk=nk,
        embedding_inventory_fingerprint=embedding_inventory,
    )
    exact_unitary_energy = _ExactUnitaryEnergyCallback(
        functional=functional,
        selected_indices=selected_indices,
        spectator_indices=spectator_indices,
        nk=nk,
        embedding_inventory_fingerprint=embedding_inventory,
    )

    hessian = build_zero_temperature_ragged_orbital_hessian(
        selected_projectors,
        selected_hamiltonians,
        basis,
        local_ranks,
        restricted_fock_derivative,
        block_weights=np.ones(nk, dtype=np.float64),
    )
    raw_total_energy = functional.energy(density)
    if exact_unitary_energy(selected_projectors) != raw_total_energy:
        raise RuntimeError("restricted scalar embedding changed the exact anchor energy")

    local_rank_array = np.asarray(local_ranks, dtype=np.int64)
    receipt = Vituri2024HFSpiralStabilityReceipt(
        density_native_sha256=density_hash,
        fresh_hamiltonian_conventional_sha256=supplied_fock_hash,
        prepared_fingerprint=prepared.fingerprint,
        functional_fingerprint=functional.fingerprint,
        choice_fingerprint=prepared.choice.fingerprint,
        gauge_receipt_fingerprint=(
            None if prepared.gauge_receipt is None else prepared.gauge_receipt.fingerprint
        ),
        interaction_fingerprint=functional.interaction_fingerprint,
        normal_order_reference_fingerprint=functional.normal_order_reference_fingerprint,
        normal_order_reference_native_sha256=_array_sha256(
            functional.normal_order_reference_native
        ),
        backend_kind=prepared.backend_kind,
        backend_type=type(functional).__name__,
        backend_fingerprint=backend_fingerprint,
        callback_embedding_inventory_fingerprint=embedding_inventory,
        selected_flavor_indices=selected_indices,
        spectator_flavor_indices=spectator_indices,
        selected_global_rank=prepared.selected_rank,
        local_ranks=local_ranks,
        local_rank_inventory_sha256=_array_sha256(local_rank_array),
        local_rank_counts_0_1_2=tuple(local_ranks.count(rank) for rank in (0, 1, 2)),
        raw_total_energy_ev=float(raw_total_energy),
    )
    result = Vituri2024HFSpiralStabilityPreparation(
        _factory_token=_STABILITY_PREPARATION_FACTORY_TOKEN,
        prepared=prepared,
        density_native=density,
        fresh_hamiltonian_conventional=supplied_fock,
        selected_projectors_conventional=selected_projectors,
        selected_hamiltonians_conventional=selected_hamiltonians,
        selected_orbital_basis_conventional=basis,
        hessian=hessian,
        fock_derivative_callback=restricted_fock_derivative,
        exact_unitary_energy_callback=exact_unitary_energy,
        receipt=receipt,
    )
    return result


# Explicit spelling for callers that want the restricted scope in the name.
prepare_vituri2024_hf_spiral_restricted_stability = (
    prepare_vituri2024_hf_spiral_stability
)
Vituri2024PreparedHFSpiralStability = Vituri2024HFSpiralStabilityPreparation
Vituri2024HFSpiralCurvatureDiagnostic = (
    Vituri2024HFSpiralStabilityCurvatureDiagnostic
)
Vituri2024HFSpiralCurvatureStep = Vituri2024HFSpiralStabilityCurvatureStep


__all__ = [
    "VITURI2024_HF_SPIRAL_STABILITY_API_VERSION",
    "VITURI2024_HF_SPIRAL_STABILITY_AUTHORITY",
    "VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_SEED",
    "VITURI2024_HF_SPIRAL_STABILITY_CURVATURE_STEPS",
    "VITURI2024_HF_SPIRAL_STABILITY_NORMALIZATION",
    "VITURI2024_HF_SPIRAL_STABILITY_OFFSET_ROUNDOFF_MULTIPLIER",
    "Vituri2024HFSpiralCurvatureDiagnostic",
    "Vituri2024HFSpiralCurvatureStep",
    "Vituri2024HFSpiralStabilityCurvatureDiagnostic",
    "Vituri2024HFSpiralStabilityCurvatureStep",
    "Vituri2024HFSpiralStabilityPreparation",
    "Vituri2024HFSpiralStabilityReceipt",
    "Vituri2024PreparedHFSpiralStability",
    "prepare_vituri2024_hf_spiral_restricted_stability",
    "prepare_vituri2024_hf_spiral_stability",
]
