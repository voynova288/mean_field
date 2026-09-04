"""Paired no-wrap selected-spin orbital Hessian for a normal spiral source.

This adapter maps source-bound Vituri momentum/valley sectors onto the generic
``PairedSectorOrbitalHessian``.  For one conjugate orbit
``(d,chi) <-> (-d,-chi)``, independent particle-hole coordinates ``x`` and
``y`` form the signed Hermitian density block

``W_d = Z_d[x] + Z_-d[y]^dagger``.

The two signed responses are evaluated independently; no q/-q averaging,
padding, symmetry postselection, or dimension equality is assumed.  The
spectator spin remains frozen, and only the selected-output projection enters
the constrained Hessian.

This remains candidate algebra.  It does not establish scalar-functional
curvature, full reciprocity, Hermitian-eigensolver authority, positivity, or
paper reproduction.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
from typing import Final

import numpy as np

from ...core.hf.zero_temperature_sector_stability import (
    OrbitalTransitionLane,
    PairedOrbitalTransitionFrame,
    PairedSectorOrbitalHessian,
)
from .vituri2024_hf_spiral_full_response import (
    Vituri2024HFSpiralSignedDisplacementResponse,
    Vituri2024HFSpiralValidatedResponseActionFactory,
    Vituri2024HFSpiralValidatedSignedDisplacementFFTAction,
)
from .vituri2024_hf_spiral_full_stability import (
    Vituri2024HFSpiralFullSectorInventory,
    Vituri2024HFSpiralFullSectorKey,
    Vituri2024HFSpiralFullSectorOrbit,
)

Array = np.ndarray

VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION: Final[str] = (
    "vituri2024_hf_spiral_full_selected_spin_paired_sector_hessian.v1"
)
VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY: Final[str] = (
    "candidate_paired_no_wrap_selected_spin_global_rank_orbital_hessian_"
    "action_only_not_reciprocity_eigensolver_stability_or_paper_authority"
)

_CONTEXT_TOKEN = object()


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


def _readonly(value: object, dtype: np.dtype, shape: tuple[int, ...], label: str) -> Array:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite exact {dtype} {shape}")
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _allowed_pairs(charge: int) -> tuple[tuple[int, int], ...]:
    if charge == 0:
        return ((0, 0), (1, 1))
    if charge == 2:
        return ((1, 0),)
    if charge == -2:
        return ((0, 1),)
    raise ValueError("unsupported selected-spin valley charge")


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralTransitionLaneEmbedding:
    """Exact source-orbital embedding for one signed particle-hole lane."""

    _factory_token: InitVar[object]
    key: Vituri2024HFSpiralFullSectorKey
    particle_k_indices: Array
    hole_k_indices: Array
    particle_valley_slots: Array
    hole_valley_slots: Array
    particle_orbital_ids: Array
    hole_orbital_ids: Array
    embedding_fingerprint: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CONTEXT_TOKEN:
            raise TypeError("transition-lane embedding is factory-only")
        if type(self.key) is not Vituri2024HFSpiralFullSectorKey:
            raise TypeError("transition-lane key type drifted")
        arrays = tuple(np.asarray(value) for value in (
            self.particle_k_indices,
            self.hole_k_indices,
            self.particle_valley_slots,
            self.hole_valley_slots,
            self.particle_orbital_ids,
            self.hole_orbital_ids,
        ))
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("transition-lane embedding arrays must be one-dimensional")
        size = int(arrays[0].size)
        if any(array.size != size for array in arrays):
            raise ValueError("transition-lane embedding arrays have unequal sizes")
        labels = (
            "particle k indices",
            "hole k indices",
            "particle valley slots",
            "hole valley slots",
            "particle orbital ids",
            "hole orbital ids",
        )
        for name, value in zip(labels, arrays, strict=True):
            object.__setattr__(
                self,
                name.replace(" ", "_"),
                _readonly(value, np.dtype(np.int64), (size,), name),
            )
        object.__setattr__(
            self,
            "embedding_fingerprint",
            _fingerprint(
                {
                    "key": (
                        self.key.displacement_x,
                        self.key.displacement_y,
                        self.key.valley_charge,
                    ),
                    "arrays": tuple(_array_sha256(value) for value in arrays),
                }
            ),
        )

    @property
    def complex_dimension(self) -> int:
        return int(self.particle_k_indices.size)


@dataclass(frozen=True, slots=True)
class _VituriPairedInteractionCallback:
    response: Vituri2024HFSpiralSignedDisplacementResponse
    first_embedding: Vituri2024HFSpiralTransitionLaneEmbedding
    second_embedding: Vituri2024HFSpiralTransitionLaneEmbedding
    first_action: Vituri2024HFSpiralValidatedSignedDisplacementFFTAction
    second_action: Vituri2024HFSpiralValidatedSignedDisplacementFFTAction
    expected_response_fingerprint: str

    def __call__(self, first: Array, second: Array) -> tuple[Array, Array]:
        if self.response.response_fingerprint != self.expected_response_fingerprint:
            raise ValueError("paired Hessian response became stale")
        first_values = np.asarray(first, dtype=np.complex128)
        second_values = np.asarray(second, dtype=np.complex128)
        if first_values.shape != (self.first_embedding.complex_dimension,) or second_values.shape != (
            self.second_embedding.complex_dimension,
        ):
            raise ValueError("paired Hessian callback coordinate shape drifted")
        block = np.zeros((2, 2, self.response.nk), dtype=np.complex128)
        a = self.first_embedding.particle_valley_slots
        b = self.first_embedding.hole_valley_slots
        k = self.first_embedding.hole_k_indices
        block[a, b, k] = first_values

        # A (-d,-chi) transition y has row=particle k and column=hole k.
        # Its adjoint contributes to W_d at base=particle k with reversed
        # flavor order.  Occupation complementarity makes these entries
        # disjoint from the first-lane assignments.
        particle = self.second_embedding.particle_valley_slots
        hole = self.second_embedding.hole_valley_slots
        base = self.second_embedding.particle_k_indices
        if np.any(block[hole, particle, base] != 0.0):
            raise RuntimeError("paired tangent embeddings unexpectedly overlap")
        block[hole, particle, base] = second_values.conj()

        if self.first_embedding.complex_dimension:
            first_response_block = self.first_action(block)
            first_output = np.asarray(
                first_response_block[a, b, k], dtype=np.complex128
            )
        else:
            first_output = np.empty(0, dtype=np.complex128)
        if self.second_embedding.complex_dimension:
            partner_block = np.zeros_like(block)
            partner_block[:, :, self.first_action.targets] = block[
                :, :, self.first_action.bases
            ].swapaxes(0, 1).conj()
            second_response_block = self.second_action(partner_block)
            second_output = np.asarray(
                second_response_block[
                    self.second_embedding.particle_valley_slots,
                    self.second_embedding.hole_valley_slots,
                    self.second_embedding.hole_k_indices,
                ],
                dtype=np.complex128,
            )
        else:
            second_output = np.empty(0, dtype=np.complex128)
        return first_output, second_output


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralFullHessianOrbit:
    """One prepared candidate paired-sector Hessian and its exact embeddings."""

    context: "Vituri2024HFSpiralFullHessianContext"
    orbit: Vituri2024HFSpiralFullSectorOrbit
    first_embedding: Vituri2024HFSpiralTransitionLaneEmbedding
    second_embedding: Vituri2024HFSpiralTransitionLaneEmbedding
    hessian: PairedSectorOrbitalHessian
    orbit_fingerprint: str
    candidate_only: bool = field(default=True, init=False)
    q_and_minus_q_both_retained: bool = field(default=True, init=False)
    conjugate_lane_dimensions_assumed_equal: bool = field(default=False, init=False)
    exact_unitary_scalar_curvature_established: bool = field(default=False, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    local_stability_established: bool = field(default=False, init=False)

    @property
    def complex_dimension(self) -> int:
        return self.hessian.complex_dimension

    @property
    def real_dimension(self) -> int:
        return self.hessian.real_dimension


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralFullHessianContext:
    """Validate-once source context that cheaply builds many sector orbits."""

    _factory_token: InitVar[object]
    response: Vituri2024HFSpiralSignedDisplacementResponse
    action_factory: Vituri2024HFSpiralValidatedResponseActionFactory
    selected_fock_diagonal_ev: Array
    context_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION, init=False
    )
    authority: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    selected_spin_only: bool = field(default=True, init=False)
    spectator_frozen_to_identity: bool = field(default=True, init=False)
    global_selected_rank: bool = field(default=True, init=False)
    all_occupation_transfers_in_inventory: bool = field(default=True, init=False)
    all_nonempty_inventory_orbits_buildable: bool = field(default=True, init=False)
    self_conjugate_sector_dimension_zero: bool = field(default=True, init=False)
    q_and_minus_q_averaged: bool = field(default=False, init=False)
    literal_float_full_functional_parity_established: bool = field(
        default=False, init=False
    )
    scalar_curvature_established: bool = field(default=False, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CONTEXT_TOKEN:
            raise TypeError("full Hessian context is factory-only")
        object.__setattr__(self, "context_fingerprint", self._expected_fingerprint())
        self.validate_live_state()

    @property
    def inventory(self) -> Vituri2024HFSpiralFullSectorInventory:
        return self.response.inventory

    @property
    def nk(self) -> int:
        return self.inventory.nk

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.context_fingerprint

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "authority": self.authority,
                "response": self.response.response_fingerprint,
                "action_factory": self.action_factory.expected_response_fingerprint,
                "selected_fock_diagonal_ev": _array_sha256(
                    self.selected_fock_diagonal_ev
                ),
                "flags": (
                    self.candidate_only,
                    self.selected_spin_only,
                    self.spectator_frozen_to_identity,
                    self.global_selected_rank,
                    self.all_occupation_transfers_in_inventory,
                    self.all_nonempty_inventory_orbits_buildable,
                    self.self_conjugate_sector_dimension_zero,
                    self.q_and_minus_q_averaged,
                    self.literal_float_full_functional_parity_established,
                    self.scalar_curvature_established,
                    self.reciprocity_established,
                    self.hermitian_eigensolver_authorized,
                    self.full_local_stability_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        if type(self.response) is not Vituri2024HFSpiralSignedDisplacementResponse:
            raise TypeError("full Hessian response type drifted")
        self.response.validate_live_state()
        if (
            type(self.action_factory)
            is not Vituri2024HFSpiralValidatedResponseActionFactory
            or self.action_factory.response is not self.response
            or self.action_factory.expected_response_fingerprint
            != self.response.response_fingerprint
        ):
            raise ValueError("full Hessian validated action factory drifted")
        expected = self.inventory.restricted_preparation.selected_hamiltonians_conventional
        off_diagonal = np.array(expected, copy=True)
        off_diagonal[0, 0] = 0.0
        off_diagonal[1, 1] = 0.0
        expected_diagonal = np.stack((expected[0, 0].real, expected[1, 1].real))
        diagonal_imaginary_residual = float(
            np.max(
                np.abs(np.stack((expected[0, 0].imag, expected[1, 1].imag))),
                initial=0.0,
            )
        )
        diagonal_scale = max(1.0, float(np.max(np.abs(expected_diagonal), initial=0.0)))
        diagonal_imaginary_tolerance = 64.0 * np.finfo(np.float64).eps * diagonal_scale
        if (
            np.count_nonzero(off_diagonal)
            or diagonal_imaginary_residual > diagonal_imaginary_tolerance
            or type(self.selected_fock_diagonal_ev) is not np.ndarray
            or self.selected_fock_diagonal_ev.dtype != np.dtype(np.float64)
            or self.selected_fock_diagonal_ev.shape != (2, self.nk)
            or self.selected_fock_diagonal_ev.flags.writeable
            or not np.array_equal(self.selected_fock_diagonal_ev, expected_diagonal)
        ):
            raise ValueError("full Hessian requires an exactly valley-diagonal normal Fock")
        locked = (
            self.api_version == VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION,
            self.authority == VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY,
            self.candidate_only is True,
            self.selected_spin_only is True,
            self.spectator_frozen_to_identity is True,
            self.global_selected_rank is True,
            self.all_occupation_transfers_in_inventory is True,
            self.all_nonempty_inventory_orbits_buildable is True,
            self.self_conjugate_sector_dimension_zero is True,
            self.inventory.sector_complex_dimension(
                Vituri2024HFSpiralFullSectorKey(0, 0, 0)
            )
            == 0,
            self.q_and_minus_q_averaged is False,
            self.literal_float_full_functional_parity_established is False,
            self.scalar_curvature_established is False,
            self.reciprocity_established is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
            self._expected_fingerprint() == self.context_fingerprint,
        )
        if not all(locked):
            raise ValueError("full Hessian context authority or binding drifted")

    def _build_embedding(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        bases: Array,
        targets: Array,
    ) -> Vituri2024HFSpiralTransitionLaneEmbedding:
        occupations = self.inventory.selected_occupations
        particle_k: list[int] = []
        hole_k: list[int] = []
        particle_slot: list[int] = []
        hole_slot: list[int] = []
        target_by_base = dict(zip(bases.tolist(), targets.tolist(), strict=True))
        for particle_valley, hole_valley in _allowed_pairs(key.valley_charge):
            for base in bases:
                base_int = int(base)
                target = int(target_by_base[base_int])
                if occupations[hole_valley, base_int] and not occupations[
                    particle_valley, target
                ]:
                    particle_k.append(target)
                    hole_k.append(base_int)
                    particle_slot.append(particle_valley)
                    hole_slot.append(hole_valley)
        dimension = len(particle_k)
        if dimension != self.inventory.sector_complex_dimension(key):
            raise RuntimeError("transition embedding dimension disagrees with inventory")
        particle_k_array = np.asarray(particle_k, dtype=np.int64)
        hole_k_array = np.asarray(hole_k, dtype=np.int64)
        particle_slot_array = np.asarray(particle_slot, dtype=np.int64)
        hole_slot_array = np.asarray(hole_slot, dtype=np.int64)
        return Vituri2024HFSpiralTransitionLaneEmbedding(
            _factory_token=_CONTEXT_TOKEN,
            key=key,
            particle_k_indices=particle_k_array,
            hole_k_indices=hole_k_array,
            particle_valley_slots=particle_slot_array,
            hole_valley_slots=hole_slot_array,
            particle_orbital_ids=particle_slot_array * self.nk + particle_k_array,
            hole_orbital_ids=hole_slot_array * self.nk + hole_k_array,
        )

    def build_orbit_hessian(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> Vituri2024HFSpiralFullHessianOrbit:
        """Build one canonical nonempty orbit without full source revalidation."""

        if self._expected_fingerprint() != self.context_fingerprint:
            raise ValueError("full Hessian context became stale")
        if type(key) is not Vituri2024HFSpiralFullSectorKey:
            raise TypeError("orbit key type drifted")
        partner = key.conjugate
        first_key, second_key = (key, partner) if key <= partner else (partner, key)
        first_dimension = self.inventory.sector_complex_dimension(first_key)
        second_dimension = self.inventory.sector_complex_dimension(second_key)
        if first_dimension + second_dimension == 0:
            raise ValueError("cannot build a zero-dimensional conjugate orbit")
        if first_key == second_key:
            raise ValueError("nonempty self-conjugate sectors require a separate adapter")
        orbit = Vituri2024HFSpiralFullSectorOrbit(
            first_key, second_key, first_dimension, second_dimension
        )
        first_action = self.action_factory.prepare_fft_action(first_key)
        second_action = self.action_factory.prepare_fft_action(second_key)
        first_embedding = self._build_embedding(
            first_key, first_action.bases, first_action.targets
        )
        second_embedding = self._build_embedding(
            second_key, second_action.bases, second_action.targets
        )
        first_particle_energy = self.selected_fock_diagonal_ev[
            first_embedding.particle_valley_slots,
            first_embedding.particle_k_indices,
        ]
        first_hole_energy = self.selected_fock_diagonal_ev[
            first_embedding.hole_valley_slots,
            first_embedding.hole_k_indices,
        ]
        second_particle_energy = self.selected_fock_diagonal_ev[
            second_embedding.particle_valley_slots,
            second_embedding.particle_k_indices,
        ]
        second_hole_energy = self.selected_fock_diagonal_ev[
            second_embedding.hole_valley_slots,
            second_embedding.hole_k_indices,
        ]
        first_lane = OrbitalTransitionLane(
            label=f"d=({first_key.displacement_x},{first_key.displacement_y}),chi={first_key.valley_charge}",
            particle_orbital_ids=first_embedding.particle_orbital_ids,
            hole_orbital_ids=first_embedding.hole_orbital_ids,
            particle_energies_ev=np.asarray(first_particle_energy, dtype=np.float64),
            hole_energies_ev=np.asarray(first_hole_energy, dtype=np.float64),
        )
        second_lane = OrbitalTransitionLane(
            label=f"d=({second_key.displacement_x},{second_key.displacement_y}),chi={second_key.valley_charge}",
            particle_orbital_ids=second_embedding.particle_orbital_ids,
            hole_orbital_ids=second_embedding.hole_orbital_ids,
            particle_energies_ev=np.asarray(second_particle_energy, dtype=np.float64),
            hole_energies_ev=np.asarray(second_hole_energy, dtype=np.float64),
        )
        frame = PairedOrbitalTransitionFrame(first_lane, second_lane)
        callback = _VituriPairedInteractionCallback(
            response=self.response,
            first_embedding=first_embedding,
            second_embedding=second_embedding,
            first_action=first_action,
            second_action=second_action,
            expected_response_fingerprint=self.response.response_fingerprint,
        )
        hessian = PairedSectorOrbitalHessian(frame, callback)
        fingerprint = _fingerprint(
            {
                "context": self.context_fingerprint,
                "orbit": (
                    first_key.displacement_x,
                    first_key.displacement_y,
                    first_key.valley_charge,
                    second_key.displacement_x,
                    second_key.displacement_y,
                    second_key.valley_charge,
                    first_dimension,
                    second_dimension,
                ),
                "first_embedding": first_embedding.embedding_fingerprint,
                "second_embedding": second_embedding.embedding_fingerprint,
                "factor_two_real_hessian": True,
            }
        )
        return Vituri2024HFSpiralFullHessianOrbit(
            context=self,
            orbit=orbit,
            first_embedding=first_embedding,
            second_embedding=second_embedding,
            hessian=hessian,
            orbit_fingerprint=fingerprint,
        )


def build_vituri2024_hf_spiral_full_hessian_context(
    response: Vituri2024HFSpiralSignedDisplacementResponse,
) -> Vituri2024HFSpiralFullHessianContext:
    """Validate the source once and prepare cheap per-orbit construction."""

    if type(response) is not Vituri2024HFSpiralSignedDisplacementResponse:
        raise TypeError("full Hessian context requires the exact response")
    response.validate_live_state()
    hamiltonians = response.inventory.restricted_preparation.selected_hamiltonians_conventional
    diagonal = np.stack((hamiltonians[0, 0].real, hamiltonians[1, 1].real))
    readonly_diagonal = _readonly(
        np.asarray(diagonal, dtype=np.float64),
        np.dtype(np.float64),
        (2, response.nk),
        "selected Fock diagonal",
    )
    return Vituri2024HFSpiralFullHessianContext(
        _factory_token=_CONTEXT_TOKEN,
        response=response,
        action_factory=response.make_validated_action_factory(),
        selected_fock_diagonal_ev=readonly_diagonal,
    )


__all__ = [
    "VITURI2024_HF_SPIRAL_FULL_HESSIAN_API_VERSION",
    "VITURI2024_HF_SPIRAL_FULL_HESSIAN_AUTHORITY",
    "Vituri2024HFSpiralFullHessianContext",
    "Vituri2024HFSpiralFullHessianOrbit",
    "Vituri2024HFSpiralTransitionLaneEmbedding",
    "build_vituri2024_hf_spiral_full_hessian_context",
]
