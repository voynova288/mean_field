from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from mean_field.core.hf import compute_hf_energy, compute_oda_parameter

from ._hf_shared import _rlg_hbn_zero_literal_q0_fock
from ._hf_interaction_path import (
    RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION,
    build_rlg_hbn_interaction_components,
    interaction_shifts_for_cutoff,
)
from ._hf_types import (
    RLGhBNHartreeFockState,
    RLGhBNInteractionComponents,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
)

RLG_HBN_TRACK_P_PROVIDER_VERSION = "rlg_hbn_track_p_provider_v1"


def _update_array_hash(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))


@dataclass(frozen=True, eq=False)
class RLGhBNTrackPInteractionProvider:
    """Executable fixed-G Track-P interaction contract shared by HF and response."""

    basis_data: RLGhBNProjectedBasisData
    overlap_blocks: RLGhBNLayerOverlapBlockSet
    beta: float = 1.0
    physical_shifts: tuple[tuple[int, int], ...] = field(init=False)
    fingerprint: str = field(init=False)
    response_cache_fingerprint: str = field(init=False)
    zero_literal_q0_fock: bool = field(init=False, default=False)
    _physical_payload_ids: tuple[tuple[str, tuple[int, int] | None, int], ...] = field(
        init=False,
        repr=False,
    )
    _response_payload_ids: tuple[tuple[str, tuple[int, int], int], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not np.isfinite(beta):
            raise ValueError(f"Track-P provider beta must be finite, got {beta}")
        if _rlg_hbn_zero_literal_q0_fock():
            raise ValueError(
                "Track-P provider refuses MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1"
            )
        expected = interaction_shifts_for_cutoff(
            self.basis_data.basis_model.lattice,
            self.basis_data.interaction,
        )
        restored = tuple(
            (int(value[0]), int(value[1]))
            for value in self.overlap_blocks.shifts
        )
        if restored != expected:
            raise ValueError(
                "Track-P provider physical shifts do not equal configured shell: "
                f"restored={restored}, expected={expected}"
            )
        if len(set(restored)) != len(restored):
            raise ValueError("Track-P provider physical shifts contain duplicates")
        if {(-first, -second) for first, second in restored} != set(restored):
            raise ValueError("Track-P provider physical shifts are not G -> -G closed")
        expected_h0_shape = (
            self.basis_data.nt,
            self.basis_data.nt,
            self.basis_data.nk,
        )
        if np.asarray(self.basis_data.h0).shape != expected_h0_shape:
            raise ValueError(
                f"Track-P provider h0 shape mismatch: {self.basis_data.h0.shape}"
            )
        for shift in restored:
            for name, table in (
                ("layer_overlaps", self.overlap_blocks.layer_overlaps),
                (
                    "layer_diagonal_overlaps",
                    self.overlap_blocks.layer_diagonal_overlaps,
                ),
                (
                    "hartree_layer_coulomb",
                    self.overlap_blocks.hartree_layer_coulomb,
                ),
                (
                    "fock_layer_coulomb",
                    self.overlap_blocks.fock_layer_coulomb,
                ),
            ):
                if shift not in table:
                    raise ValueError(
                        f"Track-P provider lacks physical shift {shift} in {name}"
                    )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "physical_shifts", restored)
        physical_ids, response_ids = self._freeze_and_record_payloads()
        object.__setattr__(self, "_physical_payload_ids", physical_ids)
        object.__setattr__(self, "_response_payload_ids", response_ids)
        object.__setattr__(self, "fingerprint", self._build_fingerprint())
        object.__setattr__(
            self,
            "response_cache_fingerprint",
            self._build_response_cache_fingerprint(),
        )

    def _freeze_and_record_payloads(
        self,
    ) -> tuple[
        tuple[tuple[str, tuple[int, int] | None, int], ...],
        tuple[tuple[str, tuple[int, int], int], ...],
    ]:
        physical: list[tuple[str, tuple[int, int] | None, int]] = []
        response: list[tuple[str, tuple[int, int], int]] = []

        def freeze(name: str, key: tuple[int, int] | None, values: np.ndarray) -> None:
            np.asarray(values).setflags(write=False)
            physical.append((name, key, id(values)))

        freeze("basis_h0", None, self.basis_data.h0)
        if self.basis_data.fixed_remote_hamiltonian is not None:
            freeze(
                "fixed_remote_hamiltonian",
                None,
                self.basis_data.fixed_remote_hamiltonian,
            )
        freeze("gvecs", None, self.overlap_blocks.gvecs)
        for name, table in (
            ("layer_overlaps", self.overlap_blocks.layer_overlaps),
            (
                "layer_diagonal_overlaps",
                self.overlap_blocks.layer_diagonal_overlaps,
            ),
            ("hartree_layer_coulomb", self.overlap_blocks.hartree_layer_coulomb),
            ("fock_layer_coulomb", self.overlap_blocks.fock_layer_coulomb),
        ):
            for key, values in table.items():
                np.asarray(values).setflags(write=False)
                if key in self.physical_shifts:
                    physical.append((name, key, id(values)))
                if name in ("layer_overlaps", "fock_layer_coulomb"):
                    response.append((name, key, id(values)))
        return tuple(physical), tuple(response)

    @property
    def convention(self) -> str:
        return RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION

    @property
    def v0(self) -> float:
        return float(self.basis_data.v0)

    def _build_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(RLG_HBN_TRACK_P_PROVIDER_VERSION.encode("utf-8"))
        digest.update(self.convention.encode("utf-8"))
        digest.update(b"zero_literal_q0_fock=0")
        digest.update(np.asarray([self.beta, self.v0], dtype=np.float64).tobytes())
        digest.update(
            np.asarray(
                [self.basis_data.nt, self.basis_data.nk], dtype=np.int64
            ).tobytes()
        )
        digest.update(np.asarray(self.physical_shifts, dtype=np.int64).tobytes())
        _update_array_hash(digest, self.basis_data.h0)
        if self.basis_data.fixed_remote_hamiltonian is not None:
            _update_array_hash(digest, self.basis_data.fixed_remote_hamiltonian)
        _update_array_hash(digest, self.overlap_blocks.gvecs)
        for shift in self.physical_shifts:
            digest.update(np.asarray(shift, dtype=np.int64).tobytes())
            _update_array_hash(digest, self.overlap_blocks.layer_overlaps[shift])
            _update_array_hash(
                digest, self.overlap_blocks.layer_diagonal_overlaps[shift]
            )
            _update_array_hash(
                digest, self.overlap_blocks.hartree_layer_coulomb[shift]
            )
            _update_array_hash(
                digest, self.overlap_blocks.fock_layer_coulomb[shift]
            )
        return digest.hexdigest()

    def _build_response_cache_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"rlg_hbn_track_p_response_cache_v1")
        for name, table in (
            ("layer_overlaps", self.overlap_blocks.layer_overlaps),
            ("fock_layer_coulomb", self.overlap_blocks.fock_layer_coulomb),
        ):
            digest.update(name.encode("ascii"))
            for key in sorted(table):
                digest.update(np.asarray(key, dtype=np.int64).tobytes())
                _update_array_hash(digest, table[key])
        return digest.hexdigest()

    def validate_runtime_policy(self) -> None:
        if _rlg_hbn_zero_literal_q0_fock():
            raise ValueError(
                "Track-P provider runtime q=0 Fock policy changed after construction"
            )

    def validate_integrity(self, *, recompute_hashes: bool = False) -> None:
        tables = {
            "layer_overlaps": self.overlap_blocks.layer_overlaps,
            "layer_diagonal_overlaps": self.overlap_blocks.layer_diagonal_overlaps,
            "hartree_layer_coulomb": self.overlap_blocks.hartree_layer_coulomb,
            "fock_layer_coulomb": self.overlap_blocks.fock_layer_coulomb,
        }
        special = {
            "basis_h0": self.basis_data.h0,
            "fixed_remote_hamiltonian": self.basis_data.fixed_remote_hamiltonian,
            "gvecs": self.overlap_blocks.gvecs,
        }
        for name, key, expected_id in self._physical_payload_ids:
            values = special[name] if key is None else tables[name].get(key)
            if values is None or id(values) != expected_id:
                raise ValueError(
                    f"Track-P provider physical payload identity changed: {name} {key}"
                )
            if np.asarray(values).flags.writeable:
                raise ValueError(
                    f"Track-P provider physical payload became writeable: {name} {key}"
                )
        current_response_entries = {
            (name, key, id(values))
            for name, table in (
                ("layer_overlaps", self.overlap_blocks.layer_overlaps),
                ("fock_layer_coulomb", self.overlap_blocks.fock_layer_coulomb),
            )
            for key, values in table.items()
        }
        if current_response_entries != set(self._response_payload_ids):
            raise ValueError("Track-P provider response-cache payload identities changed")
        for name, key, _expected_id in self._response_payload_ids:
            if np.asarray(tables[name][key]).flags.writeable:
                raise ValueError(
                    f"Track-P response-cache payload became writeable: {name} {key}"
                )
        if recompute_hashes:
            if self._build_fingerprint() != self.fingerprint:
                raise ValueError("Track-P provider physical fingerprint is stale")
            if (
                self._build_response_cache_fingerprint()
                != self.response_cache_fingerprint
            ):
                raise ValueError("Track-P provider response-cache fingerprint is stale")

    def validate_state(self, state: RLGhBNHartreeFockState) -> None:
        self.validate_runtime_policy()
        self.validate_integrity()
        expected_shape = (
            self.basis_data.nt,
            self.basis_data.nt,
            self.basis_data.nk,
        )
        if state.density.shape != expected_shape:
            raise ValueError(
                f"Track-P provider/state density shape mismatch: {state.density.shape}"
            )
        if not np.isclose(state.v0, self.v0, rtol=0.0, atol=1.0e-15):
            raise ValueError(
                f"Track-P provider/state v0 mismatch: {state.v0} != {self.v0}"
            )
        if not np.array_equal(state.h0, self.basis_data.h0):
            raise ValueError("Track-P provider/state h0 arrays are not identical")

    def scf_components(self, density: np.ndarray) -> RLGhBNInteractionComponents:
        self.validate_runtime_policy()
        self.validate_integrity()
        return build_rlg_hbn_interaction_components(
            density,
            self.overlap_blocks,
            v0=self.v0,
            beta=self.beta,
        )

    def scf_hamiltonian(self, density: np.ndarray) -> np.ndarray:
        return self.scf_components(density).total

    def tangent_components(
        self, delta_density: np.ndarray
    ) -> RLGhBNInteractionComponents:
        self.validate_runtime_policy()
        self.validate_integrity()
        return build_rlg_hbn_interaction_components(
            delta_density,
            self.overlap_blocks,
            v0=self.v0,
            beta=self.beta,
        )

    def tangent_hamiltonian(self, delta_density: np.ndarray) -> np.ndarray:
        return self.tangent_components(delta_density).total

    def energy_functional(
        self,
        interaction_hamiltonian: np.ndarray,
        h0: np.ndarray,
        density: np.ndarray,
    ) -> float:
        self.validate_runtime_policy()
        self.validate_integrity()
        if not np.array_equal(np.asarray(h0), np.asarray(self.basis_data.h0)):
            raise ValueError("Track-P provider energy h0 is not its bound basis h0")
        return compute_hf_energy(interaction_hamiltonian, h0, density)

    def oda_parameter(
        self,
        state: RLGhBNHartreeFockState,
        delta_density: np.ndarray,
    ) -> float:
        self.validate_state(state)
        return compute_oda_parameter(
            state,
            delta_density,
            interaction_builder=self.tangent_hamiltonian,
        )

    def with_overlap_blocks(
        self, overlap_blocks: RLGhBNLayerOverlapBlockSet
    ) -> "RLGhBNTrackPInteractionProvider":
        result = RLGhBNTrackPInteractionProvider(
            self.basis_data,
            overlap_blocks,
            beta=self.beta,
        )
        if result.fingerprint != self.fingerprint:
            raise ValueError(
                "auxiliary overlap completion changed the Track-P physical "
                "functional fingerprint"
            )
        return result


def build_rlg_hbn_track_p_interaction_provider(
    basis_data: RLGhBNProjectedBasisData,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    beta: float = 1.0,
) -> RLGhBNTrackPInteractionProvider:
    return RLGhBNTrackPInteractionProvider(
        basis_data=basis_data,
        overlap_blocks=overlap_blocks,
        beta=beta,
    )


__all__ = [
    "RLG_HBN_TRACK_P_PROVIDER_VERSION",
    "RLGhBNTrackPInteractionProvider",
    "build_rlg_hbn_track_p_interaction_provider",
]
