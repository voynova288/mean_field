"""Paper-direct normal-order representative for the Vituri active-band Hamiltonian.

The pinned supplementary material displays the projected active ``n=3``
Hamiltonian as an explicit quartic ``c† c† c c`` interaction plus a one-body
``Delta mu * N`` term that is absorbed into the chemical potential.  In the
conventional-density factorized functional this fixes the canonical empty
active-electron Fock-vacuum representative ``R=0``.  It does not fix the
physical neutral density, the global one-body identity gauge, or an HF q=0
background prescription.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from hashlib import sha256
import json
from typing import Final

import numpy as np

from .vituri2024 import (
    ARXIV_IDENTIFIER,
    ARXIV_SOURCE_SHA256,
    SM_TEX_AUTHORITY_PATH,
    SM_TEX_SHA256,
)
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_replay import Vituri2024HalfMetalHFReplayPayload
from .vituri2024_interaction import (
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
from .vituri2024_tdhf_full_provider_bridge import (
    Vituri2024FullFunctionalReplayBridge,
    build_vituri2024_full_functional_replay_bridge,
)

Array = np.ndarray
InteractionInput = Vituri2024InteractionChoiceReceipt | Vituri2024InteractionBinding

VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_API_VERSION: Final[str] = (
    "vituri2024_projected_hamiltonian_reference.v1"
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY: Final[str] = (
    "paper_direct_active_band_projected_H_quartic_ordering_R0_"
    "canonical_representative_only"
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_LOCATOR: Final[str] = (
    "SM.tex active-band projector and projected-interaction display immediately "
    "before Gauge Fixing (pinned extraction lines 78-93)"
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT: Final[str] = (
    "After projecting the six-band Coulomb Hamiltonian to the third-lowest "
    "active band, P H_C P is an explicit (1/(2A)) sum over flavors, "
    "sublattice indices, and k1..k4 of "
    "a_i,lambda1*(k1) a_j,lambda2*(k2) a_j,lambda2(k3) "
    "a_i,lambda1(k4) V_TF(k2-k3) c^dagger_{lambda1,k1} "
    "c^dagger_{lambda2,k2} c_{lambda2,k3} c_{lambda1,k4} with exact "
    "momentum conservation, plus Delta mu times sum_{lambda,k} "
    "c^dagger_{lambda,k} c_{lambda,k}. "
    "The supplementary material states that Delta mu is a shift in the "
    "chemical potential that can be absorbed through a redefinition of the "
    "chemical potential."
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256: Final[str] = sha256(
    VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT.encode("utf-8")
).hexdigest()
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_DENSITY_CONVENTION: Final[str] = (
    "P_ij=<c_j_dagger c_i>; canonical empty-active-electron Fock vacuum R=0"
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_MU_STATUS: Final[str] = (
    "Delta_mu_times_N_unresolved_global_one_body_identity_gauge"
)
VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_Q0_STATUS: Final[str] = (
    "no_HF_q0_background_or_physical_neutral_reference_authority"
)

_REFERENCE_TOKEN = object()
_COMPOSITE_TOKEN = object()


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


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _zero_reference(dimension: int) -> Array:
    payload = np.zeros((dimension, dimension), dtype=np.complex128).tobytes(order="C")
    result = np.frombuffer(payload, dtype=np.complex128).reshape(dimension, dimension)
    result.setflags(write=False)
    return result


def _fingerprint(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Vituri2024ProjectedHamiltonianReferenceReceipt:
    _factory_token: InitVar[object]
    nk: int
    dimension: int
    normal_order_reference_full: Array
    normal_order_reference_array_sha256: str
    canonical_equation_text_sha256: str
    construction_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_API_VERSION, init=False
    )
    authority: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY, init=False
    )
    arxiv_identifier: str = field(default=ARXIV_IDENTIFIER, init=False)
    arxiv_source_sha256: str = field(default=ARXIV_SOURCE_SHA256, init=False)
    sm_tex_authority_path: str = field(default=SM_TEX_AUTHORITY_PATH, init=False)
    sm_tex_sha256: str = field(default=SM_TEX_SHA256, init=False)
    equation_locator: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_LOCATOR, init=False
    )
    canonical_equation_text: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT, init=False
    )
    density_convention: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_DENSITY_CONVENTION,
        init=False,
    )
    orbital_order: str = field(default="flavor_major_then_k", init=False)
    chemical_potential_status: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_MU_STATUS, init=False
    )
    q0_status: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_Q0_STATUS, init=False
    )
    paper_active_quartic_R0_semantics_established: bool = field(
        default=True, init=False
    )
    canonical_empty_active_electron_reference: bool = field(default=True, init=False)
    r0_is_physical_neutral_reference: bool = field(default=False, init=False)
    physical_neutral_density_identified: bool = field(default=False, init=False)
    normal_order_authority_established: bool = field(default=False, init=False)
    absolute_identity_shift_authority_established: bool = field(
        default=False, init=False
    )
    fixed_N_ensemble_authority_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    replay_source_authority_established: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    absolute_fock_zero_authority_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_numerical_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _REFERENCE_TOKEN:
            raise TypeError("projected-Hamiltonian reference receipt is factory-only")
        self._validate_fields(check_construction=False)
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                item.name: (
                    self.normal_order_reference_array_sha256
                    if item.name == "normal_order_reference_full"
                    else getattr(self, item.name)
                )
                for item in fields(self)
                if item.name != "construction_fingerprint"
            }
        )

    def _validate_fields(self, *, check_construction: bool) -> None:
        if type(self.nk) is not int or self.nk < 1:
            raise ValueError("projected-Hamiltonian reference Nk must be positive")
        expected_dimension = len(INTERNAL_FLAVOR_ORDER) * self.nk
        if type(self.dimension) is not int or self.dimension != expected_dimension:
            raise ValueError("projected-Hamiltonian reference dimension must equal 4*Nk")
        matrix = self.normal_order_reference_full
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.dtype != np.dtype(np.complex128)
            or matrix.shape != (self.dimension, self.dimension)
            or matrix.flags.writeable
            or matrix.flags.owndata
            or np.any(matrix != 0.0)
        ):
            raise ValueError("projected-Hamiltonian reference must be bytes-backed exact R=0")
        if _array_sha256(matrix) != self.normal_order_reference_array_sha256:
            raise ValueError("projected-Hamiltonian zero-reference bytes/hash drifted")
        _sha256(self.normal_order_reference_array_sha256, "zero-reference array hash")
        if (
            self.canonical_equation_text_sha256
            != VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256
            or sha256(self.canonical_equation_text.encode("utf-8")).hexdigest()
            != self.canonical_equation_text_sha256
        ):
            raise ValueError("projected-Hamiltonian canonical equation text drifted")
        expected = (
            self.api_version == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_API_VERSION,
            self.authority == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY,
            self.arxiv_identifier == ARXIV_IDENTIFIER,
            self.arxiv_source_sha256 == ARXIV_SOURCE_SHA256,
            self.sm_tex_authority_path == SM_TEX_AUTHORITY_PATH,
            self.sm_tex_sha256 == SM_TEX_SHA256,
            self.equation_locator == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_LOCATOR,
            self.canonical_equation_text == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT,
            self.density_convention
            == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_DENSITY_CONVENTION,
            self.orbital_order == "flavor_major_then_k",
            self.chemical_potential_status
            == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_MU_STATUS,
            self.q0_status == VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_Q0_STATUS,
            self.paper_active_quartic_R0_semantics_established is True,
            self.canonical_empty_active_electron_reference is True,
            self.r0_is_physical_neutral_reference is False,
            self.physical_neutral_density_identified is False,
            self.normal_order_authority_established is False,
            self.absolute_identity_shift_authority_established is False,
            self.fixed_N_ensemble_authority_established is False,
            self.q0_background_authority_established is False,
            self.replay_source_authority_established is False,
            self.source_closure_established is False,
            self.absolute_fock_zero_authority_established is False,
            self.production_ready is False,
            self.paper_numerical_reproduction_verified is False,
        )
        if not all(expected):
            raise ValueError("projected-Hamiltonian reference authority was inflated")
        if check_construction and self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("projected-Hamiltonian reference construction drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_construction=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def make_vituri2024_projected_hamiltonian_zero_reference(
    *, nk: int
) -> Vituri2024ProjectedHamiltonianReferenceReceipt:
    if type(nk) is not int or nk < 1:
        raise ValueError("projected-Hamiltonian reference Nk must be a positive int")
    dimension = len(INTERNAL_FLAVOR_ORDER) * nk
    reference = _zero_reference(dimension)
    return Vituri2024ProjectedHamiltonianReferenceReceipt(
        _factory_token=_REFERENCE_TOKEN,
        nk=nk,
        dimension=dimension,
        normal_order_reference_full=reference,
        normal_order_reference_array_sha256=_array_sha256(reference),
        canonical_equation_text_sha256=(
            VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256
        ),
    )


@dataclass(frozen=True, slots=True)
class Vituri2024ProjectedHamiltonianReferenceBridge:
    _factory_token: InitVar[object]
    reference: Vituri2024ProjectedHamiltonianReferenceReceipt
    replay_bridge: Vituri2024FullFunctionalReplayBridge
    construction_fingerprint: str = field(init=False)
    authority: str = field(
        default=(
            "paper_R0_semantics_plus_selected_R0_identity_gauge_saved_array_parity_"
            "only_no_absolute_mu_q0_source_tdhf_production_or_paper_numerical_authority"
        ),
        init=False,
    )
    paper_active_quartic_R0_semantics_established: bool = field(
        default=True, init=False
    )
    selected_R0_identity_gauge_saved_array_parity_passed: bool = field(
        default=True, init=False
    )
    absolute_identity_shift_authority_established: bool = field(
        default=False, init=False
    )
    replay_source_authority_established: bool = field(default=False, init=False)
    absolute_fock_zero_authority_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_numerical_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _COMPOSITE_TOKEN:
            raise TypeError("projected-Hamiltonian reference bridge is factory-only")
        self._validate_fields(check_construction=False)
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "reference": self.reference.fingerprint,
                "replay_bridge": self.replay_bridge.fingerprint,
                "authority": self.authority,
                "paper_active_quartic_R0_semantics_established": (
                    self.paper_active_quartic_R0_semantics_established
                ),
                "selected_R0_identity_gauge_saved_array_parity_passed": (
                    self.selected_R0_identity_gauge_saved_array_parity_passed
                ),
                "absolute_identity_shift_authority_established": (
                    self.absolute_identity_shift_authority_established
                ),
                "replay_source_authority_established": (
                    self.replay_source_authority_established
                ),
                "absolute_fock_zero_authority_established": (
                    self.absolute_fock_zero_authority_established
                ),
                "q0_background_authority_established": (
                    self.q0_background_authority_established
                ),
                "source_closure_established": self.source_closure_established,
                "production_ready": self.production_ready,
                "paper_numerical_reproduction_verified": (
                    self.paper_numerical_reproduction_verified
                ),
            }
        )

    def _validate_fields(self, *, check_construction: bool) -> None:
        if type(self.reference) is not Vituri2024ProjectedHamiltonianReferenceReceipt:
            raise TypeError("reference bridge requires exact paper reference receipt")
        if type(self.replay_bridge) is not Vituri2024FullFunctionalReplayBridge:
            raise TypeError("reference bridge requires exact replay bridge")
        self.reference.validate_live_state()
        self.replay_bridge.validate_live_state()
        if self.reference.nk != self.replay_bridge.kernel.nk:
            raise ValueError("paper reference/replay bridge Nk mismatch")
        if (
            self.replay_bridge.kernel.normal_order_reference_fingerprint
            != self.reference.fingerprint
            or self.replay_bridge.normal_order_reference_array_sha256
            != self.reference.normal_order_reference_array_sha256
        ):
            raise ValueError("paper reference/replay bridge reference binding drifted")
        expected = (
            self.paper_active_quartic_R0_semantics_established is True,
            self.selected_R0_identity_gauge_saved_array_parity_passed is True,
            self.absolute_identity_shift_authority_established is False,
            self.replay_source_authority_established is False,
            self.absolute_fock_zero_authority_established is False,
            self.q0_background_authority_established is False,
            self.source_closure_established is False,
            self.production_ready is False,
            self.paper_numerical_reproduction_verified is False,
        )
        if not all(expected):
            raise ValueError("projected-Hamiltonian reference bridge authority inflated")
        if check_construction and self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("projected-Hamiltonian reference bridge construction drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_construction=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference(
    *,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    reference: Vituri2024ProjectedHamiltonianReferenceReceipt,
    area_angstrom_squared: float,
    interaction: InteractionInput,
    q0_policy_fingerprint: str,
    q0_background_evidence_sha256: str,
    provenance: str,
) -> Vituri2024ProjectedHamiltonianReferenceBridge:
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("paper-reference bridge requires exact replay payload")
    if type(reference) is not Vituri2024ProjectedHamiltonianReferenceReceipt:
        raise TypeError("paper-reference bridge requires exact reference receipt")
    reference.validate_live_state()
    if source_payload.mesh.shape[0] != reference.nk:
        raise ValueError("paper reference Nk differs from replay payload")
    bridge = build_vituri2024_full_functional_replay_bridge(
        source_payload=source_payload,
        normal_order_reference_full=reference.normal_order_reference_full,
        area_angstrom_squared=area_angstrom_squared,
        interaction=interaction,
        normal_order_reference_fingerprint=reference.fingerprint,
        reference_policy_evidence_sha256=reference.canonical_equation_text_sha256,
        q0_policy_fingerprint=q0_policy_fingerprint,
        q0_background_evidence_sha256=q0_background_evidence_sha256,
        provenance=provenance,
    )
    return Vituri2024ProjectedHamiltonianReferenceBridge(
        _factory_token=_COMPOSITE_TOKEN,
        reference=reference,
        replay_bridge=bridge,
    )


__all__ = [
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_API_VERSION",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_DENSITY_CONVENTION",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_LOCATOR",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_MU_STATUS",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_Q0_STATUS",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256",
    "Vituri2024ProjectedHamiltonianReferenceBridge",
    "Vituri2024ProjectedHamiltonianReferenceReceipt",
    "build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference",
    "make_vituri2024_projected_hamiltonian_zero_reference",
]
