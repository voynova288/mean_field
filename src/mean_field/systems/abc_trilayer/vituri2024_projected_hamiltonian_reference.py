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
import math
from typing import Final

import numpy as np

from mean_field.core.hf.tdhf_scalar_functional import (
    TDHFFullProjectorFunctionalBinding,
    TDHFFullProjectorSingleFockExecutionReceipt,
    TDHFFullProjectorSpace,
    TDHFScalarFunctionalInputsManifest,
    execute_tdhf_full_projector_fock_once,
)

from .vituri2024 import (
    ARXIV_IDENTIFIER,
    ARXIV_SOURCE_SHA256,
    SM_TEX_AUTHORITY_PATH,
    SM_TEX_SHA256,
)
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_replay import (
    Vituri2024HalfMetalHFReplayPayload,
    canonical_orbital_order_sha256,
)
from .vituri2024_interaction import (
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
from .vituri2024_tdhf_full_functional import (
    Vituri2024FullProjectedFunctionalKernel,
)
from .vituri2024_tdhf_full_provider_callbacks import (
    VITURI2024_FULL_PROVIDER_INPUT_NAMES,
    make_vituri2024_full_provider_binding,
    make_vituri2024_full_provider_inputs,
)
from .vituri2024_tdhf_full_provider_bridge import (
    Vituri2024FullFunctionalReplayBridge,
    build_vituri2024_full_functional_replay_bridge,
)
from .vituri2024_tdhf_full_scalar import (
    vituri2024_payload_density_to_full_projector,
    vituri2024_payload_operator_to_full_dense,
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
VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION: Final[str] = (
    "vituri2024_projected_hamiltonian_identity_gauge.v1"
)
VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV: Final[float] = (
    1.0e-10
)
VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY: Final[str] = (
    "selected_R0_fixed_rank_operator_parity_modulo_one_common_real_identity_only_"
    "no_physical_Delta_mu_source_q0_tdhf_production_or_paper_authority"
)

_REFERENCE_TOKEN = object()
_COMPOSITE_TOKEN = object()
_IDENTITY_CANDIDATE_TOKEN = object()
_IDENTITY_PARITY_TOKEN = object()


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


def _readonly_array(value: object, *, dtype: np.dtype, shape: tuple[int, ...], label: str) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise TypeError(f"{label} must be an exact {dtype} numpy array")
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    result = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _readonly_hermitian(value: object, *, dimension: int, label: str) -> Array:
    result = _readonly_array(
        value,
        dtype=np.dtype(np.complex128),
        shape=(dimension, dimension),
        label=label,
    )
    residual = float(np.max(np.abs(result - result.conj().T)))
    if residual > VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV:
        raise ValueError(f"{label} is not Hermitian")
    return result


def _target_free_source_input_fingerprint(
    payload: Vituri2024HalfMetalHFReplayPayload,
) -> str:
    return _fingerprint(
        {
            "provider_fingerprint": payload.provider_fingerprint,
            "source_commit": payload.source_commit,
            "replay_loader_implementation_fingerprint": (
                payload.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": (
                payload.replay_payload_schema_fingerprint
            ),
            "mesh": _array_sha256(payload.mesh),
            "active_band_states": _array_sha256(payload.active_band_states),
            "h0": _array_sha256(payload.h0),
            "projector": _array_sha256(payload.projector),
            "occupations": _array_sha256(payload.occupations),
            "targets_excluded": ("interaction_h", "fock", "energies"),
        }
    )


def _target_payload_fingerprint(payload: Vituri2024HalfMetalHFReplayPayload) -> str:
    return _fingerprint(
        {
            "provider_fingerprint": payload.provider_fingerprint,
            "source_commit": payload.source_commit,
            "source_artifact_sha256": payload.source_artifact_sha256,
            "spec_fingerprint": payload.spec_fingerprint,
            "source_state_sha256": payload.source_state_sha256,
            "replay_loader_implementation_fingerprint": (
                payload.replay_loader_implementation_fingerprint
            ),
            "replay_payload_schema_fingerprint": (
                payload.replay_payload_schema_fingerprint
            ),
            "arrays": {
                name: _array_sha256(getattr(payload, name))
                for name in (
                    "mesh",
                    "active_band_states",
                    "h0",
                    "interaction_h",
                    "fock",
                    "projector",
                    "energies",
                    "occupations",
                )
            },
        }
    )


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
class Vituri2024ProjectedHamiltonianIdentityGaugeCandidate:
    _factory_token: InitVar[object]
    reference: Vituri2024ProjectedHamiltonianReferenceReceipt
    space: TDHFFullProjectorSpace
    inputs: TDHFScalarFunctionalInputsManifest
    binding: TDHFFullProjectorFunctionalBinding
    source_input_fingerprint: str
    source_projector_full_sha256: str
    source_h0_full_sha256: str
    normal_order_reference_array_sha256: str
    reference_policy_evidence_sha256: str
    q0_background_evidence_sha256: str
    provenance: str
    construction_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION,
        init=False,
    )
    status: str = field(default="candidate_bound_not_executed", init=False)
    authority: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY,
        init=False,
    )
    r0_kernel_constructed: bool = field(default=True, init=False)
    saved_targets_absent_from_callback_inputs: bool = field(default=True, init=False)
    generic_E_F_dF_callbacks_bound: bool = field(default=True, init=False)
    identity_gauge_parity_executed: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    generic_functional_qualification_executed: bool = field(default=False, init=False)
    full_projector_functional_consistency: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_numerical_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _IDENTITY_CANDIDATE_TOKEN:
            raise TypeError("identity-gauge candidate is factory-only")
        self._validate_fields(check_construction=False)
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name
                not in (
                    "reference",
                    "space",
                    "inputs",
                    "binding",
                    "construction_fingerprint",
                )
            }
            | {
                "reference_fingerprint": self.reference.fingerprint,
                "space_fingerprint": self.space.fingerprint,
                "inputs_fingerprint": self.inputs.fingerprint,
                "binding_fingerprint": self.binding.fingerprint,
            }
        )

    def _validate_fields(self, *, check_construction: bool) -> None:
        if type(self.reference) is not Vituri2024ProjectedHamiltonianReferenceReceipt:
            raise TypeError("identity-gauge candidate requires exact R0 receipt")
        if type(self.space) is not TDHFFullProjectorSpace:
            raise TypeError("identity-gauge candidate requires exact full-projector space")
        if self.space.dimension != self.reference.dimension:
            raise ValueError("identity-gauge candidate space/reference dimension drifted")
        if type(self.inputs) is not TDHFScalarFunctionalInputsManifest:
            raise TypeError("identity-gauge candidate requires exact generic inputs")
        if type(self.binding) is not TDHFFullProjectorFunctionalBinding:
            raise TypeError("identity-gauge candidate requires exact generic binding")
        self.reference.validate_live_state()
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        if tuple(item.name for item in self.inputs.entries) != (
            VITURI2024_FULL_PROVIDER_INPUT_NAMES
        ):
            raise ValueError("identity-gauge candidate callback input allowlist drifted")
        if self.binding.fingerprint != make_vituri2024_full_provider_binding().fingerprint:
            raise ValueError("identity-gauge candidate callback binding is not canonical")
        for name in (
            "source_input_fingerprint",
            "source_projector_full_sha256",
            "source_h0_full_sha256",
            "normal_order_reference_array_sha256",
            "reference_policy_evidence_sha256",
            "q0_background_evidence_sha256",
        ):
            _sha256(getattr(self, name), f"identity-gauge candidate {name}")
        checks = (
            (
                self.inputs.value("source_input_fingerprint"),
                self.source_input_fingerprint,
                "source input",
            ),
            (
                _array_sha256(self.inputs.array("source_projector_full")),
                self.source_projector_full_sha256,
                "source projector",
            ),
            (
                _array_sha256(self.inputs.array("h0_full")),
                self.source_h0_full_sha256,
                "source h0",
            ),
            (
                _array_sha256(self.inputs.array("normal_order_reference_full")),
                self.normal_order_reference_array_sha256,
                "R0 matrix",
            ),
            (
                self.normal_order_reference_array_sha256,
                self.reference.normal_order_reference_array_sha256,
                "reference receipt",
            ),
            (
                self.inputs.value("normal_order_reference_fingerprint"),
                self.reference.fingerprint,
                "reference fingerprint",
            ),
            (
                self.inputs.value("reference_policy_evidence_sha256"),
                self.reference_policy_evidence_sha256,
                "reference evidence",
            ),
            (
                self.inputs.value("q0_background_evidence_sha256"),
                self.q0_background_evidence_sha256,
                "q0 evidence",
            ),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(f"identity-gauge candidate {label} drifted")
        locked = (
            self.api_version
            == VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION,
            self.status == "candidate_bound_not_executed",
            self.authority
            == VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY,
            self.r0_kernel_constructed is True,
            self.saved_targets_absent_from_callback_inputs is True,
            self.generic_E_F_dF_callbacks_bound is True,
            self.identity_gauge_parity_executed is False,
            self.source_closure_established is False,
            self.source_stationarity_established is False,
            self.q0_background_authority_established is False,
            self.generic_functional_qualification_executed is False,
            self.full_projector_functional_consistency is False,
            self.tdhf_hessian_match is False,
            self.production_ready is False,
            self.paper_numerical_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("identity-gauge candidate authority inflated")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("identity-gauge candidate provenance must be explicit")
        if check_construction and self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("identity-gauge candidate construction drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_construction=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def build_vituri2024_projected_hamiltonian_identity_gauge_candidate(
    *,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    reference: Vituri2024ProjectedHamiltonianReferenceReceipt,
    area_angstrom_squared: float,
    interaction: InteractionInput,
    q0_policy_fingerprint: str,
    q0_background_evidence_sha256: str,
    provenance: str,
) -> Vituri2024ProjectedHamiltonianIdentityGaugeCandidate:
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("identity-gauge candidate requires exact replay payload")
    if type(reference) is not Vituri2024ProjectedHamiltonianReferenceReceipt:
        raise TypeError("identity-gauge candidate requires exact R0 receipt")
    reference.validate_live_state()
    nk = int(source_payload.mesh.shape[0])
    if reference.nk != nk:
        raise ValueError("identity-gauge candidate reference/payload Nk mismatch")
    source_projector = vituri2024_payload_density_to_full_projector(
        source_payload.projector
    )
    source_h0 = vituri2024_payload_operator_to_full_dense(source_payload.h0)
    mesh_sha256 = _array_sha256(source_payload.mesh)
    space = TDHFFullProjectorSpace(
        dimension=reference.dimension,
        axis_sizes=(len(INTERNAL_FLAVOR_ORDER), nk),
        axis_order=("flavor", "k"),
        orbital_order_fingerprint=canonical_orbital_order_sha256(
            source_payload.mesh
        ),
        layout_adapter_fingerprint=_fingerprint(
            {
                "api": VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION,
                "layout": "flat=flavor*Nk+k",
                "nk": nk,
                "mesh_sha256": mesh_sha256,
            }
        ),
    )
    kernel = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=source_payload.mesh,
        active_band_states=source_payload.active_band_states,
        h0_full=source_h0,
        normal_order_reference=reference.normal_order_reference_full,
        area_angstrom_squared=area_angstrom_squared,
        interaction=interaction,
        normal_order_reference_fingerprint=reference.fingerprint,
        q0_policy_fingerprint=q0_policy_fingerprint,
        source_artifact_sha256=source_payload.source_artifact_sha256,
        provenance=provenance,
    )
    source_input_fingerprint = _target_free_source_input_fingerprint(source_payload)
    inputs = make_vituri2024_full_provider_inputs(
        kernel=kernel,
        source_projector_full=source_projector,
        provider_fingerprint=source_payload.provider_fingerprint,
        source_commit=source_payload.source_commit,
        source_input_fingerprint=source_input_fingerprint,
        reference_policy_evidence_sha256=reference.canonical_equation_text_sha256,
        q0_background_evidence_sha256=q0_background_evidence_sha256,
        provenance=provenance,
    )
    binding = make_vituri2024_full_provider_binding()
    return Vituri2024ProjectedHamiltonianIdentityGaugeCandidate(
        _factory_token=_IDENTITY_CANDIDATE_TOKEN,
        reference=reference,
        space=space,
        inputs=inputs,
        binding=binding,
        source_input_fingerprint=source_input_fingerprint,
        source_projector_full_sha256=_array_sha256(source_projector),
        source_h0_full_sha256=_array_sha256(source_h0),
        normal_order_reference_array_sha256=(
            reference.normal_order_reference_array_sha256
        ),
        reference_policy_evidence_sha256=reference.canonical_equation_text_sha256,
        q0_background_evidence_sha256=q0_background_evidence_sha256,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024ProjectedHamiltonianIdentityGaugeParityReceipt:
    _factory_token: InitVar[object]
    candidate: Vituri2024ProjectedHamiltonianIdentityGaugeCandidate
    guarded_fock_execution: TDHFFullProjectorSingleFockExecutionReceipt
    source_payload_fingerprint: str
    supplied_interaction_h_full: Array
    supplied_fock_full: Array
    supplied_energies: Array
    computed_interaction_h_full: Array
    computed_fock_full: Array
    computed_energies: Array
    supplied_interaction_h_full_sha256: str
    supplied_fock_full_sha256: str
    supplied_energies_sha256: str
    computed_interaction_h_full_sha256: str
    computed_fock_full_sha256: str
    computed_energies_sha256: str
    lambda_interaction_ev: float
    lambda_fock_ev: float
    lambda_energies_ev: float
    lambda_fit_ev: float
    maximum_lambda_consistency_residual_ev: float
    maximum_interaction_identity_quotient_residual_ev: float
    maximum_fock_identity_quotient_residual_ev: float
    maximum_energy_identity_quotient_residual_ev: float
    maximum_supplied_decomposition_residual_ev: float
    maximum_imaginary_identity_trace_ev: float
    tolerance_ev: float
    construction_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION,
        init=False,
    )
    authority: str = field(
        default=VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY,
        init=False,
    )
    paper_active_quartic_R0_semantics_established: bool = field(
        default=True, init=False
    )
    supplied_fock_h0_interaction_decomposition_passed: bool = field(
        default=True, init=False
    )
    single_real_global_identity_fit_passed: bool = field(default=True, init=False)
    selected_R0_fixed_rank_operator_parity_mod_global_identity_passed: bool = field(
        default=True, init=False
    )
    physical_delta_mu_identified: bool = field(default=False, init=False)
    absolute_fock_parity_established: bool = field(default=False, init=False)
    absolute_energy_or_cross_rank_authority_established: bool = field(
        default=False, init=False
    )
    replay_normal_order_source_authority_established: bool = field(
        default=False, init=False
    )
    q0_background_authority_established: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    generic_functional_qualification_executed: bool = field(default=False, init=False)
    source_dF_parity_established: bool = field(default=False, init=False)
    full_projector_functional_consistency: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_numerical_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _IDENTITY_PARITY_TOKEN:
            raise TypeError("identity-gauge parity receipt is factory-only")
        self._validate_fields(check_construction=False)
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        array_names = {
            "supplied_interaction_h_full",
            "supplied_fock_full",
            "supplied_energies",
            "computed_interaction_h_full",
            "computed_fock_full",
            "computed_energies",
        }
        return _fingerprint(
            {
                item.name: (
                    getattr(self, item.name + "_sha256")
                    if item.name in array_names
                    else getattr(self, item.name)
                )
                for item in fields(self)
                if item.name
                not in (
                    "candidate",
                    "guarded_fock_execution",
                    "construction_fingerprint",
                )
            }
            | {
                "candidate_fingerprint": self.candidate.fingerprint,
                "guarded_fock_execution_fingerprint": (
                    self.guarded_fock_execution.fingerprint
                ),
            }
        )

    def _recomputed_metrics(self) -> dict[str, float]:
        dimension = self.computed_fock_full.shape[0]
        identity = np.eye(dimension, dtype=np.complex128)
        interaction_difference = (
            self.supplied_interaction_h_full - self.computed_interaction_h_full
        )
        fock_difference = self.supplied_fock_full - self.computed_fock_full
        energy_difference = self.supplied_energies - self.computed_energies
        lambda_interaction = float(
            np.real(np.trace(interaction_difference)) / dimension
        )
        lambda_fock = float(np.real(np.trace(fock_difference)) / dimension)
        lambda_energies = float(np.mean(energy_difference))
        lambda_fit = 0.5 * (lambda_interaction + lambda_fock)
        imaginary_trace = max(
            abs(float(np.imag(np.trace(interaction_difference)) / dimension)),
            abs(float(np.imag(np.trace(fock_difference)) / dimension)),
        )
        decomposition = (
            self.supplied_fock_full
            - self.candidate.inputs.array("h0_full")
            - self.supplied_interaction_h_full
        )
        return {
            "lambda_interaction_ev": lambda_interaction,
            "lambda_fock_ev": lambda_fock,
            "lambda_energies_ev": lambda_energies,
            "lambda_fit_ev": lambda_fit,
            "maximum_lambda_consistency_residual_ev": max(
                abs(lambda_interaction - lambda_fock),
                abs(lambda_energies - lambda_fit),
            ),
            "maximum_interaction_identity_quotient_residual_ev": float(
                np.max(np.abs(interaction_difference - lambda_fit * identity))
            ),
            "maximum_fock_identity_quotient_residual_ev": float(
                np.max(np.abs(fock_difference - lambda_fit * identity))
            ),
            "maximum_energy_identity_quotient_residual_ev": float(
                np.max(np.abs(energy_difference - lambda_fit))
            ),
            "maximum_supplied_decomposition_residual_ev": float(
                np.max(np.abs(decomposition))
            ),
            "maximum_imaginary_identity_trace_ev": imaginary_trace,
        }

    def _validate_fields(self, *, check_construction: bool) -> None:
        if type(self.candidate) is not Vituri2024ProjectedHamiltonianIdentityGaugeCandidate:
            raise TypeError("identity-gauge parity requires exact candidate")
        if type(self.guarded_fock_execution) is not TDHFFullProjectorSingleFockExecutionReceipt:
            raise TypeError("identity-gauge parity requires guarded Fock execution")
        self.candidate.validate_live_state()
        self.guarded_fock_execution.validate_live_state()
        if (
            self.guarded_fock_execution.space_fingerprint
            != self.candidate.space.fingerprint
            or self.guarded_fock_execution.inputs_fingerprint
            != self.candidate.inputs.fingerprint
            or self.guarded_fock_execution.binding_fingerprint
            != self.candidate.binding.fingerprint
            or self.guarded_fock_execution.projector_fingerprint
            != self.candidate.source_projector_full_sha256
            or self.guarded_fock_execution.fock_fingerprint
            != self.computed_fock_full_sha256
        ):
            raise ValueError("identity-gauge guarded Fock execution binding drifted")
        dimension = self.candidate.reference.dimension
        nk = self.candidate.reference.nk
        arrays = (
            ("supplied_interaction_h_full", np.dtype(np.complex128), (dimension, dimension)),
            ("supplied_fock_full", np.dtype(np.complex128), (dimension, dimension)),
            ("supplied_energies", np.dtype(np.float64), (4, nk)),
            ("computed_interaction_h_full", np.dtype(np.complex128), (dimension, dimension)),
            ("computed_fock_full", np.dtype(np.complex128), (dimension, dimension)),
            ("computed_energies", np.dtype(np.float64), (4, nk)),
        )
        for name, dtype, shape in arrays:
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or value.flags.owndata
                or not np.all(np.isfinite(value))
                or _array_sha256(value) != getattr(self, name + "_sha256")
            ):
                raise ValueError(f"identity-gauge parity array {name} drifted")
        for name in (
            "source_payload_fingerprint",
            "supplied_interaction_h_full_sha256",
            "supplied_fock_full_sha256",
            "supplied_energies_sha256",
            "computed_interaction_h_full_sha256",
            "computed_fock_full_sha256",
            "computed_energies_sha256",
        ):
            _sha256(getattr(self, name), f"identity-gauge parity {name}")
        if self.tolerance_ev != VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV:
            raise ValueError("identity-gauge parity tolerance changed")
        metrics = self._recomputed_metrics()
        for name, actual in metrics.items():
            stored = float(getattr(self, name))
            if not math.isfinite(stored) or abs(stored - actual) > 1.0e-15:
                raise ValueError(f"identity-gauge parity metric {name} drifted")
        gated = tuple(
            metrics[name]
            for name in (
                "maximum_lambda_consistency_residual_ev",
                "maximum_interaction_identity_quotient_residual_ev",
                "maximum_fock_identity_quotient_residual_ev",
                "maximum_energy_identity_quotient_residual_ev",
                "maximum_supplied_decomposition_residual_ev",
                "maximum_imaginary_identity_trace_ev",
            )
        )
        if any(value > self.tolerance_ev for value in gated):
            raise ValueError("identity-gauge parity residual exceeds locked tolerance")
        locked = (
            self.api_version
            == VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION,
            self.authority
            == VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY,
            self.paper_active_quartic_R0_semantics_established is True,
            self.supplied_fock_h0_interaction_decomposition_passed is True,
            self.single_real_global_identity_fit_passed is True,
            self.selected_R0_fixed_rank_operator_parity_mod_global_identity_passed
            is True,
            self.physical_delta_mu_identified is False,
            self.absolute_fock_parity_established is False,
            self.absolute_energy_or_cross_rank_authority_established is False,
            self.replay_normal_order_source_authority_established is False,
            self.q0_background_authority_established is False,
            self.source_closure_established is False,
            self.source_stationarity_established is False,
            self.generic_functional_qualification_executed is False,
            self.source_dF_parity_established is False,
            self.full_projector_functional_consistency is False,
            self.tdhf_hessian_match is False,
            self.production_ready is False,
            self.paper_numerical_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("identity-gauge parity authority inflated")
        if check_construction and self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("identity-gauge parity construction drifted")

    def validate_live_state(self) -> None:
        self._validate_fields(check_construction=True)

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def validate_vituri2024_projected_hamiltonian_identity_gauge_parity(
    *,
    candidate: Vituri2024ProjectedHamiltonianIdentityGaugeCandidate,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
) -> Vituri2024ProjectedHamiltonianIdentityGaugeParityReceipt:
    if type(candidate) is not Vituri2024ProjectedHamiltonianIdentityGaugeCandidate:
        raise TypeError("identity-gauge parity requires exact candidate")
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("identity-gauge parity requires exact replay payload")
    candidate.validate_live_state()
    if _target_free_source_input_fingerprint(source_payload) != (
        candidate.source_input_fingerprint
    ):
        raise ValueError("identity-gauge parity source inputs differ from candidate")
    source_projector = vituri2024_payload_density_to_full_projector(
        source_payload.projector
    )
    source_h0 = vituri2024_payload_operator_to_full_dense(source_payload.h0)
    if (
        _array_sha256(source_projector) != candidate.source_projector_full_sha256
        or _array_sha256(source_h0) != candidate.source_h0_full_sha256
    ):
        raise ValueError("identity-gauge parity source P0/h0 drifted")
    candidate_before = candidate.fingerprint
    guarded_fock_execution = execute_tdhf_full_projector_fock_once(
        space=candidate.space,
        inputs=candidate.inputs,
        binding=candidate.binding,
        projector=source_projector,
        provenance=(
            "Vituri selected-R0 identity-gauge parity: exactly one guarded F "
            "execution; no E/dF, generic qualification, TDHF, or authority promotion."
        ),
    )
    computed_fock_raw = guarded_fock_execution.fock
    if candidate.fingerprint != candidate_before:
        raise ValueError("identity-gauge callback mutated candidate state")
    dimension = candidate.reference.dimension
    computed_fock = _readonly_hermitian(
        np.asarray(computed_fock_raw, dtype=np.complex128),
        dimension=dimension,
        label="identity-gauge computed Fock",
    )
    computed_interaction = _readonly_hermitian(
        np.asarray(computed_fock - source_h0, dtype=np.complex128),
        dimension=dimension,
        label="identity-gauge computed interaction_h",
    )
    supplied_interaction = _readonly_hermitian(
        vituri2024_payload_operator_to_full_dense(source_payload.interaction_h),
        dimension=dimension,
        label="identity-gauge supplied interaction_h",
    )
    supplied_fock = _readonly_hermitian(
        vituri2024_payload_operator_to_full_dense(source_payload.fock),
        dimension=dimension,
        label="identity-gauge supplied Fock",
    )
    supplied_energies = _readonly_array(
        source_payload.energies,
        dtype=np.dtype(np.float64),
        shape=(4, candidate.reference.nk),
        label="identity-gauge supplied energies",
    )
    diagonal = np.diag(computed_fock)
    if np.max(np.abs(np.imag(diagonal))) > (
        VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV
    ):
        raise ValueError("identity-gauge computed Fock diagonal is materially complex")
    computed_energies = _readonly_array(
        np.asarray(np.real(diagonal), dtype=np.float64).reshape(
            4, candidate.reference.nk
        ),
        dtype=np.dtype(np.float64),
        shape=(4, candidate.reference.nk),
        label="identity-gauge computed energies",
    )
    temporary = {
        "candidate": candidate,
        "guarded_fock_execution": guarded_fock_execution,
        "source_payload_fingerprint": _target_payload_fingerprint(source_payload),
        "supplied_interaction_h_full": supplied_interaction,
        "supplied_fock_full": supplied_fock,
        "supplied_energies": supplied_energies,
        "computed_interaction_h_full": computed_interaction,
        "computed_fock_full": computed_fock,
        "computed_energies": computed_energies,
    }
    dimension_float = float(dimension)
    interaction_difference = supplied_interaction - computed_interaction
    fock_difference = supplied_fock - computed_fock
    energy_difference = supplied_energies - computed_energies
    lambda_interaction = float(np.real(np.trace(interaction_difference)) / dimension_float)
    lambda_fock = float(np.real(np.trace(fock_difference)) / dimension_float)
    lambda_energies = float(np.mean(energy_difference))
    lambda_fit = 0.5 * (lambda_interaction + lambda_fock)
    identity = np.eye(dimension, dtype=np.complex128)
    metrics = {
        "lambda_interaction_ev": lambda_interaction,
        "lambda_fock_ev": lambda_fock,
        "lambda_energies_ev": lambda_energies,
        "lambda_fit_ev": lambda_fit,
        "maximum_lambda_consistency_residual_ev": max(
            abs(lambda_interaction - lambda_fock),
            abs(lambda_energies - lambda_fit),
        ),
        "maximum_interaction_identity_quotient_residual_ev": float(
            np.max(np.abs(interaction_difference - lambda_fit * identity))
        ),
        "maximum_fock_identity_quotient_residual_ev": float(
            np.max(np.abs(fock_difference - lambda_fit * identity))
        ),
        "maximum_energy_identity_quotient_residual_ev": float(
            np.max(np.abs(energy_difference - lambda_fit))
        ),
        "maximum_supplied_decomposition_residual_ev": float(
            np.max(np.abs(supplied_fock - source_h0 - supplied_interaction))
        ),
        "maximum_imaginary_identity_trace_ev": max(
            abs(float(np.imag(np.trace(interaction_difference)) / dimension_float)),
            abs(float(np.imag(np.trace(fock_difference)) / dimension_float)),
        ),
    }
    tolerance = VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV
    if any(
        metrics[name] > tolerance
        for name in (
            "maximum_lambda_consistency_residual_ev",
            "maximum_interaction_identity_quotient_residual_ev",
            "maximum_fock_identity_quotient_residual_ev",
            "maximum_energy_identity_quotient_residual_ev",
            "maximum_supplied_decomposition_residual_ev",
            "maximum_imaginary_identity_trace_ev",
        )
    ):
        raise ValueError("identity-gauge parity failed locked residual gates")
    return Vituri2024ProjectedHamiltonianIdentityGaugeParityReceipt(
        _factory_token=_IDENTITY_PARITY_TOKEN,
        **temporary,
        supplied_interaction_h_full_sha256=_array_sha256(supplied_interaction),
        supplied_fock_full_sha256=_array_sha256(supplied_fock),
        supplied_energies_sha256=_array_sha256(supplied_energies),
        computed_interaction_h_full_sha256=_array_sha256(computed_interaction),
        computed_fock_full_sha256=_array_sha256(computed_fock),
        computed_energies_sha256=_array_sha256(computed_energies),
        **metrics,
        tolerance_ev=tolerance,
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
    "VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_API_VERSION",
    "VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_AUTHORITY",
    "VITURI2024_PROJECTED_HAMILTONIAN_IDENTITY_GAUGE_TOLERANCE_EV",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_API_VERSION",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_AUTHORITY",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_DENSITY_CONVENTION",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_LOCATOR",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_MU_STATUS",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_Q0_STATUS",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT",
    "VITURI2024_PROJECTED_HAMILTONIAN_REFERENCE_TEXT_SHA256",
    "Vituri2024ProjectedHamiltonianIdentityGaugeCandidate",
    "Vituri2024ProjectedHamiltonianIdentityGaugeParityReceipt",
    "Vituri2024ProjectedHamiltonianReferenceBridge",
    "Vituri2024ProjectedHamiltonianReferenceReceipt",
    "build_vituri2024_full_functional_replay_bridge_from_projected_hamiltonian_reference",
    "build_vituri2024_projected_hamiltonian_identity_gauge_candidate",
    "make_vituri2024_projected_hamiltonian_zero_reference",
    "validate_vituri2024_projected_hamiltonian_identity_gauge_parity",
]
