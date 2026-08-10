"""Fail-closed replay bridge for the Vituri factorized full functional.

This bridge binds the exact replay P0/h0/interaction_h/Fock arrays to one
explicit caller-supplied normal reference and the actual projected-Hamiltonian
kernel.  It constructs generic E/F/dF callback inputs without exposing replay
interaction_h or Fock as callback targets.  Success is replay-array
consistency only: the replay schema does not contain the authoritative R matrix
or an executable HF q=0 background prescription.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
from typing import Final

import numpy as np

from mean_field.core.hf.tdhf_scalar_functional import (
    TDHFFullProjectorFunctionalBinding,
    TDHFScalarFunctionalInputsManifest,
)

from .vituri2024_hf_replay import Vituri2024HalfMetalHFReplayPayload
from .vituri2024_interaction import (
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
from .vituri2024_tdhf_full_functional import (
    Vituri2024FullProjectedFunctionalKernel,
    Vituri2024FullProjectedSuppliedArrayConsistencyReceipt,
    validate_vituri2024_full_projected_supplied_arrays,
)
from .vituri2024_tdhf_full_provider_callbacks import (
    VITURI2024_FULL_PROVIDER_INPUT_NAMES,
    VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND,
    make_vituri2024_full_provider_binding,
    make_vituri2024_full_provider_inputs,
)
from .vituri2024_tdhf_full_scalar import (
    vituri2024_payload_operator_to_full_dense,
    vituri2024_tdhf_full_scalar_source_from_payload,
)

Array = np.ndarray
InteractionInput = Vituri2024InteractionChoiceReceipt | Vituri2024InteractionBinding

VITURI2024_FULL_PROVIDER_BRIDGE_API_VERSION: Final[str] = (
    "vituri2024_full_provider_replay_bridge.v1"
)
VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY: Final[str] = (
    "replay_array_bound_factorized_provider_candidate_only_no_source_q0_normal_order_"
    "tdhf_scalar_hessian_production_or_paper_authority"
)
VITURI2024_FULL_PROVIDER_BRIDGE_STATUS: Final[str] = (
    "factorized_callbacks_bound_replay_arrays_consistent_not_source_qualified"
)

_BRIDGE_TOKEN = object()


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


def _stable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "sha256": _array_sha256(value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _stable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported bridge fingerprint type {type(value).__name__}")


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            _stable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _payload_fingerprint(payload: Vituri2024HalfMetalHFReplayPayload) -> str:
    return _fingerprint(
        {
            item.name: (
                _array_sha256(getattr(payload, item.name))
                if isinstance(getattr(payload, item.name), np.ndarray)
                else getattr(payload, item.name)
            )
            for item in fields(payload)
        }
    )


def _source_input_fingerprint(payload: Vituri2024HalfMetalHFReplayPayload) -> str:
    """Bind callback source inputs while excluding replay target arrays."""

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


@dataclass(frozen=True, slots=True)
class Vituri2024FullFunctionalReplayBridge:
    _factory_token: InitVar[object]
    kernel: Vituri2024FullProjectedFunctionalKernel
    inputs: TDHFScalarFunctionalInputsManifest
    binding: TDHFFullProjectorFunctionalBinding
    array_consistency: Vituri2024FullProjectedSuppliedArrayConsistencyReceipt
    source_payload_fingerprint: str
    source_input_fingerprint: str
    source_projector_full_sha256: str
    source_h0_full_sha256: str
    supplied_interaction_h_full_sha256: str
    supplied_fock_full_sha256: str
    normal_order_reference_array_sha256: str
    reference_policy_evidence_sha256: str
    q0_background_evidence_sha256: str
    provenance: str
    construction_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_FULL_PROVIDER_BRIDGE_API_VERSION, init=False
    )
    status: str = field(default=VITURI2024_FULL_PROVIDER_BRIDGE_STATUS, init=False)
    authority: str = field(
        default=VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY, init=False
    )
    generic_callbacks_bound: bool = field(default=True, init=False)
    replay_array_consistency_passed: bool = field(default=True, init=False)
    generic_validation_executed: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    normal_order_authority_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    full_projector_functional_consistency: bool = field(default=False, init=False)
    tdhf_hessian_match: bool = field(default=False, init=False)
    scalar_hessian_authority_promoted: bool = field(default=False, init=False)
    eligible_for_slurm_qualification: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BRIDGE_TOKEN:
            raise TypeError("replay bridge construction is factory-only")
        if type(self.kernel) is not Vituri2024FullProjectedFunctionalKernel:
            raise TypeError("bridge requires an exact factorized Vituri kernel")
        if type(self.inputs) is not TDHFScalarFunctionalInputsManifest:
            raise TypeError("bridge requires exact generic callback inputs")
        if type(self.binding) is not TDHFFullProjectorFunctionalBinding:
            raise TypeError("bridge requires exact generic callback binding")
        if (
            type(self.array_consistency)
            is not Vituri2024FullProjectedSuppliedArrayConsistencyReceipt
            or self.array_consistency.passed is not True
        ):
            raise TypeError("bridge requires a passing supplied-array receipt")
        for name in (
            "source_payload_fingerprint",
            "source_input_fingerprint",
            "source_projector_full_sha256",
            "source_h0_full_sha256",
            "supplied_interaction_h_full_sha256",
            "supplied_fock_full_sha256",
            "normal_order_reference_array_sha256",
            "reference_policy_evidence_sha256",
            "q0_background_evidence_sha256",
        ):
            _sha256(getattr(self, name), f"bridge {name}")
        self.kernel.validate_live_state()
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        if tuple(item.name for item in self.inputs.entries) != (
            VITURI2024_FULL_PROVIDER_INPUT_NAMES
        ):
            raise ValueError("bridge callback input allowlist changed")
        if self.inputs.value("execution_input_fingerprint") != (
            self.inputs.source_fingerprint
        ):
            raise ValueError("bridge execution/input fingerprint mismatch")
        if self.inputs.value("q0_background_action_kind") != (
            VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND
        ):
            raise ValueError("bridge q0 action kind changed")
        if self.array_consistency.kernel_fingerprint != self.kernel.fingerprint:
            raise ValueError("bridge supplied-array receipt is stale")
        receipt_bindings = (
            (
                self.array_consistency.source_projector_sha256,
                self.source_projector_full_sha256,
                "source projector",
            ),
            (
                self.array_consistency.supplied_interaction_h_sha256,
                self.supplied_interaction_h_full_sha256,
                "supplied interaction_h",
            ),
            (
                self.array_consistency.supplied_fock_sha256,
                self.supplied_fock_full_sha256,
                "supplied Fock",
            ),
        )
        for actual, expected, label in receipt_bindings:
            if actual != expected:
                raise ValueError(f"bridge receipt/{label} fingerprint mismatch")
        input_source = self.inputs.array("source_projector_full")
        input_h0 = self.inputs.array("h0_full")
        input_reference = self.inputs.array("normal_order_reference_full")
        if _array_sha256(input_source) != self.source_projector_full_sha256:
            raise ValueError("bridge source projector bytes drifted")
        if _array_sha256(input_h0) != self.source_h0_full_sha256:
            raise ValueError("bridge source h0 bytes drifted")
        if _array_sha256(input_reference) != self.normal_order_reference_array_sha256:
            raise ValueError("bridge normal-reference bytes drifted")
        if self.inputs.value("source_input_fingerprint") != (
            self.source_input_fingerprint
        ):
            raise ValueError("bridge source-input fingerprint drifted")
        if self.inputs.source_fingerprint != self.inputs.value(
            "execution_input_fingerprint"
        ):
            raise ValueError("bridge callback execution fingerprint drifted")
        array_bindings = (
            ("ordered_mesh", self.kernel.ordered_mesh),
            ("active_band_states", self.kernel.active_band_states),
            ("form_factors_by_flavor", self.kernel.form_factors_by_flavor),
            (
                "interaction_kernel_by_mesh_pair",
                self.kernel.kernel_by_mesh_pair,
            ),
            ("exact_local_mask", self.kernel.exact_local_mask),
        )
        for name, expected in array_bindings:
            if _array_sha256(self.inputs.array(name)) != _array_sha256(expected):
                raise ValueError(f"bridge input {name!r} differs from kernel")
        interaction = self.kernel.interaction_receipt
        scalar_bindings = (
            (
                self.inputs.value("area_angstrom_squared"),
                self.kernel.area_angstrom_squared,
                "area",
            ),
            (
                self.inputs.value("interaction_fingerprint"),
                interaction.fingerprint,
                "interaction fingerprint",
            ),
            (
                self.inputs.value("interaction_gate_distance_angstrom"),
                interaction.gate_distance_angstrom,
                "interaction gate distance",
            ),
            (
                self.inputs.value("interaction_coulomb_e2_ev_angstrom"),
                interaction.coulomb_e2_ev_angstrom,
                "interaction Coulomb prefactor",
            ),
            (
                self.inputs.value("interaction_q0_evaluation"),
                interaction.q0_evaluation,
                "interaction q0 evaluation",
            ),
            (
                self.inputs.value("interaction_provider_sha256"),
                interaction.provider_sha256,
                "interaction provider",
            ),
            (
                self.inputs.value("interaction_source_sha256"),
                interaction.source_sha256,
                "interaction source",
            ),
            (
                self.inputs.value("interaction_authority_kind"),
                interaction.authority_kind,
                "interaction authority kind",
            ),
            (
                self.inputs.value("interaction_source_text"),
                interaction.source_text,
                "interaction source text",
            ),
            (
                self.inputs.value("normal_order_reference_fingerprint"),
                self.kernel.normal_order_reference_fingerprint,
                "normal-reference policy",
            ),
            (
                self.inputs.value("q0_policy_fingerprint"),
                self.kernel.q0_policy_fingerprint,
                "q0 policy",
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
        for actual, expected, label in scalar_bindings:
            if actual != expected:
                raise ValueError(f"bridge input {label} drifted")
        if self.binding.fingerprint != make_vituri2024_full_provider_binding().fingerprint:
            raise ValueError("bridge callback binding differs from canonical provider binding")
        callback_names = tuple(item.name.lower() for item in self.inputs.entries)
        if any(
            name in callback_names
            for name in ("source_fock", "saved_fock", "interaction_h", "target_h")
        ):
            raise ValueError("bridge callback inputs expose replay target arrays")
        locked = (
            self.api_version == VITURI2024_FULL_PROVIDER_BRIDGE_API_VERSION,
            self.status == VITURI2024_FULL_PROVIDER_BRIDGE_STATUS,
            self.authority == VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY,
            self.generic_callbacks_bound is True,
            self.replay_array_consistency_passed is True,
            self.generic_validation_executed is False,
            self.source_closure_established is False,
            self.source_stationarity_established is False,
            self.normal_order_authority_established is False,
            self.q0_background_authority_established is False,
            self.full_projector_functional_consistency is False,
            self.tdhf_hessian_match is False,
            self.scalar_hessian_authority_promoted is False,
            self.eligible_for_slurm_qualification is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("bridge scope or authority was inflated")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("bridge provenance must be explicit nonempty text")
        object.__setattr__(
            self, "construction_fingerprint", self._current_fingerprint()
        )

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name
                not in (
                    "kernel",
                    "inputs",
                    "binding",
                    "array_consistency",
                    "construction_fingerprint",
                )
            }
            | {
                "kernel_fingerprint": self.kernel.fingerprint,
                "inputs_fingerprint": self.inputs.fingerprint,
                "binding_fingerprint": self.binding.fingerprint,
                "array_consistency_fingerprint": self.array_consistency.fingerprint,
            }
        )

    def validate_live_state(self) -> None:
        self.kernel.validate_live_state()
        self.inputs.validate_live_state()
        self.binding.validate_live_state()
        if self.binding.fingerprint != make_vituri2024_full_provider_binding().fingerprint:
            raise ValueError("bridge live callback binding is not canonical")
        if self.array_consistency.kernel_fingerprint != self.kernel.fingerprint:
            raise ValueError("bridge live supplied-array receipt is stale")
        if (
            self.array_consistency.source_projector_sha256
            != self.source_projector_full_sha256
            or self.array_consistency.supplied_interaction_h_sha256
            != self.supplied_interaction_h_full_sha256
            or self.array_consistency.supplied_fock_sha256
            != self.supplied_fock_full_sha256
        ):
            raise ValueError("bridge live receipt/source target hashes drifted")
        if self._current_fingerprint() != self.construction_fingerprint:
            raise ValueError("bridge live construction fingerprint drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint


def build_vituri2024_full_functional_replay_bridge(
    *,
    source_payload: Vituri2024HalfMetalHFReplayPayload,
    normal_order_reference_full: Array,
    area_angstrom_squared: float,
    interaction: InteractionInput,
    normal_order_reference_fingerprint: str,
    reference_policy_evidence_sha256: str,
    q0_policy_fingerprint: str,
    q0_background_evidence_sha256: str,
    provenance: str,
) -> Vituri2024FullFunctionalReplayBridge:
    if type(source_payload) is not Vituri2024HalfMetalHFReplayPayload:
        raise TypeError("replay bridge requires an exact Vituri replay payload")
    for value, label in (
        (normal_order_reference_fingerprint, "normal-order reference fingerprint"),
        (reference_policy_evidence_sha256, "reference-policy evidence"),
        (q0_policy_fingerprint, "q0 policy fingerprint"),
        (q0_background_evidence_sha256, "q0-background evidence"),
    ):
        _sha256(value, label)
    source = vituri2024_tdhf_full_scalar_source_from_payload(source_payload)
    payload_fingerprint = _payload_fingerprint(source_payload)
    source_input_fingerprint = _source_input_fingerprint(source_payload)
    kernel = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=source_payload.mesh,
        active_band_states=source_payload.active_band_states,
        h0_full=source.source_h0,
        normal_order_reference=normal_order_reference_full,
        area_angstrom_squared=area_angstrom_squared,
        interaction=interaction,
        normal_order_reference_fingerprint=normal_order_reference_fingerprint,
        q0_policy_fingerprint=q0_policy_fingerprint,
        source_artifact_sha256=source_payload.source_artifact_sha256,
        provenance=provenance,
    )
    supplied_interaction_h = vituri2024_payload_operator_to_full_dense(
        source_payload.interaction_h
    )
    supplied_fock = vituri2024_payload_operator_to_full_dense(source_payload.fock)
    array_consistency = validate_vituri2024_full_projected_supplied_arrays(
        kernel=kernel,
        source_projector=source.source_projector,
        supplied_interaction_h=supplied_interaction_h,
        supplied_fock=supplied_fock,
    )
    inputs = make_vituri2024_full_provider_inputs(
        kernel=kernel,
        source_projector_full=source.source_projector,
        provider_fingerprint=source_payload.provider_fingerprint,
        source_commit=source_payload.source_commit,
        source_input_fingerprint=source_input_fingerprint,
        reference_policy_evidence_sha256=reference_policy_evidence_sha256,
        q0_background_evidence_sha256=q0_background_evidence_sha256,
        provenance=provenance,
    )
    binding = make_vituri2024_full_provider_binding()
    return Vituri2024FullFunctionalReplayBridge(
        _factory_token=_BRIDGE_TOKEN,
        kernel=kernel,
        inputs=inputs,
        binding=binding,
        array_consistency=array_consistency,
        source_payload_fingerprint=payload_fingerprint,
        source_input_fingerprint=source_input_fingerprint,
        source_projector_full_sha256=_array_sha256(source.source_projector),
        source_h0_full_sha256=_array_sha256(source.source_h0),
        supplied_interaction_h_full_sha256=_array_sha256(supplied_interaction_h),
        supplied_fock_full_sha256=_array_sha256(supplied_fock),
        normal_order_reference_array_sha256=_array_sha256(
            kernel.normal_order_reference
        ),
        reference_policy_evidence_sha256=reference_policy_evidence_sha256,
        q0_background_evidence_sha256=q0_background_evidence_sha256,
        provenance=provenance,
    )


__all__ = [
    "VITURI2024_FULL_PROVIDER_BRIDGE_API_VERSION",
    "VITURI2024_FULL_PROVIDER_BRIDGE_AUTHORITY",
    "VITURI2024_FULL_PROVIDER_BRIDGE_STATUS",
    "Vituri2024FullFunctionalReplayBridge",
    "build_vituri2024_full_functional_replay_bridge",
]
