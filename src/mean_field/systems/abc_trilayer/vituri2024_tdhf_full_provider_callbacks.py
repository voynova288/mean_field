"""Generic full-projector callback adapter for the Vituri factorized functional.

The callbacks are separate plain functions with no defaults or closures, as
required by ``core.hf.tdhf_scalar_functional``.  They consume only immutable
manifest arrays and call the factorized projected-Hamiltonian action.  Saved
replay Fock/interaction targets are intentionally absent from callback inputs.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Final

import numpy as np

from mean_field.core.hf.tdhf_scalar_functional import (
    TDHFFullProjectorFunctionalBinding,
    TDHFScalarFunctionalInputsManifest,
    bind_tdhf_scalar_kernel,
    make_tdhf_scalar_functional_inputs_manifest,
)

from .vituri2024_interaction import Vituri2024InteractionChoiceReceipt
from .vituri2024_rpa import vituri2024_rpa_a_element, vituri2024_rpa_b_element
from .vituri2024_tdhf import assemble_vituri2024_tdhf_signed_q
from .vituri2024_tdhf_full_functional import (
    Vituri2024FullProjectedFunctionalKernel,
    vituri2024_full_projected_interaction_action,
)

Array = np.ndarray

VITURI2024_FULL_PROVIDER_CALLBACK_API_VERSION: Final[str] = (
    "vituri2024_full_provider_callbacks.v1"
)
VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND: Final[str] = (
    "analytic_kernel_limit_only_no_hf_background_authority"
)
VITURI2024_FULL_PROVIDER_INPUT_NAMES: Final[tuple[str, ...]] = (
    "active_band_states",
    "area_angstrom_squared",
    "exact_local_mask",
    "execution_input_fingerprint",
    "form_factors_by_flavor",
    "h0_full",
    "interaction_authority_kind",
    "interaction_coulomb_e2_ev_angstrom",
    "interaction_fingerprint",
    "interaction_gate_distance_angstrom",
    "interaction_kernel_by_mesh_pair",
    "interaction_provider_sha256",
    "interaction_q0_evaluation",
    "interaction_source_sha256",
    "interaction_source_text",
    "kernel_provenance",
    "normal_order_reference_array_sha256",
    "normal_order_reference_fingerprint",
    "normal_order_reference_full",
    "ordered_mesh",
    "provider_fingerprint",
    "q0_background_action_kind",
    "q0_background_evidence_sha256",
    "q0_policy_fingerprint",
    "reference_policy_evidence_sha256",
    "source_commit",
    "source_input_fingerprint",
    "source_projector_full",
)


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


def _execution_input_fingerprint(values: dict[str, object]) -> str:
    stable = {
        name: (
            {
                "dtype": str(value.dtype),
                "shape": value.shape,
                "sha256": _array_sha256(value),
            }
            if isinstance(value, np.ndarray)
            else value
        )
        for name, value in sorted(values.items())
    }
    return sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False
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


def _readonly_hermitian(value: object, label: str, dimension: int) -> Array:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{label} must be an exact complex128 numpy array")
    if value.shape != (dimension, dimension) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite with shape {(dimension, dimension)}")
    residual = float(np.max(np.abs(value - value.conj().T)))
    scale = max(1.0, float(np.max(np.abs(value))))
    if residual > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} must be Hermitian")
    result = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(
        value.shape
    )
    result.setflags(write=False)
    return result


def _execution_inputs(
    inputs: TDHFScalarFunctionalInputsManifest,
) -> tuple[Array, Array, Array, Array, Array, float, int]:
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("Vituri callbacks require exact generic functional inputs")
    inputs.validate_live_state()
    names = tuple(item.name for item in inputs.entries)
    if names != VITURI2024_FULL_PROVIDER_INPUT_NAMES:
        raise ValueError("Vituri full-provider callback input allowlist changed")
    form_factors = inputs.array("form_factors_by_flavor")
    if form_factors.ndim != 3 or form_factors.shape[0] != 4:
        raise ValueError("callback form factors must have shape (4,Nk,Nk)")
    nk = int(form_factors.shape[1])
    dimension = 4 * nk
    h0 = _readonly_hermitian(inputs.array("h0_full"), "callback h0", dimension)
    reference = _readonly_hermitian(
        inputs.array("normal_order_reference_full"),
        "callback normal-order reference",
        dimension,
    )
    source_projector = _readonly_hermitian(
        inputs.array("source_projector_full"),
        "callback source projector",
        dimension,
    )
    if _array_sha256(reference) != inputs.value(
        "normal_order_reference_array_sha256"
    ):
        raise ValueError("callback normal-reference bytes/fingerprint mismatch")
    if np.max(np.abs(source_projector @ source_projector - source_projector)) > 5.0e-10:
        raise ValueError("callback source projector is not idempotent")
    area_value = inputs.value("area_angstrom_squared")
    if isinstance(area_value, (bool, np.bool_)):
        raise TypeError("callback area must be a strict real scalar")
    area = float(area_value)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("callback area must be finite and positive")
    if inputs.value("q0_background_action_kind") != VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND:
        raise ValueError("callback q0 action kind is unsupported")
    for name in (
        "execution_input_fingerprint",
        "interaction_fingerprint",
        "interaction_provider_sha256",
        "interaction_source_sha256",
        "normal_order_reference_fingerprint",
        "provider_fingerprint",
        "q0_background_evidence_sha256",
        "q0_policy_fingerprint",
        "reference_policy_evidence_sha256",
        "source_input_fingerprint",
    ):
        _sha256(inputs.value(name), f"callback input {name}")
    if inputs.value("execution_input_fingerprint") != inputs.source_fingerprint:
        raise ValueError("callback execution/source fingerprint mismatch")
    return (
        h0,
        reference,
        form_factors,
        inputs.array("interaction_kernel_by_mesh_pair"),
        inputs.array("exact_local_mask"),
        area,
        dimension,
    )


def vituri2024_full_provider_energy(inputs, P):
    h0, reference, form_factors, interaction_kernel, mask, area, _dimension = (
        _execution_inputs(inputs)
    )
    clean = _readonly_hermitian(P, "callback conventional P", h0.shape[0])
    difference = clean - reference
    interaction = vituri2024_full_projected_interaction_action(
        difference,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction_kernel,
        exact_local_mask=mask,
        area_angstrom_squared=area,
    )
    total = complex(np.einsum("ij,ji->", h0, clean, optimize=False))
    total += 0.5 * complex(
        np.einsum("ij,ji->", difference, interaction, optimize=False)
    )
    if abs(total.imag) > 5.0e-11 * max(1.0, abs(total)):
        raise ValueError("callback scalar energy has a material imaginary part")
    return float(total.real)


def vituri2024_full_provider_fock(inputs, P):
    h0, reference, form_factors, interaction_kernel, mask, area, _dimension = (
        _execution_inputs(inputs)
    )
    clean = _readonly_hermitian(P, "callback conventional P", h0.shape[0])
    interaction = vituri2024_full_projected_interaction_action(
        clean - reference,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction_kernel,
        exact_local_mask=mask,
        area_angstrom_squared=area,
    )
    return np.asarray(h0 + interaction, dtype=np.complex128)


def vituri2024_full_provider_fock_derivative(inputs, P, D):
    h0, _reference, form_factors, interaction_kernel, mask, area, dimension = (
        _execution_inputs(inputs)
    )
    _readonly_hermitian(P, "callback conventional P anchor", dimension)
    clean_direction = _readonly_hermitian(
        D, "callback conventional Hermitian D", dimension
    )
    del h0
    return np.asarray(
        vituri2024_full_projected_interaction_action(
            clean_direction,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=interaction_kernel,
            exact_local_mask=mask,
            area_angstrom_squared=area,
        ),
        dtype=np.complex128,
    )


def make_vituri2024_full_provider_inputs(
    *,
    kernel: Vituri2024FullProjectedFunctionalKernel,
    source_projector_full: Array,
    provider_fingerprint: str,
    source_commit: str,
    source_input_fingerprint: str,
    reference_policy_evidence_sha256: str,
    q0_background_evidence_sha256: str,
    provenance: str,
) -> TDHFScalarFunctionalInputsManifest:
    if type(kernel) is not Vituri2024FullProjectedFunctionalKernel:
        raise TypeError("provider inputs require an exact Vituri functional kernel")
    kernel.validate_live_state()
    source_projector = _readonly_hermitian(
        source_projector_full,
        "provider conventional source projector",
        kernel.dimension,
    )
    if np.max(np.abs(source_projector @ source_projector - source_projector)) > 5.0e-10:
        raise ValueError("provider conventional source projector is not idempotent")
    for value, label in (
        (provider_fingerprint, "provider fingerprint"),
        (source_input_fingerprint, "source-input fingerprint"),
        (reference_policy_evidence_sha256, "reference-policy evidence"),
        (q0_background_evidence_sha256, "q0-background evidence"),
    ):
        _sha256(value, label)
    if (
        type(source_commit) is not str
        or len(source_commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("provider source commit must be a lowercase commit digest")
    interaction: Vituri2024InteractionChoiceReceipt = kernel.interaction_receipt
    values = {
        "active_band_states": kernel.active_band_states,
        "area_angstrom_squared": kernel.area_angstrom_squared,
        "exact_local_mask": kernel.exact_local_mask,
        "form_factors_by_flavor": kernel.form_factors_by_flavor,
        "h0_full": kernel.h0_full,
        "interaction_authority_kind": interaction.authority_kind,
        "interaction_coulomb_e2_ev_angstrom": interaction.coulomb_e2_ev_angstrom,
        "interaction_fingerprint": interaction.fingerprint,
        "interaction_gate_distance_angstrom": interaction.gate_distance_angstrom,
        "interaction_kernel_by_mesh_pair": kernel.kernel_by_mesh_pair,
        "interaction_provider_sha256": interaction.provider_sha256,
        "interaction_q0_evaluation": interaction.q0_evaluation,
        "interaction_source_sha256": interaction.source_sha256,
        "interaction_source_text": interaction.source_text,
        "kernel_provenance": kernel.provenance,
        "normal_order_reference_array_sha256": _array_sha256(
            kernel.normal_order_reference
        ),
        "normal_order_reference_fingerprint": (
            kernel.normal_order_reference_fingerprint
        ),
        "normal_order_reference_full": kernel.normal_order_reference,
        "ordered_mesh": kernel.ordered_mesh,
        "provider_fingerprint": provider_fingerprint,
        "q0_background_action_kind": VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND,
        "q0_background_evidence_sha256": q0_background_evidence_sha256,
        "q0_policy_fingerprint": kernel.q0_policy_fingerprint,
        "reference_policy_evidence_sha256": reference_policy_evidence_sha256,
        "source_commit": source_commit,
        "source_input_fingerprint": source_input_fingerprint,
        "source_projector_full": source_projector,
    }
    execution_fingerprint = _execution_input_fingerprint(values)
    values["execution_input_fingerprint"] = execution_fingerprint
    if tuple(sorted(values)) != VITURI2024_FULL_PROVIDER_INPUT_NAMES:
        raise RuntimeError("provider input implementation/allowlist drifted")
    return make_tdhf_scalar_functional_inputs_manifest(
        values,
        source_fingerprint=execution_fingerprint,
        provenance=provenance,
    )


def make_vituri2024_full_provider_binding() -> TDHFFullProjectorFunctionalBinding:
    dependencies = (
        _execution_inputs,
        vituri2024_full_projected_interaction_action,
    )
    return TDHFFullProjectorFunctionalBinding(
        energy=bind_tdhf_scalar_kernel(
            role="energy",
            callback=vituri2024_full_provider_energy,
            dependencies=dependencies,
            provenance="Vituri projected-H factorized raw-total energy callback.",
        ),
        fock=bind_tdhf_scalar_kernel(
            role="fock",
            callback=vituri2024_full_provider_fock,
            dependencies=dependencies,
            provenance="Vituri projected-H factorized full Fock callback.",
        ),
        fock_derivative=bind_tdhf_scalar_kernel(
            role="fock_derivative",
            callback=vituri2024_full_provider_fock_derivative,
            dependencies=dependencies,
            provenance="Vituri projected-H factorized affine dF callback.",
        ),
        forbidden_entrypoints=(
            vituri2024_rpa_a_element,
            vituri2024_rpa_b_element,
            assemble_vituri2024_tdhf_signed_q,
        ),
    )


__all__ = [
    "VITURI2024_FULL_PROVIDER_CALLBACK_API_VERSION",
    "VITURI2024_FULL_PROVIDER_INPUT_NAMES",
    "VITURI2024_FULL_PROVIDER_Q0_ACTION_KIND",
    "make_vituri2024_full_provider_binding",
    "make_vituri2024_full_provider_inputs",
    "vituri2024_full_provider_energy",
    "vituri2024_full_provider_fock",
    "vituri2024_full_provider_fock_derivative",
]
