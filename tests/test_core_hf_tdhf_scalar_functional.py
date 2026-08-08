"""Public-path tests for the dense conventional scalar-functional ABI."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from types import FunctionType

import numpy as np
import pytest

from mean_field.core.hf import (
    TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM,
    TDHFFullProjectorDirection,
    TDHFFullProjectorFunctionalBinding,
    TDHFFullProjectorSpace,
    TDHFFullProjectorValidationPlan,
    TDHFFullProjectorValidationTolerances,
    bind_tdhf_scalar_kernel,
    deterministic_complete_hermitian_basis,
    make_tdhf_full_projector_functional_approval,
    make_tdhf_full_projector_unitary_probe,
    make_tdhf_scalar_functional_inputs_manifest,
    validate_tdhf_full_projector_functional,
)


def _linear_action(inputs, P):
    basis = inputs.array("basis")
    kernel = inputs.array("kernel")
    coordinates = np.real(np.einsum("aij,ji->a", basis, P, optimize=False))
    return np.einsum(
        "ab,b,aij->ij", kernel, coordinates, basis, optimize=False
    ).astype(np.complex128)


def _quadratic_energy(inputs, P):
    coordinates = np.real(
        np.einsum("aij,ji->a", inputs.array("basis"), P, optimize=False)
    )
    one_body = np.einsum("ij,ji->", inputs.array("h0"), P, optimize=False)
    interaction = coordinates @ inputs.array("kernel") @ coordinates
    return float(np.real(one_body + 0.5 * interaction) + inputs.value("offset"))


def _quadratic_fock(inputs, P):
    return inputs.array("h0") + _linear_action(inputs, P)


def _quadratic_df(inputs, P, D):
    del P
    return _linear_action(inputs, D)


def _transpose_fock(inputs, P):
    return (inputs.array("h0") + _linear_action(inputs, P)).T


def _conjugate_fock(inputs, P):
    return (inputs.array("h0") + _linear_action(inputs, P)).conj()


def _half_energy(inputs, P):
    coordinates = np.real(
        np.einsum("aij,ji->a", inputs.array("basis"), P, optimize=False)
    )
    one_body = np.einsum("ij,ji->", inputs.array("h0"), P, optimize=False)
    interaction = coordinates @ inputs.array("kernel") @ coordinates
    return float(np.real(one_body + 0.25 * interaction) + inputs.value("offset"))


def _half_df(inputs, P, D):
    del P
    return 0.5 * _linear_action(inputs, D)


def _missing_h0_fock(inputs, P):
    return _linear_action(inputs, P)


def _constant_energy(inputs, P):
    p0 = inputs.array("p0")
    constant_fock = inputs.array("h0") + _linear_action(inputs, p0)
    return float(np.real(np.einsum("ij,ji->", constant_fock, P, optimize=False)))


def _constant_fock(inputs, P):
    del P
    return inputs.array("h0") + _linear_action(inputs, inputs.array("p0"))


def _zero_df(inputs, P, D):
    del inputs, P
    return np.zeros_like(D)


def _nonlinear_fock(inputs, P):
    delta = P - inputs.array("p0")
    amount = float(np.real(np.einsum("ij,ji->", delta, delta, optimize=False)))
    return inputs.array("h0") + _linear_action(inputs, P) + amount * inputs.array("x")


def _nonself_df(inputs, P, D):
    del P
    coefficient = np.einsum("ij,ji->", inputs.array("x"), D, optimize=False)
    return _linear_action(inputs, D) + 0.17 * float(np.real(coefficient)) * inputs.array("y")


def _alias_fock(inputs, P):
    del inputs
    return P


def _mutating_energy(inputs, P):
    value = inputs.array("h0")
    value.setflags(write=True)
    value[0, 0] += 1.0
    return float(np.real(np.einsum("ij,ji->", value, P, optimize=False)))


def _delegating_energy(inputs, P):
    fock = _quadratic_fock(inputs, P)
    return float(np.real(np.einsum("ij,ji->", fock, P, optimize=False)))


def _delegating_df(inputs, P, D):
    del P
    return _quadratic_fock(inputs, D)


def _forbidden_entrypoint(inputs, D):
    return _linear_action(inputs, D)


def _forbidden_df(inputs, P, D):
    del P
    return _forbidden_entrypoint(inputs, D)


class _Case:
    def __init__(self) -> None:
        self.space = TDHFFullProjectorSpace(
            dimension=4,
            axis_sizes=(2, 2),
            axis_order=("sector", "orbital"),
            orbital_order_fingerprint=sha256(b"n4-orbital-order").hexdigest(),
            layout_adapter_fingerprint=sha256(b"identity-dense-layout").hexdigest(),
        )
        self.directions = deterministic_complete_hermitian_basis(4)
        basis = np.asarray([item.matrix for item in self.directions])
        rng = np.random.default_rng(7183)
        raw = rng.normal(scale=0.035, size=(16, 16))
        kernel = np.asarray(0.5 * (raw + raw.T), dtype=np.complex128)
        p0 = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)
        temporary = make_tdhf_scalar_functional_inputs_manifest(
            {
                "basis": basis,
                "h0": np.zeros((4, 4), dtype=np.complex128),
                "kernel": kernel,
                "offset": 0.031,
                "p0": p0,
                "x": basis[5].copy(),
                "y": basis[9].copy(),
            },
            source_fingerprint=sha256(b"temporary-n4-input").hexdigest(),
            provenance="Temporary exact input used only to derive an independently chosen stationary h0.",
        )
        target_fock = np.diag([-0.8, -0.3, 0.4, 1.1]).astype(np.complex128)
        h0 = target_fock - _linear_action(temporary, p0)
        self.inputs = make_tdhf_scalar_functional_inputs_manifest(
            {
                "basis": basis,
                "h0": h0,
                "kernel": kernel,
                "offset": 0.031,
                "p0": p0,
                "x": basis[5].copy(),
                "y": basis[9].copy(),
            },
            source_fingerprint=sha256(b"independent-n4-quadratic-input-v1").hexdigest(),
            provenance="Independent N=4 self-adjoint real-linear quadratic functional.",
        )
        self.plan = TDHFFullProjectorValidationPlan(
            space=self.space,
            source_projector=p0,
            directions=self.directions,
            steps=(2.0e-2, 1.0e-2),
            tolerances=TDHFFullProjectorValidationTolerances(
                gradient_absolute=5.0e-9,
                gradient_relative=5.0e-9,
                derivative_absolute=5.0e-9,
                derivative_relative=5.0e-9,
                exact_absolute=5.0e-10,
                exact_relative=5.0e-10,
                stationarity_absolute=5.0e-10,
                stationarity_relative=5.0e-10,
                self_adjoint_absolute=5.0e-10,
                self_adjoint_relative=5.0e-10,
            ),
            registration_label="complete-N4-Hermitian-affine-all-steps-v1",
            probe_scope="complete_small_test_basis",
        )


def _binding(
    *,
    energy=_quadratic_energy,
    fock=_quadratic_fock,
    df=_quadratic_df,
    forbidden=(),
):
    return TDHFFullProjectorFunctionalBinding(
        energy=bind_tdhf_scalar_kernel(
            role="energy",
            callback=energy,
            dependencies=(_linear_action,),
            provenance="Independent explicit N=4 energy implementation.",
        ),
        fock=bind_tdhf_scalar_kernel(
            role="fock",
            callback=fock,
            dependencies=(_linear_action,),
            provenance="Independent explicit N=4 Fock implementation.",
        ),
        fock_derivative=bind_tdhf_scalar_kernel(
            role="fock_derivative",
            callback=df,
            dependencies=(_linear_action,),
            provenance="Independent explicit N=4 dF implementation.",
        ),
        forbidden_entrypoints=tuple(forbidden),
    )


def _validate(case: _Case, binding=None, inputs=None, plan=None):
    selected_binding = _binding() if binding is None else binding
    selected_inputs = case.inputs if inputs is None else inputs
    selected_plan = case.plan if plan is None else plan
    approval = make_tdhf_full_projector_functional_approval(
        space=case.space,
        inputs=selected_inputs,
        binding=selected_binding,
        plan=selected_plan,
        provenance="Detached public approval before every callback call.",
    )
    return validate_tdhf_full_projector_functional(
        approval=approval,
        space=case.space,
        inputs=selected_inputs,
        binding=selected_binding,
        plan=selected_plan,
    )


def test_public_full_projector_quadratic_certificate_checks_every_equation() -> None:
    case = _Case()
    binding = _binding()
    receipt = _validate(case, binding)

    assert receipt.registered_probe_functional_consistency
    assert receipt.full_projector_functional_consistency
    assert receipt.direction_inventory_fingerprint == (
        receipt.complete_basis_inventory_fingerprint
    )
    assert receipt.source_stationarity_verified
    assert receipt.dF_anchor_independence_verified
    assert receipt.dF_real_self_adjoint_verified
    assert receipt.callback_trace_verified
    assert receipt.callback_source_code_dependency_stable
    assert receipt.directions_are_complete
    assert receipt.maximum_dF_response_frobenius_norm > (
        TDHF_FULL_PROJECTOR_DF_RESPONSE_MINIMUM
    )
    assert not receipt.exact_unitary_projector_probes_executed
    assert not receipt.tdhf_hessian_match
    assert not receipt.static_hessian_authority_promoted
    assert not receipt.production_ready
    assert not receipt.paper_reproduction_verified
    assert len(receipt.step_evidence) == len(case.directions) * len(case.plan.steps)
    assert len(receipt.anchor_evidence) == (
        len(case.directions) ** 2 * len(case.plan.steps)
    )
    assert len(receipt.pairing_evidence) == len(case.directions) ** 2
    assert receipt.source_commutator_residual < 1.0e-13
    assert receipt.source_qfp_residual < 1.0e-13
    for record in receipt.step_evidence:
        assert record.energy_to_fock_residual < 1.0e-9
        assert record.energy_second_to_derivative_residual < 1.0e-8
        assert record.fock_to_derivative_residual < 1.0e-10
        assert record.exact_affine_fock_residual < 1.0e-12
        assert record.exact_quadratic_energy_residual < 1.0e-12
    assert any("imag" in item.label for item in case.directions)
    assert any(
        np.max(np.abs(item.matrix[:2, 2:])) > 0.0 for item in case.directions
    )


@pytest.mark.parametrize(
    ("binding", "match"),
    (
        (lambda: _binding(fock=_transpose_fock), "E/F/dF|E->F|stationarity"),
        (lambda: _binding(fock=_conjugate_fock), "E/F/dF|E->F|stationarity"),
        (lambda: _binding(energy=_half_energy), "E/F/dF|E->F|E''"),
        (lambda: _binding(df=_half_df), "E/F/dF|F->dF|E''"),
        (lambda: _binding(fock=_missing_h0_fock), "stationarity|E/F/dF|E->F"),
        (lambda: _binding(fock=_nonlinear_fock), "affine/quadratic"),
        (lambda: _binding(df=_nonself_df), "self-adjoint|E/F/dF"),
        (lambda: _binding(fock=_alias_fock), "aliases"),
    ),
)
def test_formula_and_alias_canaries_fail_through_public_validation(binding, match) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        _validate(_Case(), binding())


def test_input_alias_mutation_nonfinite_and_writable_canaries() -> None:
    shared = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="alias"):
        make_tdhf_scalar_functional_inputs_manifest(
            {"a": shared, "b": shared[:, :]},
            source_fingerprint=sha256(b"alias").hexdigest(),
            provenance="Alias canary.",
        )
    with pytest.raises(ValueError, match="finite"):
        make_tdhf_scalar_functional_inputs_manifest(
            {"a": np.asarray([np.nan])},
            source_fingerprint=sha256(b"nonfinite").hexdigest(),
            provenance="Nonfinite canary.",
        )

    case = _Case()
    stale = case.inputs.array("h0").copy()
    stale[0, 0] += 1.0
    object.__setattr__(case.inputs.entries[1], "value", stale)
    with pytest.raises(ValueError, match="writable|stale|input"):
        _validate(case)


def test_callback_input_mutation_is_rejected() -> None:
    with pytest.raises((ValueError, RuntimeError), match="WRITEABLE|write|writable|mutat"):
        _validate(_Case(), _binding(energy=_mutating_energy))


def test_step_and_roundoff_nonvacuity_canaries() -> None:
    case = _Case()
    with pytest.raises(ValueError, match="locked.*range"):
        replace(case.plan, steps=(1.0e-8, 5.0e-9))

    values = {item.name: item.value for item in case.inputs.entries}
    values["offset"] = 1.0e10
    huge = make_tdhf_scalar_functional_inputs_manifest(
        values,
        source_fingerprint=sha256(b"huge-offset-canary").hexdigest(),
        provenance="Huge raw energy offset must trip roundoff nonvacuity.",
    )
    with pytest.raises(ValueError, match="roundoff allowance.*vacuous"):
        _validate(case, inputs=huge)


def test_direct_peer_and_configured_forbidden_delegation_are_traced() -> None:
    with pytest.raises(RuntimeError, match="peer.*delegation"):
        _validate(_Case(), _binding(energy=_delegating_energy))
    with pytest.raises(RuntimeError, match="peer.*delegation"):
        _validate(_Case(), _binding(df=_delegating_df))
    with pytest.raises(RuntimeError, match="delegation: configured forbidden"):
        _validate(
            _Case(),
            _binding(df=_forbidden_df, forbidden=(_forbidden_entrypoint,)),
        )


def test_closure_defaults_and_same_callback_code_are_rejected() -> None:
    captured = 0.0

    def closure_energy(inputs, P):
        return _quadratic_energy(inputs, P) + captured

    with pytest.raises(ValueError, match="closures"):
        bind_tdhf_scalar_kernel(
            role="energy",
            callback=closure_energy,
            dependencies=(_linear_action,),
            provenance="Closure canary.",
        )

    def default_energy(inputs, P, extra=0.0):
        return _quadratic_energy(inputs, P) + extra

    with pytest.raises((TypeError, ValueError), match="signature|defaults"):
        bind_tdhf_scalar_kernel(
            role="energy",
            callback=default_energy,
            dependencies=(_linear_action,),
            provenance="Mutable/default canary.",
        )

    same_code_fock = FunctionType(
        _quadratic_energy.__code__, globals(), "same_code_fock"
    )
    with pytest.raises(ValueError, match="distinct code"):
        _binding(fock=same_code_fock)


def test_stale_input_source_and_callback_snapshots_fail_before_calls() -> None:
    case = _Case()
    binding = _binding()
    approval = make_tdhf_full_projector_functional_approval(
        space=case.space,
        inputs=case.inputs,
        binding=binding,
        plan=case.plan,
        provenance="Detached stale-snapshot canary approval.",
    )

    original_source = binding.energy.manifest.callback.source_fingerprint
    object.__setattr__(
        binding.energy.manifest.callback, "source_fingerprint", "f" * 64
    )
    try:
        with pytest.raises(ValueError, match="source/code/module.*drifted"):
            validate_tdhf_full_projector_functional(
                approval=approval,
                space=case.space,
                inputs=case.inputs,
                binding=binding,
                plan=case.plan,
            )
    finally:
        object.__setattr__(
            binding.energy.manifest.callback,
            "source_fingerprint",
            original_source,
        )

    original_code = _quadratic_fock.__code__
    _quadratic_fock.__code__ = _conjugate_fock.__code__
    try:
        with pytest.raises(ValueError, match="callback source/code/module.*drifted"):
            validate_tdhf_full_projector_functional(
                approval=approval,
                space=case.space,
                inputs=case.inputs,
                binding=binding,
                plan=case.plan,
            )
    finally:
        _quadratic_fock.__code__ = original_code


def test_direction_and_small_basis_boundary_are_fail_closed() -> None:
    nonhermitian = np.zeros((4, 4), dtype=np.complex128)
    nonhermitian[0, 1] = 1.0
    with pytest.raises(ValueError, match="Hermitian"):
        TDHFFullProjectorDirection("bad", nonhermitian)
    with pytest.raises(ValueError, match="small tests"):
        deterministic_complete_hermitian_basis(80)


def test_direction_normalization_tiny_signal_and_incomplete_scope_are_fail_closed() -> None:
    case = _Case()
    scaled = TDHFFullProjectorDirection(
        "scaled", 7.0 * case.directions[5].matrix
    )
    assert np.linalg.norm(scaled.matrix, ord="fro") == pytest.approx(
        1.0, abs=2.0e-15, rel=0.0
    )
    with pytest.raises(ValueError, match="zero/tiny"):
        TDHFFullProjectorDirection(
            "tiny", 1.0e-16 * case.directions[0].matrix
        )

    incomplete = replace(
        case.plan,
        directions=case.directions[:3],
        probe_scope="explicit_bound_probes",
    )
    receipt = _validate(case, plan=incomplete)
    assert receipt.registered_probe_functional_consistency
    assert not receipt.directions_are_complete
    assert not receipt.full_projector_functional_consistency


def test_required_df_informativeness_rejects_constant_interaction() -> None:
    case = _Case()
    required = replace(case.plan, require_informative_df=True)
    with pytest.raises(ValueError, match="required dF response.*uninformative"):
        _validate(
            case,
            binding=_binding(
                energy=_constant_energy,
                fock=_constant_fock,
                df=_zero_df,
            ),
            plan=required,
        )


def test_preregistered_unitary_projector_values_execute_energy_and_fock() -> None:
    case = _Case()
    angle = 0.23
    unitary = np.eye(4, dtype=np.complex128)
    unitary[np.ix_((0, 2), (0, 2))] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    projector = unitary @ case.plan.source_projector @ unitary.conj().T
    probe = make_tdhf_full_projector_unitary_probe(
        label="occupied-unoccupied-exact-rotation",
        source_projector=case.plan.source_projector,
        projector=np.asarray(projector, dtype=np.complex128),
    )
    plan = replace(case.plan, unitary_projector_probes=(probe,))
    receipt = _validate(case, plan=plan)
    assert receipt.exact_unitary_projector_probes_executed
    assert receipt.exact_unitary_projector_probe_count == 1
    assert receipt.exact_unitary_projector_probe_inventory_fingerprint
    assert receipt.unitary_probe_evidence[0].energy_executed
    assert receipt.unitary_probe_evidence[0].fock_executed
