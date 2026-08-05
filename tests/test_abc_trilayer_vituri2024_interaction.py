"""Focused lightweight checks for the next Vituri-2024 interaction layer."""

from dataclasses import FrozenInstanceError, replace
import inspect
import math

import numpy as np
import pytest

import mean_field.systems.abc_trilayer.vituri2024_interaction as interaction_module
from mean_field.systems.abc_trilayer import (
    FORM_FACTOR_GAUGE_LABEL,
    PAPER_EPSILON,
    PAPER_Q_TF_INVERSE_ANGSTROM,
    PAPER_Q_TF_PER_A0,
    SM_TEX_SHA256,
    VITURI2024_PARAMETERS,
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
    bind_vituri2024_interaction,
    c3_basis_operator,
    state_overlap_invariant,
    third_band_density_form_factor,
    third_lowest_active_band,
    vituri2024_v0,
    vituri2024_vtf,
)

_PROVIDER_SHA256 = "a" * 64
_SOURCE_TEXT = (
    "Controlled Test001 reproduction choice for d and e^2; not a paper-direct "
    "q=0 background prescription."
)


def _choice(
    q0_evaluation: str = "analytic_kernel_limit_only",
    **overrides: object,
) -> Vituri2024InteractionChoiceReceipt:
    values: dict[str, object] = {
        "gate_distance_angstrom": 250.0,
        "coulomb_e2_ev_angstrom": 14.3996454784255,
        "q0_evaluation": q0_evaluation,
        "provider_sha256": _PROVIDER_SHA256,
        "source_sha256": SM_TEX_SHA256,
        "authority_kind": "reproduction_choice",
        "source_text": _SOURCE_TEXT,
    }
    values.update(overrides)
    return Vituri2024InteractionChoiceReceipt(**values)  # type: ignore[arg-type]


def _rotate_c3(k: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi / 3.0
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    ) @ k


def test001_v0_vtf_match_sm_tex_entrywise_and_document_dimensions() -> None:
    receipt = _choice()
    prefactor = 2.0 * np.pi * receipt.coulomb_e2_ev_angstrom / PAPER_EPSILON

    for q in (1.0e-8, 0.0017, 0.025, 0.31):
        radial_direct = np.tanh(q * receipt.gate_distance_angstrom) / q
        expected_v0 = prefactor * radial_direct
        expected_vtf = expected_v0 / (
            1.0 + PAPER_Q_TF_INVERSE_ANGSTROM * radial_direct
        )
        assert vituri2024_v0(q, receipt) == pytest.approx(
            expected_v0, rel=2.0e-15
        )
        assert vituri2024_vtf(q, receipt) == pytest.approx(
            expected_vtf, rel=2.0e-15
        )

    assert "1/Angstrom" in (interaction_module.__doc__ or "")
    assert "eV*Angstrom^2" in (vituri2024_v0.__doc__ or "")
    assert "eV*Angstrom^2" in (vituri2024_vtf.__doc__ or "")
    assert receipt.epsilon == 8.0
    assert receipt.q_tf_per_a0 == 0.04
    assert receipt.q_tf_inverse_angstrom == pytest.approx(
        0.04 / VITURI2024_PARAMETERS.a0, rel=0.0, abs=0.0
    )


def test001_q0_reject_and_analytic_kernel_limit_are_strictly_separate() -> None:
    rejecting = _choice("reject")
    analytic = _choice("analytic_kernel_limit_only")
    prefactor = 2.0 * np.pi * analytic.coulomb_e2_ev_angstrom / PAPER_EPSILON
    d = analytic.gate_distance_angstrom

    for function in (vituri2024_v0, vituri2024_vtf):
        with pytest.raises(ValueError, match="q=0 evaluation rejected"):
            function(0.0, rejecting)

    assert vituri2024_v0(0.0, analytic) == pytest.approx(
        prefactor * d, rel=0.0, abs=0.0
    )
    assert vituri2024_vtf(0.0, analytic) == pytest.approx(
        prefactor * d / (1.0 + PAPER_Q_TF_INVERSE_ANGSTROM * d),
        rel=0.0,
        abs=0.0,
    )

    small_q = 1.0e-10
    assert vituri2024_v0(small_q, analytic) == pytest.approx(
        vituri2024_v0(0.0, analytic), rel=3.0e-16
    )
    assert vituri2024_vtf(small_q, analytic) == pytest.approx(
        vituri2024_vtf(0.0, analytic), rel=3.0e-16
    )
    assert analytic.establishes_hf_q0_background is False
    assert "kernel limit" in (vituri2024_vtf.__doc__ or "")

    huge_d = _choice(gate_distance_angstrom=1.0e308)
    saturated = vituri2024_vtf(0.0, huge_d)
    expected_saturation = (
        (2.0 * np.pi / PAPER_EPSILON)
        * huge_d.coulomb_e2_ev_angstrom
        / PAPER_Q_TF_INVERSE_ANGSTROM
    )
    assert saturated == pytest.approx(expected_saturation, rel=2.0e-15)


def test001_large_q_gate_and_tf_limits() -> None:
    receipt = _choice()
    q = 100.0
    prefactor = 2.0 * np.pi * receipt.coulomb_e2_ev_angstrom / PAPER_EPSILON

    # tanh(q*d) rounds to exactly one here, so r=1/q and the TF expression
    # reduces rigorously at float64 precision to prefactor/(q+qTF).
    assert math.tanh(q * receipt.gate_distance_angstrom) == 1.0
    assert vituri2024_v0(q, receipt) == pytest.approx(
        prefactor / q, rel=2.0e-15
    )
    assert vituri2024_vtf(q, receipt) == pytest.approx(
        prefactor / (q + PAPER_Q_TF_INVERSE_ANGSTROM), rel=2.0e-15
    )
    assert 0.0 < vituri2024_vtf(q, receipt) < vituri2024_v0(q, receipt)


def test001_receipt_is_strict_immutable_and_hash_tamper_closed() -> None:
    receipt = _choice()
    binding = bind_vituri2024_interaction(receipt)

    assert isinstance(binding, Vituri2024InteractionBinding)
    assert binding.receipt is receipt
    assert binding.receipt_fingerprint == receipt.fingerprint
    assert receipt.paper_direct_claim_allowed is False
    assert receipt.establishes_hf_q0_background is False
    assert binding.paper_direct_claim_allowed is False
    assert binding.establishes_hf_q0_background is False
    assert not hasattr(receipt, "executable_ready")
    assert not hasattr(binding, "executable_ready")
    assert vituri2024_v0(0.02, binding) == vituri2024_v0(0.02, receipt)

    with pytest.raises(FrozenInstanceError):
        receipt.gate_distance_angstrom = 10.0  # type: ignore[misc]
    for field_name in ("gate_distance_angstrom", "coulomb_e2_ev_angstrom"):
        for bad in (True, False, 0.0, -1.0, np.nan, np.inf, -np.inf, 1.0 + 0.0j):
            with pytest.raises((TypeError, ValueError)):
                _choice(**{field_name: bad})
    for bad_q0 in ("analytic", "zero", "", True, 0):
        with pytest.raises(ValueError):
            _choice(q0_evaluation=bad_q0)  # type: ignore[arg-type]
    for bad_authority in ("paper_explicit", "", True):
        with pytest.raises(ValueError):
            _choice(authority_kind=bad_authority)
    with pytest.raises(ValueError, match="provider_sha256"):
        _choice(provider_sha256="A" * 64)
    with pytest.raises(ValueError, match="provider_sha256"):
        _choice(provider_sha256="a" * 63)
    with pytest.raises(ValueError, match="must match"):
        _choice(source_sha256="0" * 64)
    with pytest.raises(ValueError, match="source_text"):
        _choice(source_text="  ")
    with pytest.raises(TypeError):
        Vituri2024InteractionChoiceReceipt(
            gate_distance_angstrom=250.0,
            coulomb_e2_ev_angstrom=14.4,
            q0_evaluation="reject",
            provider_sha256=_PROVIDER_SHA256,
            source_sha256=SM_TEX_SHA256,
            authority_kind="reproduction_choice",
            source_text=_SOURCE_TEXT,
            epsilon=20.0,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        Vituri2024InteractionBinding(receipt="not-a-receipt")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_q",
    (True, False, -1.0, np.nan, np.inf, -np.inf, 0.1 + 0.0j, [0.1]),
)
def test001_kernel_rejects_bool_negative_and_nonfinite_q(bad_q: object) -> None:
    receipt = _choice()
    for function in (vituri2024_v0, vituri2024_vtf):
        with pytest.raises((TypeError, ValueError)):
            function(bad_q, receipt)


def test001_third_band_fkk_and_projector_trace_identity() -> None:
    k = np.array([0.019, -0.012])
    result = third_band_density_form_factor(k, k, 1, 0.028)

    assert result.value == pytest.approx(1.0 + 0.0j, abs=2.0e-15)
    assert result.absolute_squared == pytest.approx(1.0, abs=3.0e-15)
    assert result.projector_trace_identity == pytest.approx(
        result.absolute_squared, abs=3.0e-15
    )
    assert result.projector_trace_residual <= 3.0e-15
    assert result.delta1_ev == 0.028
    assert result.bra_local_gap.valley == 1
    assert result.bra_local_gap.delta1_ev == result.delta1_ev
    assert result.bra_local_gap.lower_gap_ev > 0.0
    assert result.bra_local_gap.upper_gap_ev > 0.0
    assert result.ket_local_gap == result.bra_local_gap
    assert result.gauge_label == "numerical_eigh_phase_only_not_paper_gauge"
    assert result.gauge_label == FORM_FACTOR_GAUGE_LABEL
    assert result.paper_direct_claim_allowed is False
    assert result.establishes_hf_q0_background is False
    assert not hasattr(result, "executable_ready")
    with pytest.raises(ValueError, match="delta1 mismatch"):
        replace(result, delta1_ev=0.027)
    wrong_valley_gap = replace(result.bra_local_gap, valley=-1)
    with pytest.raises(ValueError, match="valley mismatch"):
        replace(result, bra_local_gap=wrong_valley_gap)


def test001_state_helper_independent_phase_covariance_and_projector_invariant() -> None:
    bra = third_lowest_active_band([0.021, 0.008], 1, 0.028).eigenvector
    ket = third_lowest_active_band([-0.014, 0.017], 1, 0.028).eigenvector
    base = state_overlap_invariant(bra, ket)
    phi_bra = 0.37
    phi_ket = -0.91
    transformed = state_overlap_invariant(
        np.exp(1j * phi_bra) * bra,
        np.exp(1j * phi_ket) * ket,
    )

    expected_phase = np.exp(-1j * phi_bra + 1j * phi_ket)
    assert transformed.value == pytest.approx(
        expected_phase * base.value, rel=2.0e-14, abs=2.0e-15
    )
    assert transformed.absolute_squared == pytest.approx(
        base.absolute_squared, abs=3.0e-15
    )
    assert transformed.projector_trace_identity == pytest.approx(
        base.projector_trace_identity, abs=3.0e-15
    )
    assert transformed.projector_trace_identity == pytest.approx(
        transformed.absolute_squared, abs=3.0e-15
    )


def test001_common_unitary_c3_invariance_at_state_level() -> None:
    tau = -1
    bra = third_lowest_active_band([0.027, -0.005], tau, 0.028).eigenvector
    ket = third_lowest_active_band([-0.011, 0.022], tau, 0.028).eigenvector
    c3 = c3_basis_operator(tau)
    original = state_overlap_invariant(bra, ket)
    transformed = state_overlap_invariant(c3 @ bra, c3 @ ket)

    assert transformed.value == pytest.approx(original.value, abs=3.0e-15)
    assert transformed.absolute_squared == pytest.approx(
        original.absolute_squared, abs=3.0e-15
    )
    assert transformed.projector_trace_identity == pytest.approx(
        original.projector_trace_identity, abs=3.0e-15
    )


def test001_parent_solution_c3_and_time_reversal_magnitude_covariance() -> None:
    k_bra = np.array([0.026, -0.013])
    k_ket = np.array([-0.009, 0.024])
    Delta1 = 0.028

    for tau in (-1, 1):
        original = third_band_density_form_factor(
            k_bra, k_ket, tau, Delta1
        )
        c3_related = third_band_density_form_factor(
            _rotate_c3(k_bra), _rotate_c3(k_ket), tau, Delta1
        )
        time_reversed = third_band_density_form_factor(
            -k_bra, -k_ket, -tau, Delta1
        )
        assert c3_related.absolute_squared == pytest.approx(
            original.absolute_squared, rel=2.0e-13, abs=2.0e-15
        )
        assert time_reversed.absolute_squared == pytest.approx(
            original.absolute_squared, rel=2.0e-13, abs=2.0e-15
        )
        assert c3_related.projector_trace_identity == pytest.approx(
            original.projector_trace_identity, rel=2.0e-13, abs=2.0e-15
        )
        assert time_reversed.projector_trace_identity == pytest.approx(
            original.projector_trace_identity, rel=2.0e-13, abs=2.0e-15
        )


def test001_same_valley_only_and_no_readiness_or_authority_inflation() -> None:
    state = np.arange(1.0, 7.0, dtype=np.complex128)
    algebraic = state_overlap_invariant(state, state)
    assert algebraic.authority_scope == (
        "algebraic_overlap_no_band_valley_or_hamiltonian_authority"
    )
    assert algebraic.paper_direct_claim_allowed is False
    for bad_valley in (0, 2, -2, 1.0, True, np.nan, "1"):
        with pytest.raises(ValueError):
            third_band_density_form_factor(
                [0.01, 0.0], [0.02, 0.0], bad_valley, 0.028  # type: ignore[arg-type]
            )

    signature = inspect.signature(state_overlap_invariant)
    assert "valley" not in signature.parameters
    assert not hasattr(interaction_module, "state_density_form_factor")
    assert not hasattr(interaction_module, "executable_ready")
    assert not hasattr(interaction_module, "hartree_fock")
    assert not hasattr(interaction_module, "tdhf")
    assert PAPER_EPSILON == 8.0
    assert PAPER_Q_TF_PER_A0 == 0.04
