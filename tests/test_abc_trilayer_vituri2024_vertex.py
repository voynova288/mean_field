"""Tiny Test001 checks for the authority-limited Vituri projected vertex."""

from dataclasses import FrozenInstanceError, replace
import inspect

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc_trilayer_module
import mean_field.systems.abc_trilayer.vituri2024_vertex as vertex_module
from mean_field.systems.abc_trilayer import (
    ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR,
    ANTISYMMETRIZED_VERTEX_EXCLUSIONS,
    KINEMATICS_PROVIDER_METADATA_STATUS,
    ORDERED_COEFFICIENT_EXCLUSIONS,
    ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR,
    PUBLISHED_PRB_PDF_SHA256,
    SM_TEX_SHA256,
    VERTEX_AUTHORITY,
    VERTEX_GAUGE_BEHAVIOR,
    VERTEX_NORMALIZATION_IDENTITY,
    Vituri2024Flavor,
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024InteractionChoiceReceipt,
    Vituri2024Orbital,
    bind_vituri2024_interaction,
    state_overlap_invariant,
    third_band_density_form_factor,
    vituri2024_antisymmetrized_projected_vertex,
    vituri2024_ordered_projected_coefficient,
    vituri2024_vtf,
)

_PROVIDER_SHA256 = "7" * 64
_SOURCE_TEXT = (
    "Caller-attested Test001 local continuum quartet and diagnostic tolerance; "
    "not independently verified; no reciprocal-torus or reciprocal-carry claim."
)
_DELTA1 = 0.028


def _interaction(
    q0_evaluation: str = "reject",
) -> Vituri2024InteractionChoiceReceipt:
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996454784255,
        q0_evaluation=q0_evaluation,  # type: ignore[arg-type]
        provider_sha256="8" * 64,
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Test001 interaction choice, not a paper-direct background.",
    )


def _orbital(
    momentum: tuple[float, float],
    *,
    valley: int = 1,
    spin: int = 1,
) -> Vituri2024Orbital:
    return Vituri2024Orbital(
        flavor=Vituri2024Flavor(valley=valley, spin=spin),
        momentum_inverse_angstrom=momentum,
    )


def _quartet_orbitals() -> tuple[
    Vituri2024Orbital,
    Vituri2024Orbital,
    Vituri2024Orbital,
    Vituri2024Orbital,
]:
    # Dyadic coordinates keep every tested permutation exactly conserving in
    # float64; no tolerance is used to manufacture antisymmetry/Hermiticity.
    k_alpha = (16.0 / 1024.0, -8.0 / 1024.0)
    k_beta = (-12.0 / 1024.0, 14.0 / 1024.0)
    k_gamma = (9.0 / 1024.0, 12.0 / 1024.0)
    k_delta = (
        k_alpha[0] + k_beta[0] - k_gamma[0],
        k_alpha[1] + k_beta[1] - k_gamma[1],
    )
    return tuple(_orbital(k) for k in (k_alpha, k_beta, k_gamma, k_delta))  # type: ignore[return-value]


def _kinematics(
    alpha: Vituri2024Orbital,
    beta: Vituri2024Orbital,
    gamma: Vituri2024Orbital,
    delta: Vituri2024Orbital,
    *,
    tolerance: float = 0.0,
    provider_sha256: str = _PROVIDER_SHA256,
    derivation_source_sm_sha256: str = SM_TEX_SHA256,
    source_text: str = _SOURCE_TEXT,
) -> Vituri2024FourPointKinematicsReceipt:
    return Vituri2024FourPointKinematicsReceipt(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        momentum_tolerance_inverse_angstrom=tolerance,
        provider_sha256=provider_sha256,
        derivation_source_sm_sha256=derivation_source_sm_sha256,
        source_text=source_text,
    )


def _permuted(
    receipt: Vituri2024FourPointKinematicsReceipt,
    alpha: Vituri2024Orbital,
    beta: Vituri2024Orbital,
    gamma: Vituri2024Orbital,
    delta: Vituri2024Orbital,
) -> Vituri2024FourPointKinematicsReceipt:
    return _kinematics(
        alpha,
        beta,
        gamma,
        delta,
        tolerance=receipt.momentum_tolerance_inverse_angstrom,
        provider_sha256=receipt.provider_sha256,
        derivation_source_sm_sha256=receipt.derivation_source_sm_sha256,
        source_text=receipt.source_text,
    )


def _state_level_vertex_oracle(
    states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direct_kernel: float,
    exchange_kernel: float,
) -> complex:
    """Authority-neutral overlap algebra, never a physical provenance receipt."""

    alpha, beta, gamma, delta = states
    f_ad = state_overlap_invariant(alpha, delta)
    f_bg = state_overlap_invariant(beta, gamma)
    f_ag = state_overlap_invariant(alpha, gamma)
    f_bd = state_overlap_invariant(beta, delta)
    for overlap in (f_ad, f_bg, f_ag, f_bd):
        assert overlap.authority_scope == (
            "algebraic_overlap_no_band_valley_or_hamiltonian_authority"
        )
        assert overlap.paper_direct_claim_allowed is False
    return direct_kernel * f_ad.value * f_bg.value - exchange_kernel * f_ag.value * f_bd.value


def test001_ordered_coefficient_matches_hand_expression() -> None:
    alpha, beta, gamma, delta = _quartet_orbitals()
    kinematics = _kinematics(alpha, beta, gamma, delta)
    interaction = _interaction()

    result = vituri2024_ordered_projected_coefficient(
        kinematics, _DELTA1, interaction
    )
    q_vector = np.subtract(
        beta.momentum_inverse_angstrom, gamma.momentum_inverse_angstrom
    )
    q = float(np.linalg.norm(q_vector))
    f_ad = third_band_density_form_factor(
        alpha.momentum_inverse_angstrom,
        delta.momentum_inverse_angstrom,
        alpha.flavor.valley,
        _DELTA1,
    )
    f_bg = third_band_density_form_factor(
        beta.momentum_inverse_angstrom,
        gamma.momentum_inverse_angstrom,
        beta.flavor.valley,
        _DELTA1,
    )
    expected = vituri2024_vtf(q, interaction) * f_ad.value * f_bg.value

    assert result.value == pytest.approx(expected, rel=2.0e-15, abs=2.0e-15)
    assert result.transfer_vector_inverse_angstrom == pytest.approx(q_vector)
    assert result.transfer_norm_inverse_angstrom == pytest.approx(q)
    assert result.selection_rule == "allowed_by_both_flavor_deltas"
    assert result.form_factor_alpha_delta == f_ad
    assert result.form_factor_beta_gamma == f_bg
    assert len(result.form_factor_receipts) == 2


def test001_ordered_and_antisymmetrized_full_sum_normalization_identity() -> None:
    alpha, beta, gamma, delta = _quartet_orbitals()
    kinematics = _kinematics(alpha, beta, gamma, delta)
    interaction = _interaction()
    base = vituri2024_antisymmetrized_projected_vertex(
        kinematics, _DELTA1, interaction
    )
    ket_swapped = vituri2024_antisymmetrized_projected_vertex(
        _permuted(kinematics, alpha, beta, delta, gamma),
        _DELTA1,
        interaction,
    )
    assert base.direct_ordered is not None
    assert base.exchange_ordered is not None

    area = 37.0
    operator_gamma_delta = 0.31 - 0.27j
    operator_delta_gamma = -operator_gamma_delta
    ordered_reconstruction = (
        base.direct_ordered.value * operator_gamma_delta
        + base.exchange_ordered.value * operator_delta_gamma
    ) / (2.0 * area)
    antisymmetrized_reconstruction = (
        base.value * operator_gamma_delta
        + ket_swapped.value * operator_delta_gamma
    ) / (4.0 * area)

    assert ordered_reconstruction == pytest.approx(
        antisymmetrized_reconstruction, rel=2.0e-14, abs=2.0e-15
    )
    assert ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR == "1/(2A)"
    assert ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR == "1/(4A)"
    assert base.direct_ordered.omitted_full_sum_hamiltonian_area_prefactor == (
        ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR
    )
    assert base.omitted_full_sum_hamiltonian_area_prefactor == (
        ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR
    )
    assert base.direct_ordered.exclusions == ORDERED_COEFFICIENT_EXCLUSIONS
    assert base.exclusions == ANTISYMMETRIZED_VERTEX_EXCLUSIONS
    assert base.direct_ordered.exclusions != base.exclusions
    assert base.normalization_identity == VERTEX_NORMALIZATION_IDENTITY
    assert base.direct_ordered.normalization_identity == VERTEX_NORMALIZATION_IDENTITY


def test001_flavor_selection_zero_bypasses_irrelevant_q0_and_form_factors() -> None:
    alpha = _orbital((13.0 / 1024.0, -4.0 / 1024.0), valley=1, spin=1)
    beta = _orbital((-6.0 / 1024.0, 9.0 / 1024.0), valley=1, spin=1)
    gamma = _orbital(beta.momentum_inverse_angstrom, valley=1, spin=1)
    delta = _orbital(alpha.momentum_inverse_angstrom, valley=-1, spin=1)
    kinematics = _kinematics(alpha, beta, gamma, delta)

    result = vituri2024_ordered_projected_coefficient(
        kinematics, _DELTA1, _interaction("reject")
    )

    assert result.transfer_norm_inverse_angstrom == 0.0
    assert result.value == 0.0j
    assert result.selection_rule == "zero_by_alpha_delta_flavor_delta"
    assert result.kernel_value_ev_angstrom_squared is None
    assert result.form_factor_receipts == ()
    assert result.form_factor_alpha_delta_fingerprint is None
    assert result.form_factor_beta_gamma_fingerprint is None


def test001_tolerance_is_diagnostic_but_exported_vertices_require_exact_momentum() -> None:
    alpha, beta, gamma, delta = _quartet_orbitals()
    exact = _kinematics(alpha, beta, gamma, delta, tolerance=1.0e-6)
    assert exact.residual_vector_inverse_angstrom == (0.0, 0.0)
    assert exact.residual_norm_inverse_angstrom == 0.0
    assert exact.within_declared_tolerance is True
    assert exact.require_conserving() is exact

    epsilon = 2.0e-12
    shifted_delta = _orbital(
        (
            delta.momentum_inverse_angstrom[0] + epsilon,
            delta.momentum_inverse_angstrom[1],
        )
    )
    within = _kinematics(
        alpha, beta, gamma, shifted_delta, tolerance=1.01 * epsilon
    )
    assert within.residual_vector_inverse_angstrom != (0.0, 0.0)
    assert within.residual_vector_inverse_angstrom == pytest.approx(
        (-epsilon, 0.0), abs=2.0e-18
    )
    assert within.within_declared_tolerance is True
    for evaluator in (
        vituri2024_ordered_projected_coefficient,
        vituri2024_antisymmetrized_projected_vertex,
    ):
        with pytest.raises(ValueError, match="exact local momentum conservation"):
            evaluator(within, _DELTA1, _interaction())
    with pytest.raises(ValueError, match="exact local momentum conservation"):
        within.require_conserving()

    outside = _kinematics(
        alpha, beta, gamma, shifted_delta, tolerance=0.99 * epsilon
    )
    assert outside.within_declared_tolerance is False
    with pytest.raises(ValueError, match="exact local momentum conservation"):
        outside.require_conserving()


def test001_strict_typed_inputs_hashes_and_tamper_fail_closed() -> None:
    for bad in (0, 2, -2, 1.0, True, np.int64(1), "1"):
        with pytest.raises(ValueError, match="exactly the integer"):
            Vituri2024Flavor(valley=bad, spin=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exactly the integer"):
            Vituri2024Flavor(valley=1, spin=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        Vituri2024Orbital(  # type: ignore[arg-type]
            Vituri2024Flavor(1, 1), [0.0, 0.0]
        )
    with pytest.raises(TypeError, match="Vituri2024Flavor"):
        Vituri2024Orbital((1, 1), (0.0, 0.0))  # type: ignore[arg-type]

    alpha, beta, gamma, delta = _quartet_orbitals()
    for bad_tolerance in (True, -1.0, np.nan, np.inf, 0.0 + 0.0j):
        with pytest.raises((TypeError, ValueError)):
            _kinematics(
                alpha, beta, gamma, delta, tolerance=bad_tolerance  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="provider_sha256"):
        _kinematics(alpha, beta, gamma, delta, provider_sha256="A" * 64)
    with pytest.raises(ValueError, match="provider_sha256"):
        _kinematics(alpha, beta, gamma, delta, provider_sha256="a" * 63)
    with pytest.raises(ValueError, match="must match"):
        _kinematics(
            alpha,
            beta,
            gamma,
            delta,
            derivation_source_sm_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="source_text"):
        _kinematics(alpha, beta, gamma, delta, source_text="  ")

    frozen = _kinematics(alpha, beta, gamma, delta)
    with pytest.raises(FrozenInstanceError):
        frozen.source_text = "changed"  # type: ignore[misc]
    object.__setattr__(frozen, "source_text", "forced tamper")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        frozen.require_conserving()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        vituri2024_ordered_projected_coefficient(
            frozen, _DELTA1, _interaction()
        )

    binding = bind_vituri2024_interaction(_interaction())
    object.__setattr__(binding, "receipt_fingerprint", "0" * 64)
    with pytest.raises(ValueError, match="binding fingerprint mismatch"):
        vituri2024_ordered_projected_coefficient(
            _kinematics(alpha, beta, gamma, delta), _DELTA1, binding
        )


def test001_exact_ket_bra_antisymmetry_and_pair_hermiticity() -> None:
    alpha, beta, gamma, delta = _quartet_orbitals()
    kinematics = _kinematics(alpha, beta, gamma, delta)
    interaction = _interaction()
    base = vituri2024_antisymmetrized_projected_vertex(
        kinematics, _DELTA1, interaction
    )
    ket_swapped = vituri2024_antisymmetrized_projected_vertex(
        _permuted(kinematics, alpha, beta, delta, gamma),
        _DELTA1,
        interaction,
    )
    bra_swapped = vituri2024_antisymmetrized_projected_vertex(
        _permuted(kinematics, beta, alpha, gamma, delta),
        _DELTA1,
        interaction,
    )
    pair_swapped = vituri2024_antisymmetrized_projected_vertex(
        _permuted(kinematics, gamma, delta, alpha, beta),
        _DELTA1,
        interaction,
    )

    for result in (base, ket_swapped, bra_swapped, pair_swapped):
        assert result.kinematics.residual_vector_inverse_angstrom == (0.0, 0.0)
        assert result.pauli_short_circuit is None
    assert ket_swapped.value == pytest.approx(-base.value, rel=2.0e-13, abs=2.0e-13)
    assert bra_swapped.value == pytest.approx(-base.value, rel=2.0e-13, abs=2.0e-13)
    assert pair_swapped.value == pytest.approx(
        base.value.conjugate(), rel=2.0e-13, abs=2.0e-13
    )


def test001_pauli_short_circuit_precedes_vtf_and_form_factors_and_fails_closed() -> None:
    interaction = _interaction("reject")
    same = _orbital((5.0 / 1024.0, -3.0 / 1024.0))
    all_identical = vituri2024_antisymmetrized_projected_vertex(
        _kinematics(same, same, same, same), _DELTA1, interaction
    )
    assert all_identical.value == 0.0j
    assert all_identical.raw_ordered_difference_ev_angstrom_squared is None
    assert all_identical.direct_ordered is None
    assert all_identical.exchange_ordered is None
    assert all_identical.pauli_short_circuit is not None
    assert all_identical.pauli_short_circuit.reason == (
        "both_bra_and_ket_pairs_repeated"
    )
    assert all_identical.pauli_short_circuit.vtf_evaluated is False
    assert all_identical.pauli_short_circuit.form_factors_evaluated is False

    repeated_ket = _orbital((4.0 / 1024.0, 2.0 / 1024.0))
    repeated_alpha = _orbital((12.0 / 1024.0, -4.0 / 1024.0))
    repeated_beta = _orbital(
        (
            2.0 * repeated_ket.momentum_inverse_angstrom[0]
            - repeated_alpha.momentum_inverse_angstrom[0],
            2.0 * repeated_ket.momentum_inverse_angstrom[1]
            - repeated_alpha.momentum_inverse_angstrom[1],
        )
    )
    ket_pauli = vituri2024_antisymmetrized_projected_vertex(
        _kinematics(repeated_alpha, repeated_beta, repeated_ket, repeated_ket),
        _DELTA1,
        interaction,
    )
    assert ket_pauli.pauli_short_circuit is not None
    assert ket_pauli.pauli_short_circuit.reason == "gamma_equals_delta"
    assert ket_pauli.direct_ordered is None
    assert ket_pauli.exchange_ordered is None

    repeated_bra = _orbital((3.0 / 1024.0, 4.0 / 1024.0))
    bra_gamma = _orbital((8.0 / 1024.0, -2.0 / 1024.0))
    bra_delta = _orbital(
        (
            2.0 * repeated_bra.momentum_inverse_angstrom[0]
            - bra_gamma.momentum_inverse_angstrom[0],
            2.0 * repeated_bra.momentum_inverse_angstrom[1]
            - bra_gamma.momentum_inverse_angstrom[1],
        )
    )
    bra_pauli = vituri2024_antisymmetrized_projected_vertex(
        _kinematics(repeated_bra, repeated_bra, bra_gamma, bra_delta),
        _DELTA1,
        interaction,
    )
    assert bra_pauli.pauli_short_circuit is not None
    assert bra_pauli.pauli_short_circuit.reason == "alpha_equals_beta"
    assert bra_pauli.direct_ordered is None
    assert bra_pauli.exchange_ordered is None

    alpha, beta, gamma, delta = _quartet_orbitals()
    non_pauli = vituri2024_antisymmetrized_projected_vertex(
        _kinematics(alpha, beta, gamma, delta), _DELTA1, interaction
    )
    assert non_pauli.direct_ordered is not None
    with pytest.raises(ValueError, match="requires a typed Pauli short circuit"):
        replace(all_identical, pauli_short_circuit=None)
    with pytest.raises(ValueError, match="forbids ordered-term receipts"):
        replace(all_identical, direct_ordered=non_pauli.direct_ordered)
    with pytest.raises(ValueError, match="forbids a raw ordered difference"):
        replace(all_identical, raw_ordered_difference_ev_angstrom_squared=0.0j)


def test001_complex_vertex_obeys_gauge_phase_law_not_invariance() -> None:
    states = (
        np.array([1.0, 0.2j, -0.3, 0.1 + 0.4j, 0.7, -0.2j]),
        np.array([0.1j, 1.2, 0.4 - 0.2j, -0.5, 0.3j, 0.8]),
        np.array([0.6, -0.1j, 1.1, 0.2 + 0.3j, -0.4j, 0.5]),
        np.array([0.3 - 0.2j, 0.4, -0.6j, 1.0, 0.2, 0.7j]),
    )
    direct_kernel = 2.3
    exchange_kernel = 1.4
    base = _state_level_vertex_oracle(states, direct_kernel, exchange_kernel)
    phases = (0.31, -0.47, 0.83, -0.19)
    transformed_states = tuple(
        np.exp(1j * phase) * state for phase, state in zip(phases, states)
    )
    transformed = _state_level_vertex_oracle(
        transformed_states, direct_kernel, exchange_kernel
    )
    expected_phase = np.exp(
        -1j * (phases[0] + phases[1]) + 1j * (phases[2] + phases[3])
    )

    assert abs(base) > 1.0e-3
    assert abs(expected_phase - 1.0) > 0.1
    assert transformed == pytest.approx(
        expected_phase * base, rel=2.0e-14, abs=2.0e-15
    )
    assert transformed != pytest.approx(base, rel=1.0e-6, abs=1.0e-6)


def test001_literal_printed_c3_repeated_product_breaks_antisymmetry() -> None:
    states = (
        np.array([1.0, 0.3j, -0.2, 0.5 + 0.1j, 0.7, -0.4j]),
        np.array([0.2, 1.0, 0.6j, -0.3 + 0.2j, 0.4, 0.8j]),
        np.array([0.7j, -0.1, 1.0, 0.3, -0.5j, 0.6]),
        np.array([0.4, 0.2 - 0.3j, -0.7j, 1.0, 0.5, 0.1]),
    )
    k_alpha = np.array([0.017, -0.006])
    k_beta = np.array([-0.010, 0.013])
    k_gamma = np.array([0.006, 0.011])
    k_delta = k_alpha + k_beta - k_gamma

    def kernel(q: float) -> float:
        return 1.0 + 0.7 * q + 0.2 * q * q

    def overlaps(
        ordered_states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[complex, complex, complex, complex]:
        alpha, beta, gamma, delta = ordered_states
        return (
            state_overlap_invariant(alpha, delta).value,
            state_overlap_invariant(beta, gamma).value,
            state_overlap_invariant(alpha, gamma).value,
            state_overlap_invariant(beta, delta).value,
        )

    def derived(
        ordered_states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        momenta: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> complex:
        f_ad, f_bg, f_ag, f_bd = overlaps(ordered_states)
        _, k_beta_local, k_gamma_local, k_delta_local = momenta
        return (
            kernel(float(np.linalg.norm(k_beta_local - k_gamma_local)))
            * f_ad
            * f_bg
            - kernel(float(np.linalg.norm(k_beta_local - k_delta_local)))
            * f_ag
            * f_bd
        )

    def literal_repeated_product_candidate(
        ordered_states: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        momenta: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> complex:
        f_ad, f_bg, _, _ = overlaps(ordered_states)
        k_alpha_local, _, k_gamma_local, k_delta_local = momenta
        return f_ad * f_bg * (
            kernel(float(np.linalg.norm(k_alpha_local - k_delta_local)))
            - kernel(float(np.linalg.norm(k_alpha_local - k_gamma_local)))
        )

    momenta = (k_alpha, k_beta, k_gamma, k_delta)
    ket_swapped_states = (states[0], states[1], states[3], states[2])
    ket_swapped_momenta = (k_alpha, k_beta, k_delta, k_gamma)
    derived_base = derived(states, momenta)
    derived_ket_swapped = derived(ket_swapped_states, ket_swapped_momenta)
    literal_base = literal_repeated_product_candidate(states, momenta)
    literal_ket_swapped = literal_repeated_product_candidate(
        ket_swapped_states, ket_swapped_momenta
    )

    assert derived_ket_swapped == pytest.approx(
        -derived_base, rel=2.0e-14, abs=2.0e-15
    )
    assert abs(literal_base + literal_ket_swapped) > 1.0e-5


def test001_receipts_bind_inputs_and_do_not_inflate_scope_or_readiness() -> None:
    alpha, beta, gamma, delta = _quartet_orbitals()
    kinematics = _kinematics(alpha, beta, gamma, delta)
    binding = bind_vituri2024_interaction(_interaction())
    result = vituri2024_antisymmetrized_projected_vertex(
        kinematics, _DELTA1, binding
    )

    assert result.direct_ordered is not None
    assert result.exchange_ordered is not None
    assert result.pauli_short_circuit is None
    assert result.delta1_ev == _DELTA1
    assert result.kinematics == kinematics
    assert result.kinematics_fingerprint == kinematics.fingerprint
    assert result.interaction_receipt == binding.receipt
    assert result.interaction_receipt_fingerprint == binding.receipt.fingerprint
    assert result.interaction_binding_fingerprint == binding.receipt_fingerprint
    assert result.direct_ordered.delta1_ev == _DELTA1
    assert result.direct_ordered.kinematics_fingerprint == kinematics.fingerprint
    assert result.exchange_ordered.kinematics == kinematics.ket_swapped()
    assert result.direct_ordered.form_factor_alpha_delta is not None
    assert result.direct_ordered.form_factor_beta_gamma is not None
    assert result.direct_ordered.transfer_vector_inverse_angstrom == (
        beta.momentum_inverse_angstrom[0] - gamma.momentum_inverse_angstrom[0],
        beta.momentum_inverse_angstrom[1] - gamma.momentum_inverse_angstrom[1],
    )
    assert kinematics.provider_sha256 == _PROVIDER_SHA256
    assert kinematics.derivation_source_sm_sha256 == SM_TEX_SHA256
    assert kinematics.source_text == _SOURCE_TEXT
    assert kinematics.provider_metadata_status == KINEMATICS_PROVIDER_METADATA_STATUS
    assert KINEMATICS_PROVIDER_METADATA_STATUS == (
        "caller_attested_quartet_and_tolerance_not_independently_verified"
    )
    assert result.derivation_source_sm_sha256 == SM_TEX_SHA256
    assert result.published_prb_pdf_sha256 == PUBLISHED_PRB_PDF_SHA256
    assert PUBLISHED_PRB_PDF_SHA256 == (
        "2226e17ed95bd867607787b47343fe5fc77f2c30557023e349d86e55159c0765"
    )
    assert result.authority == (
        "derived_from_projected_H_not_literal_C3_internally_inconsistent"
    )
    assert result.authority == VERTEX_AUTHORITY
    assert result.gauge_behavior == VERTEX_GAUGE_BEHAVIOR
    assert result.exclusions == ANTISYMMETRIZED_VERTEX_EXCLUSIONS
    assert result.direct_ordered.exclusions == ORDERED_COEFFICIENT_EXCLUSIONS
    assert result.production_numerical_parity is False
    assert result.paper_numerical_parity is False
    assert result.includes_area_prefactor is False
    assert result.includes_momentum_delta is False
    assert result.includes_occupation_factors is False
    assert result.establishes_hf_q0_background is False
    assert kinematics.reciprocal_torus_authority is False
    assert kinematics.reciprocal_carry_authority is False
    for receipt in (kinematics, result, result.direct_ordered):
        assert not hasattr(receipt, "executable_ready")
        assert not hasattr(receipt, "area")
        assert not hasattr(receipt, "occupations")
    for module in (vertex_module, abc_trilayer_module):
        assert not hasattr(module, "projected_four_point_vertex")
        assert not hasattr(module, "ordered_projected_coefficient")
        assert not hasattr(module, "antisymmetrized_projected_vertex")
    assert not hasattr(vertex_module, "executable_ready")
    assert not hasattr(vertex_module, "hartree_fock")
    assert not hasattr(vertex_module, "tdhf")
    module_doc = vertex_module.__doc__ or ""
    package_doc = abc_trilayer_module.__doc__ or ""
    assert "does *not* copy printed Eq. C3" in module_doc
    assert "literal_C3_internally_inconsistent" in module_doc
    assert "gauge *covariant*, not gauge invariant" in module_doc
    assert "production or paper" in module_doc
    assert "1/(2A)" in module_doc
    assert "1/(4A)" in module_doc
    assert "momentum delta" in module_doc
    assert "caller-attested" in module_doc
    assert "not independently verified" in module_doc
    assert PUBLISHED_PRB_PDF_SHA256 in module_doc
    assert PUBLISHED_PRB_PDF_SHA256 in package_doc
    assert "occupation" in module_doc
    assert "executable" in module_doc
    assert "readiness" in module_doc
    assert "Delta1" in inspect.signature(
        vituri2024_antisymmetrized_projected_vertex
    ).parameters
