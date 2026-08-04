from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import TDHFConfig, analyze_tdhf_sector, run_tdhf
from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQSector,
    TDHFSelfConjugateQSector,
    build_tdhf_self_conjugate_matrices,
    certify_tdhf_ward_identity,
    fingerprint_tdhf_sector,
)
from mean_field.systems.hubbard_1d import (
    AlaviradSauHubbard1DModel,
    AlaviradSauHubbardTDHFProvider,
    apply_saturated_ferromagnet_interspin_tdhf_action,
    build_exact_one_magnon_hamiltonian,
    build_exact_one_magnon_hamiltonian_bitstring,
    fit_exact_one_magnon_spin_stiffness,
    saturated_ferromagnet_stationarity_residual,
    spin_lowering_nambu_generator,
)
import mean_field.systems.hubbard_1d.tdhf_alavirad_sau as benchmark_module


def _model() -> AlaviradSauHubbard1DModel:
    return AlaviradSauHubbard1DModel(
        site_count=12,
        hopping=1.0,
        interaction=10.0,
    )


def _config(raw_q: int) -> TDHFConfig:
    return TDHFConfig(
        q_sector=(raw_q, 0),
        channel="interspin",
        structure_tolerance=1.0e-11,
        hessian_tolerance=1.0e-11,
        imag_tolerance=1.0e-11,
        norm_tolerance=1.0e-11,
        zero_tolerance=1.0e-11,
        degeneracy_tolerance=1.0e-11,
        pairing_tolerance=1.0e-10,
        eigensolver_tolerance=1.0e-10,
        metric_gram_tolerance=1.0e-10,
    )


def test_alavirad_sau_literal_eq14_hopping_sign_and_matrix_entry() -> None:
    model = _model()
    exact = build_exact_one_magnon_hamiltonian(model, 1)
    expected_00 = (
        2.0 * model.hopping * np.cos(2.0 * np.pi / model.site_count)
        - 2.0 * model.hopping
        + model.interaction
        - model.interaction / model.site_count
    )
    np.testing.assert_allclose(exact[0, 0], expected_00, atol=1.0e-14)
    np.testing.assert_allclose(
        exact[0, 1], -model.interaction / model.site_count, atol=1.0e-14
    )


def test_alavirad_sau_bitstring_projection_is_independent_exact_oracle() -> None:
    model = AlaviradSauHubbard1DModel(
        site_count=6,
        hopping=0.7,
        interaction=3.2,
    )
    for raw_q in range(model.site_count):
        analytical = build_exact_one_magnon_hamiltonian(model, raw_q)
        bitstring = build_exact_one_magnon_hamiltonian_bitstring(model, raw_q)
        np.testing.assert_allclose(bitstring, analytical, rtol=0.0, atol=2.0e-14)


def test_alavirad_sau_exact_q0_spin_goldstone() -> None:
    model = _model()
    exact = build_exact_one_magnon_hamiltonian(model, 0)
    uniform = np.ones(model.site_count) / np.sqrt(model.site_count)
    np.testing.assert_allclose(exact @ uniform, 0.0, atol=1.0e-13)
    eigenvalues = np.linalg.eigvalsh(exact)
    assert abs(eigenvalues[0]) < 1.0e-12
    np.testing.assert_allclose(
        eigenvalues[1:],
        model.interaction,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_alavirad_sau_exact_and_interspin_tdhf_actions_match_all_q() -> None:
    model = _model()
    identity = np.eye(model.site_count, dtype=np.complex128)
    for raw_q in range(model.site_count):
        exact = build_exact_one_magnon_hamiltonian(model, raw_q)
        action = np.column_stack(
            [
                apply_saturated_ferromagnet_interspin_tdhf_action(
                    model, raw_q, identity[:, column]
                )
                for column in range(model.site_count)
            ]
        )
        np.testing.assert_allclose(action, exact, rtol=0.0, atol=1.0e-14)


def test_alavirad_sau_provider_does_not_call_exact_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("TDHF adapter called the exact one-magnon oracle")

    monkeypatch.setattr(benchmark_module, "build_exact_one_magnon_hamiltonian", forbidden)
    sector = AlaviradSauHubbardTDHFProvider(model).build_tdhf_sector(_config(1))
    assert isinstance(sector, TDHFGenericSignedQSector)


def test_alavirad_sau_generic_q_api_matches_exact_one_magnon_spectra() -> None:
    model = _model()
    provider = AlaviradSauHubbardTDHFProvider(model)
    analysis = run_tdhf(provider, _config(1))
    assert isinstance(analysis.sector, TDHFGenericSignedQSector)
    exact_plus = build_exact_one_magnon_hamiltonian(model, 1)
    exact_minus = build_exact_one_magnon_hamiltonian(model, -1)
    np.testing.assert_allclose(
        analysis.sector.blocks.A_plus,
        exact_plus,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        analysis.sector.blocks.A_minus,
        exact_minus,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sort(analysis.assignment.plus_energies),
        np.linalg.eigvalsh(exact_plus),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sort(analysis.assignment.minus_energies),
        np.linalg.eigvalsh(exact_minus),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert analysis.dynamic.kind == "real"
    assert analysis.static.kind == "indefinite"
    assert analysis.assignment.plus_energies[0] < 0.0 or np.min(
        analysis.assignment.plus_energies
    ) < 0.0


def test_alavirad_sau_q0_ward_certificate_uses_spin_lowering_generator() -> None:
    model = _model()
    provider = AlaviradSauHubbardTDHFProvider(model)
    config = _config(0)
    sector = provider.build_tdhf_sector(config)
    assert isinstance(sector, TDHFSelfConjugateQSector)
    matrices = build_tdhf_self_conjugate_matrices(
        sector,
        structure_tolerance=config.structure_tolerance,
    )
    ward = certify_tdhf_ward_identity(
        hessian=matrices.H,
        liouvillian=matrices.L,
        generator=spin_lowering_nambu_generator(model),
        generator_label="total-spin-lowering S_minus",
        generator_provenance="Alavirad-Sau Eq.13 q=0 uniform spin flip",
        source_fingerprint=model.source_fingerprint,
        expected_source_fingerprint=sector.source_fingerprint,
        interaction_fingerprint=sector.interaction_fingerprint,
        sector_fingerprint=fingerprint_tdhf_sector(sector),
        response_scope=sector.response_scope,
        static_hessian_authority="scalar_hessian",
        source_stationarity_residual=saturated_ferromagnet_stationarity_residual(model),
        source_stationarity_tolerance=1.0e-11,
        action_tolerance=1.0e-11,
        null_eigenvalue_tolerance=config.hessian_tolerance,
        overlap_tolerance=1.0 - 1.0e-12,
    )
    assert saturated_ferromagnet_stationarity_residual(model) == 0.0
    assert sector.source_fingerprint != sector.interaction_fingerprint
    assert ward.passed
    analysis = analyze_tdhf_sector(sector, config, ward_certificate=ward)
    assert analysis.static.kind == "positive_semidefinite"
    assert analysis.static.zero_count == 2
    assert analysis.dynamic.kind == "real"
    assert analysis.zero_mode.origin == "ward_static_null"
    assert analysis.zero_mode.ward_passed


def test_alavirad_sau_first_finite_q_mode_and_stiffness_are_negative() -> None:
    model = _model()
    q0 = np.linalg.eigvalsh(build_exact_one_magnon_hamiltonian(model, 0))[0]
    q1 = np.linalg.eigvalsh(build_exact_one_magnon_hamiltonian(model, 1))[0]
    stiffness = fit_exact_one_magnon_spin_stiffness(
        model,
        max_positive_q_index=2,
    )
    assert abs(q0) < 1.0e-12
    assert q1 < 0.0
    assert stiffness.stiffness < 0.0
    assert np.all(stiffness.lowest_energies < 0.0)
    assert stiffness.fit_residual < 0.02


def test_alavirad_sau_large_u_stiffness_limit() -> None:
    model = AlaviradSauHubbard1DModel(
        site_count=120,
        hopping=1.0,
        interaction=100.0,
    )
    stiffness = fit_exact_one_magnon_spin_stiffness(
        model,
        max_positive_q_index=1,
    )
    np.testing.assert_allclose(
        stiffness.stiffness,
        -2.0 * model.hopping**2 / model.interaction,
        rtol=0.03,
        atol=0.0,
    )


def test_alavirad_sau_exact_boundary_keeps_raw_alias_diagnostic() -> None:
    model = _model()
    sector = AlaviradSauHubbardTDHFProvider(model).build_tdhf_sector(
        _config(model.site_count // 2)
    )
    assert isinstance(sector, TDHFSelfConjugateQSector)
    assert sector.q.plus_raw != sector.q.minus_raw
    matrices = build_tdhf_self_conjugate_matrices(sector)
    assert matrices.raw_signed_diagnostic is not None
    assert matrices.raw_signed_diagnostic.structure.ok
    np.testing.assert_array_equal(
        matrices.raw_signed_diagnostic.H_plus,
        matrices.raw_signed_diagnostic.H_minus,
    )
