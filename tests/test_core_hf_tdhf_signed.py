from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from mean_field.api.tdhf import TDHFConfig, run_tdhf
from mean_field.core.hf import (
    ParticleHolePair,
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFNambuSewing,
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    TDHFSignedQBlocks,
    analyze_tdhf_typed_sector,
    build_standard_nambu_sewing,
    build_tdhf_self_conjugate_matrices,
    build_tdhf_signed_q_matrices,
    certify_tdhf_ward_identity,
    classify_tdhf_signed_q,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
    solve_tdhf_wang_signed_modes,
)


def _pairs(n: int, *, offset: int = 0) -> tuple[ParticleHolePair, ...]:
    return tuple(
        ParticleHolePair(
            particle=10 + offset + index,
            hole=offset + index,
            particle_momentum=(index, 1),
            hole_momentum=(index, 0),
        )
        for index in range(n)
    )


def _generic_sector(
    A_plus: np.ndarray,
    B_plus_minus: np.ndarray,
    A_minus: np.ndarray,
    B_minus_plus: np.ndarray,
    *,
    authority: str = "scalar_hessian",
) -> TDHFGenericSignedQSector:
    n_plus = A_plus.shape[0]
    n_minus = A_minus.shape[0]
    plus_pairs = _pairs(n_plus)
    minus_pairs = _pairs(n_minus, offset=100)
    blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=A_plus,
        B_plus_minus=B_plus_minus,
        A_minus=A_minus,
        B_minus_plus=B_minus_plus,
    )
    sewing = build_standard_nambu_sewing(
        plus_pairs,
        minus_pairs,
        source_fingerprint="source",
    )
    return TDHFGenericSignedQSector(
        q=TDHFGenericSignedQ(
            plus_raw=(1, 0),
            minus_raw=(-1, 0),
            plus_canonical=(1, 0),
            minus_canonical=(9, 0),
            provenance="test-grid-v1",
        ),
        blocks=blocks,
        sewing=sewing,
        source_fingerprint="source",
        interaction_fingerprint="interaction",
        response_scope="test-scalar-hessian",
        static_hessian_authority=authority,  # type: ignore[arg-type]
    )


def test_tdhf_q_type_invariants_fail_closed() -> None:
    with pytest.raises(ValueError, match="distinct canonical"):
        TDHFGenericSignedQ(
            plus_raw=(5, 0),
            minus_raw=(-5, 0),
            plus_canonical=(5, 0),
            minus_canonical=(5, 0),
            provenance="wrong-generic-exact-m",
        )
    with pytest.raises(ValueError, match="multiplicity one"):
        TDHFSelfConjugateQ(
            plus_raw=(0, 0),
            minus_raw=(0, 0),
            canonical=(0, 0),
            provenance="q0",
            orbit_multiplicity=2,
        )


def test_tdhf_sector_q_cross_wiring_fails_closed() -> None:
    valid = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    with pytest.raises(TypeError, match="requires TDHFGenericSignedQ"):
        TDHFGenericSignedQSector(
            **{
                **valid.__dict__,
                "q": TDHFSelfConjugateQ(
                    plus_raw=(0, 0),
                    minus_raw=(0, 0),
                    canonical=(0, 0),
                    provenance="cross-wired",
                ),
            }
        )


def test_tdhf_q_types_distinguish_generic_q0_and_raw_exact_m() -> None:
    generic = classify_tdhf_signed_q(
        plus_raw=(1, 0),
        minus_raw=(-1, 0),
        plus_canonical=(1, 0),
        minus_canonical=(9, 0),
        provenance="mesh10",
    )
    q0 = classify_tdhf_signed_q(
        plus_raw=(0, 0),
        minus_raw=(0, 0),
        plus_canonical=(0, 0),
        minus_canonical=(0, 0),
        provenance="mesh10",
    )
    exact_m = classify_tdhf_signed_q(
        plus_raw=(5, 0),
        minus_raw=(-5, 0),
        plus_canonical=(5, 0),
        minus_canonical=(5, 0),
        provenance="mesh10",
    )
    assert isinstance(generic, TDHFGenericSignedQ)
    assert generic.orbit_multiplicity == 2
    assert isinstance(q0, TDHFSelfConjugateQ)
    assert isinstance(exact_m, TDHFSelfConjugateQ)
    assert exact_m.plus_raw != exact_m.minus_raw
    assert exact_m.orbit_multiplicity == 1


def test_tdhf_signed_structure_uses_cross_partner_b_not_same_q_symmetry() -> None:
    A_plus = np.asarray([[2.0, 0.2j], [-0.2j, 3.0]])
    A_minus = np.asarray([[4.0, 0.3], [0.3, 5.0]])
    B_plus = np.asarray([[0.1, 0.2j], [0.4, -0.3j]])
    assert not np.allclose(B_plus, B_plus.T)
    sector = _generic_sector(A_plus, B_plus, A_minus, B_plus.T)
    matrices = build_tdhf_signed_q_matrices(
        sector.blocks, sector.sewing, raise_on_structure_error=True
    )
    assert matrices.structure.ok
    assert matrices.structure.B_partner_transpose == 0.0
    assert matrices.structure.signed_liouvillian_covariance < 1.0e-12


def test_tdhf_complex_phase_sewing_closes_covariance_and_metric() -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[3.0]]),
        np.asarray([[0.0]]),
    )
    phase = np.exp(0.37j)
    plus_to_minus = np.asarray([[0.0, phase], [np.conj(phase), 0.0]])
    minus_to_plus = plus_to_minus.T
    sewing = TDHFNambuSewing(
        plus_to_minus=plus_to_minus,
        minus_to_plus=minus_to_plus,
        source_fingerprint="source",
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(sector.blocks.plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(sector.blocks.minus_pairs),
        construction="complex-phase-block-swap-test",
        closure_residual=0.0,
    )
    matrices = build_tdhf_signed_q_matrices(
        sector.blocks, sewing, raise_on_structure_error=True
    )
    assert matrices.structure.signed_liouvillian_covariance < 1.0e-12
    assert matrices.structure.reverse_signed_liouvillian_covariance < 1.0e-12
    assert matrices.structure.sewing_metric_anticovariance < 1.0e-12


def test_tdhf_wang_degenerate_eta_gram_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sector = _generic_sector(
        np.diag([2.0, 2.0]),
        np.zeros((2, 2)),
        np.diag([3.0, 3.0]),
        np.zeros((2, 2)),
    )
    matrices = build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    values = np.asarray([2.0, 2.0, -3.0, -3.0], dtype=np.complex128)
    vectors = np.asarray(
        [
            [1.0, 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0 / np.sqrt(2.0)],
            [0.0, 0.0, 0.0, 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )

    def fake_eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        np.testing.assert_allclose(matrix, matrices.L_plus)
        return values.copy(), vectors.copy()

    monkeypatch.setattr(
        "mean_field.core.hf.tdhf_signed.scipy_linalg.eig", fake_eig
    )
    result = solve_tdhf_wang_signed_modes(matrices, degeneracy_tol=1.0e-12)
    assert result.plus_energies.size == 2
    assert result.minus_energies.size == 2
    assert result.metric_gram_residual < 1.0e-12
    assert np.max(result.plus_residuals, initial=0.0) < 1.0e-12
    assert np.max(result.minus_residuals, initial=0.0) < 1.0e-12


def test_tdhf_wang_assignment_retains_all_norm_sign_energy_cases() -> None:
    sector = _generic_sector(
        np.diag([2.0, -1.0]),
        np.zeros((2, 2)),
        np.diag([3.0, -4.0]),
        np.zeros((2, 2)),
    )
    matrices = build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    result = solve_tdhf_wang_signed_modes(matrices)
    np.testing.assert_allclose(np.sort(result.plus_energies), [-1.0, 2.0])
    np.testing.assert_allclose(np.sort(result.minus_energies), [-4.0, 3.0])
    assert result.plus_raw_indices.size == 2
    assert result.minus_raw_indices.size == 2
    assert result.complex_indices.size == 0
    assert result.null_metric_indices.size == 0


def test_tdhf_wang_zero_modes_remain_unassigned() -> None:
    sector = _generic_sector(
        np.asarray([[0.0]]),
        np.asarray([[0.0]]),
        np.asarray([[0.0]]),
        np.asarray([[0.0]]),
    )
    result = solve_tdhf_wang_signed_modes(
        build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    )
    assert result.zero_eigenvalue_indices.size == 2
    assert result.plus_raw_indices.size == 0
    assert result.minus_raw_indices.size == 0


def test_tdhf_wang_assignment_leaves_complex_null_metric_modes_raw() -> None:
    sector = _generic_sector(
        np.asarray([[0.5]]),
        np.asarray([[1.0]]),
        np.asarray([[0.5]]),
        np.asarray([[1.0]]),
    )
    result = solve_tdhf_wang_signed_modes(
        build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    )
    assert result.complex_indices.size == 2
    assert result.null_metric_indices.size == 2
    assert result.plus_raw_indices.size == 0
    assert result.minus_raw_indices.size == 0


def test_tdhf_nonfinite_retained_assignment_residual_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    matrices = build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    assignment = solve_tdhf_wang_signed_modes(matrices)
    poisoned = replace(
        assignment,
        raw_residuals=np.full_like(assignment.raw_residuals, np.nan),
    )
    monkeypatch.setattr(
        "mean_field.core.hf.tdhf_signed.solve_tdhf_wang_signed_modes",
        lambda *args, **kwargs: poisoned,
    )
    analysis = analyze_tdhf_typed_sector(sector)
    assert analysis.dynamic.kind == "invalid"


def test_tdhf_metric_gram_gate_is_distinct_from_null_norm_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    matrices = build_tdhf_signed_q_matrices(sector.blocks, sector.sewing)
    assignment = replace(
        solve_tdhf_wang_signed_modes(matrices),
        metric_gram_residual=5.0e-10,
    )
    monkeypatch.setattr(
        "mean_field.core.hf.tdhf_signed.solve_tdhf_wang_signed_modes",
        lambda *args, **kwargs: assignment,
    )
    accepted = analyze_tdhf_typed_sector(
        sector,
        norm_tolerance=1.0e-10,
        metric_gram_tolerance=1.0e-9,
    )
    rejected = analyze_tdhf_typed_sector(
        sector,
        norm_tolerance=1.0e-10,
        metric_gram_tolerance=1.0e-11,
    )
    assert accepted.dynamic.kind == "real"
    assert rejected.dynamic.kind == "invalid"
    assert accepted.dynamic.metric_gram_tolerance == 1.0e-9


def test_tdhf_static_and_dynamic_statuses_are_independent() -> None:
    sector = _generic_sector(
        np.asarray([[0.5]]),
        np.asarray([[1.0]]),
        np.asarray([[0.5]]),
        np.asarray([[1.0]]),
    )
    analysis = analyze_tdhf_typed_sector(sector)
    assert analysis.static.kind == "indefinite"
    assert analysis.dynamic.kind == "complex"
    assert analysis.zero_mode.origin == "none"


def test_tdhf_sewing_pair_provenance_mismatch_fails_closed() -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    object.__setattr__(sector.sewing, "plus_pairs_fingerprint", "tampered")
    with np.testing.assert_raises_regex(ValueError, "pair fingerprints"):
        analyze_tdhf_typed_sector(sector)


def test_tdhf_projected_signed_blocks_do_not_claim_scalar_static_status() -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        authority="projected_signed_ab",
    )
    analysis = analyze_tdhf_typed_sector(sector)
    assert analysis.static.kind == "not_established"
    assert analysis.dynamic.kind == "real"


def test_tdhf_complex_dynamics_can_coexist_with_ward_static_null() -> None:
    pairs = _pairs(1)
    sector = TDHFSelfConjugateQSector(
        q=TDHFSelfConjugateQ(
            plus_raw=(0, 0),
            minus_raw=(0, 0),
            canonical=(0, 0),
            provenance="q0",
        ),
        canonical_pairs=pairs,
        A=np.asarray([[1.0]]),
        B=np.asarray([[1.0 + 1.0e-12]]),
        source_fingerprint="source",
        interaction_fingerprint="interaction",
        response_scope="q0-scalar",
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance="literal-q0-basis",
    )
    matrices = build_tdhf_self_conjugate_matrices(sector)
    generator = np.asarray([1.0, -1.0])
    ward = certify_tdhf_ward_identity(
        hessian=matrices.H,
        liouvillian=matrices.L,
        generator=generator,
        generator_label="spin-lowering",
        generator_provenance="exact-su2-generator",
        source_fingerprint="source",
        expected_source_fingerprint="source",
        interaction_fingerprint="interaction",
        sector_fingerprint=fingerprint_tdhf_sector(sector),
        response_scope="q0-scalar",
        static_hessian_authority="scalar_hessian",
        source_stationarity_residual=0.0,
        source_stationarity_tolerance=1.0e-10,
        action_tolerance=1.0e-10,
        null_eigenvalue_tolerance=1.0e-10,
        overlap_tolerance=0.999999,
    )
    assert ward.passed
    analysis = analyze_tdhf_typed_sector(
        sector,
        hessian_tolerance=1.0e-10,
        imag_tolerance=1.0e-10,
        ward=ward,
    )
    assert analysis.static.kind == "positive_semidefinite"
    assert analysis.dynamic.kind == "complex"
    assert analysis.zero_mode.origin == "ward_static_null"
    assert analysis.assignment is not None

    same_matrices_other_q = TDHFSelfConjugateQSector(
        **{
            **sector.__dict__,
            "q": TDHFSelfConjugateQ(
                plus_raw=(5, 0),
                minus_raw=(-5, 0),
                canonical=(5, 0),
                provenance="different-q-same-matrices",
            ),
        }
    )
    with np.testing.assert_raises_regex(ValueError, "sector_fingerprint"):
        analyze_tdhf_typed_sector(
            same_matrices_other_q,
            hessian_tolerance=1.0e-10,
            imag_tolerance=1.0e-10,
            ward=ward,
        )
    forged = replace(
        ward,
        passed=True,
        failure_reasons=(),
        source_stationarity_residual=1.0,
    )
    with np.testing.assert_raises_regex(ValueError, "certificate_gate_recheck"):
        analyze_tdhf_typed_sector(
            sector,
            hessian_tolerance=1.0e-10,
            imag_tolerance=1.0e-10,
            ward=forged,
        )


def test_tdhf_self_conjugate_raw_signed_diagnostics_are_separate() -> None:
    plus_pairs = _pairs(1)
    minus_pairs = _pairs(1, offset=100)
    raw_blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=np.asarray([[2.0]]),
        B_plus_minus=np.asarray([[0.2]]),
        A_minus=np.asarray([[2.0]]),
        B_minus_plus=np.asarray([[0.3]]),
    )
    raw_sewing = build_standard_nambu_sewing(
        plus_pairs, minus_pairs, source_fingerprint="source"
    )
    sector = TDHFSelfConjugateQSector(
        q=TDHFSelfConjugateQ(
            plus_raw=(5, 0),
            minus_raw=(-5, 0),
            canonical=(5, 0),
            provenance="mesh10-exact-m",
        ),
        canonical_pairs=_pairs(1),
        A=np.asarray([[2.0]]),
        B=np.asarray([[0.0]]),
        source_fingerprint="source",
        interaction_fingerprint="interaction",
        response_scope="exact-m-canonical-scalar",
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance="explicit-exact-m-sewing",
        raw_signed_diagnostic_blocks=raw_blocks,
        raw_signed_diagnostic_sewing=raw_sewing,
    )
    matrices = build_tdhf_self_conjugate_matrices(sector)
    assert matrices.structure.ok
    assert matrices.raw_signed_diagnostic is not None
    assert not matrices.raw_signed_diagnostic.structure.ok

    missing_sewing = TDHFSelfConjugateQSector(
        **{**sector.__dict__, "raw_signed_diagnostic_sewing": None}
    )
    with np.testing.assert_raises_regex(ValueError, "both blocks and sewing"):
        build_tdhf_self_conjugate_matrices(missing_sewing)


def test_tdhf_exact_m_uses_canonical_self_conjugate_payload() -> None:
    sector = TDHFSelfConjugateQSector(
        q=TDHFSelfConjugateQ(
            plus_raw=(5, 0),
            minus_raw=(-5, 0),
            canonical=(5, 0),
            provenance="mesh10-exact-m",
        ),
        canonical_pairs=_pairs(1),
        A=np.asarray([[2.0]]),
        B=np.asarray([[0.0]]),
        source_fingerprint="source",
        interaction_fingerprint="interaction",
        response_scope="exact-m-canonical-scalar",
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance="explicit-exact-m-sewing",
    )
    analysis = analyze_tdhf_typed_sector(sector)
    assert analysis.static.kind == "positive_definite"
    assert analysis.dynamic.kind == "real"
    np.testing.assert_allclose(analysis.assignment.energies, [2.0])


@dataclass
class _Provider:
    sector: TDHFGenericSignedQSector

    def build_tdhf_sector(self, config: object, **kwargs: object) -> TDHFGenericSignedQSector:
        assert isinstance(config, TDHFConfig)
        assert kwargs == {"adapter_receipt": "ok"}
        return self.sector


@dataclass
class _LegacyAndTypedProvider(_Provider):
    legacy_called: bool = False

    def run_tdhf(self, config: TDHFConfig, **kwargs: object) -> str:
        self.legacy_called = True
        assert kwargs == {"legacy_receipt": "ok"}
        return "legacy"


def test_tdhf_config_preserves_legacy_positional_metadata_and_dispatch() -> None:
    config = TDHFConfig("q0", "all", 5, 1.0, "auto", {"legacy": True})
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    provider = _LegacyAndTypedProvider(sector)
    assert run_tdhf(provider, config, legacy_receipt="ok") == "legacy"
    assert provider.legacy_called
    assert config.metadata == {"legacy": True}


def test_public_run_tdhf_consumes_typed_provider_api() -> None:
    sector = _generic_sector(
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
        np.asarray([[2.0]]),
        np.asarray([[0.0]]),
    )
    result = run_tdhf(
        _Provider(sector),
        TDHFConfig(q_sector=(1, 0), channel="interspin"),
        adapter_receipt="ok",
    )
    assert result.static.kind == "positive_definite"
    assert result.dynamic.kind == "real"
    np.testing.assert_allclose(result.assignment.plus_energies, [2.0])
    np.testing.assert_allclose(result.assignment.minus_energies, [2.0])
