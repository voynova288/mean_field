from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mean_field.api import TDHFConfig, analyze_tdhf_sector
from mean_field.core.hf import ParticleHolePair
from mean_field.core.hf.tdhf_goldstone import (
    TDHFSymplecticProvenance,
    analyze_tdhf_goldstone_symplectic_matrix,
    certify_tdhf_ward_subspace,
    count_tdhf_goldstones_from_rank,
    validate_tdhf_ward_subspace_certificate,
)
from mean_field.core.hf.tdhf_signed import (
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    build_tdhf_self_conjugate_matrices,
    fingerprint_tdhf_sector,
)


def _rho_provenance(
    labels: tuple[str, ...],
    *,
    source: str = "source",
    interaction: str = "interaction",
    sector: str = "sector",
    scope: str = "toy_scalar_hessian",
) -> TDHFSymplecticProvenance:
    return TDHFSymplecticProvenance(
        commutator_definition="rho_ab=-i<[Qa,Qb]>/volume",
        volume_normalization="unit test volume=1",
        generator_basis_order=labels,
        source_fingerprint=source,
        interaction_fingerprint=interaction,
        sector_fingerprint=sector,
        response_scope=scope,
        notes="ordered physical broken-charge basis",
    )


def test_goldstone_count_distinguishes_static_directions_and_branches() -> None:
    neutral = count_tdhf_goldstones_from_rank(4, 0)
    assert (neutral.type_a_count, neutral.type_b_count) == (4, 0)
    assert neutral.dynamic_branch_count == 4
    half = count_tdhf_goldstones_from_rank(5, 4)
    assert (half.type_a_count, half.type_b_count) == (1, 2)
    assert half.dynamic_branch_count == 3
    with pytest.raises(ValueError, match="must be even"):
        count_tdhf_goldstones_from_rank(5, 3)
    with pytest.raises(TypeError, match="must be an integer"):
        count_tdhf_goldstones_from_rank(4.0, 0)  # type: ignore[arg-type]


def test_goldstone_symplectic_matrix_rank_and_antisymmetry() -> None:
    rho = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    count = analyze_tdhf_goldstone_symplectic_matrix(
        rho,
        rank_atol=1.0e-12,
        rank_rtol=1.0e-12,
        antisymmetry_tolerance=1.0e-12,
    )
    assert count.symplectic_rank == 2
    assert (count.type_a_count, count.type_b_count) == (1, 1)
    with pytest.raises(ValueError, match="not resolved as real antisymmetric"):
        analyze_tdhf_goldstone_symplectic_matrix(
            np.asarray([[0.0, 1.0], [0.0, 0.0]]),
            rank_atol=1.0e-12,
            rank_rtol=1.0e-12,
            antisymmetry_tolerance=1.0e-12,
        )


def test_goldstone_rank_resolution_rejects_structural_uncertainty() -> None:
    contaminated = np.asarray([[5.0e-11, 1.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="not resolved as real antisymmetric"):
        analyze_tdhf_goldstone_symplectic_matrix(
            contaminated,
            rank_atol=1.0e-12,
            rank_rtol=0.0,
            antisymmetry_tolerance=1.0e-10,
        )
    count = analyze_tdhf_goldstone_symplectic_matrix(
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        rank_atol=1.0e-12,
        rank_rtol=1.0e-12,
        antisymmetry_tolerance=1.0e-12,
    )
    assert count.singular_values.shape == (2,)
    assert count.smallest_retained_margin > 0.0
    assert count.largest_rejected_margin == float("inf")


def test_goldstone_rank_rejects_high_dimensional_symmetric_false_rank() -> None:
    symmetric = np.kron(np.eye(2), np.ones((3, 3)))
    with pytest.raises(ValueError, match="not resolved as real antisymmetric"):
        analyze_tdhf_goldstone_symplectic_matrix(
            symmetric,
            rank_atol=2.1,
            rank_rtol=0.0,
            antisymmetry_tolerance=3.0,
        )


def test_goldstone_rank_rejects_mixed_symmetric_false_rank() -> None:
    mixed = np.asarray([[0.8, 0.8], [-0.8, 0.8]])
    with pytest.raises(ValueError, match="not separated"):
        analyze_tdhf_goldstone_symplectic_matrix(
            mixed,
            rank_atol=1.0,
            rank_rtol=0.0,
            antisymmetry_tolerance=1.0,
        )


def _certificate(
    *,
    rho: np.ndarray,
    source: str = "source",
    h00: float = 0.0,
    generators: np.ndarray | None = None,
    nonfinite_hessian: bool = False,
    nonfinite_liouvillian: bool = False,
):
    hessian = np.diag([h00, 0.0, 2.0, 3.0]).astype(np.complex128)
    liouvillian = np.diag([h00, 0.0, 2.0, -3.0]).astype(np.complex128)
    if nonfinite_hessian:
        hessian[0, 0] = np.inf
    if nonfinite_liouvillian:
        liouvillian[0, 0] = np.nan
    if generators is None:
        generators = np.asarray(
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    return certify_tdhf_ward_subspace(
        hessian=hessian,
        liouvillian=liouvillian,
        generators=generators,
        generator_labels=("Q1", "Q2"),
        generator_provenances=("exact generator 1", "exact generator 2"),
        symplectic_matrix=rho,
        symplectic_provenance=_rho_provenance(
            ("Q1", "Q2"), source=source
        ),
        source_fingerprint=source,
        expected_source_fingerprint="source",
        interaction_fingerprint="interaction",
        sector_fingerprint="sector",
        response_scope="toy_scalar_hessian",
        static_hessian_authority="scalar_hessian",
        source_stationarity_residual=0.0,
        source_stationarity_tolerance=1.0e-12,
        action_tolerance=1.0e-12,
        null_eigenvalue_tolerance=1.0e-12,
        overlap_tolerance=1.0 - 1.0e-12,
        generator_rank_atol=1.0e-12,
        generator_rank_rtol=1.0e-12,
        symplectic_rank_atol=1.0e-12,
        symplectic_rank_rtol=1.0e-12,
        antisymmetry_tolerance=1.0e-12,
    )


def test_multi_generator_ward_subspace_certifies_type_ii_pair() -> None:
    certificate = _certificate(rho=np.asarray([[0.0, 1.0], [-1.0, 0.0]]))
    assert certificate.passed
    assert certificate.failure_reasons == ()
    assert certificate.static_null_dimension == 2
    assert certificate.static_null_min_principal_overlap == pytest.approx(1.0)
    assert certificate.generator_gram_residual < 1.0e-14
    assert certificate.hessian_action_operator_norm == 0.0
    assert certificate.liouvillian_action_operator_norm == 0.0
    assert certificate.goldstone_count is not None
    assert certificate.goldstone_count.broken_generator_count == 2
    assert certificate.goldstone_count.symplectic_rank == 2
    assert certificate.goldstone_count.type_a_count == 0
    assert certificate.goldstone_count.type_b_count == 1
    assert certificate.goldstone_count.dynamic_branch_count == 1


def test_multi_generator_ward_subspace_preserves_type_i_count() -> None:
    certificate = _certificate(rho=np.zeros((2, 2)))
    assert certificate.passed
    assert certificate.goldstone_count is not None
    assert certificate.goldstone_count.symplectic_rank == 0
    assert certificate.goldstone_count.type_a_count == 2
    assert certificate.goldstone_count.type_b_count == 0
    assert certificate.goldstone_count.dynamic_branch_count == 2


def test_multi_generator_ward_subspace_fails_without_repair() -> None:
    bad_rho = _certificate(rho=np.asarray([[0.0, 1.0], [0.0, 0.0]]))
    assert not bad_rho.passed
    assert "invalid_symplectic_matrix" in bad_rho.failure_reasons
    assert bad_rho.goldstone_count is None
    bad_source = _certificate(rho=np.zeros((2, 2)), source="wrong")
    assert not bad_source.passed
    assert "source_fingerprint_mismatch" in bad_source.failure_reasons
    bad_action = _certificate(rho=np.zeros((2, 2)), h00=1.0e-3)
    assert not bad_action.passed
    assert "ward_action_residual" in bad_action.failure_reasons
    assert "static_null_overlap" in bad_action.failure_reasons


def test_multi_generator_ward_subspace_rejects_nonfinite_matrices() -> None:
    with pytest.raises(ValueError, match="Hessian must be finite"):
        _certificate(rho=np.zeros((2, 2)), nonfinite_hessian=True)
    with pytest.raises(ValueError, match="Liouvillian must be finite"):
        _certificate(rho=np.zeros((2, 2)), nonfinite_liouvillian=True)
    with pytest.raises(ValueError, match="symplectic matrix must be finite"):
        _certificate(rho=np.asarray([[0.0, np.nan], [np.nan, 0.0]]))


def test_ward_action_operator_norm_is_generator_basis_invariant() -> None:
    first = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    second = first @ rotation
    direct = _certificate(rho=np.zeros((2, 2)), h00=1.0e-3, generators=first)
    rotated = _certificate(rho=np.zeros((2, 2)), h00=1.0e-3, generators=second)
    assert direct.hessian_action_operator_norm == pytest.approx(1.0e-3)
    assert rotated.hessian_action_operator_norm == pytest.approx(1.0e-3)


def test_multi_generator_ward_subspace_rejects_dependent_tangents() -> None:
    hessian = np.zeros((3, 3), dtype=np.complex128)
    generators = np.asarray(
        [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.complex128
    )
    with pytest.raises(ValueError, match="linearly dependent"):
        certify_tdhf_ward_subspace(
            hessian=hessian,
            liouvillian=hessian,
            generators=generators,
            generator_labels=("Q1", "Q2"),
            generator_provenances=("p1", "p2"),
            symplectic_matrix=np.zeros((2, 2)),
            symplectic_provenance=_rho_provenance(
                ("Q1", "Q2"), scope="scope"
            ),
            source_fingerprint="source",
            expected_source_fingerprint="source",
            interaction_fingerprint="interaction",
            sector_fingerprint="sector",
            response_scope="scope",
            static_hessian_authority="scalar_hessian",
            source_stationarity_residual=0.0,
            source_stationarity_tolerance=1.0e-12,
            action_tolerance=1.0e-12,
            null_eigenvalue_tolerance=1.0e-12,
            overlap_tolerance=1.0,
            generator_rank_atol=1.0e-12,
            generator_rank_rtol=1.0e-12,
            symplectic_rank_atol=1.0e-12,
            symplectic_rank_rtol=1.0e-12,
            antisymmetry_tolerance=1.0e-12,
        )


def _self_conjugate_sector() -> TDHFSelfConjugateQSector:
    pairs = (
        ParticleHolePair(0, 2, particle_momentum=(0, 0), hole_momentum=(0, 0)),
        ParticleHolePair(1, 3, particle_momentum=(0, 0), hole_momentum=(0, 0)),
    )
    return TDHFSelfConjugateQSector(
        q=TDHFSelfConjugateQ(
            plus_raw=(0, 0),
            minus_raw=(0, 0),
            canonical=(0, 0),
            orbit_multiplicity=1,
            provenance="literal_q0",
        ),
        canonical_pairs=pairs,
        A=np.diag([0.0, 2.0]).astype(np.complex128),
        B=np.zeros((2, 2), dtype=np.complex128),
        source_fingerprint="source",
        interaction_fingerprint="interaction",
        response_scope="toy_multi_ward_scalar_hessian",
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance="literal_q0_basis",
    )


def test_typed_analysis_rebinds_multi_generator_certificate() -> None:
    sector = _self_conjugate_sector()
    matrices = build_tdhf_self_conjugate_matrices(sector)
    generators = np.zeros((4, 2), dtype=np.complex128)
    generators[0, 0] = 1.0
    generators[2, 1] = 1.0
    certificate = certify_tdhf_ward_subspace(
        hessian=matrices.H,
        liouvillian=matrices.L,
        generators=generators,
        generator_labels=("Q1", "Q2"),
        generator_provenances=("exact tangent 1", "exact tangent 2"),
        symplectic_matrix=np.zeros((2, 2)),
        symplectic_provenance=_rho_provenance(
            ("Q1", "Q2"),
            source=sector.source_fingerprint,
            interaction=sector.interaction_fingerprint,
            sector=fingerprint_tdhf_sector(sector),
            scope=sector.response_scope,
        ),
        source_fingerprint=sector.source_fingerprint,
        expected_source_fingerprint=sector.source_fingerprint,
        interaction_fingerprint=sector.interaction_fingerprint,
        sector_fingerprint=fingerprint_tdhf_sector(sector),
        response_scope=sector.response_scope,
        static_hessian_authority=sector.static_hessian_authority,
        source_stationarity_residual=0.0,
        source_stationarity_tolerance=1.0e-12,
        action_tolerance=1.0e-12,
        null_eigenvalue_tolerance=1.0e-12,
        overlap_tolerance=1.0 - 1.0e-12,
        generator_rank_atol=1.0e-12,
        generator_rank_rtol=1.0e-12,
        symplectic_rank_atol=1.0e-12,
        symplectic_rank_rtol=1.0e-12,
        antisymmetry_tolerance=1.0e-12,
    )
    count = validate_tdhf_ward_subspace_certificate(
        certificate,
        sector=sector,
        hessian=matrices.H,
        liouvillian=matrices.L,
        static_zero_count=2,
        expected_static_null_tolerance=1.0e-12,
    )
    assert count.type_a_count == 2
    analysis = analyze_tdhf_sector(
        sector,
        TDHFConfig(
            q_sector=(0, 0),
            channel="all",
            hessian_tolerance=1.0e-12,
        ),
        ward_subspace_certificate=certificate,
    )
    assert analysis.zero_mode.origin == "ward_static_null"
    assert analysis.zero_mode.ward_passed
    assert analysis.ward_subspace is certificate

    assert certificate.goldstone_count is not None
    changed_generator = certificate.generator_matrix.copy()
    changed_generator[0, 0] += 0.1
    changed_basis = certificate.orthonormal_generator_basis.copy()
    changed_basis[0, 0] = 0.5
    changed_rho = certificate.symplectic_matrix.copy()
    changed_rho[0, 1] = 1.0
    changed_rho[1, 0] = -1.0
    tampered_certificates = (
        replace(certificate, interaction_fingerprint="wrong"),
        replace(certificate, hessian_fingerprint="0" * 64),
        replace(certificate, liouvillian_fingerprint="0" * 64),
        replace(certificate, generator_matrix=changed_generator),
        replace(certificate, generator_provenances=("tampered", "tampered")),
        replace(certificate, orthonormal_generator_basis=changed_basis),
        replace(certificate, symplectic_matrix=changed_rho),
        replace(certificate, symplectic_matrix=np.zeros((3, 3))),
        replace(
            certificate,
            goldstone_count=replace(
                certificate.goldstone_count,
                dynamic_branch_count=99,
            ),
        ),
        replace(
            certificate,
            symplectic_provenance=replace(
                certificate.symplectic_provenance,
                notes="tampered provenance",
            ),
        ),
        replace(
            certificate,
            hessian_action_operator_norm=1.0,
        ),
    )
    for tampered in tampered_certificates:
        with pytest.raises(ValueError, match="not bound to analyzed sector"):
            analyze_tdhf_sector(
                sector,
                TDHFConfig(
                    q_sector=(0, 0),
                    channel="all",
                    hessian_tolerance=1.0e-12,
                ),
                ward_subspace_certificate=tampered,
            )
