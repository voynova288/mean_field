from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFSelfConjugateQ,
)
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.khalaf_fig3 import (
    KHALAF_FIG3_PDF_SHA256,
    KhalafCutoffAuthority,
    KhalafFig3ExecutableBinding,
    KhalafFig3Spec,
    KhalafParentOneBodyAuthority,
    KhalafPinnedValue,
    KhalafRectangularTorus,
    KhalafRemoteClosureAuthority,
)
from mean_field.systems.tbg.zero_field.model import (
    build_tbg_zero_field_half_open_torus_mesh,
)


def _resolved_spec(filling: int) -> KhalafFig3Spec:
    target = KhalafFig3Spec.paper_target(filling)
    return replace(
        target,
        w1_mev=KhalafPinnedValue(
            110.0,
            "cited_reference_explicit",
            "Bultinck et al. arXiv:1911.02045 default w1; controlled carry-over",
        ),
        parent_one_body=KhalafParentOneBodyAuthority(
            model_convention="full_monolayer_graphene_t0_parent",
            t0_ev=2.8,
            provider_fingerprint="b" * 64,
            source_artifact_sha256="c" * 64,
            authority_kind="cited_reference_explicit",
            source="Bultinck et al. arXiv:1911.02045 Supplement Hamiltonian",
        ),
        plane_wave_cutoff=KhalafCutoffAuthority(
            cutoff_kind="plane_wave",
            shell=5,
            generator_fingerprint="d" * 64,
            source_artifact_sha256="4" * 64,
            convergence_family="G_shell_4_5_6",
            authority_kind="reproduction_choice",
            source="pre-registered convergence family, not a Fig.3 paper value",
        ),
        interaction_transfer_cutoff=KhalafCutoffAuthority(
            cutoff_kind="interaction_transfer",
            shell=5,
            generator_fingerprint="e" * 64,
            source_artifact_sha256="5" * 64,
            convergence_family="Q_shell_4_5_6",
            authority_kind="reproduction_choice",
            source="pre-registered convergence family, not a Fig.3 paper value",
        ),
        remote_closure=KhalafRemoteClosureAuthority(
            reference_density="decoupled_graphene_layers_neutrality",
            subtraction_prefactor=0.5,
            projected_out_valence_policy=(
                "filled_remote_valence_plus_matching_P0_subtraction"
            ),
            provider_fingerprint="a" * 64,
            source_artifact_sha256="f" * 64,
            authority_kind="cited_reference_explicit",
            source="Bultinck et al. arXiv:1911.02045 Supplement Hartree-Fock",
        ),
    )


class _BoundProvider:
    def __init__(self, spec: KhalafFig3Spec) -> None:
        assert spec.remote_closure is not None
        self.fingerprint = "9" * 64
        self.spec_fingerprint = spec.fingerprint
        self.source_commit = "8" * 40
        assert spec.parent_one_body is not None
        assert spec.plane_wave_cutoff is not None
        assert spec.interaction_transfer_cutoff is not None
        self.parent_provider_fingerprint = spec.parent_one_body.provider_fingerprint
        self.parent_source_artifact_sha256 = spec.parent_one_body.source_artifact_sha256
        self.plane_wave_generator_fingerprint = (
            spec.plane_wave_cutoff.generator_fingerprint
        )
        self.plane_wave_source_artifact_sha256 = (
            spec.plane_wave_cutoff.source_artifact_sha256
        )
        self.interaction_transfer_generator_fingerprint = (
            spec.interaction_transfer_cutoff.generator_fingerprint
        )
        self.interaction_transfer_source_artifact_sha256 = (
            spec.interaction_transfer_cutoff.source_artifact_sha256
        )
        self.remote_closure_fingerprint = spec.remote_closure.provider_fingerprint
        self.remote_source_artifact_sha256 = spec.remote_closure.source_artifact_sha256

    def build_parent_one_body(self) -> object:
        return object()

    def build_interaction_kernel(self) -> object:
        return object()


def test_khalaf_fig3_paper_meshes_and_six_band_ranks() -> None:
    neutral = KhalafFig3Spec.paper_target(0)
    half = KhalafFig3Spec.paper_target(-2)
    assert neutral.mesh_shape == (18, 18)
    assert half.mesh_shape == (18, 12)
    assert neutral.retained_bands_per_spin_valley == 6
    assert half.retained_bands_per_spin_valley == 6
    assert neutral.retained_dimension_per_k == half.retained_dimension_per_k == 24
    assert neutral.occupied_remote_valence_per_k == 8
    assert neutral.occupied_flat_per_k == 4
    assert half.occupied_flat_per_k == 2
    assert neutral.occupied_rank_per_k == 12
    assert half.occupied_rank_per_k == 10
    assert neutral.expected_hf_gap_mev == 25.0
    assert half.expected_hf_gap_mev == 14.0
    assert len(KHALAF_FIG3_PDF_SHA256) == 64


def test_khalaf_fig3_goldstone_and_soft_mode_counts_close() -> None:
    neutral = KhalafFig3Spec.paper_target(0).expected_modes
    assert (
        neutral.total_soft_modes,
        neutral.static_ward_directions,
        neutral.symplectic_rank,
        neutral.linear_goldstones,
        neutral.quadratic_goldstones,
        neutral.gapped_degeneracies,
    ) == (16, 4, 0, 4, 0, (4, 4, 4))
    half = KhalafFig3Spec.paper_target(-2).expected_modes
    assert (
        half.total_soft_modes,
        half.static_ward_directions,
        half.symplectic_rank,
        half.linear_goldstones,
        half.quadratic_goldstones,
        half.gapped_degeneracies,
    ) == (12, 5, 4, 1, 2, (2, 1, 4, 2))


def test_khalaf_fig3_rejects_non_strict_integer_metadata() -> None:
    with pytest.raises(TypeError, match="mesh dimensions"):
        KhalafFig3Spec(filling=0, mesh_shape=(18.0, 18))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retained-band counts"):
        KhalafFig3Spec(
            filling=0,
            mesh_shape=(18, 18),
            retained_flat_per_spin_valley=2.0,  # type: ignore[arg-type]
        )


def test_khalaf_fig3_rejects_square_nu_minus_two_and_strain() -> None:
    with pytest.raises(ValueError, match="published Khalaf Fig.3 mesh"):
        KhalafFig3Spec(filling=-2, mesh_shape=(18, 18))
    with pytest.raises(ValueError, match="does not permit strain"):
        KhalafFig3Spec(filling=0, mesh_shape=(18, 18), strain_percent=0.3)


def test_khalaf_fig3_unresolved_authority_fails_closed() -> None:
    target = KhalafFig3Spec.paper_target(0)
    assert target.unresolved_authorities == (
        "w1_mev",
        "parent_one_body",
        "plane_wave_cutoff",
        "interaction_transfer_cutoff",
        "remote_closure",
    )
    assert not target.metadata_resolved
    assert not target.paper_direct_claim_allowed
    with pytest.raises(RuntimeError, match="authority closure is incomplete"):
        target.require_metadata_resolved()


def test_khalaf_fig3_resolved_metadata_needs_bound_executable_provider() -> None:
    spec = _resolved_spec(-2)
    spec.require_metadata_resolved()
    assert spec.metadata_resolved
    assert spec.resolved_w0_mev == 82.5
    assert not spec.paper_direct_claim_allowed
    assert len(spec.fingerprint) == 64
    assert spec.fingerprint == _resolved_spec(-2).fingerprint
    binding = KhalafFig3ExecutableBinding(spec, _BoundProvider(spec))
    assert binding.executable_ready
    bad_provider = _BoundProvider(spec)
    bad_provider.spec_fingerprint = "7" * 64
    with pytest.raises(ValueError, match="provider/spec fingerprint mismatch"):
        KhalafFig3ExecutableBinding(spec, bad_provider)

    receipt_attributes = (
        "parent_provider_fingerprint",
        "parent_source_artifact_sha256",
        "plane_wave_generator_fingerprint",
        "plane_wave_source_artifact_sha256",
        "interaction_transfer_generator_fingerprint",
        "interaction_transfer_source_artifact_sha256",
        "remote_closure_fingerprint",
        "remote_source_artifact_sha256",
    )
    for attribute in receipt_attributes:
        mismatched = _BoundProvider(spec)
        setattr(mismatched, attribute, "6" * 64)
        with pytest.raises(ValueError, match="Khalaf provider/"):
            KhalafFig3ExecutableBinding(spec, mismatched)


def test_khalaf_fig3_forbids_fabricated_caption_authority() -> None:
    target = KhalafFig3Spec.paper_target(0)
    with pytest.raises(ValueError, match="not directly specified"):
        replace(
            target,
            w1_mev=KhalafPinnedValue(
                110.0,
                "paper_explicit",
                "Fig.3 caption does not actually state this",
            ),
        )
    with pytest.raises(ValueError, match="does not directly specify"):
        KhalafCutoffAuthority(
            cutoff_kind="plane_wave",
            shell=5,
            generator_fingerprint="d" * 64,
            source_artifact_sha256="4" * 64,
            convergence_family="test",
            authority_kind="invalid",  # type: ignore[arg-type]
            source="unsupported",
        )
    with pytest.raises(ValueError, match="remote closure cannot be claimed"):
        KhalafRemoteClosureAuthority(
            reference_density="decoupled_graphene_layers_neutrality",
            subtraction_prefactor=0.5,
            projected_out_valence_policy=(
                "filled_remote_valence_plus_matching_P0_subtraction"
            ),
            provider_fingerprint="b" * 64,
            source_artifact_sha256="c" * 64,
            authority_kind="paper_explicit",
            source="unsupported caption claim",
        )


def test_khalaf_rectangular_torus_raw_signed_exact_m_aliases() -> None:
    torus = KhalafRectangularTorus.for_fig3(
        -2, reciprocal_basis_fingerprint="1" * 64
    )
    assert torus.shape == (18, 12)
    plus, minus = torus.signed_pair((9, 0))
    assert plus.raw == (9, 0)
    assert minus.raw == (-9, 0)
    assert plus.canonical == minus.canonical == (9, 0)
    assert plus.reciprocal_carry == (0, 0)
    assert minus.reciprocal_carry == (-1, 0)
    assert plus.fingerprint != minus.fingerprint
    assert isinstance(torus.signed_q_kind((9, 0)), TDHFSelfConjugateQ)


def test_khalaf_rectangular_label_binds_shape_filling_basis_and_q_type() -> None:
    neutral = KhalafRectangularTorus.for_fig3(
        0, reciprocal_basis_fingerprint="3" * 64
    )
    half = KhalafRectangularTorus.for_fig3(
        -2, reciprocal_basis_fingerprint="3" * 64
    )
    assert neutral.label((0, 6)).fingerprint != half.label((0, 6)).fingerprint
    assert isinstance(neutral.signed_q_kind((0, 6)), TDHFGenericSignedQ)
    assert isinstance(half.signed_q_kind((0, 6)), TDHFSelfConjugateQ)


def test_khalaf_rectangular_torus_binds_actual_rectangular_bm_mesh() -> None:
    # The bridge test binds only reciprocal geometry; it does not resolve the
    # still-unpublished Fig.3 tunnelling amplitudes.
    params = TBGParameters.from_degrees(1.08)
    bm_mesh = build_tbg_zero_field_half_open_torus_mesh(params, (18, 12))
    torus = KhalafRectangularTorus.from_bm_mesh(-2, bm_mesh)
    assert torus.shape == bm_mesh.mesh_shape == (18, 12)
    assert torus.reciprocal_basis_fingerprint == (
        bm_mesh.reciprocal_basis_fingerprint
    )
    for canonical in ((0, 0), (9, 0), (0, 6), (17, 11)):
        assert torus.flatten(canonical) == canonical[0] + 18 * canonical[1]
        assert bm_mesh.k_grid_frac[torus.flatten(canonical)].tolist() == [
            canonical[0] / 18.0,
            canonical[1] / 12.0,
        ]
    for raw in ((9, 0), (0, 6)):
        plus, minus = torus.signed_pair(raw)
        assert plus.canonical == minus.canonical
        assert plus.raw != minus.raw
        assert plus.reciprocal_carry != minus.reciprocal_carry
        assert isinstance(torus.signed_q_kind(raw), TDHFSelfConjugateQ)
    wrong_mesh = build_tbg_zero_field_half_open_torus_mesh(params, 18)
    with pytest.raises(ValueError, match="does not match"):
        KhalafRectangularTorus.from_bm_mesh(-2, wrong_mesh)


def test_khalaf_rectangular_torus_m_gamma_m_path_and_roundtrip() -> None:
    torus = KhalafRectangularTorus.for_fig3(
        0, reciprocal_basis_fingerprint="2" * 64
    )
    labels = torus.m_gamma_m_x_labels()
    assert len(labels) == 19
    assert labels[0].raw == (-9, 0)
    assert labels[9].raw == (0, 0)
    assert labels[-1].raw == (9, 0)
    assert labels[0].canonical == labels[-1].canonical == (9, 0)
    assert labels[0].reciprocal_carry != labels[-1].reciprocal_carry
    numpy_torus = KhalafRectangularTorus(
        (np.int64(18), np.int64(18)),
        np.int64(0),
        "2" * 64,
    )
    assert numpy_torus.shape == (18, 18)
    assert isinstance(numpy_torus.fingerprint, str)
    assert numpy_torus.label((np.int64(1), np.int64(0))).raw == (1, 0)
    assert numpy_torus.unflatten(np.int64(1)) == (1, 0)
    assert torus.flatten((1, 0)) == 1
    assert torus.flatten((0, 1)) == torus.shape[0]
    assert isinstance(torus.signed_q_kind((1, 0)), TDHFGenericSignedQ)
    with pytest.raises(TypeError, match="two integers"):
        torus.flatten((1.5, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="two integers"):
        torus.flatten((1.0, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="two integers"):
        torus.flatten((True, 0))
    with pytest.raises(ValueError, match="out of range"):
        torus.unflatten(1.0)  # type: ignore[arg-type]
    for x in range(torus.shape[0]):
        for y in range(torus.shape[1]):
            assert torus.unflatten(torus.flatten((x, y))) == (x, y)
