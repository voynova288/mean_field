from __future__ import annotations

from dataclasses import replace

import pytest

from mean_field.core.hf.tdhf_signed import TDHFSelfConjugateQ
from mean_field.systems.tbg.zero_field.wang2025 import (
    WANG_2025_MAIN_TEX_SHA256,
    WANG_2025_PDF_SHA256,
    WANG_2025_SOURCE_ARCHIVE_SHA256,
    Wang2025ProviderBinding,
    Wang2025Spec,
    WangAuthorFamilyTBGHFReceipt,
    WangCutoffReceipt,
    WangImplementationReceipt,
    WangMeshSignedQReceipt,
    WangModeScales,
    WangPrimarySourceReceipt,
)


def _implementation_receipt(role: str, value: int | float | str) -> WangImplementationReceipt:
    return WangImplementationReceipt(
        role=role,  # type: ignore[arg-type]
        value=value,
        provider_fingerprint="a" * 64,
        source_artifact_sha256=WANG_2025_MAIN_TEX_SHA256,
        authority_kind="reproduction_choice",
        source=f"controlled {role} receipt; not paper-direct authority",
    )


def _mesh_receipt() -> WangMeshSignedQReceipt:
    return WangMeshSignedQReceipt(
        shape=(10, 10),
        reciprocal_basis_fingerprint="b" * 64,
        mesh_generator_fingerprint="c" * 64,
        index_order="first_coordinate_fastest_fortran_v1",
        raw_tdhf_endpoints=((5, 0), (-5, 0)),
        canonical_tdhf_endpoints=((5, 0), (5, 0)),
        raw_reciprocal_carries=((0, 0), (-1, 0)),
        endpoint_classification="self_conjugate_exact_M_raw_alias_pair",
        source_artifact_sha256=WANG_2025_MAIN_TEX_SHA256,
        source="typed 10x10 mesh registration and raw signed-q carry receipt",
    )


def _resolved_spec() -> Wang2025Spec:
    target = Wang2025Spec.paper_target()
    return replace(
        target,
        strain_beta=_implementation_receipt("strain_beta", 3.14),
        parent_plane_wave_cutoff=WangCutoffReceipt(
            cutoff_kind="parent_plane_wave",
            value="hex_shell_convergence_family",
            generator_fingerprint="d" * 64,
            source_artifact_sha256=WANG_2025_MAIN_TEX_SHA256,
            authority_kind="reproduction_choice",
            source="controlled parent cutoff receipt",
        ),
        interaction_transfer_cutoff=WangCutoffReceipt(
            cutoff_kind="interaction_transfer",
            value="hex_transfer_convergence_family",
            generator_fingerprint="e" * 64,
            source_artifact_sha256=WANG_2025_MAIN_TEX_SHA256,
            authority_kind="reproduction_choice",
            source="controlled transfer cutoff receipt",
        ),
        q0_policy=_implementation_receipt("q0_policy", "explicit_finite_q0_policy"),
        exact_p_ref=_implementation_receipt(
            "exact_p_ref", "typed_half_identity_active_reference"
        ),
        mesh_registration_and_carries=_mesh_receipt(),
        iks_scan_and_scf_policy=_implementation_receipt(
            "iks_scan_and_scf_policy", "pre_registered_multiseed_q_scan_and_scf"
        ),
        exact_tdhf_q_list_and_tolerances=_implementation_receipt(
            "exact_tdhf_q_list_and_tolerances",
            "raw_signed_q_list_with_stability_and_goldstone_tolerances",
        ),
        scalar_energy_functional=_implementation_receipt(
            "scalar_energy_functional", "2" * 64
        ),
        scf_hamiltonian_derivative=_implementation_receipt(
            "scf_hamiltonian_derivative", "3" * 64
        ),
        finite_q_hessian_derivative=_implementation_receipt(
            "finite_q_hessian_derivative", "4" * 64
        ),
        shared_functional_source=_implementation_receipt(
            "shared_functional_source", "5" * 64
        ),
        tdhf_implementation_provider=_implementation_receipt(
            "tdhf_implementation_provider", "typed_A_B_L_provider_receipt"
        ),
    )


class _BoundProvider:
    def __init__(self, spec: Wang2025Spec) -> None:
        assert spec.strain_beta is not None
        assert spec.parent_plane_wave_cutoff is not None
        assert spec.interaction_transfer_cutoff is not None
        assert spec.q0_policy is not None
        assert spec.exact_p_ref is not None
        assert spec.mesh_registration_and_carries is not None
        assert spec.iks_scan_and_scf_policy is not None
        assert spec.exact_tdhf_q_list_and_tolerances is not None
        assert spec.scalar_energy_functional is not None
        assert spec.scf_hamiltonian_derivative is not None
        assert spec.finite_q_hessian_derivative is not None
        assert spec.shared_functional_source is not None
        assert spec.tdhf_implementation_provider is not None
        self.fingerprint = "f" * 64
        self.spec_fingerprint = spec.fingerprint
        self.source_commit = "1" * 40
        self.strain_beta_receipt_fingerprint = spec.strain_beta.fingerprint
        self.parent_plane_wave_cutoff_receipt_fingerprint = (
            spec.parent_plane_wave_cutoff.fingerprint
        )
        self.interaction_transfer_cutoff_receipt_fingerprint = (
            spec.interaction_transfer_cutoff.fingerprint
        )
        self.q0_policy_receipt_fingerprint = spec.q0_policy.fingerprint
        self.exact_p_ref_receipt_fingerprint = spec.exact_p_ref.fingerprint
        self.mesh_registration_and_carries_receipt_fingerprint = (
            spec.mesh_registration_and_carries.fingerprint
        )
        self.iks_scan_and_scf_policy_receipt_fingerprint = (
            spec.iks_scan_and_scf_policy.fingerprint
        )
        self.exact_tdhf_q_list_and_tolerances_receipt_fingerprint = (
            spec.exact_tdhf_q_list_and_tolerances.fingerprint
        )
        self.scalar_energy_functional_receipt_fingerprint = (
            spec.scalar_energy_functional.fingerprint
        )
        self.scf_hamiltonian_derivative_receipt_fingerprint = (
            spec.scf_hamiltonian_derivative.fingerprint
        )
        self.finite_q_hessian_derivative_receipt_fingerprint = (
            spec.finite_q_hessian_derivative.fingerprint
        )
        self.shared_functional_source_receipt_fingerprint = (
            spec.shared_functional_source.fingerprint
        )
        self.tdhf_implementation_provider_receipt_fingerprint = (
            spec.tdhf_implementation_provider.fingerprint
        )


def test_wang_primary_source_and_paper_direct_model_are_pinned() -> None:
    target = Wang2025Spec.paper_target()
    paper = target.paper
    assert paper.fillings == (2, 3)
    assert (paper.theta_deg, paper.w_aa_mev, paper.w_ab_mev) == (1.05, 80.0, 110.0)
    assert paper.dirac_velocity_m_per_s == 8.8e5
    assert (paper.strain_percent, paper.strain_angle_deg, paper.poisson_ratio) == (
        0.3,
        0.0,
        0.16,
    )
    assert paper.hopping_model == "local_three_matrix_bm_nonlocal_tunnelling_excluded"
    assert paper.coulomb_kernel == "e2_tanh_qd_over_2_epsilon0_epsilonr_q"
    assert (paper.gate_distance_nm, paper.epsilon_r) == (25.0, 10.0)
    assert paper.subtraction_name == "average"
    assert paper.active_bands_per_spin_valley == 2
    assert paper.active_dimension_per_k == 8
    assert paper.tdhf_mesh_shape == (10, 10)
    assert paper.field_tdhf_mesh_shape == (10, 10)
    assert paper.full_bz_tdhf_mesh_shape == (10, 10)
    assert paper.linear_quadratic_confirmation_mesh_shape == (20, 20)
    assert paper.q_iks_fractional_G1_G2 == (0.5, 0.0)
    assert paper.tdhf_q_role == "independent_transfer_q_not_ground_state_qIKS"
    assert paper.phonon_irreps == ("A1", "B1")
    assert paper.epc_vertices == ("tau_x_sigma_x", "tau_y_sigma_x")
    assert paper.epc_treatment.startswith("Schrieffer_Wolff")
    assert paper.intervalley_coulomb_treatment.endswith("tunable_V_inter")
    assert paper.intervalley_coulomb_estimates_mev_nm2 == (50.0, 120.0)
    source = target.primary_source
    assert source.source_archive_sha256 == WANG_2025_SOURCE_ARCHIVE_SHA256
    assert source.main_tex_sha256 == WANG_2025_MAIN_TEX_SHA256
    assert source.local_pdf_sha256 == WANG_2025_PDF_SHA256
    assert source.source_archive_size_bytes == 6_524_220


def test_wang_linear_and_quadratic_scenario_counts_are_locked() -> None:
    modes = Wang2025Spec.paper_target().paper.modes
    expected = {
        (2, "unperturbed"): (4, 0),
        (2, "epc"): (1, 0),
        (2, "intervalley_coulomb"): (3, 0),
        (2, "zeeman"): (2, 0),
        (2, "ising_soc"): (2, 0),
        (3, "unperturbed"): (1, 2),
        (3, "epc"): (3, 0),
        (3, "intervalley_coulomb"): (1, 1),
        (3, "zeeman"): (1, 0),
        (3, "ising_soc"): (1, 0),
    }
    for key, counts in expected.items():
        item = modes.for_scenario(*key)  # type: ignore[arg-type]
        assert (item.linear_goldstones, item.quadratic_goldstones) == counts
        assert item.total_goldstone_branches == sum(counts)
    assert (
        modes.total_soft_modes_nu2,
        modes.total_soft_modes_nu3,
        modes.additional_gapped_modes_nu2,
        modes.additional_gapped_modes_nu3,
    ) == (12, 7, 8, 4)
    assert modes.for_scenario(2, "zeeman").infinitesimal_perturbation_only
    assert modes.for_scenario(2, "ising_soc").infinitesimal_perturbation_only


def test_wang_approximate_scales_and_raster_only_values_keep_qualifiers() -> None:
    paper = Wang2025Spec.paper_target().paper
    scales = paper.mode_scales
    assert (
        scales.additional_gapped_modes.value_mev,
        scales.additional_gapped_modes.qualifier,
    ) == (15.0, "approximately")
    assert scales.figure_hf_charge_gap.value_mev == 20.0
    assert scales.figure_hf_charge_gap.qualifier == (
        "approximately_figure_scale_actual_likely_much_smaller"
    )
    assert scales.former_ngm_gaps.value_mev is None
    assert scales.former_ngm_gaps.qualifier == "small_unquantified"
    for item in (
        scales.additional_gapped_modes,
        scales.figure_hf_charge_gap,
        scales.former_ngm_gaps,
    ):
        assert item.acceptance_use == "context_only_not_an_exact_threshold"

    raster = paper.raster_only_targets
    assert (raster.epc_g_mev_nm2, raster.intervalley_coulomb_mev_nm2) == (
        50.0,
        100.0,
    )
    assert (raster.zeeman_mev, raster.ising_soc_mev) == (0.4, 0.4)
    assert raster.large_grid_epc_g_mev_nm2 == 100.0
    assert raster.main_path_q1_extent == (-0.5, 0.5)
    assert raster.main_path_q2 == 0.0
    assert raster.qualifier == "paper_source_raster_label_only_not_caption_text"


def test_wang_qiks_boost_is_distinct_from_raw_signed_tdhf_q() -> None:
    receipt = _mesh_receipt()
    boost = receipt.iks_boost
    plus = receipt.plus_m
    minus = receipt.minus_m
    assert boost.fractional_G1_G2 == (0.5, 0.0)
    assert boost.raw_grid_shift == (5, 0)
    assert boost.role == "ground_state_iks_valley_boost_not_tdhf_transfer_q"
    assert plus.raw == (5, 0)
    assert minus.raw == (-5, 0)
    assert plus.canonical == minus.canonical == (5, 0)
    assert plus.reciprocal_carry == (0, 0)
    assert minus.reciprocal_carry == (-1, 0)
    assert plus.raw != minus.raw
    assert plus.fingerprint != minus.fingerprint
    assert boost.fingerprint != plus.fingerprint
    assert "tdhf_external_transfer_q" in plus.role
    assert plus.q_classification == "self_conjugate_exact_M_raw_alias"
    assert minus.q_classification == "self_conjugate_exact_M_raw_alias"
    assert isinstance(receipt.signed_q_kind, TDHFSelfConjugateQ)


def test_wang_target_fails_closed_on_all_unresolved_authorities() -> None:
    target = Wang2025Spec.paper_target()
    assert target.unresolved_authorities == (
        "strain_beta",
        "parent_plane_wave_cutoff",
        "interaction_transfer_cutoff",
        "q0_policy",
        "exact_p_ref",
        "mesh_registration_and_carries",
        "iks_scan_and_scf_policy",
        "exact_tdhf_q_list_and_tolerances",
        "scalar_energy_functional",
        "scf_hamiltonian_derivative",
        "finite_q_hessian_derivative",
        "shared_functional_source",
        "tdhf_implementation_provider",
    )
    assert not target.metadata_resolved
    assert not target.paper_direct_claim_allowed
    with pytest.raises(RuntimeError, match="executable authority is incomplete"):
        target.require_metadata_resolved()


def test_later_author_family_receipt_is_separate_and_not_target_authority() -> None:
    receipt = WangAuthorFamilyTBGHFReceipt()
    assert receipt.commit == "0d2a3d742aa901fa45ce46690c1385887165f58c"
    assert receipt.tag == "v1.0.0"
    assert receipt.tag_commit == "b180f8f2b627d8a80b74b61a99d96b2cd56a76db"
    assert len(receipt.archive_sha256) == 64
    assert len(receipt.file_sha256) == 7
    assert receipt.beta == 3.14
    assert receipt.parent_envelope_ng == (4, 4)
    assert receipt.transfer_envelope_ng == (5, 5)
    assert receipt.include_q0
    assert "half_identity" in receipt.reference_policy
    assert receipt.grid_registration.startswith("Gamma_registered")
    assert (receipt.theta_deg, receipt.w_aa_mev, receipt.w_ab_mev) == (
        1.08,
        70.0,
        110.0,
    )
    assert not receipt.has_epc
    assert not receipt.has_intervalley_coulomb
    assert not receipt.has_tdhf
    assert not receipt.target_parameter_match
    assert not receipt.target_run_authority
    assert not receipt.tdhf_authority
    target = Wang2025Spec.paper_target()
    assert target.author_family is receipt or target.author_family == receipt
    assert "strain_beta" in target.unresolved_authorities
    assert "tdhf_implementation_provider" in target.unresolved_authorities


def test_wang_paper_and_author_evidence_reject_tampering() -> None:
    target = Wang2025Spec.paper_target()
    with pytest.raises(ValueError, match="paper-direct BM/strain"):
        replace(target.paper, theta_deg=1.08)
    with pytest.raises(ValueError, match="paper-direct BM/strain"):
        replace(target.paper, w_aa_mev=70.0)
    with pytest.raises(TypeError, match="active-band counts"):
        replace(target.paper, active_bands_per_spin_valley=2.0)
    with pytest.raises(TypeError, match="soft-mode totals"):
        replace(target.paper.modes, total_soft_modes_nu2=12.0)
    with pytest.raises(ValueError, match="primary-source receipt"):
        replace(target.primary_source, main_tex_sha256="0" * 64)
    with pytest.raises(ValueError, match="raster-only target labels"):
        replace(target.paper.raster_only_targets, epc_g_mev_nm2=70.0)

    changed_scale = replace(
        target.paper.mode_scales.additional_gapped_modes,
        value_mev=16.0,
    )
    with pytest.raises(ValueError, match="qualified mode scales"):
        WangModeScales(additional_gapped_modes=changed_scale)
    with pytest.raises(ValueError, match="author-family TBG-HF receipt"):
        replace(target.author_family, has_tdhf=True)
    with pytest.raises(ValueError, match="author-family TBG-HF receipt"):
        replace(target.author_family, theta_deg=1.05)
    with pytest.raises(TypeError, match="Ng/NG values"):
        replace(target.author_family, parent_envelope_ng=(4.0, 4))

    unperturbed_nu3 = target.paper.modes.for_scenario(3, "unperturbed")
    with pytest.raises(ValueError, match="linear/quadratic mode inventory"):
        replace(unperturbed_nu3, linear_goldstones=3, quadratic_goldstones=0)


def test_wang_mesh_receipt_rejects_endpoint_carry_and_index_tampering() -> None:
    receipt = _mesh_receipt()
    with pytest.raises(ValueError, match="raw carries"):
        replace(receipt, raw_reciprocal_carries=((0, 0), (0, 0)))
    with pytest.raises(ValueError, match=r"raw \+M and -M"):
        replace(receipt, raw_tdhf_endpoints=((5, 0), (5, 0)))
    with pytest.raises(TypeError, match="strict integers"):
        replace(receipt, raw_tdhf_endpoints=((5.0, 0), (-5, 0)))
    with pytest.raises(ValueError, match="bind index order"):
        replace(receipt, index_order="")
    with pytest.raises(ValueError, match="exact-M classification"):
        replace(receipt, endpoint_classification="generic_q_pair")
    with pytest.raises(ValueError, match="exact-M transfer classification"):
        replace(receipt.plus_m, q_classification="generic_q")
    with pytest.raises(ValueError, match="strict 10x10"):
        replace(receipt, shape=(20, 20))


def test_wang_resolved_metadata_binding_never_claims_execution_readiness() -> None:
    spec = _resolved_spec()
    spec.require_metadata_resolved()
    assert spec.metadata_resolved
    assert not spec.paper_direct_claim_allowed
    binding = Wang2025ProviderBinding(spec, _BoundProvider(spec))
    assert binding.provider_bound
    assert not hasattr(binding, "executable_ready")
    assert not hasattr(binding.provider, "build_tdhf")

    provider = _BoundProvider(spec)
    provider.spec_fingerprint = "0" * 64
    with pytest.raises(ValueError, match="provider/spec fingerprint mismatch"):
        Wang2025ProviderBinding(spec, provider)


def test_wang_provider_binding_rejects_every_tampered_receipt() -> None:
    spec = _resolved_spec()
    attributes = (
        "strain_beta_receipt_fingerprint",
        "parent_plane_wave_cutoff_receipt_fingerprint",
        "interaction_transfer_cutoff_receipt_fingerprint",
        "q0_policy_receipt_fingerprint",
        "exact_p_ref_receipt_fingerprint",
        "mesh_registration_and_carries_receipt_fingerprint",
        "iks_scan_and_scf_policy_receipt_fingerprint",
        "exact_tdhf_q_list_and_tolerances_receipt_fingerprint",
        "scalar_energy_functional_receipt_fingerprint",
        "scf_hamiltonian_derivative_receipt_fingerprint",
        "finite_q_hessian_derivative_receipt_fingerprint",
        "shared_functional_source_receipt_fingerprint",
        "tdhf_implementation_provider_receipt_fingerprint",
    )
    for attribute in attributes:
        provider = _BoundProvider(spec)
        current = getattr(provider, attribute)
        replacement = ("0" if current[0] != "0" else "1") + current[1:]
        setattr(provider, attribute, replacement)
        with pytest.raises(ValueError, match="Wang provider/"):
            Wang2025ProviderBinding(spec, provider)


def test_wang_shared_functional_receipts_require_hashes_and_one_source() -> None:
    target = Wang2025Spec.paper_target()
    with pytest.raises(ValueError, match="SHA256"):
        replace(
            target,
            scalar_energy_functional=_implementation_receipt(
                "scalar_energy_functional", "not-a-functional-hash"
            ),
        )
    resolved = _resolved_spec()
    assert resolved.scf_hamiltonian_derivative is not None
    changed = replace(
        resolved.scf_hamiltonian_derivative,
        source_artifact_sha256="6" * 64,
    )
    with pytest.raises(ValueError, match="must share one source artifact"):
        replace(resolved, scf_hamiltonian_derivative=changed)


def test_wang_receipt_roles_and_cutoff_kinds_fail_closed() -> None:
    target = Wang2025Spec.paper_target()
    wrong_role = _implementation_receipt("q0_policy", "wrong role")
    with pytest.raises(ValueError, match="strain_beta receipt has the wrong role"):
        replace(target, strain_beta=wrong_role)
    transfer = WangCutoffReceipt(
        cutoff_kind="interaction_transfer",
        value=5,
        generator_fingerprint="1" * 64,
        source_artifact_sha256=WANG_2025_MAIN_TEX_SHA256,
        authority_kind="reproduction_choice",
        source="controlled transfer receipt",
    )
    with pytest.raises(ValueError, match="parent plane-wave cutoff has the wrong kind"):
        replace(target, parent_plane_wave_cutoff=transfer)
    with pytest.raises(ValueError, match="source commit"):
        bad = _BoundProvider(_resolved_spec())
        bad.source_commit = "not-a-commit"
        Wang2025ProviderBinding(_resolved_spec(), bad)


def test_wang_fingerprints_are_deterministic_and_source_sensitive() -> None:
    target = Wang2025Spec.paper_target()
    assert len(target.fingerprint) == 64
    assert target.fingerprint == Wang2025Spec.paper_target().fingerprint
    assert len(target.paper.fingerprint) == 64
    assert len(target.primary_source.fingerprint) == 64
    assert len(target.author_family.fingerprint) == 64
    altered = replace(target, strain_beta=_implementation_receipt("strain_beta", 3.14))
    assert altered.fingerprint != target.fingerprint


def test_wang_primary_source_constructor_rejects_nonlocal_hash_substitution() -> None:
    with pytest.raises(ValueError, match="primary-source receipt"):
        WangPrimarySourceReceipt(local_pdf_sha256="f" * 64)
