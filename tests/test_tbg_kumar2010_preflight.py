from __future__ import annotations

from dataclasses import replace

import pytest

from mean_field.systems.tbg.zero_field.kumar2010 import (
    KUMAR_2010_PDF_SHA256,
    KUMAR_2010_SOURCE_ARCHIVE_SHA256,
    Kumar2010ProviderBinding,
    Kumar2010Spec,
    KumarC3MeshReceipt,
    KumarCutoffAuthority,
    KumarFunctionalAuthority,
    KumarMeshAuthority,
    KumarParentAuthority,
    KumarPinnedValue,
    KumarRemoteClosureAuthority,
    KumarRemotePrescription,
)


def _resolved_spec() -> Kumar2010Spec:
    source_hash = KUMAR_2010_SOURCE_ARCHIVE_SHA256
    prescription = KumarRemotePrescription()
    c3_receipt = KumarC3MeshReceipt(
        shape=(18, 18),
        reciprocal_basis_fingerprint="5" * 64,
        mesh_generator_fingerprint="6" * 64,
        c3_map_fingerprint="7" * 64,
        orbit_map_fingerprint="8" * 64,
        reciprocal_carry_fingerprint="9" * 64,
        quadrature_weight_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
        carry_residual=0.0,
        weight_residual=0.0,
        source="explicit C3 map/orbit/carry/weight receipt",
    )
    return Kumar2010Spec(
        w1_mev=KumarPinnedValue(
            110.0,
            "reproduction_choice",
            "controlled choice; not stated by target paper",
        ),
        parent_one_body=KumarParentAuthority(
            model_convention="linear_dirac_bm",
            vf_mev_b0=2482.0,
            provider_fingerprint="1" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="controlled parent convention",
        ),
        plane_wave_cutoff=KumarCutoffAuthority(
            cutoff_kind="plane_wave",
            value=5,
            generator_fingerprint="2" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="pre-registered convergence choice",
        ),
        interaction_transfer_cutoff=KumarCutoffAuthority(
            cutoff_kind="interaction_transfer",
            value=5,
            generator_fingerprint="3" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="pre-registered convergence choice",
        ),
        remote_window=KumarCutoffAuthority(
            cutoff_kind="remote_window",
            value=6,
            generator_fingerprint="4" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="pre-registered remote convergence choice",
        ),
        k_mesh=KumarMeshAuthority(
            shape=(18, 18),
            reciprocal_basis_fingerprint="5" * 64,
            generator_fingerprint="6" * 64,
            source_artifact_sha256=source_hash,
            c3_receipt=c3_receipt,
            authority_kind="reproduction_choice",
            source="pre-registered and executable C3 orbit receipt",
        ),
        q0_background=KumarPinnedValue(
            "neutralizing_background_delete_q0",
            "reproduction_choice",
            "controlled q0 choice; target paper is silent",
        ),
        q_path_coordinates=KumarPinnedValue(
            "explicit_Kprime_Gamma_K_fractional_receipt",
            "reproduction_choice",
            "controlled path receipt; target gives labels only",
        ),
        scf_branch_policy=KumarPinnedValue(
            "fresh_multiseed_full_gradient_qualified",
            "reproduction_choice",
            "pre-registered SCF branch policy",
        ),
        remote_closure=KumarRemoteClosureAuthority(
            prescription_fingerprint=prescription.fingerprint,
            provider_fingerprint="c" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="executable implementation of the paper prescription",
        ),
        functional_authority=KumarFunctionalAuthority(
            scalar_energy_fingerprint="d" * 64,
            scf_hamiltonian_derivative_fingerprint="e" * 64,
            finite_q_hessian_derivative_fingerprint="f" * 64,
            shared_source_fingerprint="0" * 64,
            provider_fingerprint="1" * 64,
            source_artifact_sha256=source_hash,
            authority_kind="reproduction_choice",
            source="same-source scalar HF and TDHFA derivative receipt",
        ),
    )


class _BoundProvider:
    def __init__(self, spec: Kumar2010Spec) -> None:
        assert spec.parent_one_body is not None
        assert spec.plane_wave_cutoff is not None
        assert spec.interaction_transfer_cutoff is not None
        assert spec.remote_window is not None
        assert spec.k_mesh is not None
        assert spec.remote_closure is not None
        assert spec.functional_authority is not None
        assert spec.scf_branch_policy is not None
        self.fingerprint = "2" * 64
        self.spec_fingerprint = spec.fingerprint
        self.source_commit = "e" * 40
        self.scf_branch_policy_fingerprint = spec.scf_branch_policy.fingerprint
        self.parent_provider_fingerprint = spec.parent_one_body.provider_fingerprint
        self.parent_source_artifact_sha256 = spec.parent_one_body.source_artifact_sha256
        self.plane_wave_generator_fingerprint = spec.plane_wave_cutoff.generator_fingerprint
        self.plane_wave_source_artifact_sha256 = spec.plane_wave_cutoff.source_artifact_sha256
        self.interaction_transfer_generator_fingerprint = (
            spec.interaction_transfer_cutoff.generator_fingerprint
        )
        self.interaction_transfer_source_artifact_sha256 = (
            spec.interaction_transfer_cutoff.source_artifact_sha256
        )
        self.remote_window_generator_fingerprint = spec.remote_window.generator_fingerprint
        self.remote_window_source_artifact_sha256 = spec.remote_window.source_artifact_sha256
        self.mesh_generator_fingerprint = spec.k_mesh.generator_fingerprint
        self.mesh_source_artifact_sha256 = spec.k_mesh.source_artifact_sha256
        assert spec.k_mesh.c3_receipt is not None
        self.c3_receipt_fingerprint = spec.k_mesh.c3_receipt.fingerprint
        self.remote_provider_fingerprint = spec.remote_closure.provider_fingerprint
        self.remote_source_artifact_sha256 = spec.remote_closure.source_artifact_sha256
        self.scalar_energy_fingerprint = (
            spec.functional_authority.scalar_energy_fingerprint
        )
        self.scf_hamiltonian_derivative_fingerprint = (
            spec.functional_authority.scf_hamiltonian_derivative_fingerprint
        )
        self.finite_q_hessian_derivative_fingerprint = (
            spec.functional_authority.finite_q_hessian_derivative_fingerprint
        )
        self.shared_functional_source_fingerprint = (
            spec.functional_authority.shared_source_fingerprint
        )
        self.functional_provider_fingerprint = spec.functional_authority.provider_fingerprint
        self.functional_source_artifact_sha256 = (
            spec.functional_authority.source_artifact_sha256
        )


def test_kumar_paper_target_and_seven_mode_inventory() -> None:
    target = Kumar2010Spec.paper_target()
    assert (target.filling, target.theta_deg, target.kappa) == (-3, 1.1, 0.8)
    assert target.epsilon_inverse == 0.06
    assert target.coulomb_kernel == "bare_2pi_e2_over_epsilon_q"
    assert target.active_dimension_per_k == 8
    assert target.occupied_active_rank_per_k == 1
    modes = target.expected_modes
    assert (
        modes.intraflavor_excitons,
        modes.spin_waves,
        modes.valley_waves,
        modes.spin_valley_waves,
    ) == (1, 2, 2, 2)
    assert modes.total_modes == 7
    assert modes.quadratic_spin_goldstones == 1
    assert modes.derived_spin_commutator_rank == 2
    assert modes.low_mode_energy_scale.value_mev == 2.0
    assert modes.low_mode_energy_scale.qualifier == "below_approximately"
    assert modes.mode_bandwidth_scale.qualifier == "lesssim"
    assert modes.valley_gap_scale.qualifier == "approximately"
    assert modes.charge_gap_scale.qualifier == "approximately"
    assert len(KUMAR_2010_PDF_SHA256) == 64
    assert len(KUMAR_2010_SOURCE_ARCHIVE_SHA256) == 64


def test_kumar_paper_mode_inventory_and_approximate_scales_are_locked() -> None:
    modes = Kumar2010Spec.paper_target().expected_modes
    with pytest.raises(ValueError, match="mode inventory was changed"):
        replace(modes, intraflavor_excitons=0, spin_waves=3)
    with pytest.raises(ValueError, match="approximate paper scales were changed"):
        replace(
            modes,
            valley_gap_scale=replace(modes.valley_gap_scale, value_mev=0.3),
        )
    with pytest.raises(ValueError, match="mode inventory was changed"):
        replace(modes, derived_spin_commutator_rank=0)


def test_kumar_target_fails_closed_on_unpublished_authorities() -> None:
    target = Kumar2010Spec.paper_target()
    assert target.unresolved_authorities == (
        "w1_mev",
        "parent_one_body",
        "plane_wave_cutoff",
        "interaction_transfer_cutoff",
        "remote_window",
        "k_mesh",
        "q0_background",
        "q_path_coordinates",
        "scf_branch_policy",
        "remote_closure",
        "functional_authority",
    )
    assert not target.metadata_resolved
    assert not target.paper_direct_claim_allowed
    with pytest.raises(RuntimeError, match="executable authority is incomplete"):
        target.require_metadata_resolved()


def test_kumar_rejects_gate_kernel_and_wrong_active_inventory() -> None:
    with pytest.raises(ValueError, match="does not specify dual-gate"):
        Kumar2010Spec(coulomb_kernel="dual_gate_tanh_qd_over_q")
    with pytest.raises(ValueError, match="one occupied state"):
        Kumar2010Spec(active_bands_per_spin_valley=3)
    with pytest.raises(ValueError, match="one occupied state"):
        Kumar2010Spec(occupied_active_rank_per_k=2)


def test_kumar_forbids_fabricated_paper_direct_numerics() -> None:
    target = Kumar2010Spec.paper_target()
    with pytest.raises(ValueError, match="absolute w1 is not stated"):
        replace(
            target,
            w1_mev=KumarPinnedValue(110.0, "paper_explicit", "unsupported"),
        )
    for label in ("q0_background", "q_path_coordinates", "scf_branch_policy"):
        with pytest.raises(ValueError, match="not numerically stated"):
            replace(
                target,
                **{
                    label: KumarPinnedValue(
                        "unsupported",
                        "paper_explicit",
                        "paper does not state this numerical policy",
                    )
                },
            )


def test_kumar_mesh_is_candidate_until_typed_c3_receipt() -> None:
    with pytest.raises(ValueError, match="square-torus candidate"):
        KumarMeshAuthority(
            shape=(18, 12),
            reciprocal_basis_fingerprint="1" * 64,
            generator_fingerprint="2" * 64,
            source_artifact_sha256="3" * 64,
            c3_receipt=None,
            authority_kind="reproduction_choice",
            source="rectangular control",
        )
    candidate = KumarMeshAuthority(
        shape=(18, 18),
        reciprocal_basis_fingerprint="1" * 64,
        generator_fingerprint="2" * 64,
        source_artifact_sha256="3" * 64,
        c3_receipt=None,
        authority_kind="reproduction_choice",
        source="shape-only candidate",
    )
    with pytest.raises(ValueError, match="square candidate"):
        replace(Kumar2010Spec.paper_target(), k_mesh=candidate)

    receipt = KumarC3MeshReceipt(
        shape=(18, 18),
        reciprocal_basis_fingerprint="1" * 64,
        mesh_generator_fingerprint="2" * 64,
        c3_map_fingerprint="4" * 64,
        orbit_map_fingerprint="5" * 64,
        reciprocal_carry_fingerprint="6" * 64,
        quadrature_weight_fingerprint="7" * 64,
        source_fingerprint="8" * 64,
        carry_residual=0.0,
        weight_residual=0.0,
        source="typed C3 closure receipt",
    )
    certified = replace(candidate, c3_receipt=receipt)
    assert certified.c3_receipt is receipt
    for field in (
        "reciprocal_basis_fingerprint",
        "mesh_generator_fingerprint",
        "reciprocal_carry_fingerprint",
        "quadrature_weight_fingerprint",
    ):
        changed = replace(receipt, **{field: "9" * 64})
        if field in ("reciprocal_basis_fingerprint", "mesh_generator_fingerprint"):
            with pytest.raises(ValueError, match="mismatch"):
                replace(candidate, c3_receipt=changed)
        else:
            assert changed.fingerprint != receipt.fingerprint


def test_kumar_paper_prescription_is_separate_from_executable_remote_provider() -> None:
    prescription = KumarRemotePrescription()
    assert prescription.reference_density == "isolated_neutral_graphene_sheets"
    assert prescription.full_density_policy == "rho_full_minus_rho_iso"
    assert prescription.coulomb_kernel == "bare_2pi_e2_over_epsilon_q"
    with pytest.raises(ValueError, match="invalid Kumar authority kind"):
        KumarRemoteClosureAuthority(
            prescription_fingerprint=prescription.fingerprint,
            provider_fingerprint="4" * 64,
            source_artifact_sha256=KUMAR_2010_SOURCE_ARCHIVE_SHA256,
            authority_kind="paper_explicit",
            source="paper gives a formula but no executable provider",
        )
    closure = KumarRemoteClosureAuthority(
        prescription_fingerprint=prescription.fingerprint,
        provider_fingerprint="4" * 64,
        source_artifact_sha256=KUMAR_2010_SOURCE_ARCHIVE_SHA256,
        authority_kind="reproduction_choice",
        source="executable implementation receipt",
    )
    with pytest.raises(ValueError, match="implementation/prescription mismatch"):
        replace(closure, prescription_fingerprint="5" * 64)


def test_kumar_cutoffs_cannot_be_claimed_from_paper() -> None:
    with pytest.raises(ValueError, match="invalid Kumar authority kind"):
        KumarCutoffAuthority(
            cutoff_kind="plane_wave",
            value=5,
            generator_fingerprint="5" * 64,
            source_artifact_sha256=KUMAR_2010_SOURCE_ARCHIVE_SHA256,
            authority_kind="paper_explicit",
            source="paper gives no cutoff",
        )


def test_kumar_resolved_metadata_binds_receipts_without_claiming_execution() -> None:
    spec = _resolved_spec()
    spec.require_metadata_resolved()
    assert spec.metadata_resolved
    assert not spec.paper_direct_claim_allowed
    binding = Kumar2010ProviderBinding(spec, _BoundProvider(spec))
    assert binding.provider_bound
    assert not hasattr(binding, "executable_ready")
    receipt_attributes = (
        "spec_fingerprint",
        "scf_branch_policy_fingerprint",
        "parent_provider_fingerprint",
        "parent_source_artifact_sha256",
        "plane_wave_generator_fingerprint",
        "plane_wave_source_artifact_sha256",
        "interaction_transfer_generator_fingerprint",
        "interaction_transfer_source_artifact_sha256",
        "remote_window_generator_fingerprint",
        "remote_window_source_artifact_sha256",
        "mesh_generator_fingerprint",
        "mesh_source_artifact_sha256",
        "c3_receipt_fingerprint",
        "remote_provider_fingerprint",
        "remote_source_artifact_sha256",
        "scalar_energy_fingerprint",
        "scf_hamiltonian_derivative_fingerprint",
        "finite_q_hessian_derivative_fingerprint",
        "shared_functional_source_fingerprint",
        "functional_provider_fingerprint",
        "functional_source_artifact_sha256",
    )
    for attribute in receipt_attributes:
        provider = _BoundProvider(spec)
        current = getattr(provider, attribute)
        replacement = ("0" if current[0] != "0" else "1") + current[1:]
        setattr(provider, attribute, replacement)
        with pytest.raises(ValueError, match="Kumar provider/"):
            Kumar2010ProviderBinding(spec, provider)


def test_kumar_source_hashes_enter_fingerprint() -> None:
    target = Kumar2010Spec.paper_target()
    assert len(target.fingerprint) == 64
    assert target.fingerprint == Kumar2010Spec.paper_target().fingerprint
