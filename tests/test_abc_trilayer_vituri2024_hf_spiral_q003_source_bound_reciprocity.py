from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mean_field.systems import abc_trilayer
from mean_field.systems.abc_trilayer import (
    vituri2024_hf_spiral_q003_source_bound_reciprocity as certification,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_q003_source_bound_reciprocity import (
    ARTIFACT_ROLES,
    REVIEWED_CAPSULE_MEMBERS,
    Vituri2024Q003SourceBoundReciprocityCertificate,
    certify_vituri2024_q003_source_bound_reciprocity,
    vituri2024_q003_source_bound_reciprocity_implementation_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vituri2024_q003_reciprocity_source_bound_v1"


def _evidence() -> tuple[dict[str, bytes], dict[str, bytes]]:
    artifact_paths = {
        "candidate_contract": FIXTURE / "candidate_contract.json",
        "candidate_detached_approval": FIXTURE / "candidate_detached_approval.json",
        "candidate_input_manifest": FIXTURE / "candidate_input_manifest.json",
        "candidate_prelaunch_review": FIXTURE / "candidate_prelaunch_review.json",
        "candidate_reduced_qualification": FIXTURE
        / "candidate_reduced_qualification.json",
        "candidate_result": FIXTURE / "candidate_result.json",
        "candidate_ready": FIXTURE / "candidate_ready.json",
        "certifier_review": ROOT
        / "reports/data/vituri2024_fig2_q003_reciprocity_certifier_review_20260907.json",
        "historical_capsule_manifest": FIXTURE / "historical_capsule_manifest.json",
        "historical_source_commit": FIXTURE / "historical_source_commit.txt",
        "historical_source_manifest": FIXTURE / "historical_source_manifest.txt",
        "postrun_attestation": ROOT
        / "reports/data/vituri2024_fig2_q003_whole_inventory_reciprocity_candidate_493109_attestation.json",
        "recovery_complete": FIXTURE / "recovery_complete.json",
        "recovery_summary": FIXTURE / "recovery_summary.json",
        "theorem_review": ROOT
        / "reports/data/vituri2024_fig2_q003_reciprocity_theorem_review_20260907.json",
    }
    member_paths = {
        "bin/run_qualifier.py": FIXTURE / "historical_run_qualifier.py",
        "submit.sbatch": FIXTURE / "historical_submit.sbatch",
    }
    for path in REVIEWED_CAPSULE_MEMBERS:
        if path.startswith("source/"):
            member_paths[path] = ROOT / path.removeprefix("source/")
    return (
        {role: artifact_paths[role].read_bytes() for role in ARTIFACT_ROLES},
        {
            path: member_paths[path].read_bytes()
            for path in REVIEWED_CAPSULE_MEMBERS
        },
    )


def _certificate() -> Vituri2024Q003SourceBoundReciprocityCertificate:
    artifacts, members = _evidence()
    return certify_vituri2024_q003_source_bound_reciprocity(
        artifacts=artifacts,
        reviewed_capsule_members=members,
    )


def test_exact_pinned_evidence_establishes_only_algebraic_reciprocity() -> None:
    certificate = _certificate()
    assert certificate.artifact_only
    assert certificate.algebraic_reciprocity_only
    assert certificate.source_bound_q003_reciprocity_established
    assert certificate.whole_inventory_reciprocity_established
    assert certificate.both_groups_certified
    assert certificate.no_postselection
    assert not certificate.runtime_bitwise_reciprocity_established
    assert not certificate.full_inventory_exact_unitary_scalar_curvature_established
    assert not certificate.scalar_hessian_authority_established
    assert not certificate.linear_operator_authorized
    assert not certificate.hermitian_eigensolver_authorized
    assert not certificate.full_local_stability_established
    assert not certificate.production_ready
    assert not certificate.paper_reproduction_verified
    assert tuple(group.group_label for group in certificate.groups) == ("3307", "40fd")
    assert all(group.nonempty_sector_count == 38259 for group in certificate.groups)
    assert all(group.canonical_orbit_count == 21170 for group in certificate.groups)
    certificate.validate_live_state()


def test_certificate_is_factory_only_frozen_and_has_no_operator_surface() -> None:
    certificate = _certificate()
    with pytest.raises(FrozenInstanceError):
        certificate.whole_inventory_reciprocity_established = False
    with pytest.raises(TypeError, match="factory-only"):
        Vituri2024Q003SourceBoundReciprocityCertificate(
            _factory_token=object(),
            certifier_implementation_fingerprint=certificate.certifier_implementation_fingerprint,
            evidence_root_sha256=certificate.evidence_root_sha256,
            reviewed_member_root_sha256=certificate.reviewed_member_root_sha256,
            historical_source_commit=certificate.historical_source_commit,
            historical_implementation_fingerprint=certificate.historical_implementation_fingerprint,
            candidate_result_sha256=certificate.candidate_result_sha256,
            recovery_complete_sha256=certificate.recovery_complete_sha256,
            postrun_attestation_sha256=certificate.postrun_attestation_sha256,
            theorem_review_sha256=certificate.theorem_review_sha256,
            certifier_review_sha256=certificate.certifier_review_sha256,
            groups=certificate.groups,
        )
    for name in ("matvec", "rmatvec", "linear_operator", "eigenvalues"):
        assert not hasattr(certificate, name)
        assert name not in certification.__all__


def test_module_token_cannot_forge_changed_certificate_or_group() -> None:
    certificate = _certificate()
    fields = {
        "certifier_implementation_fingerprint": certificate.certifier_implementation_fingerprint,
        "evidence_root_sha256": certificate.evidence_root_sha256,
        "reviewed_member_root_sha256": certificate.reviewed_member_root_sha256,
        "historical_source_commit": certificate.historical_source_commit,
        "historical_implementation_fingerprint": certificate.historical_implementation_fingerprint,
        "candidate_result_sha256": certificate.candidate_result_sha256,
        "recovery_complete_sha256": certificate.recovery_complete_sha256,
        "postrun_attestation_sha256": certificate.postrun_attestation_sha256,
        "theorem_review_sha256": certificate.theorem_review_sha256,
        "certifier_review_sha256": certificate.certifier_review_sha256,
        "groups": certificate.groups,
    }
    with pytest.raises(ValueError, match="evidence root is not pinned"):
        Vituri2024Q003SourceBoundReciprocityCertificate(
            _factory_token=certification._TOKEN,
            **{**fields, "evidence_root_sha256": "0" * 64},
        )
    group = certificate.groups[0]
    forged_group = certification.Vituri2024Q003SourceBoundReciprocityGroup(
        group_label=group.group_label,
        path_id="forged-path",
        endpoint_sha256=group.endpoint_sha256,
        density_sha256=group.density_sha256,
        fresh_hamiltonian_sha256=group.fresh_hamiltonian_sha256,
        comparison_fingerprint=group.comparison_fingerprint,
        context_fingerprint=group.context_fingerprint,
        response_fingerprint=group.response_fingerprint,
        inventory_fingerprint=group.inventory_fingerprint,
        canonical_orbit_fingerprint=group.canonical_orbit_fingerprint,
        sector_dimension_fingerprint=group.sector_dimension_fingerprint,
        support_inventory_fingerprint=group.support_inventory_fingerprint,
        nonempty_sector_count=group.nonempty_sector_count,
        canonical_orbit_count=group.canonical_orbit_count,
    )
    with pytest.raises(ValueError, match="group payload is not pinned"):
        Vituri2024Q003SourceBoundReciprocityCertificate(
            _factory_token=certification._TOKEN,
            **{**fields, "groups": (forged_group, certificate.groups[1])},
        )


def test_public_surface_has_no_caller_selectable_trust_root() -> None:
    for name in certification.__all__:
        assert getattr(abc_trilayer, name) is getattr(certification, name)
        assert name in abc_trilayer.__all__
    assert not hasattr(certification, "approve_vituri2024_q003_source_bound_reciprocity")
    source = Path(certification.__file__).read_text()
    assert "import numpy" not in source
    assert "full_response import" not in source
    assert "full_hessian import" not in source


@pytest.mark.parametrize("role", ARTIFACT_ROLES)
def test_every_artifact_byte_string_is_pinned(role: str) -> None:
    artifacts, members = _evidence()
    artifacts[role] += b"\n"
    with pytest.raises(ValueError, match="pinned trust root"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )


@pytest.mark.parametrize("path", REVIEWED_CAPSULE_MEMBERS)
def test_every_reviewed_member_byte_string_is_pinned(path: str) -> None:
    artifacts, members = _evidence()
    members[path] += b"\n"
    with pytest.raises(ValueError, match="pinned trust root"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )


def test_missing_extra_and_nonbytes_evidence_fail_closed() -> None:
    artifacts, members = _evidence()
    del artifacts["candidate_ready"]
    with pytest.raises(ValueError, match="role inventory"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )
    artifacts, members = _evidence()
    members["extra"] = b"x"
    with pytest.raises(ValueError, match="member inventory"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )
    artifacts, members = _evidence()
    artifacts["candidate_ready"] = bytearray(artifacts["candidate_ready"])
    with pytest.raises(TypeError, match="exact bytes"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )


def test_live_binding_and_constant_drift_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, members = _evidence()
    original = certification._validate_comparison
    monkeypatch.setattr(certification, "_validate_comparison", lambda *_args: None)
    with pytest.raises(RuntimeError, match="binding drifted"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )
    monkeypatch.setattr(certification, "_validate_comparison", original)
    monkeypatch.setattr(certification, "_REVIEW_GATES", certification._REVIEW_GATES[:-1])
    with pytest.raises(RuntimeError, match="constant drifted"):
        certify_vituri2024_q003_source_bound_reciprocity(
            artifacts=artifacts,
            reviewed_capsule_members=members,
        )


def test_implementation_fingerprint_is_stable_and_source_based() -> None:
    first = vituri2024_q003_source_bound_reciprocity_implementation_fingerprint()
    second = vituri2024_q003_source_bound_reciprocity_implementation_fingerprint()
    assert first == second
    assert len(first) == 64
