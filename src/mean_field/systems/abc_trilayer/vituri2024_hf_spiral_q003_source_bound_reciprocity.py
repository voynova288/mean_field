"""Pinned artifact-only reciprocity certificate for the two Vituri q003 sources.

The trust root is deliberately not caller-selectable: this module pins the exact
job-492881/job-493109 artifacts, archived production members, and independently
reviewed theorem record. It imports no historical numerical implementation and
exposes no matrix action or eigensolver API.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final, Mapping

from .vituri2024_hf_spiral_q003_reciprocity_trust import (
    PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256,
    PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256,
)

VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_API_VERSION: Final[str] = (
    "vituri2024_q003_source_bound_whole_inventory_reciprocity.v1"
)
VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_SCOPE: Final[str] = (
    "exact_job492881_job493109_two_q003_density_groups_algebraic_reciprocity"
)
VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_THEOREM: Final[str] = (
    "real_direct_rank_one_and_hermitian_exchange_convolution_with_signed_"
    "conjugation_adjoint_transition_restriction_imply_real_reciprocity"
)
VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_AUTHORITY: Final[str] = (
    "whole_inventory_algebraic_reciprocity_only_without_scalar_hessian_"
    "linear_operator_eigensolver_local_stability_production_or_paper_authority"
)

ARTIFACT_ROLES: Final[tuple[str, ...]] = (
    "candidate_contract",
    "candidate_detached_approval",
    "candidate_input_manifest",
    "candidate_prelaunch_review",
    "candidate_reduced_qualification",
    "candidate_result",
    "candidate_ready",
    "certifier_review",
    "historical_capsule_manifest",
    "historical_source_commit",
    "historical_source_manifest",
    "postrun_attestation",
    "recovery_complete",
    "recovery_summary",
    "theorem_review",
)
REVIEWED_CAPSULE_MEMBERS: Final[tuple[str, ...]] = (
    "bin/control_common.py",
    "bin/finalize_after_srun.py",
    "bin/publish_wrapper_failure.py",
    "bin/release_held.py",
    "bin/run_qualifier.py",
    "bin/submit_held.py",
    "source/src/mean_field/core/hf/zero_temperature_sector_stability.py",
    "source/src/mean_field/systems/abc_trilayer/vituri2024_hf_fft.py",
    "source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_hessian.py",
    "source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_reciprocity.py",
    "source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_response.py",
    "source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_stability.py",
    "submit.sbatch",
)

_PINNED_ARTIFACT_SHA256: Final[tuple[tuple[str, str], ...]] = (
    ("candidate_contract", "c07a6aeadcba0cfe5eff5dab00f8c24ccb3ec2171a882c2899f809923314d6ca"),
    ("candidate_detached_approval", "54947fedb0dc5512658470a62652aff5caf62ab268fb61ea4de245e8b07f0bd8"),
    ("candidate_input_manifest", "f11130012df63b5a40e806374f615326e639540c5dfd97cf8995bc651ce4912e"),
    ("candidate_prelaunch_review", "c794df3ba86a02bee4aac86f035c0229630f400e274b0d332c5b54e8d98553fe"),
    ("candidate_reduced_qualification", "7c03c34d2fde19d514fba1bf4b93d7eee4b22dcf42e702cd97d714c0c9f29108"),
    ("candidate_result", "614bdd6bcabb8c88441fe7e201e66088c0629a36dee99229a2aa00b07dc512df"),
    ("candidate_ready", "d60a458fbde44f4083de229094ca51fa37db2e780661a7eae7a9909f96f88954"),
    ("historical_capsule_manifest", "33fd6d462d553cbecd395494051426527d32e9ca24de48efcf042e154b8c1823"),
    ("historical_source_commit", "c00f1d6a3c0eb026e13471862c29d118023280efa0e7199f78075aa9f052169a"),
    ("historical_source_manifest", "1c148ac014daa16d4f646ea6331465a4083926779b445cdd1dd0f6c40ef79aec"),
    ("postrun_attestation", "8971432f18f60237199ceafa983a385b2f1aad27266bab57cf6e69d2fdb670ce"),
    ("recovery_complete", "c72701e537c24ac1b18a84dc3a877f3d2299194e191fbc9273861b21713e12e7"),
    ("recovery_summary", "54be6dc934c36aa66cc8e5cb2e64039ab91fe3b63b97b7e39ff241013f9822aa"),
    ("theorem_review", "e4fac5c79a2154cf8de1048b51f8cd21b2317b30cb4250a527e9a8cb924f2af2"),
)
_PINNED_MEMBER_SHA256: Final[tuple[tuple[str, str], ...]] = (
    ("bin/control_common.py", "1e06b1bc38ddb66d792c23857e80f5a4b536c1c719ab32a730cf198296b789b5"),
    ("bin/finalize_after_srun.py", "b044b3964aaab54734c7718e43f25458011d844b93715b7b36e6a0b88e570577"),
    ("bin/publish_wrapper_failure.py", "41cbb29b87409f9b5066b559bb6963b4b6f5f7d1c308b104f44a964f521bc319"),
    ("bin/release_held.py", "e09572917a531f89f45fa45eebbf1b575e911636f79c6d8b2278ed873f102d6d"),
    ("bin/run_qualifier.py", "73b6117e07e620e5c9d48c27b85cecd143535221c8f0cdd255320e8dc7749902"),
    ("bin/submit_held.py", "3c9815df4f2566228b642fce0ba6d11035ee299955063e34c82437e457a0f8c4"),
    ("source/src/mean_field/core/hf/zero_temperature_sector_stability.py", "e8c9e5b21d132807ddad6c1b7884bba29bec0b10600a4b1ebf0a3aacb7d2a0eb"),
    ("source/src/mean_field/systems/abc_trilayer/vituri2024_hf_fft.py", "dac25398f5e776bde242ed1e3edd627dc8f0569f7b9b51b1376b05dab0ed8854"),
    ("source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_hessian.py", "82a30f77a643a2867508905745aba85bfd7ebd4d73f4b5e44f387341ca343afc"),
    ("source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_reciprocity.py", "956ad5af24a93c7ea3fa94d85a694f8beec2fa533839fc900df059e3b000472c"),
    ("source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_response.py", "7dd8f9d0874121f7258facc89aa7615b123360ae8839fa847c2e1eb14247aa91"),
    ("source/src/mean_field/systems/abc_trilayer/vituri2024_hf_spiral_full_stability.py", "e1f63a0bfe43a722ba0d04260a5fab67fe060d10217b2ef1b973b7c4fd8a623d"),
    ("submit.sbatch", "8a7284eef38ac7cd7d7bf67bf89071b75ad6bd7293c46175ceea293f5262770d"),
)
_PINNED_SOURCE_COMMIT: Final[str] = "6dcc7248ae581dfd3f1d5e98a4c25cfa2d543e86"
_PINNED_HISTORICAL_IMPLEMENTATION: Final[str] = (
    "8fca779ec244dc4ac4ca5cbf5dd1707d86e7f73278979d63cecf3618a90c7d96"
)
_PINNED_ENDPOINTS: Final[tuple[tuple[str, str, int], ...]] = (
    ("input/group_3307_endpoint_38b984fdb7a3.npz", "5b010d357ce5d9c06d4a03fb0d44e1542b610ca96a47aa322eb091ddbcd795b8", 5041058),
    ("input/group_40fd_endpoint_73ed6f5bf440.npz", "c212c7d83255f99c955b3646ed0c313ab5b43b3d0208f71f0a3233a632bf87f6", 5041058),
)
_PINNED_GROUPS: Final[tuple[tuple[str, str], ...]] = (
    ("3307", "26690d7010d70829910341aecf9e61f45c8e598c0328388238ee4389c16d6a52"),
    ("40fd", "30fcc30b0001c65f0d3bceb52ef06d53883d89bd988f3bfff0758e05d3088a1a"),
)
_PINNED_CERTIFICATE_GROUP_FINGERPRINTS: Final[tuple[tuple[str, str], ...]] = (
    ("3307", "8ddf9c691e0f7a0511583d31e8e0d5e43bb5b3aadd4b64b1bbeac784e39b1b65"),
    ("40fd", "55dabcd5cb7f082008ba4060da193838f004487af79084b33f45d491b4ddf66d"),
)
_FALSE_AUTHORITIES: Final[tuple[str, ...]] = (
    "stationarity_established",
    "literal_float_full_functional_parity_established",
    "scientific_authority_promoted",
    "full_inventory_exact_unitary_scalar_curvature_established",
    "scalar_hessian_authority_established",
    "hermitian_eigensolver_authorized",
    "full_local_stability_established",
    "production_ready",
    "paper_reproduction_verified",
)
_STRUCTURAL_GATES: Final[tuple[str, ...]] = (
    "direct_rank_one_identity_bound",
    "exchange_mdagger_ck_m_identity_bound",
    "factor_two_real_packing_bound",
    "one_body_real_diagonal",
    "signed_conjugation_covariance_derived",
    "transition_injection_extraction_adjoint_derived",
)
_REVIEW_GATES: Final[tuple[str, ...]] = (
    "complete_orbit_coverage_reviewed",
    "direct_coefficient_real_established",
    "direct_sign_irrelevant_to_reciprocity",
    "exchange_convolution_hermitian_established",
    "exchange_convolution_psd_not_required",
    "factor_two_real_packing_reviewed",
    "no_cross_orbit_leakage_reviewed",
    "nonzero_self_conjugate_sector_absent_reviewed",
    "production_factorizations_independently_reviewed",
    "scalar_hessian_separated_from_reciprocity",
    "signed_conjugation_covariance_reviewed",
    "stationarity_separated_from_reciprocity",
    "transition_assignment_injective_reviewed",
    "transition_assignment_surjective_reviewed",
    "transition_injection_extraction_adjoint_reviewed",
)
_TOKEN = object()


def _expected_artifact_sha256() -> tuple[tuple[str, str], ...]:
    pinned = dict(_PINNED_ARTIFACT_SHA256)
    pinned["certifier_review"] = PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256
    return tuple((role, pinned[role]) for role in ARTIFACT_ROLES)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fingerprint(value: object) -> str:
    return _sha256_bytes(_canonical(value))


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json(data: bytes, role: str) -> dict[str, object]:
    if type(data) is not bytes:
        raise TypeError(f"{role} must be exact bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{role} root must be an object")
    return value


def _strict_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _exact_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be exact {expected}")


def _exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be exact integer {expected}")


def _zero_number(value: object, label: str) -> None:
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{label} must be an exact JSON number")
    if not math.isfinite(float(value)) or value != 0:
        raise ValueError(f"{label} must be finite zero")


def _module_sha256() -> str:
    return _sha256_bytes(Path(__file__).resolve().read_bytes())


def vituri2024_q003_source_bound_reciprocity_implementation_fingerprint() -> str:
    _validate_live_bindings()
    current = _module_sha256()
    if current != _IMPORT_IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("q003 reciprocity certifier source bytes drifted after import")
    if current != PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256:
        raise RuntimeError("q003 reciprocity certifier lacks external source approval")
    return current


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SourceBoundReciprocityGroup:
    group_label: str
    path_id: str
    endpoint_sha256: str
    density_sha256: str
    fresh_hamiltonian_sha256: str
    comparison_fingerprint: str
    context_fingerprint: str
    response_fingerprint: str
    inventory_fingerprint: str
    canonical_orbit_fingerprint: str
    sector_dimension_fingerprint: str
    support_inventory_fingerprint: str
    nonempty_sector_count: int
    canonical_orbit_count: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.group_label not in dict(_PINNED_GROUPS):
            raise ValueError("q003 group label is not pinned")
        if type(self.path_id) is not str or not self.path_id:
            raise ValueError("q003 path id is invalid")
        for name in (
            "endpoint_sha256", "density_sha256", "fresh_hamiltonian_sha256",
            "comparison_fingerprint", "context_fingerprint", "response_fingerprint",
            "inventory_fingerprint", "canonical_orbit_fingerprint",
            "sector_dimension_fingerprint", "support_inventory_fingerprint",
        ):
            _strict_sha256(getattr(self, name), name)
        if type(self.nonempty_sector_count) is not int or self.nonempty_sector_count <= 0:
            raise ValueError("nonempty sector count is invalid")
        if type(self.canonical_orbit_count) is not int or self.canonical_orbit_count <= 0:
            raise ValueError("canonical orbit count is invalid")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fingerprint"
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SourceBoundReciprocityCertificate:
    """Certificate for algebraic reciprocity of exactly the two pinned q003 sources."""

    _factory_token: InitVar[object]
    certifier_implementation_fingerprint: str
    evidence_root_sha256: str
    reviewed_member_root_sha256: str
    historical_source_commit: str
    historical_implementation_fingerprint: str
    candidate_result_sha256: str
    recovery_complete_sha256: str
    postrun_attestation_sha256: str
    theorem_review_sha256: str
    certifier_review_sha256: str
    groups: tuple[Vituri2024Q003SourceBoundReciprocityGroup, ...]
    fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_API_VERSION, init=False
    )
    scope: str = field(default=VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_SCOPE, init=False)
    theorem: str = field(default=VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_THEOREM, init=False)
    authority: str = field(default=VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_AUTHORITY, init=False)
    artifact_only: bool = field(default=True, init=False)
    algebraic_reciprocity_only: bool = field(default=True, init=False)
    runtime_bitwise_reciprocity_established: bool = field(default=False, init=False)
    stationarity_established: bool = field(default=False, init=False)
    literal_float_full_functional_parity_established: bool = field(
        default=False, init=False
    )
    scientific_authority_promoted: bool = field(default=False, init=False)
    both_groups_certified: bool = field(default=True, init=False)
    no_postselection: bool = field(default=True, init=False)
    source_bound_q003_reciprocity_established: bool = field(default=True, init=False)
    whole_inventory_reciprocity_established: bool = field(default=True, init=False)
    full_inventory_exact_unitary_scalar_curvature_established: bool = field(default=False, init=False)
    scalar_hessian_authority_established: bool = field(default=False, init=False)
    linear_operator_authorized: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _TOKEN:
            raise TypeError("q003 reciprocity certificates are factory-only")
        object.__setattr__(self, "fingerprint", _fingerprint(self._payload()))
        self.validate_live_state()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("_factory_token", "fingerprint", "groups")
        }
        payload["groups"] = tuple(
            {
                name: getattr(group, name)
                for name in group.__dataclass_fields__
            }
            for group in self.groups
        )
        return payload

    def validate_live_state(self) -> None:
        _validate_live_bindings()
        for name in (
            "certifier_implementation_fingerprint", "evidence_root_sha256",
            "reviewed_member_root_sha256", "historical_implementation_fingerprint",
            "candidate_result_sha256", "recovery_complete_sha256",
            "postrun_attestation_sha256", "theorem_review_sha256",
            "certifier_review_sha256",
        ):
            _strict_sha256(getattr(self, name), name)
        if self.certifier_implementation_fingerprint != PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256:
            raise ValueError("certificate certifier fingerprint is not externally pinned")
        if self.evidence_root_sha256 != _fingerprint(_expected_artifact_sha256()):
            raise ValueError("certificate evidence root is not pinned")
        if self.reviewed_member_root_sha256 != _fingerprint(_PINNED_MEMBER_SHA256):
            raise ValueError("certificate member root is not pinned")
        if self.historical_source_commit != _PINNED_SOURCE_COMMIT:
            raise ValueError("certificate source commit drifted")
        if self.historical_implementation_fingerprint != _PINNED_HISTORICAL_IMPLEMENTATION:
            raise ValueError("certificate historical implementation drift")
        if self.candidate_result_sha256 != dict(_PINNED_ARTIFACT_SHA256)["candidate_result"]:
            raise ValueError("certificate candidate result drifted")
        if self.recovery_complete_sha256 != dict(_PINNED_ARTIFACT_SHA256)["recovery_complete"]:
            raise ValueError("certificate recovery record drifted")
        if self.postrun_attestation_sha256 != dict(_PINNED_ARTIFACT_SHA256)["postrun_attestation"]:
            raise ValueError("certificate postrun attestation drifted")
        if self.theorem_review_sha256 != dict(_PINNED_ARTIFACT_SHA256)["theorem_review"]:
            raise ValueError("certificate theorem review drifted")
        if self.certifier_review_sha256 != PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256:
            raise ValueError("certificate certifier review drifted")
        if tuple(group.group_label for group in self.groups) != ("3307", "40fd"):
            raise ValueError("certificate group inventory drifted")
        for group in self.groups:
            if group.comparison_fingerprint != dict(_PINNED_GROUPS)[group.group_label]:
                raise ValueError("certificate group receipt is not pinned")
            if group.fingerprint != dict(_PINNED_CERTIFICATE_GROUP_FINGERPRINTS)[
                group.group_label
            ]:
                raise ValueError("certificate group payload is not pinned")
        true_flags = (
            self.artifact_only, self.algebraic_reciprocity_only,
            self.both_groups_certified, self.no_postselection,
            self.source_bound_q003_reciprocity_established,
            self.whole_inventory_reciprocity_established,
        )
        false_flags = (
            self.runtime_bitwise_reciprocity_established,
            self.stationarity_established,
            self.literal_float_full_functional_parity_established,
            self.scientific_authority_promoted,
            self.full_inventory_exact_unitary_scalar_curvature_established,
            self.scalar_hessian_authority_established, self.linear_operator_authorized,
            self.hermitian_eigensolver_authorized, self.full_local_stability_established,
            self.production_ready, self.paper_reproduction_verified,
        )
        if any(value is not True for value in true_flags) or any(
            value is not False for value in false_flags
        ):
            raise ValueError("certificate authority drifted")
        if self.fingerprint != _fingerprint(self._payload()):
            raise RuntimeError("certificate fingerprint drifted")


def _parse_source_manifest(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("historical source manifest is not UTF-8") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in result:
            raise ValueError("historical source manifest line is invalid or duplicated")
        result[parts[1]] = _strict_sha256(parts[0], f"source manifest {parts[1]}")
    return result


def _parse_capsule_manifest(data: bytes) -> dict[str, dict[str, object]]:
    manifest = _load_json(data, "historical_capsule_manifest")
    if manifest.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_"
        "comparison_capsule_manifest.v24"
    ):
        raise ValueError("historical capsule manifest schema is invalid")
    files = manifest.get("files")
    if type(files) is not list:
        raise TypeError("historical capsule file inventory is invalid")
    result: dict[str, dict[str, object]] = {}
    for item in files:
        if type(item) is not dict or set(item) != {"mode", "path", "sha256", "size_bytes"}:
            raise ValueError("historical capsule file entry is invalid")
        path = item["path"]
        if type(path) is not str or path in result:
            raise ValueError("historical capsule file path is invalid or duplicated")
        _strict_sha256(item["sha256"], f"capsule member {path}")
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            raise ValueError("historical capsule member size is invalid")
        result[path] = item
    return result


def _validate_false_authorities(value: Mapping[str, object], label: str) -> None:
    if "linear_operator_authorized" in value:
        _exact_bool(
            value["linear_operator_authorized"],
            False,
            f"{label}.linear_operator_authorized",
        )
    for name in _FALSE_AUTHORITIES:
        if name in value:
            _exact_bool(value[name], False, f"{label}.{name}")


def _validate_comparison(comparison: dict[str, object], label: str) -> None:
    fingerprint = _strict_sha256(comparison.get("fingerprint"), "comparison fingerprint")
    payload = dict(comparison)
    del payload["fingerprint"]
    if _fingerprint(payload) != fingerprint:
        raise ValueError("comparison receipt fingerprint does not reproduce")
    if fingerprint != dict(_PINNED_GROUPS)[label]:
        raise ValueError("comparison fingerprint is not the pinned group receipt")
    if comparison.get("implementation_fingerprint") != _PINNED_HISTORICAL_IMPLEMENTATION:
        raise ValueError("historical comparison implementation differs")
    if comparison.get("approved_implementation_fingerprint") != _PINNED_HISTORICAL_IMPLEMENTATION:
        raise ValueError("approved historical implementation differs")
    if comparison.get("approved_source_commit") != _PINNED_SOURCE_COMMIT:
        raise ValueError("comparison source commit differs")
    for gate in _STRUCTURAL_GATES:
        _exact_bool(comparison.get(gate), True, f"comparison.{gate}")
    _exact_bool(comparison.get("passed"), True, "comparison.passed")
    _exact_bool(comparison.get("candidate_only"), True, "comparison.candidate_only")
    _exact_bool(
        comparison.get("whole_inventory_reciprocity_established"),
        False,
        "candidate comparison reciprocity authority",
    )
    _validate_false_authorities(comparison, "candidate comparison")
    if comparison.get("failed_gates") != []:
        raise ValueError("candidate comparison has failed gates")
    for name in (
        "support_mismatch_count", "conjugation_structure_mismatch_count",
        "paired_lane_overlap_count", "sector_dimension_mismatch_count",
        "canonical_orbit_mismatch_count",
    ):
        _exact_int(comparison.get(name), 0, f"comparison.{name}")
    _zero_number(comparison.get("maximum_kernel_imaginary_residual"), "kernel imaginary residual")
    _zero_number(comparison.get("maximum_kernel_even_residual"), "kernel even residual")
    expected_counts = {
        "mesh_size": 81,
        "nk": 6561,
        "displacement_count_checked": 25921,
        "signed_sector_count_checked": 77763,
        "ordered_mesh_pair_count_covered": 43046721,
        "independently_recomputed_complex_dimension": 15052040,
        "inventory_complex_dimension": 15052040,
        "inventory_real_dimension": 30104080,
        "canonical_orbit_complex_dimension_sum": 15052040,
    }
    for name, expected in expected_counts.items():
        _exact_int(comparison.get(name), expected, f"comparison.{name}")
    for name in ("nonempty_sector_count", "canonical_orbit_count"):
        value = comparison.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"comparison.{name} must be a positive exact integer")
    for name in (
        "context_fingerprint", "response_fingerprint", "inventory_fingerprint",
        "canonical_orbit_fingerprint", "sector_dimension_fingerprint",
        "support_inventory_fingerprint",
    ):
        _strict_sha256(comparison.get(name), f"comparison.{name}")


def certify_vituri2024_q003_source_bound_reciprocity(
    *,
    artifacts: Mapping[str, bytes],
    reviewed_capsule_members: Mapping[str, bytes],
) -> Vituri2024Q003SourceBoundReciprocityCertificate:
    """Certify the exact pinned q003 evidence without executing numerical code."""

    _validate_live_bindings()
    if type(artifacts) is not dict or tuple(sorted(artifacts)) != tuple(
        sorted(ARTIFACT_ROLES)
    ):
        raise ValueError("certification artifact role inventory is not exact")
    if type(reviewed_capsule_members) is not dict or tuple(
        sorted(reviewed_capsule_members)
    ) != tuple(sorted(REVIEWED_CAPSULE_MEMBERS)):
        raise ValueError("reviewed capsule member inventory is not exact")
    if any(type(value) is not bytes for value in artifacts.values()):
        raise TypeError("certification artifacts must be exact bytes")
    if any(type(value) is not bytes for value in reviewed_capsule_members.values()):
        raise TypeError("reviewed capsule members must be exact bytes")
    artifact_hashes = {role: _sha256_bytes(artifacts[role]) for role in ARTIFACT_ROLES}
    member_hashes = {
        path: _sha256_bytes(reviewed_capsule_members[path])
        for path in REVIEWED_CAPSULE_MEMBERS
    }
    if tuple(artifact_hashes.items()) != _expected_artifact_sha256():
        raise ValueError("artifact bytes differ from the pinned trust root")
    if tuple(member_hashes.items()) != _PINNED_MEMBER_SHA256:
        raise ValueError("reviewed member bytes differ from the pinned trust root")

    contract = _load_json(artifacts["candidate_contract"], "candidate_contract")
    detached = _load_json(
        artifacts["candidate_detached_approval"], "candidate_detached_approval"
    )
    input_manifest = _load_json(
        artifacts["candidate_input_manifest"], "candidate_input_manifest"
    )
    prelaunch = _load_json(
        artifacts["candidate_prelaunch_review"], "candidate_prelaunch_review"
    )
    reduced = _load_json(
        artifacts["candidate_reduced_qualification"], "candidate_reduced_qualification"
    )
    result = _load_json(artifacts["candidate_result"], "candidate_result")
    ready = _load_json(artifacts["candidate_ready"], "candidate_ready")
    certifier_review = _load_json(artifacts["certifier_review"], "certifier_review")
    capsule_files = _parse_capsule_manifest(artifacts["historical_capsule_manifest"])
    source_manifest = _parse_source_manifest(artifacts["historical_source_manifest"])
    postrun = _load_json(artifacts["postrun_attestation"], "postrun_attestation")
    recovery_complete = _load_json(
        artifacts["recovery_complete"], "recovery_complete"
    )
    recovery_summary = _load_json(artifacts["recovery_summary"], "recovery_summary")
    theorem_review = _load_json(artifacts["theorem_review"], "theorem_review")

    static_roles = {
        "candidate_contract": "config/contract.json",
        "candidate_detached_approval": "config/DETACHED_APPROVAL_RECORD.json",
        "candidate_input_manifest": "input/INPUT_MANIFEST.json",
        "candidate_prelaunch_review": "config/PRELAUNCH_REVIEW.json",
        "candidate_reduced_qualification": "config/REDUCED_EXHAUSTIVE_QUALIFICATION.json",
        "historical_source_commit": "provenance/SOURCE_COMMIT.txt",
        "historical_source_manifest": "provenance/SOURCE_SHA256SUMS.txt",
    }
    for role, path in static_roles.items():
        item = capsule_files.get(path)
        if type(item) is not dict or item.get("sha256") != artifact_hashes[role]:
            raise ValueError(f"capsule manifest does not bind {role}")
        _exact_int(item.get("size_bytes"), len(artifacts[role]), f"capsule {role} size")
    for path in REVIEWED_CAPSULE_MEMBERS:
        item = capsule_files.get(path)
        if type(item) is not dict or item.get("sha256") != member_hashes[path]:
            raise ValueError(f"capsule manifest does not bind reviewed member {path}")
        _exact_int(
            item.get("size_bytes"),
            len(reviewed_capsule_members[path]),
            f"capsule reviewed member {path} size",
        )
    source_paths = tuple(
        path.removeprefix("source/")
        for path in REVIEWED_CAPSULE_MEMBERS
        if path.startswith("source/")
    )
    for capsule_path in REVIEWED_CAPSULE_MEMBERS:
        if capsule_path.startswith("source/"):
            source_path = capsule_path.removeprefix("source/")
            if source_manifest.get(source_path) != member_hashes[capsule_path]:
                raise ValueError(f"source manifest does not bind {source_path}")
    if len(source_paths) != 6:
        raise RuntimeError("reviewed source path inventory drifted")
    try:
        source_commit_text = artifacts["historical_source_commit"].decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("historical source commit is not ASCII") from error
    if source_commit_text != _PINNED_SOURCE_COMMIT:
        raise ValueError("historical source commit text differs")

    if input_manifest.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_"
        "comparison_input_manifest.v24"
    ):
        raise ValueError("candidate input manifest schema differs")
    input_files = input_manifest.get("files")
    if type(input_files) is not list:
        raise TypeError("candidate input manifest file inventory is invalid")
    input_by_path = {
        item.get("path"): item for item in input_files if type(item) is dict
    }
    if len(input_by_path) != len(input_files):
        raise ValueError("candidate input manifest has invalid or duplicate paths")
    for path, digest, size in _PINNED_ENDPOINTS:
        input_item = input_by_path.get(path)
        capsule_item = capsule_files.get(path)
        for item, label in ((input_item, "input"), (capsule_item, "capsule")):
            if type(item) is not dict or item.get("sha256") != digest:
                raise ValueError(f"{label} manifest does not bind endpoint {path}")
            _exact_int(item.get("size_bytes"), size, f"{label} endpoint size")

    if contract.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_"
        "comparison_contract.v24"
    ) or contract.get("source_commit") != _PINNED_SOURCE_COMMIT:
        raise ValueError("candidate contract lineage differs")
    physical = contract.get("physical")
    contract_authority = contract.get("authority")
    if type(physical) is not dict or type(contract_authority) is not dict:
        raise TypeError("candidate contract sections are invalid")
    _exact_int(physical.get("mesh_size"), 81, "candidate mesh size")
    if type(physical.get("q_a0")) is not float or physical.get("q_a0") != 0.03:
        raise ValueError("candidate q_a0 must be exact float 0.03")
    _exact_bool(contract_authority.get("candidate_only"), True, "contract candidate_only")
    _exact_bool(contract_authority.get("no_postselection"), True, "contract no_postselection")
    _exact_bool(
        contract_authority.get("whole_inventory_reciprocity_established"),
        False,
        "contract reciprocity authority",
    )
    if any(contract_authority.get(name) is True for name in (
        "scalar_hessian_authority_established", "hermitian_eigensolver_authorized",
        "full_local_stability_established", "scientific_authority_promoted",
        "linear_operator_authorized",
    )):
        raise ValueError("candidate contract contains promoted later authority")

    if detached.get("schema") != (
        "mean_field.vituri2024.whole_inventory_reciprocity_detached_approval_record.v24"
    ) or detached.get("source_commit") != _PINNED_SOURCE_COMMIT:
        raise ValueError("historical detached approval lineage differs")
    if detached.get("expected_implementation_fingerprint") != (
        _PINNED_HISTORICAL_IMPLEMENTATION
    ):
        raise ValueError("historical detached approval implementation differs")
    if detached.get("review_record_sha256") != artifact_hashes[
        "candidate_prelaunch_review"
    ] or detached.get("reduced_exhaustive_qualification_sha256") != artifact_hashes[
        "candidate_reduced_qualification"
    ]:
        raise ValueError("historical detached review lineage differs")
    contract_detached = contract.get("detached_approval")
    if type(contract_detached) is not dict:
        raise TypeError("contract detached approval is invalid")
    detached_fingerprint = _strict_sha256(
        contract_detached.get("approval_fingerprint"),
        "historical detached approval fingerprint",
    )
    approval_payload = {
        "expected_implementation_fingerprint": detached[
            "expected_implementation_fingerprint"
        ],
        "source_commit": detached["source_commit"],
        "review_record_sha256": detached["review_record_sha256"],
        "reduced_exhaustive_qualification_sha256": detached[
            "reduced_exhaustive_qualification_sha256"
        ],
        "approval_record_sha256": artifact_hashes[
            "candidate_detached_approval"
        ],
        "rationale": contract_detached.get("rationale"),
        "api_version": (
            "vituri2024_selected_spin_exact_integer_whole_inventory_reciprocity.v1"
        ),
        "scope": (
            "selected_spin_fixed_rank_exact_integer_no_wrap_whole_transition_inventory"
        ),
        "detached_preexecution_approval": True,
    }
    if _fingerprint(approval_payload) != detached_fingerprint:
        raise ValueError("historical detached approval fingerprint does not reproduce")
    for name in (
        "expected_implementation_fingerprint", "source_commit",
        "review_record_sha256", "reduced_exhaustive_qualification_sha256",
        "approval_record_sha256", "rationale",
    ):
        if contract_detached.get(name) != approval_payload[name]:
            raise ValueError(f"contract detached approval field differs: {name}")

    if prelaunch.get("schema") != (
        "mean_field.vituri2024.whole_inventory_reciprocity_prelaunch_review.v24"
    ) or reduced.get("schema") != (
        "mean_field.vituri2024.whole_inventory_reciprocity_reduced_qualification.v24"
    ):
        raise ValueError("historical review schemas differ")
    for record, label in ((prelaunch, "prelaunch"), (reduced, "reduced")):
        if record.get("source_commit") != _PINNED_SOURCE_COMMIT or record.get(
            "implementation_fingerprint"
        ) != _PINNED_HISTORICAL_IMPLEMENTATION:
            raise ValueError(f"historical {label} lineage differs")
    _exact_int(prelaunch.get("blocker_count"), 0, "prelaunch blocker count")
    if prelaunch.get("verdict") != "candidate_comparison_safe_to_submit":
        raise ValueError("prelaunch verdict differs")
    if reduced.get("pytest_result") != "62 passed":
        raise ValueError("reduced qualification result differs")
    for name in (
        "reduced_every_paired_orbit_real_matrix_symmetric_checked",
        "reduced_exhaustive_signed_action_hermitian_checked",
        "reduced_signed_conjugation_covariance_checked",
        "runtime_primitive_and_inventory_method_drift_canaries_checked",
    ):
        _exact_bool(reduced.get(name), True, f"reduced qualification {name}")

    if result.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_"
        "comparison_result.v24"
    ) or result.get("source_commit") != _PINNED_SOURCE_COMMIT:
        raise ValueError("candidate result lineage differs")
    _exact_bool(result.get("both_groups_evaluated"), True, "result both groups")
    _exact_bool(
        result.get("all_candidate_structural_comparisons_passed"),
        True,
        "result all comparisons passed",
    )
    _exact_bool(result.get("candidate_only"), True, "result candidate_only")
    _exact_bool(
        result.get("whole_inventory_reciprocity_established"),
        False,
        "result candidate reciprocity authority",
    )
    _validate_false_authorities(result, "candidate result")
    representatives = contract.get("representatives")
    groups = result.get("groups")
    if type(representatives) is not list or type(groups) is not list:
        raise TypeError("candidate representatives or groups are invalid")
    representative_by_label = {
        item.get("group_label"): item for item in representatives if type(item) is dict
    }
    if set(representative_by_label) != {"3307", "40fd"} or len(representatives) != 2:
        raise ValueError("candidate representative inventory differs")
    if len(groups) != 2:
        raise ValueError("candidate group count differs")
    certificate_groups: list[Vituri2024Q003SourceBoundReciprocityGroup] = []
    seen: set[str] = set()
    for group in groups:
        if type(group) is not dict:
            raise TypeError("candidate group is not an object")
        label = group.get("group_label")
        if label not in ("3307", "40fd") or label in seen:
            raise ValueError("candidate group labels differ or duplicate")
        seen.add(label)
        representative = representative_by_label[label]
        if group.get("path_id") != representative.get("path_id"):
            raise ValueError("candidate path lineage differs")
        endpoint_path, endpoint_digest, _size = next(
            item for item in _PINNED_ENDPOINTS if label in item[0]
        )
        if representative.get("artifact") != endpoint_path or group.get(
            "source_artifact_sha256"
        ) != endpoint_digest:
            raise ValueError("candidate endpoint lineage differs")
        for name in (
            "density_sha256_stability_v2", "fresh_hamiltonian_sha256_stability_v2"
        ):
            if group.get(name) != representative.get(name):
                raise ValueError(f"candidate {name} lineage differs")
        comparison = group.get("comparison")
        if type(comparison) is not dict:
            raise TypeError("candidate comparison is not an object")
        _validate_comparison(comparison, label)
        if comparison.get("approval_fingerprint") != detached_fingerprint:
            raise ValueError("comparison detached approval fingerprint differs")
        if comparison.get("approval_record_sha256") != artifact_hashes[
            "candidate_detached_approval"
        ] or comparison.get("review_record_sha256") != artifact_hashes[
            "candidate_prelaunch_review"
        ] or comparison.get("reduced_exhaustive_qualification_sha256") != artifact_hashes[
            "candidate_reduced_qualification"
        ]:
            raise ValueError("comparison review lineage differs")
        certificate_groups.append(
            Vituri2024Q003SourceBoundReciprocityGroup(
                group_label=label,
                path_id=group["path_id"],
                endpoint_sha256=endpoint_digest,
                density_sha256=group["density_sha256_stability_v2"],
                fresh_hamiltonian_sha256=group[
                    "fresh_hamiltonian_sha256_stability_v2"
                ],
                comparison_fingerprint=comparison["fingerprint"],
                context_fingerprint=comparison["context_fingerprint"],
                response_fingerprint=comparison["response_fingerprint"],
                inventory_fingerprint=comparison["inventory_fingerprint"],
                canonical_orbit_fingerprint=comparison[
                    "canonical_orbit_fingerprint"
                ],
                sector_dimension_fingerprint=comparison[
                    "sector_dimension_fingerprint"
                ],
                support_inventory_fingerprint=comparison[
                    "support_inventory_fingerprint"
                ],
                nonempty_sector_count=comparison["nonempty_sector_count"],
                canonical_orbit_count=comparison["canonical_orbit_count"],
            )
        )
    if seen != {"3307", "40fd"}:
        raise ValueError("both pinned groups were not evaluated")

    if ready.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_"
        "comparison_ready.v24"
    ) or ready.get("result_sha256") != artifact_hashes["candidate_result"]:
        raise ValueError("candidate READY lineage differs")
    _exact_bool(
        ready.get("execution_ready_for_wrapper_finalization"),
        True,
        "candidate READY execution gate",
    )
    if recovery_summary.get("schema") != (
        "mean_field.vituri2024.fig2_q003_reciprocity_postflight_recovery_summary.v8"
    ) or recovery_complete.get("schema") != (
        "mean_field.vituri2024.fig2_q003_reciprocity_postflight_recovery_complete.v8"
    ):
        raise ValueError("recovery schemas differ")
    if recovery_summary.get("original_result_sha256") != artifact_hashes[
        "candidate_result"
    ] or recovery_summary.get("original_ready_sha256") != artifact_hashes[
        "candidate_ready"
    ]:
        raise ValueError("recovery source lineage differs")
    for record, label in (
        (recovery_summary, "recovery summary"),
        (recovery_complete, "recovery complete"),
    ):
        _exact_bool(record.get("zero_scientific_updates"), True, f"{label} zero science")
        _exact_bool(
            record.get("whole_inventory_reciprocity_established"),
            False,
            f"{label} candidate authority",
        )
        _validate_false_authorities(record, label)
    _exact_bool(
        recovery_summary.get("project_modules_imported"),
        False,
        "recovery project imports",
    )
    if recovery_complete.get("summary_sha256") != artifact_hashes[
        "recovery_summary"
    ] or recovery_complete.get("original_result_sha256") != artifact_hashes[
        "candidate_result"
    ]:
        raise ValueError("recovery COMPLETE lineage differs")

    if postrun.get("schema") != (
        "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_candidate_"
        "comparison_postrun_attestation.v1"
    ):
        raise ValueError("postrun attestation schema differs")
    aggregate = postrun.get("aggregate")
    postrun_authority = postrun.get("authority")
    original_job = postrun.get("original_science_job")
    recovery_job = postrun.get("recovery_job")
    postrun_groups = postrun.get("groups")
    if any(type(item) is not dict for item in (
        aggregate, postrun_authority, original_job, recovery_job
    )) or type(postrun_groups) is not list:
        raise TypeError("postrun attestation sections are invalid")
    _exact_bool(aggregate.get("both_groups_evaluated"), True, "postrun both groups")
    _exact_bool(aggregate.get("no_postselection"), True, "postrun no postselection")
    _exact_bool(
        postrun_authority.get("candidate_structural_comparison_evidence_established"),
        True,
        "postrun candidate evidence",
    )
    _exact_bool(
        postrun_authority.get("whole_inventory_reciprocity_established"),
        False,
        "postrun candidate reciprocity authority",
    )
    _validate_false_authorities(postrun_authority, "postrun authority")
    if original_job.get("result", {}).get("sha256") != artifact_hashes[
        "candidate_result"
    ] or original_job.get("capsule_manifest_sha256") != artifact_hashes[
        "historical_capsule_manifest"
    ]:
        raise ValueError("postrun original-job lineage differs")
    if recovery_job.get("summary", {}).get("sha256") != artifact_hashes[
        "recovery_summary"
    ] or recovery_job.get("complete", {}).get("sha256") != artifact_hashes[
        "recovery_complete"
    ]:
        raise ValueError("postrun recovery lineage differs")
    postrun_by_label = {
        item.get("group_label"): item
        for item in postrun_groups
        if type(item) is dict
    }
    if set(postrun_by_label) != {"3307", "40fd"} or len(postrun_groups) != 2:
        raise ValueError("postrun group inventory differs")
    for group in certificate_groups:
        reviewed = postrun_by_label[group.group_label]
        if reviewed.get("comparison_fingerprint") != group.comparison_fingerprint:
            raise ValueError("postrun comparison fingerprint differs")
        _exact_bool(reviewed.get("passed"), True, "postrun group passed")
        _exact_bool(reviewed.get("candidate_only"), True, "postrun group candidate_only")

    if set(theorem_review) != {
        "authority", "blocker_count", "derivation", "derivation_gates",
        "historical_implementation_fingerprint", "historical_source_commit",
        "reviewed_capsule_member_sha256", "reviewers", "schema", "theorem",
        "verdict",
    }:
        raise ValueError("theorem review key set differs")
    if theorem_review.get("schema") != (
        "mean_field.vituri2024.q003_source_bound_reciprocity_theorem_review.v1"
    ) or theorem_review.get("verdict") != "source_bound_reciprocity_theorem_approved":
        raise ValueError("theorem review verdict differs")
    _exact_int(theorem_review.get("blocker_count"), 0, "theorem review blocker count")
    if theorem_review.get("reviewers") != ["reviewer", "mf-reviewer"]:
        raise ValueError("theorem review reviewer inventory differs")
    if theorem_review.get("historical_source_commit") != _PINNED_SOURCE_COMMIT or (
        theorem_review.get("historical_implementation_fingerprint")
        != _PINNED_HISTORICAL_IMPLEMENTATION
    ):
        raise ValueError("theorem review historical lineage differs")
    if theorem_review.get("theorem") != VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_THEOREM:
        raise ValueError("theorem review theorem differs")
    if theorem_review.get("reviewed_capsule_member_sha256") != member_hashes:
        raise ValueError("theorem review member inventory differs")
    gates = theorem_review.get("derivation_gates")
    if type(gates) is not dict or set(gates) != set(_REVIEW_GATES):
        raise ValueError("theorem review derivation gate inventory differs")
    for gate in _REVIEW_GATES:
        _exact_bool(gates[gate], True, f"theorem review {gate}")
    theorem_authority = theorem_review.get("authority")
    if type(theorem_authority) is not dict or set(theorem_authority) != {
        "all_later_authorities_forbidden",
        "whole_inventory_reciprocity_authorized_for_exact_sources_only",
    }:
        raise ValueError("theorem review authority key set differs")
    _exact_bool(
        theorem_authority["all_later_authorities_forbidden"],
        True,
        "theorem review later-authority lock",
    )
    _exact_bool(
        theorem_authority["whole_inventory_reciprocity_authorized_for_exact_sources_only"],
        True,
        "theorem review exact-source authority",
    )
    derivation = theorem_review.get("derivation")
    if type(derivation) is not dict or derivation.get("numerical_scope") != (
        "algebraic reciprocity of the exact finite-square discrete operator represented "
        "by the reviewed source; bitwise FFT self-adjointness is not claimed"
    ):
        raise ValueError("theorem review algebraic scope differs")

    if set(certifier_review) != {
        "authority", "blocker_count", "certifier_source_sha256", "reviewers",
        "schema", "theorem_review_sha256", "verdict",
    }:
        raise ValueError("certifier review key set differs")
    if certifier_review.get("schema") != (
        "mean_field.vituri2024.q003_source_bound_reciprocity_certifier_review.v1"
    ) or certifier_review.get("verdict") != "exact_pinned_certifier_approved":
        raise ValueError("certifier review verdict differs")
    _exact_int(certifier_review.get("blocker_count"), 0, "certifier review blocker count")
    if certifier_review.get("reviewers") != ["reviewer", "mf-reviewer"]:
        raise ValueError("certifier review reviewer inventory differs")
    if certifier_review.get("certifier_source_sha256") != (
        PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256
    ):
        raise ValueError("certifier review source fingerprint differs")
    if certifier_review.get("theorem_review_sha256") != artifact_hashes[
        "theorem_review"
    ]:
        raise ValueError("certifier review theorem lineage differs")
    certifier_authority = certifier_review.get("authority")
    if type(certifier_authority) is not dict or set(certifier_authority) != {
        "all_later_authorities_forbidden",
        "whole_inventory_algebraic_reciprocity_for_exact_pins_only",
    }:
        raise ValueError("certifier review authority key set differs")
    _exact_bool(
        certifier_authority["all_later_authorities_forbidden"],
        True,
        "certifier review later-authority lock",
    )
    _exact_bool(
        certifier_authority["whole_inventory_algebraic_reciprocity_for_exact_pins_only"],
        True,
        "certifier review exact-pin authority",
    )

    implementation = vituri2024_q003_source_bound_reciprocity_implementation_fingerprint()
    certificate_groups.sort(key=lambda group: group.group_label)
    certificate = Vituri2024Q003SourceBoundReciprocityCertificate(
        _factory_token=_TOKEN,
        certifier_implementation_fingerprint=implementation,
        evidence_root_sha256=_fingerprint(_expected_artifact_sha256()),
        reviewed_member_root_sha256=_fingerprint(_PINNED_MEMBER_SHA256),
        historical_source_commit=_PINNED_SOURCE_COMMIT,
        historical_implementation_fingerprint=_PINNED_HISTORICAL_IMPLEMENTATION,
        candidate_result_sha256=artifact_hashes["candidate_result"],
        recovery_complete_sha256=artifact_hashes["recovery_complete"],
        postrun_attestation_sha256=artifact_hashes["postrun_attestation"],
        theorem_review_sha256=artifact_hashes["theorem_review"],
        certifier_review_sha256=artifact_hashes["certifier_review"],
        groups=tuple(certificate_groups),
    )
    _validate_live_bindings()
    if implementation != vituri2024_q003_source_bound_reciprocity_implementation_fingerprint():
        raise RuntimeError("q003 reciprocity certifier changed during certification")
    return certificate


def _validate_live_bindings() -> None:
    for name, expected in _IMPORT_BINDINGS:
        if globals().get(name) is not expected:
            raise RuntimeError(f"q003 reciprocity certifier binding drifted: {name}")
    if (
        ARTIFACT_ROLES != _IMPORT_ARTIFACT_ROLES
        or REVIEWED_CAPSULE_MEMBERS != _IMPORT_REVIEWED_CAPSULE_MEMBERS
        or _PINNED_ARTIFACT_SHA256 != _IMPORT_PINNED_ARTIFACT_SHA256
        or _PINNED_MEMBER_SHA256 != _IMPORT_PINNED_MEMBER_SHA256
        or _REVIEW_GATES != _IMPORT_REVIEW_GATES
        or _STRUCTURAL_GATES != _IMPORT_STRUCTURAL_GATES
        or _FALSE_AUTHORITIES != _IMPORT_FALSE_AUTHORITIES
        or _PINNED_CERTIFICATE_GROUP_FINGERPRINTS
        != _IMPORT_PINNED_CERTIFICATE_GROUP_FINGERPRINTS
        or PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256 != _IMPORT_EXTERNAL_CERTIFIER_SHA256
        or PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256
        != _IMPORT_EXTERNAL_CERTIFIER_REVIEW_SHA256
    ):
        raise RuntimeError("q003 reciprocity certifier constant drifted")


_IMPORT_BINDINGS = tuple(
    (name, globals()[name])
    for name in (
        "_expected_artifact_sha256", "_sha256_bytes", "_canonical",
        "_fingerprint", "_load_json",
        "_parse_source_manifest", "_parse_capsule_manifest",
        "_validate_false_authorities", "_validate_comparison",
        "certify_vituri2024_q003_source_bound_reciprocity",
        "vituri2024_q003_source_bound_reciprocity_implementation_fingerprint",
        "Vituri2024Q003SourceBoundReciprocityGroup",
        "Vituri2024Q003SourceBoundReciprocityCertificate",
    )
)
_IMPORT_ARTIFACT_ROLES = ARTIFACT_ROLES
_IMPORT_REVIEWED_CAPSULE_MEMBERS = REVIEWED_CAPSULE_MEMBERS
_IMPORT_PINNED_ARTIFACT_SHA256 = _PINNED_ARTIFACT_SHA256
_IMPORT_PINNED_MEMBER_SHA256 = _PINNED_MEMBER_SHA256
_IMPORT_REVIEW_GATES = _REVIEW_GATES
_IMPORT_STRUCTURAL_GATES = _STRUCTURAL_GATES
_IMPORT_FALSE_AUTHORITIES = _FALSE_AUTHORITIES
_IMPORT_PINNED_CERTIFICATE_GROUP_FINGERPRINTS = (
    _PINNED_CERTIFICATE_GROUP_FINGERPRINTS
)
_IMPORT_EXTERNAL_CERTIFIER_SHA256 = PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256
_IMPORT_EXTERNAL_CERTIFIER_REVIEW_SHA256 = (
    PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256
)
_IMPORT_IMPLEMENTATION_FINGERPRINT = _module_sha256()

__all__ = [
    "ARTIFACT_ROLES",
    "REVIEWED_CAPSULE_MEMBERS",
    "VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_API_VERSION",
    "VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_AUTHORITY",
    "VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_SCOPE",
    "VITURI2024_Q003_SOURCE_BOUND_RECIPROCITY_THEOREM",
    "Vituri2024Q003SourceBoundReciprocityCertificate",
    "Vituri2024Q003SourceBoundReciprocityGroup",
    "certify_vituri2024_q003_source_bound_reciprocity",
    "vituri2024_q003_source_bound_reciprocity_implementation_fingerprint",
]
