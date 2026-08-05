"""Fail-closed authority contract for Kumar--Xie--MacDonald collective modes.

The paper (arXiv:2010.05946v2) gives a strong seven-mode benchmark at
``nu=-3`` but does not publish a complete executable numerical configuration.
This module records paper-direct facts separately from cited-reference or
reproduction choices.  It deliberately contains no HF/TDHFA implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from mean_field.core.hf.tdhf_goldstone import count_tdhf_goldstones_from_rank

KUMAR_2010_ARXIV = "2010.05946v2"
KUMAR_2010_PDF_SHA256 = (
    "4839d31fd922923875e800a5e5208a94aab018a6714c6b4ee148decff2d4e9a4"
)
KUMAR_2010_SOURCE_ARCHIVE_SHA256 = (
    "4cb4f5990d691349bb362279ad3f726e0db21e24ddd9aff9990ae710eedafdec"
)
KUMAR_2010_RESPONSE_SCOPE = "kumar_2010_nu_minus3_seven_mode_tdhfa_v1"

AuthorityKind = Literal[
    "paper_explicit",
    "cited_reference_explicit",
    "author_source_explicit",
    "reproduction_choice",
]


def _strict_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _authority(kind: str, *, allow_paper: bool) -> None:
    allowed = {
        "cited_reference_explicit",
        "author_source_explicit",
        "reproduction_choice",
    }
    if allow_paper:
        allowed.add("paper_explicit")
    if kind not in allowed:
        raise ValueError("invalid Kumar authority kind")


@dataclass(frozen=True, slots=True)
class KumarPinnedValue:
    value: int | float | str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        _authority(self.authority_kind, allow_paper=True)
        if isinstance(self.value, bool):
            raise TypeError("Kumar pinned value cannot be boolean")
        if _strict_integer(self.value):
            object.__setattr__(self, "value", int(self.value))
        elif isinstance(self.value, (float, np.floating)):
            value = float(self.value)
            if not np.isfinite(value):
                raise ValueError("Kumar pinned value must be finite")
            object.__setattr__(self, "value", value)
        if not str(self.value).strip() or not self.source.strip():
            raise ValueError("Kumar pinned value requires value/source provenance")

    @property
    def fingerprint(self) -> str:
        payload = {
            "value": self.value,
            "authority_kind": self.authority_kind,
            "source": self.source,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KumarParentAuthority:
    model_convention: str
    vf_mev_b0: float
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.model_convention != "linear_dirac_bm":
            raise ValueError("Kumar parent must explicitly identify linear Dirac BM")
        vf = float(self.vf_mev_b0)
        if not np.isfinite(vf) or vf <= 0.0:
            raise ValueError("Kumar parent vf must be finite and positive")
        object.__setattr__(self, "vf_mev_b0", vf)
        _sha256(self.provider_fingerprint, "Kumar parent provider fingerprint")
        _sha256(self.source_artifact_sha256, "Kumar parent source artifact")
        _authority(self.authority_kind, allow_paper=False)
        if not self.source.strip():
            raise ValueError("Kumar parent requires source provenance")


@dataclass(frozen=True, slots=True)
class KumarCutoffAuthority:
    cutoff_kind: Literal["plane_wave", "interaction_transfer", "remote_window"]
    value: int | str
    generator_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.cutoff_kind not in (
            "plane_wave",
            "interaction_transfer",
            "remote_window",
        ):
            raise ValueError("unknown Kumar cutoff kind")
        if _strict_integer(self.value):
            if int(self.value) <= 0:
                raise ValueError("Kumar cutoff integer must be positive")
            object.__setattr__(self, "value", int(self.value))
        elif not isinstance(self.value, str) or not self.value.strip():
            raise TypeError("Kumar cutoff must be a positive integer or nonempty label")
        _sha256(self.generator_fingerprint, "Kumar cutoff generator fingerprint")
        _sha256(self.source_artifact_sha256, "Kumar cutoff source artifact")
        _authority(self.authority_kind, allow_paper=False)
        if not self.source.strip():
            raise ValueError("Kumar cutoff requires source provenance")


@dataclass(frozen=True, slots=True)
class KumarC3MeshReceipt:
    shape: tuple[int, int]
    reciprocal_basis_fingerprint: str
    mesh_generator_fingerprint: str
    c3_map_fingerprint: str
    orbit_map_fingerprint: str
    reciprocal_carry_fingerprint: str
    quadrature_weight_fingerprint: str
    source_fingerprint: str
    carry_residual: float
    weight_residual: float
    source: str

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(
            not _strict_integer(value) or int(value) <= 0 for value in self.shape
        ):
            raise TypeError("Kumar C3 receipt shape must contain positive integers")
        object.__setattr__(
            self,
            "shape",
            (int(self.shape[0]), int(self.shape[1])),
        )
        for label, value in (
            ("reciprocal basis", self.reciprocal_basis_fingerprint),
            ("mesh generator", self.mesh_generator_fingerprint),
            ("C3 map", self.c3_map_fingerprint),
            ("orbit map", self.orbit_map_fingerprint),
            ("reciprocal carry", self.reciprocal_carry_fingerprint),
            ("quadrature weight", self.quadrature_weight_fingerprint),
            ("source", self.source_fingerprint),
        ):
            _sha256(value, f"Kumar C3 {label}")
        carry = float(self.carry_residual)
        weight = float(self.weight_residual)
        if not np.isfinite(carry) or carry < 0.0 or not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Kumar C3 residuals must be finite and non-negative")
        object.__setattr__(self, "carry_residual", carry)
        object.__setattr__(self, "weight_residual", weight)
        if carry != 0.0 or weight != 0.0:
            raise ValueError("Kumar production C3 receipt requires exact carry/weight closure")
        if not self.source.strip():
            raise ValueError("Kumar C3 receipt requires source provenance")

    @property
    def fingerprint(self) -> str:
        payload = {
            "shape": self.shape,
            "reciprocal_basis_fingerprint": self.reciprocal_basis_fingerprint,
            "mesh_generator_fingerprint": self.mesh_generator_fingerprint,
            "c3_map_fingerprint": self.c3_map_fingerprint,
            "orbit_map_fingerprint": self.orbit_map_fingerprint,
            "reciprocal_carry_fingerprint": self.reciprocal_carry_fingerprint,
            "quadrature_weight_fingerprint": self.quadrature_weight_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "carry_residual": self.carry_residual,
            "weight_residual": self.weight_residual,
            "source": self.source,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KumarMeshAuthority:
    shape: tuple[int, int]
    reciprocal_basis_fingerprint: str
    generator_fingerprint: str
    source_artifact_sha256: str
    c3_receipt: KumarC3MeshReceipt | None
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(
            not _strict_integer(value) or int(value) <= 0 for value in self.shape
        ):
            raise TypeError("Kumar mesh shape must contain two positive integers")
        shape = (int(self.shape[0]), int(self.shape[1]))
        object.__setattr__(self, "shape", shape)
        if shape[0] != shape[1]:
            raise ValueError(
                "Kumar C3-breaking benchmark requires a square-torus candidate; "
                "rectangular meshes are controls only"
            )
        _sha256(self.reciprocal_basis_fingerprint, "Kumar reciprocal basis")
        _sha256(self.generator_fingerprint, "Kumar mesh generator")
        _sha256(self.source_artifact_sha256, "Kumar mesh source artifact")
        if self.c3_receipt is not None:
            if self.c3_receipt.shape != shape:
                raise ValueError("Kumar mesh/C3 receipt shape mismatch")
            if self.c3_receipt.reciprocal_basis_fingerprint != self.reciprocal_basis_fingerprint:
                raise ValueError("Kumar mesh/C3 reciprocal-basis mismatch")
            if self.c3_receipt.mesh_generator_fingerprint != self.generator_fingerprint:
                raise ValueError("Kumar mesh/C3 generator mismatch")
        _authority(self.authority_kind, allow_paper=False)
        if not self.source.strip():
            raise ValueError("Kumar mesh requires source provenance")


@dataclass(frozen=True, slots=True)
class KumarRemotePrescription:
    reference_density: str = "isolated_neutral_graphene_sheets"
    full_density_policy: str = "rho_full_minus_rho_iso"
    frozen_remote_policy: str = "occupied_remote_single_particle_wavefunctions_frozen"
    coulomb_kernel: str = "bare_2pi_e2_over_epsilon_q"
    paper_source: str = "CollectiveModes.tex lines 236-276"

    def __post_init__(self) -> None:
        if self.reference_density != "isolated_neutral_graphene_sheets":
            raise ValueError("Kumar requires isolated neutral-sheet rho_iso")
        if self.full_density_policy != "rho_full_minus_rho_iso":
            raise ValueError("Kumar subtraction applies to the full R/A density closure")
        if self.frozen_remote_policy != (
            "occupied_remote_single_particle_wavefunctions_frozen"
        ):
            raise ValueError("Kumar frozen-remote prescription mismatch")
        if self.coulomb_kernel != "bare_2pi_e2_over_epsilon_q":
            raise ValueError("Kumar printed kernel is bare 2pi e2/(epsilon q)")
        if not self.paper_source.strip():
            raise ValueError("Kumar prescription requires paper provenance")

    @property
    def fingerprint(self) -> str:
        payload = {
            "reference_density": self.reference_density,
            "full_density_policy": self.full_density_policy,
            "frozen_remote_policy": self.frozen_remote_policy,
            "coulomb_kernel": self.coulomb_kernel,
            "paper_source": self.paper_source,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KumarFunctionalAuthority:
    scalar_energy_fingerprint: str
    scf_hamiltonian_derivative_fingerprint: str
    finite_q_hessian_derivative_fingerprint: str
    shared_source_fingerprint: str
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        for label, value in (
            ("scalar energy", self.scalar_energy_fingerprint),
            ("SCF Hamiltonian derivative", self.scf_hamiltonian_derivative_fingerprint),
            ("finite-q Hessian derivative", self.finite_q_hessian_derivative_fingerprint),
            ("shared functional source", self.shared_source_fingerprint),
            ("functional provider", self.provider_fingerprint),
            ("functional source artifact", self.source_artifact_sha256),
        ):
            _sha256(value, f"Kumar {label} fingerprint")
        _authority(self.authority_kind, allow_paper=False)
        if not self.source.strip():
            raise ValueError("Kumar functional binding requires source provenance")


@dataclass(frozen=True, slots=True)
class KumarRemoteClosureAuthority:
    prescription_fingerprint: str
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.prescription_fingerprint != KumarRemotePrescription().fingerprint:
            raise ValueError("Kumar remote implementation/prescription mismatch")
        _sha256(self.prescription_fingerprint, "Kumar remote prescription")
        _sha256(self.provider_fingerprint, "Kumar remote provider fingerprint")
        _sha256(self.source_artifact_sha256, "Kumar remote source artifact")
        _authority(self.authority_kind, allow_paper=False)
        if not self.source.strip():
            raise ValueError("Kumar remote closure requires source provenance")


@dataclass(frozen=True, slots=True)
class KumarPaperScale:
    value_mev: float
    qualifier: Literal["approximately", "lesssim", "below_approximately"]
    source: str

    def __post_init__(self) -> None:
        value = float(self.value_mev)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Kumar paper scale must be finite and non-negative")
        object.__setattr__(self, "value_mev", value)
        if self.qualifier not in (
            "approximately",
            "lesssim",
            "below_approximately",
        ):
            raise ValueError("invalid Kumar paper-scale qualifier")
        if not self.source.strip():
            raise ValueError("Kumar paper scale requires source provenance")


@dataclass(frozen=True, slots=True)
class Kumar2010ExpectedModes:
    intraflavor_excitons: int = 1
    spin_waves: int = 2
    valley_waves: int = 2
    spin_valley_waves: int = 2
    broken_spin_generators: int = 2
    derived_spin_commutator_rank: int = 2
    spin_counting_derivation: str = "saturated_SU2_to_U1_ferromagnet_Watanabe_Murayama"
    quadratic_spin_goldstones: int = 1
    total_modes: int = 7
    low_mode_energy_scale: KumarPaperScale = KumarPaperScale(
        2.0, "below_approximately", "CollectiveModes.tex lines 476-483"
    )
    mode_bandwidth_scale: KumarPaperScale = KumarPaperScale(
        1.0, "lesssim", "CollectiveModes.tex lines 476-483"
    )
    valley_gap_scale: KumarPaperScale = KumarPaperScale(
        0.2, "approximately", "CollectiveModes.tex lines 492-500"
    )
    charge_gap_scale: KumarPaperScale = KumarPaperScale(
        10.0, "approximately", "CollectiveModes.tex lines 476-483"
    )

    def __post_init__(self) -> None:
        total = (
            self.intraflavor_excitons
            + self.spin_waves
            + self.valley_waves
            + self.spin_valley_waves
        )
        if (
            self.intraflavor_excitons,
            self.spin_waves,
            self.valley_waves,
            self.spin_valley_waves,
            self.broken_spin_generators,
            self.derived_spin_commutator_rank,
            self.quadratic_spin_goldstones,
            self.total_modes,
        ) != (1, 2, 2, 2, 2, 2, 1, 7) or total != 7:
            raise ValueError("Kumar paper mode inventory was changed")
        expected_scales = (
            (self.low_mode_energy_scale, 2.0, "below_approximately"),
            (self.mode_bandwidth_scale, 1.0, "lesssim"),
            (self.valley_gap_scale, 0.2, "approximately"),
            (self.charge_gap_scale, 10.0, "approximately"),
        )
        if any(
            scale.value_mev != value or scale.qualifier != qualifier
            for scale, value, qualifier in expected_scales
        ):
            raise ValueError("Kumar approximate paper scales were changed")
        count = count_tdhf_goldstones_from_rank(
            self.broken_spin_generators,
            self.derived_spin_commutator_rank,
        )
        if self.spin_counting_derivation != (
            "saturated_SU2_to_U1_ferromagnet_Watanabe_Murayama"
        ):
            raise ValueError("Kumar spin commutator rank requires derivation provenance")
        if count.type_b_count != self.quadratic_spin_goldstones:
            raise ValueError("Kumar quadratic spin Goldstone count is inconsistent")
        if count.type_a_count != 0:
            raise ValueError("Kumar spin sector should have one type-B branch")


@dataclass(frozen=True, slots=True)
class Kumar2010Spec:
    filling: int = -3
    theta_deg: float = 1.1
    kappa: float = 0.8
    epsilon_inverse: float = 0.06
    coulomb_kernel: str = "bare_2pi_e2_over_epsilon_q"
    active_bands_per_spin_valley: int = 2
    occupied_active_rank_per_k: int = 1
    flavor_order: str = "fully_spin_and_valley_polarized_no_flavor_mixing"
    w1_mev: KumarPinnedValue | None = None
    parent_one_body: KumarParentAuthority | None = None
    plane_wave_cutoff: KumarCutoffAuthority | None = None
    interaction_transfer_cutoff: KumarCutoffAuthority | None = None
    remote_window: KumarCutoffAuthority | None = None
    k_mesh: KumarMeshAuthority | None = None
    q0_background: KumarPinnedValue | None = None
    q_path_coordinates: KumarPinnedValue | None = None
    scf_branch_policy: KumarPinnedValue | None = None
    remote_closure: KumarRemoteClosureAuthority | None = None
    functional_authority: KumarFunctionalAuthority | None = None

    def __post_init__(self) -> None:
        if not _strict_integer(self.filling) or int(self.filling) != -3:
            raise ValueError("Kumar benchmark target is nu=-3")
        object.__setattr__(self, "filling", -3)
        if (
            self.theta_deg != 1.1
            or self.kappa != 0.8
            or self.epsilon_inverse != 0.06
        ):
            raise ValueError("paper-direct Kumar parameters were changed")
        object.__setattr__(self, "theta_deg", float(self.theta_deg))
        object.__setattr__(self, "kappa", float(self.kappa))
        object.__setattr__(self, "epsilon_inverse", float(self.epsilon_inverse))
        if self.coulomb_kernel != "bare_2pi_e2_over_epsilon_q":
            raise ValueError("Kumar does not specify dual-gate screening")
        if (
            not _strict_integer(self.active_bands_per_spin_valley)
            or int(self.active_bands_per_spin_valley) != 2
            or not _strict_integer(self.occupied_active_rank_per_k)
            or int(self.occupied_active_rank_per_k) != 1
        ):
            raise ValueError("Kumar requires one occupied state in eight active bands")
        object.__setattr__(self, "active_bands_per_spin_valley", 2)
        object.__setattr__(self, "occupied_active_rank_per_k", 1)
        if self.flavor_order != "fully_spin_and_valley_polarized_no_flavor_mixing":
            raise ValueError("Kumar source flavor order mismatch")
        for expected_kind, cutoff in (
            ("plane_wave", self.plane_wave_cutoff),
            ("interaction_transfer", self.interaction_transfer_cutoff),
            ("remote_window", self.remote_window),
        ):
            if cutoff is not None and cutoff.cutoff_kind != expected_kind:
                raise ValueError(f"Kumar {expected_kind} cutoff has the wrong kind")
        if self.k_mesh is not None and self.k_mesh.c3_receipt is None:
            raise ValueError(
                "Kumar production mesh is only a square candidate until C3 closure is certified"
            )
        if self.w1_mev is not None and self.w1_mev.authority_kind == "paper_explicit":
            raise ValueError("absolute w1 is not stated in the Kumar paper")
        for label, value in (
            ("q0_background", self.q0_background),
            ("q_path_coordinates", self.q_path_coordinates),
            ("scf_branch_policy", self.scf_branch_policy),
        ):
            if value is not None and value.authority_kind == "paper_explicit":
                raise ValueError(f"{label} is not numerically stated in the Kumar paper")

    @classmethod
    def paper_target(cls) -> "Kumar2010Spec":
        return cls()

    @property
    def active_dimension_per_k(self) -> int:
        return 4 * self.active_bands_per_spin_valley

    @property
    def expected_modes(self) -> Kumar2010ExpectedModes:
        return Kumar2010ExpectedModes()

    @property
    def unresolved_authorities(self) -> tuple[str, ...]:
        return tuple(
            label
            for label in (
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
            if getattr(self, label) is None
        )

    @property
    def metadata_resolved(self) -> bool:
        return not self.unresolved_authorities

    @property
    def paper_direct_claim_allowed(self) -> bool:
        return False

    def require_metadata_resolved(self) -> None:
        if self.unresolved_authorities:
            raise RuntimeError(
                "Kumar 2010 executable authority is incomplete: "
                + ", ".join(self.unresolved_authorities)
            )

    @property
    def fingerprint(self) -> str:
        def encode(value: object) -> object:
            if value is None:
                return None
            if isinstance(value, KumarPinnedValue):
                return {
                    "value": value.value,
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            if isinstance(value, KumarParentAuthority):
                return {
                    "model_convention": value.model_convention,
                    "vf_mev_b0": value.vf_mev_b0,
                    "provider_fingerprint": value.provider_fingerprint,
                    "source_artifact_sha256": value.source_artifact_sha256,
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            if isinstance(value, KumarCutoffAuthority):
                return {
                    "cutoff_kind": value.cutoff_kind,
                    "value": value.value,
                    "generator_fingerprint": value.generator_fingerprint,
                    "source_artifact_sha256": value.source_artifact_sha256,
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            if isinstance(value, KumarMeshAuthority):
                return {
                    "shape": value.shape,
                    "reciprocal_basis_fingerprint": value.reciprocal_basis_fingerprint,
                    "generator_fingerprint": value.generator_fingerprint,
                    "source_artifact_sha256": value.source_artifact_sha256,
                    "c3_receipt_fingerprint": (
                        None if value.c3_receipt is None else value.c3_receipt.fingerprint
                    ),
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            if isinstance(value, KumarRemoteClosureAuthority):
                return {
                    "prescription_fingerprint": value.prescription_fingerprint,
                    "provider_fingerprint": value.provider_fingerprint,
                    "source_artifact_sha256": value.source_artifact_sha256,
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            if isinstance(value, KumarFunctionalAuthority):
                return {
                    "scalar_energy_fingerprint": value.scalar_energy_fingerprint,
                    "scf_hamiltonian_derivative_fingerprint": value.scf_hamiltonian_derivative_fingerprint,
                    "finite_q_hessian_derivative_fingerprint": value.finite_q_hessian_derivative_fingerprint,
                    "shared_source_fingerprint": value.shared_source_fingerprint,
                    "provider_fingerprint": value.provider_fingerprint,
                    "source_artifact_sha256": value.source_artifact_sha256,
                    "authority_kind": value.authority_kind,
                    "source": value.source,
                }
            raise TypeError("unsupported Kumar authority value")

        payload = {
            "paper": KUMAR_2010_ARXIV,
            "pdf_sha256": KUMAR_2010_PDF_SHA256,
            "source_archive_sha256": KUMAR_2010_SOURCE_ARCHIVE_SHA256,
            "filling": self.filling,
            "theta_deg": self.theta_deg,
            "kappa": self.kappa,
            "epsilon_inverse": self.epsilon_inverse,
            "coulomb_kernel": self.coulomb_kernel,
            "active_bands_per_spin_valley": self.active_bands_per_spin_valley,
            "occupied_active_rank_per_k": self.occupied_active_rank_per_k,
            "flavor_order": self.flavor_order,
            "w1_mev": encode(self.w1_mev),
            "parent_one_body": encode(self.parent_one_body),
            "plane_wave_cutoff": encode(self.plane_wave_cutoff),
            "interaction_transfer_cutoff": encode(
                self.interaction_transfer_cutoff
            ),
            "remote_window": encode(self.remote_window),
            "k_mesh": encode(self.k_mesh),
            "q0_background": encode(self.q0_background),
            "q_path_coordinates": encode(self.q_path_coordinates),
            "scf_branch_policy": encode(self.scf_branch_policy),
            "remote_closure": encode(self.remote_closure),
            "functional_authority": encode(self.functional_authority),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@runtime_checkable
class Kumar2010ProviderReceiptProtocol(Protocol):
    fingerprint: str
    spec_fingerprint: str
    source_commit: str
    scf_branch_policy_fingerprint: str
    parent_provider_fingerprint: str
    parent_source_artifact_sha256: str
    plane_wave_generator_fingerprint: str
    plane_wave_source_artifact_sha256: str
    interaction_transfer_generator_fingerprint: str
    interaction_transfer_source_artifact_sha256: str
    remote_window_generator_fingerprint: str
    remote_window_source_artifact_sha256: str
    mesh_generator_fingerprint: str
    mesh_source_artifact_sha256: str
    c3_receipt_fingerprint: str
    remote_provider_fingerprint: str
    remote_source_artifact_sha256: str
    scalar_energy_fingerprint: str
    scf_hamiltonian_derivative_fingerprint: str
    finite_q_hessian_derivative_fingerprint: str
    shared_functional_source_fingerprint: str
    functional_provider_fingerprint: str
    functional_source_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class Kumar2010ProviderBinding:
    """Bind metadata receipts only; this is not execution readiness."""

    spec: Kumar2010Spec
    provider: Kumar2010ProviderReceiptProtocol

    def __post_init__(self) -> None:
        self.spec.require_metadata_resolved()
        if not isinstance(self.provider, Kumar2010ProviderReceiptProtocol):
            raise TypeError("Kumar provider binding requires typed receipts")
        _sha256(self.provider.fingerprint, "Kumar provider receipt")
        if len(self.provider.source_commit) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.provider.source_commit
        ):
            raise ValueError("Kumar provider receipt must bind a source commit")
        if self.provider.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("Kumar provider/spec fingerprint mismatch")
        assert self.spec.scf_branch_policy is not None
        assert self.spec.parent_one_body is not None
        assert self.spec.plane_wave_cutoff is not None
        assert self.spec.interaction_transfer_cutoff is not None
        assert self.spec.remote_window is not None
        assert self.spec.k_mesh is not None
        assert self.spec.k_mesh.c3_receipt is not None
        assert self.spec.remote_closure is not None
        assert self.spec.functional_authority is not None
        expected = (
            ("SCF policy", self.provider.scf_branch_policy_fingerprint, self.spec.scf_branch_policy.fingerprint),
            ("parent", self.provider.parent_provider_fingerprint, self.spec.parent_one_body.provider_fingerprint),
            ("parent source", self.provider.parent_source_artifact_sha256, self.spec.parent_one_body.source_artifact_sha256),
            ("plane-wave", self.provider.plane_wave_generator_fingerprint, self.spec.plane_wave_cutoff.generator_fingerprint),
            ("plane-wave source", self.provider.plane_wave_source_artifact_sha256, self.spec.plane_wave_cutoff.source_artifact_sha256),
            ("interaction-transfer", self.provider.interaction_transfer_generator_fingerprint, self.spec.interaction_transfer_cutoff.generator_fingerprint),
            ("interaction-transfer source", self.provider.interaction_transfer_source_artifact_sha256, self.spec.interaction_transfer_cutoff.source_artifact_sha256),
            ("remote-window", self.provider.remote_window_generator_fingerprint, self.spec.remote_window.generator_fingerprint),
            ("remote-window source", self.provider.remote_window_source_artifact_sha256, self.spec.remote_window.source_artifact_sha256),
            ("mesh", self.provider.mesh_generator_fingerprint, self.spec.k_mesh.generator_fingerprint),
            ("mesh source", self.provider.mesh_source_artifact_sha256, self.spec.k_mesh.source_artifact_sha256),
            ("C3 receipt", self.provider.c3_receipt_fingerprint, self.spec.k_mesh.c3_receipt.fingerprint),
            ("remote", self.provider.remote_provider_fingerprint, self.spec.remote_closure.provider_fingerprint),
            ("remote source", self.provider.remote_source_artifact_sha256, self.spec.remote_closure.source_artifact_sha256),
            ("scalar energy", self.provider.scalar_energy_fingerprint, self.spec.functional_authority.scalar_energy_fingerprint),
            ("SCF Hamiltonian derivative", self.provider.scf_hamiltonian_derivative_fingerprint, self.spec.functional_authority.scf_hamiltonian_derivative_fingerprint),
            ("finite-q Hessian derivative", self.provider.finite_q_hessian_derivative_fingerprint, self.spec.functional_authority.finite_q_hessian_derivative_fingerprint),
            ("shared functional source", self.provider.shared_functional_source_fingerprint, self.spec.functional_authority.shared_source_fingerprint),
            ("functional provider", self.provider.functional_provider_fingerprint, self.spec.functional_authority.provider_fingerprint),
            ("functional source artifact", self.provider.functional_source_artifact_sha256, self.spec.functional_authority.source_artifact_sha256),
        )
        for label, actual, required in expected:
            _sha256(actual, f"Kumar provider {label}")
            if actual != required:
                raise ValueError(f"Kumar provider/{label} fingerprint mismatch")

    @property
    def provider_bound(self) -> bool:
        return True


__all__ = [
    "KUMAR_2010_ARXIV",
    "KUMAR_2010_PDF_SHA256",
    "KUMAR_2010_RESPONSE_SCOPE",
    "KUMAR_2010_SOURCE_ARCHIVE_SHA256",
    "Kumar2010ProviderBinding",
    "Kumar2010ProviderReceiptProtocol",
    "Kumar2010ExpectedModes",
    "Kumar2010Spec",
    "KumarC3MeshReceipt",
    "KumarCutoffAuthority",
    "KumarFunctionalAuthority",
    "KumarMeshAuthority",
    "KumarPaperScale",
    "KumarParentAuthority",
    "KumarPinnedValue",
    "KumarRemoteClosureAuthority",
    "KumarRemotePrescription",
]
