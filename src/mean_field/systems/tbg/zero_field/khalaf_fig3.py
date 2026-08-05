"""Fail-closed authority contract for Khalaf et al. Fig. 3.

This module does not implement the expensive six-band HF/TDHF calculation.
It records what arXiv:2009.14827v2 states, separates cited-reference and
reproduction choices from paper-direct inputs, and provides the rectangular
raw-q torus needed by the published meshes.  A production runner must consume
a fully resolved instance instead of silently inheriting Kwan-companion or
central-two-band defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import json
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from mean_field.core.hf.tdhf_goldstone import count_tdhf_goldstones_from_rank
from mean_field.core.hf.tdhf_signed import TDHFSignedQ, classify_tdhf_signed_q

from .model import TBGZeroFieldTorusMesh

KHALAF_FIG3_ARXIV = "2009.14827v2"
KHALAF_FIG3_PDF_SHA256 = (
    "991c348090db78d25d2d06e69858f9df58ff35c411bc5a199478b70326280730"
)
KHALAF_REFERENCE_SUBTRACTION_ARXIV = "1911.02045"
KHALAF_FIG3_RESPONSE_SCOPE = "khalaf_fig3_six_band_kivc_tdhf_v1"

AuthorityKind = Literal[
    "paper_explicit",
    "cited_reference_explicit",
    "author_source_explicit",
    "reproduction_choice",
]


def _is_strict_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


@dataclass(frozen=True, slots=True)
class KhalafPinnedValue:
    """One executable value with an explicit level and source of authority."""

    value: int | float | str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise TypeError("Khalaf pinned value cannot be boolean")
        if _is_strict_integer(self.value):
            object.__setattr__(self, "value", int(self.value))
        elif isinstance(self.value, (float, np.floating)):
            object.__setattr__(self, "value", float(self.value))
        if self.authority_kind not in (
            "paper_explicit",
            "cited_reference_explicit",
            "author_source_explicit",
            "reproduction_choice",
        ):
            raise ValueError("invalid Khalaf authority kind")
        if not str(self.source).strip():
            raise ValueError("Khalaf pinned value requires a source")
        if isinstance(self.value, float) and not np.isfinite(self.value):
            raise ValueError("Khalaf pinned value must be finite")

    @property
    def is_paper_direct(self) -> bool:
        return self.authority_kind == "paper_explicit"


@dataclass(frozen=True, slots=True)
class KhalafParentOneBodyAuthority:
    model_convention: str
    t0_ev: float
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.model_convention != "full_monolayer_graphene_t0_parent":
            raise ValueError("Khalaf parent must pin the full-monolayer convention")
        if not np.isfinite(self.t0_ev) or self.t0_ev <= 0.0:
            raise ValueError("Khalaf parent t0 must be finite and positive")
        object.__setattr__(self, "t0_ev", float(self.t0_ev))
        _validate_sha256(self.provider_fingerprint, "parent provider fingerprint")
        _validate_sha256(self.source_artifact_sha256, "parent source artifact")
        if self.authority_kind not in (
            "cited_reference_explicit",
            "author_source_explicit",
            "reproduction_choice",
        ):
            raise ValueError("Khalaf parent convention is not Fig.3-caption direct")
        if not self.source.strip():
            raise ValueError("Khalaf parent convention requires source provenance")


@dataclass(frozen=True, slots=True)
class KhalafCutoffAuthority:
    cutoff_kind: Literal["plane_wave", "interaction_transfer"]
    shell: int
    generator_fingerprint: str
    source_artifact_sha256: str
    convergence_family: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.cutoff_kind not in ("plane_wave", "interaction_transfer"):
            raise ValueError("unknown Khalaf cutoff kind")
        if not _is_strict_integer(self.shell) or self.shell < 1:
            raise ValueError("Khalaf cutoff shell must be a positive integer")
        object.__setattr__(self, "shell", int(self.shell))
        _validate_sha256(self.generator_fingerprint, "cutoff generator fingerprint")
        _validate_sha256(self.source_artifact_sha256, "cutoff source artifact")
        if not self.convergence_family.strip() or not self.source.strip():
            raise ValueError("Khalaf cutoff requires convergence/source provenance")
        if self.authority_kind not in (
            "cited_reference_explicit",
            "author_source_explicit",
            "reproduction_choice",
        ):
            raise ValueError(
                "Fig.3 does not directly specify parent/transfer cutoffs"
            )


@dataclass(frozen=True, slots=True)
class KhalafRemoteClosureAuthority:
    """Executable Ref. 27 remote/reference/subtraction closure.

    The cited prescription uses the neutral density matrix of two decoupled
    graphene layers as ``P0``, replaces ``h_BM`` by
    ``h_BM - 1/2 H_MF^C[P0]``, and includes filled projected-out valence bands
    together with their matching reference subtraction.  The provider digest
    binds an implementation of that complete operation; prose alone is not
    executable authority.
    """

    reference_density: str
    subtraction_prefactor: float
    projected_out_valence_policy: str
    provider_fingerprint: str
    source_artifact_sha256: str
    authority_kind: AuthorityKind
    source: str

    def __post_init__(self) -> None:
        if self.reference_density != "decoupled_graphene_layers_neutrality":
            raise ValueError("Khalaf closure requires the Ref.27 decoupled-layer P0")
        if self.subtraction_prefactor != 0.5:
            raise ValueError("Khalaf closure requires h=h_BM-1/2 H_MF[P0]")
        object.__setattr__(
            self, "subtraction_prefactor", float(self.subtraction_prefactor)
        )
        if self.projected_out_valence_policy != (
            "filled_remote_valence_plus_matching_P0_subtraction"
        ):
            raise ValueError("Khalaf projected-out valence policy mismatch")
        _validate_sha256(
            self.provider_fingerprint, "remote closure provider fingerprint"
        )
        _validate_sha256(
            self.source_artifact_sha256, "remote closure source artifact"
        )
        if self.authority_kind not in (
            "cited_reference_explicit",
            "author_source_explicit",
            "reproduction_choice",
        ):
            raise ValueError("remote closure cannot be claimed from the Fig.3 caption")
        if not self.source.strip():
            raise ValueError("remote closure requires source provenance")


@dataclass(frozen=True, slots=True)
class KhalafFig3ExpectedModes:
    total_soft_modes: int
    static_ward_directions: int
    symplectic_rank: int
    linear_goldstones: int
    quadratic_goldstones: int
    gapped_degeneracies: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.linear_goldstones + self.quadratic_goldstones + sum(
            self.gapped_degeneracies
        ) != self.total_soft_modes:
            raise ValueError("Khalaf mode-count decomposition does not close")
        count = count_tdhf_goldstones_from_rank(
            self.static_ward_directions,
            self.symplectic_rank,
        )
        # Khalaf's regular-gradient sigma model maps theorem type-A/type-B
        # counts to its reported linear/quadratic branches.
        if self.linear_goldstones != count.type_a_count:
            raise ValueError("Khalaf linear count is inconsistent with rank rho")
        if self.quadratic_goldstones != count.type_b_count:
            raise ValueError("Khalaf quadratic count is inconsistent with rank rho")


@dataclass(frozen=True, slots=True)
class KhalafFig3Spec:
    """Paper-direct fields plus unresolved executable authority fields."""

    filling: int
    mesh_shape: tuple[int, int]
    theta_deg: float = 1.08
    kappa: float = 0.75
    epsilon_r: float = 12.5
    gate_distance_nm: float = 40.0
    retained_remote_valence_per_spin_valley: int = 2
    retained_flat_per_spin_valley: int = 2
    retained_remote_conduction_per_spin_valley: int = 2
    strain_percent: float = 0.0
    w1_mev: KhalafPinnedValue | None = None
    parent_one_body: KhalafParentOneBodyAuthority | None = None
    plane_wave_cutoff: KhalafCutoffAuthority | None = None
    interaction_transfer_cutoff: KhalafCutoffAuthority | None = None
    remote_closure: KhalafRemoteClosureAuthority | None = None

    def __post_init__(self) -> None:
        expected_mesh = {0: (18, 18), -2: (18, 12)}
        if not _is_strict_integer(self.filling) or self.filling not in expected_mesh:
            raise ValueError("Khalaf Fig.3 supports only nu=0 and nu=-2")
        if len(self.mesh_shape) != 2 or any(
            not _is_strict_integer(value) for value in self.mesh_shape
        ):
            raise TypeError("Khalaf mesh dimensions must be strict integers")
        filling = int(self.filling)
        mesh_shape = (int(self.mesh_shape[0]), int(self.mesh_shape[1]))
        object.__setattr__(self, "filling", filling)
        object.__setattr__(self, "mesh_shape", mesh_shape)
        if mesh_shape != expected_mesh[filling]:
            raise ValueError(
                f"published Khalaf Fig.3 mesh at nu={filling} is "
                f"{expected_mesh[filling]}, not {mesh_shape}"
            )
        if (
            self.theta_deg != 1.08
            or self.kappa != 0.75
            or self.epsilon_r != 12.5
            or self.gate_distance_nm != 40.0
        ):
            raise ValueError("paper-direct Khalaf Fig.3 parameters were changed")
        for name in ("theta_deg", "kappa", "epsilon_r", "gate_distance_nm"):
            object.__setattr__(self, name, float(getattr(self, name)))
        retained = (
            self.retained_remote_valence_per_spin_valley,
            self.retained_flat_per_spin_valley,
            self.retained_remote_conduction_per_spin_valley,
        )
        if any(not _is_strict_integer(value) for value in retained):
            raise TypeError("Khalaf retained-band counts must be strict integers")
        retained = tuple(int(value) for value in retained)
        object.__setattr__(self, "retained_remote_valence_per_spin_valley", retained[0])
        object.__setattr__(self, "retained_flat_per_spin_valley", retained[1])
        object.__setattr__(self, "retained_remote_conduction_per_spin_valley", retained[2])
        if retained != (2, 2, 2):
            raise ValueError("Khalaf Fig.3 requires six retained bands per spin/valley")
        if self.strain_percent != 0.0:
            raise ValueError("Khalaf Fig.3 preflight does not permit strain")
        object.__setattr__(self, "strain_percent", float(self.strain_percent))
        if self.w1_mev is not None:
            w1 = float(self.w1_mev.value)
            if not np.isfinite(w1) or w1 <= 0.0:
                raise ValueError("w1 must be finite and positive")
        if self.w1_mev is not None and self.w1_mev.authority_kind == "paper_explicit":
            raise ValueError(
                "w1_mev is not directly specified by the Khalaf Fig.3 paper"
            )
        if (
            self.plane_wave_cutoff is not None
            and self.plane_wave_cutoff.cutoff_kind != "plane_wave"
        ):
            raise ValueError("plane-wave cutoff binding has the wrong kind")
        if (
            self.interaction_transfer_cutoff is not None
            and self.interaction_transfer_cutoff.cutoff_kind
            != "interaction_transfer"
        ):
            raise ValueError("interaction-transfer cutoff binding has the wrong kind")

    @classmethod
    def paper_target(cls, filling: int) -> "KhalafFig3Spec":
        meshes = {0: (18, 18), -2: (18, 12)}
        if filling not in meshes:
            raise ValueError("Khalaf Fig.3 supports only nu=0 and nu=-2")
        return cls(filling=filling, mesh_shape=meshes[filling])

    @property
    def retained_bands_per_spin_valley(self) -> int:
        return (
            self.retained_remote_valence_per_spin_valley
            + self.retained_flat_per_spin_valley
            + self.retained_remote_conduction_per_spin_valley
        )

    @property
    def retained_dimension_per_k(self) -> int:
        return 4 * self.retained_bands_per_spin_valley

    @property
    def occupied_remote_valence_per_k(self) -> int:
        return 4 * self.retained_remote_valence_per_spin_valley

    @property
    def occupied_flat_per_k(self) -> int:
        return 4 + self.filling

    @property
    def occupied_rank_per_k(self) -> int:
        return self.occupied_remote_valence_per_k + self.occupied_flat_per_k

    @property
    def expected_hf_gap_mev(self) -> float:
        return 25.0 if self.filling == 0 else 14.0

    @property
    def resolved_w0_mev(self) -> float | None:
        if self.w1_mev is None:
            return None
        return self.kappa * float(self.w1_mev.value)

    @property
    def expected_modes(self) -> KhalafFig3ExpectedModes:
        if self.filling == 0:
            return KhalafFig3ExpectedModes(
                total_soft_modes=16,
                static_ward_directions=4,
                symplectic_rank=0,
                linear_goldstones=4,
                quadratic_goldstones=0,
                gapped_degeneracies=(4, 4, 4),
            )
        return KhalafFig3ExpectedModes(
            total_soft_modes=12,
            static_ward_directions=5,
            symplectic_rank=4,
            linear_goldstones=1,
            quadratic_goldstones=2,
            gapped_degeneracies=(2, 1, 4, 2),
        )

    @property
    def unresolved_authorities(self) -> tuple[str, ...]:
        missing: list[str] = []
        for label in (
            "w1_mev",
            "parent_one_body",
            "plane_wave_cutoff",
            "interaction_transfer_cutoff",
            "remote_closure",
        ):
            if getattr(self, label) is None:
                missing.append(label)
        return tuple(missing)

    @property
    def metadata_resolved(self) -> bool:
        return not self.unresolved_authorities

    @property
    def paper_direct_claim_allowed(self) -> bool:
        # The Fig.3 paper does not directly pin all five executable fields.
        return False

    def require_metadata_resolved(self) -> None:
        if self.unresolved_authorities:
            raise RuntimeError(
                "Khalaf Fig.3 authority closure is incomplete: "
                + ", ".join(self.unresolved_authorities)
            )

    @property
    def fingerprint(self) -> str:
        def scalar(value: KhalafPinnedValue | None) -> object:
            if value is None:
                return None
            return {
                "value": value.value,
                "authority_kind": value.authority_kind,
                "source": value.source,
            }

        parent: object = None
        if self.parent_one_body is not None:
            parent = {
                "model_convention": self.parent_one_body.model_convention,
                "t0_ev": self.parent_one_body.t0_ev,
                "provider_fingerprint": self.parent_one_body.provider_fingerprint,
                "source_artifact_sha256": self.parent_one_body.source_artifact_sha256,
                "authority_kind": self.parent_one_body.authority_kind,
                "source": self.parent_one_body.source,
            }

        def cutoff(value: KhalafCutoffAuthority | None) -> object:
            if value is None:
                return None
            return {
                "cutoff_kind": value.cutoff_kind,
                "shell": value.shell,
                "generator_fingerprint": value.generator_fingerprint,
                "source_artifact_sha256": value.source_artifact_sha256,
                "convergence_family": value.convergence_family,
                "authority_kind": value.authority_kind,
                "source": value.source,
            }

        remote: object = None
        if self.remote_closure is not None:
            remote = {
                "reference_density": self.remote_closure.reference_density,
                "subtraction_prefactor": self.remote_closure.subtraction_prefactor,
                "projected_out_valence_policy": self.remote_closure.projected_out_valence_policy,
                "provider_fingerprint": self.remote_closure.provider_fingerprint,
                "source_artifact_sha256": self.remote_closure.source_artifact_sha256,
                "authority_kind": self.remote_closure.authority_kind,
                "source": self.remote_closure.source,
            }
        payload = {
            "paper": KHALAF_FIG3_ARXIV,
            "pdf_sha256": KHALAF_FIG3_PDF_SHA256,
            "filling": self.filling,
            "mesh_shape": self.mesh_shape,
            "theta_deg": self.theta_deg,
            "kappa": self.kappa,
            "epsilon_r": self.epsilon_r,
            "gate_distance_nm": self.gate_distance_nm,
            "retained_bands_per_spin_valley": self.retained_bands_per_spin_valley,
            "occupied_rank_per_k": self.occupied_rank_per_k,
            "strain_percent": self.strain_percent,
            "w1_mev": scalar(self.w1_mev),
            "parent_one_body": parent,
            "plane_wave_cutoff": cutoff(self.plane_wave_cutoff),
            "interaction_transfer_cutoff": cutoff(
                self.interaction_transfer_cutoff
            ),
            "remote_closure": remote,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@runtime_checkable
class KhalafFig3ExecutableProviderProtocol(Protocol):
    fingerprint: str
    spec_fingerprint: str
    source_commit: str
    parent_provider_fingerprint: str
    parent_source_artifact_sha256: str
    plane_wave_generator_fingerprint: str
    plane_wave_source_artifact_sha256: str
    interaction_transfer_generator_fingerprint: str
    interaction_transfer_source_artifact_sha256: str
    remote_closure_fingerprint: str
    remote_source_artifact_sha256: str

    def build_parent_one_body(self) -> Any: ...

    def build_interaction_kernel(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class KhalafFig3ExecutableBinding:
    spec: KhalafFig3Spec
    provider: KhalafFig3ExecutableProviderProtocol

    def __post_init__(self) -> None:
        self.spec.require_metadata_resolved()
        if not isinstance(self.provider, KhalafFig3ExecutableProviderProtocol):
            raise TypeError("Khalaf executable binding requires a typed provider")
        _validate_sha256(self.provider.fingerprint, "executable provider fingerprint")
        if len(self.provider.source_commit) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.provider.source_commit
        ):
            raise ValueError("Khalaf executable provider must bind a source commit")
        if self.provider.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("Khalaf provider/spec fingerprint mismatch")
        assert self.spec.parent_one_body is not None
        assert self.spec.plane_wave_cutoff is not None
        assert self.spec.interaction_transfer_cutoff is not None
        assert self.spec.remote_closure is not None
        comparisons = (
            (
                "parent provider",
                self.provider.parent_provider_fingerprint,
                self.spec.parent_one_body.provider_fingerprint,
            ),
            (
                "parent source artifact",
                self.provider.parent_source_artifact_sha256,
                self.spec.parent_one_body.source_artifact_sha256,
            ),
            (
                "plane-wave generator",
                self.provider.plane_wave_generator_fingerprint,
                self.spec.plane_wave_cutoff.generator_fingerprint,
            ),
            (
                "plane-wave source artifact",
                self.provider.plane_wave_source_artifact_sha256,
                self.spec.plane_wave_cutoff.source_artifact_sha256,
            ),
            (
                "interaction-transfer generator",
                self.provider.interaction_transfer_generator_fingerprint,
                self.spec.interaction_transfer_cutoff.generator_fingerprint,
            ),
            (
                "interaction-transfer source artifact",
                self.provider.interaction_transfer_source_artifact_sha256,
                self.spec.interaction_transfer_cutoff.source_artifact_sha256,
            ),
            (
                "remote closure",
                self.provider.remote_closure_fingerprint,
                self.spec.remote_closure.provider_fingerprint,
            ),
            (
                "remote source artifact",
                self.provider.remote_source_artifact_sha256,
                self.spec.remote_closure.source_artifact_sha256,
            ),
        )
        for label, actual, expected in comparisons:
            _validate_sha256(actual, f"provider {label}")
            if actual != expected:
                raise ValueError(f"Khalaf provider/{label} fingerprint mismatch")

    @property
    def executable_ready(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class KhalafRectangularMomentumLabel:
    raw: tuple[int, int]
    canonical: tuple[int, int]
    reciprocal_carry: tuple[int, int]
    torus_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("raw", "canonical", "reciprocal_carry"):
            values = getattr(self, name)
            if len(values) != 2 or any(
                not _is_strict_integer(value) for value in values
            ):
                raise TypeError(f"{name} momentum data must contain strict integers")
            object.__setattr__(self, name, (int(values[0]), int(values[1])))
        _validate_sha256(self.torus_fingerprint, "momentum-label torus fingerprint")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "raw": self.raw,
                "canonical": self.canonical,
                "reciprocal_carry": self.reciprocal_carry,
                "torus_fingerprint": self.torus_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class KhalafRectangularTorus:
    shape: tuple[int, int]
    filling: int
    reciprocal_basis_fingerprint: str
    index_order: str = "first_coordinate_fastest_fortran_v1"

    def __post_init__(self) -> None:
        nx, ny = self.shape
        if not _is_strict_integer(nx) or not _is_strict_integer(ny):
            raise TypeError("rectangular torus dimensions must be integers")
        if nx < 2 or ny < 2:
            raise ValueError("rectangular torus dimensions must be integers >= 2")
        if not _is_strict_integer(self.filling):
            raise TypeError("Khalaf torus filling must be a strict integer")
        shape = (int(nx), int(ny))
        filling = int(self.filling)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "filling", filling)
        if filling not in (0, -2):
            raise ValueError("Khalaf rectangular torus requires nu=0 or nu=-2")
        if shape != KhalafFig3Spec.paper_target(filling).mesh_shape:
            raise ValueError("Khalaf torus shape/filling mismatch")
        _validate_sha256(
            self.reciprocal_basis_fingerprint, "reciprocal basis fingerprint"
        )
        if self.index_order != "first_coordinate_fastest_fortran_v1":
            raise ValueError("Khalaf torus must match production TBG Fortran order")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "shape": self.shape,
                "filling": self.filling,
                "reciprocal_basis_fingerprint": self.reciprocal_basis_fingerprint,
                "index_order": self.index_order,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def for_fig3(
        cls,
        filling: int,
        *,
        reciprocal_basis_fingerprint: str,
    ) -> "KhalafRectangularTorus":
        return cls(
            KhalafFig3Spec.paper_target(filling).mesh_shape,
            filling,
            reciprocal_basis_fingerprint,
        )

    @classmethod
    def from_bm_mesh(
        cls,
        filling: int,
        mesh: TBGZeroFieldTorusMesh,
    ) -> "KhalafRectangularTorus":
        if not isinstance(mesh, TBGZeroFieldTorusMesh):
            raise TypeError("Khalaf torus bridge requires a typed TBG BM mesh")
        target = KhalafFig3Spec.paper_target(filling)
        if mesh.mesh_shape != target.mesh_shape:
            raise ValueError("typed BM mesh does not match the Khalaf Fig.3 target")
        return cls(
            target.mesh_shape,
            filling,
            mesh.reciprocal_basis_fingerprint,
        )

    def label(self, raw: tuple[int, int]) -> KhalafRectangularMomentumLabel:
        if len(raw) != 2 or any(
            not _is_strict_integer(value) for value in raw
        ):
            raise TypeError("raw momentum must contain two integers")
        nx, ny = self.shape
        rx, ry = int(raw[0]), int(raw[1])
        canonical = (rx % nx, ry % ny)
        carry = ((rx - canonical[0]) // nx, (ry - canonical[1]) // ny)
        return KhalafRectangularMomentumLabel(
            (rx, ry), canonical, carry, self.fingerprint
        )

    def signed_pair(
        self, raw: tuple[int, int]
    ) -> tuple[KhalafRectangularMomentumLabel, KhalafRectangularMomentumLabel]:
        plus = self.label(raw)
        minus = self.label((-int(raw[0]), -int(raw[1])))
        return plus, minus

    def signed_q_kind(self, raw: tuple[int, int]) -> TDHFSignedQ:
        plus, minus = self.signed_pair(raw)
        return classify_tdhf_signed_q(
            plus_raw=plus.raw,
            minus_raw=minus.raw,
            plus_canonical=plus.canonical,
            minus_canonical=minus.canonical,
            provenance=f"khalaf_rectangular_torus={self.fingerprint}",
        )

    def flatten(self, canonical: tuple[int, int]) -> int:
        nx, ny = self.shape
        if len(canonical) != 2 or any(
            not _is_strict_integer(value) for value in canonical
        ):
            raise TypeError("canonical momentum must contain two integers")
        cx, cy = int(canonical[0]), int(canonical[1])
        if not (0 <= cx < nx and 0 <= cy < ny):
            raise ValueError("canonical momentum lies outside the torus")
        return cx + nx * cy

    def unflatten(self, index: int) -> tuple[int, int]:
        nx, ny = self.shape
        if not _is_strict_integer(index) or not 0 <= index < nx * ny:
            raise ValueError("flattened momentum index is out of range")
        return int(index) % nx, int(index) // nx

    def m_gamma_m_x_labels(self) -> tuple[KhalafRectangularMomentumLabel, ...]:
        nx, _ = self.shape
        if nx % 2:
            raise ValueError("M-Gamma-M endpoint aliases require even Nx")
        return tuple(self.label((qx, 0)) for qx in range(-nx // 2, nx // 2 + 1))


__all__ = [
    "KHALAF_FIG3_ARXIV",
    "KHALAF_FIG3_PDF_SHA256",
    "KHALAF_FIG3_RESPONSE_SCOPE",
    "KHALAF_REFERENCE_SUBTRACTION_ARXIV",
    "KhalafCutoffAuthority",
    "KhalafFig3ExecutableBinding",
    "KhalafFig3ExecutableProviderProtocol",
    "KhalafFig3ExpectedModes",
    "KhalafFig3Spec",
    "KhalafParentOneBodyAuthority",
    "KhalafPinnedValue",
    "KhalafRectangularMomentumLabel",
    "KhalafRectangularTorus",
    "KhalafRemoteClosureAuthority",
]
