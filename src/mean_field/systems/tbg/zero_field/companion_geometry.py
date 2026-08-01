"""Companion-faithful finite plane-wave and interaction-cutoff geometry.

This bookkeeping-only module exposes the companion rectangular label orders,
circular cutoffs, selected basis indices, and zero-filled tunnelling edges
without constructing Hamiltonian or interaction matrices.  The zero-field TBG
package front door exports these records, but the production solver/HF/TDHF
paths do not consume them.
"""

from __future__ import annotations


from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Complex, Real
from typing import Final, Literal


TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY: Final[str] = "reference/TBG-HF"
TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT: Final[str] = (
    "0d2a3d742aa901fa45ce46690c1385887165f58c"
)
TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE: Final[str] = "singleParticle.py"
TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256: Final[str] = (
    "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
)
TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION: Final[str] = "gen_coeff"
TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES: Final[str] = "132-175"
TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION: Final[str] = (
    "gen_moire_hamiltonian"
)
TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES: Final[str] = "101-109"
TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION: Final[str] = "gen_interaction"
TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES: Final[str] = "220-255"

TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA = (
    "mean_field.tbg.zero_field.companion_plane_wave_geometry"
)
TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA = (
    "mean_field.tbg.zero_field.companion_interaction_geometry"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION = 1

TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_LABEL_ORDER = (
    "nested_g1_then_g2_then_layer1_species_0_1_then_layer2_species_2_3"
)
TBG_ZERO_FIELD_COMPANION_FLATTENED_INDEX_CONVENTION = (
    "4*((g1+Ng1)*(2*Ng2)+(g2+Ng2))+species"
)
TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_CUTOFF_CONVENTION = (
    "rectangular_Ng1_Ng2_labels_but_pinned_gen_coeff_radius_uses_Ng1_in_all_four_"
    "RX_RY_terms_including_b2_terms;strict_abs_Q_lt_radius_minus_margin"
)
TBG_ZERO_FIELD_COMPANION_HOPPING_CONVENTION = (
    "directed_layer2_to_layer1_zero_fill:T1=(0,0),T2=(1,1),T3=(0,1)"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_LABEL_ORDER = "nested_ik1_then_ik2_then_G1_then_G2"
TBG_ZERO_FIELD_COMPANION_INTERACTION_INDEX_CONVENTION = (
    "(((ik1*N2+ik2)*(2*NG1)+(G1+NG1))*(2*NG2)+(G2+NG2))"
)
TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION = (
    "rectangular_NG1_NG2_labels_but_pinned_gen_interaction_radius_uses_NG1_for_"
    "both_R1_R2;strict_abs_total_Q_lt_radius_minus_margin"
)
TBG_ZERO_FIELD_COMPANION_MASK_DIGEST_CONVENTION = (
    "sha256_one_byte_per_entry_false_0_true_1_in_label_order"
)


def _strict_int(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "a positive integer" if positive else "an integer"
        raise TypeError(f"{name} must be {qualifier} (bool is not accepted), got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _finite_complex(value: object, *, name: str) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError(f"{name} must be a complex-compatible finite scalar, got {value!r}")
    resolved = complex(value)
    if not math.isfinite(resolved.real) or not math.isfinite(resolved.imag):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if resolved == 0.0j:
        raise ValueError(f"{name} must be nonzero")
    return resolved


def _finite_margin(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"margin must be a finite nonnegative real scalar, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"margin must be finite and nonnegative, got {value!r}")
    return resolved


def _validate_schema_version(value: object, *, expected: int, schema: str) -> int:
    resolved = _strict_int(value, name="schema_version", positive=True)
    if resolved != expected:
        raise ValueError(f"Unsupported {schema} schema version {value!r}")
    return resolved


def _dot_2d(left: complex, right: complex) -> float:
    return float(left.real * right.real + left.imag * right.imag)


def tbg_zero_field_companion_perpendicular_norm(vector: complex, axis: complex) -> float:
    """Return ``|v-axis*dot(v,axis)/dot(axis,axis)|`` in complex 2D notation."""

    resolved_vector = _finite_complex_allow_zero(vector, name="vector")
    resolved_axis = _finite_complex(axis, name="axis")
    projection = resolved_axis * (
        _dot_2d(resolved_vector, resolved_axis) / _dot_2d(resolved_axis, resolved_axis)
    )
    return float(abs(resolved_vector - projection))


def _finite_complex_allow_zero(value: object, *, name: str) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError(f"{name} must be a complex-compatible finite scalar, got {value!r}")
    resolved = complex(value)
    if not math.isfinite(resolved.real) or not math.isfinite(resolved.imag):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return resolved


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mask_sha256(mask: tuple[bool, ...]) -> str:
    return hashlib.sha256(bytes(1 if value else 0 for value in mask)).hexdigest()


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


@dataclass(frozen=True)
class TBGZeroFieldCompanionPlaneWaveSpec:
    """Pinned companion plane-wave cutoff and parent-index convention."""

    Ng1: int
    Ng2: int
    b1: complex
    b2: complex
    margin: float = 1.0e-5
    schema: str = TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA_VERSION
    label_order: str = TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_LABEL_ORDER
    flattened_index_convention: str = TBG_ZERO_FIELD_COMPANION_FLATTENED_INDEX_CONVENTION
    cutoff_convention: str = TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_CUTOFF_CONVENTION
    hopping_convention: str = TBG_ZERO_FIELD_COMPANION_HOPPING_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA:
            raise ValueError(f"Unsupported companion plane-wave schema {self.schema!r}")
        schema_version = _validate_schema_version(
            self.schema_version,
            expected=TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA_VERSION,
            schema="companion plane-wave",
        )
        if self.label_order != TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_LABEL_ORDER:
            raise ValueError("Companion plane-wave label_order differs from the pinned convention")
        if self.flattened_index_convention != TBG_ZERO_FIELD_COMPANION_FLATTENED_INDEX_CONVENTION:
            raise ValueError("Companion flattened_index_convention differs from the pinned convention")
        if self.cutoff_convention != TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_CUTOFF_CONVENTION:
            raise ValueError("Companion plane-wave cutoff_convention differs from the pinned convention")
        if self.hopping_convention != TBG_ZERO_FIELD_COMPANION_HOPPING_CONVENTION:
            raise ValueError("Companion hopping_convention differs from the pinned convention")

        Ng1 = _strict_int(self.Ng1, name="Ng1", positive=True)
        Ng2 = _strict_int(self.Ng2, name="Ng2", positive=True)
        b1 = _finite_complex(self.b1, name="b1")
        b2 = _finite_complex(self.b2, name="b2")
        margin = _finite_margin(self.margin)
        object.__setattr__(self, "Ng1", Ng1)
        object.__setattr__(self, "Ng2", Ng2)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "schema_version", schema_version)
        if self.radius <= 0.0:
            raise ValueError("b1 and b2 must span 2D so that the companion radius is positive")

    @property
    def g1_labels(self) -> tuple[int, ...]:
        return tuple(range(-self.Ng1, self.Ng1))

    @property
    def g2_labels(self) -> tuple[int, ...]:
        return tuple(range(-self.Ng2, self.Ng2))

    @property
    def X(self) -> complex:
        return complex((2.0 / 3.0) * self.b1 + (1.0 / 3.0) * self.b2)

    @property
    def Y(self) -> complex:
        return complex((1.0 / 3.0) * self.b1 - (1.0 / 3.0) * self.b2)

    @property
    def radius(self) -> float:
        """Return the literal pinned ``gen_coeff`` RX/RY cutoff radius.

        Rectangular labels retain independent ``Ng1``/``Ng2`` extents, but the
        source uses ``Ng1`` in all four radius terms, including both ``b2``
        terms.  This asymmetry is intentional companion behavior.
        """

        return float(
            min(
                tbg_zero_field_companion_perpendicular_norm(
                    self.Ng1 * self.b1 - self.X, self.b2
                ),
                tbg_zero_field_companion_perpendicular_norm(
                    self.Ng1 * self.b2 - self.X, self.b1
                ),
                tbg_zero_field_companion_perpendicular_norm(
                    self.Ng1 * self.b1 - self.Y, self.b2
                ),
                tbg_zero_field_companion_perpendicular_norm(
                    -self.Ng1 * self.b2 - self.Y, self.b1
                ),
            )
        )

    @property
    def parent_g_count(self) -> int:
        return int(4 * self.Ng1 * self.Ng2)

    @property
    def parent_dimension(self) -> int:
        return int(4 * self.parent_g_count)

    def g_index(self, g1: int, g2: int) -> int:
        resolved_g1 = _strict_int(g1, name="g1")
        resolved_g2 = _strict_int(g2, name="g2")
        if resolved_g1 not in range(-self.Ng1, self.Ng1):
            raise ValueError(f"g1={resolved_g1} is outside range(-Ng1,Ng1)")
        if resolved_g2 not in range(-self.Ng2, self.Ng2):
            raise ValueError(f"g2={resolved_g2} is outside range(-Ng2,Ng2)")
        return int((resolved_g1 + self.Ng1) * (2 * self.Ng2) + resolved_g2 + self.Ng2)

    def companion_index(self, g1: int, g2: int, species: int) -> int:
        resolved_species = _strict_int(species, name="species")
        if resolved_species not in (0, 1, 2, 3):
            raise ValueError(f"species must be one of 0,1,2,3, got {species!r}")
        return int(4 * self.g_index(g1, g2) + resolved_species)

    def _payload(self) -> dict[str, object]:
        return {
            "Ng1": self.Ng1,
            "Ng2": self.Ng2,
            "X": _complex_pair(self.X),
            "Y": _complex_pair(self.Y),
            "b1": _complex_pair(self.b1),
            "b2": _complex_pair(self.b2),
            "cutoff_convention": self.cutoff_convention,
            "flattened_index_convention": self.flattened_index_convention,
            "hopping_convention": self.hopping_convention,
            "hopping_reference_function": TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION,
            "hopping_reference_lines": TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES,
            "label_order": self.label_order,
            "margin": self.margin,
            "parent_dimension": self.parent_dimension,
            "parent_g_count": self.parent_g_count,
            "radius": self.radius,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_function": TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION,
            "reference_lines": TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES,
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "schema": self.schema,
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class TBGZeroFieldCompanionBasisEntry:
    """One active species entry in exact companion ``sub_index`` order."""

    g1: int
    g2: int
    layer: Literal[1, 2]
    species: Literal[0, 1, 2, 3]
    companion_index: int
    subspace_index: int
    Q: complex

    @property
    def g_label(self) -> tuple[int, int]:
        return (self.g1, self.g2)


@dataclass(frozen=True)
class TBGZeroFieldCompanionHoppingEdge:
    """Directed 2x2 block geometry; no hopping matrix is stored."""

    channel: Literal["T1", "T2", "T3"]
    source_label: tuple[int, int]
    target_label: tuple[int, int]
    source_g_index: int
    target_g_index: int
    source_companion_indices: tuple[int, int]
    target_companion_indices: tuple[int, int]
    source_subspace_indices: tuple[int, int] | None
    target_subspace_indices: tuple[int, int] | None

    @property
    def is_active(self) -> bool:
        return self.source_subspace_indices is not None and self.target_subspace_indices is not None


@dataclass(frozen=True)
class TBGZeroFieldCompanionPlaneWaveGeometry:
    """Selected basis and zero-filled parent edges at one directly solved K point."""

    spec: TBGZeroFieldCompanionPlaneWaveSpec
    N1: int
    N2: int
    ik1: int
    ik2: int
    stau: Literal[1]
    k: complex
    basis_entries: tuple[TBGZeroFieldCompanionBasisEntry, ...]
    sub_index: tuple[int, ...]
    companion_to_subspace_index: tuple[int, ...]
    hopping_edges: tuple[TBGZeroFieldCompanionHoppingEdge, ...]

    @property
    def basis_count(self) -> int:
        return len(self.sub_index)

    @property
    def active_hopping_edges(self) -> tuple[TBGZeroFieldCompanionHoppingEdge, ...]:
        return tuple(edge for edge in self.hopping_edges if edge.is_active)

    @property
    def parent_hopping_edge_count(self) -> int:
        return len(self.hopping_edges)

    @property
    def active_hopping_edge_count(self) -> int:
        return len(self.active_hopping_edges)

    @property
    def sub_index_sha256(self) -> str:
        return _json_sha256(list(self.sub_index))

    @property
    def hopping_edges_sha256(self) -> str:
        payload = [
            {
                "channel": edge.channel,
                "source_companion_indices": list(edge.source_companion_indices),
                "source_label": list(edge.source_label),
                "source_subspace_indices": (
                    None
                    if edge.source_subspace_indices is None
                    else list(edge.source_subspace_indices)
                ),
                "target_companion_indices": list(edge.target_companion_indices),
                "target_label": list(edge.target_label),
                "target_subspace_indices": (
                    None
                    if edge.target_subspace_indices is None
                    else list(edge.target_subspace_indices)
                ),
            }
            for edge in self.hopping_edges
        ]
        return _json_sha256(payload)

    def _payload(self) -> dict[str, object]:
        return {
            "N1": self.N1,
            "N2": self.N2,
            "active_hopping_edge_count": self.active_hopping_edge_count,
            "basis_count": self.basis_count,
            "hopping_edges_sha256": self.hopping_edges_sha256,
            "ik1": self.ik1,
            "ik2": self.ik2,
            "k": _complex_pair(self.k),
            "parent_hopping_edge_count": self.parent_hopping_edge_count,
            "schema": self.spec.schema,
            "schema_version": self.spec.schema_version,
            "spec_fingerprint": self.spec.fingerprint,
            "stau": self.stau,
            "sub_index_sha256": self.sub_index_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def build_tbg_zero_field_companion_plane_wave_geometry(
    spec: TBGZeroFieldCompanionPlaneWaveSpec,
    *,
    N1: int,
    N2: int,
    ik1: int,
    ik2: int,
    stau: int,
) -> TBGZeroFieldCompanionPlaneWaveGeometry:
    """Build the exact companion ``stau=+1`` K basis and parent-edge list."""

    if not isinstance(spec, TBGZeroFieldCompanionPlaneWaveSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionPlaneWaveSpec")
    resolved_N1 = _strict_int(N1, name="N1", positive=True)
    resolved_N2 = _strict_int(N2, name="N2", positive=True)
    resolved_ik1 = _strict_int(ik1, name="ik1")
    resolved_ik2 = _strict_int(ik2, name="ik2")
    resolved_stau = _strict_int(stau, name="stau")
    if not 0 <= resolved_ik1 < resolved_N1:
        raise ValueError(f"ik1={resolved_ik1} must satisfy 0 <= ik1 < N1={resolved_N1}")
    if not 0 <= resolved_ik2 < resolved_N2:
        raise ValueError(f"ik2={resolved_ik2} must satisfy 0 <= ik2 < N2={resolved_N2}")
    if resolved_stau != 1:
        raise ValueError(
            "Only the directly solved K valley stau=+1 is supported; K' must come from "
            "the separate companion time-reversal mesh/wrapped-G construction, not "
            "direct cutoff reuse"
        )

    k = complex(
        (resolved_ik1 / resolved_N1) * spec.b1
        + (resolved_ik2 / resolved_N2) * spec.b2
    )
    threshold = spec.radius - spec.margin
    basis_entries: list[TBGZeroFieldCompanionBasisEntry] = []
    sub_index: list[int] = []
    companion_to_subspace = [-1] * spec.parent_dimension

    for g1 in spec.g1_labels:
        for g2 in spec.g2_labels:
            base = k + g1 * spec.b1 + g2 * spec.b2
            Q1 = complex(base + resolved_stau * (spec.b1 - spec.b2) / 3.0)
            Q2 = complex(base + resolved_stau * (2.0 * spec.b1 + spec.b2) / 3.0)
            layer_data = ((1, (0, 1), Q1), (2, (2, 3), Q2))
            for layer, species_pair, Q in layer_data:
                if abs(Q) >= threshold:
                    continue
                for species in species_pair:
                    companion_index = spec.companion_index(g1, g2, species)
                    subspace_index = len(sub_index)
                    sub_index.append(companion_index)
                    companion_to_subspace[companion_index] = subspace_index
                    basis_entries.append(
                        TBGZeroFieldCompanionBasisEntry(
                            g1=g1,
                            g2=g2,
                            layer=layer,  # type: ignore[arg-type]
                            species=species,  # type: ignore[arg-type]
                            companion_index=companion_index,
                            subspace_index=subspace_index,
                            Q=Q,
                        )
                    )

    edges: list[TBGZeroFieldCompanionHoppingEdge] = []
    channel_shifts = (("T1", 0, 0), ("T2", 1, 1), ("T3", 0, 1))
    for source_g1 in spec.g1_labels:
        for source_g2 in spec.g2_labels:
            for channel, shift_g1, shift_g2 in channel_shifts:
                target_g1 = source_g1 + shift_g1
                target_g2 = source_g2 + shift_g2
                if target_g1 not in range(-spec.Ng1, spec.Ng1):
                    continue
                if target_g2 not in range(-spec.Ng2, spec.Ng2):
                    continue
                source_companion_indices = (
                    spec.companion_index(source_g1, source_g2, 2),
                    spec.companion_index(source_g1, source_g2, 3),
                )
                target_companion_indices = (
                    spec.companion_index(target_g1, target_g2, 0),
                    spec.companion_index(target_g1, target_g2, 1),
                )
                source_positions = tuple(
                    companion_to_subspace[index] for index in source_companion_indices
                )
                target_positions = tuple(
                    companion_to_subspace[index] for index in target_companion_indices
                )
                edges.append(
                    TBGZeroFieldCompanionHoppingEdge(
                        channel=channel,  # type: ignore[arg-type]
                        source_label=(source_g1, source_g2),
                        target_label=(target_g1, target_g2),
                        source_g_index=spec.g_index(source_g1, source_g2),
                        target_g_index=spec.g_index(target_g1, target_g2),
                        source_companion_indices=source_companion_indices,
                        target_companion_indices=target_companion_indices,
                        source_subspace_indices=(
                            None
                            if any(position < 0 for position in source_positions)
                            else (source_positions[0], source_positions[1])
                        ),
                        target_subspace_indices=(
                            None
                            if any(position < 0 for position in target_positions)
                            else (target_positions[0], target_positions[1])
                        ),
                    )
                )

    return TBGZeroFieldCompanionPlaneWaveGeometry(
        spec=spec,
        N1=resolved_N1,
        N2=resolved_N2,
        ik1=resolved_ik1,
        ik2=resolved_ik2,
        stau=resolved_stau,  # type: ignore[arg-type]
        k=k,
        basis_entries=tuple(basis_entries),
        sub_index=tuple(sub_index),
        companion_to_subspace_index=tuple(companion_to_subspace),
        hopping_edges=tuple(edges),
    )


@dataclass(frozen=True)
class TBGZeroFieldCompanionInteractionSpec:
    """Pinned companion circular total-Q interaction cutoff."""

    NG1: int
    NG2: int
    b1: complex
    b2: complex
    margin: float = 1.0e-5
    schema: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION
    label_order: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_LABEL_ORDER
    index_convention: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_INDEX_CONVENTION
    cutoff_convention: str = TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION
    mask_digest_convention: str = TBG_ZERO_FIELD_COMPANION_MASK_DIGEST_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA:
            raise ValueError(f"Unsupported companion interaction schema {self.schema!r}")
        schema_version = _validate_schema_version(
            self.schema_version,
            expected=TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION,
            schema="companion interaction",
        )
        if self.label_order != TBG_ZERO_FIELD_COMPANION_INTERACTION_LABEL_ORDER:
            raise ValueError("Companion interaction label_order differs from the pinned convention")
        if self.index_convention != TBG_ZERO_FIELD_COMPANION_INTERACTION_INDEX_CONVENTION:
            raise ValueError("Companion interaction index_convention differs from the pinned convention")
        if self.cutoff_convention != TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION:
            raise ValueError("Companion interaction cutoff_convention differs from the pinned convention")
        if self.mask_digest_convention != TBG_ZERO_FIELD_COMPANION_MASK_DIGEST_CONVENTION:
            raise ValueError("Companion mask_digest_convention differs from the pinned convention")

        NG1 = _strict_int(self.NG1, name="NG1", positive=True)
        NG2 = _strict_int(self.NG2, name="NG2", positive=True)
        b1 = _finite_complex(self.b1, name="b1")
        b2 = _finite_complex(self.b2, name="b2")
        margin = _finite_margin(self.margin)
        object.__setattr__(self, "NG1", NG1)
        object.__setattr__(self, "NG2", NG2)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "schema_version", schema_version)
        if self.radius <= 0.0:
            raise ValueError("b1 and b2 must span 2D so that the interaction radius is positive")

    @property
    def G1_labels(self) -> tuple[int, ...]:
        return tuple(range(-self.NG1, self.NG1))

    @property
    def G2_labels(self) -> tuple[int, ...]:
        return tuple(range(-self.NG2, self.NG2))

    @property
    def radius(self) -> float:
        """Return the literal pinned ``gen_interaction`` R1/R2 cutoff radius.

        Rectangular labels retain independent ``NG1``/``NG2`` extents, but the
        source multiplies both perpendicular norms by ``NG1``.  This asymmetry
        is intentional companion behavior.
        """

        return float(
            min(
                self.NG1 * tbg_zero_field_companion_perpendicular_norm(self.b1, self.b2),
                self.NG1 * tbg_zero_field_companion_perpendicular_norm(self.b2, self.b1),
            )
        )

    def flat_index(self, *, N2: int, ik1: int, ik2: int, G1: int, G2: int) -> int:
        resolved_N2 = _strict_int(N2, name="N2", positive=True)
        resolved_ik1 = _strict_int(ik1, name="ik1")
        resolved_ik2 = _strict_int(ik2, name="ik2")
        resolved_G1 = _strict_int(G1, name="G1")
        resolved_G2 = _strict_int(G2, name="G2")
        if resolved_ik1 < 0:
            raise ValueError("ik1 must be nonnegative")
        if not 0 <= resolved_ik2 < resolved_N2:
            raise ValueError(f"ik2={resolved_ik2} must satisfy 0 <= ik2 < N2={resolved_N2}")
        if resolved_G1 not in range(-self.NG1, self.NG1):
            raise ValueError(f"G1={resolved_G1} is outside range(-NG1,NG1)")
        if resolved_G2 not in range(-self.NG2, self.NG2):
            raise ValueError(f"G2={resolved_G2} is outside range(-NG2,NG2)")
        return int(
            (((resolved_ik1 * resolved_N2 + resolved_ik2) * (2 * self.NG1) + resolved_G1 + self.NG1)
            * (2 * self.NG2))
            + resolved_G2
            + self.NG2
        )

    def _payload(self) -> dict[str, object]:
        return {
            "NG1": self.NG1,
            "NG2": self.NG2,
            "b1": _complex_pair(self.b1),
            "b2": _complex_pair(self.b2),
            "cutoff_convention": self.cutoff_convention,
            "index_convention": self.index_convention,
            "label_order": self.label_order,
            "margin": self.margin,
            "mask_digest_convention": self.mask_digest_convention,
            "radius": self.radius,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "reference_function": TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION,
            "reference_lines": TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES,
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "reference_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "schema": self.schema,
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class TBGZeroFieldCompanionInteractionEntry:
    """One total-Q label in exact nested interaction ordering."""

    index: int
    ik1: int
    ik2: int
    G1: int
    G2: int
    Q: complex
    active: bool

    @property
    def label(self) -> tuple[int, int, int, int]:
        return (self.ik1, self.ik2, self.G1, self.G2)


@dataclass(frozen=True)
class TBGZeroFieldCompanionInteractionGeometry:
    """Exact total-Q labels, indices, and circular-cutoff mask."""

    spec: TBGZeroFieldCompanionInteractionSpec
    N1: int
    N2: int
    entries: tuple[TBGZeroFieldCompanionInteractionEntry, ...]
    labels: tuple[tuple[int, int, int, int], ...]
    active_mask: tuple[bool, ...]
    active_indices: tuple[int, ...]

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def active_count(self) -> int:
        return len(self.active_indices)

    @property
    def active_mask_sha256(self) -> str:
        return _mask_sha256(self.active_mask)

    @property
    def labels_sha256(self) -> str:
        return _json_sha256([list(label) for label in self.labels])

    def _payload(self) -> dict[str, object]:
        return {
            "N1": self.N1,
            "N2": self.N2,
            "active_count": self.active_count,
            "active_mask_sha256": self.active_mask_sha256,
            "label_order": self.spec.label_order,
            "labels_sha256": self.labels_sha256,
            "mask_digest_convention": self.spec.mask_digest_convention,
            "schema": self.spec.schema,
            "schema_version": self.spec.schema_version,
            "spec_fingerprint": self.spec.fingerprint,
            "total_count": self.total_count,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self._payload())

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def build_tbg_zero_field_companion_interaction_geometry(
    spec: TBGZeroFieldCompanionInteractionSpec,
    *,
    N1: int,
    N2: int,
) -> TBGZeroFieldCompanionInteractionGeometry:
    """Build total-Q labels and the strict companion circular-cutoff mask."""

    if not isinstance(spec, TBGZeroFieldCompanionInteractionSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionInteractionSpec")
    resolved_N1 = _strict_int(N1, name="N1", positive=True)
    resolved_N2 = _strict_int(N2, name="N2", positive=True)
    threshold = spec.radius - spec.margin

    entries: list[TBGZeroFieldCompanionInteractionEntry] = []
    labels: list[tuple[int, int, int, int]] = []
    active_mask: list[bool] = []
    active_indices: list[int] = []
    for ik1 in range(resolved_N1):
        for ik2 in range(resolved_N2):
            for G1 in spec.G1_labels:
                for G2 in spec.G2_labels:
                    index = spec.flat_index(
                        N2=resolved_N2,
                        ik1=ik1,
                        ik2=ik2,
                        G1=G1,
                        G2=G2,
                    )
                    Q = complex(
                        (ik1 / resolved_N1 + G1) * spec.b1
                        + (ik2 / resolved_N2 + G2) * spec.b2
                    )
                    active = bool(abs(Q) < threshold)
                    label = (ik1, ik2, G1, G2)
                    labels.append(label)
                    active_mask.append(active)
                    if active:
                        active_indices.append(index)
                    entries.append(
                        TBGZeroFieldCompanionInteractionEntry(
                            index=index,
                            ik1=ik1,
                            ik2=ik2,
                            G1=G1,
                            G2=G2,
                            Q=Q,
                            active=active,
                        )
                    )

    return TBGZeroFieldCompanionInteractionGeometry(
        spec=spec,
        N1=resolved_N1,
        N2=resolved_N2,
        entries=tuple(entries),
        labels=tuple(labels),
        active_mask=tuple(active_mask),
        active_indices=tuple(active_indices),
    )


__all__ = [
    "TBGZeroFieldCompanionBasisEntry",
    "TBGZeroFieldCompanionHoppingEdge",
    "TBGZeroFieldCompanionInteractionEntry",
    "TBGZeroFieldCompanionInteractionGeometry",
    "TBGZeroFieldCompanionInteractionSpec",
    "TBGZeroFieldCompanionPlaneWaveGeometry",
    "TBGZeroFieldCompanionPlaneWaveSpec",
    "TBG_ZERO_FIELD_COMPANION_FLATTENED_INDEX_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HOPPING_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_FUNCTION",
    "TBG_ZERO_FIELD_COMPANION_HOPPING_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_CUTOFF_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_INDEX_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_LABEL_ORDER",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_FUNCTION",
    "TBG_ZERO_FIELD_COMPANION_INTERACTION_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_MASK_DIGEST_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_CUTOFF_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_LABEL_ORDER",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_FUNCTION",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_PLANE_WAVE_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT",
    "TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY",
    "TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256",
    "build_tbg_zero_field_companion_interaction_geometry",
    "build_tbg_zero_field_companion_plane_wave_geometry",
    "tbg_zero_field_companion_perpendicular_norm",
]
