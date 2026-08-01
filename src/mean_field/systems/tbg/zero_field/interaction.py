from __future__ import annotations

"""Typed interaction contract for zero-field TBG Hartree--Fock sources.

The low-level B0 kernel uses dimensionless momenta measured in units of the
inverse graphene lattice constant and evaluates ``tanh(2 * |q| * lm)``.  A
physical dual-gate distance therefore maps to
``lm = dsc_nm / (2 * graphene_a_nm)`` so that the argument is
``|q_physical| * dsc_nm``.

Evidence paths:
- ``reference/2511.21683v1.pdf``, Fig. 8 caption: 10x10, 1.05 degrees,
  wAA=80 meV, wAB=110 meV, and epsr=10.
- ``reference/TBG-HF/int_input.json`` and ``HF_input.json``: dual gate,
  dsc=25 nm, included q=0, and ``average central`` reference.
- ``reference/TBG-HF/singleParticle.py::gen_interaction``: physical
  ``tanh(|q|*dsc)/|q|`` dual-gate kernel and finite q=0 limit.

Uncertainty: the executable transfer cutoff below is the explicit reciprocal
G-label shell ``max(|m|,|n|,|m+n|)<=3``, not the companion code's circular
total-Q interaction cutoff.  Companion cutoff parity is recorded as not
established and must pass separately before any Fig. 8 reproduction claim.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal


TBG_ZERO_FIELD_INTERACTION_SCHEMA = "mean_field.tbg.zero_field.interaction"
TBG_ZERO_FIELD_INTERACTION_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY = "g_label_hex_shell_3"
TBG_ZERO_FIELD_REFERENCE_SCHEME = "central_average_active_two_band"
TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY = "not_established"
TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1 = 0.246


@dataclass(frozen=True)
class TBGZeroFieldInteractionSpec:
    """Executable physical interaction identity for the central two-band model.

    ``transfer_cutoff_policy`` deliberately names the reciprocal-label cutoff
    executed by the typed block builder. It is not the circular ``NG1=NG2=5`` cutoff of the
    tutorial companion implementation; a future companion-parity branch must
    use a distinct policy rather than silently changing this contract.
    """

    gate_geometry: Literal["dual"] = "dual"
    dsc_nm: float = 25.0
    epsr: float = 10.0
    graphene_a_nm: float = 0.246
    finite_zero_limit: bool = True
    zero_cutoff: float = 1.0e-6
    reference_scheme: Literal["central_average_active_two_band"] = (
        TBG_ZERO_FIELD_REFERENCE_SCHEME
    )
    transfer_cutoff_policy: Literal["g_label_hex_shell_3"] = (
        TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY
    )
    companion_circular_total_q_cutoff_parity: Literal["not_established"] = (
        TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY
    )
    schema: str = TBG_ZERO_FIELD_INTERACTION_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_INTERACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_INTERACTION_SCHEMA:
            raise ValueError(f"Unsupported TBG zero-field interaction schema {self.schema!r}")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != TBG_ZERO_FIELD_INTERACTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported TBG zero-field interaction schema version "
                f"{self.schema_version!r}"
            )
        if self.gate_geometry != "dual":
            raise ValueError("TBG zero-field typed interaction currently supports only gate_geometry='dual'")
        if self.reference_scheme != TBG_ZERO_FIELD_REFERENCE_SCHEME:
            raise ValueError(
                "TBG zero-field typed interaction requires "
                f"reference_scheme={TBG_ZERO_FIELD_REFERENCE_SCHEME!r}"
            )
        if self.transfer_cutoff_policy != TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY:
            raise ValueError(
                "Unsupported TBG zero-field transfer cutoff policy "
                f"{self.transfer_cutoff_policy!r}"
            )
        if self.companion_circular_total_q_cutoff_parity != TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY:
            raise ValueError(
                "TBG zero-field schema-v1 must state that companion circular total-Q "
                "cutoff parity is not established"
            )
        for name in ("dsc_nm", "epsr", "graphene_a_nm", "zero_cutoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            resolved = float(value)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
            object.__setattr__(self, name, resolved)
        if self.graphene_a_nm != TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1:
            raise ValueError(
                "TBG zero-field interaction schema-v1 freezes graphene_a_nm exactly "
                f"at {TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1}"
            )
        object.__setattr__(self, "graphene_a_nm", TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1)
        if not isinstance(self.finite_zero_limit, bool):
            raise TypeError("finite_zero_limit must be bool")
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def screening_lm(self) -> float:
        """Dimensionless ``lm`` consumed by the legacy ``tanh(2 q lm)`` kernel."""

        return float(self.dsc_nm / (2.0 * self.graphene_a_nm))

    def _payload(self) -> dict[str, object]:
        return {
            "companion_circular_total_q_cutoff_parity": self.companion_circular_total_q_cutoff_parity,
            "dsc_nm": self.dsc_nm,
            "epsr": self.epsr,
            "finite_zero_limit": self.finite_zero_limit,
            "gate_geometry": self.gate_geometry,
            "graphene_a_nm": self.graphene_a_nm,
            "reference_scheme": self.reference_scheme,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "screening_lm": self.screening_lm,
            "transfer_cutoff_policy": self.transfer_cutoff_policy,
            "zero_cutoff": self.zero_cutoff,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json(include_fingerprint=False).encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def to_json(self, *, include_fingerprint: bool = True) -> str:
        payload = self.to_metadata() if include_fingerprint else self._payload()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "TBGZeroFieldInteractionSpec":
        payload = dict(metadata)
        expected_keys = {
            "companion_circular_total_q_cutoff_parity",
            "dsc_nm",
            "epsr",
            "finite_zero_limit",
            "fingerprint",
            "gate_geometry",
            "graphene_a_nm",
            "reference_scheme",
            "schema",
            "schema_version",
            "screening_lm",
            "transfer_cutoff_policy",
            "zero_cutoff",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "TBG zero-field interaction metadata keys differ from the supported schema: "
                f"expected={sorted(expected_keys)}, got={sorted(payload)}"
            )
        spec = cls(
            gate_geometry=payload["gate_geometry"],  # type: ignore[arg-type]
            dsc_nm=payload["dsc_nm"],  # type: ignore[arg-type]
            epsr=payload["epsr"],  # type: ignore[arg-type]
            graphene_a_nm=payload["graphene_a_nm"],  # type: ignore[arg-type]
            finite_zero_limit=payload["finite_zero_limit"],  # type: ignore[arg-type]
            zero_cutoff=payload["zero_cutoff"],  # type: ignore[arg-type]
            reference_scheme=payload["reference_scheme"],  # type: ignore[arg-type]
            transfer_cutoff_policy=payload["transfer_cutoff_policy"],  # type: ignore[arg-type]
            companion_circular_total_q_cutoff_parity=payload[
                "companion_circular_total_q_cutoff_parity"
            ],  # type: ignore[arg-type]
            schema=payload["schema"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        supplied_lm = payload["screening_lm"]
        if isinstance(supplied_lm, bool) or not isinstance(supplied_lm, (int, float)):
            raise ValueError("screening_lm metadata must be a real scalar")
        if not math.isclose(float(supplied_lm), spec.screening_lm, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError("TBG zero-field interaction screening_lm does not match dsc_nm/(2*graphene_a_nm)")
        supplied_fingerprint = payload["fingerprint"]
        if not isinstance(supplied_fingerprint, str) or supplied_fingerprint != spec.fingerprint:
            raise ValueError("TBG zero-field interaction metadata fingerprint does not match its fields")
        return spec

    @classmethod
    def from_json(cls, payload: str) -> "TBGZeroFieldInteractionSpec":
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("TBG zero-field interaction JSON must encode an object")
        return cls.from_metadata(decoded)


__all__ = [
    "TBGZeroFieldInteractionSpec",
    "TBG_ZERO_FIELD_INTERACTION_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY",
    "TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1",
    "TBG_ZERO_FIELD_INTERACTION_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_REFERENCE_SCHEME",
    "TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY",
]
