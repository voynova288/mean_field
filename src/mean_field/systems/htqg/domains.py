from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .lattice import HTQGLattice

DomainKey = Literal[
    "alpha_alpha_alpha",
    "alpha_beta_alpha",
    "beta_alpha_beta",
    "alpha_beta_gamma",
    "gamma_beta_alpha",
]

_DOMAIN_ALIASES: dict[str, DomainKey] = {
    "alpha_alpha_alpha": "alpha_alpha_alpha",
    "alpha-alpha-alpha": "alpha_alpha_alpha",
    "alphaalphaalpha": "alpha_alpha_alpha",
    "aaa": "alpha_alpha_alpha",
    "ααα": "alpha_alpha_alpha",
    "alpha_beta_alpha": "alpha_beta_alpha",
    "alpha-beta-alpha": "alpha_beta_alpha",
    "alphabetaalpha": "alpha_beta_alpha",
    "aba": "alpha_beta_alpha",
    "αβα": "alpha_beta_alpha",
    "beta_alpha_beta": "beta_alpha_beta",
    "beta-alpha-beta": "beta_alpha_beta",
    "betaalphabeta": "beta_alpha_beta",
    "bab": "beta_alpha_beta",
    "βαβ": "beta_alpha_beta",
    "alpha_beta_gamma": "alpha_beta_gamma",
    "alpha-beta-gamma": "alpha_beta_gamma",
    "alphabetagamma": "alpha_beta_gamma",
    "abg": "alpha_beta_gamma",
    "αβγ": "alpha_beta_gamma",
    "gamma_beta_alpha": "gamma_beta_alpha",
    "gamma-beta-alpha": "gamma_beta_alpha",
    "gammabetaalpha": "gamma_beta_alpha",
    "gba": "gamma_beta_alpha",
    "γβα": "gamma_beta_alpha",
}

_GREEK_LABELS: dict[DomainKey, str] = {
    "alpha_alpha_alpha": "ααα",
    "alpha_beta_alpha": "αβα",
    "beta_alpha_beta": "βαβ",
    "alpha_beta_gamma": "αβγ",
    "gamma_beta_alpha": "γβα",
}

_DOMAIN_TYPES: dict[DomainKey, str] = {
    "alpha_alpha_alpha": "AA-like metallic",
    "alpha_beta_alpha": "Type-I Bernal",
    "beta_alpha_beta": "Type-I Bernal",
    "alpha_beta_gamma": "Type-II rhombohedral",
    "gamma_beta_alpha": "Type-II rhombohedral",
}

_C2ZT_PARTNERS: dict[DomainKey, DomainKey] = {
    "alpha_alpha_alpha": "alpha_alpha_alpha",
    "alpha_beta_alpha": "beta_alpha_beta",
    "beta_alpha_beta": "alpha_beta_alpha",
    "alpha_beta_gamma": "gamma_beta_alpha",
    "gamma_beta_alpha": "alpha_beta_gamma",
}


@dataclass(frozen=True)
class HTQGExplicitDisplacements:
    """JSON-native explicit adjacent-interface moiré displacements.

    Coordinates are Cartesian nanometres in the same frame as
    :class:`HTQGLattice`.  Keeping real/imaginary parts in fixed length-two
    tuples makes projected-HF configs hashable and JSON serializable without
    hiding a workflow-specific interpolation inside a named domain.
    """

    d12_nm_xy: tuple[float, float]
    d34_nm_xy: tuple[float, float]
    label: str = "explicit"

    def __post_init__(self) -> None:
        for name in ("d12_nm_xy", "d34_nm_xy"):
            values = getattr(self, name)
            try:
                canonical = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain exactly two real Cartesian-nm values") from exc
            if len(canonical) != 2:
                raise ValueError(f"{name} must contain exactly two real Cartesian-nm values")
            object.__setattr__(self, name, canonical)
        object.__setattr__(self, "label", str(self.label))

    @classmethod
    def from_complex(
        cls,
        d12: complex,
        d34: complex,
        *,
        label: str = "explicit",
    ) -> "HTQGExplicitDisplacements":
        d12_value = complex(d12)
        d34_value = complex(d34)
        return cls(
            d12_nm_xy=(float(d12_value.real), float(d12_value.imag)),
            d34_nm_xy=(float(d34_value.real), float(d34_value.imag)),
            label=str(label),
        )

    @property
    def d12(self) -> complex:
        return complex(float(self.d12_nm_xy[0]), float(self.d12_nm_xy[1]))

    @property
    def d34(self) -> complex:
        return complex(float(self.d34_nm_xy[0]), float(self.d34_nm_xy[1]))

    def to_dict(self) -> dict[str, object]:
        return {
            "label": str(self.label),
            "d12_nm": [float(self.d12.real), float(self.d12.imag)],
            "d34_nm": [float(self.d34.real), float(self.d34.imag)],
        }

@dataclass(frozen=True)
class HTQGDomain:
    """Relaxed single-moiré-domain displacement data for HTQG."""

    key: DomainKey
    label: str
    domain_type: str
    d12: complex
    d34: complex
    c2zt_partner: DomainKey

    @property
    def is_aa_like(self) -> bool:
        return self.key == "alpha_alpha_alpha"

    @property
    def is_type_i(self) -> bool:
        return self.key in {"alpha_beta_alpha", "beta_alpha_beta"}

    @property
    def is_type_ii(self) -> bool:
        return self.key in {"alpha_beta_gamma", "gamma_beta_alpha"}

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "domain_type": self.domain_type,
            "d12_nm": [float(self.d12.real), float(self.d12.imag)],
            "d34_nm": [float(self.d34.real), float(self.d34.imag)],
            "c2zt_partner": self.c2zt_partner,
        }


def canonical_domain_key(domain: str | HTQGDomain) -> DomainKey:
    if isinstance(domain, HTQGDomain):
        return domain.key
    normalized = str(domain).strip().lower().replace(" ", "_")
    normalized = normalized.replace("__", "_")
    try:
        return _DOMAIN_ALIASES[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_DOMAIN_ALIASES))
        raise ValueError(f"Unknown HTQG domain {domain!r}. Known aliases: {known}") from exc


def domain_displacements(lattice: HTQGLattice, domain: str | HTQGDomain) -> HTQGDomain:
    key = canonical_domain_key(domain)
    d = complex(lattice.d_ba)
    if key == "alpha_alpha_alpha":
        # Shin et al., arXiv:2604.19608v1, Appendix B.1: the AA-like
        # alpha-alpha-alpha family has d_ll'=0 on every adjacent interface.
        # The middle d23 displacement is fixed to zero by this HTQG gauge,
        # so the named domain is represented by d12=d34=0 here.
        d12, d34 = 0.0j, 0.0j
    elif key == "alpha_beta_alpha":
        d12, d34 = d, -d
    elif key == "beta_alpha_beta":
        d12, d34 = -d, d
    elif key == "alpha_beta_gamma":
        d12, d34 = d, d
    elif key == "gamma_beta_alpha":
        d12, d34 = -d, -d
    else:  # pragma: no cover; canonical_domain_key exhausts the keys.
        raise AssertionError(key)
    return HTQGDomain(
        key=key,
        label=_GREEK_LABELS[key],
        domain_type=_DOMAIN_TYPES[key],
        d12=complex(d12),
        d34=complex(d34),
        c2zt_partner=_C2ZT_PARTNERS[key],
    )


def all_domains(lattice: HTQGLattice) -> tuple[HTQGDomain, ...]:
    return tuple(domain_displacements(lattice, key) for key in _GREEK_LABELS)


def representative_domains(lattice: HTQGLattice) -> tuple[HTQGDomain, HTQGDomain]:
    """Return the two domains that must be explicitly recomputed in the paper."""

    return (
        domain_displacements(lattice, "alpha_beta_alpha"),
        domain_displacements(lattice, "alpha_beta_gamma"),
    )


def mirror_x(value: complex) -> complex:
    return complex(value.real, -value.imag)


def mirror_y(value: complex) -> complex:
    return complex(-value.real, value.imag)


__all__ = [
    "DomainKey",
    "HTQGDomain",
    "HTQGExplicitDisplacements",
    "all_domains",
    "canonical_domain_key",
    "domain_displacements",
    "mirror_x",
    "mirror_y",
    "representative_domains",
]
