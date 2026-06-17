from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .lattice import HTQGLattice

DomainKey = Literal["alpha_beta_alpha", "beta_alpha_beta", "alpha_beta_gamma", "gamma_beta_alpha"]

_DOMAIN_ALIASES: dict[str, DomainKey] = {
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
    "alpha_beta_alpha": "αβα",
    "beta_alpha_beta": "βαβ",
    "alpha_beta_gamma": "αβγ",
    "gamma_beta_alpha": "γβα",
}

_DOMAIN_TYPES: dict[DomainKey, str] = {
    "alpha_beta_alpha": "Type-I Bernal",
    "beta_alpha_beta": "Type-I Bernal",
    "alpha_beta_gamma": "Type-II rhombohedral",
    "gamma_beta_alpha": "Type-II rhombohedral",
}

_C2ZT_PARTNERS: dict[DomainKey, DomainKey] = {
    "alpha_beta_alpha": "beta_alpha_beta",
    "beta_alpha_beta": "alpha_beta_alpha",
    "alpha_beta_gamma": "gamma_beta_alpha",
    "gamma_beta_alpha": "alpha_beta_gamma",
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
    if key == "alpha_beta_alpha":
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
    "all_domains",
    "canonical_domain_key",
    "domain_displacements",
    "mirror_x",
    "mirror_y",
    "representative_domains",
]
