from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MagneticFlux:
    """Rational magnetic flux ``phi/phi0 = p/q`` with coprime integers."""

    p: int
    q: int

    def __post_init__(self) -> None:
        p = int(self.p)
        q = int(self.q)
        if q <= 0:
            raise ValueError(f"Flux denominator q must be positive, got {self.q}")
        if p == 0:
            raise ValueError("Finite-field calculations expect nonzero flux numerator p")
        frac = Fraction(p, q)
        object.__setattr__(self, "p", int(frac.numerator))
        object.__setattr__(self, "q", int(frac.denominator))

    @classmethod
    def from_value(cls, value: Fraction | tuple[int, int] | str) -> "MagneticFlux":
        if isinstance(value, Fraction):
            return cls(value.numerator, value.denominator)
        if isinstance(value, tuple):
            return cls(int(value[0]), int(value[1]))
        frac = Fraction(value)
        return cls(frac.numerator, frac.denominator)

    @property
    def ratio(self) -> float:
        return float(self.p / self.q)


def magnetic_reciprocal_vector(m: int, n: int, *, g1: complex, g2: complex, q: int) -> complex:
    """Return the finite-field reciprocal vector ``G = m*g1 + (n/q)*g2``."""

    q_int = int(q)
    if q_int <= 0:
        raise ValueError(f"q must be positive, got {q}")
    return complex(int(m) * g1 + int(n) / q_int * g2)


def in_hex_shell(m: int, n: int, *, g1: complex, g2: complex, q: int, shell_ng: int = 3) -> bool:
    """Hexagonal shell cutoff for finite-field reciprocal vectors.

    This is system agnostic once the moire reciprocal vectors are supplied. It
    mirrors the cutoff used by the TBG author magnetic-HF code and is useful for
    any moire finite-B adapter that indexes interaction shifts as
    ``G=m*g1+(n/q)*g2``.
    """

    shell_ng = int(shell_ng)
    if shell_ng < 0:
        raise ValueError(f"shell_ng must be non-negative, got {shell_ng}")
    g = magnetic_reciprocal_vector(m, n, g1=g1, g2=g2, q=q)
    if abs(g) < 1e-15:
        return True
    g0 = abs(shell_ng * g1 + shell_ng * g2) * 1.00001
    angle_mod = np.mod(np.angle(g), np.pi / 3.0) - np.pi / 6.0
    radius = g0 * np.cos(np.pi / 6.0) / abs(np.cos(angle_mod))
    return bool(abs(g) < radius)


def magnetic_shell_shifts(*, g1: complex, g2: complex, q: int, shell_ng: int = 3) -> tuple[tuple[int, int], ...]:
    """Return finite-B interaction-shell shifts in author-code ordering."""

    q = int(q)
    shell_ng = int(shell_ng)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if shell_ng < 0:
        raise ValueError(f"shell_ng must be non-negative, got {shell_ng}")
    shifts: list[tuple[int, int]] = []
    for m in range(-shell_ng, shell_ng + 1):
        for n in range(shell_ng * q, -shell_ng * q - 1, -1):
            if in_hex_shell(m, n, g1=g1, g2=g2, q=q, shell_ng=shell_ng):
                shifts.append((int(m), int(n)))
    return tuple(shifts)


def choose_magnetic_nq(q: int, *, max_q_points: int = 12) -> int:
    """Return a small magnetic-mesh ``nq`` for denominator ``q``.

    The default reproduces the finite-B TBG author-code/paper convention:
    ``12//q`` with a special ``q=7 -> nq=2`` case. Other systems can pass a
    different ``max_q_points`` or bypass this helper.
    """

    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    nq = int(max_q_points) // q
    if q == 7 and max_q_points >= 12:
        nq = max(nq, 2)
    return max(int(nq), 1)


def magnetic_orbit_indices(q: int, nq: int) -> Array:
    """Return full-strip indices ``(q, nq**2)`` for magnetic-orbit reductions."""

    q = int(q)
    nq = int(nq)
    if q <= 0 or nq <= 0:
        raise ValueError(f"q and nq must be positive, got q={q}, nq={nq}")
    return np.arange(q * nq * nq, dtype=int).reshape((q, nq * nq), order="F")


def magnetic_r_orbit_positions(p: int, q: int) -> Array:
    """Return the zero-based ``r*p mod q`` orbit used by magnetic translations."""

    p = int(p)
    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    return (np.arange(q, dtype=int) * p) % q


def magnetic_k_vectors(
    *,
    g1: complex,
    g2: complex,
    flux: MagneticFlux | Fraction | tuple[int, int] | str,
    nq: int,
    reduced_translation: bool = False,
) -> Array:
    """Return magnetic-BZ k vectors in the common finite-B flattening order."""

    flux_obj = flux if isinstance(flux, MagneticFlux) else MagneticFlux.from_value(flux)
    q = int(flux_obj.q)
    nq = int(nq)
    if nq <= 0:
        raise ValueError(f"nq must be positive, got {nq}")
    fine = np.arange(nq, dtype=float) / float(nq * q)
    if reduced_translation:
        k = fine.reshape(nq, 1) * g1 + fine.reshape(1, nq) * g2
        return np.asarray(k.reshape(-1, order="F"), dtype=np.complex128)
    strips = np.arange(q, dtype=float) / float(q)
    k = strips.reshape(q, 1, 1) * g1 + fine.reshape(1, nq, 1) * g1 + fine.reshape(1, 1, nq) * g2
    return np.asarray(k.reshape(-1, order="F"), dtype=np.complex128)


def magnetic_normalization_count(flux: MagneticFlux | Fraction | tuple[int, int] | str, nq: int) -> int:
    """Return the full magnetic-mesh normalization count ``(q*nq)^2``."""

    flux_obj = flux if isinstance(flux, MagneticFlux) else MagneticFlux.from_value(flux)
    return int(flux_obj.q * flux_obj.q * int(nq) * int(nq))


def diophantine_filling(s: int, t: int, flux: MagneticFlux | Fraction | tuple[int, int] | str) -> float:
    """Return the Streda/Diophantine filling ``nu = s + t*phi/phi0``."""

    flux_obj = flux if isinstance(flux, MagneticFlux) else MagneticFlux.from_value(flux)
    return float(int(s) + int(t) * flux_obj.ratio)


def diophantine_branch_cases(
    s: int,
    t: int,
    *,
    fluxes: Sequence[MagneticFlux | Fraction | tuple[int, int] | str],
) -> tuple[tuple[MagneticFlux, float], ...]:
    """Return ``(flux, nu)`` cases for one Streda/Diophantine branch."""

    selected_fluxes = tuple(flux if isinstance(flux, MagneticFlux) else MagneticFlux.from_value(flux) for flux in fluxes)
    return tuple((flux, diophantine_filling(s, t, flux)) for flux in selected_fluxes)


__all__ = [
    "MagneticFlux",
    "choose_magnetic_nq",
    "diophantine_branch_cases",
    "diophantine_filling",
    "in_hex_shell",
    "magnetic_k_vectors",
    "magnetic_normalization_count",
    "magnetic_orbit_indices",
    "magnetic_r_orbit_positions",
    "magnetic_reciprocal_vector",
    "magnetic_shell_shifts",
]
