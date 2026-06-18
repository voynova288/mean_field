"""Commensurate HTQG angle geometry from Fujimoto et al. Appendix A.

The appendix parameterizes commensurate supermoire geometry by four
integers ``(n12, m12, n23, m23)``.  They define supermoire lattice vectors
for the 12/23 and 23/34 moire pairs as integer combinations of the adjacent
moire lattice vectors, then impose equality of the two supermoire unit cells.
Solving this condition gives the adjacent-layer twist angles.

This module exposes the paper's rounded checkpoint
``(n12, m12, n23, m23) = (8, 7, 8, 8) -> (2.13, 2.27, 2.13) degrees``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

FUJIMOTO_2025_FIG2_INTEGERS: tuple[int, int, int, int] = (8, 7, 8, 8)
FUJIMOTO_2025_FIG2_REPORTED_DEG: tuple[float, float, float] = (2.13, 2.27, 2.13)


@dataclass(frozen=True)
class HTQGCommensurateGeometry:
    """Commensurate HTQG supermoire geometry determined by four integers.

    Attributes use the paper labels ``12`` and ``23`` for the two adjacent
    moire lattices entering the supermoire construction.  The returned twist
    angles follow the positive adjacent-twist convention used in the main text
    for Fig. 2: ``(theta_12, theta_23, theta_34)`` are positive magnitudes.
    The signed helper :func:`theta_function_rad` is also available when the raw
    orientation of the integer pair is needed.
    """

    n12: int
    m12: int
    n23: int
    m23: int
    theta12_deg: float
    theta23_deg: float
    theta34_deg: float
    theta12_rad: float
    theta23_rad: float
    theta34_rad: float
    supermoire_period_factor_12: float
    supermoire_period_factor_23: float

    @property
    def integers(self) -> tuple[int, int, int, int]:
        return (int(self.n12), int(self.m12), int(self.n23), int(self.m23))

    @property
    def twist_angles_deg(self) -> tuple[float, float, float]:
        return (float(self.theta12_deg), float(self.theta23_deg), float(self.theta34_deg))

    @property
    def twist_angles_rad(self) -> tuple[float, float, float]:
        return (float(self.theta12_rad), float(self.theta23_rad), float(self.theta34_rad))

    def to_dict(self) -> dict[str, object]:
        return {
            "n12": int(self.n12),
            "m12": int(self.m12),
            "n23": int(self.n23),
            "m23": int(self.m23),
            "theta12_deg": float(self.theta12_deg),
            "theta23_deg": float(self.theta23_deg),
            "theta34_deg": float(self.theta34_deg),
            "theta12_rad": float(self.theta12_rad),
            "theta23_rad": float(self.theta23_rad),
            "theta34_rad": float(self.theta34_rad),
            "supermoire_period_factor_12": float(self.supermoire_period_factor_12),
            "supermoire_period_factor_23": float(self.supermoire_period_factor_23),
        }


def _validate_pair(n: int, m: int, *, name: str) -> tuple[int, int]:
    n_int = int(n)
    m_int = int(m)
    if n_int != n or m_int != m:
        raise ValueError(f"{name} integers must be exact integers, got {(n, m)!r}")
    if n_int == 0 and m_int == 0:
        raise ValueError(f"{name} integer pair must not be (0, 0)")
    return n_int, m_int


def triangular_norm_squared(n: int, m: int) -> int:
    """Return ``n^2 + n m + m^2`` for a triangular-lattice integer vector."""

    n_int, m_int = _validate_pair(n, m, name="triangular vector")
    return int(n_int * n_int + n_int * m_int + m_int * m_int)


def supermoire_period_factor(n: int, m: int) -> float:
    """Return the period multiplier ``sqrt(n^2 + n m + m^2)``."""

    return math.sqrt(float(triangular_norm_squared(n, m)))


def theta_function_rad(n12: int, m12: int, n23: int, m23: int) -> float:
    """Return the signed angle function entering Appendix A Eq. (A6).

    This implements the positive-angle convention used by the paper's Fig. 2
    checkpoint.  In formulas copied from PDF text extraction, the numerator can
    appear with the opposite sign depending on the rotation-matrix convention;
    the convention here is fixed by the explicit paper statement
    ``(8, 7, 8, 8) -> theta12 = +2.13 deg``.
    """

    n12, m12 = _validate_pair(n12, m12, name="(n12, m12)")
    n23, m23 = _validate_pair(n23, m23, name="(n23, m23)")
    numerator = math.sqrt(3.0) * ((2 * n12 + m12) * m23 - m12 * (2 * n23 + m23))
    denominator = (
        (2 * n12 + m12) * (2 * n23 + m23)
        + 3 * m12 * m23
        + (2 * n23 + m23) ** 2
        + 3 * m23**2
    )
    if denominator == 0:
        raise ValueError("Commensurate-angle denominator vanished; check integer inputs.")
    return float(2.0 * math.atan2(numerator, denominator))


def theta_function_deg(n12: int, m12: int, n23: int, m23: int) -> float:
    """Return :func:`theta_function_rad` in degrees."""

    return math.degrees(theta_function_rad(n12, m12, n23, m23))


def commensurate_twist_angles_rad(
    n12: int,
    m12: int,
    n23: int,
    m23: int,
    *,
    positive: bool = True,
) -> tuple[float, float, float]:
    """Return ``(theta12, theta23, theta34)`` from the four appendix integers.

    Appendix A Eq. (A6) has

    ``theta12 = theta34 = theta(n12, m12, n23, m23)`` and
    ``theta23 = -theta(n23, m23, n12, m12)``.

    With ``positive=True`` (default), the returned adjacent twist angles are
    positive magnitudes, matching the main-text/reporting convention.
    """

    theta12 = theta_function_rad(n12, m12, n23, m23)
    theta23 = -theta_function_rad(n23, m23, n12, m12)
    theta34 = theta12
    if positive:
        return (abs(float(theta12)), abs(float(theta23)), abs(float(theta34)))
    return (float(theta12), float(theta23), float(theta34))


def commensurate_twist_angles_deg(
    n12: int,
    m12: int,
    n23: int,
    m23: int,
    *,
    positive: bool = True,
) -> tuple[float, float, float]:
    """Return ``(theta12, theta23, theta34)`` in degrees."""

    return tuple(math.degrees(angle) for angle in commensurate_twist_angles_rad(n12, m12, n23, m23, positive=positive))


def build_commensurate_geometry(n12: int, m12: int, n23: int, m23: int) -> HTQGCommensurateGeometry:
    """Build a self-describing commensurate geometry record."""

    n12, m12 = _validate_pair(n12, m12, name="(n12, m12)")
    n23, m23 = _validate_pair(n23, m23, name="(n23, m23)")
    theta_rad = commensurate_twist_angles_rad(n12, m12, n23, m23, positive=True)
    theta_deg = tuple(math.degrees(angle) for angle in theta_rad)
    return HTQGCommensurateGeometry(
        n12=n12,
        m12=m12,
        n23=n23,
        m23=m23,
        theta12_deg=float(theta_deg[0]),
        theta23_deg=float(theta_deg[1]),
        theta34_deg=float(theta_deg[2]),
        theta12_rad=float(theta_rad[0]),
        theta23_rad=float(theta_rad[1]),
        theta34_rad=float(theta_rad[2]),
        supermoire_period_factor_12=supermoire_period_factor(n12, m12),
        supermoire_period_factor_23=supermoire_period_factor(n23, m23),
    )


def fujimoto_2025_fig2_checkpoint(*, atol_deg: float = 5.0e-3) -> bool:
    """Check the Appendix A/Fig. 2 integer example against the paper values.

    The paper reports the angles rounded to two decimals:
    ``(8, 7, 8, 8) -> (2.13, 2.27, 2.13)`` degrees.  The exact values from
    Eq. (A6)-(A7) in this convention are approximately
    ``(2.1339297, 2.2745253, 2.1339297)`` degrees.
    """

    computed = commensurate_twist_angles_deg(*FUJIMOTO_2025_FIG2_INTEGERS)
    return all(
        abs(float(value) - float(target)) <= float(atol_deg)
        for value, target in zip(computed, FUJIMOTO_2025_FIG2_REPORTED_DEG, strict=True)
    )


__all__ = [
    "FUJIMOTO_2025_FIG2_INTEGERS",
    "FUJIMOTO_2025_FIG2_REPORTED_DEG",
    "HTQGCommensurateGeometry",
    "build_commensurate_geometry",
    "commensurate_twist_angles_deg",
    "commensurate_twist_angles_rad",
    "fujimoto_2025_fig2_checkpoint",
    "supermoire_period_factor",
    "theta_function_deg",
    "theta_function_rad",
    "triangular_norm_squared",
]
