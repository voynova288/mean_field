"""Analytical graphene THG benchmark with scattering-term relaxation.

This module implements Mikhailov, Phys. Rev. B 93, 085403 (2016),
Eqs. (59)-(78), specialized to equal-input ``xxxx`` third-harmonic
conductivity.  Passos et al. use this result for their Figs. 1 and 2.

The relaxation convention is deliberately distinct from adiabatic switching:
every cumulative density-matrix pole contains one ``+i*gamma``.  In terms of
``Omega = hbar*omega/E_F`` and ``Gamma = gamma/E_F`` this gives

``O_1=(Omega+i*Gamma)/2``, ``O_12=(2*Omega+i*Gamma)/2``, and
``O_123=(3*Omega+i*Gamma)/2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import operator
from typing import Literal

import numpy as np


IntegralMethod = Literal["closed_form", "gauss_legendre"]
IntegralTuple = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True, kw_only=True, eq=False)
class Mikhailov2016THGResult:
    """Dimensionless ``sigma_xxxx^(3)/sigma_0`` and its four sectors."""

    omega_over_fermi_energy: np.ndarray
    gamma_over_fermi_energy: float
    s_30: np.ndarray
    s_21: np.ndarray
    s_12: np.ndarray
    s_03: np.ndarray
    total: np.ndarray
    integral_method: IntegralMethod
    quadrature_order: int
    coincident_argument_fallback_count: int = 0
    quadrature_max_absolute_change: float = 0.0
    quadrature_max_relative_change: float = 0.0

    def __post_init__(self) -> None:
        omega = np.array(self.omega_over_fermi_energy, dtype=float, copy=True)
        if omega.ndim != 1 or omega.size == 0:
            raise ValueError("omega_over_fermi_energy must be a nonempty 1D array")
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
            raise ValueError("omega_over_fermi_energy must be finite and positive")
        arrays: dict[str, np.ndarray] = {}
        for name in ("s_30", "s_21", "s_12", "s_03", "total"):
            value = np.array(getattr(self, name), dtype=complex, copy=True)
            if value.shape != omega.shape:
                raise ValueError(f"{name} must have shape {omega.shape}")
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value.setflags(write=False)
            arrays[name] = value
        if isinstance(self.gamma_over_fermi_energy, (bool, np.bool_)):
            raise TypeError("gamma_over_fermi_energy must be a real scalar, not bool")
        gamma = float(self.gamma_over_fermi_energy)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma_over_fermi_energy must be finite and positive")
        method = str(self.integral_method)
        if method not in {"closed_form", "gauss_legendre"}:
            raise ValueError("integral_method must be 'closed_form' or 'gauss_legendre'")
        order = _positive_integer(self.quadrature_order, name="quadrature_order")
        if order < 32:
            raise ValueError("quadrature_order must be at least 32")
        fallback_count = _nonnegative_integer(
            self.coincident_argument_fallback_count,
            name="coincident_argument_fallback_count",
        )
        if fallback_count > omega.size:
            raise ValueError("coincident_argument_fallback_count exceeds grid size")
        maximum_absolute_change = float(self.quadrature_max_absolute_change)
        maximum_relative_change = float(self.quadrature_max_relative_change)
        if (
            not np.isfinite(maximum_absolute_change)
            or maximum_absolute_change < 0.0
            or not np.isfinite(maximum_relative_change)
            or maximum_relative_change < 0.0
        ):
            raise ValueError("quadrature convergence changes must be finite and non-negative")
        sector_sum = arrays["s_30"] + arrays["s_21"] + arrays["s_12"] + arrays["s_03"]
        if not np.allclose(arrays["total"], sector_sum, rtol=2.0e-13, atol=2.0e-13):
            raise ValueError("total must equal the four sector contributions")
        omega.setflags(write=False)
        object.__setattr__(self, "omega_over_fermi_energy", omega)
        object.__setattr__(self, "gamma_over_fermi_energy", gamma)
        object.__setattr__(self, "integral_method", method)
        object.__setattr__(self, "quadrature_order", order)
        object.__setattr__(self, "coincident_argument_fallback_count", fallback_count)
        object.__setattr__(self, "quadrature_max_absolute_change", maximum_absolute_change)
        object.__setattr__(self, "quadrature_max_relative_change", maximum_relative_change)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def evaluation_method(self) -> str:
        if self.coincident_argument_fallback_count:
            return "closed_form_with_converged_quadrature_limits"
        return self.integral_method


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _positive_integer(value: object, *, name: str) -> int:
    integer = _nonnegative_integer(value, name=name)
    if integer == 0:
        raise ValueError(f"{name} must be positive")
    return integer


@lru_cache(maxsize=8)
def _one_to_infinity_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights for ``integral_1^infinity dx``."""

    legendre_x, legendre_weight = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (legendre_x + 1.0)
    one_minus_unit = 1.0 - unit
    x = 1.0 + unit / one_minus_unit
    weight = 0.5 * legendre_weight / one_minus_unit**2
    x.setflags(write=False)
    weight.setflags(write=False)
    return x, weight


def _mikhailov_integrals_quadrature(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    quadrature_order: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Mikhailov Eqs. (69)-(73) from their defining integrals."""

    x, weight = _one_to_infinity_quadrature(quadrature_order)
    outputs = [np.empty(a.shape, dtype=complex) for _ in range(5)]
    for start in range(0, a.size, chunk_size):
        stop = min(start + chunk_size, a.size)
        aa = a[start:stop][None, :]
        bb = b[start:stop][None, :]
        cc = c[start:stop][None, :]
        xx = x[:, None]
        ww = weight[:, None]

        xpa = xx + aa
        xpb = xx + bb
        xpc = xx + cc
        common = xpa * xpb * xpc
        j1_integrand = (
            1.0 / xx**2
            + 1.0 / (xx * xpc)
            - 1.0 / (xpb * xpc)
            - 2.0 / xpc**2
        ) / common
        j2_integrand = 1.0 / (xx * xpa * xpb**2 * xpc)
        j3_integrand = 1.0 / (xx**2 * xpa * xpb)
        j4_integrand = 1.0 / (xx * xpa * xpb**2)
        j5_integrand = (
            (1.0 / xpa - 1.0 / (xx - aa))
            * (1.0 / xpc - 1.0 / (xx - cc))
            / xx**2
        )
        for output, integrand in zip(
            outputs,
            (j1_integrand, j2_integrand, j3_integrand, j4_integrand, j5_integrand),
        ):
            output[start:stop] = np.sum(ww * integrand, axis=0)
    return tuple(outputs)  # type: ignore[return-value]


def _mikhailov_integrals_closed(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Mikhailov Eqs. (74)-(78) on their principal branches."""

    log_a = np.log(1.0 + a)
    log_b = np.log(1.0 + b)
    log_c = np.log(1.0 + c)
    j1 = (
        1.0 / (a * b * c)
        + (-a * b + 2.0 * a * c + 3.0 * b * c - 4.0 * c**2)
        / (c * (c - a) ** 2 * (b - c) ** 2 * (c + 1.0))
        - 1.0 / ((a - b) * (b - c) ** 2 * (b + 1.0))
        - 1.0 / ((a - c) * (b - c) * (c + 1.0) ** 2)
        - (-a**3 - 2.0 * a**2 * c + 3.0 * a * b * c + a * c**2 - b * c**2)
        * log_a
        / (a**2 * (a - b) ** 2 * (a - c) ** 3)
        - (-2.0 * a * b + a * c + 3.0 * b**2 - b * c)
        * log_b
        / (b**2 * (a - b) ** 2 * (b - c) ** 2)
        - (a**2 + a * b - 4.0 * a * c - 3.0 * b * c + 5.0 * c**2)
        * log_c
        / (c * (a - c) ** 3 * (c - b) ** 2)
    )
    j2 = (
        1.0 / (b * (a - b) * (b - c) * (b + 1.0))
        - log_a / (a * (a - b) ** 2 * (a - c))
        - (2.0 * a * b - a * c - 3.0 * b**2 + 2.0 * b * c)
        * log_b
        / (b**2 * (a - b) ** 2 * (b - c) ** 2)
        - log_c / (c * (c - a) * (b - c) ** 2)
    )
    j3 = 1.0 / (a * b) + log_a / (a**2 * (a - b)) - log_b / (
        b**2 * (a - b)
    )
    j4 = (
        -1.0 / (b * (a - b) * (1.0 + b))
        + log_a / (a * (a - b) ** 2)
        + (a - 2.0 * b) * log_b / (b**2 * (a - b) ** 2)
    )
    j5 = (
        4.0 / (a * c)
        - 2.0
        * c
        * np.log((1.0 - a) / (1.0 + a))
        / (a**2 * (a - c) * (a + c))
        + 2.0
        * a
        * np.log((1.0 - c) / (1.0 + c))
        / (c**2 * (a - c) * (a + c))
    )
    return j1, j2, j3, j4, j5


def _mikhailov_integrals(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    integral_method: IntegralMethod,
    quadrature_order: int,
    coincident_argument_mask: np.ndarray | None = None,
) -> tuple[IntegralTuple, float, float]:
    if integral_method == "gauss_legendre":
        values = _mikhailov_integrals_quadrature(
            a,
            b,
            c,
            quadrature_order=quadrature_order,
        )
        return values, 0.0, 0.0
    if coincident_argument_mask is None or not np.any(coincident_argument_mask):
        return _mikhailov_integrals_closed(a, b, c), 0.0, 0.0

    mask = np.asarray(coincident_argument_mask, dtype=bool)
    if mask.shape != a.shape:
        raise ValueError("coincident_argument_mask must match the frequency grid")
    outputs = [np.empty(a.shape, dtype=complex) for _ in range(5)]
    if np.any(~mask):
        closed = _mikhailov_integrals_closed(a[~mask], b[~mask], c[~mask])
        for output, values in zip(outputs, closed):
            output[~mask] = values
    lower_order = max(32, quadrature_order // 2)
    lower = _mikhailov_integrals_quadrature(
        a[mask],
        b[mask],
        c[mask],
        quadrature_order=lower_order,
    )
    integrated = _mikhailov_integrals_quadrature(
        a[mask],
        b[mask],
        c[mask],
        quadrature_order=quadrature_order,
    )
    maximum_absolute_change = 0.0
    maximum_relative_change = 0.0
    for integral_index, (coarse, converged) in enumerate(zip(lower, integrated), start=1):
        change = np.abs(converged - coarse)
        relative = change / np.maximum(np.abs(converged), 1.0e-300)
        maximum_absolute_change = max(maximum_absolute_change, float(np.max(change)))
        maximum_relative_change = max(maximum_relative_change, float(np.max(relative)))
        if not np.allclose(converged, coarse, rtol=2.0e-9, atol=2.0e-11):
            raise RuntimeError(
                "Mikhailov coincident-argument quadrature did not converge for "
                f"J{integral_index} between orders {lower_order} and "
                f"{quadrature_order}; increase quadrature_order"
            )
    for output, values in zip(outputs, integrated):
        output[mask] = values
    return (
        tuple(outputs),  # type: ignore[arg-type]
        maximum_absolute_change,
        maximum_relative_change,
    )


def _sector_21(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    positive_integrals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    negative_integrals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    j1, j2, _, _, _ = positive_integrals
    j1_negative, j2_negative, _, _, _ = negative_integrals

    def expression(
        aa: np.ndarray,
        bb: np.ndarray,
        cc: np.ndarray,
        jj1: np.ndarray,
        jj2: np.ndarray,
    ) -> np.ndarray:
        return (
            (1.0 + 0.75 * cc) / (aa * bb * (1.0 + cc) ** 2)
            + 0.25 / (aa * (1.0 + bb) * (1.0 + cc))
            + 0.25 / (aa * (1.0 + bb) * (1.0 + cc) ** 2)
            - 0.25 * jj1
            + 0.25 * jj2
        )

    positive = expression(a, b, c, j1, j2)
    negative = expression(-a, -b, -c, j1_negative, j2_negative)
    return -0.25j * (positive - negative)


def _sector_12(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    positive_integrals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    negative_integrals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    _, _, j3, j4, _ = positive_integrals
    _, _, j3_negative, j4_negative, _ = negative_integrals

    def expression(
        aa: np.ndarray,
        bb: np.ndarray,
        cc: np.ndarray,
        jj3: np.ndarray,
        jj4: np.ndarray,
    ) -> np.ndarray:
        return (
            0.75 * np.log(1.0 + aa) / (aa**2 * cc * bb)
            - 0.75 / (cc * bb * aa)
            + 0.25 / (cc * aa * (1.0 + bb))
            - 0.25 * jj3 / cc
            + 0.25 * jj4 / cc
        )

    positive = expression(a, b, c, j3, j4)
    negative = expression(-a, -b, -c, j3_negative, j4_negative)
    return 0.25j * (positive - negative)


def mikhailov2016_thg_xxxx_scattering(
    omega_over_fermi_energy: np.ndarray,
    *,
    gamma_over_fermi_energy: float,
    integral_method: IntegralMethod = "closed_form",
    quadrature_order: int = 512,
) -> Mikhailov2016THGResult:
    """Return Mikhailov's scattering-term graphene THG benchmark.

    Parameters are dimensionless: ``Omega=hbar*omega/E_F`` and
    ``Gamma=gamma/E_F``.  The output is normalized by Mikhailov Eq. (61),
    which equals the Passos graphene ``sigma_0`` normalization.
    """

    omega = np.asarray(omega_over_fermi_energy, dtype=float)
    if omega.ndim != 1 or omega.size == 0:
        raise ValueError("omega_over_fermi_energy must be a nonempty 1D array")
    if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("omega_over_fermi_energy must be finite and positive")
    if isinstance(gamma_over_fermi_energy, (bool, np.bool_)):
        raise TypeError("gamma_over_fermi_energy must be a real scalar, not bool")
    gamma = float(gamma_over_fermi_energy)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma_over_fermi_energy must be finite and positive")
    method = str(integral_method)
    if method not in {"closed_form", "gauss_legendre"}:
        raise ValueError("integral_method must be 'closed_form' or 'gauss_legendre'")
    order = _positive_integer(quadrature_order, name="quadrature_order")
    if order < 32:
        raise ValueError("quadrature_order must be at least 32")

    a = 0.5 * (omega + 1.0j * gamma)
    b = 0.5 * (2.0 * omega + 1.0j * gamma)
    c = 0.5 * (3.0 * omega + 1.0j * gamma)
    # Eqs. (74)-(78) have removable coincident-argument singularities as
    # Omega -> 0. Evaluate their defining integrals in that region rather
    # than subtracting large closed-form terms.
    fallback_mask = (method == "closed_form") & (omega < 0.1)
    effective_order = 2 * max(order, 512) if np.any(fallback_mask) else order
    positive_integrals, positive_absolute_change, positive_relative_change = (
        _mikhailov_integrals(
            a,
            b,
            c,
            integral_method=method,  # type: ignore[arg-type]
            quadrature_order=effective_order,
            coincident_argument_mask=fallback_mask,
        )
    )
    negative_integrals, negative_absolute_change, negative_relative_change = (
        _mikhailov_integrals(
            -a,
            -b,
            -c,
            integral_method=method,  # type: ignore[arg-type]
            quadrature_order=effective_order,
            coincident_argument_mask=fallback_mask,
        )
    )
    s_30 = 3.0j / (8.0 * c * b * a)
    s_21 = _sector_21(
        a,
        b,
        c,
        positive_integrals=positive_integrals,
        negative_integrals=negative_integrals,
    )
    s_12 = _sector_12(
        a,
        b,
        c,
        positive_integrals=positive_integrals,
        negative_integrals=negative_integrals,
    )
    _, _, _, _, j5 = positive_integrals
    s_03 = 3.0j * j5 / (32.0 * b)
    total = s_30 + s_21 + s_12 + s_03
    return Mikhailov2016THGResult(
        omega_over_fermi_energy=omega,
        gamma_over_fermi_energy=gamma,
        s_30=s_30,
        s_21=s_21,
        s_12=s_12,
        s_03=s_03,
        total=total,
        integral_method=method,  # type: ignore[arg-type]
        quadrature_order=effective_order,
        coincident_argument_fallback_count=int(np.count_nonzero(fallback_mask)),
        quadrature_max_absolute_change=max(
            positive_absolute_change,
            negative_absolute_change,
        ),
        quadrature_max_relative_change=max(
            positive_relative_change,
            negative_relative_change,
        ),
    )


__all__ = [
    "IntegralMethod",
    "Mikhailov2016THGResult",
    "mikhailov2016_thg_xxxx_scattering",
]
