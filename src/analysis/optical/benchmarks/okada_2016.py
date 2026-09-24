"""Okada et al. 2016 QAH Faraday/Kerr formula benchmarks.

Reference: K. N. Okada et al., Nature Communications 7, 12245 (2016),
"Terahertz spectroscopy on Faraday and Kerr rotations in a quantum anomalous
Hall state", especially Eqs. (1)-(3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.optical.linear import E2_OVER_H_S
from analysis.optical.magneto import (
    MagnetoOpticalAmplitudes,
    VACUUM_IMPEDANCE_OHM,
    faraday_kerr_from_sheet_conductivity_on_substrate,
)

OKADA_2016_INP_REFRACTIVE_INDEX = 3.47
OKADA_2016_DC_ESTIMATED_FARADAY_RAD = 3.1e-3
OKADA_2016_DC_ESTIMATED_KERR_RAD = 8.7e-3
OKADA_2016_MEASURED_FARADAY_RAD = 2.6e-3
OKADA_2016_MEASURED_KERR_RAD = 6.9e-3


@dataclass(frozen=True)
class Okada2016ScalingUncertainty:
    """Okada Eq. (3) value and first-order propagated uncertainty."""

    value: np.ndarray
    standard_deviation: np.ndarray
    derivative_faraday: np.ndarray
    derivative_kerr: np.ndarray


@dataclass(frozen=True)
class Okada2016QAHBenchmark:
    """Quantized-QAH sheet prediction in the Okada substrate geometry."""

    sheet_conductivity: np.ndarray
    angles: MagnetoOpticalAmplitudes
    fine_structure_constant: float
    scaling_function: float


def okada2016_qah_sheet_conductivity(
    *,
    longitudinal_e2_over_h: complex | float = 0.0,
    hall_e2_over_h: float = 1.0,
    conductance_quantum_siemens: float = E2_OVER_H_S,
) -> np.ndarray:
    """Return the isotropic QAH sheet tensor in the physical Hall convention."""

    longitudinal = complex(longitudinal_e2_over_h) * float(conductance_quantum_siemens)
    hall = float(hall_e2_over_h) * float(conductance_quantum_siemens)
    sigma = np.zeros((2, 2), dtype=np.complex128)
    sigma[0, 0] = longitudinal
    sigma[1, 1] = longitudinal
    sigma[0, 1] = hall
    sigma[1, 0] = -hall
    return sigma


def okada2016_scaling_function(
    faraday_angle_rad: np.ndarray | float,
    kerr_angle_rad: np.ndarray | float,
) -> np.ndarray:
    """Evaluate Okada Eq. (3) in a cancellation-resistant tangent form."""

    theta_f = np.asarray(faraday_angle_rad, dtype=float)
    theta_k = np.asarray(kerr_angle_rad, dtype=float)
    p = np.tan(theta_f)
    k = np.tan(theta_k)
    denominator = k - 2.0 * p - p * p * k
    with np.errstate(divide="ignore", invalid="ignore"):
        return p * (k - p) / denominator


def okada2016_scaling_from_dc_conductivity(
    longitudinal_siemens: np.ndarray | float,
    hall_siemens: np.ndarray | float,
    *,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> np.ndarray:
    """Evaluate the dc form of Eq. (3) directly from real sheet conductance.

    For the isotropic Hall tensor this exact identity is independent of the
    substrate refractive index:
    ``f=Z0*sigma_xy/[2*(1+Z0*sigma_xx)]``.
    """

    longitudinal = np.asarray(longitudinal_siemens, dtype=float)
    hall = np.asarray(hall_siemens, dtype=float)
    z0 = float(vacuum_impedance_ohm)
    return z0 * hall / (2.0 * (1.0 + z0 * longitudinal))


def okada2016_scaling_uncertainty(
    faraday_angle_rad: np.ndarray | float,
    kerr_angle_rad: np.ndarray | float,
    faraday_standard_deviation_rad: np.ndarray | float,
    kerr_standard_deviation_rad: np.ndarray | float,
    *,
    covariance_rad2: np.ndarray | float = 0.0,
) -> Okada2016ScalingUncertainty:
    """Propagate angle uncertainty through Okada Eq. (3) to first order.

    The paper does not report the Faraday/Kerr covariance. Callers must pass it
    when known; the default makes the explicit independence assumption.
    """

    theta_f = np.asarray(faraday_angle_rad, dtype=float)
    theta_k = np.asarray(kerr_angle_rad, dtype=float)
    std_f = np.asarray(faraday_standard_deviation_rad, dtype=float)
    std_k = np.asarray(kerr_standard_deviation_rad, dtype=float)
    covariance = np.asarray(covariance_rad2, dtype=float)
    if np.any(~np.isfinite(std_f)) or np.any(~np.isfinite(std_k)):
        raise ValueError("angle standard deviations must be finite")
    if np.any(std_f < 0.0) or np.any(std_k < 0.0):
        raise ValueError("angle standard deviations must be non-negative")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("angle covariance must be finite")
    covariance_bound = std_f * std_k
    tolerance = 32.0 * np.finfo(float).eps * np.maximum(covariance_bound, 1.0e-300)
    if np.any(np.abs(covariance) > covariance_bound + tolerance):
        raise ValueError("angle covariance is inconsistent with a positive-semidefinite covariance matrix")
    p = np.tan(theta_f)
    k = np.tan(theta_k)
    denominator = k - 2.0 * p - p * p * k
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative_f = (
            (1.0 + p * p)
            * (k * k - 2.0 * p * k + 2.0 * p * p + p * p * k * k)
            / (denominator * denominator)
        )
        derivative_k = (
            -p * p * (1.0 + p * p) * (1.0 + k * k)
            / (denominator * denominator)
        )
        variance = (
            (derivative_f * std_f) ** 2
            + (derivative_k * std_k) ** 2
            + 2.0 * derivative_f * derivative_k * covariance
        )
    variance = np.maximum(variance, 0.0)
    return Okada2016ScalingUncertainty(
        value=okada2016_scaling_function(theta_f, theta_k),
        standard_deviation=np.sqrt(variance),
        derivative_faraday=derivative_f,
        derivative_kerr=derivative_k,
    )


def okada2016_qah_benchmark(
    *,
    substrate_refractive_index: float = OKADA_2016_INP_REFRACTIVE_INDEX,
    longitudinal_e2_over_h: complex | float = 0.0,
    hall_e2_over_h: float = 1.0,
    vacuum_impedance_ohm: float = VACUUM_IMPEDANCE_OHM,
) -> Okada2016QAHBenchmark:
    """Return the ideal low-frequency QAH benchmark for Okada Eqs. (1)-(3).

    For ``n_s=3.47``, ``sigma_xx=0``, and ``sigma_xy=e^2/h``, the ideal
    prediction is approximately 3.265 mrad Faraday and 9.174 mrad Kerr. The
    paper quotes 3.1 and 8.7 mrad from its measured, slightly non-quantized dc
    conductances, so those experimental numbers are comparison anchors rather
    than exact targets for this ideal helper.
    """

    sigma = okada2016_qah_sheet_conductivity(
        longitudinal_e2_over_h=longitudinal_e2_over_h,
        hall_e2_over_h=hall_e2_over_h,
    )
    angles = faraday_kerr_from_sheet_conductivity_on_substrate(
        sigma,
        substrate_refractive_index=float(substrate_refractive_index),
        vacuum_impedance_ohm=float(vacuum_impedance_ohm),
    )
    alpha = 0.5 * float(vacuum_impedance_ohm) * E2_OVER_H_S
    scaling = float(okada2016_scaling_function(angles.faraday_angle_rad, angles.kerr_angle_rad))
    return Okada2016QAHBenchmark(
        sheet_conductivity=sigma,
        angles=angles,
        fine_structure_constant=alpha,
        scaling_function=scaling,
    )


__all__ = [
    "OKADA_2016_INP_REFRACTIVE_INDEX",
    "OKADA_2016_DC_ESTIMATED_FARADAY_RAD",
    "OKADA_2016_DC_ESTIMATED_KERR_RAD",
    "OKADA_2016_MEASURED_FARADAY_RAD",
    "OKADA_2016_MEASURED_KERR_RAD",
    "Okada2016QAHBenchmark",
    "Okada2016ScalingUncertainty",
    "okada2016_qah_benchmark",
    "okada2016_qah_sheet_conductivity",
    "okada2016_scaling_from_dc_conductivity",
    "okada2016_scaling_function",
    "okada2016_scaling_uncertainty",
]
