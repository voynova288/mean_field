"""Qi-Wu-Zhang Chern-insulator helper for end-to-end optical benchmarks.

This is not a model used by Okada et al. 2016.  It is a minimal, independently
solvable two-band source of a quantized Hall sheet used to test the chain

    Hamiltonian -> interband Kubo conductivity -> Kerr/Faraday boundary solve.

The dimensionless crystal momenta cover ``[-pi, pi)^2`` and the Hamiltonian is

    H(k) = t [sin(kx) sigma_x + sin(ky) sigma_y
              + (m + cos(kx) + cos(ky)) sigma_z].
"""

from __future__ import annotations

import math

import numpy as np

from analysis.optical.kmesh import OpticalKPointData, optical_kpoint_data_from_eigensystem

_PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def qwz_hamiltonian_and_derivatives(
    kx: float,
    ky: float,
    *,
    mass: float = 1.0,
    energy_scale_ev: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``H`` and ``(dH/dkx,dH/dky)`` for the QWZ model."""

    scale = float(energy_scale_ev)
    sx = math.sin(float(kx))
    sy = math.sin(float(ky))
    cx = math.cos(float(kx))
    cy = math.cos(float(ky))
    hamiltonian = scale * (sx * _PAULI_X + sy * _PAULI_Y + (float(mass) + cx + cy) * _PAULI_Z)
    dhdk = scale * np.asarray(
        [
            cx * _PAULI_X - sx * _PAULI_Z,
            cy * _PAULI_Y - sy * _PAULI_Z,
        ],
        dtype=np.complex128,
    )
    return hamiltonian, dhdk


def qwz_lower_band_chern_number(mass: float) -> int:
    """Return the analytic lower-band Chern number away from gap closings.

    For the Hamiltonian convention in this module,

    ```text
    C_lower = -1,  -2 < m < 0
    C_lower = +1,   0 < m < 2
    C_lower =  0,  |m| > 2.
    ```
    """

    value = float(mass)
    if np.isclose(abs(value), 2.0) or np.isclose(value, 0.0):
        raise ValueError(f"QWZ gap closes at mass={value}")
    if -2.0 < value < 0.0:
        return -1
    if 0.0 < value < 2.0:
        return 1
    return 0


def qwz_lower_band_chern_dvector(
    mesh: int,
    *,
    mass: float = 1.0,
) -> float:
    """Integrate the analytic lower-band d-vector Berry curvature.

    This provides an oracle independent of eigensolver phases and the Kubo
    implementation.  Midpoint sampling avoids placing points on BZ boundaries.
    """

    n = int(mesh)
    if n <= 0:
        raise ValueError(f"mesh must be positive, got {mesh}")
    dk = 2.0 * math.pi / n
    integral = 0.0
    for ix in range(n):
        kx = -math.pi + (ix + 0.5) * dk
        sx = math.sin(kx)
        cx = math.cos(kx)
        for iy in range(n):
            ky = -math.pi + (iy + 0.5) * dk
            sy = math.sin(ky)
            cy = math.cos(ky)
            d = np.asarray([sx, sy, float(mass) + cx + cy], dtype=float)
            dx = np.asarray([cx, 0.0, -sx], dtype=float)
            dy = np.asarray([0.0, cy, -sy], dtype=float)
            omega_lower = -0.5 * float(np.dot(d, np.cross(dx, dy))) / float(np.linalg.norm(d) ** 3)
            integral += omega_lower * dk * dk
    return integral / (2.0 * math.pi)


def qwz_optical_kpoint_data(
    mesh: int,
    *,
    mass: float = 1.0,
    energy_scale_ev: float = 1.0,
) -> list[OpticalKPointData]:
    """Build midpoint-grid optical payloads with the lower band occupied."""

    n = int(mesh)
    if n <= 0:
        raise ValueError(f"mesh must be positive, got {mesh}")
    dk = 2.0 * math.pi / n
    weight = dk * dk
    payloads: list[OpticalKPointData] = []
    for ix in range(n):
        kx = -math.pi + (ix + 0.5) * dk
        for iy in range(n):
            ky = -math.pi + (iy + 0.5) * dk
            hamiltonian, dhdk = qwz_hamiltonian_and_derivatives(
                kx,
                ky,
                mass=mass,
                energy_scale_ev=energy_scale_ev,
            )
            energies, eigenvectors = np.linalg.eigh(hamiltonian)
            payloads.append(
                optical_kpoint_data_from_eigensystem(
                    energies,
                    eigenvectors,
                    dhdk,
                    weight=weight,
                    mu_ev=0.0,
                    temperature_k=0.0,
                )
            )
    return payloads


__all__ = [
    "qwz_hamiltonian_and_derivatives",
    "qwz_lower_band_chern_dvector",
    "qwz_lower_band_chern_number",
    "qwz_optical_kpoint_data",
]
