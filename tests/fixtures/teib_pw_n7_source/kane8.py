"""Plane-wave eight-band Kane Hamiltonian for an InAs/GaSb quantum well."""

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .materials import HBAR2_OVER_2M0_EV_NM2, MATERIALS, KaneMaterial


FOURIER_PARAMETERS = (
    "Eg",
    "Ev",
    "delta_so",
    "Ac",
    "gamma1",
    "gamma2",
    "gamma3",
    "epsilon_r",
    "EP",
    "P0",
)

M_SIGN_CONVENTION = (
    "M uses kx gamma2 kx minus sign ky gamma2 ky. This is the conventional "
    "anisotropic form and consciously corrects the supplement's likely "
    "plus-sign typo."
)


@dataclass(frozen=True)
class Layer:
    material: KaneMaterial
    start_nm: float
    stop_nm: float

    def __post_init__(self) -> None:
        if self.stop_nm <= self.start_nm:
            raise ValueError("layer stop must be greater than layer start")

    @property
    def width_nm(self) -> float:
        return self.stop_nm - self.start_nm


@dataclass(frozen=True)
class PeriodicProfile:
    """Periodic, piecewise-constant material profile along the growth axis."""

    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("profile must contain at least one layer")
        if self.layers[0].start_nm != 0.0:
            raise ValueError("profile must start at z=0")
        for left, right in zip(self.layers, self.layers[1:]):
            if left.stop_nm != right.start_nm:
                raise ValueError("profile layers must be contiguous")

    @property
    def period_nm(self) -> float:
        return self.layers[-1].stop_nm

    def material_at(self, z_nm: float) -> KaneMaterial:
        z_periodic = float(z_nm) % self.period_nm
        for layer in self.layers:
            if layer.start_nm <= z_periodic < layer.stop_nm:
                return layer.material
        raise RuntimeError("periodic coordinate did not map to a layer")

    def fourier_coefficient(self, parameter: str, harmonic: int) -> complex:
        """Return ``<exp(iG_m z)|parameter|exp(iG_n z)>`` coefficient.

        The returned coefficient uses harmonic ``m-n`` and the convention
        ``f_q = integral f(z) exp(-i G_q z) dz / period``.
        """

        if parameter not in FOURIER_PARAMETERS:
            raise KeyError(f"unknown material parameter: {parameter}")
        if harmonic == 0:
            return sum(
                layer.width_nm * getattr(layer.material, parameter)
                for layer in self.layers
            ) / self.period_nm

        wavevector = 2.0 * np.pi * harmonic / self.period_nm
        coefficient = 0.0j
        for layer in self.layers:
            phase_start = np.exp(-1j * wavevector * layer.start_nm)
            phase_stop = np.exp(-1j * wavevector * layer.stop_nm)
            coefficient += (
                getattr(layer.material, parameter)
                * (phase_start - phase_stop)
                / (1j * wavevector * self.period_nm)
            )
        return coefficient

    def fourier_matrix(self, parameter: str, N: int) -> np.ndarray:
        """Lift a scalar material parameter to the basis ``m=-N,...,N``."""

        indices = plane_wave_indices(N)
        matrix = np.empty((indices.size, indices.size), dtype=complex)
        for row, m in enumerate(indices):
            for column, n in enumerate(indices):
                matrix[row, column] = self.fourier_coefficient(
                    parameter, int(m - n)
                )
        return matrix

    @lru_cache(maxsize=32)
    def parameter_matrices(self, N: int) -> Mapping[str, np.ndarray]:
        """Return immutable parameter matrices from a 32-entry LRU cache."""

        matrices = {}
        for parameter in FOURIER_PARAMETERS:
            matrix = self.fourier_matrix(parameter, N)
            frozen = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype)
            matrices[parameter] = frozen.reshape(matrix.shape)
        return MappingProxyType(matrices)


def default_heterostructure(barrier_nm: float = 15.0) -> PeriodicProfile:
    """Return AlSb / 11.5 nm InAs / 8.0 nm GaSb / AlSb."""

    if barrier_nm <= 0.0:
        raise ValueError("barrier_nm must be positive")
    inas_start = float(barrier_nm)
    gasb_start = inas_start + 11.5
    right_barrier_start = gasb_start + 8.0
    period = right_barrier_start + barrier_nm
    return PeriodicProfile(
        layers=(
            Layer(MATERIALS["AlSb"], 0.0, inas_start),
            Layer(MATERIALS["InAs"], inas_start, gasb_start),
            Layer(MATERIALS["GaSb"], gasb_start, right_barrier_start),
            Layer(MATERIALS["AlSb"], right_barrier_start, period),
        )
    )


def plane_wave_indices(N: int) -> np.ndarray:
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N must be a non-negative integer")
    return np.arange(-N, N + 1, dtype=int)


def kz_matrix(profile: PeriodicProfile, N: int, kz0: float = 0.0) -> np.ndarray:
    wavevectors = kz0 + 2.0 * np.pi * plane_wave_indices(N) / profile.period_nm
    return np.diag(wavevectors.astype(complex))


def _symmetrized(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left @ right + right @ left) / 2.0


def kane_operators(
    profile: PeriodicProfile,
    N: int,
    kx: float,
    ky: float,
    kz0: float = 0.0,
) -> dict[str, np.ndarray]:
    """Construct the scalar/operator entries of Supplementary Note 6.

    Coefficients containing two growth-direction momenta use flux ordering,
    ``kz @ coefficient @ kz``. Linear interface terms use the Hermitian half
    anticommutator, in particular ``{P0, kz}/2`` in ``U``.
    """

    params = profile.parameter_matrices(N)
    kz = kz_matrix(profile, N, kz0=kz0)
    h2 = HBAR2_OVER_2M0_EV_NM2
    k_parallel_sq = kx**2 + ky**2
    k_minus = kx - 1j * ky

    A = (
        params["Ev"]
        + params["Eg"]
        + k_parallel_sq * params["Ac"]
        + kz @ params["Ac"] @ kz
    )
    P = -params["Ev"] + h2 * (
        k_parallel_sq * params["gamma1"]
        + kz @ params["gamma1"] @ kz
    )
    Q = h2 * (
        k_parallel_sq * params["gamma2"]
        - 2.0 * kz @ params["gamma2"] @ kz
    )
    L = (
        1j
        * np.sqrt(3.0)
        * (2.0 * h2)
        * k_minus
        * _symmetrized(params["gamma3"], kz)
    )
    M = -np.sqrt(3.0) * h2 * (
        (kx**2 - ky**2) * params["gamma2"]
        - 2j * kx * ky * params["gamma3"]
    )
    U = _symmetrized(params["P0"], kz) / np.sqrt(3.0)
    V = k_minus * params["P0"] / np.sqrt(6.0)

    return {
        "A": A,
        "P": P,
        "Q": Q,
        "L": L,
        "M": M,
        "U": U,
        "V": V,
        "Delta": params["delta_so"],
        "kz": kz,
    }


def kane_hamiltonian(
    profile: PeriodicProfile,
    N: int,
    kx: float,
    ky: float,
    kz0: float = 0.0,
) -> np.ndarray:
    """Lift the exact Supplementary Note 6 phase convention to plane waves."""

    operators = kane_operators(profile, N=N, kx=kx, ky=ky, kz0=kz0)
    A = operators["A"]
    P = operators["P"]
    Q = operators["Q"]
    L = operators["L"]
    M = operators["M"]
    U = operators["U"]
    V = operators["V"]
    Delta = operators["Delta"]
    zero = np.zeros_like(A)
    dagger = lambda matrix: matrix.conj().T
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)

    return np.block(
        [
            [A, zero, 1j * sqrt3 * dagger(V), sqrt2 * U, 1j * V, zero, 1j * U, sqrt2 * V],
            [zero, A, zero, -dagger(V), 1j * sqrt2 * U, -sqrt3 * V, 1j * sqrt2 * dagger(V), -U],
            [-1j * sqrt3 * V, zero, -(P + Q), L, M, zero, 1j * L / sqrt2, -1j * sqrt2 * M],
            [sqrt2 * U, -V, dagger(L), -(P - Q), zero, M, 1j * sqrt2 * Q, 1j * sqrt3 * L / sqrt2],
            [-1j * dagger(V), -1j * sqrt2 * U, dagger(M), zero, -(P - Q), -L, -1j * sqrt3 * dagger(L) / sqrt2, 1j * sqrt2 * Q],
            [zero, -sqrt3 * dagger(V), zero, dagger(M), -dagger(L), -(P + Q), -1j * sqrt2 * dagger(M), -1j * dagger(L) / sqrt2],
            [-1j * U, -1j * sqrt2 * V, -1j * dagger(L) / sqrt2, -1j * sqrt2 * Q, 1j * sqrt3 * L / sqrt2, 1j * sqrt2 * M, -P - Delta, zero],
            [sqrt2 * dagger(V), -U, 1j * sqrt2 * dagger(M), -1j * sqrt3 * dagger(L) / sqrt2, -1j * sqrt2 * Q, 1j * L / sqrt2, zero, -P - Delta],
        ]
    )


__all__ = [
    "FOURIER_PARAMETERS",
    "Layer",
    "M_SIGN_CONVENTION",
    "PeriodicProfile",
    "default_heterostructure",
    "kane_hamiltonian",
    "kane_operators",
    "kz_matrix",
    "plane_wave_indices",
]
