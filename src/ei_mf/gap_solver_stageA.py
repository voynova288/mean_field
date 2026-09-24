"""Stage-A paper-convention EI gap solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bands import effective_kF, xi0_parabolic_mev
from .coulomb import angular_averaged_kernel
from .grids import radial_grid
from .params import EIParams
from .units import KB_MEV_PER_K

Array = NDArray[np.float64]


@dataclass(frozen=True)
class StageAResult:
    k_nm_inv: Array
    weights_nm2: Array
    xi_mev: Array
    Delta_order_mev: Array
    E_pair_mev: Array
    iterations: int
    residual_mev: float
    converged: bool
    kF_nm_inv: float

    @property
    def transport_gap_proxy_mev(self) -> float:
        return float(np.min(self.E_pair_mev))

    @property
    def Delta_order_max_mev(self) -> float:
        return float(np.max(np.abs(self.Delta_order_mev)))


def initial_delta(k_nm_inv: Array, params: EIParams, *, amplitude_mev: float = 2.0) -> Array:
    """Smooth positive seed centered near kF."""

    k = np.asarray(k_nm_inv, dtype=np.float64)
    kF = effective_kF(params)
    sigma = max(0.01, 0.35 * kF)
    return amplitude_mev * np.exp(-0.5 * ((k - kF) / sigma) ** 2) + 1e-6


def solve_stageA(
    k_nm_inv: Array,
    weights_nm2: Array,
    xi_mev: Array,
    K_ab_mev_nm2: Array,
    params: EIParams,
    *,
    initial: Array | None = None,
) -> StageAResult:
    """Solve the Stage-A gap equation in the paper convention.

    The implemented equation is

        Delta_i = sum_j K_ab(i,j) w_j Delta_j/E_j tanh(E_j/(4 kB T)).

    No factor of two is applied to ``E_pair``.
    """

    params.validate()
    k = np.asarray(k_nm_inv, dtype=np.float64)
    w = np.asarray(weights_nm2, dtype=np.float64)
    xi = np.asarray(xi_mev, dtype=np.float64)
    K = np.asarray(K_ab_mev_nm2, dtype=np.float64)
    if K.shape != (k.size, k.size):
        raise ValueError("K_ab shape does not match k grid")
    if params.pairing_scale == 0.0:
        Delta_zero = np.zeros_like(k)
        E_pair = np.sqrt(xi * xi + Delta_zero * Delta_zero)
        return StageAResult(k, w, xi, Delta_zero, E_pair, 0, 0.0, True, effective_kF(params))
    Delta = initial_delta(k, params) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    if Delta.shape != k.shape:
        raise ValueError("initial Delta shape does not match k grid")
    kBT = KB_MEV_PER_K * float(params.temperature_K)
    residual = np.inf
    converged = False
    for iteration in range(1, int(params.max_iter) + 1):
        E_pair = np.sqrt(xi * xi + Delta * Delta)
        E_safe = np.maximum(E_pair, params.energy_floor_meV)
        if params.temperature_K <= 0:
            thermal = np.ones_like(E_pair)
        else:
            # Workdoc/Supplementary Note 2 paper convention: tanh(E_pair/(4*kBT)).
            thermal = np.tanh(E_pair / (4.0 * kBT))
        rhs = K @ (w * Delta / E_safe * thermal)
        rhs *= params.pairing_scale
        Delta_new = (1.0 - params.mix) * Delta + params.mix * rhs
        if not np.all(np.isfinite(Delta_new)):
            raise FloatingPointError("non-finite Delta during Stage-A iteration")
        residual = float(np.max(np.abs(Delta_new - Delta)))
        Delta = Delta_new
        if residual < params.tol_meV:
            converged = True
            break
    E_pair = np.sqrt(xi * xi + Delta * Delta)
    return StageAResult(k, w, xi, Delta, E_pair, iteration, residual, converged, effective_kF(params))


def solve_stageA_case(
    params: EIParams,
    *,
    K_ab_mev_nm2: Array | None = None,
    initial: Array | None = None,
    block_rows: int = 16,
    kernel_progress: bool = False,
) -> tuple[StageAResult, Array]:
    """Build grid/kernel if needed and solve a complete Stage-A case."""

    params.validate()
    k, w, _edges = radial_grid(params.k_max_nm_inv, params.nk)
    K_ab = (
        angular_averaged_kernel(k, params, kind="interlayer", block_rows=block_rows, progress=kernel_progress)
        if K_ab_mev_nm2 is None
        else np.asarray(K_ab_mev_nm2, dtype=np.float64)
    )
    xi = xi0_parabolic_mev(k, params)
    result = solve_stageA(k, w, xi, K_ab, params, initial=initial)
    return result, K_ab
