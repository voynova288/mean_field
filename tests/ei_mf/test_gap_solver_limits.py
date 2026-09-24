from __future__ import annotations

import numpy as np

from ei_mf.gap_solver_stageA import solve_stageA
from ei_mf.params import EIParams


def _toy_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = np.linspace(0.01, 0.04, 5)
    w = np.ones_like(k) / k.size
    xi = np.linspace(-0.15, 0.15, k.size)
    K = 3.0 * np.eye(k.size)
    return k, w, xi, K


def test_pairing_scale_zero_gives_zero_delta() -> None:
    k, w, xi, K = _toy_inputs()
    params = EIParams(nk=k.size, nphi=16, pairing_scale=0.0, max_iter=100)
    result = solve_stageA(k, w, xi, K, params)
    assert result.converged
    assert np.allclose(result.Delta_order_mev, 0.0)


def test_higher_temperature_reduces_toy_gap() -> None:
    k, w, xi, K = _toy_inputs()
    low = EIParams(nk=k.size, nphi=16, temperature_K=0.1, mix=0.25, max_iter=2000, tol_meV=1e-10)
    high = low.updated(temperature_K=20.0)
    low_result = solve_stageA(k, w, xi, K, low)
    high_result = solve_stageA(k, w, xi, K, high, initial=low_result.Delta_order_mev)
    assert low_result.converged
    assert high_result.converged
    assert high_result.transport_gap_proxy_mev < low_result.transport_gap_proxy_mev


def test_initial_guess_independence_for_toy_gap() -> None:
    k, w, xi, K = _toy_inputs()
    params = EIParams(nk=k.size, nphi=16, temperature_K=0.1, mix=0.25, max_iter=2000, tol_meV=1e-10)
    a = solve_stageA(k, w, xi, K, params, initial=np.full_like(k, 0.2))
    b = solve_stageA(k, w, xi, K, params, initial=np.full_like(k, 2.0))
    assert a.converged
    assert b.converged
    assert np.allclose(a.Delta_order_mev, b.Delta_order_mev, atol=1e-7)
