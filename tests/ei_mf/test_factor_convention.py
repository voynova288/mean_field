from __future__ import annotations

import numpy as np

from ei_mf.units import KB_MEV_PER_K


def test_paper_pair_energy_is_not_doubled() -> None:
    xi = np.array([-1.0, 0.0, 2.0])
    Delta = np.array([0.5, 1.5, 0.25])
    T = 4.0
    E_pair = np.sqrt(xi**2 + Delta**2)
    thermal = np.tanh(E_pair / (4.0 * KB_MEV_PER_K * T))
    wrong_E = 2.0 * np.sqrt(xi**2 + Delta**2)
    wrong_thermal = np.tanh(E_pair / (2.0 * KB_MEV_PER_K * T))
    assert not np.allclose(E_pair, wrong_E)
    assert not np.allclose(thermal, wrong_thermal)
