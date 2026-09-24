from __future__ import annotations

import numpy as np

from ei_mf.units import HBAR2_OVER_2M0_MEV_NM2, KB_MEV_PER_K


def test_units_constants_and_density_conversion() -> None:
    assert np.isclose(5.5e10 * 1e-14, 5.5e-4)
    assert np.isclose(HBAR2_OVER_2M0_MEV_NM2, 38.0998212)
    assert np.isclose(KB_MEV_PER_K, 0.08617333262)
