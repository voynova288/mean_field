from __future__ import annotations

import math

from mean_field.systems.tmbg import TMBGParameters


def test_tmbg_parameter_presets_match_park_2020_defaults() -> None:
    full = TMBGParameters.full()
    minimal = TMBGParameters.minimal()

    assert full.model_name == "full"
    assert minimal.model_name == "minimal"
    assert minimal.delta == 0.0
    assert minimal.t3 == 0.0
    assert minimal.t4 == 0.0

    assert math.isclose(full.omega, 0.12, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(full.omega_prime, 0.09393946666666667, rel_tol=0.0, abs_tol=1.0e-6)
    assert math.isclose(full.vf, 0.6603448878476401, rel_tol=0.01, abs_tol=0.0)
