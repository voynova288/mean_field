from __future__ import annotations

import math

from mean_field.systems.tdbg import TDBGParameters


def test_tdbg_parameter_presets_match_koshino_defaults() -> None:
    full = TDBGParameters.full()
    minimal = TDBGParameters.minimal()
    no_corrugation = TDBGParameters.no_corrugation()

    assert full.model_name == "full"
    assert minimal.model_name == "minimal"
    assert no_corrugation.model_name == "no_corrugation"

    assert minimal.gamma3 == 0.0
    assert minimal.gamma4 == 0.0
    assert minimal.delta_prime == 0.0
    assert no_corrugation.u == no_corrugation.u_prime == 0.0975

    assert math.isclose(full.vf, 0.5253084, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(full.v3, 0.068173519785911, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(full.v4, 0.009373858970562763, rel_tol=0.0, abs_tol=1.0e-12)
