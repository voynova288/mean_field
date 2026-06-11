from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.htg.mao2025 import (
    MaoHTGConfig,
    build_mao_hamiltonian,
    central_band_indices,
    make_mao_model,
    stacking_phase_pair,
    validate_analytic_dhdk,
)


def test_mao2025_stacking_phase_conventions_are_explicit() -> None:
    phi1, phi2 = stacking_phase_pair("ABA")
    assert np.isclose(phi1, 2.0 * np.pi / 3.0)
    assert np.isclose(phi2, -2.0 * np.pi / 3.0)
    assert stacking_phase_pair("AAA") == (0.0, 0.0)
    with pytest.raises(ValueError, match="Unsupported stacking"):
        stacking_phase_pair("ABC")


def test_mao2025_model_builds_massive_hermitian_hamiltonian_and_central_window() -> None:
    config = MaoHTGConfig(n_shells=1)
    model = make_mao_model(config)
    hamiltonian = build_mao_hamiltonian(0.01 + 0.02j, model, config)

    assert hamiltonian.shape == (model.matrix_dim, model.matrix_dim)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conjugate().T, atol=1.0e-12)
    assert central_band_indices(model.matrix_dim, 6) == (18, 19, 20, 21, 22, 23)


def test_mao2025_analytic_dhdk_matches_finite_difference_for_tiny_model() -> None:
    config = MaoHTGConfig(n_shells=1)
    model = make_mao_model(config)
    result = validate_analytic_dhdk(0.01 + 0.02j, model, config, step_nm_inv=1.0e-6)

    assert result.max_abs_ev_nm < 2.0e-9
