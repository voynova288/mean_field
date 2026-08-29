from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.stability import leading_fixed_point_map_eigenvalues


def test_fixed_point_map_spectrum_recovers_leading_linear_eigenvalues() -> None:
    matrix = np.diag([1.2, 0.8, -0.4, 0.1])
    root = np.zeros(4)
    spectrum = leading_fixed_point_map_eigenvalues(
        lambda vector: matrix @ vector,
        root,
        count=2,
        relative_step=1.0e-6,
        tolerance=1.0e-11,
    )
    assert spectrum.converged
    assert spectrum.root_map_residual_max == 0.0
    assert np.sort(np.abs(spectrum.eigenvalues)) == pytest.approx([0.8, 1.2], rel=1.0e-9)
    assert spectrum.eigenvectors.shape == (4, 2)
    for value, vector in zip(spectrum.eigenvalues, spectrum.eigenvectors.T, strict=True):
        assert matrix @ vector == pytest.approx(value * vector, abs=1.0e-9)
    assert spectrum.spectral_radius == pytest.approx(1.2, rel=1.0e-9)


def test_fixed_point_map_spectrum_respects_tangent_sector_projector() -> None:
    matrix = np.diag([1.4, 0.9, 0.5, 0.2])
    root = np.zeros(4)

    def last_two(vector: np.ndarray) -> np.ndarray:
        projected = np.asarray(vector).copy()
        projected[:2] = 0.0
        return projected

    spectrum = leading_fixed_point_map_eigenvalues(
        lambda vector: matrix @ vector,
        root,
        count=1,
        projector=last_two,
        tolerance=1.0e-11,
    )
    assert spectrum.converged
    assert spectrum.eigenvectors.shape == (4, 1)
    assert spectrum.eigenvectors[:2, 0] == pytest.approx(0.0, abs=1.0e-10)
    assert spectrum.spectral_radius == pytest.approx(0.5, rel=1.0e-8)
