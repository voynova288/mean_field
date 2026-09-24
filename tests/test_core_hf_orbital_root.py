from __future__ import annotations

import numpy as np

from mean_field.core.hf.orbital_root import (
    OrbitalRootEvaluation,
    run_orbital_rotation_root,
)


def test_orbital_rotation_root_recovers_complex_two_level_stationary_projector() -> None:
    phase = np.exp(0.37j)
    hamiltonian = np.asarray(
        [[-1.0, 0.0], [0.0, 1.0]], dtype=np.complex128
    )[:, :, None]
    angle = 0.2
    occupied = np.asarray(
        [np.cos(angle), phase * np.sin(angle)], dtype=np.complex128
    )
    initial = np.outer(occupied, occupied.conj())[:, :, None]

    run = run_orbital_rotation_root(
        initial,
        lambda projector: OrbitalRootEvaluation(
            hamiltonian=hamiltonian,
            payload=float(np.trace(projector[:, :, 0]).real),
        ),
        occupied_per_k=1,
        residual_tolerance=1.0e-12,
        max_iter=20,
    )

    expected = np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None]
    assert run.converged
    assert run.residual_max_abs <= 1.0e-12
    np.testing.assert_allclose(run.projector, expected, atol=1.0e-10)
    np.testing.assert_allclose(
        run.projector[:, :, 0] @ run.projector[:, :, 0],
        run.projector[:, :, 0],
        atol=1.0e-12,
    )
