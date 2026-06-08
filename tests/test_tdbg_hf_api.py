from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mean_field.core.hf import HartreeFockProblem
from mean_field.systems.tdbg import (
    GridBandsResult,
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
    active_band_flavor_data_from_grid,
    build_tdbg_projected_hf_data,
    build_tdbg_projected_hf_problem,
    layer_sublattice_weights,
)


def test_tdbg_hf_facade_builds_common_hf_problem() -> None:
    data = build_tdbg_projected_hf_data(
        TDBGProjectedHFConfig(
            theta_deg=1.38,
            cut=1.0,
            mesh_size=1,
            paper_ud_ev=0.09,
            paper_ud_convention="minus_xi_ud_over3",
            window=TDBGProjectedWindow("two_flat"),
            filling=2,
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        )
    )

    problem = build_tdbg_projected_hf_problem(data)

    assert isinstance(problem, HartreeFockProblem)


def test_tdbg_hf_lightweight_active_band_data_uses_system_facade() -> None:
    eigenvectors = np.zeros((1, 1, 4, 1), dtype=np.complex128)
    eigenvectors[0, 0, :, 0] = 0.5
    grid = GridBandsResult(
        k_grid_frac=np.zeros((1, 1, 2), dtype=float),
        kvec=np.zeros((1, 1), dtype=np.complex128),
        energies=np.asarray([[[0.125]]], dtype=float),
        eigenvectors=eigenvectors,
    )
    lattice = SimpleNamespace()

    data = active_band_flavor_data_from_grid(
        grid,
        lattice=lattice,  # type: ignore[arg-type]
        valley=1,
        band_index=0,
        compute_topology=False,
    )

    assert data.valley == 1
    assert data.band_index == 0
    assert data.mean_energy_ev == 0.125
    assert data.topology is None
    np.testing.assert_allclose(data.layer_sublattice_weights, [0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(layer_sublattice_weights(eigenvectors[:, :, :, 0]), data.layer_sublattice_weights)
