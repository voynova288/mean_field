from __future__ import annotations

import numpy as np

from mean_field.systems.atmg import ATMGModel, ATMGParameters, validate_physics


def test_atmg_model_exposes_standard_builders_and_mapped_bands() -> None:
    params = ATMGParameters.chiral(3, 1.53)
    model = ATMGModel.from_config(3, 1.53, n_shells=1, params=params)

    summary = model.lattice_summary()
    path = model.standard_kpath(points_per_segment=6)
    path_result = model.bands_along_standard_path(points_per_segment=6, n_bands=12, include_mapped=True)
    grid_result = model.bands_on_grid(3, n_bands=8, include_mapped=True)

    assert summary["theta_deg"] == 1.53
    assert summary["n_layers"] == 3
    assert summary["matrix_dim"] == model.matrix_dim
    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path_result.energies.shape == (path.kvec.size, 12)
    assert path_result.mapped_energies is not None
    assert path_result.mapped_energies.shape == (path.kvec.size, 12)
    assert path_result.subspace_labels == ("TBG-1", "MLG")
    assert path_result.subspace_energies is not None
    assert grid_result.energies.shape == (3, 3, 8)
    assert grid_result.mapped_energies is not None
    assert grid_result.mapped_energies.shape == (3, 3, 8)


def test_chiral_trilayer_validation_hits_exact_mapping_and_dirac_crossing() -> None:
    params = ATMGParameters.chiral(3, 1.53)
    model = ATMGModel.from_config(3, 1.53, n_shells=1, params=params)
    report = validate_physics(model)

    assert not report.has_failures

    evals_gamma, _ = model.diagonalize(model.lattice.gamma_m, valley=1)
    assert np.min(np.abs(evals_gamma)) < 1.0e-10
