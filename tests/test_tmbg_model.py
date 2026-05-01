from __future__ import annotations

from mean_field.systems.tmbg import TMBGModel, TMBGParameters


def test_tmbg_model_exposes_standard_builders_and_summary() -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full(interlayer_potential=0.02))

    summary = model.lattice_summary()
    path = model.standard_kpath(points_per_segment=6)
    path_result = model.bands_along_standard_path(points_per_segment=6, n_bands=10)
    grid_result = model.bands_on_grid(3, n_bands=6)

    assert summary["theta_deg"] == 1.21
    assert summary["n_shells"] == 1
    assert summary["N_G"] == model.lattice.n_g
    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path_result.energies.shape == (path.kvec.size, 10)
    assert grid_result.energies.shape == (3, 3, 6)
