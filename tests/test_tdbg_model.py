from __future__ import annotations

from mean_field.systems.tdbg import TDBGModel, TDBGParameters, validate_physics


def test_tdbg_model_exposes_standard_path_and_grid_helpers() -> None:
    params = TDBGParameters.full(stacking="AB-BA")
    model = TDBGModel.from_config(1.33, cut=1.0, params=params)

    summary = model.lattice_summary()
    path = model.standard_kpath(resolution=8)
    path_result = model.bands_along_standard_path(resolution=8, n_bands=12)
    grid_result = model.bands_on_grid(3, n_bands=8)
    report = validate_physics(model)

    assert summary["theta_deg"] == 1.33
    assert summary["stacking"] == "AB-BA"
    assert path.labels == ("K", "Gamma", "M", "Kprime")
    assert path_result.energies.shape == (path.kvec.size, 12)
    assert grid_result.energies.shape == (3, 3, 8)
    assert not report.has_failures
