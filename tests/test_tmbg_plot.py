from __future__ import annotations

import numpy as np

from mean_field.systems.tmbg import (
    TMBGBandPlotPanel,
    TMBGModel,
    TMBGParameters,
    infer_flat_band_indices,
    write_tmbg_band_plot,
    write_tmbg_berry_curvature_plot,
    write_tmbg_lattice_plot,
    write_tmbg_paper_band_figure,
)


def test_tmbg_plot_helpers_write_band_and_berry_artifacts(tmp_path) -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full(interlayer_potential=0.06))
    path_result = model.bands_along_standard_path(points_per_segment=4, n_bands=10)
    topology_result = model.topology_on_grid(4, 0, valley=1)

    band_paths = write_tmbg_band_plot(tmp_path, path_result)
    berry_paths = write_tmbg_berry_curvature_plot(tmp_path, topology_result)
    lattice_paths = write_tmbg_lattice_plot(tmp_path, model.lattice)

    for path in [*band_paths.values(), *berry_paths.values(), *lattice_paths.values()]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_tmbg_paper_band_figure_writes_three_panel_artifacts(tmp_path) -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full())
    path_result = model.bands_along_standard_path(points_per_segment=4, n_bands=12)
    flat_pair = infer_flat_band_indices(path_result.energies)
    panels = (
        TMBGBandPlotPanel(
            label="Δ = 0 meV",
            path_result=path_result,
            band_indices=tuple(range(2, 10)),
            flat_band_indices=flat_pair,
            annotation="flat_gap @ Δ=0 meV: 1.00 meV at k=K",
        ),
        TMBGBandPlotPanel(
            label="Δ = +60 meV",
            path_result=path_result,
            band_indices=tuple(range(2, 10)),
            flat_band_indices=flat_pair,
            annotation="flat_gap @ Δ=+60 meV: 2.00 meV at k=Γ",
        ),
        TMBGBandPlotPanel(
            label="Δ = -40 meV",
            path_result=path_result,
            band_indices=tuple(range(2, 10)),
            flat_band_indices=flat_pair,
            annotation="flat_gap @ Δ=-40 meV: 3.00 meV at k=M",
        ),
    )

    figure_paths = write_tmbg_paper_band_figure(tmp_path, panels)

    for path in figure_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_infer_flat_band_indices_prefers_pair_near_neutrality() -> None:
    energies = np.asarray(
        [
            [-0.30, -0.06, -0.01, 0.02, 0.18, 0.34],
            [-0.28, -0.05, -0.02, 0.01, 0.20, 0.36],
            [-0.31, -0.07, -0.01, 0.03, 0.17, 0.35],
        ],
        dtype=float,
    )

    assert infer_flat_band_indices(energies) == (2, 3)
