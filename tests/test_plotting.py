from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mean_field.plotting import PathBandPlotTrace, write_path_band_plot
from mean_field.systems.RnG_hBN.plot import RLGhBNPathPlotTrace, write_rlg_hbn_path_band_plot
from mean_field.systems.tdbg.plot import TDBGPathPlotTrace, write_tdbg_path_band_plot


def _fake_path_result() -> SimpleNamespace:
    path = SimpleNamespace(
        kdist=np.asarray([0.0, 1.0, 2.0], dtype=float),
        nodes=(
            SimpleNamespace(k_dist=0.0, label="Gamma"),
            SimpleNamespace(k_dist=2.0, label="K"),
        ),
    )
    return SimpleNamespace(
        path=path,
        energies=np.asarray(
            [
                [0.0, 1.0],
                [0.2, 1.2],
                [0.4, 1.4],
            ],
            dtype=float,
        ),
    )



def test_write_path_band_plot_smoke(tmp_path) -> None:
    result = _fake_path_result()
    paths = write_path_band_plot(
        tmp_path,
        (PathBandPlotTrace("demo", result, energy_scale=2.0, energy_shift=0.5),),
        stem="path_band_plot_smoke",
        ylabel="Energy (arb.)",
        label_map={"Gamma": "Γ"},
        horizontal_lines=({"y": 0.0, "color": "#777777", "linewidth": 0.5},),
        annotate="smoke",
    )

    assert set(paths) == {"band_plot_png", "band_plot_pdf"}
    assert paths["band_plot_png"].is_file()
    assert paths["band_plot_pdf"].is_file()


def test_simple_system_path_band_plot_wrappers_smoke(tmp_path) -> None:
    result = _fake_path_result()

    tdbg_paths = write_tdbg_path_band_plot(
        tmp_path / "tdbg",
        (TDBGPathPlotTrace("demo", result),),
        stem="tdbg_smoke",
    )
    rlg_paths = write_rlg_hbn_path_band_plot(
        tmp_path / "rlg",
        (RLGhBNPathPlotTrace("demo", result, energy_shift_mev=0.1),),
        stem="rlg_smoke",
    )

    assert tdbg_paths["band_plot_png"].is_file()
    assert tdbg_paths["band_plot_pdf"].is_file()
    assert rlg_paths["band_plot_png"].is_file()
    assert rlg_paths["band_plot_pdf"].is_file()
