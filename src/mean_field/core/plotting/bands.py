from __future__ import annotations

"""Common band/path plotting surface under ``mean_field.core``.

The implementation currently re-exports the mature shared helpers from
``mean_field.plotting`` so systems can migrate imports to the intended core
namespace without changing rendered output.  Future generic band-plot writers
should be added here and keep system-specific labels/default paths in
``mean_field.systems.<system>``.
"""

from ...plotting import (
    BandStyleCallback,
    LabelFormatter,
    PlotOutputPaths,
    format_kpath_axis,
    kpath_node_ticks,
    load_plot_backend,
    plot_band_columns,
    prepare_plot_paths,
    save_figure_pair,
    write_kpath_band_tsv,
    write_kpath_nodes_tsv,
)

__all__ = [
    "BandStyleCallback",
    "LabelFormatter",
    "PlotOutputPaths",
    "format_kpath_axis",
    "kpath_node_ticks",
    "load_plot_backend",
    "plot_band_columns",
    "prepare_plot_paths",
    "save_figure_pair",
    "write_kpath_band_tsv",
    "write_kpath_nodes_tsv",
]
