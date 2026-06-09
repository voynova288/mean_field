"""Shared plotting and lightweight output helpers for Mean_Field modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

PathLike = Path | str
LabelFormatter = Callable[[str], str]
BandStyleCallback = Callable[[int], Mapping[str, Any] | None]

@dataclass(frozen=True)
class PlotOutputPaths:
    """PNG/PDF output path pair produced by the common plot writers."""

    png: Path
    pdf: Path

    def as_dict(self, key_prefix: str) -> dict[str, Path]:
        """Return the historical ``{"<prefix>_png", "<prefix>_pdf"}`` mapping."""

        return {f"{key_prefix}_png": self.png, f"{key_prefix}_pdf": self.pdf}

def load_plot_backend(*, include_line2d: bool = False) -> Any:
    """Load Matplotlib with a safe non-interactive backend.

    System modules often render plots in batch jobs, on test nodes, or in
    headless CI-style validations.  Keep the backend setup in one place so each
    physical system only owns its plot content and labels.
    """

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    if include_line2d:
        from matplotlib.lines import Line2D

        return plt, Line2D
    return plt

def prepare_plot_paths(output_dir: PathLike, stem: str) -> PlotOutputPaths:
    """Create an output directory and return standard PNG/PDF plot paths."""

    if not stem:
        raise ValueError("Expected a non-empty plot stem.")
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return PlotOutputPaths(png=resolved_dir / f"{stem}.png", pdf=resolved_dir / f"{stem}.pdf")

def save_figure_pair(
    fig: Any,
    output_dir: PathLike,
    stem: str,
    *,
    key_prefix: str = "band_plot",
    png_dpi: int = 300,
    bbox_inches: str | None = "tight",
) -> dict[str, Path]:
    """Save one Matplotlib figure as PNG and PDF using repository defaults."""

    paths = prepare_plot_paths(output_dir, stem)
    fig.savefig(paths.png, dpi=png_dpi, bbox_inches=bbox_inches)
    fig.savefig(paths.pdf, bbox_inches=bbox_inches)
    return paths.as_dict(key_prefix)

def _format_node_label(
    label: str,
    *,
    label_map: Mapping[str, str] | None = None,
    label_formatter: LabelFormatter | None = None,
) -> str:
    if label_formatter is not None:
        return label_formatter(label)
    if label_map is not None:
        return label_map.get(label, label)
    return label

def kpath_node_ticks(
    path: Any,
    *,
    label_map: Mapping[str, str] | None = None,
    label_formatter: LabelFormatter | None = None,
) -> tuple[list[float], list[str]]:
    """Return high-symmetry tick positions and labels for a ``KPath``-like object."""

    nodes = tuple(path.nodes)
    node_x = [float(node.k_dist) for node in nodes]
    node_labels = [
        _format_node_label(str(node.label), label_map=label_map, label_formatter=label_formatter)
        for node in nodes
    ]
    return node_x, node_labels

def format_kpath_axis(
    ax: Any,
    path: Any,
    *,
    label_map: Mapping[str, str] | None = None,
    label_formatter: LabelFormatter | None = None,
    vertical_line_kwargs: Mapping[str, Any] | None = None,
    xlabel: str | None = "k-path",
    set_xlim: bool = True,
) -> tuple[list[float], list[str]]:
    """Apply common high-symmetry ticks, guide lines, and x-label to a path axis."""

    node_x, node_labels = kpath_node_ticks(path, label_map=label_map, label_formatter=label_formatter)
    line_style: dict[str, Any] = {"color": "#999999", "linestyle": ":", "linewidth": 0.8}
    if vertical_line_kwargs is not None:
        line_style.update(dict(vertical_line_kwargs))
    for xpos in node_x:
        ax.axvline(x=xpos, **line_style)
    ax.set_xticks(node_x)
    ax.set_xticklabels(node_labels)
    if set_xlim and node_x:
        ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return node_x, node_labels

def plot_band_columns(
    ax: Any,
    kdist: Sequence[float] | np.ndarray,
    energies: np.ndarray,
    *,
    bands_axis: Literal["columns", "rows"] = "columns",
    energy_scale: float = 1.0,
    energy_shift: float = 0.0,
    style_by_band: BandStyleCallback | None = None,
    **plot_kwargs: Any,
) -> np.ndarray:
    """Plot a 2D band-energy array and return the transformed ``(nk, nb)`` data.

    ``bands_axis="columns"`` means input shape ``(nk, nb)``.  ``"rows"``
    accepts ``(nb, nk)`` and transposes before plotting.
    """

    x_values = np.asarray(kdist, dtype=float)
    values = np.asarray(energies, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D energy array, got {values.shape}")
    if bands_axis == "rows":
        values = values.T
    elif bands_axis != "columns":
        raise ValueError(f"Unsupported bands_axis={bands_axis!r}")
    if values.shape[0] != x_values.size:
        raise ValueError(f"Expected {x_values.size} k-points, got energy shape {values.shape}")

    values = values * float(energy_scale) - float(energy_shift)
    for band_index in range(values.shape[1]):
        line_kwargs = dict(plot_kwargs)
        if style_by_band is not None:
            line_kwargs.update(dict(style_by_band(band_index) or {}))
        ax.plot(x_values, values[:, band_index], **line_kwargs)
    return values

def write_kpath_band_tsv(
    path: PathLike,
    *,
    kdist: Sequence[float] | np.ndarray,
    energies: np.ndarray,
    band_labels: Sequence[str],
    bands_axis: Literal["columns", "rows"] = "columns",
    kdist_header: str = "k_dist",
    float_format: str = ".16f",
) -> None:
    """Write path-band data as a TSV with one row per k-point."""

    x_values = np.asarray(kdist, dtype=float)
    values = np.asarray(energies, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D energy array, got {values.shape}")
    if bands_axis == "rows":
        values = values.T
    elif bands_axis != "columns":
        raise ValueError(f"Unsupported bands_axis={bands_axis!r}")
    labels = tuple(str(label) for label in band_labels)
    if values.shape != (x_values.size, len(labels)):
        raise ValueError(
            f"Expected energy shape {(x_values.size, len(labels))} for {len(labels)} labels, got {values.shape}"
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join([kdist_header, *labels]) + "\n")
        for ik, kdist_value in enumerate(x_values):
            row = [format(float(kdist_value), float_format)]
            row.extend(format(float(values[ik, band_index]), float_format) for band_index in range(values.shape[1]))
            handle.write("\t".join(row) + "\n")

def write_kpath_nodes_tsv(path: PathLike, kpath: Any, *, float_format: str = ".16f") -> None:
    """Write high-symmetry ``KPath`` node metadata as a TSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("label\tindex\tk_dist\tkx\tky\n")
        for node in kpath.nodes:
            handle.write(
                "\t".join(
                    [
                        str(node.label),
                        str(int(node.index)),
                        format(float(node.k_dist), float_format),
                        format(float(node.kx), float_format),
                        format(float(node.ky), float_format),
                    ]
                )
                + "\n"
            )

__all__ = [
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
