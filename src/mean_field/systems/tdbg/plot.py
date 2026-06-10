from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...plotting import PathBandPlotTrace, write_path_band_plot
from .bands import PathBandsResult

@dataclass(frozen=True)
class TDBGPathPlotTrace:
    label: str
    path_result: PathBandsResult
    color: str = "#1f1f1f"
    linestyle: str = "-"
    linewidth: float = 0.8
    alpha: float = 0.85

def _display_node_label(label: str) -> str:
    return {"Gamma": "Γ", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(label, label)

def write_tdbg_path_band_plot(
    output_dir: Path | str,
    traces: tuple[TDBGPathPlotTrace, ...],
    *,
    stem: str = "bands_path",
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    common_traces = tuple(
        PathBandPlotTrace(
            label=trace.label,
            path_result=trace.path_result,
            color=trace.color,
            linestyle=trace.linestyle,
            linewidth=trace.linewidth,
            alpha=trace.alpha,
        )
        for trace in traces
    )
    return write_path_band_plot(
        output_dir,
        common_traces,
        stem=stem,
        title=title,
        ylabel="Energy (eV)",
        ylim=ylim,
        label_formatter=_display_node_label,
        figsize=(7.2, 4.8),
    )
