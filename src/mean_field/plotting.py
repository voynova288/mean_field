"""Shared plotting backend helpers for Mean_Field modules."""

from __future__ import annotations

import os
import tempfile
from typing import Any


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


__all__ = ["load_plot_backend"]
