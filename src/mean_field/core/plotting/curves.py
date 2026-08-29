from __future__ import annotations

"""Generic exact-grid curve and plan-bound raster plotting helper."""

from typing import Any

from ..curve_workflow.analysis import CurveValueKind, compatible_value_convention
from ..curve_workflow.contracts import ExactSavedGridCurveBundle
from ..curve_workflow.raster import RasterBundleComparison


def plot_exact_grid_curve_bundle(
    bundle: ExactSavedGridCurveBundle,
    *,
    comparison: RasterBundleComparison | None = None,
    value_kind: CurveValueKind = "output",
    ax: Any | None = None,
) -> Any:
    """Plot every computed terminal curve and an optional plan-bound raster evaluation."""

    if comparison is not None:
        if not isinstance(comparison, RasterBundleComparison):
            raise TypeError("comparison must be a bound RasterBundleComparison, not a bare raster")
        if comparison.plan.compute_bundle_fingerprint != bundle.bundle_fingerprint:
            raise ValueError("comparison is bound to a different compute bundle")
        if value_kind != comparison.value_kind:
            raise ValueError("plot value_kind must match the bound comparison plan")
        expected_ids = tuple(curve.terminal_id for curve in bundle.curves)
        if tuple(item.terminal_id for item in comparison.branch_metrics) != expected_ids:
            raise ValueError("comparison does not contain every computed terminal curve")
    units, semantics, transforms_identical = compatible_value_convention(bundle, value_kind)
    if ax is None:
        import matplotlib.pyplot as plt

        _figure, ax = plt.subplots()
    for curve in bundle.curves:
        y = curve.raw_y if value_kind == "raw" else curve.output_y
        ax.plot(
            curve.saved_grid.x,
            y,
            marker="o",
            linestyle="-",
            label=f"computed terminal curve: {curve.terminal_id}",
        )
    if comparison is not None:
        raster = comparison.raster
        lineage = comparison.plan.selection_lineage
        raster_label = (
            "held-out raster evaluation"
            if lineage.solver_target_isolated
            and lineage.raster_evaluation_postfreeze
            else "plan-bound raster evaluation (held-out status not established)"
        )
        ax.errorbar(
            raster.x,
            raster.y,
            yerr=raster.total_uncertainty_y,
            fmt=".",
            linestyle="none",
            label=raster_label,
        )
    first = bundle.curves[0]
    identity_note = "identical transforms" if transforms_identical else "branch-specific transforms"
    transform_semantics = sorted({curve.value_transform.semantics for curve in bundle.curves})
    transform_note = "; ".join(transform_semantics)
    ax.set_xlabel(f"x [{first.saved_grid.x_units}]")
    ax.set_ylabel(f"{first.observable.kind}: {value_kind} [{units}]")
    ax.set_title(
        f"{value_kind} convention — {semantics}; transform semantics: {transform_note} ({identity_note})"
    )
    ax.legend()
    return ax


__all__ = ["plot_exact_grid_curve_bundle"]
