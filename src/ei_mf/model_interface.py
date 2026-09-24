"""Minimal interface tests between normal-state bands and scalar EI fields.

These helpers diagnose whether two saved calculations differ only by a constant
relative band alignment.  They do not solve a new mean-field model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ConstantAlignmentFit:
    delta_ls_mev: float
    delta_minimax_mev: float
    residual_ls_mev: Array
    residual_minimax_mev: Array
    rms_ls_mev: float
    max_abs_ls_mev: float
    minimax_lower_bound_mev: float
    required_delta_min_mev: float
    required_delta_max_mev: float


def fit_constant_relative_alignment(
    signed_normal_detuning_mev: Array,
    target_xi_mev: Array,
    *,
    weights: Array | None = None,
) -> ConstantAlignmentFit:
    """Fit ``target_xi = signed_normal_detuning + delta``.

    ``delta_ls`` minimizes the weighted squared residual.  ``delta_minimax``
    minimizes the largest absolute residual and has the analytic value halfway
    between the minimum and maximum pointwise required alignments.  The latter
    provides a weight-independent no-go bound for a constant alignment.
    """

    normal = np.asarray(signed_normal_detuning_mev, dtype=np.float64)
    target = np.asarray(target_xi_mev, dtype=np.float64)
    if normal.ndim != 1 or target.shape != normal.shape or normal.size == 0:
        raise ValueError("normal detuning and target xi must be matching 1D arrays")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(target)):
        raise ValueError("alignment inputs must be finite")
    if weights is None:
        fit_weights = np.ones_like(normal)
    else:
        fit_weights = np.asarray(weights, dtype=np.float64)
        if fit_weights.shape != normal.shape:
            raise ValueError("weights must match the detuning arrays")
        if not np.all(np.isfinite(fit_weights)):
            raise ValueError("weights must be finite")
        if np.any(fit_weights < 0.0) or not np.any(fit_weights > 0.0):
            raise ValueError("weights must be nonnegative with positive total")
    required = target - normal
    delta_ls = float(np.sum(fit_weights * required) / np.sum(fit_weights))
    required_min = float(np.min(required))
    required_max = float(np.max(required))
    delta_minimax = 0.5 * (required_min + required_max)
    residual_ls = normal + delta_ls - target
    residual_minimax = normal + delta_minimax - target
    return ConstantAlignmentFit(
        delta_ls_mev=delta_ls,
        delta_minimax_mev=delta_minimax,
        residual_ls_mev=np.asarray(residual_ls, dtype=np.float64),
        residual_minimax_mev=np.asarray(residual_minimax, dtype=np.float64),
        rms_ls_mev=float(
            np.sqrt(np.sum(fit_weights * residual_ls**2) / np.sum(fit_weights))
        ),
        max_abs_ls_mev=float(np.max(np.abs(residual_ls))),
        minimax_lower_bound_mev=0.5 * (required_max - required_min),
        required_delta_min_mev=required_min,
        required_delta_max_mev=required_max,
    )


def aligned_pair_energy_mev(
    signed_normal_detuning_mev: Array,
    delta_alignment_mev: float,
    delta_order_mev: Array,
) -> Array:
    """Paper-convention pair energy after a constant relative alignment."""

    normal = np.asarray(signed_normal_detuning_mev, dtype=np.float64)
    order = np.asarray(delta_order_mev, dtype=np.float64)
    if normal.shape != order.shape:
        raise ValueError("normal detuning and order parameter must match")
    return np.sqrt((normal + float(delta_alignment_mev)) ** 2 + order**2)
