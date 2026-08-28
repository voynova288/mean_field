from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf.continuation import (
    PseudoArclengthConfig,
    continue_pseudo_arclength,
)
from mean_field.core.hf.stationary import StationarySolveConfig


def test_pseudo_arclength_crosses_analytic_fold() -> None:
    # R(x,p)=x^2-p=0 has a fold in p at x=0. Parameter stepping cannot
    # continue through the turning point, whereas arclength continuation can.
    result = continue_pseudo_arclength(
        lambda state, parameter: np.asarray([state[0] ** 2 - parameter]),
        np.asarray([0.8]),
        0.64,
        np.asarray([0.5]),
        0.25,
        config=PseudoArclengthConfig(
            step_size=0.12,
            state_scale=1.0,
            parameter_scale=1.0,
            residual_scale=1.0,
            max_steps=12,
            max_step_retries=3,
            minimum_step_size=0.01,
            root=StationarySolveConfig(
                residual_rms_tolerance=1.0e-11,
                residual_max_tolerance=1.0e-11,
                anderson_max_iterations=40,
                anderson_memory=3,
                krylov_max_iterations=40,
            ),
        ),
    )
    assert result.completed_steps == 12
    assert result.exit_reason == "max_steps"
    states = np.asarray([point.state[0] for point in result.points])
    parameters = np.asarray([point.parameter for point in result.points])
    assert np.min(states) < 0.0
    assert np.min(parameters) < 2.0e-3
    assert any(point.fold_detected for point in result.points)
    for point in result.points:
        assert point.root.converged
        assert abs(point.state[0] ** 2 - point.parameter) < 2.0e-10
        assert abs(point.arclength_residual) < 2.0e-10


def test_pseudo_arclength_is_invariant_under_declared_coordinate_scaling() -> None:
    root = StationarySolveConfig(
        residual_rms_tolerance=1.0e-11,
        residual_max_tolerance=1.0e-11,
        anderson_max_iterations=40,
        anderson_memory=2,
        krylov_max_iterations=40,
    )
    base = continue_pseudo_arclength(
        lambda state, parameter: np.asarray([state[0] - parameter]),
        np.asarray([0.0]),
        0.0,
        np.asarray([0.1]),
        0.1,
        config=PseudoArclengthConfig(step_size=0.1, max_steps=3, root=root),
    )
    scaled = continue_pseudo_arclength(
        lambda state, parameter: np.asarray([state[0] - parameter]),
        np.asarray([0.0]),
        0.0,
        np.asarray([0.2]),
        0.2,
        config=PseudoArclengthConfig(
            step_size=0.1,
            state_scale=2.0,
            parameter_scale=2.0,
            residual_scale=2.0,
            max_steps=3,
            root=root,
        ),
    )
    assert np.asarray([point.state for point in scaled.points]) == pytest.approx(
        np.asarray([point.state for point in base.points]) * 2.0
    )
