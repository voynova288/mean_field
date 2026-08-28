"""Generic pseudo-arclength continuation for stationary residual branches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .stationary import StationarySolveConfig, StationarySolveResult, solve_stationary_residual

Array = np.ndarray
ParameterizedResidual = Callable[[Array, float], Array]


@dataclass(frozen=True)
class PseudoArclengthConfig:
    step_size: float
    state_scale: float = 1.0
    parameter_scale: float = 1.0
    residual_scale: float = 1.0
    max_steps: int = 20
    max_step_retries: int = 4
    minimum_step_size: float = 1.0e-4
    root: StationarySolveConfig = StationarySolveConfig()

    def __post_init__(self) -> None:
        for value in (
            self.step_size,
            self.state_scale,
            self.parameter_scale,
            self.residual_scale,
            self.minimum_step_size,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("continuation scales and step sizes must be finite and positive")
        if self.max_steps < 1 or self.max_step_retries < 0:
            raise ValueError("continuation step controls are invalid")
        if self.minimum_step_size > self.step_size:
            raise ValueError("minimum_step_size cannot exceed step_size")


@dataclass(frozen=True)
class PseudoArclengthPoint:
    state: Array
    parameter: float
    tangent_scaled: Array
    step_size: float
    arclength_residual: float
    root: StationarySolveResult
    fold_detected: bool


@dataclass(frozen=True)
class PseudoArclengthResult:
    points: tuple[PseudoArclengthPoint, ...]
    completed_steps: int
    exit_reason: str


def _scaled_join(state: Array, parameter: float, config: PseudoArclengthConfig) -> Array:
    vector = np.asarray(state, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("continuation state must be one-dimensional and finite")
    if not np.isfinite(parameter):
        raise ValueError("continuation parameter must be finite")
    return np.concatenate(
        [vector / float(config.state_scale), [float(parameter) / float(config.parameter_scale)]]
    )


def _scaled_split(vector: Array, config: PseudoArclengthConfig) -> tuple[Array, float]:
    scaled = np.asarray(vector, dtype=np.float64)
    if scaled.ndim != 1 or scaled.size < 2 or not np.all(np.isfinite(scaled)):
        raise ValueError("scaled continuation vector must be finite and contain state plus parameter")
    return (
        scaled[:-1] * float(config.state_scale),
        float(scaled[-1] * float(config.parameter_scale)),
    )


def _oriented_unit_tangent(newer: Array, older: Array, previous: Array | None = None) -> Array:
    difference = np.asarray(newer, dtype=np.float64) - np.asarray(older, dtype=np.float64)
    norm = float(np.linalg.norm(difference))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("continuation seed points must be distinct in the scaled metric")
    tangent = difference / norm
    if previous is not None and float(np.dot(tangent, previous)) < 0.0:
        tangent = -tangent
    return tangent


def continue_pseudo_arclength(
    residual_builder: ParameterizedResidual,
    first_state: Array,
    first_parameter: float,
    second_state: Array,
    second_parameter: float,
    *,
    config: PseudoArclengthConfig,
) -> PseudoArclengthResult:
    """Continue ``R(x,p)=0`` through folds using a secant predictor/corrector."""

    first = _scaled_join(first_state, first_parameter, config)
    second = _scaled_join(second_state, second_parameter, config)
    tangent = _oriented_unit_tangent(second, first)
    points: list[PseudoArclengthPoint] = []
    current_step = float(config.step_size)
    previous_parameter_tangent = float(tangent[-1])

    for _index in range(int(config.max_steps)):
        accepted = False
        attempted_step = current_step
        last_root: StationarySolveResult | None = None
        for _retry in range(int(config.max_step_retries) + 1):
            predictor = second + attempted_step * tangent

            def augmented_residual(scaled_vector: Array) -> Array:
                state, parameter = _scaled_split(scaled_vector, config)
                physical = np.asarray(residual_builder(state, parameter), dtype=np.float64)
                if physical.shape != state.shape or not np.all(np.isfinite(physical)):
                    raise ValueError("parameterized residual must be finite and match the state shape")
                arclength = float(np.dot(scaled_vector - predictor, tangent))
                return np.concatenate(
                    [physical / float(config.residual_scale), [arclength]]
                )

            root = solve_stationary_residual(augmented_residual, predictor, config=config.root)
            last_root = root
            if root.converged:
                corrected = np.asarray(root.vector, dtype=np.float64)
                new_tangent = _oriented_unit_tangent(corrected, second, tangent)
                state, parameter = _scaled_split(corrected, config)
                arclength_residual = float(np.dot(corrected - predictor, tangent))
                parameter_tangent = float(new_tangent[-1])
                fold = previous_parameter_tangent * parameter_tangent < 0.0
                points.append(
                    PseudoArclengthPoint(
                        state=state,
                        parameter=parameter,
                        tangent_scaled=new_tangent,
                        step_size=attempted_step,
                        arclength_residual=arclength_residual,
                        root=root,
                        fold_detected=fold,
                    )
                )
                first, second = second, corrected
                tangent = new_tangent
                previous_parameter_tangent = parameter_tangent
                current_step = min(float(config.step_size), 1.25 * attempted_step)
                accepted = True
                break
            attempted_step *= 0.5
            if attempted_step < float(config.minimum_step_size):
                break
        if not accepted:
            reason = "corrector_failed"
            if last_root is not None:
                reason = f"corrector_failed:{last_root.exit_reason}"
            return PseudoArclengthResult(
                points=tuple(points),
                completed_steps=len(points),
                exit_reason=reason,
            )
    return PseudoArclengthResult(
        points=tuple(points),
        completed_steps=len(points),
        exit_reason="max_steps",
    )


__all__ = [
    "PseudoArclengthConfig",
    "PseudoArclengthPoint",
    "PseudoArclengthResult",
    "continue_pseudo_arclength",
]
