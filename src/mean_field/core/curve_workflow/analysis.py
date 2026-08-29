from __future__ import annotations

"""Exact-node and explicit open-interval piecewise-linear curve analyses."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .contracts import (
    CurveDomainReceipt,
    ExactSavedGridCurveBundle,
    canonical_array_sha256,
    immutable_finite_array,
)

CrossingKind = Literal["exact_zero", "zero_plateau", "piecewise_linear"]
ExtremumKind = Literal["minimum", "maximum"]
CurveValueKind = Literal["raw", "output"]


@dataclass(frozen=True)
class ZeroCrossing:
    """A zero node/plateau or a linear crossing between adjacent exact nodes."""

    kind: CrossingKind
    x: float
    x_left: float
    x_right: float

    def __post_init__(self) -> None:
        for name in ("x", "x_left", "x_right"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.kind not in {"exact_zero", "zero_plateau", "piecewise_linear"}:
            raise ValueError(f"unsupported crossing kind {self.kind!r}")
        if not self.x_left <= self.x <= self.x_right:
            raise ValueError("crossing x must lie inside [x_left, x_right]")


@dataclass(frozen=True)
class ExactGridExtremum:
    """Strict local extremum at an exact saved-grid node."""

    kind: ExtremumKind
    point_index: int
    array_index: int
    x: float
    y: float


@dataclass(frozen=True)
class CenteredExactGridCurve:
    """A curve centered only at an explicitly requested exact node."""

    requested_x: float
    array_index: int
    center_value: float
    y: np.ndarray
    y_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        centered = immutable_finite_array(self.y, dtype=np.dtype("<f8"), name="centered_y")
        object.__setattr__(self, "y", centered)
        object.__setattr__(self, "y_sha256", canonical_array_sha256(centered))


@dataclass(frozen=True)
class PointwiseBranchSpread:
    """Pointwise min/max/spread over every computed terminal curve."""

    point_indices: np.ndarray
    x: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    spread: np.ndarray
    value_kind: CurveValueKind
    value_units: str
    value_semantics: str
    transforms_identical: bool

    def __post_init__(self) -> None:
        arrays = {
            "point_indices": (self.point_indices, np.dtype("<i8")),
            "x": (self.x, np.dtype("<f8")),
            "minimum": (self.minimum, np.dtype("<f8")),
            "maximum": (self.maximum, np.dtype("<f8")),
            "spread": (self.spread, np.dtype("<f8")),
        }
        shape: tuple[int, ...] | None = None
        for name, (value, dtype) in arrays.items():
            frozen = immutable_finite_array(value, dtype=dtype, name=name)
            if shape is None:
                shape = frozen.shape
            elif frozen.shape != shape:
                raise ValueError("all pointwise spread arrays must have one common shape")
            object.__setattr__(self, name, frozen)
        if self.value_kind not in {"raw", "output"}:
            raise ValueError(f"unsupported value_kind {self.value_kind!r}")
        if not str(self.value_units).strip() or not str(self.value_semantics).strip():
            raise ValueError("value units and semantics must be nonempty")
        if type(self.transforms_identical) is not bool:
            raise TypeError("transforms_identical must be bool")
        if not np.array_equal(self.spread, self.maximum - self.minimum):
            raise ValueError("spread must equal maximum minus minimum")


def _validated_xy(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    x_array = immutable_finite_array(x, dtype=np.dtype("<f8"), name="x")
    y_array = immutable_finite_array(y, dtype=np.dtype("<f8"), name="y")
    if x_array.shape != y_array.shape:
        raise ValueError(f"x and y shapes differ: {x_array.shape} and {y_array.shape}")
    if x_array.size > 1 and np.any(np.diff(x_array) <= 0.0):
        raise ValueError("x must be strictly increasing")
    return x_array, y_array


def exact_zero_piecewise_linear_crossings(x: object, y: object) -> tuple[ZeroCrossing, ...]:
    """Find exact zeros and explicit linear sign-changing crossings."""

    x_array, y_array = _validated_xy(x, y)
    crossings: list[ZeroCrossing] = []
    index = 0
    while index < y_array.size:
        if y_array[index] == 0.0:
            stop = index
            while stop + 1 < y_array.size and y_array[stop + 1] == 0.0:
                stop += 1
            left = float(x_array[index])
            right = float(x_array[stop])
            kind: CrossingKind = "exact_zero" if index == stop else "zero_plateau"
            crossings.append(ZeroCrossing(kind=kind, x=0.5 * (left + right), x_left=left, x_right=right))
            index = stop + 1
            continue
        if index + 1 < y_array.size and y_array[index + 1] != 0.0:
            left_y = float(y_array[index])
            right_y = float(y_array[index + 1])
            if np.signbit(left_y) != np.signbit(right_y):
                left_x = float(x_array[index])
                right_x = float(x_array[index + 1])
                crossing_x = left_x - left_y * (right_x - left_x) / (right_y - left_y)
                crossings.append(
                    ZeroCrossing(
                        kind="piecewise_linear",
                        x=crossing_x,
                        x_left=left_x,
                        x_right=right_x,
                    )
                )
        index += 1
    return tuple(crossings)


def exact_grid_local_extrema(
    point_indices: object,
    x: object,
    y: object,
) -> tuple[ExactGridExtremum, ...]:
    """Return strict interior local extrema at exact nodes; plateaus are omitted."""

    x_array, y_array = _validated_xy(x, y)
    indices = immutable_finite_array(point_indices, dtype=np.dtype("<i8"), name="point_indices")
    if indices.shape != x_array.shape:
        raise ValueError("point_indices must match x and y")
    extrema: list[ExactGridExtremum] = []
    for array_index in range(1, y_array.size - 1):
        left = y_array[array_index - 1]
        value = y_array[array_index]
        right = y_array[array_index + 1]
        if value > left and value > right:
            kind: ExtremumKind = "maximum"
        elif value < left and value < right:
            kind = "minimum"
        else:
            continue
        extrema.append(
            ExactGridExtremum(
                kind=kind,
                point_index=int(indices[array_index]),
                array_index=array_index,
                x=float(x_array[array_index]),
                y=float(value),
            )
        )
    return tuple(extrema)


def center_at_exact_requested_x(
    x: object,
    y: object,
    *,
    requested_x: float,
) -> CenteredExactGridCurve:
    """Subtract the value at an explicitly requested exact x node."""

    x_array, y_array = _validated_xy(x, y)
    target = float(requested_x)
    if not np.isfinite(target):
        raise ValueError("requested_x must be finite")
    matches = np.flatnonzero(x_array == target)
    if matches.size != 1:
        raise ValueError("requested_x must match exactly one saved-grid node")
    array_index = int(matches[0])
    center_value = float(y_array[array_index])
    return CenteredExactGridCurve(
        requested_x=target,
        array_index=array_index,
        center_value=center_value,
        y=y_array - center_value,
    )


def compatible_value_convention(
    bundle: ExactSavedGridCurveBundle,
    value_kind: CurveValueKind,
) -> tuple[str, str, bool]:
    """Validate all-curve units/semantics and report transform identity."""

    if value_kind not in {"raw", "output"}:
        raise ValueError(f"unsupported value_kind {value_kind!r}")
    transforms_identical = bundle.transforms_identical
    if value_kind == "raw":
        units = {curve.value_transform.input_units for curve in bundle.curves}
        if len(units) != 1:
            raise ValueError("raw all-branch comparisons require common input units")
        return next(iter(units)), f"raw {bundle.curves[0].observable.kind}", transforms_identical
    units = {curve.value_transform.output_units for curve in bundle.curves}
    semantics = {curve.value_transform.semantics for curve in bundle.curves}
    if len(units) != 1:
        raise ValueError("output all-branch comparisons require compatible output units")
    if len(semantics) != 1:
        raise ValueError("output all-branch comparisons require compatible transform semantics")
    return next(iter(units)), next(iter(semantics)), transforms_identical


def all_branch_pointwise_spread(
    bundle: ExactSavedGridCurveBundle,
    *,
    value_kind: CurveValueKind = "output",
) -> PointwiseBranchSpread:
    """Compute spread over all computed terminal curves on an open interval."""

    bundle.domain.require_open_interval(operation="all-branch spread")
    units, semantics, transforms_identical = compatible_value_convention(bundle, value_kind)
    values = np.stack(
        [curve.raw_y if value_kind == "raw" else curve.output_y for curve in bundle.curves],
        axis=0,
    )
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    return PointwiseBranchSpread(
        point_indices=bundle.point_indices,
        x=bundle.x,
        minimum=minimum,
        maximum=maximum,
        spread=maximum - minimum,
        value_kind=value_kind,
        value_units=units,
        value_semantics=semantics,
        transforms_identical=transforms_identical,
    )


def explicit_piecewise_linear_interpolation(
    x: object,
    y: object,
    query_x: object,
    *,
    domain: CurveDomainReceipt,
) -> np.ndarray:
    """Interpolate adjacent exact nodes only; periodic seams are unsupported."""

    if not isinstance(domain, CurveDomainReceipt):
        raise TypeError("domain must be a CurveDomainReceipt")
    domain.require_open_interval(operation="piecewise-linear interpolation")
    x_array, y_array = _validated_xy(x, y)
    query = immutable_finite_array(query_x, dtype=np.dtype("<f8"), name="query_x")
    if query.size > 1 and np.any(np.diff(query) <= 0.0):
        raise ValueError("query_x must be strictly increasing")
    if query[0] < x_array[0] or query[-1] > x_array[-1]:
        raise ValueError("piecewise-linear interpolation does not extrapolate")
    result = np.interp(query, x_array, y_array)
    return immutable_finite_array(result, dtype=np.dtype("<f8"), name="interpolated_y")


__all__ = [
    "CenteredExactGridCurve",
    "CrossingKind",
    "CurveValueKind",
    "ExactGridExtremum",
    "ExtremumKind",
    "PointwiseBranchSpread",
    "ZeroCrossing",
    "all_branch_pointwise_spread",
    "center_at_exact_requested_x",
    "compatible_value_convention",
    "exact_grid_local_extrema",
    "exact_zero_piecewise_linear_crossings",
    "explicit_piecewise_linear_interpolation",
]
