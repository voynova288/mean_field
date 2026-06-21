from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

Axis = int
Component = tuple[Axis, Axis, Axis]

_AXIS_LABELS: Mapping[Any, int] = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2, 0: 0, 1: 1, 2: 2}
_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ShiftCurrentComponent:
    """Rank-3 shift-current component ``sigma^a_{bc}``."""

    current_axis: int
    optical_axis_1: int
    optical_axis_2: int

    @property
    def as_tuple(self) -> Component:
        return (int(self.current_axis), int(self.optical_axis_1), int(self.optical_axis_2))

    @property
    def compact_label(self) -> str:
        return "".join(_AXIS_NAMES[i] if 0 <= i < len(_AXIS_NAMES) else str(i) for i in self.as_tuple)

    @property
    def semicolon_label(self) -> str:
        a, b, c = self.as_tuple
        label = lambda i: _AXIS_NAMES[i] if 0 <= i < len(_AXIS_NAMES) else str(i)
        return f"{label(a)};{label(b)}{label(c)}"


def axis_index(axis: str | int) -> int:
    """Return an integer axis index for ``x/y/z`` or ``0/1/2`` labels."""

    key: str | int = axis.strip().lower() if isinstance(axis, str) else axis
    try:
        return int(_AXIS_LABELS[key])
    except KeyError as exc:
        raise ValueError(f"Unsupported axis {axis!r}; expected x/y/z or 0/1/2") from exc


def component_from_any(component: ShiftCurrentComponent | Sequence[int] | str) -> ShiftCurrentComponent:
    """Normalize a component label/tuple/dataclass to ``ShiftCurrentComponent``."""

    if isinstance(component, ShiftCurrentComponent):
        return component
    if isinstance(component, str):
        return parse_component(component)
    if len(component) != 3:  # type: ignore[arg-type]
        raise ValueError(f"Component must have three axes, got {component!r}")
    a, b, c = component  # type: ignore[misc]
    return ShiftCurrentComponent(axis_index(a), axis_index(b), axis_index(c))


def parse_component(text: str) -> ShiftCurrentComponent:
    """Parse labels such as ``'xxx'``, ``'x;yy'``, or ``'x_yy'``."""

    raw = str(text).strip().lower().replace("σ", "").replace("^", "").replace("_", ";").replace(" ", "")
    if ";" in raw:
        left, right = raw.split(";", 1)
        if len(left) != 1 or len(right) != 2:
            raise ValueError(f"Component must look like 'xxx' or 'x;yy', got {text!r}")
        return ShiftCurrentComponent(axis_index(left), axis_index(right[0]), axis_index(right[1]))
    if len(raw) == 3:
        return ShiftCurrentComponent(axis_index(raw[0]), axis_index(raw[1]), axis_index(raw[2]))
    raise ValueError(f"Component must look like 'xxx' or 'x;yy', got {text!r}")


def component_label(component: ShiftCurrentComponent | Sequence[int] | str, *, style: Literal["compact", "semicolon"] = "semicolon") -> str:
    comp = component_from_any(component)
    return comp.compact_label if style == "compact" else comp.semicolon_label


__all__ = [
    "Axis",
    "Component",
    "ShiftCurrentComponent",
    "axis_index",
    "component_from_any",
    "component_label",
    "parse_component",
]
