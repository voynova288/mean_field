from __future__ import annotations

"""Wavefunction layout helpers for the minimal topology core.

The FHS kernels consume canonical arrays with shape
``(mesh_1, mesh_2, basis_dim, n_state)``. This module only flattens already
computed wavefunction state axes and records labels for the flattened columns;
it does not reconstruct microscopic wavefunctions or infer system-specific
sewing conventions.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .core import WavefunctionIndex, normalize_state_indices


@dataclass(frozen=True)
class WavefunctionLayout:
    """Describe how to canonicalize a wavefunction grid.

    Mesh axes are fixed to the first two axes. ``basis_axis`` identifies the
    Hilbert-space basis dimension, and ``state_axes`` are flattened into the
    final state-column axis. For example, an array shaped
    ``(nk1, nk2, basis, band, flavor)`` can use
    ``WavefunctionLayout(basis_axis=2, state_axes=(3, 4),
    state_axis_names=("band", "flavor"))``.
    """

    basis_axis: int = 2
    state_axes: tuple[int, ...] = (-1,)
    state_axis_names: tuple[str, ...] = ("state",)
    state_axis_labels: Mapping[str, Sequence[object]] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalWavefunctionGrid:
    """Canonical wavefunctions plus flattened state-label metadata."""

    wavefunctions: np.ndarray
    state_labels: tuple[Mapping[str, object], ...]
    layout: WavefunctionLayout

    def index_for(
        self,
        state_indices: int | Iterable[int],
        *,
        role: str = "state",
        system: str | None = None,
        valley: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> WavefunctionIndex:
        """Build metadata for selected flattened state columns."""

        normalized = normalize_state_indices(state_indices)
        selected_labels = [dict(self.state_labels[index]) for index in normalized]
        payload: dict[str, object] = {"selected_state_labels": selected_labels}
        if metadata:
            payload.update(dict(metadata))
        return WavefunctionIndex(
            indices=normalized,
            role=role,
            labels=tuple(_format_state_label(label) for label in selected_labels),
            system=system,
            valley=valley,
            metadata=payload,
        )


def _positive_axis(axis: int, ndim: int) -> int:
    resolved = int(axis)
    if resolved < 0:
        resolved += int(ndim)
    if resolved < 0 or resolved >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for ndim={ndim}")
    return resolved


def _format_state_label(label: Mapping[str, object]) -> str:
    return "/".join(f"{key}={value}" for key, value in label.items())


def _axis_labels(name: str, size: int, layout: WavefunctionLayout) -> tuple[object, ...]:
    labels = layout.state_axis_labels.get(name)
    if labels is None:
        return tuple(range(int(size)))
    if len(labels) != int(size):
        raise ValueError(f"Axis label list for {name!r} has length {len(labels)}, expected {size}")
    return tuple(labels)


def canonicalize_wavefunction_grid(
    wavefunctions: np.ndarray,
    layout: WavefunctionLayout | None = None,
) -> CanonicalWavefunctionGrid:
    """Move basis/state axes into canonical ``(k1, k2, basis, state)`` form."""

    array = np.asarray(wavefunctions, dtype=np.complex128)
    if array.ndim < 4:
        raise ValueError(
            "Expected at least four axes: (mesh_1, mesh_2, basis_dim, state...), "
            f"got shape {array.shape}"
        )
    resolved_layout = WavefunctionLayout() if layout is None else layout
    if len(resolved_layout.state_axes) != len(resolved_layout.state_axis_names):
        raise ValueError("state_axes and state_axis_names must have the same length")

    ndim = array.ndim
    basis_axis = _positive_axis(resolved_layout.basis_axis, ndim)
    state_axes = tuple(_positive_axis(axis, ndim) for axis in resolved_layout.state_axes)
    mesh_axes = (0, 1)
    ordered_axes = mesh_axes + (basis_axis,) + state_axes
    if len(set(ordered_axes)) != len(ordered_axes):
        raise ValueError(f"Mesh, basis, and state axes must be distinct, got {ordered_axes}")
    expected_axes = set(range(ndim))
    if set(ordered_axes) != expected_axes:
        extra = sorted(expected_axes - set(ordered_axes))
        missing = sorted(set(ordered_axes) - expected_axes)
        raise ValueError(
            "All non-mesh axes must be assigned as basis_axis or state_axes; "
            f"extra={extra}, missing={missing}"
        )

    moved = np.transpose(array, axes=ordered_axes)
    mesh_1, mesh_2, basis_dim = moved.shape[:3]
    state_shape = moved.shape[3:]
    n_state = int(np.prod(state_shape, dtype=np.int64))
    canonical = moved.reshape((mesh_1, mesh_2, basis_dim, n_state), order="C")

    per_axis_labels = [
        _axis_labels(name, size, resolved_layout)
        for name, size in zip(resolved_layout.state_axis_names, state_shape, strict=True)
    ]
    flat_labels: list[Mapping[str, object]] = []
    for multi_index in np.ndindex(*state_shape):
        flat_labels.append(
            {
                name: per_axis_labels[axis_position][state_index]
                for axis_position, (name, state_index) in enumerate(
                    zip(resolved_layout.state_axis_names, multi_index, strict=True)
                )
            }
        )
    return CanonicalWavefunctionGrid(
        wavefunctions=canonical,
        state_labels=tuple(flat_labels),
        layout=resolved_layout,
    )


def reshape_flat_mesh_to_grid(
    values: np.ndarray,
    mesh_shape: tuple[int, int],
    *,
    k_axis: int = -1,
    order: str = "C",
) -> np.ndarray:
    """Reshape a flat k-axis into ``(mesh_1, mesh_2)`` and move it to the front."""

    array = np.asarray(values)
    if len(mesh_shape) != 2:
        raise ValueError(f"mesh_shape must have length two, got {mesh_shape}")
    mesh_1, mesh_2 = int(mesh_shape[0]), int(mesh_shape[1])
    if mesh_1 <= 0 or mesh_2 <= 0:
        raise ValueError(f"mesh_shape must be positive, got {mesh_shape}")
    axis = _positive_axis(k_axis, array.ndim)
    if array.shape[axis] != mesh_1 * mesh_2:
        raise ValueError(f"k_axis length {array.shape[axis]} is incompatible with mesh_shape={mesh_shape}")
    moved = np.moveaxis(array, axis, 0)
    return moved.reshape((mesh_1, mesh_2) + moved.shape[1:], order=order)


def wavefunction_index_from_state_labels(
    state_indices: int | Iterable[int],
    state_labels: Sequence[Mapping[str, object]],
    *,
    role: str = "state",
    system: str | None = None,
    valley: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> WavefunctionIndex:
    """Build ``WavefunctionIndex`` metadata for already-flattened state labels."""

    normalized = normalize_state_indices(state_indices)
    selected = [dict(state_labels[index]) for index in normalized]
    payload: dict[str, object] = {"selected_state_labels": selected}
    if metadata:
        payload.update(dict(metadata))
    return WavefunctionIndex(
        indices=normalized,
        role=role,
        labels=tuple(_format_state_label(label) for label in selected),
        system=system,
        valley=valley,
        metadata=payload,
    )


__all__ = [
    "CanonicalWavefunctionGrid",
    "WavefunctionLayout",
    "canonicalize_wavefunction_grid",
    "reshape_flat_mesh_to_grid",
    "wavefunction_index_from_state_labels",
]
