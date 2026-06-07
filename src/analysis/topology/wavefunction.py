from __future__ import annotations

"""Wavefunction layout helpers for band/flavor-labelled state axes.

The topology and quantum-geometry kernels consume canonical arrays with shape
``(mesh_1, mesh_2, basis_dim, n_state)``.  Many mean-field artifacts naturally
store extra axes such as ``band``, ``flavor``, ``valley`` or ``spin``.  This
module provides a small adapter that flattens those state axes while preserving
metadata that can be attached to :class:`analysis.topology.WavefunctionIndex`.
"""

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .core import WavefunctionIndex, normalize_state_indices


@dataclass(frozen=True)
class WavefunctionLayout:
    """Describe how to canonicalize a wavefunction grid.

    Mesh axes are assumed to be the first two axes.  ``basis_axis`` identifies
    the Hilbert-space basis dimension, and ``state_axes`` are flattened into the
    final state-column axis.  For example, an array with shape
    ``(nk1, nk2, basis, band, flavor)`` can use

    ``WavefunctionLayout(basis_axis=2, state_axes=(3, 4), state_axis_names=("band", "flavor"))``.
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
        """Build a :class:`WavefunctionIndex` for selected flattened columns."""

        normalized = normalize_state_indices(state_indices)
        selected_labels = [dict(self.state_labels[index]) for index in normalized]
        labels = tuple(_format_state_label(label) for label in selected_labels)
        payload: dict[str, object] = {"selected_state_labels": selected_labels}
        if metadata:
            payload.update(dict(metadata))
        return WavefunctionIndex(
            indices=normalized,
            role=role,
            labels=labels,
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
    """Move basis/state axes into the canonical ``(k1, k2, basis, state)`` form."""

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
    """Reshape a flat k-axis into ``(mesh_1, mesh_2)`` and move it to the front.

    Many HF archives store k-points as a flat axis.  The topology kernels expect
    mesh axes first.  This helper keeps every non-k axis in its original order.
    """

    array = np.asarray(values)
    if len(mesh_shape) != 2:
        raise ValueError(f"mesh_shape must have length two, got {mesh_shape}")
    mesh_1, mesh_2 = int(mesh_shape[0]), int(mesh_shape[1])
    if mesh_1 <= 0 or mesh_2 <= 0:
        raise ValueError(f"mesh_shape must be positive, got {mesh_shape}")
    axis = _positive_axis(k_axis, array.ndim)
    if array.shape[axis] != mesh_1 * mesh_2:
        raise ValueError(
            f"k_axis length {array.shape[axis]} is incompatible with mesh_shape={mesh_shape}"
        )
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape((mesh_1, mesh_2) + moved.shape[1:], order=order)
    return reshaped


def reconstruct_projected_micro_wavefunctions(
    projected_basis_wavefunctions: np.ndarray,
    mixing_vectors: np.ndarray,
    mesh_shape: tuple[int, int],
    *,
    n_spin: int = 1,
    flatten_order: str = "F",
    include_spin_flavor_blocks: bool = True,
) -> np.ndarray:
    """Reconstruct microscopic wavefunctions from projected-HF eigenvectors.

    Parameters
    ----------
    projected_basis_wavefunctions:
        Array with shape ``(basis_dim, n_band, n_flavor, nk)``.  These are the
        k-dependent microscopic Bloch wavefunctions of the projected basis.
    mixing_vectors:
        HF eigenvectors in projected space with shape ``(nt, n_state, nk)`` or
        ``(nt, nk)`` for one state, where ``nt = n_spin*n_flavor*n_band``.
    mesh_shape:
        Two-dimensional k-mesh shape used to unflatten ``nk``.

    Returns
    -------
    np.ndarray
        Canonical grid ``(mesh_1, mesh_2, micro_basis, n_state)``.  By default
        ``micro_basis`` includes explicit direct-sum spin/flavor blocks so that
        spin/valley sectors are orthogonal.  This is the safe representation for
        RLG/hBN Fig. 6 Fubini-Study/Berry-curvature maps after reconstructing
        from active HF eigenvectors.
    """

    basis = np.asarray(projected_basis_wavefunctions, dtype=np.complex128)
    if basis.ndim != 4:
        raise ValueError(
            "projected_basis_wavefunctions must have shape (basis_dim, n_band, n_flavor, nk), "
            f"got {basis.shape}"
        )
    basis_dim, n_band, n_flavor, nk = basis.shape
    spin_count = int(n_spin)
    if spin_count <= 0:
        raise ValueError(f"n_spin must be positive, got {n_spin}")
    coefficients = np.asarray(mixing_vectors, dtype=np.complex128)
    if coefficients.ndim == 2:
        coefficients = coefficients[:, np.newaxis, :]
    if coefficients.ndim != 3:
        raise ValueError(f"mixing_vectors must have shape (nt, n_state, nk), got {coefficients.shape}")
    expected_nt = spin_count * n_flavor * n_band
    if coefficients.shape[0] != expected_nt:
        raise ValueError(f"mixing nt={coefficients.shape[0]} does not match n_spin*n_flavor*n_band={expected_nt}")
    if coefficients.shape[2] != nk:
        raise ValueError(f"mixing nk={coefficients.shape[2]} does not match basis nk={nk}")
    if int(mesh_shape[0]) * int(mesh_shape[1]) != nk:
        raise ValueError(f"mesh_shape={mesh_shape} is incompatible with nk={nk}")
    if flatten_order not in {"C", "F"}:
        raise ValueError("flatten_order must be 'C' or 'F'")

    n_state = int(coefficients.shape[1])
    if include_spin_flavor_blocks:
        micro_dim = basis_dim * spin_count * n_flavor
        reconstructed_flat = np.zeros((nk, micro_dim, n_state), dtype=np.complex128)
        coeff = coefficients.reshape((spin_count, n_flavor, n_band, n_state, nk), order=flatten_order)
        for ispin in range(spin_count):
            for iflavor in range(n_flavor):
                block = (ispin * n_flavor + iflavor) * basis_dim
                block_slice = slice(block, block + basis_dim)
                # basis[:, band, flavor, k] times coeff[spin, flavor, band, state, k]
                reconstructed_flat[:, block_slice, :] = np.einsum(
                    "xbk,bsk->kxs",
                    basis[:, :, iflavor, :],
                    coeff[ispin, iflavor],
                    optimize=True,
                )
    else:
        micro_dim = basis_dim
        reconstructed_flat = np.zeros((nk, micro_dim, n_state), dtype=np.complex128)
        coeff = coefficients.reshape((spin_count, n_flavor, n_band, n_state, nk), order=flatten_order)
        for ispin in range(spin_count):
            for iflavor in range(n_flavor):
                reconstructed_flat += np.einsum(
                    "xbk,bsk->kxs",
                    basis[:, :, iflavor, :],
                    coeff[ispin, iflavor],
                    optimize=True,
                )

    return reconstructed_flat.reshape((int(mesh_shape[0]), int(mesh_shape[1]), micro_dim, n_state), order="C")


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
    "reconstruct_projected_micro_wavefunctions",
    "reshape_flat_mesh_to_grid",
    "wavefunction_index_from_state_labels",
]
