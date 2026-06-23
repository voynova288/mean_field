from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import numpy as np
from mean_field.core.contracts import MicroscopicWavefunctionBundle, MicroscopicWavefunctionSource


def _axis(axis: int, ndim: int) -> int:
    out = int(axis) + (int(ndim) if int(axis) < 0 else 0)
    if out < 0 or out >= int(ndim): raise ValueError(f"Axis {axis} is out of bounds for ndim={ndim}")
    return out


def canonicalize_projected_micro_basis(micro_basis: np.ndarray, *, k_axis: int = 0, microscopic_basis_axis: int = 1, active_axis: int = 2) -> np.ndarray:
    """Return projected microscopic basis as ``(k, microscopic_basis, active_basis)``."""
    arr = np.asarray(micro_basis, dtype=np.complex128)
    if arr.ndim != 3: raise ValueError(f"micro_basis must have exactly three axes: k, microscopic_basis, active_basis; got shape {arr.shape}")
    axes = (_axis(k_axis, arr.ndim), _axis(microscopic_basis_axis, arr.ndim), _axis(active_axis, arr.ndim))
    if len(set(axes)) != 3: raise ValueError(f"k, microscopic_basis, and active axes must be distinct, got {axes}")
    return np.transpose(arr, axes=axes)


def _labels(labels: Sequence[Mapping[str, object]] | None, n_state: int) -> tuple[dict[str, object], ...]:
    if labels is None: return tuple({"hf_state_index": int(i)} for i in range(int(n_state)))
    out = tuple(dict(label) for label in labels)
    if len(out) != int(n_state): raise ValueError(f"state_labels length {len(out)} must match n_state={n_state}")
    return out


def _grid_shape(shape: tuple[int, int] | None, n_k: int) -> tuple[int, int] | None:
    if shape is None: return None
    if len(shape) != 2: raise ValueError(f"grid_shape must have length two, got {shape}")
    out = (int(shape[0]), int(shape[1]))
    if out[0] <= 0 or out[1] <= 0 or out[0] * out[1] != int(n_k): raise ValueError(f"grid_shape={shape} is incompatible with n_k={n_k}")
    return out


def reconstruct_projected_micro_wavefunctions(
    micro_basis: np.ndarray,
    active_eigenvectors: np.ndarray,
    *,
    kvec: np.ndarray | None = None,
    k_grid_frac: np.ndarray | None = None,
    grid_shape: tuple[int, int] | None = None,
    k_axis: int = 0,
    microscopic_basis_axis: int = 1,
    active_axis: int = 2,
    state_labels: Sequence[Mapping[str, object]] | None = None,
    sewing_transforms: Sequence[Callable[..., Any]] = (),
    basis_metadata: Mapping[str, Any] | None = None,
    source: MicroscopicWavefunctionSource = "hf_reconstructed",
    unitarity_atol: float | None = 1.0e-8,
) -> MicroscopicWavefunctionBundle:
    """Reconstruct ``psi_micro`` from canonical micro basis and active HF eigenvectors."""
    basis = canonicalize_projected_micro_basis(micro_basis, k_axis=k_axis, microscopic_basis_axis=microscopic_basis_axis, active_axis=active_axis)
    n_k, micro_dim, n_active = (int(v) for v in basis.shape)
    coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
    if coeffs.shape != (n_active, n_active, n_k): raise ValueError(f"active_eigenvectors must have shape ({n_active}, {n_active}, {n_k}), got {coeffs.shape}")
    residual = None
    if unitarity_atol is not None:
        if float(unitarity_atol) < 0.0: raise ValueError(f"unitarity_atol must be non-negative or None, got {unitarity_atol}")
        gram = np.einsum("ahk,amk->hmk", coeffs.conjugate(), coeffs, optimize=True)
        residual = float(np.max(np.abs(gram - np.eye(n_active, dtype=np.complex128)[:, :, None])))
        if residual > float(unitarity_atol): raise ValueError(f"active_eigenvectors must be unitary at each k point; max column-Gram residual {residual:.6e} exceeds {float(unitarity_atol):.6e}")
    kvec_arr = np.arange(n_k, dtype=np.complex128) if kvec is None else np.asarray(kvec, dtype=np.complex128).reshape(-1)
    if kvec_arr.shape != (n_k,): raise ValueError(f"kvec must have shape ({n_k},), got {kvec_arr.shape}")
    if k_grid_frac is not None and np.asarray(k_grid_frac, dtype=float).shape != (n_k, 2): raise ValueError(f"k_grid_frac must have shape ({n_k}, 2), got {np.asarray(k_grid_frac).shape}")
    shape = _grid_shape(grid_shape, n_k)
    psi_flat = np.einsum("kba,ahk->kbh", basis, coeffs, optimize=True)
    psi = psi_flat if shape is None else psi_flat.reshape((*shape, micro_dim, n_active), order="C")
    metadata = dict(basis_metadata or {})
    metadata.update({"micro_basis_axis_order": "k,microscopic_basis,active_basis", "input_micro_basis_axes": {"k_axis": int(k_axis), "microscopic_basis_axis": int(microscopic_basis_axis), "active_axis": int(active_axis)}, "active_eigenvectors_axis_order": "active_basis,hf_state,k", "psi_micro_axis_order": "k,microscopic_basis,hf_state" if shape is None else "mesh_1,mesh_2,microscopic_basis,hf_state", "n_k": n_k, "microscopic_basis_dim": micro_dim, "n_active": n_active, "state_labels": _labels(state_labels, n_active), "kvec_provided": kvec is not None})
    if residual is not None: metadata["active_eigenvectors_unitarity_residual"] = residual
    if shape is not None: metadata["grid_shape"] = shape
    if k_grid_frac is not None: metadata["k_grid_frac_shape"] = [n_k, 2]
    return MicroscopicWavefunctionBundle(kvec=kvec_arr, psi_micro=psi, sewing_transforms=tuple(sewing_transforms), basis_metadata=metadata, source=source)


__all__ = ["canonicalize_projected_micro_basis", "reconstruct_projected_micro_wavefunctions"]
