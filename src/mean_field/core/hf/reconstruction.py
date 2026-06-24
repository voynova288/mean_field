from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
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

def direct_sum_active_index(dimensions: Iterable[int], *, context: str = "direct-sum active basis") -> np.ndarray:
    """Return Fortran-ordered active indices for a spin/flavor/band direct sum."""
    resolved = tuple(int(value) for value in dimensions)
    if not resolved or any(value <= 0 for value in resolved): raise ValueError(f"{context} dimensions must be positive, got {resolved}")
    return np.arange(int(np.prod(resolved)), dtype=int).reshape(resolved, order="F")

def normalize_reconstruction_state_indices(state_indices: int | Iterable[int] | None, n_state: int, *, label: str = "HF state") -> tuple[int, ...]:
    """Normalize selected reconstructed-state indices with explicit bounds checks."""
    n_state_i = int(n_state)
    if n_state_i <= 0: raise ValueError(f"n_state must be positive, got {n_state}")
    if state_indices is None: return tuple(range(n_state_i))
    if isinstance(state_indices, (str, bytes)): raise TypeError(f"{label} indices must be an integer, an iterable of integers, or None")
    out = (int(state_indices),) if isinstance(state_indices, (int, np.integer)) else tuple(int(index) for index in state_indices)
    if not out: raise ValueError(f"At least one {label} index is required")
    if len(set(out)) != len(out): raise ValueError(f"Duplicate {label} indices {out}")
    if min(out) < 0 or max(out) >= n_state_i: raise ValueError(f"{label} indices {out} outside [0, {n_state_i})")
    return out

def active_eigenvector_unitarity_residual(coeffs: np.ndarray, *, unitarity_atol: float | None, context: str = "active_eigenvectors") -> float | None:
    """Return/max-check the column Gram residual for active-HF eigenvectors."""
    if unitarity_atol is None: return None
    tolerance = float(unitarity_atol)
    if tolerance < 0.0: raise ValueError(f"unitarity_atol must be non-negative or None, got {unitarity_atol}")
    arr = np.asarray(coeffs, dtype=np.complex128)
    if arr.ndim != 3: raise ValueError(f"{context} must have shape (active_basis, hf_state, k), got {arr.shape}")
    n_state = int(arr.shape[1])
    gram = np.einsum("ahk,amk->hmk", arr.conjugate(), arr, optimize=True)
    residual = float(np.max(np.abs(gram - np.eye(n_state, dtype=np.complex128)[:, :, None])))
    if residual > tolerance: raise ValueError(f"{context} must be orthonormal at each k point; max column-Gram residual {residual:.6e} exceeds {tolerance:.6e}")
    return residual

def _direct_sum_basis_inputs(raw_projected_basis: np.ndarray, active_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int]:
    raw = np.asarray(raw_projected_basis, dtype=np.complex128)
    active = np.asarray(active_index, dtype=int)
    if raw.ndim != 4: raise ValueError(f"raw_projected_basis must have shape (basis, band, flavor, k), got {raw.shape}")
    if active.ndim != 3: raise ValueError(f"active_index must have shape (spin, flavor, band), got {active.shape}")
    basis_dim, n_band, n_flavor, n_k = (int(value) for value in raw.shape)
    n_spin, active_flavor, active_band = (int(value) for value in active.shape)
    if (active_flavor, active_band) != (n_flavor, n_band): raise ValueError(f"active_index shape {active.shape} is incompatible with raw basis shape {raw.shape}")
    n_active = int(active.size)
    flat = active.reshape(-1)
    if n_active <= 0:
        raise ValueError("active_index must be a permutation of the compact direct-sum active basis")
    if int(flat.min()) < 0 or int(flat.max()) >= n_active or len(np.unique(flat)) != n_active:
        raise ValueError("active_index must be a permutation of the compact direct-sum active basis")
    return raw, active, basis_dim, n_band, n_flavor, n_k, n_spin, n_active

def expand_direct_sum_projected_micro_basis(raw_projected_basis: np.ndarray, active_index: np.ndarray) -> np.ndarray:
    """Expand compact ``(basis, band, flavor, k)`` basis into ``(k, micro, active)``."""
    raw, active, basis_dim, _n_band, n_flavor, n_k, n_spin, n_active = _direct_sum_basis_inputs(raw_projected_basis, active_index)
    expanded = np.zeros((n_k, n_spin * n_flavor * basis_dim, n_active), dtype=np.complex128)
    for ispin in range(n_spin):
        for iflavor in range(n_flavor):
            row_start = (ispin * n_flavor + iflavor) * basis_dim
            expanded[:, row_start : row_start + basis_dim, active[ispin, iflavor, :]] = np.transpose(raw[:, :, iflavor, :], (2, 0, 1))
    return expanded

def contract_direct_sum_projected_micro_wavefunctions(raw_projected_basis: np.ndarray, active_coefficients: np.ndarray, active_index: np.ndarray) -> np.ndarray:
    """Contract compact direct-sum projected basis with selected active coefficients."""
    raw, active, basis_dim, _n_band, n_flavor, n_k, n_spin, n_active = _direct_sum_basis_inputs(raw_projected_basis, active_index)
    coeffs = np.asarray(active_coefficients, dtype=np.complex128)
    if coeffs.ndim != 3 or coeffs.shape[0] != n_active or coeffs.shape[2] != n_k:
        raise ValueError(f"active_coefficients must have shape ({n_active}, n_state, {n_k}), got {coeffs.shape}")
    psi_flat = np.zeros((n_k, n_spin * n_flavor * basis_dim, int(coeffs.shape[1])), dtype=np.complex128)
    for ispin in range(n_spin):
        for iflavor in range(n_flavor):
            row_start = (ispin * n_flavor + iflavor) * basis_dim
            rows = active[ispin, iflavor, :]
            psi_flat[:, row_start : row_start + basis_dim, :] = np.einsum("pbk,bhk->kph", raw[:, :, iflavor, :], coeffs[rows, :, :], optimize=True)
    return psi_flat


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


__all__ = [
    "active_eigenvector_unitarity_residual",
    "canonicalize_projected_micro_basis",
    "contract_direct_sum_projected_micro_wavefunctions",
    "direct_sum_active_index",
    "expand_direct_sum_projected_micro_basis",
    "normalize_reconstruction_state_indices",
    "reconstruct_projected_micro_wavefunctions",
]
