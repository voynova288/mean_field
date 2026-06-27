from __future__ import annotations

from collections.abc import Iterable, Sequence
import numpy as np

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    SewingTransform,
    fhs_state_from_grid_result as _state_from_grid,
    fhs_state_from_wavefunctions,
    normalize_state_indices,
)

from ._polshyn_reconstruction import reconstruct_polshyn_wang_hf_micro_wavefunctions
from ._polshyn_types import PolshynProjectedBasis
from .bands import compute_bands_on_grid


def _reshape_flat_mesh_to_grid(values: np.ndarray, mesh_shape: tuple[int, int], *, k_axis: int = 0, order: str = "C") -> np.ndarray:
    array = np.asarray(values)
    mesh_1, mesh_2 = int(mesh_shape[0]), int(mesh_shape[1])
    moved = np.moveaxis(array, int(k_axis), 0)
    if moved.shape[0] != mesh_1 * mesh_2:
        raise ValueError(f"flat k-axis length {moved.shape[0]} is incompatible with mesh_shape={mesh_shape}")
    return moved.reshape((mesh_1, mesh_2) + moved.shape[1:], order=order)


def tmbg_basis_sewing(lattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    return BlockSewingSpec(
        block_coordinates=np.asarray(lattice.g_indices, dtype=float),
        local_block_size=6,
        translations=((1.0, 0.0), (0.0, 1.0)),
        atol=float(atol),
    )


def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    payload = {"boundary_sewing": basis_sewing is not None or sewing_transforms is not None}
    payload.update(dict(metadata or {}))
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        sewing_transforms=sewing_transforms,
        orientation_sign=float(orientation_sign),
        system="tmbg",
        valley=valley,
        reported_indices=band_indices,
        metadata=payload,
    )


def fhs_state_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    payload = {"boundary_sewing": basis_sewing is not None or sewing_transforms is not None}
    payload.update(dict(metadata or {}))
    return _state_from_grid(
        grid_result,
        band_indices,
        basis_sewing=basis_sewing,
        sewing_transforms=sewing_transforms,
        orientation_sign=float(orientation_sign),
        system="tmbg",
        valley=valley,
        metadata=payload,
    )


def fhs_state_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    n_bands: int | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float = 1.0,
) -> FHSState:
    requested = normalize_state_indices(band_indices)
    if n_bands is not None and int(n_bands) <= max(requested):
        raise ValueError(f"n_bands={int(n_bands)} does not include requested band index {max(requested)}")
    if endpoint:
        raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(
        int(mesh_size),
        lattice,
        params,
        valley=int(valley),
        n_bands=None if n_bands is None else int(n_bands),
        return_eigenvectors=True,
        endpoint=False,
        frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    return fhs_state_from_grid_result(
        grid,
        requested,
        valley=int(valley),
        basis_sewing=tmbg_basis_sewing(lattice) if use_boundary_sewing else None,
        orientation_sign=float(orientation_sign),
    )


def _polshyn_mesh_shape(basis: PolshynProjectedBasis, *, n_k: int, mesh_shape: tuple[int, int] | None) -> tuple[int, int]:
    if mesh_shape is not None:
        shape = tuple(int(v) for v in mesh_shape)
    elif basis.k_grid_frac is not None:
        arr = np.asarray(basis.k_grid_frac, dtype=float)
        unique_b1 = np.unique(np.round(arr[:, 0], decimals=14))
        unique_b2 = np.unique(np.round(arr[:, 1], decimals=14))
        shape = (int(unique_b1.size), int(unique_b2.size))
    else:
        raise ValueError("Polshyn projected-HF FHS state requires explicit mesh_shape or basis.k_grid_frac")
    if len(shape) != 2 or shape[0] * shape[1] != int(n_k):
        raise ValueError(f"Polshyn projected-HF mesh_shape={shape} is incompatible with n_k={n_k}")
    return shape


def _polshyn_k_grid_frac(basis: PolshynProjectedBasis, mesh_shape: tuple[int, int], k_grid_frac) -> np.ndarray | None:
    raw = basis.k_grid_frac if k_grid_frac is None else k_grid_frac
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.shape == mesh_shape + (2,):
        return arr
    if arr.shape == (mesh_shape[0] * mesh_shape[1], 2):
        return _reshape_flat_mesh_to_grid(arr, mesh_shape, k_axis=0, order="F")
    raise ValueError(f"Polshyn projected-HF k_grid_frac has incompatible shape {arr.shape}")


def fhs_state_from_polshyn_projected_hf(
    basis: PolshynProjectedBasis,
    active_eigenvectors: np.ndarray,
    state_indices: int | Iterable[int] | None = None,
    *,
    band_indices: int | Iterable[int] | None = None,
    valley: int = 0,
    mesh_shape: tuple[int, int] | None = None,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
) -> FHSState:
    if state_indices is not None and band_indices is not None:
        raise ValueError("Pass either state_indices or band_indices, not both")
    selected = normalize_state_indices(0 if state_indices is None and band_indices is None else (state_indices if state_indices is not None else band_indices))
    bundle = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        active_eigenvectors,
        state_indices=selected,
        include_sewing=False,
    )
    psi_flat = np.asarray(bundle.psi_micro, dtype=np.complex128)
    shape = _polshyn_mesh_shape(basis, n_k=int(psi_flat.shape[0]), mesh_shape=mesh_shape)
    psi_grid = _reshape_flat_mesh_to_grid(psi_flat, shape, k_axis=0, order="F")
    payload = dict(getattr(bundle, "basis_metadata", {}))
    payload.update(
        {
            "topology_adapter": "mean_field.systems.tmbg.topology.fhs_state_from_polshyn_projected_hf",
            "topology_input_axis_order": "mesh,mesh,basis,state",
            "topology_grid_shape": [int(shape[0]), int(shape[1])],
            "absolute_band_indices": [int(v) for v in selected],
            "column_indices": list(range(psi_grid.shape[-1])),
        }
    )
    payload.update(dict(metadata or {}))
    return fhs_state_from_wavefunctions(
        psi_grid,
        tuple(range(psi_grid.shape[-1])),
        k_grid_frac=_polshyn_k_grid_frac(basis, shape, k_grid_frac),
        sewing_transforms=sewing_transforms,
        system="tmbg",
        valley=int(valley),
        labels=tuple(f"hf_state={idx}" for idx in selected),
        reported_indices=selected,
        metadata=payload,
    )


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "SewingTransform",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_from_polshyn_projected_hf",
    "fhs_state_on_grid",
    "tmbg_basis_sewing",
]
