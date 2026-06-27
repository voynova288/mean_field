from __future__ import annotations

from collections.abc import Iterable, Mapping
import numpy as np

from analysis.topology import BlockSewingSpec, FHSState, fhs_state_from_grid_result as _state_from_grid, fhs_state_from_wavefunctions, normalize_state_indices

from .bands import compute_bands_on_grid


def htg_basis_sewing(lattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    return BlockSewingSpec(
        block_coordinates=np.asarray(lattice.g_indices, dtype=float),
        local_block_size=6,
        translations=((1.0, 0.0), (0.0, 1.0)),
        atol=float(atol),
    )


def _metadata(basis_sewing: BlockSewingSpec | None, metadata: Mapping[str, object] | None) -> dict[str, object]:
    payload = {"boundary_sewing": basis_sewing is not None}
    payload.update(dict(metadata or {}))
    return payload


def _window(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    requested = normalize_state_indices(band_indices)
    return tuple(range(min(requested), max(requested) + 1))


def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
    metadata: Mapping[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> FHSState:
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        system="htg",
        valley=valley,
        reported_indices=band_indices,
        orientation_sign=float(orientation_sign),
        metadata=_metadata(basis_sewing, metadata),
    )


def fhs_state_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    basis_sewing: BlockSewingSpec | None = None,
    metadata: Mapping[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> FHSState:
    return _state_from_grid(
        grid_result,
        band_indices,
        basis_sewing=basis_sewing,
        system="htg",
        valley=valley,
        orientation_sign=float(orientation_sign),
        metadata=_metadata(basis_sewing, metadata),
    )


def fhs_state_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    boundary_sewing: bool = True,
    orientation_sign: float = 1.0,
) -> FHSState:
    requested = normalize_state_indices(band_indices)
    if endpoint:
        raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(
        int(mesh_size),
        lattice,
        params,
        valley=int(valley),
        d_top=d_top,
        d_bot=d_bot,
        band_indices=_window(requested),
        return_eigenvectors=True,
        endpoint=False,
        frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    return fhs_state_from_grid_result(
        grid,
        requested,
        valley=int(valley),
        basis_sewing=htg_basis_sewing(lattice) if bool(boundary_sewing) else None,
        orientation_sign=float(orientation_sign),
    )


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
    "htg_basis_sewing",
]
