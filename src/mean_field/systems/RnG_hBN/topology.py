from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from analysis.topology import BlockSewingSpec, FHSState, fhs_state_from_grid_result as _state_from_grid, fhs_state_from_wavefunctions, normalize_state_indices

from .bands import compute_bands_on_grid


def rlg_hbn_basis_sewing(lattice, params, *, valley: int = 1, atol: float = 1.0e-8) -> BlockSewingSpec:
    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    return BlockSewingSpec(
        block_coordinates=np.asarray(lattice.g_indices, dtype=float),
        local_block_size=2 * int(params.layer_count),
        translations=((float(valley_sign), 0.0), (0.0, float(valley_sign))),
        atol=float(atol),
    )


def _orientation_sign(*, orientation_sign: float | None, paper_orientation: bool) -> float:
    return float(orientation_sign) if orientation_sign is not None else (-1.0 if bool(paper_orientation) else 1.0)


def fhs_state_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    basis_sewing: BlockSewingSpec | None = None,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> FHSState:
    sign = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        orientation_sign=sign,
        system="RLG_hBN",
        valley=valley,
        metadata={"boundary_sewing": basis_sewing is not None, "orientation_sign": float(sign)},
        reported_indices=band_indices,
    )


def fhs_state_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    lattice=None,
    params=None,
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> FHSState:
    if basis_sewing is None and use_boundary_sewing and lattice is not None and params is not None:
        basis_sewing = rlg_hbn_basis_sewing(lattice, params, valley=valley)
    sign = _orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return _state_from_grid(
        grid_result,
        band_indices,
        basis_sewing=basis_sewing,
        orientation_sign=sign,
        system="RLG_hBN",
        valley=valley,
        metadata={"boundary_sewing": basis_sewing is not None, "orientation_sign": float(sign)},
    )


def fhs_state_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    basis_sewing: BlockSewingSpec | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
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
        lattice=lattice,
        params=params,
        basis_sewing=basis_sewing,
        use_boundary_sewing=use_boundary_sewing,
        orientation_sign=orientation_sign,
        paper_orientation=paper_orientation,
    )


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_on_grid",
    "rlg_hbn_basis_sewing",
]
