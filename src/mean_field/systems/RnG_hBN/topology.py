from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np

from analysis.topology import (
    SewingTransform,
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    compute_system_topology_on_grid,
    normalize_state_indices,
)

from .bands import GridBandsResult, compute_bands_on_grid
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    return normalize_state_indices(band_indices)


def _shift_rectangular_g_components(
    vectors: np.ndarray,
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
) -> np.ndarray:
    """Shift embedded raw-G components, dropping entries outside the padded grid.

    This is a system-specific boundary transition map for RLG/hBN microscopic
    wavefunctions embedded on the rectangular reciprocal grid used by the
    projected-basis cache.  The Berry/FHS computation itself remains in
    :mod:`analysis.topology`.
    """

    local = int(local_basis_size)
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    sx, sy = (int(shift[0]), int(shift[1]))
    block_dim = local * nx * ny
    array = np.asarray(vectors, dtype=np.complex128)
    if array.shape[0] != block_dim:
        raise ValueError(f"Expected first axis {block_dim}, got {array.shape[0]}")
    frame_count = int(np.prod(array.shape[1:], dtype=int)) if array.ndim > 1 else 1
    reshaped = array.reshape((local, nx, ny, frame_count), order="F")
    shifted = np.zeros_like(reshaped)
    for ix in range(nx):
        tx = ix + sx
        if tx < 0 or tx >= nx:
            continue
        for iy in range(ny):
            ty = iy + sy
            if ty < 0 or ty >= ny:
                continue
            shifted[:, tx, ty, :] = reshaped[:, ix, iy, :]
    return shifted.reshape(array.shape, order="F")


def rlg_hbn_reciprocal_shift_sewing_transform(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
    valley: int,
) -> SewingTransform:
    """Return a single-valley RLG/hBN raw-G boundary sewing transform.

    For valley ``+1``/``-1``, a reciprocal boundary shift ``G`` is represented
    in the cached periodic gauge by relabeling raw components with ``-valley*G``.
    Pass the returned callable to ``analysis.topology.compute_lattice_topology``
    as one entry of ``sewing_transforms``.
    """

    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    component_shift = (-valley_sign * int(shift[0]), -valley_sign * int(shift[1]))

    def transform(vectors: np.ndarray) -> np.ndarray:
        return _shift_rectangular_g_components(
            vectors,
            local_basis_size=int(local_basis_size),
            grid_shape=grid_shape,
            shift=component_shift,
        )

    return transform


def rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    shift: tuple[int, int],
    spin_count: int = 2,
    valley_signs: Sequence[int] = (1, -1),
) -> SewingTransform:
    """Return a full spin/flavor-block RLG/hBN raw-G sewing transform.

    Use this for HF source-state microscopic wavefunctions reconstructed with
    ``include_spin_flavor_blocks=True``.  The vector layout is block-major in
    ``(spin, valley)`` with each block containing the embedded raw-G basis.
    """

    local = int(local_basis_size)
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    block_dim = local * nx * ny
    valleys = tuple(int(value) for value in valley_signs)
    if any(value not in {-1, 1} for value in valleys):
        raise ValueError(f"Expected valley signs ±1, got {valley_signs!r}")
    n_spin = int(spin_count)
    n_flavor = len(valleys)
    total_dim = n_spin * n_flavor * block_dim

    def transform(vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.complex128)
        one_dimensional = array.ndim == 1
        matrix = array[:, None] if one_dimensional else array.reshape((array.shape[0], -1), order="F")
        if matrix.shape[0] != total_dim:
            raise ValueError(f"Expected first axis {total_dim}, got {matrix.shape[0]}")
        out = np.zeros_like(matrix)
        for ispin in range(n_spin):
            for iflavor, valley_sign in enumerate(valleys):
                start = (ispin * n_flavor + iflavor) * block_dim
                stop = start + block_dim
                component_shift = (-valley_sign * int(shift[0]), -valley_sign * int(shift[1]))
                out[start:stop, :] = _shift_rectangular_g_components(
                    matrix[start:stop, :],
                    local_basis_size=local,
                    grid_shape=grid_shape,
                    shift=component_shift,
                )
        if one_dimensional:
            return out[:, 0]
        return out.reshape(array.shape, order="F")

    return transform


def rlg_hbn_projected_micro_sewing_transforms(
    *,
    local_basis_size: int,
    grid_shape: tuple[int, int],
    spin_count: int = 2,
    valley_signs: Sequence[int] = (1, -1),
) -> tuple[SewingTransform, SewingTransform]:
    """Boundary sewing transforms for reconstructed RLG/hBN HF microstates."""

    return (
        rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
            local_basis_size=local_basis_size,
            grid_shape=grid_shape,
            shift=(1, 0),
            spin_count=spin_count,
            valley_signs=valley_signs,
        ),
        rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform(
            local_basis_size=local_basis_size,
            grid_shape=grid_shape,
            shift=(0, 1),
            spin_count=spin_count,
            valley_signs=valley_signs,
        ),
    )


def _reciprocal_translation(
    lattice: RLGhBNLattice,
    *,
    layer_count: int,
    dn1: int,
    dn2: int,
    valley: int = 1,
) -> SewingTransform:
    """Return the plane-wave G-relabeling map for an RLG/hBN BZ edge.

    ``build_hamiltonian(k)`` orders the microscopic basis by ``G`` and then by
    layer/sublattice.  Since ``H(k + b_i)`` equals ``H(k)`` only after
    relabeling ``G -> G + b_i`` (with the sign reversed for the conjugated
    ``K'`` implementation), FHS links across the torus boundary must apply this
    transition map.  Omitting it glues two incompatible plane-wave gauges.
    """

    valley_sign = int(valley)
    if valley_sign not in {-1, 1}:
        raise ValueError(f"Expected valley ±1, got {valley!r}")
    index_by_g = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}
    block = 2 * int(layer_count)
    source_dn1 = valley_sign * int(dn1)
    source_dn2 = valley_sign * int(dn2)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        if array.shape[0] != block * lattice.n_g:
            raise ValueError(f"Expected first axis {block * lattice.n_g}, got {array.shape[0]}")
        out = np.zeros_like(array)
        for target_index, (n1, n2) in enumerate(lattice.g_indices):
            source_index = index_by_g.get((int(n1) + source_dn1, int(n2) + source_dn2))
            if source_index is None:
                continue
            out[block * target_index : block * (target_index + 1), ...] = array[
                block * source_index : block * (source_index + 1), ...
            ]
        return out

    return apply


def rlg_hbn_boundary_sewing_transforms(
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
) -> tuple[SewingTransform, SewingTransform]:
    """Boundary sewing transforms for ordinary RLG/hBN plane-wave eigenvectors."""

    return (
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=1, dn2=0, valley=valley),
        _reciprocal_translation(lattice, layer_count=params.layer_count, dn1=0, dn2=1, valley=valley),
    )


def _resolve_orientation_sign(*, orientation_sign: float | None, paper_orientation: bool) -> float:
    if orientation_sign is not None:
        return float(orientation_sign)
    return -1.0 if bool(paper_orientation) else 1.0


def _resolve_boundary_sewing(
    sewing_transforms: Sequence[SewingTransform | None] | None,
    *,
    lattice: RLGhBNLattice | None,
    params: RLGhBNParams | None,
    valley: int,
    use_boundary_sewing: bool,
) -> Sequence[SewingTransform | None] | None:
    if sewing_transforms is not None or not bool(use_boundary_sewing):
        return sewing_transforms
    if lattice is None or params is None:
        return None
    return rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> TopologyResult:
    """Compute topology from an already-built RLG/hBN eigenvector grid.

    Boundary sewing is only automatic for APIs that receive ``lattice`` and
    ``params``.  For raw eigenvectors, pass ``sewing_transforms`` explicitly if
    the vectors are in the ordinary plane-wave gauge.
    """

    resolved_orientation = _resolve_orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system="RLG_hBN",
        valley=valley,
        k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms,
        orientation_sign=resolved_orientation,
        index_metadata={"orientation_sign": float(resolved_orientation)},
    )


def compute_topology_from_grid_result(
    grid_result: GridBandsResult,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    lattice: RLGhBNLattice | None = None,
    params: RLGhBNParams | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> TopologyResult:
    resolved_orientation = _resolve_orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    resolved_sewing = _resolve_boundary_sewing(
        sewing_transforms,
        lattice=lattice,
        params=params,
        valley=valley,
        use_boundary_sewing=use_boundary_sewing,
    )
    return compute_system_topology_from_grid_result(
        grid_result,
        band_indices,
        system="RLG_hBN",
        valley=valley,
        sewing_transforms=resolved_sewing,
        orientation_sign=resolved_orientation,
        index_metadata={
            "boundary_sewing": resolved_sewing is not None,
            "orientation_sign": float(resolved_orientation),
        },
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    use_boundary_sewing: bool = True,
    orientation_sign: float | None = None,
    paper_orientation: bool = False,
) -> TopologyResult:
    def grid_builder(trial_mesh: int, frac_shift: tuple[float, float], resolved_n_bands: int) -> GridBandsResult:
        return compute_bands_on_grid(
            trial_mesh,
            lattice,
            params,
            valley=valley,
            n_bands=resolved_n_bands,
            return_eigenvectors=True,
            endpoint=endpoint,
            frac_shift=frac_shift,
        )

    resolved_orientation = _resolve_orientation_sign(orientation_sign=orientation_sign, paper_orientation=paper_orientation)
    sewing_builder = None
    if sewing_transforms is None and bool(use_boundary_sewing):
        sewing_builder = lambda: rlg_hbn_boundary_sewing_transforms(lattice, params, valley=valley)

    return compute_system_topology_on_grid(
        mesh_size,
        band_indices,
        system="RLG_hBN",
        grid_builder=grid_builder,
        valley=valley,
        n_bands=n_bands,
        sewing_transforms=sewing_transforms,
        sewing_transforms_builder=sewing_builder,
        orientation_sign=resolved_orientation,
        index_metadata={
            "boundary_sewing": sewing_transforms is not None or sewing_builder is not None,
            "orientation_sign": float(resolved_orientation),
        },
    )


__all__ = [
    "SewingTransform",
    "TopologyResult",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "rlg_hbn_boundary_sewing_transforms",
    "rlg_hbn_projected_micro_sewing_transforms",
    "rlg_hbn_reciprocal_shift_sewing_transform",
    "rlg_hbn_spin_flavor_reciprocal_shift_sewing_transform",
]
