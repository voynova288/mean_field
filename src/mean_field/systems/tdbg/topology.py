from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import numpy as np

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    SewingTransform,
    fhs_state_from_grid_result as _state_from_grid,
    fhs_state_from_wavefunctions,
    normalize_state_indices,
)

from .bands import compute_bands_on_grid
from .lattice import TDBGLattice
from .params import TDBGParameters
from .projected_hf_geometry import (
    tdbg_projected_hf_boundary_sewing_transforms as projected_hf_boundary_sewing_transforms,
    tdbg_projected_hf_q_site_sewing_transform as projected_hf_q_site_sewing_transform,
    translation_srcmap,
)
from .projected_hf_state import reconstruct_tdbg_projected_hf_micro_wavefunctions


def _reshape_flat_mesh_to_grid(values: np.ndarray, mesh_shape: tuple[int, int], *, k_axis: int = 0, order: str = "C") -> np.ndarray:
    array = np.asarray(values)
    mesh_1, mesh_2 = int(mesh_shape[0]), int(mesh_shape[1])
    moved = np.moveaxis(array, int(k_axis), 0)
    if moved.shape[0] != mesh_1 * mesh_2:
        raise ValueError(f"flat k-axis length {moved.shape[0]} is incompatible with mesh_shape={mesh_shape}")
    return moved.reshape((mesh_1, mesh_2) + moved.shape[1:], order=order)


def tdbg_basis_sewing(lattice: TDBGLattice, *, atol: float = 1.0e-8) -> BlockSewingSpec:
    q_sites = np.asarray(lattice.q_sites, dtype=float)
    if q_sites.ndim != 2 or q_sites.shape[1] < 3:
        raise ValueError(f"Expected lattice.q_sites shape (n, >=3), got {q_sites.shape}")
    return BlockSewingSpec(
        block_coordinates=q_sites[:, :2],
        local_block_size=4,
        translations=((float(lattice.g_m1.real), float(lattice.g_m1.imag)), (float(lattice.g_m2.real), float(lattice.g_m2.imag))),
        block_labels=q_sites[:, 2].astype(int),
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
    metadata: Mapping[str, object] | None = None,
) -> FHSState:
    payload = {"boundary_sewing": basis_sewing is not None or sewing_transforms is not None}
    payload.update(dict(metadata or {}))
    return fhs_state_from_wavefunctions(
        eigenvectors,
        band_indices,
        k_grid_frac=k_grid_frac,
        basis_sewing=basis_sewing,
        sewing_transforms=sewing_transforms,
        system="tdbg",
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
    metadata: Mapping[str, object] | None = None,
) -> FHSState:
    payload = {"boundary_sewing": basis_sewing is not None}
    payload.update(dict(metadata or {}))
    return _state_from_grid(grid_result, band_indices, basis_sewing=basis_sewing, system="tdbg", valley=valley, metadata=payload)


def fhs_state_on_grid(
    mesh_size: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    n_bands: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    boundary_sewing: bool = True,
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
        basis_sewing=tdbg_basis_sewing(lattice) if bool(boundary_sewing) else None,
        metadata={"boundary_sewing": bool(boundary_sewing)},
    )


def _projected_hf_mesh_shape(basis_metadata: Mapping[str, object], *, n_k: int, mesh_shape: tuple[int, int] | None) -> tuple[tuple[int, int], str]:
    raw = mesh_shape
    source = "mesh_shape argument"
    if raw is None:
        raw = basis_metadata.get("topology_grid_shape", basis_metadata.get("grid_shape"))
        source = str(basis_metadata.get("topology_grid_shape_source", "basis_metadata grid_shape/topology_grid_shape"))
    if raw is None:
        raise ValueError("TDBG projected-HF FHS state requires topology_grid_shape/grid_shape metadata or mesh_shape")
    shape = tuple(int(v) for v in raw)  # type: ignore[arg-type]
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[0] * shape[1] != int(n_k):
        raise ValueError(f"TDBG projected-HF mesh shape {shape} from {source} is incompatible with n_k={n_k}")
    return shape, source


def _projected_hf_k_grid_frac(result, mesh_shape: tuple[int, int], k_grid_frac) -> np.ndarray | None:
    if k_grid_frac is None:
        data = getattr(result, "data", None)
        if data is None or not hasattr(data, "k_grid_frac"):
            return None
        raw = np.asarray(getattr(data, "k_grid_frac"), dtype=float)
    else:
        raw = np.asarray(k_grid_frac, dtype=float)
    if raw.shape == mesh_shape + (2,):
        return raw
    if raw.shape == (mesh_shape[0] * mesh_shape[1], 2):
        return _reshape_flat_mesh_to_grid(raw, mesh_shape, k_axis=0, order="C")
    raise ValueError(f"TDBG projected-HF k_grid_frac has incompatible shape {raw.shape}")


def fhs_state_from_projected_hf(
    result,
    state_indices: int | Iterable[int] | None = None,
    *,
    band_indices: int | Iterable[int] | None = None,
    valley: int = 0,
    mesh_shape: tuple[int, int] | None = None,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    max_dense_elements: int | None = 5_000_000,
    hermiticity_atol: float = 1.0e-8,
    unitarity_atol: float | None = 1.0e-8,
    metadata: Mapping[str, object] | None = None,
) -> FHSState:
    core_bundle = reconstruct_tdbg_projected_hf_micro_wavefunctions(
        result,
        state_indices=state_indices,
        band_indices=band_indices,
        max_dense_elements=max_dense_elements,
        hermiticity_atol=hermiticity_atol,
        unitarity_atol=unitarity_atol,
    )
    basis_metadata = dict(core_bundle.basis_metadata)
    psi_flat = np.asarray(core_bundle.psi_micro, dtype=np.complex128)
    shape, shape_source = _projected_hf_mesh_shape(basis_metadata, n_k=int(psi_flat.shape[0]), mesh_shape=mesh_shape)
    psi_grid = _reshape_flat_mesh_to_grid(psi_flat, shape, k_axis=0, order="C")
    active_sewing = tuple(core_bundle.sewing_transforms) if sewing_transforms is None else tuple(sewing_transforms)
    selected = tuple(int(index) for index in basis_metadata.get("selected_hf_state_indices", range(psi_grid.shape[-1])))
    payload = dict(basis_metadata)
    payload.update(
        {
            "topology_adapter": "mean_field.systems.tdbg.topology.fhs_state_from_projected_hf",
            "topology_input_axis_order": "mesh,mesh,basis,state",
            "topology_flat_input_axis_order": "nk,basis,state",
            "topology_grid_shape": [int(shape[0]), int(shape[1])],
            "topology_grid_shape_source": shape_source,
            "topology_sewing_transforms_count": int(len(active_sewing)),
            "absolute_band_indices": [int(index) for index in selected],
            "column_indices": list(range(psi_grid.shape[-1])),
        }
    )
    payload.update(dict(metadata or {}))
    return fhs_state_from_wavefunctions(
        psi_grid,
        tuple(range(psi_grid.shape[-1])),
        k_grid_frac=_projected_hf_k_grid_frac(result, shape, k_grid_frac),
        sewing_transforms=active_sewing,
        system="tdbg",
        valley=int(valley),
        labels=tuple(f"hf_state={index}" for index in selected),
        reported_indices=selected,
        metadata=payload,
    )


__all__ = [
    "BlockSewingSpec",
    "FHSState",
    "SewingTransform",
    "fhs_state_from_eigenvectors",
    "fhs_state_from_grid_result",
    "fhs_state_from_projected_hf",
    "fhs_state_on_grid",
    "projected_hf_boundary_sewing_transforms",
    "projected_hf_q_site_sewing_transform",
    "tdbg_basis_sewing",
    "translation_srcmap",
]
