from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from analysis.topology import (
    LinkMethod,
    SewingTransform,
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    normalize_state_indices,
    reshape_flat_mesh_to_grid,
)

from .bands import compute_bands_on_grid
from .projected_hf_geometry import (
    tdbg_projected_hf_boundary_sewing_transforms as projected_hf_boundary_sewing_transforms,
    tdbg_projected_hf_q_site_sewing_transform as projected_hf_q_site_sewing_transform,
    translation_srcmap,
)
from .projected_hf_state import reconstruct_tdbg_projected_hf_micro_wavefunctions


def _q_site_sewing_transform(lattice, gvec: complex) -> SewingTransform:
    src = translation_srcmap(lattice, complex(gvec))
    n_q = int(lattice.n_q)
    valid = src >= 0

    def transform(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.complex128)
        if arr.shape[0] != 4 * n_q:
            raise ValueError(f"Expected first axis {4 * n_q} for TDBG sewing, got {arr.shape}")
        reshaped = arr.reshape((n_q, 4) + arr.shape[1:], order="C")
        out = np.zeros_like(reshaped)
        out[valid] = reshaped[src[valid]]
        return out.reshape(arr.shape, order="C")

    return transform


def boundary_sewing_transforms(lattice) -> tuple[SewingTransform, SewingTransform]:
    return (_q_site_sewing_transform(lattice, lattice.g_m1), _q_site_sewing_transform(lattice, lattice.g_m2))


def _coerce_projected_hf_mesh_shape(raw_shape: object, *, n_k: int, source: str) -> tuple[int, int]:
    if raw_shape is None:
        raise ValueError(
            "TDBG projected-HF topology requires metadata topology_grid_shape/grid_shape to reshape "
            "flat (nk,basis,state) wavefunctions to (mesh,mesh,basis,state). "
            "Rebuild the result on a square endpoint=False mesh or pass mesh_shape explicitly."
        )
    try:
        mesh_shape = tuple(int(value) for value in raw_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"Could not read TDBG projected-HF mesh shape from {source}: {raw_shape!r}") from exc
    if len(mesh_shape) != 2:
        raise ValueError(f"TDBG projected-HF mesh shape from {source} must have length two, got {mesh_shape}")
    if mesh_shape[0] <= 0 or mesh_shape[1] <= 0 or mesh_shape[0] * mesh_shape[1] != int(n_k):
        raise ValueError(
            f"TDBG projected-HF mesh shape {mesh_shape} from {source} is incompatible with n_k={int(n_k)}"
        )
    return (int(mesh_shape[0]), int(mesh_shape[1]))


def _projected_hf_mesh_shape(
    basis_metadata: Mapping[str, object],
    *,
    n_k: int,
    mesh_shape: tuple[int, int] | None,
) -> tuple[tuple[int, int], str]:
    if mesh_shape is not None:
        return _coerce_projected_hf_mesh_shape(mesh_shape, n_k=n_k, source="mesh_shape argument"), "mesh_shape argument"
    if basis_metadata.get("topology_grid_shape") is not None:
        source = str(basis_metadata.get("topology_grid_shape_source", "basis_metadata['topology_grid_shape']"))
        return _coerce_projected_hf_mesh_shape(basis_metadata.get("topology_grid_shape"), n_k=n_k, source=source), source
    source = "basis_metadata['grid_shape']"
    return _coerce_projected_hf_mesh_shape(basis_metadata.get("grid_shape"), n_k=n_k, source=source), source


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
        return reshape_flat_mesh_to_grid(raw, mesh_shape, k_axis=0, order="C")
    raise ValueError(
        "TDBG projected-HF k_grid_frac must have shape "
        f"({mesh_shape[0] * mesh_shape[1]}, 2) or {mesh_shape + (2,)}, got {raw.shape}"
    )


def compute_projected_hf_topology(
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
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    """Compute FHS topology for selected TDBG projected-HF eigenstates.

    The public :meth:`TDBGProjectedHFResult.reconstruct_micro_wavefunctions`
    returns a flat ``(nk,basis,state)`` API bundle and deliberately drops the
    core sewing transforms. This system helper is the topology-ready path: it
    reconstructs through ``reconstruct_tdbg_projected_hf_micro_wavefunctions``,
    reshapes the flat k axis to ``(mesh,mesh,basis,state)`` using recorded mesh
    metadata, preserves the projected-HF sewing transforms, and delegates the
    FHS plaquette math to ``analysis.topology``.
    """

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
    if psi_flat.ndim != 3:
        raise ValueError(f"Expected reconstructed TDBG projected-HF psi_micro shape (nk,basis,state), got {psi_flat.shape}")
    resolved_mesh_shape, mesh_shape_source = _projected_hf_mesh_shape(
        basis_metadata,
        n_k=int(psi_flat.shape[0]),
        mesh_shape=mesh_shape,
    )
    psi_grid = reshape_flat_mesh_to_grid(psi_flat, resolved_mesh_shape, k_axis=0, order="C")
    k_grid = _projected_hf_k_grid_frac(result, resolved_mesh_shape, k_grid_frac)
    active_sewing_transforms = tuple(core_bundle.sewing_transforms) if sewing_transforms is None else tuple(sewing_transforms)
    selected = tuple(int(index) for index in basis_metadata.get("selected_hf_state_indices", range(psi_grid.shape[-1])))
    if len(selected) != int(psi_grid.shape[-1]):
        raise ValueError(
            "TDBG projected-HF reconstruction metadata selected_hf_state_indices length "
            f"{len(selected)} does not match reconstructed state count {psi_grid.shape[-1]}"
        )
    column_indices = tuple(range(int(psi_grid.shape[-1])))
    labels = tuple(f"hf_state={index}" for index in selected)
    payload = dict(basis_metadata)
    payload.update(
        {
            "topology_adapter": "mean_field.systems.tdbg.topology.compute_projected_hf_topology",
            "topology_input_axis_order": "mesh,mesh,basis,state",
            "topology_flat_input_axis_order": "nk,basis,state",
            "topology_grid_shape": [int(resolved_mesh_shape[0]), int(resolved_mesh_shape[1])],
            "topology_grid_shape_source": mesh_shape_source,
            "topology_k_grid_frac_shape": None if k_grid is None else [int(v) for v in k_grid.shape],
            "topology_sewing_transforms_count": int(len(active_sewing_transforms)),
            "band_indices_semantics": "HF eigenstate indices after np.linalg.eigh sorting",
            "column_indices": [int(index) for index in column_indices],
            "absolute_band_indices": [int(index) for index in selected],
            "valley_argument_semantics": "diagnostic label; reconstructed projected-HF rows contain spin and both valleys as direct-sum blocks",
            "evidence_paths": [
                "src/mean_field/systems/tdbg/topology.py",
                "src/mean_field/systems/tdbg/projected_hf_state.py",
                "src/mean_field/systems/tdbg/projected_hf_geometry.py",
                "src/analysis/topology/system.py",
                "src/analysis/topology/core.py",
            ],
        }
    )
    payload.update(dict(metadata or {}))
    return compute_system_topology_from_eigenvectors(
        psi_grid,
        column_indices,
        system="tdbg",
        valley=int(valley),
        k_grid_frac=k_grid,
        sewing_transforms=active_sewing_transforms,
        index_metadata=payload,
        result_band_indices=selected,
        role="hf_state",
        labels=labels,
        link_method=link_method,
        orientation_sign=float(orientation_sign),
        atol=float(atol),
        regularization=float(regularization),
    )


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(
        eigenvectors, band_indices, system="tdbg", valley=valley, k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms, index_metadata=metadata, orientation_sign=orientation_sign,
    )


def compute_topology_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    metadata: dict[str, object] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_grid_result(
        grid_result, band_indices, system="tdbg", valley=valley, sewing_transforms=sewing_transforms,
        index_metadata=metadata, orientation_sign=orientation_sign,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    n_bands: int | None = None,
    boundary_sewing: bool = True,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    requested = normalize_state_indices(band_indices)
    if n_bands is not None and int(n_bands) <= max(requested):
        raise ValueError(f"n_bands={int(n_bands)} does not include requested band index {max(requested)}")
    if endpoint:
        raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(
        int(mesh_size), lattice, params, valley=int(valley), n_bands=None if n_bands is None else int(n_bands), return_eigenvectors=True,
        endpoint=False, frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    return compute_topology_from_grid_result(
        grid, requested, valley=int(valley), sewing_transforms=boundary_sewing_transforms(lattice) if bool(boundary_sewing) else None,
        metadata={"boundary_sewing": bool(boundary_sewing)}, orientation_sign=float(orientation_sign),
    )


__all__ = [
    "TopologyResult", "boundary_sewing_transforms", "compute_projected_hf_topology", "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result", "compute_topology_on_grid", "projected_hf_boundary_sewing_transforms",
    "projected_hf_q_site_sewing_transform", "translation_srcmap",
]
