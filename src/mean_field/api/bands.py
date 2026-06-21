from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mean_field.core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from mean_field.core.lattice import KPath as CoreKPath, cumulative_distance

from .artifacts import ConventionBundle
from .models import component_group_records


@dataclass(frozen=True)
class KGrid:
    mesh: tuple[int, int]
    kvec: np.ndarray
    frac: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    labels: tuple[str, ...] = ()
    node_indices: tuple[int, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BandBundle:
    k: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    basis_metadata: dict[str, object] = field(default_factory=dict)
    convention: ConventionBundle = field(default_factory=ConventionBundle)
    source: Literal["path", "grid", "raw"] = "raw"



def band_bundle_from_result(result: PathBandsResult | GridBandsResult) -> BandBundle:
    if isinstance(result, PathBandsResult):
        metadata = dict(result.metadata)
        metadata.update({"labels": list(result.path.labels), "node_indices": list(result.path.node_indices)})
        return BandBundle(
            k=np.asarray(result.path.kvec),
            energies=np.asarray(result.energies, dtype=float),
            eigenvectors=result.eigenvectors,
            basis_metadata=metadata,
            source="path",
        )
    if isinstance(result, GridBandsResult):
        metadata = dict(result.metadata)
        metadata.update({"k_grid_frac_shape": list(np.asarray(result.k_grid_frac).shape)})
        return BandBundle(
            k=np.asarray(result.kvec),
            energies=np.asarray(result.energies, dtype=float),
            eigenvectors=result.eigenvectors,
            basis_metadata=metadata,
            source="grid",
        )
    raise TypeError(f"Unsupported band result type: {type(result)!r}")


def _with_model_metadata(bundle: BandBundle, model: object) -> BandBundle:
    metadata = dict(bundle.basis_metadata)
    records = component_group_records(model)
    if records:
        metadata.setdefault("component_groups", [dict(record) for record in records])
    return BandBundle(
        k=bundle.k,
        energies=bundle.energies,
        eigenvectors=bundle.eigenvectors,
        basis_metadata=metadata,
        convention=bundle.convention,
        source=bundle.source,
    )


def _model_matrix_dim(model: object) -> int:
    value = getattr(model, "matrix_dim", None)
    if value is None:
        raise TypeError(f"Model {model!r} does not expose matrix_dim")
    return int(value() if callable(value) else value)


def _diagonalize_direct(model: object, k_tilde: complex, n_bands: int, return_eigenvectors: bool, call_kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    if not hasattr(model, "diagonalize"):
        raise TypeError(f"Model {model!r} does not expose diagonalize")
    kwargs = dict(call_kwargs)
    kwargs.pop("return_eigenvectors", None)
    kwargs["n_bands"] = int(n_bands)
    try:
        return model.diagonalize(complex(k_tilde), return_eigenvectors=bool(return_eigenvectors), **kwargs)  # type: ignore[attr-defined]
    except TypeError:
        result = model.diagonalize(complex(k_tilde), **kwargs)  # type: ignore[attr-defined]
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Model diagonalize must return (energies, eigenvectors_or_none)")
        energies, eigenvectors = result
        if return_eigenvectors and eigenvectors is None:
            raise ValueError("Model diagonalize returned no eigenvectors despite return_eigenvectors=True")
        return np.asarray(energies, dtype=float), None if eigenvectors is None else np.asarray(eigenvectors)


def _public_path_to_core_path(path: KPath) -> CoreKPath:
    kvec = np.asarray(path.kvec, dtype=np.complex128)
    if kvec.ndim != 1:
        raise ValueError(f"KPath.kvec must be one-dimensional, got shape {kvec.shape}")
    if path.labels and path.node_indices:
        labels = tuple(path.labels)
        node_indices = tuple(int(index) for index in path.node_indices)
    else:
        labels = ()
        node_indices = ()
    return CoreKPath(kvec=kvec, kdist=cumulative_distance(kvec), labels=labels, node_indices=node_indices)


def _compute_direct_grid(model: object, grid: KGrid, call_kwargs: dict[str, Any]) -> BandBundle:
    kvec = np.asarray(grid.kvec, dtype=np.complex128)
    if kvec.ndim != 2:
        raise ValueError(f"KGrid.kvec must be a two-dimensional grid, got shape {kvec.shape}")
    frac = np.zeros(kvec.shape + (2,), dtype=float) if grid.frac is None else np.asarray(grid.frac, dtype=float)
    if frac.shape != kvec.shape + (2,):
        raise ValueError(f"KGrid.frac shape {frac.shape} incompatible with kvec shape {kvec.shape}")
    result = compute_grid_bands(
        k_grid_frac=frac,
        kvec=kvec,
        matrix_dim=_model_matrix_dim(model),
        n_bands=call_kwargs.get("n_bands"),
        return_eigenvectors=bool(call_kwargs.get("return_eigenvectors", False)),
        diagonalize=lambda kval, n_bands, ret: _diagonalize_direct(model, kval, n_bands, ret, call_kwargs),
        result_metadata=dict(grid.metadata),
    )
    return _with_model_metadata(band_bundle_from_result(result), model)


def _compute_direct_path(model: object, path: KPath, call_kwargs: dict[str, Any]) -> BandBundle:
    core_path = _public_path_to_core_path(path)
    result = compute_path_bands(
        core_path,
        matrix_dim=_model_matrix_dim(model),
        n_bands=call_kwargs.get("n_bands"),
        return_eigenvectors=bool(call_kwargs.get("return_eigenvectors", False)),
        diagonalize=lambda kval, n_bands, ret: _diagonalize_direct(model, kval, n_bands, ret, call_kwargs),
        result_metadata=dict(path.metadata),
    )
    return _with_model_metadata(band_bundle_from_result(result), model)


def compute_bands(
    model: object,
    *,
    path: Any | None = None,
    grid_mesh: int | tuple[int, int] | KGrid | None = None,
    valley: int | None = 1,
    n_bands: int | None = None,
    points_per_segment: int = 120,
    return_eigenvectors: bool = False,
    **kwargs: Any,
) -> BandBundle:
    """Compute non-interacting bands through the public façade.

    This delegates to the existing system model methods and only normalizes the
    result container.  It does not change system-specific Hamiltonians, gauges,
    paths, or band selection conventions.
    """

    if path is not None and grid_mesh is not None:
        raise ValueError("Pass either path or grid_mesh, not both")
    call_kwargs: dict[str, Any] = dict(kwargs)
    if valley is not None:
        call_kwargs["valley"] = int(valley)
    if n_bands is not None:
        call_kwargs["n_bands"] = int(n_bands)
    call_kwargs["return_eigenvectors"] = bool(return_eigenvectors)
    if grid_mesh is not None:
        if isinstance(grid_mesh, KGrid):
            return _compute_direct_grid(model, grid_mesh, call_kwargs)
        if isinstance(grid_mesh, tuple):
            if len(grid_mesh) != 2:
                raise ValueError(f"grid_mesh tuple must have two entries, got {grid_mesh!r}")
            if int(grid_mesh[0]) != int(grid_mesh[1]):
                raise NotImplementedError("Pass a KGrid with explicit kvec/frac to compute non-square grids")
            mesh_size = int(grid_mesh[0])
        else:
            mesh_size = int(grid_mesh)
        if not hasattr(model, "bands_on_grid"):
            raise TypeError(f"Model {model!r} does not expose bands_on_grid")
        try:
            return _with_model_metadata(band_bundle_from_result(model.bands_on_grid(mesh_size, **call_kwargs)), model)
        except TypeError:
            fallback_kwargs = _central_band_count_kwargs(call_kwargs)
            return _with_model_metadata(band_bundle_from_result(model.bands_on_grid(mesh_size, **fallback_kwargs)), model)
    if isinstance(path, KPath):
        return _compute_direct_path(model, path, call_kwargs)
    if path is None:
        if not hasattr(model, "standard_kpath"):
            raise TypeError(f"Model {model!r} does not expose standard_kpath")
        try:
            path = model.standard_kpath(points_per_segment=int(points_per_segment))
        except TypeError:
            path = model.standard_kpath(resolution=int(points_per_segment))
    if not hasattr(model, "bands_along_path"):
        raise TypeError(f"Model {model!r} does not expose bands_along_path")
    try:
        return _with_model_metadata(band_bundle_from_result(model.bands_along_path(path, **call_kwargs)), model)
    except TypeError:
        fallback_kwargs = _central_band_count_kwargs(call_kwargs)
        return _with_model_metadata(band_bundle_from_result(model.bands_along_path(path, **fallback_kwargs)), model)


def _central_band_count_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    fallback = dict(call_kwargs)
    if "n_bands" in fallback:
        fallback["central_band_count"] = fallback.pop("n_bands")
    return fallback


__all__ = [
    "BandBundle",
    "KGrid",
    "KPath",
    "band_bundle_from_result",
    "compute_bands",
]
