from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mean_field.core.bands import GridBandsResult, PathBandsResult

from .artifacts import ConventionBundle


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


def compute_bands(
    model: object,
    *,
    path: Any | None = None,
    grid_mesh: int | tuple[int, int] | None = None,
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
        if not hasattr(model, "bands_on_grid"):
            raise TypeError(f"Model {model!r} does not expose bands_on_grid")
        if isinstance(grid_mesh, tuple):
            if len(grid_mesh) != 2 or int(grid_mesh[0]) != int(grid_mesh[1]):
                raise NotImplementedError("The current façade only delegates square mesh_size grids")
            mesh_size = int(grid_mesh[0])
        else:
            mesh_size = int(grid_mesh)
        try:
            return band_bundle_from_result(model.bands_on_grid(mesh_size, **call_kwargs))
        except TypeError:
            fallback_kwargs = _central_band_count_kwargs(call_kwargs)
            return band_bundle_from_result(model.bands_on_grid(mesh_size, **fallback_kwargs))
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
        return band_bundle_from_result(model.bands_along_path(path, **call_kwargs))
    except TypeError:
        fallback_kwargs = _central_band_count_kwargs(call_kwargs)
        return band_bundle_from_result(model.bands_along_path(path, **fallback_kwargs))


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
