"""Exact saved-grid views of converged TBG zero-field HF states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mean_field.core.hf.flavors import (
    FlavorBandData,
    build_flavor_band_data,
)
from ....core.lattice import KPath
from ..params import TBGParameters
from mean_field.systems.tbg.zero_field._hf_basis_overlap import RestrictedHartreeFockRun
from .hf_contracts import validate_tbg_zero_field_typed_hf_run_source
from .model import BMSolution


@dataclass(frozen=True)
class HFExactSavedGridPathResult:
    """HF bands selected only at exact points of the saved SCF grid."""

    params: TBGParameters
    path: KPath
    kdist: np.ndarray
    path_sample_indices: np.ndarray
    path_kvec: np.ndarray
    grid_kvec: np.ndarray
    grid_indices: np.ndarray
    distance_to_path: np.ndarray
    band_data: FlavorBandData
    mu: float
    nu: float
    lk: int | None
    lg: int
    init_mode: str
    seed: int
    exit_reason: str
    mesh_shape: tuple[int, int] | None = None


def _reported_grid_shape(grid_solution: BMSolution) -> tuple[int, int] | None:
    if grid_solution.torus_mesh is None:
        return None
    shape = grid_solution.torus_mesh.mesh_shape
    return int(shape[0]), int(shape[1])


def _reported_grid_size(grid_solution: BMSolution) -> int | None:
    shape = _reported_grid_shape(grid_solution)
    if shape is not None:
        return shape[0] if shape[0] == shape[1] else None
    return int(round(np.sqrt(grid_solution.nk))) - 1


def select_restricted_hf_saved_grid_path(
    hf_run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
    *,
    path: KPath,
) -> HFExactSavedGridPathResult:
    """Select exact path/grid intersections without off-grid reconstruction.

    The returned Hamiltonians come directly from the saved converged SCF state.
    Only machine-precision coordinate equality is accepted; callers cannot
    enlarge the threshold. Nearest-grid substitution and interpolation are
    never performed.
    """

    if not isinstance(hf_run, RestrictedHartreeFockRun):
        raise TypeError("hf_run must be RestrictedHartreeFockRun")
    if not isinstance(grid_solution, BMSolution):
        raise TypeError("grid_solution must be BMSolution")
    validate_tbg_zero_field_typed_hf_run_source(hf_run, grid_solution)
    if path.kvec.size == 0:
        raise ValueError("At least two path nodes are required")

    grid_kvec = np.asarray(grid_solution.lattice_kvec, dtype=np.complex128)
    path_kvec_all = np.asarray(path.kvec, dtype=np.complex128)
    if not (
        np.all(np.isfinite(grid_kvec.real))
        and np.all(np.isfinite(grid_kvec.imag))
        and np.all(np.isfinite(path_kvec_all.real))
        and np.all(np.isfinite(path_kvec_all.imag))
    ):
        raise ValueError("saved-grid and path coordinates must be finite")
    trusted_grid_scale = max(1.0, float(np.max(np.abs(grid_kvec))))
    exact_tolerance = 64.0 * np.finfo(np.float64).eps * trusted_grid_scale
    distance_matrix = np.abs(path_kvec_all[:, None] - grid_kvec[None, :])
    nearest_indices = np.argmin(distance_matrix, axis=1).astype(int)
    nearest_distances = distance_matrix[np.arange(path.kvec.size), nearest_indices]
    path_indices = np.flatnonzero(nearest_distances <= exact_tolerance).astype(int)
    grid_indices = nearest_indices[path_indices].astype(int)
    path_kvec = np.asarray(path.kvec[path_indices], dtype=np.complex128)
    selected_hamiltonian = np.asarray(
        hf_run.state.hamiltonian[:, :, grid_indices], dtype=np.complex128
    )
    band_data = build_flavor_band_data(
        selected_hamiltonian,
        n_spin=hf_run.state.n_spin,
        n_eta=hf_run.state.n_eta,
        n_band=hf_run.state.n_band,
    )
    return HFExactSavedGridPathResult(
        params=grid_solution.params,
        path=path,
        kdist=np.asarray(path.kdist[path_indices], dtype=float),
        path_sample_indices=path_indices,
        path_kvec=path_kvec,
        grid_kvec=np.asarray(grid_kvec[grid_indices], dtype=np.complex128),
        grid_indices=grid_indices,
        distance_to_path=np.asarray(nearest_distances[path_indices], dtype=float),
        band_data=band_data,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        lk=_reported_grid_size(grid_solution),
        lg=grid_solution.lg,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        mesh_shape=_reported_grid_shape(grid_solution),
    )


__all__ = ["HFExactSavedGridPathResult", "select_restricted_hf_saved_grid_path"]
