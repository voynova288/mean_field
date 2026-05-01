from __future__ import annotations

import math

import numpy as np
import pytest

from mean_field.systems.tmbg import (
    GridBandsResult,
    TMBGModel,
    TMBGParameters,
    compute_topology_from_eigenvectors,
)


def _qiwuzhang_eigenvectors(mesh_size: int, *, mass: float) -> np.ndarray:
    eigenvectors = np.zeros((mesh_size, mesh_size, 2, 2), dtype=np.complex128)
    frac = np.arange(mesh_size, dtype=float) / float(mesh_size)
    for ix, fx in enumerate(frac):
        kx = 2.0 * math.pi * fx
        for iy, fy in enumerate(frac):
            ky = 2.0 * math.pi * fy
            dx = math.sin(kx)
            dy = math.sin(ky)
            dz = mass + math.cos(kx) + math.cos(ky)
            hamiltonian = np.asarray(
                [
                    [dz, dx - 1j * dy],
                    [dx + 1j * dy, -dz],
                ],
                dtype=np.complex128,
            )
            _, vecs = np.linalg.eigh(hamiltonian)
            eigenvectors[ix, iy] = vecs
    return eigenvectors


def test_fukui_hatsugai_finds_nontrivial_qwz_band() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(21, mass=-1.0)
    result = compute_topology_from_eigenvectors(eigenvectors, 0)

    assert result.is_nearly_integer
    assert abs(result.chern_number) == pytest.approx(1.0, abs=1.0e-8)


def test_fukui_hatsugai_total_two_band_subspace_is_trivial() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(21, mass=-1.0)
    result = compute_topology_from_eigenvectors(eigenvectors, (0, 1))

    assert result.is_nearly_integer
    assert result.chern_number == pytest.approx(0.0, abs=1.0e-8)


def test_tmbg_model_topology_smoke_runs_and_returns_grid_flux() -> None:
    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full(interlayer_potential=0.06))
    result = model.topology_on_grid(4, 0, valley=1)

    assert result.band_indices == (0,)
    assert result.berry_curvature.shape == (4, 4)
    assert np.isfinite(result.chern_number)


def test_topology_on_grid_retries_shifted_and_refined_mesh(monkeypatch) -> None:
    import mean_field.systems.tmbg.topology as topology_module

    attempts: list[tuple[int, tuple[float, float]]] = []

    def fake_compute_bands_on_grid(
        mesh_size,
        lattice,
        params,
        *,
        valley,
        n_bands,
        return_eigenvectors,
        endpoint,
        frac_shift,
    ):
        attempts.append((int(mesh_size), tuple(float(value) for value in frac_shift)))
        return GridBandsResult(
            k_grid_frac=np.zeros((mesh_size, mesh_size, 2), dtype=float),
            kvec=np.zeros((mesh_size, mesh_size), dtype=np.complex128),
            energies=np.zeros((mesh_size, mesh_size, n_bands), dtype=float),
            eigenvectors=np.zeros((mesh_size, mesh_size, 2, n_bands), dtype=np.complex128),
        )

    call_count = {"count": 0}

    def fake_topology_from_grid_result(grid_result, band_indices, *, valley):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise ValueError("near-degenerate link")
        return topology_module.TopologyResult(
            band_indices=tuple(band_indices),
            valley=int(valley),
            k_grid_frac=grid_result.k_grid_frac,
            berry_curvature=np.zeros(grid_result.k_grid_frac.shape[:2], dtype=float),
            chern_number=0.0,
            rounded_chern_number=0,
        )

    monkeypatch.setattr(topology_module, "compute_bands_on_grid", fake_compute_bands_on_grid)
    monkeypatch.setattr(topology_module, "_topology_from_grid_result", fake_topology_from_grid_result)

    model = TMBGModel.from_config(1.21, n_shells=1, params=TMBGParameters.full())
    result = model.topology_on_grid(4, 0, valley=1)

    assert result.chern_number == pytest.approx(0.0, abs=1.0e-12)
    assert attempts == [
        (4, (0.0, 0.0)),
        (4, (0.125, 0.125)),
        (8, (0.0, 0.0)),
    ]
