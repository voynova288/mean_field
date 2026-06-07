from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import TopologyResult, compute_system_topology_on_grid


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


@pytest.mark.parametrize(
    ("module_name", "system_label"),
    [
        ("mean_field.systems.tmbg.topology", "tmbg"),
        ("mean_field.systems.tdbg.topology", "tdbg"),
        ("mean_field.systems.atmg.topology", "atmg"),
        ("mean_field.systems.RnG_hBN.topology", "RLG_hBN"),
    ],
)
def test_system_topology_wrappers_delegate_to_unified_wavefunction_framework(
    module_name: str,
    system_label: str,
) -> None:
    module = importlib.import_module(module_name)
    eigenvectors = _qiwuzhang_eigenvectors(17, mass=-1.0)

    result = module.compute_topology_from_eigenvectors(eigenvectors, 0, valley=-1)

    assert isinstance(result, TopologyResult)
    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.berry_connection.shape == (2, 17, 17)
    assert result.min_link_magnitude is not None
    assert result.index_metadata["system"] == system_label
    assert result.index_metadata["indices"] == [0]
    assert result.is_nearly_integer
    assert abs(result.chern_number) == pytest.approx(1.0, abs=1.0e-8)


def test_compute_system_topology_on_grid_uses_common_retry_and_metadata_api() -> None:
    calls: list[tuple[int, tuple[float, float], int]] = []

    def grid_builder(trial_mesh: int, frac_shift: tuple[float, float], resolved_n_bands: int) -> SimpleNamespace:
        calls.append((int(trial_mesh), frac_shift, int(resolved_n_bands)))
        frac = np.arange(int(trial_mesh), dtype=float) / float(trial_mesh)
        f1, f2 = np.meshgrid(frac + frac_shift[0], frac + frac_shift[1], indexing="ij")
        return SimpleNamespace(
            eigenvectors=_qiwuzhang_eigenvectors(int(trial_mesh), mass=-1.0),
            k_grid_frac=np.stack((f1, f2), axis=-1),
        )

    result = compute_system_topology_on_grid(
        19,
        0,
        system="toy_system",
        grid_builder=grid_builder,
        valley=2,
        index_metadata={"source": "unit_test"},
    )

    assert isinstance(result, TopologyResult)
    assert result.band_indices == (0,)
    assert result.valley == 2
    assert result.index_metadata["system"] == "toy_system"
    assert result.index_metadata["metadata"] == {"source": "unit_test"}
    assert result.k_grid_frac.shape == (19, 19, 2)
    assert result.is_nearly_integer
    assert abs(result.chern_number) == pytest.approx(1.0, abs=1.0e-8)
    assert calls == [(19, (0.0, 0.0), 1)]
