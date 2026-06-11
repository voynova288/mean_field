from __future__ import annotations

import numpy as np

from analysis.topology import make_topology_adapter


def _trivial_eigenvectors(mesh: int = 3) -> np.ndarray:
    out = np.zeros((mesh, mesh, 1, 1), dtype=np.complex128)
    out[:, :, 0, 0] = 1.0
    return out


def test_make_topology_adapter_from_eigenvectors_preserves_system_metadata() -> None:
    adapter = make_topology_adapter(system="toy", valley=-1, index_metadata={"sewing": False})
    result = adapter["from_eigenvectors"](_trivial_eigenvectors(), 0)

    assert result.chern_number == 0.0
    assert result.valley == -1
    assert result.index_metadata["system"] == "toy"
    assert result.index_metadata["metadata"] == {"sewing": False}


def test_make_topology_adapter_on_grid_uses_supplied_grid_builder() -> None:
    def grid_builder(mesh: int, frac_shift: tuple[float, float], n_bands: int):
        del frac_shift
        assert n_bands == 1
        return type(
            "Grid",
            (),
            {
                "eigenvectors": _trivial_eigenvectors(mesh),
                "k_grid_frac": np.zeros((mesh, mesh, 2), dtype=float),
            },
        )()

    adapter = make_topology_adapter(system="toy", grid_builder=grid_builder, valley=1)
    result = adapter["on_grid"](3, 0)

    assert result.rounded_chern_number == 0
    assert result.index_metadata["system"] == "toy"


def test_make_topology_adapter_on_grid_forwards_custom_topology_builder() -> None:
    calls: list[tuple[int, tuple[int, ...]]] = []

    def grid_builder(mesh: int, frac_shift: tuple[float, float], n_bands: int):
        del frac_shift, n_bands
        return type(
            "Grid",
            (),
            {
                "mesh": mesh,
                "eigenvectors": _trivial_eigenvectors(mesh),
                "k_grid_frac": np.zeros((mesh, mesh, 2), dtype=float),
            },
        )()

    def topology_builder(grid_result, band_indices: tuple[int, ...]):
        calls.append((grid_result.mesh, band_indices))
        return make_topology_adapter(system="toy", valley=-1)["from_grid_result"](grid_result, band_indices)

    adapter = make_topology_adapter(system="toy", grid_builder=grid_builder, topology_builder=topology_builder)
    result = adapter["on_grid"](3, 0)

    assert calls == [(3, (0,))]
    assert result.rounded_chern_number == 0
    assert result.valley == -1
