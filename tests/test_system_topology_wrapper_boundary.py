from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import BlockSewingSpec, FHSState, compute_lattice_topology, sewing_transforms_from_block_spec
import mean_field.systems.RnG_hBN.topology as rlg_topology
import mean_field.systems.htg.topology as htg_topology
import mean_field.systems.tdbg.topology as tdbg_topology
import mean_field.systems.tmbg.topology as tmbg_topology


MODULES = (tmbg_topology, tdbg_topology, rlg_topology, htg_topology)


def _fake_grid(*, basis_dim: int = 2, band_label: int = 7):
    return SimpleNamespace(
        k_grid_frac=np.zeros((2, 2, 2), dtype=float),
        kvec=np.zeros((2, 2), dtype=np.complex128),
        energies=np.zeros((2, 2, 1), dtype=float),
        eigenvectors=np.ones((2, 2, basis_dim, 1), dtype=np.complex128),
        band_indices=(int(band_label),),
    )


def _trivial_basis_sewing() -> BlockSewingSpec:
    return BlockSewingSpec(
        block_coordinates=np.asarray([[0.0]], dtype=float),
        local_block_size=2,
        translations=((0.0,), (0.0,)),
    )


@pytest.mark.parametrize("module", MODULES)
def test_system_topology_modules_are_state_builders_not_chern_calculators(module) -> None:
    for removed in ("compute_topology_from_eigenvectors", "compute_topology_from_grid_result", "compute_topology_on_grid", "TopologyResult"):
        assert not hasattr(module, removed)

    state = module.fhs_state_from_grid_result(_fake_grid(), 7, valley=1, basis_sewing=_trivial_basis_sewing())

    assert isinstance(state, FHSState)
    assert state.reported_indices == (7,)
    assert state.metadata["absolute_band_indices"] == [7]
    assert state.metadata["column_indices"] == [0]
    assert state.metadata["boundary_sewing"] is True

    result = compute_lattice_topology(state)
    assert result.band_indices == (7,)
    assert result.rounded_chern_number == 0


def test_common_grid_state_maps_band_labels_to_state_columns() -> None:
    state = tmbg_topology.fhs_state_from_grid_result(_fake_grid(basis_dim=3, band_label=11), 11, valley=1)

    assert state.state_indices == (0,)
    assert state.reported_indices == (11,)
    assert state.metadata["absolute_band_indices"] == [11]
    assert state.metadata["column_indices"] == [0]

    with pytest.raises(ValueError, match="Requested band labels"):
        tmbg_topology.fhs_state_from_grid_result(_fake_grid(basis_dim=3, band_label=11), 12, valley=1)


def test_generic_basis_sewing_specs_are_metadata_not_topology_calculators() -> None:
    tmbg_spec = tmbg_topology.tmbg_basis_sewing(SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)))
    htg_spec = htg_topology.htg_basis_sewing(SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)))
    tdbg_spec = tdbg_topology.tdbg_basis_sewing(
        SimpleNamespace(
            q_sites=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            g_m1=1.0 + 0.0j,
            g_m2=0.0 + 1.0j,
        )
    )
    rlg_spec = rlg_topology.rlg_hbn_basis_sewing(
        SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)),
        SimpleNamespace(layer_count=1),
        valley=-1,
    )

    assert tmbg_spec.local_block_size == 6
    assert htg_spec.local_block_size == 6
    assert tdbg_spec.local_block_size == 4
    assert tdbg_spec.block_labels.tolist() == [0, 0]
    assert rlg_spec.local_block_size == 2
    assert rlg_spec.translations == ((-1.0, 0.0), (0.0, -1.0))


def test_block_sewing_spec_relabels_plane_wave_blocks() -> None:
    spec = tmbg_topology.tmbg_basis_sewing(SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)))
    sew_1, sew_2 = sewing_transforms_from_block_spec(spec)
    values = np.arange(12, dtype=np.complex128)

    np.testing.assert_allclose(sew_1(values), np.concatenate((values[6:12], np.zeros(6, dtype=np.complex128))))
    np.testing.assert_allclose(sew_2(values), 0.0)
    with pytest.raises(ValueError, match="Expected first axis"):
        sew_1(np.zeros((11,), dtype=np.complex128))
