from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    compute_lattice_topology,
    fhs_state_from_wavefunctions,
    sewing_transforms_from_block_spec,
)
from mean_field.core.bands import GridBandsResult
from mean_field.systems.RnG_hBN import RLGhBNModel
from mean_field.systems.atmg import ATMGModel, ATMGParameters
from mean_field.systems.tdbg import TDBGModel, TDBGParameters
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
import mean_field.systems.RnG_hBN.topology as rlg_topology
import mean_field.systems.atmg.topology as atmg_topology
import mean_field.systems.htqg.topology as htqg_topology
import mean_field.systems.tdbg.topology as tdbg_topology
import mean_field.systems.tmbg.topology as tmbg_topology


def _fake_grid(*, basis_dim: int = 2, band_label: int = 7) -> GridBandsResult:
    return GridBandsResult(
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



def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> np.ndarray:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    for ix in range(mesh):
        kx = 2.0 * np.pi * ix / mesh
        for iy in range(mesh):
            ky = 2.0 * np.pi * iy / mesh
            dz = mass + np.cos(kx) + np.cos(ky)
            hamiltonian = np.asarray(
                [
                    [dz, np.sin(kx) - 1j * np.sin(ky)],
                    [np.sin(kx) + 1j * np.sin(ky), -dz],
                ],
                dtype=np.complex128,
            )
            _, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions

@pytest.mark.parametrize(
    "module,kwargs",
    [
        (tmbg_topology, {}),
        (atmg_topology, {}),
        (tdbg_topology, {"basis_sewing": _trivial_basis_sewing()}),
        (rlg_topology, {"basis_sewing": _trivial_basis_sewing()}),
        (htqg_topology, {"basis_sewing": _trivial_basis_sewing()}),
    ],
)
def test_system_topology_modules_build_fhs_state_not_chern_result(module, kwargs) -> None:
    for removed in ("compute_topology_from_eigenvectors", "compute_topology_from_grid_result", "compute_topology_on_grid", "TopologyResult"):
        assert not hasattr(module, removed)

    state = module.fhs_state_from_grid_result(_fake_grid(), 7, valley=1, **kwargs)

    assert isinstance(state, FHSState)
    assert state.reported_indices == (7,)
    assert state.metadata["absolute_band_indices"] == [7]
    assert state.metadata["column_indices"] == [0]

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


def test_common_fhs_state_computes_nonzero_chern_without_system_code() -> None:
    state = fhs_state_from_wavefunctions(
        _qiwuzhang_wavefunctions(mesh=21, mass=1.0),
        0,
        system="qiwuzhang",
        labels=("lower",),
    )

    result = compute_lattice_topology(state)

    assert result.rounded_chern_number == 1
    assert result.is_nearly_integer
    assert result.index_metadata["system"] == "qiwuzhang"
    assert result.index_metadata["labels"] == ["lower"]


def test_block_sewing_spec_is_common_equivalent_to_explicit_sewing() -> None:
    spec = BlockSewingSpec(
        block_coordinates=np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        local_block_size=2,
        translations=((1.0,), (0.0,)),
    )
    sew_1, sew_2 = sewing_transforms_from_block_spec(spec)
    vector = np.arange(6, dtype=np.complex128)

    np.testing.assert_array_equal(sew_1(vector), np.asarray([2, 3, 4, 5, 0, 0], dtype=np.complex128))

    wavefunctions = np.ones((3, 2, 6, 1), dtype=np.complex128)
    from_state = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, basis_sewing=spec))
    explicit = compute_lattice_topology(wavefunctions, 0, sewing_transforms=(sew_1, sew_2))

    np.testing.assert_allclose(from_state.link_1, explicit.link_1)
    np.testing.assert_allclose(from_state.link_2, explicit.link_2)
    np.testing.assert_allclose(from_state.berry_curvature, explicit.berry_curvature)
    assert from_state.chern_number == pytest.approx(explicit.chern_number)


def test_tdbg_actual_state_fhs_chern_uses_common_topology_pipeline() -> None:
    model = TDBGModel.from_config(
        1.38,
        cut=2.0,
        params=TDBGParameters.full(stacking="AB-BA", Delta=0.09),
    )
    conduction_band = model.matrix_dim // 2

    for mesh in (5, 7):
        frac_shift = (0.37 / mesh, 0.19 / mesh)
        chern_k = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, conduction_band, valley=1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )
        chern_kprime = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, conduction_band, valley=-1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )

        assert chern_k.chern_number == pytest.approx(1.0, abs=1.0e-8)
        assert chern_kprime.chern_number == pytest.approx(-1.0, abs=1.0e-8)
        assert chern_k.metadata["boundary_sewing"] is True
        assert chern_k.min_link_magnitude > 0.2
        assert chern_kprime.min_link_magnitude > 0.2


def test_tmbg_actual_shell_converged_subspaces_use_common_topology_pipeline() -> None:
    model = TMBGModel.from_config(
        1.05,
        n_shells=2,
        params=TMBGParameters.minimal(interlayer_potential=0.0),
    )
    conduction_band = model.lattice.matrix_dim // 2

    for mesh in (5, 7):
        frac_shift = (0.37 / mesh, 0.19 / mesh)
        conduction = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, conduction_band, valley=1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )
        flat_pair = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, (conduction_band - 1, conduction_band), valley=1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )

        assert conduction.rounded_chern_number == 1
        assert conduction.chern_number == pytest.approx(1.0, abs=1.0e-8)
        assert conduction.min_link_magnitude > 0.2
        assert flat_pair.rounded_chern_number == -1
        assert flat_pair.chern_number == pytest.approx(-1.0, abs=1.0e-8)
        assert flat_pair.min_link_magnitude > 0.08


def test_atmg_actual_chiral_central_subspace_uses_common_topology_pipeline() -> None:
    model = ATMGModel.from_config(
        3,
        1.05,
        n_shells=2,
        params=ATMGParameters.chiral(3, 1.05),
    )
    conduction_band = model.matrix_dim // 2

    for mesh in (5, 7):
        frac_shift = (0.37 / mesh, 0.19 / mesh)
        central_pair = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, (conduction_band - 1, conduction_band), valley=1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )

        assert central_pair.rounded_chern_number == 0
        assert central_pair.chern_number == pytest.approx(0.0, abs=1.0e-8)
        assert central_pair.min_link_magnitude > 0.6


def test_rlg_hbn_actual_central_pair_uses_common_topology_pipeline() -> None:
    model = RLGhBNModel.from_config(
        layer_count=5,
        xi=1,
        theta_deg=0.77,
        shell_count=2,
        displacement_field_mev=0.0,
    )
    valence_band, conduction_band = model.flat_band_indices

    for mesh in (5, 7):
        frac_shift = (0.37 / mesh, 0.19 / mesh)
        central_pair = compute_lattice_topology(
            model.fhs_state_on_grid(mesh, (valence_band, conduction_band), valley=1, n_bands=conduction_band + 1, frac_shift=frac_shift)
        )

        assert central_pair.rounded_chern_number == -4
        assert central_pair.chern_number == pytest.approx(-4.0, abs=1.0e-8)
        assert central_pair.min_link_magnitude > 0.09


def test_generic_basis_sewing_specs_are_metadata_not_topology_calculators() -> None:
    tmbg_spec = tmbg_topology.tmbg_basis_sewing(
        SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int))
    )
    atmg_spec = atmg_topology.atmg_basis_sewing(
        SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)),
        SimpleNamespace(n_layers=4),
    )
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
    htqg_spec = htqg_topology.htqg_basis_sewing(
        SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int))
    )

    assert isinstance(tmbg_spec, BlockSewingSpec)
    assert tmbg_spec.local_block_size == 6
    assert tmbg_spec.translations == ((1.0, 0.0), (0.0, 1.0))

    assert isinstance(atmg_spec, BlockSewingSpec)
    assert atmg_spec.local_block_size == 8
    assert atmg_spec.translations == ((1.0, 0.0), (0.0, 1.0))

    assert isinstance(tdbg_spec, BlockSewingSpec)
    assert tdbg_spec.local_block_size == 4
    assert tdbg_spec.block_labels.tolist() == [0, 0]

    assert isinstance(rlg_spec, BlockSewingSpec)
    assert rlg_spec.local_block_size == 2
    assert rlg_spec.translations == ((-1.0, 0.0), (0.0, -1.0))

    assert isinstance(htqg_spec, BlockSewingSpec)
    assert htqg_spec.local_block_size == 8
    assert htqg_spec.translations == ((1.0, 0.0), (0.0, 1.0))
