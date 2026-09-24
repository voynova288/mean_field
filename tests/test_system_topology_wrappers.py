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
from mean_field.systems.RnG_hBN import (
    RLGhBNModel,
)
from mean_field.systems.RnG_hBN.sewing import (
    rlg_hbn_projected_micro_basis_sewing,
)
from mean_field.systems.atmg import ATMGModel
from mean_field.systems.tdbg import TDBGModel, TDBGParameters
from mean_field.systems.tmbg import TMBGModel
import mean_field.systems.RnG_hBN.topology as rlg_topology
import mean_field.systems.atmg.topology as atmg_topology
import mean_field.systems.htqg.topology as htqg_topology
import mean_field.systems.tdbg.topology as tdbg_topology
import mean_field.systems.tmbg.topology as tmbg_topology


def _fake_grid(*, basis_dim: int = 2, band_label: int = 7) -> GridBandsResult:
    frac = np.arange(2, dtype=float) / 2.0
    return GridBandsResult(
        k_grid_frac=np.stack(np.meshgrid(frac, frac, indexing="ij"), axis=-1),
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
        (tmbg_topology, {"basis_sewing": _trivial_basis_sewing()}),
        (atmg_topology, {"basis_sewing": _trivial_basis_sewing()}),
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
    state = tmbg_topology.fhs_state_from_grid_result(
        _fake_grid(basis_dim=3, band_label=11),
        11,
        valley=1,
        use_boundary_sewing=False,
    )

    assert state.state_indices == (0,)
    assert state.reported_indices == (11,)
    assert state.metadata["absolute_band_indices"] == [11]
    assert state.metadata["column_indices"] == [0]

    with pytest.raises(ValueError, match="Requested band labels"):
        tmbg_topology.fhs_state_from_grid_result(
            _fake_grid(basis_dim=3, band_label=11),
            12,
            valley=1,
            use_boundary_sewing=False,
        )


def test_rlg_hbn_grid_state_fails_closed_when_requested_sewing_lacks_metadata() -> None:
    grid = _fake_grid()
    with pytest.raises(ValueError, match="lattice and params are required"):
        rlg_topology.fhs_state_from_grid_result(grid, 7, valley=1)

    diagnostic = rlg_topology.fhs_state_from_grid_result(
        grid,
        7,
        valley=1,
        use_boundary_sewing=False,
    )
    assert diagnostic.basis_sewing is None
    assert diagnostic.metadata["boundary_sewing"] is False


def test_raw_finite_basis_state_builders_require_sewing_or_diagnostic_opt_out() -> None:
    grid = _fake_grid(basis_dim=2)
    eigenvectors = grid.eigenvectors
    assert eigenvectors is not None

    for builder, value, state_index in (
        (tdbg_topology.fhs_state_from_eigenvectors, eigenvectors, 0),
        (tdbg_topology.fhs_state_from_grid_result, grid, 7),
    ):
        with pytest.raises(ValueError, match="lattice is required"):
            builder(value, state_index, valley=1)
        diagnostic = builder(
            value,
            state_index,
            valley=1,
            use_boundary_sewing=False,
        )
        assert diagnostic.basis_sewing is None
        assert diagnostic.metadata["boundary_sewing"] is False

    with pytest.raises(ValueError, match="lattice and params are required"):
        rlg_topology.fhs_state_from_eigenvectors(eigenvectors, 0, valley=1)
    rlg_diagnostic = rlg_topology.fhs_state_from_eigenvectors(
        eigenvectors,
        0,
        valley=1,
        use_boundary_sewing=False,
    )
    assert rlg_diagnostic.basis_sewing is None
    assert rlg_diagnostic.metadata["boundary_sewing"] is False


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
    explicit = compute_lattice_topology(
        fhs_state_from_wavefunctions(
            wavefunctions, 0, sewing_transforms=(sew_1, sew_2)
        )
    )

    np.testing.assert_allclose(from_state.link_1, explicit.link_1)
    np.testing.assert_allclose(from_state.link_2, explicit.link_2)
    np.testing.assert_allclose(from_state.berry_curvature, explicit.berry_curvature)
    assert from_state.chern_number == pytest.approx(explicit.chern_number)


@pytest.mark.parametrize(
    "name,model",
    [
        ("tmbg", TMBGModel.from_config(1.2, n_shells=2)),
        ("atmg", ATMGModel.from_config(3, 1.2, n_shells=2)),
    ],
)
def test_plane_wave_basis_sewing_matches_interior_hamiltonian_covariance(
    name,
    model,
) -> None:
    spec = (
        tmbg_topology.tmbg_basis_sewing(model.lattice)
        if name == "tmbg"
        else atmg_topology.atmg_basis_sewing(model.lattice, model.params)
    )
    seams = sewing_transforms_from_block_spec(spec)
    reciprocal_vectors = (model.lattice.g_m1, model.lattice.g_m2)
    dimension = (
        model.lattice.matrix_dim if name == "tmbg" else model.matrix_dim
    )
    momentum = 0.137 * model.lattice.g_m1 + 0.211 * model.lattice.g_m2

    for seam, reciprocal_vector in zip(seams, reciprocal_vectors, strict=True):
        transform = seam(np.eye(dimension, dtype=np.complex128))
        retained = np.flatnonzero(np.linalg.norm(transform, axis=1) > 0.5)
        assert retained.size < dimension
        assert retained.size > 0
        for valley in (1, -1):
            source = model.build_hamiltonian(momentum, valley=valley)
            target = model.build_hamiltonian(
                momentum + reciprocal_vector,
                valley=valley,
            )
            mapped = transform @ source @ transform.conjugate().T
            np.testing.assert_allclose(
                target[np.ix_(retained, retained)],
                mapped[np.ix_(retained, retained)],
                atol=5.0e-13,
                rtol=0.0,
            )


def test_rlg_hbn_valley_sewing_matches_interior_hamiltonian_covariance() -> None:
    model = RLGhBNModel.from_config(layer_count=3, xi=1, shell_count=2)
    momentum = 0.137 * model.lattice.g_m1 + 0.211 * model.lattice.g_m2
    for valley in (1, -1):
        spec = rlg_topology.rlg_hbn_basis_sewing(
            model.lattice,
            model.params,
            valley=valley,
        )
        seams = sewing_transforms_from_block_spec(spec)
        reciprocal_vectors = (model.lattice.g_m1, model.lattice.g_m2)
        for seam, reciprocal_vector in zip(
            seams,
            reciprocal_vectors,
            strict=True,
        ):
            transform = seam(
                np.eye(model.lattice.matrix_dim, dtype=np.complex128)
            )
            retained = np.flatnonzero(np.linalg.norm(transform, axis=1) > 0.5)
            source = model.build_hamiltonian(momentum, valley=valley)
            target = model.build_hamiltonian(
                momentum + reciprocal_vector,
                valley=valley,
            )
            mapped = transform @ source @ transform.conjugate().T
            np.testing.assert_allclose(
                target[np.ix_(retained, retained)],
                mapped[np.ix_(retained, retained)],
                atol=5.0e-13,
                rtol=0.0,
            )


def test_tdbg_actual_state_fhs_chern_uses_common_topology_pipeline() -> None:
    model = TDBGModel.from_config(
        1.38,
        cut=2.0,
        params=TDBGParameters.full(stacking="AB-BA", Delta=0.09),
    )
    conduction_band = model.matrix_dim // 2

    chern_k = compute_lattice_topology(
        model.fhs_state_on_grid(5, conduction_band, valley=1, n_bands=conduction_band + 1)
    )
    chern_kprime = compute_lattice_topology(
        model.fhs_state_on_grid(5, conduction_band, valley=-1, n_bands=conduction_band + 1)
    )

    assert chern_k.chern_number == pytest.approx(1.0, abs=1.0e-8)
    assert chern_kprime.chern_number == pytest.approx(-1.0, abs=1.0e-8)
    assert chern_k.metadata["boundary_sewing"] is True
    assert chern_k.min_link_magnitude > 0.1
    assert chern_k.minimum_plaquette_branch_margin > 0.05
    assert chern_kprime.minimum_plaquette_branch_margin > 0.05


def test_generic_basis_sewing_specs_are_metadata_not_topology_calculators() -> None:
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
    plane_wave_lattice = SimpleNamespace(
        g_indices=np.asarray([[0, 0], [1, 0]], dtype=int)
    )
    tmbg_spec = tmbg_topology.tmbg_basis_sewing(plane_wave_lattice)
    atmg_spec = atmg_topology.atmg_basis_sewing(
        plane_wave_lattice,
        SimpleNamespace(n_layers=3),
    )
    htqg_spec = htqg_topology.htqg_basis_sewing(
        plane_wave_lattice,
        valley=-1,
    )

    assert isinstance(tdbg_spec, BlockSewingSpec)
    assert tdbg_spec.local_block_size == 4
    assert tdbg_spec.block_labels.tolist() == [0, 0]

    assert isinstance(rlg_spec, BlockSewingSpec)
    assert rlg_spec.local_block_size == 2
    assert rlg_spec.translations == ((-1.0, 0.0), (0.0, -1.0))

    assert isinstance(tmbg_spec, BlockSewingSpec)
    assert tmbg_spec.local_block_size == 6
    assert tmbg_spec.translations == ((1.0, 0.0), (0.0, 1.0))

    assert isinstance(atmg_spec, BlockSewingSpec)
    assert atmg_spec.local_block_size == 6
    assert atmg_spec.translations == ((1.0, 0.0), (0.0, 1.0))

    assert isinstance(htqg_spec, BlockSewingSpec)
    assert htqg_spec.local_block_size == 8
    assert htqg_spec.translations == ((-1.0, 0.0), (0.0, -1.0))

    projected_micro = rlg_hbn_projected_micro_basis_sewing(
        local_basis_size=1,
        grid_shape=(3, 2),
        spin_count=1,
        valley_signs=(1, -1),
    )
    sew_x, _ = sewing_transforms_from_block_spec(projected_micro)
    vector = np.zeros(12, dtype=np.complex128)
    vector[1] = 2.0
    vector[6 + 1 + 3] = 3.0
    shifted = sew_x(vector)
    assert shifted[0] == 2.0
    assert shifted[6 + 2 + 3] == 3.0
    assert np.count_nonzero(shifted) == 2
