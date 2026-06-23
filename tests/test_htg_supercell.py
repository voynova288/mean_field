from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.htg import HTGModel, HTGParams, InteractionParams
from mean_field.systems.htg.mean_field_adapter import reconstruct_htg_projected_hf_micro_wavefunctions
from mean_field.systems.htg.supercell import (
    build_htg_supercell_hf_wavefunction_grid,
    build_htg_supercell_projected_basis,
    extract_htg_supercell_inspection_scf_grid_path,
    htg_doubled_fractional_supercell,
    htg_minimal_fractional_supercell,
    htg_supercell_filling_from_density,
    htg_supercell_full_boundary_sewing_transform,
    htg_supercell_occupied_count_per_k,
    htg_supercell_reference_diagonal,
    htg_tripled_fractional_supercell,
    run_htg_supercell_hf,
    supercell_fold_representatives,
)
from mean_field.systems.htg.supercell_contracts import reconstruct_htg_supercell_projected_hf_micro_wavefunctions


def _f_order_basis_index(local: int, ix: int, iy: int, *, local_size: int, nx: int, ny: int) -> int:
    return int(np.ravel_multi_index((int(local), int(ix), int(iy)), (int(local_size), int(nx), int(ny)), order="F"))


def _direct_sum_row(
    ispin: int,
    ieta: int,
    basis_index: int,
    *,
    n_eta: int,
    basis_dimension: int,
) -> int:
    return int((int(ispin) * int(n_eta) + int(ieta)) * int(basis_dimension) + int(basis_index))


def _toy_projected_hf_run(*, supercell: bool) -> SimpleNamespace:
    n_spin, n_eta, n_band = 2, 2, 2
    local_size, nx, ny = 2, 2, 3
    basis_dimension = local_size * nx * ny
    nk = 4
    nt = n_spin * n_eta * n_band
    raw = np.zeros((basis_dimension, n_band, n_eta, nk), dtype=np.complex128)
    for ik in range(nk):
        for ieta in range(n_eta):
            for iband in range(n_band):
                for local in range(local_size):
                    for ix in range(nx):
                        for iy in range(ny):
                            p = _f_order_basis_index(local, ix, iy, local_size=local_size, nx=nx, ny=ny)
                            raw[p, iband, ieta, ik] = complex(
                                1000 * ik + 100 * ieta + 10 * iband + 3 * local + ix + 0.1 * iy
                            )
    basis = SimpleNamespace(
        wavefunctions=raw,
        n_spin=n_spin,
        n_flavor=n_eta,
        n_band=n_band,
        basis_dimension=basis_dimension,
        local_basis_size=local_size,
        grid_shape=(nx, ny),
        nt=nt,
        nk=nk,
    )
    hamiltonian = np.repeat(np.diag(np.arange(nt, dtype=float)).astype(np.complex128)[:, :, None], nk, axis=2)
    k_grid_frac = np.stack(np.meshgrid(np.arange(2) / 2.0, np.arange(2) / 2.0, indexing="ij"), axis=-1)
    data_kwargs = {
        "mesh_size": 2,
        "kvec": np.arange(nk, dtype=np.complex128),
        "k_grid_frac": k_grid_frac,
        "basis": basis,
        "h0": hamiltonian,
        "primitive_projected_indices": (10, 11),
        "primitive_band_count": 2,
        "fold_representatives": ((0, 0),),
        "reciprocal_grid_shape": (nx, ny),
        "reciprocal_grid_origin": (0, 0),
    }
    if supercell:
        data_kwargs.update(
            {
                "supercell": SimpleNamespace(area_ratio=1, n11=1, n12=0, n21=0, n22=1),
                "nk": nk,
                "nt": nt,
            }
        )
    else:
        data_kwargs.update(
            {
                "projected_band_indices": (10, 11),
                "central_band_indices": (10, 11),
                "nk": nk,
                "nt": nt,
            }
        )
    return SimpleNamespace(
        basis_data=SimpleNamespace(**data_kwargs),
        state=SimpleNamespace(hamiltonian=hamiltonian, energies=np.arange(nt, dtype=float)[:, None] + np.zeros((nt, nk))),
    )


def test_htg_primitive_projected_hf_reconstruction_is_explicit_and_guarded() -> None:
    run = _toy_projected_hf_run(supercell=False)
    state_index = np.arange(8, dtype=int).reshape((2, 2, 2), order="F")
    selected = int(state_index[1, 0, 1])

    selected_elements = int(run.basis_data.nk * 48)
    full_state_elements = int(run.basis_data.nk * 48 * run.basis_data.basis.nt)
    bundle = reconstruct_htg_projected_hf_micro_wavefunctions(
        run,
        band_indices=(selected,),
        max_dense_elements=selected_elements,
    )

    assert bundle.psi_micro.shape == (2, 2, 48, 1)
    assert bundle.sewing_transforms == ()
    assert bundle.basis_metadata["projected_hf_reconstruction"] == "explicit_dense_opt_in"
    assert bundle.basis_metadata["canonical_wrapping_dense_by_default"] is False
    assert bundle.basis_metadata["selected_hf_band_indices"] == [selected]
    assert bundle.basis_metadata["dense_reconstruction_estimated_elements"] == selected_elements
    assert bundle.basis_metadata["dense_reconstruction_estimated_all_state_elements"] == full_state_elements
    assert bundle.basis_metadata["n_reconstructed_hf_states"] == 1
    assert bundle.basis_metadata["selected_state_allocation"] == "output_axis_contains_only_selected_hf_states"
    flat = bundle.psi_micro.reshape((4, 48, 1))
    expected = np.zeros(48, dtype=np.complex128)
    row_start = _direct_sum_row(1, 0, 0, n_eta=2, basis_dimension=12)
    expected[row_start : row_start + 12] = run.basis_data.basis.wavefunctions[:, 1, 0, 0]
    np.testing.assert_allclose(flat[0, :, 0], expected)

    with pytest.raises(ValueError, match="dense reconstruction"):
        reconstruct_htg_projected_hf_micro_wavefunctions(
            run,
            band_indices=(selected,),
            max_dense_elements=selected_elements - 1,
        )


def test_htg_primitive_projected_hf_reconstruction_contracts_selected_mixed_state() -> None:
    run = _toy_projected_hf_run(supercell=False)
    state_index = np.arange(8, dtype=int).reshape((2, 2, 2), order="F")
    active_a = int(state_index[0, 0, 0])
    active_b = int(state_index[0, 0, 1])
    hamiltonian = np.zeros_like(run.state.hamiltonian)
    for ik in range(int(run.basis_data.nk)):
        hamiltonian[:, :, ik] = np.diag(np.arange(run.basis_data.basis.nt, dtype=float) + 10.0)
        hamiltonian[active_a, active_a, ik] = -0.25
        hamiltonian[active_b, active_b, ik] = 0.50
        hamiltonian[active_a, active_b, ik] = 0.20
        hamiltonian[active_b, active_a, ik] = 0.20
    run.state.hamiltonian = hamiltonian
    selected = 0

    selected_elements = int(run.basis_data.nk * 48)
    bundle = reconstruct_htg_projected_hf_micro_wavefunctions(
        run,
        band_indices=(selected,),
        max_dense_elements=selected_elements,
    )

    assert bundle.psi_micro.shape == (2, 2, 48, 1)
    assert bundle.basis_metadata["n_reconstructed_hf_states"] == 1
    flat = bundle.psi_micro.reshape((4, 48, 1))
    evals, evecs = np.linalg.eigh(hamiltonian[:, :, 0])
    assert evals[selected] < 0.0
    expected = np.zeros(48, dtype=np.complex128)
    row_start = _direct_sum_row(0, 0, 0, n_eta=2, basis_dimension=12)
    expected[row_start : row_start + 12] = (
        run.basis_data.basis.wavefunctions[:, 0, 0, 0] * evecs[active_a, selected]
        + run.basis_data.basis.wavefunctions[:, 1, 0, 0] * evecs[active_b, selected]
    )
    np.testing.assert_allclose(flat[0, :, 0], expected)


def test_htg_supercell_boundary_sewing_respects_direct_sum_f_order_row_layout() -> None:
    n_spin, n_eta = 2, 2
    local_size, nx, ny = 2, 2, 3
    basis_dimension = local_size * nx * ny
    basis_data = SimpleNamespace(
        basis=SimpleNamespace(n_spin=n_spin, n_flavor=n_eta, local_basis_size=local_size, grid_shape=(nx, ny))
    )
    vector = np.zeros(n_spin * n_eta * basis_dimension, dtype=np.complex128)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for local in range(local_size):
                for ix in range(nx):
                    for iy in range(ny):
                        p = _f_order_basis_index(local, ix, iy, local_size=local_size, nx=nx, ny=ny)
                        row = _direct_sum_row(ispin, ieta, p, n_eta=n_eta, basis_dimension=basis_dimension)
                        vector[row] = complex(1000 * ispin + 100 * ieta + 10 * local + ix + 0.1 * iy)

    sew_x = htg_supercell_full_boundary_sewing_transform(basis_data, 1, 0)
    shifted = sew_x(vector)
    expected = np.zeros_like(vector)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for local in range(local_size):
                for ix in range(nx - 1):
                    for iy in range(ny):
                        dst_p = _f_order_basis_index(local, ix, iy, local_size=local_size, nx=nx, ny=ny)
                        src_p = _f_order_basis_index(local, ix + 1, iy, local_size=local_size, nx=nx, ny=ny)
                        dst = _direct_sum_row(ispin, ieta, dst_p, n_eta=n_eta, basis_dimension=basis_dimension)
                        src = _direct_sum_row(ispin, ieta, src_p, n_eta=n_eta, basis_dimension=basis_dimension)
                        expected[dst] = vector[src]
    np.testing.assert_allclose(shifted, expected)

    frame = np.column_stack([vector, 10_000.0 + vector])
    expected_frame = np.column_stack([expected, np.where(expected != 0, 10_000.0 + expected, 0.0)])
    np.testing.assert_allclose(sew_x(frame), expected_frame)


def test_htg_supercell_projected_hf_reconstruction_attaches_validated_sewing_only_on_explicit_call() -> None:
    run = _toy_projected_hf_run(supercell=True)
    state_index = np.arange(8, dtype=int).reshape((2, 2, 2), order="F")
    selected = int(state_index[0, 1, 0])

    selected_elements = int(run.basis_data.nk * 48)
    full_state_elements = int(run.basis_data.nk * 48 * run.basis_data.basis.nt)
    bundle = reconstruct_htg_supercell_projected_hf_micro_wavefunctions(
        run,
        band_indices=selected,
        max_dense_elements=selected_elements,
    )

    assert bundle.psi_micro.shape == (2, 2, 48, 1)
    assert len(bundle.sewing_transforms) == 2
    assert bundle.basis_metadata["sewing_policy"] == "htg_supercell_full_boundary_sewing_transforms"
    assert "tests/test_htg_supercell.py" in bundle.basis_metadata["sewing_row_order_validation"]
    assert bundle.basis_metadata["dense_reconstruction_estimated_elements"] == selected_elements
    assert bundle.basis_metadata["dense_reconstruction_estimated_all_state_elements"] == full_state_elements
    assert bundle.basis_metadata["n_reconstructed_hf_states"] == 1
    assert bundle.basis_metadata["selected_state_allocation"] == "output_axis_contains_only_selected_hf_states"
    flat = bundle.psi_micro.reshape((4, 48, 1))
    expected = np.zeros(48, dtype=np.complex128)
    row_start = _direct_sum_row(0, 1, 0, n_eta=2, basis_dimension=12)
    expected[row_start : row_start + 12] = run.basis_data.basis.wavefunctions[:, 0, 1, 0]
    np.testing.assert_allclose(flat[0, :, 0], expected)

    without_sewing = reconstruct_htg_supercell_projected_hf_micro_wavefunctions(
        run,
        band_indices=selected,
        attach_sewing=False,
    )
    assert without_sewing.sewing_transforms == ()
    assert without_sewing.basis_metadata["sewing_policy"] == "not_attached_by_request"

    with pytest.raises(ValueError, match="dense reconstruction"):
        reconstruct_htg_supercell_projected_hf_micro_wavefunctions(
            run,
            band_indices=selected,
            max_dense_elements=selected_elements - 1,
        )


def test_htg_minimal_supercell_filling_counts_for_requested_fractions() -> None:
    third = htg_tripled_fractional_supercell()
    half = htg_doubled_fractional_supercell()

    assert htg_minimal_fractional_supercell(3.0 + 1.0 / 3.0) == third
    assert htg_minimal_fractional_supercell(3.5) == half
    assert htg_minimal_fractional_supercell(3.0 + 2.0 / 3.0) == third

    third_reference = htg_supercell_reference_diagonal(2, third.area_ratio)
    half_reference = htg_supercell_reference_diagonal(2, half.area_ratio)
    assert third.area_ratio == 3
    assert half.area_ratio == 2
    assert len(supercell_fold_representatives(third)) == 3
    assert len(supercell_fold_representatives(half)) == 2
    assert third_reference.shape == (6,)
    assert half_reference.shape == (4,)
    assert np.allclose(third_reference, 0.5)
    assert np.allclose(half_reference, 0.5)

    assert htg_supercell_occupied_count_per_k(
        3.0 + 1.0 / 3.0,
        reference_diagonal=third_reference,
        area_ratio=third.area_ratio,
    ) == 22
    assert htg_supercell_occupied_count_per_k(
        3.5,
        reference_diagonal=half_reference,
        area_ratio=half.area_ratio,
    ) == 15
    assert htg_supercell_occupied_count_per_k(
        3.0 + 2.0 / 3.0,
        reference_diagonal=third_reference,
        area_ratio=third.area_ratio,
    ) == 23


def test_htg_minimal_supercell_basis_has_folded_band_count() -> None:
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    half_basis = build_htg_supercell_projected_basis(
        model,
        InteractionParams(n_k=1, g_shells=0),
        supercell=htg_doubled_fractional_supercell(),
        mesh_size=1,
        projected_band_count=2,
    )
    third_basis = build_htg_supercell_projected_basis(
        model,
        InteractionParams(n_k=1, g_shells=0),
        supercell=htg_tripled_fractional_supercell(),
        mesh_size=1,
        projected_band_count=2,
    )
    assert half_basis.basis.n_band == 4
    assert half_basis.basis.nt == 16
    assert half_basis.h0.shape == (16, 16, 1)
    assert third_basis.basis.n_band == 6
    assert third_basis.basis.nt == 24
    assert third_basis.h0.shape == (24, 24, 1)


def test_htg_supercell_scf_grid_path_uses_saved_grid_indices() -> None:
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    basis = build_htg_supercell_projected_basis(
        model,
        InteractionParams(n_k=6, g_shells=0),
        supercell=htg_tripled_fractional_supercell(),
        mesh_size=6,
        projected_band_count=2,
    )
    samples = extract_htg_supercell_inspection_scf_grid_path(basis)
    assert samples.unique_grid_count > 0
    assert samples.exact_node_hit_mask.tolist() == [True, True, True, True]
    assert np.all(samples.grid_indices >= 0)
    assert np.all(samples.grid_indices < basis.nk)


def test_htg_supercell_hf_wavefunction_grid_reconstructs_micro_basis_columns() -> None:
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    run = run_htg_supercell_hf(
        model,
        InteractionParams(n_k=1, g_shells=0),
        primitive_nu=3.5,
        mesh_size=1,
        g_shells=0,
        max_iter=1,
        init_mode="bm",
        seed=1,
        use_numba=False,
    )

    grid = build_htg_supercell_hf_wavefunction_grid(run.state.hamiltonian, run.basis_data, band_indices=(0, 1))

    expected_micro_dim = int(
        run.basis_data.basis.n_spin * run.basis_data.basis.n_flavor * run.basis_data.basis.basis_dimension
    )
    assert grid.wavefunctions.shape == (1, 1, expected_micro_dim, 2)
    assert grid.energies.shape == (2, 1, 1)
    assert grid.k_grid_frac.shape == (1, 1, 2)
    assert grid.band_indices == (0, 1)
    norms = np.einsum("ijbs,ijbs->s", grid.wavefunctions.conj(), grid.wavefunctions).real
    np.testing.assert_allclose(norms, np.ones(2), atol=1.0e-12)


def test_htg_supercell_tiny_hf_run_preserves_fractional_filling() -> None:
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    run = run_htg_supercell_hf(
        model,
        InteractionParams(n_k=1, g_shells=0),
        primitive_nu=3.5,
        mesh_size=1,
        g_shells=0,
        max_iter=1,
        init_mode="bm",
        seed=1,
        use_numba=False,
    )
    assert run.basis_data.supercell == htg_doubled_fractional_supercell()
    assert run.state.n_band == 4
    assert run.state.nt == 16
    assert np.isclose(
        htg_supercell_filling_from_density(
            run.state.density,
            reference_diagonal=run.state.reference_diagonal,
            area_ratio=run.basis_data.supercell.area_ratio,
        ),
        3.5,
    )
