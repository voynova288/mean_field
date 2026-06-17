from __future__ import annotations

import numpy as np

from mean_field.systems.htg import HTGModel, HTGParams, InteractionParams
from mean_field.systems.htg.supercell import (
    build_htg_supercell_projected_basis,
    extract_htg_supercell_inspection_scf_grid_path,
    htg_doubled_fractional_supercell,
    htg_minimal_fractional_supercell,
    htg_supercell_filling_from_density,
    htg_supercell_occupied_count_per_k,
    htg_supercell_reference_diagonal,
    htg_tripled_fractional_supercell,
    run_htg_supercell_hf,
    supercell_fold_representatives,
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
