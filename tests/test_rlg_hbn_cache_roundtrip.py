from __future__ import annotations

import numpy as np

from mean_field.systems.RnG_hBN.screening import (
    InterlayerHartreeResult,
    LayerChargeResult,
    ScreenedInterlayerPotentialResult,
    ScreeningIteration,
)
from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
)
from mean_field.systems.RnG_hBN.cache import (
    load_layer_overlap_blocks_cache,
    load_or_build_layer_overlap_blocks,
    load_or_build_projected_basis,
    load_or_solve_screening,
    load_projected_basis_cache,
)


def _small_model() -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )


def _small_interaction(**overrides: object) -> RLGhBNInteractionParams:
    values = {
        "scheme": "average",
        "active_valence_bands": 1,
        "active_conduction_bands": 1,
        "k_mesh_size": 1,
        "interaction_cutoff_q1": 1.0,
        "use_screened_basis": True,
    }
    values.update(overrides)
    return RLGhBNInteractionParams(**values)


def _fake_screening(model: RLGhBNModel, *, screened_u_mev: float = 11.0) -> ScreenedInterlayerPotentialResult:
    layer_count = model.params.layer_count
    charge = LayerChargeResult(
        layer_charge=np.arange(layer_count, dtype=float),
        reference_layer_charge=np.zeros(layer_count, dtype=float),
        delta_layer_charge=np.arange(layer_count, dtype=float),
        mesh_size=1,
        n_spin=2,
        valleys=(1, -1),
        n_valence_bands=0,
    )
    hartree = InterlayerHartreeResult(
        layer_potential_mev=np.linspace(-1.0, 1.0, layer_count, dtype=float),
        interlayer_slope_mev=float(screened_u_mev - model.params.displacement_field_mev),
        delta_layer_charge=charge.delta_layer_charge.copy(),
        moire_cell_area_nm2=123.0,
    )
    return ScreenedInterlayerPotentialResult(
        external_v_mev=float(model.params.displacement_field_mev),
        screened_u_mev=float(screened_u_mev),
        converged=True,
        iterations=(
            ScreeningIteration(
                iteration=0,
                screened_u_mev=float(screened_u_mev),
                interlayer_hartree_mev=float(hartree.interlayer_slope_mev),
                candidate_u_mev=float(screened_u_mev),
                residual_mev=0.0,
            ),
        ),
        layer_charge=charge,
        hartree=hartree,
        residual_mev=0.0,
        method="test_fixture",
        mesh_size=1,
    )


def test_projected_basis_cache_roundtrip(tmp_path) -> None:
    model = _small_model()
    interaction = _small_interaction()
    built = load_or_build_projected_basis(
        model,
        interaction,
        cache_dir=tmp_path,
        cache_policy="refresh",
        mesh_size=1,
        screening=_fake_screening(model),
    )

    loaded = load_projected_basis_cache(tmp_path, built.key)
    original = built.value

    assert np.array_equal(loaded.h0, original.h0)
    assert np.array_equal(loaded.physical_h0, original.physical_h0)
    assert np.array_equal(loaded.fixed_remote_hamiltonian, original.fixed_remote_hamiltonian)
    assert np.array_equal(loaded.basis.wavefunctions, original.basis.wavefunctions)
    assert np.array_equal(loaded.band_energies, original.band_energies)
    assert loaded.active_band_indices == original.active_band_indices
    assert loaded.flat_band_indices == original.flat_band_indices
    assert loaded.valleys == original.valleys


def test_overlap_cache_roundtrip(tmp_path) -> None:
    model = _small_model()
    interaction = _small_interaction()
    basis = load_or_build_projected_basis(
        model,
        interaction,
        cache_dir=tmp_path,
        cache_policy="refresh",
        mesh_size=1,
        screening=_fake_screening(model),
    )
    built = load_or_build_layer_overlap_blocks(
        basis.value,
        cache_dir=tmp_path,
        cache_policy="refresh",
        basis_cache_key=basis.key,
        shifts=((0, 0),),
    )

    loaded = load_layer_overlap_blocks_cache(tmp_path, built.key)
    original = built.value

    assert loaded.shifts == original.shifts
    assert np.array_equal(loaded.gvecs, original.gvecs)
    for shift in original.shifts:
        assert np.array_equal(loaded.layer_overlaps[shift], original.layer_overlaps[shift])
        assert np.array_equal(loaded.layer_diagonal_overlaps[shift], original.layer_diagonal_overlaps[shift])
        assert np.array_equal(loaded.hartree_layer_coulomb[shift], original.hartree_layer_coulomb[shift])
        assert np.array_equal(loaded.fock_layer_coulomb[shift], original.fock_layer_coulomb[shift])


def test_screening_cache_reuse_roundtrip(tmp_path) -> None:
    model = _small_model()
    interaction = _small_interaction(k_mesh_size=1)
    first = load_or_solve_screening(
        model,
        interaction,
        cache_dir=tmp_path,
        cache_policy="refresh",
        solver="fixed_point",
        mesh_size=1,
        fixed_point_max_iter=2,
    )
    second = load_or_solve_screening(
        model,
        interaction,
        cache_dir=tmp_path,
        cache_policy="reuse",
        solver="fixed_point",
        mesh_size=1,
        fixed_point_max_iter=2,
    )

    assert not first.hit
    assert second.hit
    assert second.key == first.key
    assert np.isclose(second.value.screened_u_mev, first.value.screened_u_mev)
    assert np.array_equal(second.value.layer_charge.layer_charge, first.value.layer_charge.layer_charge)
