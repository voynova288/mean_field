from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import HFConfig, run_hf
from mean_field.api.hf import list_hf_adapters
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_projected_basis_consistent,
)
from mean_field.core.hf import (
    HartreeFockProblem,
    conventional_projector_to_stored,
    empty_overlap_block_set,
    run_hartree_fock_problem,
)
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
from mean_field.systems.tmbg.polshyn_supercell import (
    PolshynDoubledCell,
    PolshynProjectedBasis,
    PolshynRunHFConfig,
    PolshynWangHFState,
    build_polshyn_projected_basis,
    build_wang_hf_problem,
    cdw_density_blocks,
    flatten_sector_blocks,
    polshyn_nu_7over2_filling_summary,
    polshyn_wang_hf_bundle_to_hf_run_result,
    translation_order_parameters,
    unflatten_sector_blocks,
    wang_sector_density_blocks,
)


def test_polshyn_hf_filling_summary_matches_nu_7over2_convention() -> None:
    projected = (25, 26, 27, 28)
    summary = polshyn_nu_7over2_filling_summary(projected, target_band_index=27)

    assert summary.projected_indices == projected
    assert summary.target_primitive_position == 2
    assert summary.target_fold_indices == (4, 5)
    assert summary.nb == 8
    assert summary.area_ratio == 2
    assert np.allclose(summary.reference_diagonal, [1, 1, 1, 1, 0, 0, 0, 0])
    assert summary.occupation_counts.tolist() == [[5, 6], [6, 6]]
    assert np.isclose(summary.primitive_nu, 3.5, atol=1.0e-12)
    assert summary.matches_expected_filling

    as_dict = summary.to_dict()
    assert as_dict["target_fold_indices"] == [4, 5]
    assert as_dict["matches_expected_filling"] is True


def test_polshyn_cdw_initializer_has_maximal_target_fold_order() -> None:
    projected = (25, 26, 27, 28)
    summary = polshyn_nu_7over2_filling_summary(projected, target_band_index=27)

    density = cdw_density_blocks(
        projected_indices=projected,
        target_band_index=27,
        n_spin=2,
        n_eta=2,
        nb=summary.nb,
        nk=3,
        reference_diagonal=summary.reference_diagonal,
    )
    order = translation_order_parameters(
        density,
        projected_indices=projected,
        target_band_index=27,
        spin_index=0,
        valley_index=0,
    )

    assert np.allclose(order["target_x2"], np.ones(3))
    assert np.isclose(order["target_x2_mean"], 1.0, atol=1.0e-12)
    assert np.allclose(density[0, 0, 4, 5, :], 0.5)
    assert np.allclose(density[0, 0, 5, 4, :], 0.5)
    assert np.allclose(density[1, 0, 4, 5, :], 0.0)
    assert np.allclose(density[0, 1, 4, 5, :], 0.0)


def test_polshyn_wang_hf_problem_builder_uses_common_problem_api() -> None:
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    h0[:, :, 0] = np.diag([-1.0, 1.0])
    state = PolshynWangHFState(
        h0=h0.copy(),
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=np.zeros((2, 1), dtype=float),
        mu=0.0,
        precision=1.0e-12,
        v0=1.0,
        diagnostics={},
    )

    problem = build_wang_hf_problem(
        state,
        empty_overlap_block_set(),
        occupation_counts=np.asarray([[1]], dtype=int),
        reference_diagonal=np.asarray([0.0, 0.0], dtype=float),
        n_spin=1,
        n_eta=1,
        nb=2,
    )
    run = run_hartree_fock_problem(state, problem, init_mode="toy_wang", seed=0, max_iter=1)

    assert isinstance(problem, HartreeFockProblem)
    assert run.iterations == 1
    assert state.density.shape == h0.shape
    assert np.isclose(state.density[0, 0, 0].real, 1.0)
    assert np.isclose(state.density[1, 1, 0].real, 0.0)


def test_polshyn_sector_flatten_round_trip_preserves_block_layout() -> None:
    blocks = np.zeros((2, 2, 3, 3, 2), dtype=np.complex128)
    for ispin in range(2):
        for ieta in range(2):
            for ib in range(3):
                for jb in range(3):
                    for ik in range(2):
                        blocks[ispin, ieta, ib, jb, ik] = (
                            1000 * ispin
                            + 100 * ieta
                            + 10 * ib
                            + jb
                            + 0.1 * ik
                            + 1j * (ib - jb)
                        )

    flat = flatten_sector_blocks(blocks)
    restored = unflatten_sector_blocks(flat, n_spin=2, n_eta=2, nb=3)

    assert flat.shape == (12, 12, 2)
    assert np.allclose(restored, blocks)


def _toy_polshyn_wang_bundle() -> tuple[PolshynProjectedBasis, PolshynWangHFState, np.ndarray]:
    model = TMBGModel.from_config(1.25, n_shells=0, params=TMBGParameters.minimal())
    supercell = PolshynDoubledCell()
    super_b1, super_b2 = supercell.reciprocal_vectors(model.lattice)
    nk = 2
    n_spin = 2
    n_eta = 2
    nb = 2
    nt = n_spin * n_eta * nb
    h0_blocks = np.zeros((n_spin, n_eta, nb, nb, nk), dtype=np.complex128)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for ik in range(nk):
                h0_blocks[ispin, ieta, :, :, ik] = np.diag([0.01 * (1 + ispin + ieta), 0.2 + 0.01 * ik])
    h0_flat = flatten_sector_blocks(h0_blocks)

    state_index = np.arange(nt, dtype=int).reshape((n_spin, n_eta, nb), order="F")
    i0 = int(state_index[0, 0, 0])
    i1 = int(state_index[0, 0, 1])
    conventional_delta = np.zeros_like(h0_flat)
    conventional_projector_block = np.asarray(
        [[0.5, -0.5j], [0.5j, 0.5]],
        dtype=np.complex128,
    )
    for ik in range(nk):
        conventional_delta[np.ix_([i0, i1], [i0, i1], [ik])] = conventional_projector_block[:, :, None]
    wang_stored_delta = conventional_projector_to_stored(conventional_delta)

    basis = PolshynProjectedBasis(
        model=model,
        supercell=supercell,
        kvec=np.asarray([0.0 + 0.0j, 0.25 * super_b1], dtype=np.complex128),
        k_grid_frac=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=float),
        projected_indices=(27,),
        target_band_index=27,
        wavefunctions=np.zeros((6, nb, n_eta, nk), dtype=np.complex128),
        h0_blocks=h0_blocks,
        reference_diagonal=np.zeros((nb,), dtype=float),
        super_b1=complex(super_b1),
        super_b2=complex(super_b2),
        embedding_shape=(1, 1),
        embedding_origin=(0, 0),
        embedding_positions={},
    )
    state = PolshynWangHFState(
        h0=h0_flat.copy(),
        density=wang_stored_delta.copy(),
        hamiltonian=h0_flat.copy(),
        energies=np.stack([np.diag(h0_flat[:, :, ik]).real for ik in range(nk)], axis=1),
        mu=0.0,
        precision=1.0e-8,
        v0=1.0,
        diagnostics={"hf_energy": -0.25},
    )
    return basis, state, conventional_delta


def test_polshyn_wang_bundle_adapter_preserves_stored_density_orientation_without_fake_history() -> None:
    basis, state, conventional_delta = _toy_polshyn_wang_bundle()
    info = {
        "mode": "polshyn_projected_hf_wang",
        "iterations": 5,
        "converged": True,
        "exit_reason": "converged",
        "init_mode": "toy_wang",
        "seed": 11,
        "final_raw_norm": 0.0,
    }

    canonical = polshyn_wang_hf_bundle_to_hf_run_result(basis, state, info, archive_manifest={"state": "toy.npz"})

    assert isinstance(canonical, ContractHFRunResult)
    assert canonical.archive_manifest == {"state": "toy.npz"}
    assert canonical.best_seed == 11
    assert canonical.iteration_history == []
    assert canonical.final_state.observables["iteration_history_source"] == "unavailable_in_polshyn_wang_info"

    final = canonical.final_state
    assert_projected_basis_consistent(final.basis)
    assert_hamiltonian_parts_consistent(final.hamiltonian)
    assert_density_state_consistent(final.density)
    np.testing.assert_allclose(final.density.density_delta, state.density)
    assert final.density.metadata["raw_density_projector_orientation"] == "wang_xiaoyu_stored_P_star"
    assert final.basis.metadata["density_projector_orientation"] == "wang_xiaoyu_stored_P_star"
    assert np.isclose(final.density.filling, 0.5)
    assert final.density.n_occupied_total == 2

    state_index = np.arange(8, dtype=int).reshape((2, 2, 2), order="F")
    i0 = int(state_index[0, 0, 0])
    i1 = int(state_index[0, 0, 1])
    assert np.isclose(conventional_delta[i0, i1, 0], -0.5j)
    assert np.isclose(final.density.density_delta[i0, i1, 0], 0.5j)

    conventional_blocks = unflatten_sector_blocks(conventional_delta, n_spin=2, n_eta=2, nb=2)
    np.testing.assert_allclose(wang_sector_density_blocks(state, basis), conventional_blocks)


def test_polshyn_wang_bundle_adapter_preserves_explicit_iteration_history_only() -> None:
    basis, state, _conventional_delta = _toy_polshyn_wang_bundle()
    history = [{"iteration": 1, "energy": -0.2, "error": 0.1, "oda_lambda": 0.75}]
    info = {
        "converged": False,
        "exit_reason": "max_iter",
        "init_mode": "toy_wang",
        "seed": 3,
        "iteration_history": history,
    }

    canonical = polshyn_wang_hf_bundle_to_hf_run_result(basis, state, info)

    assert canonical.iteration_history == history
    assert canonical.final_state.observables["iteration_history_available"] is True
    assert canonical.final_state.observables["iteration_history_source"] == "info.iteration_history"


def test_polshyn_wang_bundle_adapter_refuses_to_invent_seed() -> None:
    basis, state, _conventional_delta = _toy_polshyn_wang_bundle()
    info = {"converged": True, "exit_reason": "converged", "init_mode": "toy_wang"}

    with pytest.raises(ValueError, match="seed"):
        polshyn_wang_hf_bundle_to_hf_run_result(basis, state, info)


def test_polshyn_projected_basis_builder_embeds_doubled_cell_shapes() -> None:
    model = TMBGModel.from_config(1.25, n_shells=0, params=TMBGParameters.minimal())
    basis = build_polshyn_projected_basis(
        model,
        mesh_size=1,
        projected_indices=(2,),
        target_band_index=2,
    )
    assert basis.nk == 1
    assert basis.n_spin == 2
    assert basis.n_eta == 2
    assert basis.nb == 2
    assert basis.wavefunctions.shape == (12, 2, 2, 1)
    assert basis.h0_blocks.shape == (2, 2, 2, 2, 1)
    assert np.count_nonzero(np.abs(basis.h0_blocks[:, :, 0, 1, :])) == 0
    np.testing.assert_allclose(basis.reference_diagonal, [0.0, 0.0])


def test_polshyn_public_run_hf_explicit_config_smoke_attaches_canonical_result() -> None:
    adapters = list_hf_adapters(system_name="tmbg_polshyn", adapter_type="run_hf")
    assert any(adapter.name == "tmbg_polshyn_explicit_run_hf" for adapter in adapters)
    model = TMBGModel.from_config(1.25, n_shells=0, params=TMBGParameters.minimal())
    polshyn_config = PolshynRunHFConfig(
        mesh_size=1,
        projected_indices=(2,),
        target_band_index=2,
        shifts=(),
        v0=0.0,
        epsilon_r=9.0,
        d_sc_nm=12.0,
        max_iter=1,
        precision=1.0e-7,
        seed=5,
    )
    cfg = HFConfig(
        filling=3.5,
        mesh=(1, 1),
        active_band_indices=(2,),
        density_convention="stored_delta",
        epsilon_r=9.0,
        dsc_nm=12.0,
        max_iter=1,
        precision=1.0e-7,
    )
    result = run_hf(model, cfg, tmbg_polshyn_config=polshyn_config)
    assert result.model.system_name == "tmbg_polshyn"
    assert result.canonical_run_result is not None
    assert result.canonical_run_result.best_seed == 5
    assert result.observables["explicit_config_type"] == "PolshynRunHFConfig"
    assert result.observables["primitive_nu"] == pytest.approx(3.5)
    assert result.artifacts is not None
    assert result.artifacts.metadata["workflow"] == "tmbg.polshyn_wang.explicit_config"
