from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import HFConfig, load_result, run_hf
from mean_field.api.hf import list_hf_adapters
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_projected_basis_consistent,
)
from dataclasses import replace

from mean_field.core.hf import (
    HFOverlapBlockSet,
    HartreeFockProblem,
    build_projected_interaction_hamiltonian,
    conventional_projector_to_stored,
    empty_overlap_block_set,
    run_hartree_fock_problem,
)
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
from mean_field.systems.tmbg.polshyn_supercell import (
    PolshynDoubledCell,
    PolshynH0SubtractionConfig,
    PolshynH0SubtractionResult,
    PolshynProjectedBasis,
    PolshynRunHFConfig,
    PolshynWangHFState,
    apply_polshyn_h0_subtraction,
    basis_with_polshyn_h0_correction,
    build_polshyn_projected_basis,
    build_wang_hf_problem,
    cdw_density_blocks,
    compute_polshyn_active_reference_h0_correction,
    compute_polshyn_minus_full_p0_h0_correction,
    flatten_sector_blocks,
    polshyn_nu_7over2_filling_summary,
    polshyn_reference_projector_blocks,
    polshyn_wang_hf_bundle_to_hf_run_result,
    translation_order_parameters,
    unflatten_sector_blocks,
    wang_sector_density_blocks,
    wang_stored_density_from_sector_blocks,
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


def _toy_hartree_q0_overlap(nt: int, nk: int) -> HFOverlapBlockSet:
    overlap = np.zeros((int(nt), int(nk), int(nt), int(nk)), dtype=np.complex128)
    diagonal = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    for ik in range(int(nk)):
        diagonal[:, :, ik] = np.eye(int(nt), dtype=np.complex128)
    return HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): diagonal},
        hartree_screening={(0, 0): 1.0},
        fock_screening={},
    )


def test_polshyn_h0_subtraction_config_normalizes_modes_and_fixed_signs() -> None:
    assert PolshynH0SubtractionConfig("active_reference").mode == "active-reference"
    assert PolshynH0SubtractionConfig("active-reference").applied_sign == 1.0
    assert PolshynH0SubtractionConfig("minus_full_p0").mode == "minus-full-p0"
    assert PolshynH0SubtractionConfig("minus-full-p0").applied_sign == -1.0
    assert PolshynH0SubtractionConfig("none").enabled is False
    with pytest.raises(ValueError, match="Unsupported Polshyn h0_subtraction"):
        PolshynH0SubtractionConfig("paper-figure")
    with pytest.raises(ValueError, match="decoupled-layers"):
        PolshynH0SubtractionConfig("minus-full-p0", p0_reference="bernal-bilayer")


def test_polshyn_reference_projector_blocks_match_reference_diagonal() -> None:
    basis, _state, _conventional_delta = _toy_polshyn_wang_bundle()
    basis = replace(basis, reference_diagonal=np.asarray([1.0, 0.0], dtype=float))
    blocks = polshyn_reference_projector_blocks(basis)
    assert blocks.shape == (basis.n_spin, basis.n_eta, basis.nb, basis.nb, basis.nk)
    for ispin in range(basis.n_spin):
        for ieta in range(basis.n_eta):
            for ik in range(basis.nk):
                np.testing.assert_allclose(blocks[ispin, ieta, :, :, ik], np.diag([1.0, 0.0]))


def test_polshyn_active_reference_h0_correction_matches_common_interaction_and_q0_policy() -> None:
    basis, _state, _conventional_delta = _toy_polshyn_wang_bundle()
    basis = replace(basis, reference_diagonal=np.asarray([1.0, 0.0], dtype=float))
    nt = basis.n_spin * basis.n_eta * basis.nb
    overlaps = _toy_hartree_q0_overlap(nt, basis.nk)
    correction_zeroed, diag_zeroed = compute_polshyn_active_reference_h0_correction(
        basis,
        overlaps,
        v0=2.0,
        zero_hartree_q0=True,
    )
    np.testing.assert_allclose(correction_zeroed, 0.0)
    assert diag_zeroed["mode"] == "active-reference"
    correction, diag = compute_polshyn_active_reference_h0_correction(
        basis,
        overlaps,
        v0=2.0,
        zero_hartree_q0=False,
    )
    ref_blocks = polshyn_reference_projector_blocks(basis)
    ref_flat = wang_stored_density_from_sector_blocks(ref_blocks)
    expected = unflatten_sector_blocks(
        build_projected_interaction_hamiltonian(ref_flat, overlaps, v0=2.0, beta=1.0),
        n_spin=basis.n_spin,
        n_eta=basis.n_eta,
        nb=basis.nb,
    )
    np.testing.assert_allclose(correction, expected)
    assert diag["zero_hartree_q0"] is False
    assert diag["h0_correction_norm_ev"] > 0.0


def test_basis_with_polshyn_h0_correction_validates_shape_and_symmetrizes() -> None:
    basis, _state, _conventional_delta = _toy_polshyn_wang_bundle()
    with pytest.raises(ValueError, match="incompatible"):
        basis_with_polshyn_h0_correction(basis, np.zeros((1, 1), dtype=np.complex128))
    correction = np.zeros_like(basis.h0_blocks)
    correction[0, 0, 0, 1, 0] = 1.0 + 2.0j
    corrected = basis_with_polshyn_h0_correction(basis, correction)
    np.testing.assert_allclose(corrected.h0_blocks, np.swapaxes(corrected.h0_blocks.conjugate(), 2, 3))


def test_minus_full_p0_h0_correction_empty_overlap_is_shape_safe() -> None:
    basis, _state, _conventional_delta = _toy_polshyn_wang_bundle()
    raw, diag = compute_polshyn_minus_full_p0_h0_correction(
        basis,
        empty_overlap_block_set(),
        v0=0.0,
    )
    np.testing.assert_allclose(raw, 0.0)
    assert raw.shape == basis.h0_blocks.shape
    assert diag["mode"] == "minus-full-p0"
    assert "projected_p0_trace_mean" in diag
    result = apply_polshyn_h0_subtraction(
        basis,
        empty_overlap_block_set(),
        config=PolshynH0SubtractionConfig("minus-full-p0"),
        v0=0.0,
    )
    assert isinstance(result, PolshynH0SubtractionResult)
    assert result.diagnostics["applied_sign"] == -1.0
    np.testing.assert_allclose(result.corrected_basis.h0_blocks, basis.h0_blocks)


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


def _minimal_polshyn_public_run_hf_result(h0_subtraction=None):
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
        h0_subtraction=PolshynH0SubtractionConfig() if h0_subtraction is None else h0_subtraction,
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
    return run_hf(model, cfg, tmbg_polshyn_config=polshyn_config)


def test_polshyn_public_run_hf_explicit_config_smoke_attaches_canonical_result() -> None:
    adapters = list_hf_adapters(system_name="tmbg_polshyn", adapter_type="run_hf")
    assert any(adapter.name == "tmbg_polshyn_explicit_run_hf" for adapter in adapters)
    result = _minimal_polshyn_public_run_hf_result()
    assert result.model.system_name == "tmbg_polshyn"
    assert result.canonical_run_result is not None
    assert result.canonical_run_result.best_seed == 5
    assert result.observables["explicit_config_type"] == "PolshynRunHFConfig"
    assert result.observables["h0_subtraction_mode"] == "none"
    assert result.observables["primitive_nu"] == pytest.approx(3.5)
    assert result.artifacts is not None
    assert result.artifacts.metadata["workflow"] == "tmbg.polshyn_wang.explicit_config"
    assert result.artifacts.metadata["h0_subtraction"]["mode"] == "none"


def test_polshyn_public_run_hf_records_active_reference_h0_metadata() -> None:
    result = _minimal_polshyn_public_run_hf_result(PolshynH0SubtractionConfig("active-reference"))
    assert result.observables["h0_subtraction_mode"] == "active-reference"
    assert result.observables["h0_subtraction_applied_sign"] == 1.0
    assert result.artifacts is not None
    assert result.artifacts.metadata["h0_subtraction"]["mode"] == "active-reference"
    assert result.canonical_run_result is not None
    assert result.canonical_run_result.archive_manifest["h0_subtraction"]["mode"] == "active-reference"


def test_polshyn_public_run_hf_records_minus_full_p0_h0_metadata() -> None:
    result = _minimal_polshyn_public_run_hf_result(PolshynH0SubtractionConfig("minus_full_p0"))
    assert result.observables["h0_subtraction_mode"] == "minus-full-p0"
    assert result.observables["h0_subtraction_applied_sign"] == -1.0
    assert result.artifacts is not None
    assert result.artifacts.metadata["h0_subtraction"]["mode"] == "minus-full-p0"
    assert result.canonical_run_result is not None
    assert result.canonical_run_result.archive_manifest["h0_subtraction"]["mode"] == "minus-full-p0"


def test_polshyn_public_facade_exports_h0_helpers_without_private_compact_helpers() -> None:
    from mean_field.systems.tmbg import polshyn_supercell

    assert polshyn_supercell.PolshynH0SubtractionConfig is PolshynH0SubtractionConfig
    assert hasattr(polshyn_supercell, "apply_polshyn_h0_subtraction")
    assert not hasattr(polshyn_supercell, "_compact_overlap_between")


def test_polshyn_public_run_hf_metadata_only_save_is_cheap_and_loadable(tmp_path: Path) -> None:
    result = _minimal_polshyn_public_run_hf_result()
    manifest_path = result.save(tmp_path / "polshyn_hf", canonical_payload="metadata_only")
    loaded = load_result(manifest_path.parent)
    assert manifest_path.name == "manifest.json"
    assert loaded.model is not None
    assert loaded.model["system_name"] == "tmbg_polshyn"
    assert loaded.canonical_hf_run_result is not None
    assert loaded.canonical_hf_run_result["contract_type"] == "mean_field.core.contracts.HFRunResult"
    assert loaded.manifest["files"]["canonical_hf_run_result"] == "canonical_hf_run_result.json"
    assert "canonical_hf_arrays" not in loaded.manifest["files"]
    assert not (manifest_path.parent / "canonical_hf_arrays.npz").exists()
