from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from mean_field.api import HFConfig, HFResult, HFState as APIHFState, ModelRecord, load_result
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_no_screened_diag_h0_for_RnG,
    assert_projected_basis_consistent,
)
from mean_field.core.hf import ProjectedWavefunctionBasis
from mean_field.systems.RnG_hBN import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    RLGhBNProjectedBasisData,
    active_band_indices_for_interaction,
    rlg_hbn_hf_run_to_hf_run_result,
    rlg_hbn_reference_density,
)


def _toy_rlg_hbn_run(
    *,
    scheme: str = "average",
    active_valence_bands: int = 0,
    active_conduction_bands: int = 1,
    nu: float = 1.0,
    use_screened_basis: bool = False,
) -> RLGhBNHartreeFockRun:
    nk = 2
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=40.0,
        shell_count=1,
    )
    basis_model = (
        RLGhBNModel.from_config(
            layer_count=3,
            xi=1,
            theta_deg=0.77,
            displacement_field_mev=25.0,
            shell_count=1,
        )
        if use_screened_basis
        else model
    )
    interaction = RLGhBNInteractionParams(
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        active_conduction_bands=active_conduction_bands,
        k_mesh_size=nk,
        use_screened_basis=use_screened_basis,
    )
    n_band = int(active_valence_bands + active_conduction_bands)
    valleys = (1, -1)
    basis = ProjectedWavefunctionBasis(
        wavefunctions=np.ones((1, n_band, len(valleys), nk), dtype=np.complex128),
        grid_shape=(1, 1),
        n_spin=2,
        local_basis_size=1,
        name="toy_rlg_hbn_contract_basis",
    )
    nt = basis.nt
    state_index = np.arange(nt, dtype=int).reshape((basis.n_spin, basis.n_flavor, n_band), order="F")
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    fixed = np.zeros_like(h0)
    for ik in range(nk):
        for istate in range(nt):
            h0[istate, istate, ik] = 0.01 * (istate + 1) + 0.001 * ik
            fixed[istate, istate, ik] = -0.0001 * (istate + 1)
    band_energies = np.zeros((n_band, len(valleys), nk), dtype=float)
    for iband in range(n_band):
        for ieta in range(len(valleys)):
            band_energies[iband, ieta, :] = 1.0 + 0.1 * iband + 0.01 * ieta + np.arange(nk)
    active_indices = active_band_indices_for_interaction(basis_model, interaction)
    basis_data = RLGhBNProjectedBasisData(
        model=model,
        basis_model=basis_model,
        interaction=interaction,
        screening=None,
        mesh_size=nk,
        kvec=np.asarray([0.0 + 0.0j, 0.25 + 0.0j], dtype=np.complex128),
        k_grid_frac=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=float),
        basis=basis,
        h0=h0,
        band_energies=band_energies,
        active_band_indices=active_indices,
        flat_band_indices=(active_indices[0], active_indices[-1]),
        valleys=valleys,
        reciprocal_grid_shape=(1, 1),
        reciprocal_grid_origin=(0, 0),
        moire_cell_area_nm2=2.0,
        physical_h0=h0.copy(),
        fixed_remote_hamiltonian=np.zeros_like(h0),
    )
    reference = rlg_hbn_reference_density(
        nt,
        nk,
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        n_spin=basis.n_spin,
        n_eta=basis.n_flavor,
    )
    projector = np.zeros_like(reference)
    occupied_per_k = int(round(basis.n_spin * basis.n_flavor * active_valence_bands + nu))
    for ik in range(nk):
        occupied: list[int] = []
        for ispin in range(basis.n_spin):
            for ieta in range(basis.n_flavor):
                for iband in range(active_valence_bands):
                    occupied.append(int(state_index[ispin, ieta, iband]))
        if len(occupied) < occupied_per_k:
            for ispin in range(basis.n_spin):
                for ieta in range(basis.n_flavor):
                    for iband in range(active_valence_bands, n_band):
                        candidate = int(state_index[ispin, ieta, iband])
                        if candidate not in occupied:
                            occupied.append(candidate)
                        if len(occupied) == occupied_per_k:
                            break
                    if len(occupied) == occupied_per_k:
                        break
                if len(occupied) == occupied_per_k:
                    break
        for istate in occupied:
            projector[istate, istate, ik] = 1.0
    state = RLGhBNHartreeFockState(
        h0=h0.copy(),
        density=projector - reference,
        hamiltonian=h0 + fixed,
        energies=np.stack([np.diag((h0 + fixed)[:, :, ik]).real for ik in range(nk)], axis=1),
        reference_density=reference,
        nu=float(nu),
        v0=float(basis_data.v0),
        active_valence_bands=active_valence_bands,
        scheme=scheme,
        mu=0.123,
        precision=1.0e-8,
        n_spin=basis.n_spin,
        n_eta=basis.n_flavor,
        n_band=n_band,
        occupation_counts=tuple(1 if index < occupied_per_k else 0 for index in range(nt)),
        diagnostics={"hf_energy": -1.25, "nonfinite_ignored": float("nan")},
    )
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=np.asarray([-1.0, -1.25], dtype=float),
        iter_err=np.asarray([0.2, 0.05], dtype=float),
        iter_oda=np.asarray([1.0, 0.75], dtype=float),
        init_mode="toy",
        seed=7,
        converged=True,
        exit_reason="converged",
        overlap_blocks=SimpleNamespace(),
        basis_data=basis_data,
    )


def test_rlg_hbn_hf_run_to_hf_run_result_preserves_canonical_arrays() -> None:
    run = _toy_rlg_hbn_run()

    canonical = rlg_hbn_hf_run_to_hf_run_result(run, archive_manifest={"state": "hf_ground_state.npz"})

    assert isinstance(canonical, ContractHFRunResult)
    assert canonical.archive_manifest == {"state": "hf_ground_state.npz"}
    assert canonical.best_seed == 7
    assert canonical.init_mode == "toy"
    assert canonical.converged is True
    assert canonical.exit_reason == "converged"
    assert canonical.iteration_history[-1] == {"iteration": 2, "energy": -1.25, "error": 0.05, "oda_lambda": 0.75}

    final = canonical.final_state
    assert_projected_basis_consistent(final.basis)
    assert_density_state_consistent(final.density)
    assert_hamiltonian_parts_consistent(final.hamiltonian)
    np.testing.assert_allclose(final.density.density_delta, run.state.density)
    np.testing.assert_allclose(final.density.projector, run.state.density + run.state.reference_density)
    assert final.density.reference.scheme == "average"
    assert final.density.n_occupied_total == 2
    assert np.isclose(final.density.metadata["filling_from_density"], 1.0)
    assert final.density.reference.metadata["raw_density_convention"] == "stored_delta"
    np.testing.assert_allclose(final.hamiltonian.fixed, run.state.hamiltonian - run.state.h0)
    np.testing.assert_allclose(final.hamiltonian.hartree, np.zeros_like(run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.fock, np.zeros_like(run.state.h0))
    assert final.hamiltonian.metadata["component_resolution"] == "collapsed_total_minus_h0"
    assert final.hamiltonian.metadata["supports_crpa"] is False
    assert final.basis.metadata["supports_crpa"] is False
    assert final.basis.metadata["projection_mode"] == "bare"
    assert final.eigenvectors_active.size == 0
    assert final.observables["eigenvectors_active_available"] is False
    assert "nonfinite_ignored" not in final.diagnostics


def test_rlg_hbn_contract_adapter_normalizes_cn_scheme_and_preserves_trace() -> None:
    run = _toy_rlg_hbn_run(scheme="cn", active_valence_bands=1, active_conduction_bands=1, nu=1.0)

    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    assert canonical.final_state.density.reference.scheme == "CN"
    assert canonical.final_state.density.n_occupied_total == 10
    assert np.isclose(canonical.final_state.density.filling, 1.0)
    assert np.isclose(canonical.final_state.density.metadata["filling_from_density"], 1.0)
    assert_density_state_consistent(canonical.final_state.density)


def test_rlg_hbn_contract_adapter_records_screened_basis_h0_rule() -> None:
    run = _toy_rlg_hbn_run(use_screened_basis=True)

    canonical = rlg_hbn_hf_run_to_hf_run_result(run)

    assert canonical.final_state.basis.metadata["projection_mode"] == "screened"
    assert canonical.final_state.basis.metadata["h0_rule"] == "project_H_sp_V_into_H_sp_U_basis"
    assert canonical.final_state.basis.metadata["physical_model_displacement_mev"] == 40.0
    assert canonical.final_state.basis.metadata["basis_model_displacement_mev"] == 25.0
    assert_no_screened_diag_h0_for_RnG(canonical.final_state.basis)


def test_rlg_hbn_hf_result_save_writes_canonical_sidecar(tmp_path) -> None:
    run = _toy_rlg_hbn_run()
    canonical = rlg_hbn_hf_run_to_hf_run_result(run)
    result = HFResult(
        model=ModelRecord(system_name="RnG_hBN"),
        config=HFConfig(filling=1.0, mesh=(1, 2)),
        state=APIHFState(density=run.state.density),
        canonical_run_result=canonical,
    )

    result.save(tmp_path)

    sidecar = json.loads((tmp_path / "canonical_hf_run_result.json").read_text(encoding="utf-8"))
    loaded = load_result(tmp_path)
    assert loaded.canonical_hf_run_result is not None
    assert sidecar["final_state"]["basis"]["system"] == "RnG_hBN"
    assert sidecar["final_state"]["basis"]["metadata"]["projection_mode"] == "bare"
    assert sidecar["final_state"]["basis"]["metadata"]["supports_crpa"] is False
    assert sidecar["final_state"]["density"]["density_delta_definition"] == "P-R"
    assert sidecar["final_state"]["density"]["reference_scheme"] == "average"
    assert sidecar["final_state"]["density"]["density_delta_shape"] == [run.state.nt, run.state.nt, run.state.nk]
    assert sidecar["final_state"]["hamiltonian"]["metadata"]["supports_crpa"] is False
    assert sidecar["iteration_history"]["count"] == len(run.iter_err)
