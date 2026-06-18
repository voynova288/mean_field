from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from mean_field.api import HFResult, load_result
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_projected_basis_consistent,
)
from mean_field.systems.htg import (
    HTGModel,
    HTGParams,
    InteractionParams,
    htg_supercell_hf_run_to_hf_result,
    htg_supercell_hf_run_to_hf_run_result,
    run_htg_hf,
)
from mean_field.systems.htg.mean_field_adapter import htg_hf_run_to_hf_result, htg_hf_run_to_hf_run_result
from mean_field.systems.htg.supercell import run_htg_supercell_hf


def _tiny_htg_supercell_run():
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    return run_htg_supercell_hf(
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


def _tiny_htg_primitive_run():
    model = HTGModel.from_config(1.8, n_shells=0, params=HTGParams.kwan2023())
    return run_htg_hf(
        model,
        InteractionParams(n_k=1, g_shells=0),
        nu=3.0,
        mesh_size=1,
        g_shells=0,
        max_iter=1,
        init_mode="bm",
        seed=2,
        use_numba=False,
    )


def test_htg_supercell_hf_run_to_hf_run_result_preserves_canonical_arrays() -> None:
    run = _tiny_htg_supercell_run()

    canonical = htg_supercell_hf_run_to_hf_run_result(run, archive_manifest={"state": "hf_supercell.npz"})

    assert isinstance(canonical, ContractHFRunResult)
    assert canonical.archive_manifest == {"state": "hf_supercell.npz"}
    assert canonical.best_seed == 1
    assert canonical.init_mode == "bm"
    assert canonical.converged == run.converged
    assert canonical.exit_reason == run.exit_reason
    assert len(canonical.iteration_history) == len(run.iter_err)

    final = canonical.final_state
    assert_projected_basis_consistent(final.basis)
    assert_hamiltonian_parts_consistent(final.hamiltonian)
    assert_density_state_consistent(final.density, require_projector=False)
    np.testing.assert_allclose(final.density.density_delta, run.state.density)
    np.testing.assert_allclose(final.hamiltonian.fixed, run.state.hamiltonian - run.state.h0)
    np.testing.assert_allclose(final.hamiltonian.hartree, np.zeros_like(run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.fock, np.zeros_like(run.state.h0))
    assert final.hamiltonian.metadata["component_resolution"] == "collapsed_total_minus_h0"
    assert final.hamiltonian.metadata["supports_crpa"] is False
    assert final.eigenvectors_active.size == 0
    assert final.observables["eigenvectors_active_available"] is False
    assert np.isclose(final.density.metadata["filling_from_density"], 3.5)
    assert final.density.reference.metadata["raw_density_convention"] == "stored_delta"


def test_htg_supercell_hf_result_helper_save_writes_canonical_sidecar(tmp_path) -> None:
    run = _tiny_htg_supercell_run()
    result = htg_supercell_hf_run_to_hf_result(run)

    assert isinstance(result, HFResult)
    assert result.state is run
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.model.system_name == "htg_supercell"
    assert result.config.density_convention == "stored_delta"
    assert result.observables["raw_density_convention"] == "stored_delta"

    result.save(tmp_path)

    sidecar = json.loads((tmp_path / "canonical_hf_run_result.json").read_text(encoding="utf-8"))
    loaded = load_result(tmp_path)
    assert loaded.canonical_hf_run_result is not None
    assert loaded.canonical_hf_run_result == sidecar
    assert sidecar["final_state"]["density"]["density_delta_definition"] == "P-R"
    assert sidecar["final_state"]["density"]["convention"] == "delta"
    assert sidecar["final_state"]["density"]["metadata"]["raw_density_convention"] == "stored_delta"
    assert sidecar["final_state"]["density"]["density_delta_shape"] == [run.state.nt, run.state.nt, run.state.nk]
    assert sidecar["final_state"]["basis"]["k_grid_frac_shape"] == [run.state.nk, 2]
    assert sidecar["final_state"]["hamiltonian"]["metadata"]["supports_crpa"] is False


def test_htg_supercell_contract_adapter_requires_k_grid_frac() -> None:
    run = _tiny_htg_supercell_run()
    bad_run = replace(run, basis_data=replace(run.basis_data, k_grid_frac=None))

    with pytest.raises(ValueError, match="k_grid_frac"):
        htg_supercell_hf_run_to_hf_run_result(bad_run)


def test_htg_primitive_hf_run_to_hf_run_result_preserves_canonical_arrays() -> None:
    run = _tiny_htg_primitive_run()

    canonical = htg_hf_run_to_hf_run_result(run, archive_manifest={"state": "hf_primitive.npz"})

    assert isinstance(canonical, ContractHFRunResult)
    assert canonical.archive_manifest == {"state": "hf_primitive.npz"}
    assert canonical.best_seed == 2
    assert canonical.init_mode == "bm"
    assert canonical.converged == run.converged
    assert canonical.exit_reason == run.exit_reason
    assert len(canonical.iteration_history) == len(run.iter_err)

    final = canonical.final_state
    assert_projected_basis_consistent(final.basis)
    assert_hamiltonian_parts_consistent(final.hamiltonian)
    assert_density_state_consistent(final.density, require_projector=False)
    assert final.basis.physical_model.system == "htg"
    assert final.basis.k_grid_frac.shape == (run.state.nk, 2)
    assert final.hamiltonian.h0.shape == run.state.h0.shape == (run.state.nt, run.state.nt, run.state.nk)
    assert final.hamiltonian.total.shape == run.state.hamiltonian.shape
    assert final.energies.shape == run.state.energies.shape == (run.state.nt, run.state.nk)
    np.testing.assert_allclose(final.density.density_delta, run.state.density)
    np.testing.assert_allclose(final.hamiltonian.h0, run.state.h0)
    np.testing.assert_allclose(final.hamiltonian.fixed, run.state.hamiltonian - run.state.h0)
    np.testing.assert_allclose(final.hamiltonian.hartree, np.zeros_like(run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.fock, np.zeros_like(run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.total, run.state.hamiltonian)
    assert final.hamiltonian.metadata["component_resolution"] == "collapsed_total_minus_h0"
    assert final.hamiltonian.metadata["supports_crpa"] is False
    assert final.eigenvectors_active.size == 0
    assert final.observables["eigenvectors_active_available"] is False
    assert np.isclose(final.density.metadata["filling_from_density"], 3.0)
    assert final.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert final.density.reference.metadata["system"] == "htg"


def test_htg_primitive_hf_result_helper_save_writes_canonical_sidecar(tmp_path) -> None:
    run = _tiny_htg_primitive_run()
    result = htg_hf_run_to_hf_result(run)

    assert isinstance(result, HFResult)
    assert result.state is run
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.model.system_name == "htg"
    assert result.config.density_convention == "stored_delta"
    assert result.observables["raw_density_convention"] == "stored_delta"

    result.save(tmp_path)

    sidecar = json.loads((tmp_path / "canonical_hf_run_result.json").read_text(encoding="utf-8"))
    loaded = load_result(tmp_path)
    assert loaded.canonical_hf_run_result is not None
    assert loaded.canonical_hf_run_result == sidecar
    assert sidecar["final_state"]["basis"]["system"] == "htg"
    assert sidecar["final_state"]["basis"]["k_grid_frac_shape"] == [run.state.nk, 2]
    assert sidecar["final_state"]["density"]["density_delta_definition"] == "P-R"
    assert sidecar["final_state"]["density"]["convention"] == "delta"
    assert sidecar["final_state"]["density"]["metadata"]["raw_density_convention"] == "stored_delta"
    assert sidecar["final_state"]["density"]["density_delta_shape"] == [run.state.nt, run.state.nt, run.state.nk]
    assert sidecar["final_state"]["hamiltonian"]["h0_shape"] == [run.state.nt, run.state.nt, run.state.nk]
    assert sidecar["final_state"]["hamiltonian"]["total_shape"] == [run.state.nt, run.state.nt, run.state.nk]
    assert sidecar["final_state"]["hamiltonian"]["metadata"]["supports_crpa"] is False
