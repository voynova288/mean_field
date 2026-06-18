from __future__ import annotations

import json

import numpy as np

from mean_field.api import (
    ArtifactManifest,
    ConventionBundle,
    HFConfig,
    HFResult,
    HFState,
    ModelRecord,
    load_result,
    required_artifact_files,
    update_artifact_manifest,
    write_contract_artifacts,
)
from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    ReferenceDensity as ContractReferenceDensity,
    SingleParticleModel as ContractSingleParticleModel,
)


def test_required_artifact_schema_names_are_stable() -> None:
    required = required_artifact_files()

    assert required == (
        "manifest.json",
        "model.json",
        "config.yaml",
        "conventions.json",
        "environment.json",
        "validation.json",
        "observables.json",
    )


def test_convention_bundle_serializes_density_axis_contract() -> None:
    payload = ConventionBundle().to_dict()

    assert payload["energy_unit"] == "meV"
    assert payload["momentum_unit"] == "nm^-1"
    assert payload["density_convention"] == "stored_delta"
    assert payload["density_axis_order"] == "abk"
    assert payload["valley_labels"] == {"K": 1, "Kprime": -1}


def test_hf_result_save_writes_public_manifest_files(tmp_path) -> None:
    model = ModelRecord(system_name="toy")
    cfg = HFConfig(filling=0.0, mesh=(1, 1))
    state = HFState(density=np.zeros((1, 1, 1), dtype=np.complex128))
    result = HFResult(model=model, config=cfg, state=state, observables={"gap_mev": 1.0})

    manifest_path = result.save(tmp_path)
    loaded = load_result(tmp_path)

    assert manifest_path == tmp_path / "manifest.json"
    assert {path.name for path in tmp_path.iterdir()} >= set(required_artifact_files())
    assert json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))["system_name"] == "toy"
    assert json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8"))["mesh"] == [1, 1]
    assert loaded.config is not None and loaded.config["mesh"] == [1, 1]
    assert loaded.conventions is not None and loaded.conventions["density_convention"] == "stored_delta"
    assert loaded.validation == {}
    assert loaded.observables == {"gap_mev": 1.0}
    assert loaded.manifest["root"] == str(tmp_path)


def test_hf_result_save_writes_canonical_hf_run_result_sidecar(tmp_path) -> None:
    def h_builder(kvec: np.ndarray) -> np.ndarray:
        return np.zeros((1, 1, np.asarray(kvec).size), dtype=np.complex128)

    def diagonalizer(kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = np.asarray(kvec).size
        return np.zeros((1, size), dtype=float), np.ones((1, 1, size), dtype=np.complex128)

    model_contract = ContractSingleParticleModel(
        system="toy",
        lattice=None,
        params={},
        hamiltonian_builder=h_builder,
        diagonalizer=diagonalizer,
    )
    h0 = np.zeros((1, 1, 1), dtype=np.complex128)
    basis = ContractProjectedBasis(
        physical_model=model_contract,
        basis_model=model_contract,
        kvec=np.asarray([0.0 + 0.0j]),
        k_grid_frac=np.asarray([[0.0, 0.0]], dtype=float),
        h0=h0,
        basis_energies=np.zeros((1, 1), dtype=float),
        active_band_indices=(0,),
        active_valence_bands=0,
        active_conduction_bands=1,
        micro_wavefunctions=np.ones((1, 1, 1), dtype=np.complex128),
        metadata={"source": "unit_test"},
    )
    reference = ContractReferenceDensity(scheme="custom", reference=np.zeros_like(h0))
    density = ContractDensityState(
        density_delta=np.ones_like(h0),
        reference=reference,
        filling=1.0,
        n_occupied_total=1,
    )
    hamiltonian = ContractHamiltonianParts(
        h0=h0,
        fixed=np.zeros_like(h0),
        hartree=np.zeros_like(h0),
        fock=np.zeros_like(h0),
        total=h0,
        density_input_convention="delta",
        metadata={"supports_crpa": False},
    )
    final_state = ContractHFState(
        basis=basis,
        density=density,
        hamiltonian=hamiltonian,
        energies=np.zeros((1, 1), dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=0.0,
        observables={"eigenvectors_active_available": False},
        diagnostics={"final_raw_norm": 0.0},
    )
    canonical = ContractHFRunResult(
        final_state=final_state,
        iteration_history=[{"iteration": 1, "energy": 0.0, "error": 0.0, "oda_lambda": 1.0}],
        converged=True,
        exit_reason="converged",
        best_seed=1,
        init_mode="toy",
    )
    result = HFResult(
        model=ModelRecord(system_name="toy"),
        config=HFConfig(filling=1.0, mesh=(1, 1)),
        state=HFState(density=np.ones_like(h0)),
        canonical_run_result=canonical,
    )

    result.save(tmp_path)

    sidecar = json.loads((tmp_path / "canonical_hf_run_result.json").read_text(encoding="utf-8"))
    loaded = load_result(tmp_path)
    assert loaded.manifest["files"]["canonical_hf_run_result"] == "canonical_hf_run_result.json"
    assert loaded.manifest["metadata"]["canonical_hf_run_result"]["contract_type"] == "mean_field.core.contracts.HFRunResult"
    assert sidecar["contract_type"] == "mean_field.core.contracts.HFRunResult"
    assert sidecar["iteration_history"]["count"] == 1
    assert sidecar["final_state"]["density"]["reference_scheme"] == "custom"
    assert sidecar["final_state"]["density"]["density_delta_shape"] == [1, 1, 1]
    assert sidecar["final_state"]["basis"]["h0_shape"] == [1, 1, 1]
    assert sidecar["final_state"]["hamiltonian"]["metadata"]["supports_crpa"] is False


def test_write_contract_artifacts_writes_schema_sidecars_and_npz_summary(tmp_path) -> None:
    state_path = tmp_path / "hf_state.npz"
    np.savez(state_path, density=np.zeros((2, 2, 3), dtype=np.complex128))

    paths = write_contract_artifacts(
        tmp_path,
        workflow="toy.workflow",
        system_name="toy",
        model=ModelRecord(system_name="toy", params={"theta_deg": 1.0}),
        config={"mesh": (1, 3)},
        conventions={"energy_unit": "eV", "density_convention": "projector", "extra_convention": "toy"},
        environment={"host": "test001"},
        validation={"status": "pass"},
        observables={"gap_ev": 0.1},
        files={"state": "hf_state.npz"},
        array_files=(state_path,),
    )

    assert tuple(sorted(path.name for path in paths.values())) == tuple(sorted(required_artifact_files()))
    loaded = load_result(tmp_path)
    assert loaded.model is not None and loaded.model["system_name"] == "toy"
    assert loaded.conventions is not None
    assert loaded.conventions["energy_unit"] == "eV"
    assert loaded.conventions["density_convention"] == "projector"
    assert loaded.conventions["extra_convention"] == "toy"
    assert loaded.environment == {"host": "test001"}
    assert loaded.observables == {"gap_ev": 0.1}

    config_payload = json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert config_payload["mesh"] == [1, 3]
    assert loaded.config == {"mesh": [1, 3]}

    manifest = loaded.manifest
    assert manifest["metadata"]["workflow"] == "toy.workflow"
    assert manifest["metadata"]["system_name"] == "toy"
    assert manifest["files"]["state"] == "hf_state.npz"
    assert manifest["metadata"]["array_summaries"][0]["keys"] == ["density"]
    assert manifest["metadata"]["array_summaries"][0]["arrays"][0]["shape"] == [2, 2, 3]


def test_update_artifact_manifest_preserves_contract_sidecars(tmp_path) -> None:
    write_contract_artifacts(
        tmp_path,
        workflow="toy.workflow",
        system_name="toy",
        model=ModelRecord(system_name="toy"),
        config={"mesh": (1, 1)},
        validation={"status": "pass"},
        observables={"gap": 1.0},
    )
    original_config = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    derived_path = tmp_path / "derived.npz"
    np.savez(derived_path, values=np.arange(3))

    manifest_path = update_artifact_manifest(
        tmp_path,
        files={"derived": "derived.npz"},
        metadata={"derived_workflow": "toy.derived"},
        array_files=(derived_path,),
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == original_config
    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "toy.workflow"
    assert loaded.manifest["metadata"]["derived_workflow"] == "toy.derived"
    assert loaded.manifest["files"]["derived"] == "derived.npz"
    assert loaded.manifest["metadata"]["array_summaries"][-1]["keys"] == ["values"]
    assert loaded.observables == {"gap": 1.0}


def test_rlg_hbn_tdhf_contract_sidecars_are_metadata_only(tmp_path) -> None:
    from mean_field.devtools.run_rlg_hbn_tdhf_q0 import _write_contract_sidecars

    spectrum_path = tmp_path / "tdhf_q0_spectrum.npz"
    np.savez(spectrum_path, energies_mev=np.asarray([1.0, 2.0]), A=np.eye(2), B=np.zeros((2, 2)))
    config = {
        "hf_archive": "/tmp/source_hf.npz",
        "channel": "intraflavor",
        "max_dense_memory_gb": 1.0,
        "max_pairs": 2,
        "summary_converged": True,
        "runtime": {"hostname": "test001"},
    }
    summary = {
        "channel": "intraflavor",
        "channel_counts": {"intraflavor": 2},
        "n_pairs": 2,
        "liouvillian_dim": 4,
        "estimated_dense_memory_gib": 0.01,
        "single_flavor_shortcut_used": False,
        "single_flavor_shortcut_reason": "test",
        "structure": {"ok": True, "A_hermitian": 0.0, "B_symmetric": 0.0, "particle_hole_symmetry": 0.0},
        "spectrum": {
            "selected_count": 2,
            "first_positive_energies_mev": [1.0, 2.0],
            "max_residual": 0.0,
            "pairing_residual": 0.0,
        },
        "hf_summary": {"final_energy_mev": -1.0},
    }

    paths = _write_contract_sidecars(tmp_path, config_payload=config, summary_payload=summary, spectrum_path=spectrum_path)
    assert set(paths) == set(required_artifact_files())

    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "rlg_hbn.tdhf_q0"
    assert loaded.conventions is not None
    assert loaded.conventions["q_sector"] == "q0"
    assert loaded.conventions["density_convention"] == "stored_delta"
    assert loaded.validation is not None and loaded.validation["status"] == "pass"
    assert loaded.observables is not None
    assert loaded.observables["first_positive_energies_mev"] == [1.0, 2.0]
    assert loaded.manifest["files"]["tdhf_spectrum"] == "tdhf_q0_spectrum.npz"
    assert loaded.manifest["metadata"]["array_summaries"][0]["keys"] == ["energies_mev", "A", "B"]


def test_rlg_hbn_paper_hf_contract_sidecars_are_metadata_only(tmp_path) -> None:
    from mean_field.devtools.run_rlg_hbn_paper_hf import _write_contract_sidecars

    config = {
        "paper_target": "fig5",
        "layer_count": 5,
        "theta_deg": 0.77,
        "shell_count": 1,
        "xi_values": (1,),
        "v_values_mev": (40.0,),
        "hbn_moire_scale": 1.0,
        "epsilon_r": 6.25,
        "gate_distance_nm": 10.0,
        "scheme": "average",
        "active_valence_bands": 4,
        "active_conduction_bands": 4,
        "k_mesh_size": 3,
        "interaction_cutoff_q1": 3.0,
        "nu": 1.0,
        "init_modes": ("flavor",),
        "seeds": (1,),
        "candidate_count": 1,
        "max_iter": 1,
        "precision": 1.0e-6,
        "beta": 1.0,
        "cache_policy": "off",
        "screening_solver": "grid",
    }
    run_preflight = {"status": "ok", "run_specs": [{"init_mode": "flavor", "seed": 1}]}
    summary = {
        "output_dir": str(tmp_path),
        "paper_target": "fig5",
        "elapsed_sec": 0.25,
        "panels": [{"panel": "xi1_V040meV", "best": {"final_energy_mev": -1.0}}],
    }

    paths = _write_contract_sidecars(
        tmp_path,
        paper_target="fig5",
        config=config,
        run_preflight=run_preflight,
        cache_dir=tmp_path / "cache",
        runtime_metadata={"hostname": "test001", "dry_run": False},
        workflow_statuses={"preflight": "succeeded", "panel_xi1_V040meV": "succeeded", "summary": "succeeded"},
        workflow_messages={"summary": "paper_hf_summary.json written"},
        summary_payload=summary,
    )

    assert set(paths) == set(required_artifact_files())
    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "rlg_hbn.paper_hf"
    assert loaded.conventions is not None
    assert loaded.conventions["density_convention"] == "stored_delta"
    assert loaded.conventions["form_factor_convention"]
    assert loaded.validation is not None and loaded.validation["status"] == "pass"
    assert loaded.observables is not None
    assert loaded.observables["panels"][0]["panel"] == "xi1_V040meV"
    assert loaded.manifest["files"]["paper_hf_summary"] == "paper_hf_summary.json"


def test_rlg_hbn_parallel_merge_contract_sidecars_are_metadata_only(tmp_path) -> None:
    from mean_field.devtools.merge_rlg_hbn_parallel_hf import _write_contract_sidecars

    config = {
        "paper_target": "fig5",
        "layer_count": 5,
        "theta_deg": 0.77,
        "shell_count": 1,
        "xi_values": (1,),
        "v_values_mev": (40.0,),
        "hbn_moire_scale": 1.0,
        "init_modes": ("flavor",),
        "seeds": (1,),
    }
    selected = [
        {
            "panel": "xi1_V040meV",
            "selected_from": "/tmp/task/xi1_V040meV",
            "candidate_count": 1,
            "selected_final_energy_mev": -10.0,
            "selected_init_mode": "flavor",
            "selected_seed": 1,
        }
    ]

    paths = _write_contract_sidecars(
        tmp_path,
        paper_target="fig5",
        merged_config=config,
        selected_rows=selected,
        ignored_panel_dirs=[],
        tasks_root=tmp_path / "tasks",
    )

    assert set(paths) == set(required_artifact_files())
    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "rlg_hbn.parallel_hf_merge"
    assert loaded.validation is not None and loaded.validation["status"] == "pass"
    assert loaded.observables is not None
    assert loaded.observables["selected"][0]["panel"] == "xi1_V040meV"
    assert loaded.manifest["files"]["parallel_selection_summary"] == "parallel_selection_summary.json"


def test_rlg_hbn_hf_archive_records_density_convention_metadata(tmp_path) -> None:
    from types import SimpleNamespace

    from mean_field.core.hf import summarize_hf_state_archive
    from mean_field.devtools.run_rlg_hbn_paper_hf import _save_state_archive

    state = SimpleNamespace(
        density=np.zeros((2, 2, 1), dtype=np.complex128),
        hamiltonian=np.zeros((2, 2, 1), dtype=np.complex128),
        h0=np.zeros((2, 2, 1), dtype=np.complex128),
        energies=np.zeros((2, 1), dtype=float),
        reference_density=np.eye(2, dtype=np.complex128)[:, :, None] * 0.5,
        nu=1.0,
        active_valence_bands=1,
        scheme="average",
        n_spin=1,
        n_eta=1,
        n_band=2,
        occupation_counts=None,
        mu=0.0,
    )
    basis_data = SimpleNamespace(
        kvec=np.asarray([0.0 + 0.0j]),
        k_grid_frac=np.zeros((1, 2), dtype=float),
        band_energies=np.zeros((2, 1), dtype=float),
        active_band_indices=np.asarray([0, 1], dtype=int),
        flat_band_indices=np.asarray([0, 1], dtype=int),
    )
    archive = tmp_path / "hf_run_state.npz"
    _save_state_archive(archive, SimpleNamespace(state=state, basis_data=basis_data), {"energy_mev": [], "err": [], "oda": []})

    with np.load(archive, allow_pickle=False) as payload:
        assert payload["density_convention"].item() == "stored_delta"
        assert payload["density_axis_order"].item() == "abk"
        assert payload["reference_density_convention"].item() == "average"
        assert payload["basis_periodic_gauge"].item()
        assert payload["form_factor_convention"].item()

    summary = summarize_hf_state_archive(archive)
    assert summary.metadata["density_convention"] == "stored_delta"
    assert summary.metadata["density_axis_order"] == "abk"
    assert summary.metadata["reference_density_convention"] == "average"
    assert summary.metadata["form_factor_convention"]


def test_rlg_hbn_band_plot_updates_manifest_without_overwriting_contract(tmp_path) -> None:
    from mean_field.devtools.plot_rlg_hbn_paper_hf_bands import _update_band_plot_manifest

    write_contract_artifacts(
        tmp_path,
        workflow="rlg_hbn.paper_hf",
        system_name="rlg_hbn",
        model=ModelRecord(system_name="rlg_hbn"),
        config={"paper_target": "fig5"},
        observables={"hf_energy_mev": -1.0},
    )
    original_observables = (tmp_path / "observables.json").read_text(encoding="utf-8")

    _update_band_plot_manifest(
        tmp_path,
        paper_target="fig5",
        panel_names=["xi1_V040meV"],
        status="complete",
    )

    assert (tmp_path / "observables.json").read_text(encoding="utf-8") == original_observables
    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "rlg_hbn.paper_hf"
    assert loaded.manifest["metadata"]["band_plot"]["workflow"] == "rlg_hbn.paper_hf_bands"
    assert loaded.manifest["metadata"]["band_plot"]["status"] == "complete"
    assert loaded.manifest["files"]["hf_band_plot_summary"] == "hf_band_plot_summary.json"
    assert loaded.observables == {"hf_energy_mev": -1.0}
