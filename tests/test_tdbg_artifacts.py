from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api import load_result, required_artifact_files
from mean_field.systems.tdbg import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedHFResult
from mean_field.systems.tdbg.artifacts import write_tdbg_projected_hf_artifacts
from mean_field.systems.tdbg.projected_hf import TDBGStateLabel


def _fake_tdbg_result() -> TDBGProjectedHFResult:
    nt = 4
    nk = 2
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    density = np.zeros_like(h0)
    hamiltonian = np.zeros_like(h0)
    energies = np.asarray(
        [
            [-1.0, -0.9],
            [-0.4, -0.3],
            [0.2, 0.3],
            [0.8, 0.9],
        ],
        dtype=float,
    )
    state = SimpleNamespace(
        h0=h0,
        density=density,
        hamiltonian=hamiltonian,
        energies=energies,
        diagnostics={"final_raw_norm": 0.0, "hf_energy": -0.25},
    )
    config = TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=1.0,
        mesh_size=1,
        paper_ud_convention="minus_xi_ud_over3",
        interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        max_iter=1,
    )
    labels = (
        TDBGStateLabel(index=0, spin="up", valley=1, band_position=-1, band_index=10),
        TDBGStateLabel(index=1, spin="down", valley=1, band_position=-1, band_index=10),
        TDBGStateLabel(index=2, spin="up", valley=-1, band_position=1, band_index=11),
        TDBGStateLabel(index=3, spin="down", valley=-1, band_position=1, band_index=11),
    )
    data = SimpleNamespace(
        config=config,
        k_grid_frac=np.zeros((1, nk, 2), dtype=float),
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.2j], dtype=np.complex128),
        band_indices=(10, 11),
        labels=labels,
        h0=h0,
        reference_density=np.zeros_like(h0),
        n_occupied_per_k=2,
        moire_area_nm2=100.0,
    )
    return TDBGProjectedHFResult(
        run=SimpleNamespace(state=state, converged=True, exit_reason="converged", iterations=1),
        data=data,  # type: ignore[arg-type]
        init_mode="sp",
        seed=1,
        order_parameters={"classification": "SP_up"},
        energy_components={"total_ev": -0.25},
    )


def test_write_tdbg_projected_hf_artifacts_writes_contract_sidecars(tmp_path) -> None:
    result = _fake_tdbg_result()

    paths = write_tdbg_projected_hf_artifacts(tmp_path, result)

    assert paths["hf_state_npz"] == tmp_path / "hf_state.npz"
    assert {path.name for path in tmp_path.iterdir()} >= set(required_artifact_files()) | {
        "hf_state.npz",
        "projected_hf_summary.json",
        "state_labels.json",
    }
    loaded = load_result(tmp_path)
    assert loaded.manifest["metadata"]["workflow"] == "tdbg.projected_hf"
    assert loaded.manifest["files"]["hf_state"] == "hf_state.npz"
    assert loaded.manifest["metadata"]["array_summaries"][0]["keys"]
    assert loaded.conventions is not None and loaded.conventions["density_convention"] == "projector"
    assert loaded.conventions["energy_unit"] == "eV"
    assert loaded.validation == {"status": "pass", "converged": True, "exit_reason": "converged", "iterations": 1}
    assert loaded.observables is not None and loaded.observables["init_mode"] == "sp"
    assert loaded.observables["grid_band_summary"]["hf_grid_gap_ev"] > 0.0

    with np.load(tmp_path / "hf_state.npz", allow_pickle=False) as archive:
        assert archive["density"].shape == (4, 4, 2)
        assert str(archive["density_convention"].item()) == "projector"

    with pytest.raises(FileExistsError, match="non-empty root"):
        write_tdbg_projected_hf_artifacts(tmp_path, result)
