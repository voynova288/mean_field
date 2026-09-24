from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

import mean_field.systems.tbg.zero_field.artifacts as artifact_impl
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    TBGZeroFieldInteractionSpec,
    solve_bm_model_on_torus,
)
from mean_field.systems.tbg.zero_field.artifacts import (
    load_tbg_zero_field_complete_hf_state_archive_npz,
    write_tbg_zero_field_complete_hf_state_archive_npz,
)
from mean_field.systems.tbg.zero_field._hf_restricted import run_restricted_hf_from_bm_solution


def _write_typed_archive(
    tmp_path: Path,
    *,
    max_iter: int = 0,
) -> tuple[Path, object]:
    params = TBGParameters.from_degrees(1.05)
    solution = solve_bm_model_on_torus(
        params,
        (2, 3),
        lg=7,
        calculate_chern_operator=False,
    )
    hf_run = run_restricted_hf_from_bm_solution(
        solution,
        nu=0.0,
        beta=0.0,
        max_iter=max_iter,
        overlap_lg=7,
        interaction_spec=TBGZeroFieldInteractionSpec(),
    )
    archive_path = write_tbg_zero_field_complete_hf_state_archive_npz(
        tmp_path / f"rectangular_typed_torus_iter{max_iter}_v1.npz",
        hf_run=hf_run,
        grid_solution=solution,
    )
    return archive_path, solution


def test_rectangular_schema_v1_archive_round_trip_keeps_shape_and_npz_keys(
    tmp_path: Path,
) -> None:
    archive_path, solution = _write_typed_archive(tmp_path)

    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == artifact_impl._COMPLETE_HF_STATE_ARCHIVE_ARRAY_KEYS
        assert "mesh_shape" not in archive.files
        mesh_metadata = json.loads(str(archive["mesh_json"].item()))
    assert mesh_metadata == solution.torus_mesh.to_metadata()
    assert mesh_metadata["mesh_shape"] == [2, 3]

    loaded = load_tbg_zero_field_complete_hf_state_archive_npz(archive_path)
    assert loaded.mesh.mesh_size == (2, 3)
    assert loaded.mesh.mesh_shape == (2, 3)
    assert loaded.mesh.to_metadata() == solution.torus_mesh.to_metadata()
    assert loaded.mesh.fingerprint == solution.torus_mesh.fingerprint
    assert loaded.receipt.mesh_fingerprint == solution.torus_mesh.fingerprint
    assert (
        loaded.source_attestation_metadata["torus_mesh_fingerprint"]
        == solution.torus_mesh.fingerprint
    )
    assert loaded.bundle_metadata["mesh_fingerprint"] == solution.torus_mesh.fingerprint
    assert loaded.provenance_metadata["mesh_fingerprint"] == solution.torus_mesh.fingerprint


def test_complete_archive_round_trip_after_one_real_oda_iteration(
    tmp_path: Path,
) -> None:
    archive_path, _solution = _write_typed_archive(tmp_path, max_iter=1)
    loaded = load_tbg_zero_field_complete_hf_state_archive_npz(archive_path)
    assert loaded.iter_energy.shape == (1,)
    assert loaded.iter_err.shape == (1,)
    assert loaded.iter_oda.shape == (1,)
    assert loaded.provenance_metadata["requested_max_iterations"] == 1
    assert loaded.state.diagnostics["requested_max_iterations"] == 1.0


@pytest.mark.parametrize(
    ("array_key", "message"),
    [
        ("state_density", "final-state hash mismatch|density-derived filling"),
        ("bm_uk", "Archived BM uk hash does not match source attestation"),
        ("screened_overlaps", "screened block inventory hash mismatch"),
    ],
)
def test_complete_archive_rejects_tampered_science_arrays(
    tmp_path: Path,
    array_key: str,
    message: str,
) -> None:
    source_path, _solution = _write_typed_archive(tmp_path)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    arrays[array_key].flat[0] += 1.0e-8
    tampered_path = tmp_path / f"tampered_{array_key}.npz"
    np.savez_compressed(tampered_path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_tbg_zero_field_complete_hf_state_archive_npz(tampered_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema": "wrong"}, "provenance schema"),
        ({"schema_version": 2.0}, "schema version"),
        ({"issuer": "manual"}, "issuer"),
        ({"hf_mode": "full"}, "receipt hf_mode"),
        ({"filling": 1.0}, "filling does not equal nu"),
        ({"converged": 1}, "converged must be bool"),
        ({"requested_max_iterations": -1}, "requested_max_iterations"),
        ({"requested_max_iterations": 1}, "does not match state diagnostics"),
        ({"oda_stall_threshold": 2.0e-3}, "does not match state diagnostics"),
        ({"normalized_init_mode": "invented"}, "invalid for hf_mode"),
    ],
)
def test_complete_archive_rejects_self_consistent_but_invalid_provenance(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    source_path, _solution = _write_typed_archive(tmp_path)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    provenance = json.loads(str(arrays["hf_run_provenance_json"].item()))
    provenance.update(updates)
    fingerprint_payload = dict(provenance)
    fingerprint_payload.pop("fingerprint")
    provenance["fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    arrays["hf_run_provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    tampered_path = tmp_path / f"tampered_provenance_{next(iter(updates))}.npz"
    np.savez_compressed(tampered_path, **arrays)
    with pytest.raises(ValueError, match=message):
        load_tbg_zero_field_complete_hf_state_archive_npz(tampered_path)
