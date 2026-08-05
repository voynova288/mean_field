from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import mean_field.systems.tbg.zero_field.artifacts as artifact_impl
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    TBGZeroFieldInteractionSpec,
    build_tbg_zero_field_half_open_torus_mesh,
    load_tbg_zero_field_complete_hf_state_archive_npz,
    run_restricted_hf_from_bm_solution,
    solve_bm_model_on_torus,
    write_tbg_zero_field_complete_hf_state_archive_npz,
)


def test_rectangular_schema_v1_archive_round_trip_keeps_shape_and_npz_keys(
    tmp_path: Path,
) -> None:
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
        max_iter=0,
        overlap_lg=7,
        interaction_spec=TBGZeroFieldInteractionSpec(),
    )
    result = SimpleNamespace(grid_solution=solution, hf_run=hf_run)
    archive_path = write_tbg_zero_field_complete_hf_state_archive_npz(
        tmp_path / "rectangular_typed_torus_v1.npz",
        result,
    )

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


def test_artifact_grid_descriptor_preserves_square_lk_and_never_infers_rectangular_lk() -> None:
    params = TBGParameters.from_degrees(1.05)
    square_mesh = build_tbg_zero_field_half_open_torus_mesh(params, 2)
    rectangular_mesh = build_tbg_zero_field_half_open_torus_mesh(params, (2, 3))

    square_result = SimpleNamespace(
        grid_solution=SimpleNamespace(torus_mesh=square_mesh)
    )
    rectangular_result = SimpleNamespace(
        grid_solution=SimpleNamespace(torus_mesh=rectangular_mesh)
    )

    assert artifact_impl._b0_reported_grid_descriptor(square_result) == {"lk": 2}
    rectangular_descriptor = artifact_impl._b0_reported_grid_descriptor(
        rectangular_result
    )
    assert rectangular_descriptor == {"mesh_shape": [2, 3]}
    assert "lk" not in rectangular_descriptor
