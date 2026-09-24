from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api.hf import HFConfig, run_hf
from mean_field.core.lattice import KPath
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    TBGZeroFieldBMModel,
    TBGZeroFieldInteractionSpec,
    TBGZeroFieldRunHFConfig,
    build_tbg_zero_field_half_open_torus_mesh,
    solve_bm_model_on_torus,
)
from mean_field.systems.tbg.zero_field.hf_runners import (
    select_restricted_hf_saved_grid_path,
)
from mean_field.systems.tbg.zero_field.hf_contracts import (
    tbg_zero_field_hf_run_to_hf_result,
)
import mean_field.systems.tbg.zero_field.hf_contracts as hf_contracts_module
import mean_field.systems.tbg.zero_field.hf_runners as hf_runners_module


@pytest.fixture(scope="module")
def rectangular_typed_hf_result():
    params = TBGParameters.from_degrees(1.05)
    grid_solution = solve_bm_model_on_torus(
        params,
        (1, 2),
        lg=7,
        calculate_chern_operator=True,
    )
    interaction_spec = TBGZeroFieldInteractionSpec()
    config = HFConfig(
        filling=0.0,
        mesh=(1, 2),
        max_iter=0,
        precision=1.0e-6,
        density_convention="stored_delta",
        interaction_scheme="average",
        epsilon_r=interaction_spec.epsr,
        dsc_nm=interaction_spec.dsc_nm,
        coulomb_kernel="2d_gate",
        seeds=("5",),
        metadata={"grid_mesh_shape": [1, 2]},
    )
    tbg_config = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=0.0,
        init_mode="bm",
        seed=5,
        max_iter=0,
        overlap_lg=7,
        precision=1.0e-6,
        interaction_spec=interaction_spec,
    )
    model = TBGZeroFieldBMModel.from_config(
        1.05,
        lg=grid_solution.lg,
        params=params,
    )
    result = run_hf(
        model,
        config,
        tbg_zero_field_config=tbg_config,
    )
    assert result is not None
    return result, grid_solution, tbg_config


def test_rectangular_typed_hf_contracts_carry_exact_shape_without_scalar_alias(
    rectangular_typed_hf_result,
) -> None:
    result, grid_solution, _tbg_config = rectangular_typed_hf_result

    assert result.config.mesh == (1, 2)
    assert result.config.metadata["grid_mesh_shape"] == [1, 2]
    assert "grid_mesh_size" not in result.config.metadata
    assert result.observables["grid_mesh_shape"] == [1, 2]
    assert "grid_mesh_size" not in result.observables

    final_state = result.canonical_run_result.final_state
    assert final_state.observables["grid_mesh_shape"] == [1, 2]
    assert "grid_mesh_size" not in final_state.observables
    assert final_state.basis.metadata["grid_mesh_shape"] == [1, 2]

    default_view = tbg_zero_field_hf_run_to_hf_result(
        result.state,
        grid_solution=grid_solution,
    )
    assert default_view.config.mesh == (1, 2)
    assert default_view.config.metadata["grid_mesh_shape"] == [1, 2]
    assert "grid_mesh_size" not in default_view.config.metadata


def test_rectangular_public_config_rejects_square_shape_claims_and_two_band_overrides(
    rectangular_typed_hf_result,
) -> None:
    result, _grid_solution, tbg_config = rectangular_typed_hf_result

    misleading_scalar = replace(
        result.config,
        metadata={"grid_mesh_shape": [1, 2], "grid_mesh_size": 1},
    )
    with pytest.raises(ValueError, match="grid_mesh_size"):
        hf_contracts_module._validate_tbg_zero_field_public_hf_config(
            misleading_scalar,
            tbg_config,
        )

    wrong_shape = replace(result.config, mesh=(1, 1))
    with pytest.raises(ValueError, match="half-open torus mesh \\(1, 2\\)"):
        hf_contracts_module._validate_tbg_zero_field_public_hf_config(
            wrong_shape,
            tbg_config,
        )

    active_window_override = replace(result.config, active_window=(1, 1))
    with pytest.raises(NotImplementedError, match="active two-band BM window"):
        hf_contracts_module._validate_tbg_zero_field_public_hf_config(
            active_window_override,
            tbg_config,
        )


def test_square_scalar_metadata_and_legacy_endpoint_lk_are_preserved() -> None:
    params = TBGParameters.from_degrees(1.05)
    square_mesh = build_tbg_zero_field_half_open_torus_mesh(params, 18)
    rectangular_mesh = build_tbg_zero_field_half_open_torus_mesh(params, (18, 12))
    square_grid = SimpleNamespace(
        torus_mesh=square_mesh,
        lattice_kvec=square_mesh.kvec,
        nk=square_mesh.nk,
    )
    rectangular_grid = SimpleNamespace(
        torus_mesh=rectangular_mesh,
        lattice_kvec=rectangular_mesh.kvec,
        nk=rectangular_mesh.nk,
    )
    legacy_endpoint_grid = SimpleNamespace(torus_mesh=None, nk=19 * 19)

    assert hf_contracts_module._tbg_zero_field_grid_metadata(square_grid) == {
        "grid_mesh_shape": [18, 18],
        "grid_mesh_size": 18,
    }
    assert hf_contracts_module._tbg_zero_field_grid_metadata(rectangular_grid) == {
        "grid_mesh_shape": [18, 12],
    }
    assert hf_runners_module._reported_grid_shape(square_grid) == (18, 18)
    assert hf_runners_module._reported_grid_size(square_grid) == 18
    assert hf_runners_module._reported_grid_shape(rectangular_grid) == (18, 12)
    assert hf_runners_module._reported_grid_size(rectangular_grid) is None
    assert hf_runners_module._reported_grid_shape(legacy_endpoint_grid) is None
    assert hf_runners_module._reported_grid_size(legacy_endpoint_grid) == 18


def test_rectangular_exact_saved_grid_path_reports_shape_not_lk(
    rectangular_typed_hf_result,
) -> None:
    result, grid_solution, _tbg_config = rectangular_typed_hf_result
    path_kvec = np.asarray(grid_solution.lattice_kvec[:2], dtype=np.complex128)
    path = KPath(
        kvec=path_kvec,
        kdist=np.asarray([0.0, abs(path_kvec[1] - path_kvec[0])], dtype=float),
        labels=("G", "P"),
        node_indices=(1, 2),
    )
    scf_result = select_restricted_hf_saved_grid_path(
        result.state,
        grid_solution,
        path=path,
    )

    assert scf_result.mesh_shape == (1, 2)
    assert scf_result.lk is None
    np.testing.assert_array_equal(scf_result.grid_indices, np.asarray([0, 1]))
    np.testing.assert_allclose(
        scf_result.band_data.energies,
        np.linalg.eigvalsh(
            np.moveaxis(result.state.state.hamiltonian[:, :, :2], -1, 0)
        ).T,
        rtol=0.0,
        atol=5.0e-14,
    )


def test_saved_grid_path_never_substitutes_nearest_grid_point(
    rectangular_typed_hf_result,
) -> None:
    result, grid_solution, _tbg_config = rectangular_typed_hf_result
    shifted = np.asarray(grid_solution.lattice_kvec[:2]) + (1.0e-6 + 2.0e-6j)
    path = KPath(
        kvec=shifted,
        kdist=np.asarray([0.0, abs(shifted[1] - shifted[0])], dtype=float),
        labels=("A", "B"),
        node_indices=(1, 2),
    )
    selected = select_restricted_hf_saved_grid_path(
        result.state,
        grid_solution,
        path=path,
    )
    assert selected.grid_indices.size == 0
    assert selected.path_kvec.size == 0
    assert selected.band_data.energies.shape[1] == 0

    injected_scale = KPath(
        kvec=np.asarray([shifted[0], 1.0e12 + 0.0j]),
        kdist=np.asarray([0.0, 1.0]),
        labels=("A", "X"),
        node_indices=(1, 2),
    )
    scale_selected = select_restricted_hf_saved_grid_path(
        result.state,
        grid_solution,
        path=injected_scale,
    )
    assert scale_selected.grid_indices.size == 0

    for nonfinite in (np.nan + 0.0j, np.inf + 0.0j):
        invalid_path = KPath(
            kvec=np.asarray([grid_solution.lattice_kvec[0], nonfinite]),
            kdist=np.asarray([0.0, 1.0]),
            labels=("A", "X"),
            node_indices=(1, 2),
        )
        with pytest.raises(ValueError, match="coordinates must be finite"):
            select_restricted_hf_saved_grid_path(
                result.state,
                grid_solution,
                path=invalid_path,
            )

    with pytest.raises(TypeError, match="path_tolerance"):
        select_restricted_hf_saved_grid_path(
            result.state,
            grid_solution,
            path=path,
            path_tolerance=1.0,  # type: ignore[call-arg]
        )


def test_saved_grid_path_rejects_mismatched_grid_lineage(
    rectangular_typed_hf_result,
) -> None:
    result, grid_solution, _tbg_config = rectangular_typed_hf_result
    wrong_grid = solve_bm_model_on_torus(
        grid_solution.params,
        (2, 1),
        lg=grid_solution.lg,
        calculate_chern_operator=True,
    )
    path = KPath(
        kvec=np.asarray(grid_solution.lattice_kvec[:2]),
        kdist=np.asarray([0.0, abs(grid_solution.lattice_kvec[1] - grid_solution.lattice_kvec[0])]),
        labels=("A", "B"),
        node_indices=(1, 2),
    )
    with pytest.raises(ValueError, match="source|mesh|fingerprint|lattice"):
        select_restricted_hf_saved_grid_path(
            result.state,
            wrong_grid,
            path=path,
        )
