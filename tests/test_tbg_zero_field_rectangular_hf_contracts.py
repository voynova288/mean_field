from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api.hf import HFConfig, run_hf
from mean_field.core.hf import FlavorBandData
from mean_field.core.lattice import KPath
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    HFPathResult,
    TBGZeroFieldBMModel,
    TBGZeroFieldInteractionSpec,
    TBGZeroFieldRunHFConfig,
    build_restricted_hf_scf_path_plot_result,
    build_tbg_zero_field_half_open_torus_mesh,
    solve_bm_model_on_torus,
    tbg_zero_field_hf_run_to_hf_result,
    write_hf_path_summary,
)
from mean_field.systems.tbg.zero_field import hf_contracts as hf_contracts_module
from mean_field.systems.tbg.zero_field import hf_runners as hf_runners_module


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


def test_rectangular_scf_result_and_summary_report_shape_not_lk(
    rectangular_typed_hf_result,
    tmp_path,
) -> None:
    result, grid_solution, _tbg_config = rectangular_typed_hf_result
    path_kvec = np.asarray(grid_solution.lattice_kvec[:2], dtype=np.complex128)
    path = KPath(
        kvec=path_kvec,
        kdist=np.asarray([0.0, abs(path_kvec[1] - path_kvec[0])], dtype=float),
        labels=("G", "P"),
        node_indices=(1, 2),
    )
    scf_result = build_restricted_hf_scf_path_plot_result(
        result.state,
        grid_solution,
        path=path,
    )

    assert scf_result.mesh_shape == (1, 2)
    assert scf_result.lk is None

    path_result = HFPathResult(
        params=grid_solution.params,
        path=path,
        hamiltonian=np.asarray(result.state.state.hamiltonian[:, :, :2]),
        band_data=FlavorBandData(
            band_labels=scf_result.band_data.band_labels,
            energies=scf_result.band_data.energies,
            mean_weights=scf_result.band_data.mean_weights,
        ),
        mu=result.state.state.mu,
        nu=result.state.state.nu,
        lk=None,
        lg=grid_solution.lg,
        points_per_segment=1,
        init_mode=result.state.init_mode,
        normalized_init_mode=result.state.init_mode,
        seed=result.state.seed,
        exit_reason=result.state.exit_reason,
        mesh_shape=(1, 2),
    )
    summary_path = tmp_path / "rectangular_hf_path_summary.txt"
    write_hf_path_summary(summary_path, path_result)
    summary = summary_path.read_text(encoding="utf-8")

    assert "mesh_shape=1x2\n" in summary
    assert "lk=\n" in summary
    assert "lk=1\n" not in summary
