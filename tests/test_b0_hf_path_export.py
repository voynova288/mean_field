from __future__ import annotations

import numpy as np

from mean_field import load_b0_suite
from mean_field.benchmarks import HFPathReference
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    build_b0_uniform_lattice,
    build_fig6_kpath,
    compare_hf_path_to_reference,
    evaluate_restricted_hf_path,
    run_restricted_hf_from_bm_solution,
    solve_bm_model,
)


def test_benchmark_case_reference_loaders_expose_hf_path_metadata() -> None:
    case = load_b0_suite().get("theta_120_nu_-1_ivc_ground")
    reference = case.load_reference_path()
    parity = case.load_parity_summary()
    runtime = case.load_runtime_summary()

    assert len(reference.band_labels) == 8
    assert reference.energies.shape[1] == 8
    assert len(reference.kdist) == reference.energies.shape[0]
    assert parity.energy_sorting == "ascending_per_k"
    assert float(runtime.entries["theta_deg"]) == case.theta_deg
    assert runtime.entries["init_mode"] == case.init_mode


def test_restricted_hf_path_export_matches_noninteracting_lg1_limit() -> None:
    params = TBGParameters.from_degrees(1.2)
    grid = build_b0_uniform_lattice(params, lk=1)
    grid_solution = solve_bm_model(params, grid.kvec, lg=1, sigma_rotation=True)
    hf_run = run_restricted_hf_from_bm_solution(
        grid_solution,
        nu=-2.0,
        init_mode="bm",
        beta=0.75,
        max_iter=3,
        overlap_lg=1,
        precision=1e-8,
    )

    path_result = evaluate_restricted_hf_path(
        hf_run,
        grid_solution,
        points_per_segment=2,
        lg=1,
        init_mode="bm",
        include_interaction=False,
    )

    path = build_fig6_kpath(params, 2)
    path_solution = solve_bm_model(params, path.kvec, lg=1, sigma_rotation=True)
    expected_energies = np.sort(path_solution.flattened_energies(), axis=0).T

    assert np.allclose(path_result.path.kvec, path.kvec)
    assert np.allclose(path_result.path.kdist, path.kdist)
    assert np.allclose(path_result.band_data.energies.T, expected_energies)
    assert np.isclose(path_result.beta, 0.75)
    assert path_result.overlap_lg == 1
    assert not path_result.finite_zero_limit
    assert not path_result.include_interaction

    reference = HFPathReference(
        band_labels=path_result.band_data.band_labels,
        kdist=tuple(float(value) for value in path_result.path.kdist),
        energies=path_result.band_data.energies.T.copy(),
    )
    parity = compare_hf_path_to_reference(reference, path_result)
    assert np.isclose(parity.kdist_max_abs_diff, 0.0)
    assert np.isclose(parity.max_abs_band_diff_mev, 0.0)
