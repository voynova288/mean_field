from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.xue2018_hf import (
    build_xue2018_hf_state,
    run_xue2018_hf,
    xue2018_global_neutral_projector,
    xue2018_seed_hamiltonian,
    xue2018_square_mesh,
)
from mean_field.systems.inas_gasb.zeng2022 import ZengSlabBasis


def test_xue2018_square_mesh_has_exact_gamma_and_physical_weights() -> None:
    mesh = xue2018_square_mesh(kmax_ab_inv=1.5, points_per_axis=5)
    assert mesh.shape == (5, 5)
    assert np.min(np.linalg.norm(mesh.points_ab_inv, axis=1)) == pytest.approx(0.0)
    assert np.sum(mesh.weights_ab2) == pytest.approx(9.0 / (2.0 * np.pi) ** 2)
    with pytest.raises(ValueError, match="odd"):
        xue2018_square_mesh(kmax_ab_inv=1.5, points_per_axis=6)
    nodes = xue2018_square_mesh(
        kmax_ab_inv=1.5,
        points_per_axis=5,
        policy="inclusive_nodes_uniform_weight_diagnostic",
    )
    assert np.min(nodes.points_ab_inv[:, 0]) == pytest.approx(-1.5)
    assert np.max(nodes.points_ab_inv[:, 0]) == pytest.approx(1.5)
    assert nodes.cell_widths_ab_inv == pytest.approx((0.75, 0.75))


def test_xue2018_global_neutral_projector_fills_two_states_per_k_on_average() -> None:
    state = build_xue2018_hf_state(
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        kmax_ab_inv=1.0,
        points_per_axis=3,
    )
    update = xue2018_global_neutral_projector(
        state.h0,
        reference_density=state.reference_density,
    )
    projector = np.asarray(update.observables["raw_projector"])
    assert np.trace(projector, axis1=0, axis2=1).real.sum() == pytest.approx(2 * state.nk)
    assert np.max(np.abs(projector - np.swapaxes(projector.conj(), 0, 1))) < 1e-14
    for ik in range(state.nk):
        assert np.max(np.abs(projector[:, :, ik] @ projector[:, :, ik] - projector[:, :, ik])) < 2e-14


def test_xue2018_nematic_seed_signs_match_paper_trsb_and_trs_channels() -> None:
    basis = ZengSlabBasis((0,))
    cup = basis.index("c", "up", 0)
    vup = basis.index("v", "up", 0)
    cdown = basis.index("c", "down", 0)
    vdown = basis.index("v", "down", 0)
    trsb = xue2018_seed_hamiltonian(
        basis, 2, mode="trsb_nematic", amplitude_ry=0.3, seed=0
    )
    trs = xue2018_seed_hamiltonian(
        basis, 2, mode="trs_nematic", amplitude_ry=0.3, seed=0
    )
    assert trsb[cup, vdown, 0] == pytest.approx(0.3)
    assert trsb[vup, cdown, 0] == pytest.approx(0.3)
    assert trs[cup, vdown, 0] == pytest.approx(0.3)
    assert trs[vup, cdown, 0] == pytest.approx(-0.3)
    assert np.max(np.abs(trsb - np.swapaxes(trsb.conj(), 0, 1))) == 0.0
    assert np.max(np.abs(trs - np.swapaxes(trs.conj(), 0, 1))) == 0.0


def test_xue2018_omitted_self_cell_is_explicit_diagnostic_lane() -> None:
    state = build_xue2018_hf_state(
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        kmax_ab_inv=1.0,
        points_per_axis=3,
        self_cell_policy="omitted_diagnostic",
    )
    assert state.self_cell_policy == "omitted_diagnostic"
    assert np.all(np.diag(state.q0_kernel.intra_ry_ab2) == 0.0)
    assert np.all(np.diag(state.q0_kernel.inter_ry_ab2) == 0.0)
    with pytest.raises(ValueError, match="self_cell_policy"):
        build_xue2018_hf_state(
            eg_ry=-0.5,
            hybridization_ab_ry=0.2,
            kmax_ab_inv=1.0,
            points_per_axis=3,
            self_cell_policy="unknown",  # type: ignore[arg-type]
        )


def test_xue2018_toeplitz_fft_backend_matches_dense_scf_steps() -> None:
    kwargs = dict(
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        kmax_ab_inv=1.0,
        points_per_axis=3,
        precision=1e-20,
    )
    dense = build_xue2018_hf_state(**kwargs, q0_kernel_backend="dense")
    fft = build_xue2018_hf_state(**kwargs, q0_kernel_backend="toeplitz_fft")
    dense_result = run_xue2018_hf(
        dense, init_mode="trsb_nematic", max_iter=2, max_oda_lambda=0.5
    )
    fft_result = run_xue2018_hf(
        fft, init_mode="trsb_nematic", max_iter=2, max_oda_lambda=0.5
    )
    assert np.max(np.abs(dense_result.run.state.density - fft_result.run.state.density)) < 2e-14
    assert np.max(np.abs(dense_result.interaction_h - fft_result.interaction_h)) < 2e-14
    assert np.max(np.abs(dense_result.run.iter_energy - fft_result.run.iter_energy)) < 2e-14


def test_xue2018_concave_oda_chord_advances_instead_of_false_stall() -> None:
    state = build_xue2018_hf_state(
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        kmax_ab_inv=1.0,
        points_per_axis=3,
        precision=1e-20,
    )
    result = run_xue2018_hf(
        state,
        init_mode="trsb_nematic",
        seed_amplitude_ry=0.2,
        max_iter=2,
        max_oda_lambda=0.5,
    )
    assert result.run.exit_reason == "max_iter"
    assert result.run.iterations == 2
    assert np.all(result.run.iter_oda > 0.0)
