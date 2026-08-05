from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import solve_bm_model_band_window_on_torus
from mean_field.systems.tbg.zero_field.model import (
    BMSolution,
    _c2t_operator,
    _c2t_takagi_factor,
    _solve_c2t_real_band_window,
    _sigma_z_operator,
    solve_bm_model,
    solve_bm_model_on_torus,
    tbg_zero_field_bm_generation_fingerprint,
)


@pytest.fixture(scope="module")
def band_window_solutions() -> tuple[BMSolution, BMSolution, BMSolution, BMSolution]:
    params = TBGParameters.from_degrees(1.05)
    solve_kwargs = {
        "lg": 3,
        "calculate_chern_operator": False,
    }
    default = solve_bm_model_on_torus(params, (1, 2), **solve_kwargs)
    direct_default = solve_bm_model(
        params,
        default.lattice_kvec,
        **solve_kwargs,
    )
    explicit_two = solve_bm_model_band_window_on_torus(
        params,
        (1, 2),
        retained_band_count=2,
        **solve_kwargs,
    )
    central_six = solve_bm_model_band_window_on_torus(
        params,
        (1, 2),
        **solve_kwargs,
    )
    return default, direct_default, explicit_two, central_six


def test_default_two_band_solvers_remain_byte_identical_and_fingerprint_pinned(
    band_window_solutions: tuple[BMSolution, BMSolution, BMSolution, BMSolution],
) -> None:
    default, direct_default, explicit_two, _central_six = band_window_solutions

    assert default.nb == direct_default.nb == explicit_two.nb == 2
    for name in ("hamiltonian", "sigma_z", "uk", "spectrum", "gvec"):
        default_array = np.asarray(getattr(default, name))
        assert default_array.tobytes(order="C") == np.asarray(
            getattr(direct_default, name)
        ).tobytes(order="C")
        assert default_array.tobytes(order="C") == np.asarray(
            getattr(explicit_two, name)
        ).tobytes(order="C")

    assert default.generation_fingerprint == (
        "ef7c114fe9bac3998352fcdcbcf828800d51100fed6bc328f0c5e3180c5a8279"
    )
    assert explicit_two.generation_fingerprint == default.generation_fingerprint
    assert default.source_attestation is not None
    assert default.source_attestation.fingerprint == (
        "0644069e7a9f089635d363b48e1a284a14d85acaf4b6bcdb01e20b302fabe930"
    )


def test_six_band_window_is_the_central_symmetric_parent_subset(
    band_window_solutions: tuple[BMSolution, BMSolution, BMSolution, BMSolution],
) -> None:
    default, _direct_default, _explicit_two, central_six = band_window_solutions

    assert central_six.nb == 6
    assert central_six.spectrum.shape == (6, 2, 2)
    assert central_six.uk.shape == (36, 6, 2, 2)
    c2t = _c2t_operator(central_six.lg)
    np.testing.assert_allclose(
        central_six.spectrum[2:4],
        default.spectrum,
        rtol=0.0,
        atol=1.0e-10,
    )

    for valley_index in range(central_six.n_eta):
        for k_index in range(central_six.nk):
            default_frame = default.uk[:, :, valley_index, k_index]
            full_hamiltonian = central_six.hamiltonian[:, :, valley_index, k_index]
            central_frame = central_six.uk[:, 2:4, valley_index, k_index]
            wide_frame = central_six.uk[:, :, valley_index, k_index]
            default_projector = default_frame @ default_frame.conj().T
            central_projector = central_frame @ central_frame.conj().T
            wide_projector = wide_frame @ wide_frame.conj().T
            np.testing.assert_allclose(
                wide_frame.conj().T @ wide_frame,
                np.eye(central_six.nb),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                c2t @ wide_frame.conj(),
                wide_frame,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                full_hamiltonian @ wide_frame,
                wide_frame * central_six.spectrum[:, valley_index, k_index][None, :],
                rtol=0.0,
                atol=1.0e-9,
            )
            np.testing.assert_allclose(
                wide_projector @ wide_projector,
                wide_projector,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                central_projector,
                default_projector,
                rtol=0.0,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                wide_projector @ default_projector,
                default_projector,
                rtol=0.0,
                atol=1.0e-10,
            )


def test_multiband_attestation_and_generation_fingerprint_bind_band_count(
    band_window_solutions: tuple[BMSolution, BMSolution, BMSolution, BMSolution],
) -> None:
    default, _direct_default, _explicit_two, central_six = band_window_solutions
    attestation = central_six.source_attestation
    mesh = central_six.torus_mesh
    assert attestation is not None
    assert mesh is not None
    assert attestation.solver_entrypoint == "solve_bm_model_band_window_on_torus"
    assert attestation.nb == central_six.nb == 6
    central_six.validate_source_attestation(require_torus=True)

    fingerprint_kwargs = {
        "lg": central_six.lg,
        "periodic_g_grid": central_six.periodic_g_grid,
        "sigma_rotation": central_six.sigma_rotation,
        "calculate_chern_operator": central_six.calculate_chern_operator,
        "torus_mesh_fingerprint": mesh.fingerprint,
    }
    default_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        central_six.params,
        **fingerprint_kwargs,
    )
    explicit_two_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        central_six.params,
        retained_band_count=2,
        **fingerprint_kwargs,
    )
    four_band_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        central_six.params,
        retained_band_count=4,
        **fingerprint_kwargs,
    )
    six_band_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        central_six.params,
        retained_band_count=6,
        **fingerprint_kwargs,
    )
    assert default_fingerprint == explicit_two_fingerprint == default.generation_fingerprint
    assert central_six.generation_fingerprint == six_band_fingerprint
    assert six_band_fingerprint == (
        "e5743a325b308557a9016974320ad3522bb635e2d30c3ae2bdadb4e7ff2a5a70"
    )
    assert len({default_fingerprint, four_band_fingerprint, six_band_fingerprint}) == 3

    original = float(central_six.spectrum[0, 0, 0])
    original_attestation_fingerprint = attestation.fingerprint
    central_six.spectrum[0, 0, 0] = original + 1.0
    try:
        with pytest.raises(ValueError, match="spectrum.sha256"):
            central_six.validate_source_attestation(require_torus=True)
    finally:
        central_six.spectrum[0, 0, 0] = original
    assert attestation.fingerprint == original_attestation_fingerprint
    central_six.validate_source_attestation(require_torus=True)


def test_nonabelian_c2t_real_solver_handles_exact_internal_degeneracy() -> None:
    c2t = _c2t_operator(1)
    factor = _c2t_takagi_factor(1)
    real_parent = np.diag([0.0, 0.0, 1.0, 2.0])
    hamiltonian = factor @ real_parent @ factor.conj().T
    eigenvalues, eigenvectors = _solve_c2t_real_band_window(
        hamiltonian,
        c2t=c2t,
        c2t_factor=factor,
        start=0,
        stop=1,
    )
    np.testing.assert_array_equal(eigenvalues, np.zeros(2))
    np.testing.assert_allclose(
        eigenvectors.conj().T @ eigenvectors,
        np.eye(2),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        c2t @ eigenvectors.conj(),
        eigenvectors,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.linalg.matrix_rank(eigenvectors, tol=1.0e-12) == 2


def test_multiband_window_rejects_boundary_degeneracy() -> None:
    c2t = _c2t_operator(1)
    factor = _c2t_takagi_factor(1)
    real_parent = np.diag([0.0, 1.0, 1.0, 2.0])
    hamiltonian = factor @ real_parent @ factor.conj().T
    with pytest.raises(RuntimeError, match="upper boundary degeneracy"):
        _solve_c2t_real_band_window(
            hamiltonian,
            c2t=c2t,
            c2t_factor=factor,
            start=0,
            stop=1,
        )


def test_multiband_chern_operator_uses_full_retained_window() -> None:
    params = TBGParameters.from_degrees(1.05)
    solution = solve_bm_model_band_window_on_torus(
        params,
        (1, 2),
        retained_band_count=6,
        lg=3,
        calculate_chern_operator=True,
    )
    assert solution.sigma_z.shape == (24, 24, 2)
    parent_sigma_z = _sigma_z_operator(solution.lg)
    expected = np.zeros_like(solution.sigma_z)
    for k_index in range(solution.nk):
        for valley_index, valley_sign in enumerate((1, -1)):
            frame = solution.uk[:, :, valley_index, k_index]
            block = valley_sign * frame.conj().T @ parent_sigma_z @ frame
            for spin_index in range(solution.n_spin):
                first = solution.n_spin * valley_index + spin_index
                indices = np.arange(
                    first,
                    solution.n_spin * solution.n_eta * solution.nb,
                    solution.n_spin * solution.n_eta,
                )
                expected[np.ix_(indices, indices, [k_index])] = block[:, :, None]
    np.testing.assert_allclose(
        solution.sigma_z,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    solution.validate_source_attestation(require_torus=True)


@pytest.mark.parametrize(
    "retained_band_count",
    [True, np.bool_(False), 0, -2, 1, 3, 2.0, "6", 38],
)
def test_multiband_window_rejects_non_strict_invalid_counts(
    retained_band_count: object,
) -> None:
    params = TBGParameters.from_degrees(1.05)
    with pytest.raises((TypeError, ValueError), match="retained_band_count"):
        solve_bm_model_band_window_on_torus(
            params,
            (1, 2),
            retained_band_count=retained_band_count,  # type: ignore[arg-type]
            lg=3,
            calculate_chern_operator=False,
        )
