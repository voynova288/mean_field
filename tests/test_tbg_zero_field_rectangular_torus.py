from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.model import (
    TBGZeroFieldBMModel,
    build_tbg_zero_field_half_open_torus_mesh,
    solve_bm_model_on_torus,
)


def test_rectangular_torus_uses_first_coordinate_fastest_order() -> None:
    params = TBGParameters.from_degrees(1.05)
    mesh = build_tbg_zero_field_half_open_torus_mesh(params, (2, 3))
    repeated = build_tbg_zero_field_half_open_torus_mesh(params, (2, 3))

    expected_frac = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0 / 3.0],
            [0.5, 1.0 / 3.0],
            [0.0, 2.0 / 3.0],
            [0.5, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    expected_kvec = expected_frac[:, 0] * params.g1 + expected_frac[:, 1] * params.g2

    assert mesh.mesh_size == (2, 3)
    assert mesh.mesh_shape == (2, 3)
    assert mesh.nk == 6
    np.testing.assert_array_equal(mesh.k_grid_frac, expected_frac)
    np.testing.assert_array_equal(mesh.kvec, expected_kvec)
    for j in range(3):
        for i in range(2):
            flat_index = i + 2 * j
            np.testing.assert_array_equal(
                mesh.k_grid_frac[flat_index],
                np.asarray([i / 2.0, j / 3.0]),
            )
    assert mesh.to_metadata()["mesh_shape"] == [2, 3]
    assert mesh.to_metadata()["point_count"] == 6
    assert mesh.to_metadata()["index_order"] == "F"
    assert mesh.fingerprint == repeated.fingerprint
    assert not mesh.k_grid_frac.flags.writeable
    assert not mesh.kvec.flags.writeable


def test_scalar_square_torus_remains_byte_identical_and_fingerprint_pinned() -> None:
    params = TBGParameters.from_degrees(1.05)
    scalar = build_tbg_zero_field_half_open_torus_mesh(params, 2)
    tuple_square = build_tbg_zero_field_half_open_torus_mesh(params, (2, 2))

    assert scalar.mesh_size == 2
    assert scalar.mesh_shape == (2, 2)
    assert scalar.k_grid_frac.tobytes(order="C") == tuple_square.k_grid_frac.tobytes(order="C")
    assert scalar.kvec.tobytes(order="C") == tuple_square.kvec.tobytes(order="C")
    assert scalar.to_metadata() == tuple_square.to_metadata()
    assert scalar.fractional_coordinates_sha256 == (
        "fb8ee2b249d3bfa8644774a7f33062c1c8d39b128db65fdc3a35ee1d0e7937d4"
    )
    assert scalar.kvec_sha256 == (
        "9c222aad30c3cdf88239a49d163fa651faba3477cb9886e38477473c06b7538a"
    )
    assert scalar.fingerprint == (
        "7e533f71aa4834f2bbbdc0c5d82fbd092cb4a80c55e403dfd034e2771db06e33"
    )


@pytest.mark.parametrize(
    "mesh_size",
    [
        True,
        np.bool_(False),
        0,
        -1,
        2.0,
        [2, 3],
        (2,),
        (2, 3, 4),
        (True, 3),
        (2, np.bool_(True)),
        (2.0, 3),
        (2, 0),
        (-2, 3),
    ],
)
def test_torus_shape_rejects_non_strict_or_nonpositive_dimensions(
    mesh_size: object,
) -> None:
    params = TBGParameters.from_degrees(1.05)
    with pytest.raises(ValueError, match="mesh_size"):
        build_tbg_zero_field_half_open_torus_mesh(params, mesh_size)  # type: ignore[arg-type]


def test_rectangular_torus_accepts_numpy_integer_dimensions_and_rejects_wrong_order() -> None:
    params = TBGParameters.from_degrees(1.05)
    mesh = build_tbg_zero_field_half_open_torus_mesh(
        params,
        (np.int64(2), np.int32(3)),
    )
    assert mesh.mesh_size == (2, 3)

    wrong_order = np.arange(mesh.nk).reshape(2, 3, order="F").ravel(order="C")
    with pytest.raises(ValueError, match=r"i\+N1\*j"):
        type(mesh)(
            mesh_size=(2, 3),
            g1=mesh.g1,
            g2=mesh.g2,
            k_grid_frac=mesh.k_grid_frac[wrong_order],
            kvec=mesh.kvec[wrong_order],
        )


def test_rectangular_bm_solve_carries_torus_without_changing_g_basis() -> None:
    params = TBGParameters.from_degrees(1.05)
    rectangular = solve_bm_model_on_torus(
        params,
        (2, 3),
        lg=1,
        calculate_chern_operator=False,
    )
    square = solve_bm_model_on_torus(
        params,
        2,
        lg=1,
        calculate_chern_operator=False,
    )

    assert rectangular.torus_mesh is not None
    assert rectangular.torus_mesh.mesh_shape == (2, 3)
    assert rectangular.nk == 6
    assert rectangular.lg == square.lg == 1
    np.testing.assert_array_equal(rectangular.gvec, square.gvec)
    assert rectangular.gvec.shape == (rectangular.lg * rectangular.lg,)
    np.testing.assert_array_equal(rectangular.lattice_kvec, rectangular.torus_mesh.kvec)
    assert rectangular.hamiltonian.shape == (4, 4, 2, 6)
    assert rectangular.spectrum.shape == (2, 2, 6)
    assert rectangular.uk.shape == (4, 2, 2, 6)
    rectangular.validate_source_attestation(require_torus=True)
    assert rectangular.source_attestation is not None
    assert rectangular.source_attestation.nk == 6
    assert (
        rectangular.source_attestation.torus_mesh_fingerprint
        == rectangular.torus_mesh.fingerprint
    )


def test_bands_on_rectangular_grid_preserves_grid_axes_and_torus_point_mapping() -> None:
    params = TBGParameters.from_degrees(1.05)
    model = TBGZeroFieldBMModel.from_config(1.05, params=params, lg=1)
    result = model.bands_on_grid((2, 3), return_eigenvectors=True)
    solution = solve_bm_model_on_torus(
        params,
        (2, 3),
        lg=1,
        calculate_chern_operator=False,
    )
    mesh = solution.torus_mesh
    assert mesh is not None

    assert result.k_grid_frac.shape == (2, 3, 2)
    assert result.kvec.shape == (2, 3)
    assert result.energies.shape == (2, 3, 2)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 3, 4, 2)
    for j in range(3):
        for i in range(2):
            flat_index = i + 2 * j
            np.testing.assert_array_equal(result.k_grid_frac[i, j], mesh.k_grid_frac[flat_index])
            assert result.kvec[i, j] == mesh.kvec[flat_index]
            np.testing.assert_array_equal(
                result.energies[i, j],
                solution.spectrum[:, 0, flat_index],
            )
            np.testing.assert_array_equal(
                result.eigenvectors[i, j],
                solution.uk[:, :, 0, flat_index],
            )

    scalar_square = model.bands_on_grid(2, return_eigenvectors=True)
    tuple_square = model.bands_on_grid((2, 2), return_eigenvectors=True)
    np.testing.assert_array_equal(scalar_square.k_grid_frac, tuple_square.k_grid_frac)
    np.testing.assert_array_equal(scalar_square.kvec, tuple_square.kvec)
    np.testing.assert_array_equal(scalar_square.energies, tuple_square.energies)
    np.testing.assert_array_equal(scalar_square.eigenvectors, tuple_square.eigenvectors)
    assert scalar_square.metadata == tuple_square.metadata
