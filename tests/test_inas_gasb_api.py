from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import compute_bands, make_model
import mean_field.systems.inas_gasb as inas_gasb_api
from mean_field.systems.inas_gasb import (
    InAsGaSbBHZModel,
    InAsGaSbHFConfig,
    build_inas_gasb_model,
    compute_kane4_bands,
    run_inas_gasb_hf,
)
from mean_field.systems.inas_gasb.kane4_bundle import Kane4Bundle
from mean_field.systems.inas_gasb.xue2018_hf import (
    Xue2018HFResult,
    build_xue2018_hf_state,
    run_xue2018_hf,
)
from mean_field.systems.inas_gasb.zeng2022 import (
    Zeng2022Parameters,
    build_zeng2022_folded_h0,
)


EXPECTED_PACKAGE_EXPORTS = {
    "InAsGaSbBHZModel",
    "InAsGaSbHFConfig",
    "Kane4BandResult",
    "build_inas_gasb_model",
    "compute_kane4_bands",
    "run_inas_gasb_hf",
}


def test_package_facade_is_exactly_the_maintained_api() -> None:
    assert set(inas_gasb_api.__all__) == EXPECTED_PACKAGE_EXPORTS
    assert len(inas_gasb_api.__all__) == len(EXPECTED_PACKAGE_EXPORTS)
    assert all(hasattr(inas_gasb_api, name) for name in EXPECTED_PACKAGE_EXPORTS)


def _model() -> InAsGaSbBHZModel:
    return build_inas_gasb_model(
        variant="xue2018_q0_bhz",
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        path_extent_ab_inv=0.3,
    )


def test_public_model_factory_requires_explicit_variant_and_parameters() -> None:
    with pytest.raises(TypeError, match="explicit variant"):
        make_model("inas_gasb")
    with pytest.raises(TypeError, match="requires eg_ry"):
        make_model("inas_gasb", variant="xue2018_q0_bhz")
    model = make_model(
        "inas_gasb",
        variant="xue2018_q0_bhz",
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        path_extent_ab_inv=0.3,
    )
    assert isinstance(model, InAsGaSbBHZModel)
    assert model.variant == "xue2018_q0_bhz"


def test_public_model_factory_rejects_silently_ignored_physical_parameters() -> None:
    params = Zeng2022Parameters(
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
        q_ab_inv=0.1,
    )
    with pytest.raises(TypeError, match="params or scalar"):
        build_inas_gasb_model(
            variant="zeng2022_folded_bhz",
            params=params,
            d_over_ab=0.9,
        )
    with pytest.raises(TypeError, match="source policy"):
        build_inas_gasb_model(
            variant="xue2018_q0_bhz",
            eg_ry=-0.5,
            hybridization_ab_ry=0.2,
            d_over_ab=0.3,
        )
    with pytest.raises(TypeError, match="source policy"):
        build_inas_gasb_model(
            variant="xue2018_q0_bhz",
            eg_ry=-0.5,
            hybridization_ab_ry=0.2,
            mass_e_over_reduced=2.0,
        )


def test_public_model_delegates_exact_xue_zeng_hamiltonian() -> None:
    model = _model()
    k_value = 0.13 - 0.07j
    expected = build_zeng2022_folded_h0(
        np.asarray([[k_value.real, k_value.imag]]),
        model.basis,
        model.params,
    )[:, :, 0]
    actual = model.build_hamiltonian(k_value)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=1.0e-14)


def test_public_band_api_supports_explicit_continuum_path_and_grid() -> None:
    model = _model()
    path = compute_bands(
        model,
        points_per_segment=2,
        n_bands=4,
        return_eigenvectors=True,
    )
    assert path.source == "path"
    assert path.energies.shape == (7, 4)
    assert path.eigenvectors is not None
    assert path.eigenvectors.shape == (7, 4, 4)
    assert path.basis_metadata["periodic_brillouin_zone"] is False
    assert path.basis_metadata["energy_unit"] == "Ry*"

    grid = compute_bands(model, grid_mesh=3, n_bands=2)
    assert grid.source == "grid"
    assert grid.energies.shape == (3, 3, 2)
    assert grid.eigenvectors is None
    assert grid.basis_metadata["momentum_unit"] == "a_B*^-1"


def test_public_band_convenience_requires_explicit_cutoff() -> None:
    model = build_inas_gasb_model(
        variant="xue2018_q0_bhz",
        eg_ry=-0.5,
        hybridization_ab_ry=0.2,
    )
    with pytest.raises(ValueError, match="path_extent_ab_inv"):
        compute_bands(model, points_per_segment=2)


def test_kane4_band_api_uses_exact_saved_hamiltonians() -> None:
    h0 = np.zeros((4, 4, 2), dtype=np.complex128)
    h0[:, :, 0] = np.diag([-3.0, -1.0, 2.0, 4.0])
    h0[:, :, 1] = np.diag([-2.5, -0.5, 1.5, 5.0])
    micro = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    for ik in range(2):
        for iz in range(2):
            micro[ik, iz] = np.eye(4)
    bundle = Kane4Bundle(
        k_cart_nm_inv=np.asarray([[0.0, 0.0], [0.1, 0.0]]),
        weights_nm2=np.asarray([0.5, 0.5]),
        z_nm=np.asarray([0.0, 1.0]),
        z_weights_nm=np.asarray([0.5, 0.5]),
        h0_mev=h0,
        micro_wavefunctions=micro,
    )
    result = compute_kane4_bands(bundle, return_eigenvectors=True)
    np.testing.assert_array_equal(result.k_cart_nm_inv, bundle.k_cart_nm_inv)
    np.testing.assert_array_equal(result.energies_mev, np.diagonal(h0, axis1=0, axis2=1))
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 4, 4)
    assert result.bundle_fingerprint == bundle.fingerprint()


def test_typed_mean_field_api_is_exact_direct_solver_delegation() -> None:
    model = _model()
    config = InAsGaSbHFConfig(
        kmax_ab_inv=0.2,
        points_per_axis=3,
        init_mode="normal",
        max_iter=1,
    )
    result = run_inas_gasb_hf(model, config)
    direct_state = build_xue2018_hf_state(
        eg_ry=model.params.eg_ry,
        hybridization_ab_ry=model.params.hybridization_ab_ry,
        kmax_ab_inv=config.kmax_ab_inv,
        points_per_axis=config.points_per_axis,
        precision=config.precision,
        self_cell_policy=config.self_cell_policy,
        mesh_policy=config.mesh_policy,
        q0_kernel_backend=config.q0_kernel_backend,
    )
    direct = run_xue2018_hf(
        direct_state,
        init_mode=config.init_mode,
        seed=config.seed,
        seed_amplitude_ry=config.seed_amplitude_ry,
        max_iter=config.max_iter,
        max_oda_lambda=config.max_oda_lambda,
        oda_stall_threshold=config.oda_stall_threshold,
    )
    assert isinstance(result, Xue2018HFResult)
    np.testing.assert_array_equal(result.total_hamiltonian, direct.total_hamiltonian)
    np.testing.assert_array_equal(result.energies, direct.energies)
    np.testing.assert_array_equal(result.raw_projector, direct.raw_projector)
    assert result.chemical_potential_ry == direct.chemical_potential_ry
    assert result.reference_relative_energy_ry_ab2 == direct.reference_relative_energy_ry_ab2
