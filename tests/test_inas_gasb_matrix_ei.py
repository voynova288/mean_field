from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.compressed_fock import (
    CompressedProjectedFockOperator,
    precompute_uniform_toeplitz_exchange_tensor_mev_nm2,
)
from mean_field.systems.inas_gasb.kane4_bundle import (
    Kane4Bundle,
    lift_active_frame_to_reference,
    load_kane4_bundle,
    remove_normal_e1_h1_hybridization,
    save_kane4_bundle,
)
from mean_field.systems.inas_gasb.projected_fock import (
    ProjectedFockOperator,
    fock_energy_density_mev_nm2,
    gauge_transform_k_matrices,
    gauge_transform_local_vertices,
    projected_fock_self_energy,
    uniform_dielectric_green_mev_nm2,
    uniform_dielectric_green_on_mesh_mev_nm2,
)
from mean_field.systems.inas_gasb.projected_vertices import (
    density_vertex_reciprocity_error,
    integrated_density_vertices,
    local_density_vertices,
)
from mean_field.systems.inas_gasb.normal_reference import (
    KaneCarrierProjectors,
    charge_neutral_fermi_density,
)
from mean_field.systems.inas_gasb.matrix_ei import (
    MatrixEIConfig,
    fixed_mu_fermi_density,
    solve_reference_subtracted_matrix_ei,
    weighted_fermi_density,
)
from mean_field.systems.inas_gasb.matrix_ei_interaction import (
    bind_matrix_ei_interaction,
)
from mean_field.systems.inas_gasb.hartree import (
    ProjectedHartreeFockOperator,
    ProjectedHartreeOperator,
    periodic_poisson_electron_energy_mev,
    reference_subtracted_charge_density,
    wafer_b_dielectric_profile,
)
from mean_field.systems.inas_gasb.band_carriers import (
    SignedRadialBandSpectrum,
    signed_band_carrier_density,
)
from mean_field.systems.inas_gasb.conventions import (
    active_electron_hole_areal_densities,
    kane8_time_reversal_unitary,
    ordinary_electron_pair_channels_mev,
    spinful_time_reversal_errors,
)
from mean_field.systems.inas_gasb.axial import (
    axial_eh_seed_hamiltonian,
    axial_time_reversal_residuals,
    expand_axial_radial_bundle,
    matrix_time_reversal_error,
    project_axial_time_reversal_matrices,
)
from mean_field.systems.inas_gasb.layered_green import (
    open_layered_green_mev_nm2,
    open_layered_green_on_mesh_mev_nm2,
    open_layered_poisson_electron_energy_mev,
    open_layered_poisson_matrix_nm2,
    one_sided_gated_layered_green_mev_nm2,
    one_sided_gated_layered_green_on_mesh_mev_nm2,
    one_sided_gated_layered_poisson_matrix_nm2,
    one_sided_gated_poisson_electron_energy_mev,
    periodic_layered_green_mev_nm2,
    periodic_layered_poisson_matrix_nm2,
    uniform_one_sided_gated_discrete_green_mev_nm2,
    uniform_open_discrete_green_mev_nm2,
    uniform_periodic_discrete_green_mev_nm2,
)


def _unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q * phases.conj()[None, :]


def _toy_bundle() -> Kane4Bundle:
    rng = np.random.default_rng(7)
    nk, nz, nmicro, n = 3, 2, 5, 4
    z = np.array([-0.5, 0.5])
    wz = np.ones(nz)
    raw = rng.normal(size=(nz * nmicro, n)) + 1j * rng.normal(size=(nz * nmicro, n))
    base, _ = np.linalg.qr(raw)
    phi = np.empty((nk, nz, nmicro, n), dtype=np.complex128)
    h0 = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        gauge = _unitary(n, rng)
        phi[ik] = (base @ gauge).reshape(nz, nmicro, n)
        diagonal = np.diag(np.array([-2.0, -1.0, 1.0, 2.0]) + 0.1 * ik)
        h0[:, :, ik] = gauge.conj().T @ diagonal @ gauge
    return Kane4Bundle(
        k_cart_nm_inv=np.array([[0.01, 0.0], [0.02, 0.01], [0.03, -0.01]]),
        weights_nm2=np.array([0.1, 0.2, 0.3]),
        z_nm=z,
        z_weights_nm=wz,
        h0_mev=h0,
        micro_wavefunctions=phi,
        provenance={"source": "analytic-test"},
    )


def _bundle_from_h0(h0: np.ndarray, weights: np.ndarray) -> Kane4Bundle:
    n, _, nk = h0.shape
    if n != 4:
        raise ValueError("test bundle requires four active states")
    phi = np.zeros((nk, 2, 4, 4), dtype=np.complex128)
    phi[:, 0, :, :] = np.eye(4)[None, :, :] / np.sqrt(2.0)
    phi[:, 1, :, :] = np.eye(4)[None, :, :] / np.sqrt(2.0)
    return Kane4Bundle(
        k_cart_nm_inv=np.column_stack([np.arange(nk, dtype=float) * 0.01, np.zeros(nk)]),
        weights_nm2=np.asarray(weights, dtype=float),
        z_nm=np.array([-0.5, 0.5]),
        z_weights_nm=np.ones(2),
        h0_mev=np.asarray(h0, dtype=np.complex128),
        micro_wavefunctions=phi,
        provenance={"source": "analytic-scf-test"},
    )


def _zero_fock(bundle: Kane4Bundle) -> ProjectedFockOperator:
    green = np.zeros((bundle.nk, bundle.nk, bundle.nz, bundle.nz), dtype=float)
    return ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="analytic zero-interaction test",
    )


def test_matrix_ei_interaction_binding_is_closed_and_component_exact() -> None:
    bundle = _toy_bundle()
    fock = _zero_fock(bundle)
    bound = bind_matrix_ei_interaction(bundle, fock)
    density = np.zeros_like(bundle.h0_mev)
    total = fock(density)
    hartree_component, fock_component = bound.components(density)

    assert bound.spec.kind == "exchange_only"
    assert bound.spec.required_reference_policy == "normal_ordered_exchange_only"
    assert bound.spec.electrostatic_ensemble == "hartree_disabled_exchange_only"
    assert bound.bundle_fingerprint == bundle.fingerprint()
    assert bound.fingerprint() == fock.fingerprint()
    np.testing.assert_array_equal(bound(density), total)
    np.testing.assert_array_equal(hartree_component, np.zeros_like(total))
    np.testing.assert_array_equal(fock_component, total)

    class DuckInteraction:
        def validate_against_bundle(self, source: Kane4Bundle) -> None:
            del source

        def fingerprint(self) -> str:
            return "f" * 64

        def __call__(self, values: np.ndarray) -> np.ndarray:
            return np.zeros_like(values)

        def components(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(values), np.zeros_like(values)

    with pytest.raises(TypeError, match="registered"):
        bind_matrix_ei_interaction(bundle, DuckInteraction())

    class ForgedProjectedFockOperator(ProjectedFockOperator):
        pass

    forged = ForgedProjectedFockOperator(
        local_vertices=fock.local_vertices,
        green_mev_nm2=fock.green_mev_nm2,
        k_weights_nm2=fock.k_weights_nm2,
        z_weights_nm=fock.z_weights_nm,
        bundle_fingerprint=fock.bundle_fingerprint,
        self_cell_description=fock.self_cell_description,
        exchange_tensor_mev_nm2=fock.exchange_tensor_mev_nm2,
        electrostatics_fingerprint=fock.electrostatics_fingerprint,
        electrostatics_token=fock.electrostatics_token,
    )
    with pytest.raises(TypeError, match="exact registered interaction type"):
        bind_matrix_ei_interaction(bundle, forged)


def _two_layer_scalar_bundle(xi: float) -> Kane4Bundle:
    h0 = np.diag([xi, xi, -xi, -xi]).astype(np.complex128)[:, :, None]
    phi = np.zeros((1, 2, 4, 4), dtype=np.complex128)
    phi[0, 0, 0, 0] = 1.0
    phi[0, 0, 1, 1] = 1.0
    phi[0, 1, 2, 2] = 1.0
    phi[0, 1, 3, 3] = 1.0
    return Kane4Bundle(
        k_cart_nm_inv=np.zeros((1, 2)),
        weights_nm2=np.ones(1),
        z_nm=np.array([-0.5, 0.5]),
        z_weights_nm=np.ones(2),
        h0_mev=h0,
        micro_wavefunctions=phi,
        provenance={"source": "analytic-two-layer-scalar-limit"},
    )


def _two_layer_hartree_bundle() -> Kane4Bundle:
    phi = np.zeros((1, 4, 4, 4), dtype=np.complex128)
    phi[0, 1, 0, 0] = 1.0
    phi[0, 1, 1, 1] = 1.0
    phi[0, 2, 2, 2] = 1.0
    phi[0, 2, 3, 3] = 1.0
    return Kane4Bundle(
        k_cart_nm_inv=np.zeros((1, 2)),
        weights_nm2=np.ones(1),
        z_nm=np.array([-1.5, -0.5, 0.5, 1.5]),
        z_weights_nm=np.ones(4),
        h0_mev=np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.complex128)[:, :, None],
        micro_wavefunctions=phi,
        provenance={"source": "analytic-periodic-hartree-test"},
    )


def _random_hermitian_density(n: int, nk: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        out[:, :, ik] = 0.5 * (raw + raw.conj().T)
    return out


def test_standard_kane_time_reversal_is_spinful() -> None:
    unitary = kane8_time_reversal_unitary()
    unitarity_error, square_error = spinful_time_reversal_errors(unitary)
    assert unitarity_error == 0.0
    assert square_error == 0.0


def test_kane4_bundle_and_density_vertex_identities() -> None:
    bundle = _toy_bundle()
    diagnostics = bundle.validate()
    assert diagnostics["wavefunction_orthonormality_error"] < 1e-12
    assert len(bundle.fingerprint()) == 64

    local = local_density_vertices(bundle.micro_wavefunctions)
    integrated = integrated_density_vertices(local, bundle.z_weights_nm)
    assert density_vertex_reciprocity_error(local) < 1e-12
    for ik in range(bundle.nk):
        np.testing.assert_allclose(integrated[ik, ik], np.eye(4), atol=1e-12)


def test_kane4_bundle_rejects_nonfinite_current_and_invalid_time_reversal() -> None:
    bundle = _toy_bundle()
    bad_current = Kane4Bundle(
        **{**bundle.__dict__, "dhdk_mev_nm": np.full((2, 4, 4, bundle.nk), np.nan + 0j)}
    )
    with pytest.raises(ValueError, match="non-finite"):
        bad_current.validate()
    bad_tr = Kane4Bundle(
        **{**bundle.__dict__, "time_reversal_unitary": np.eye(4, dtype=np.complex128)}
    )
    with pytest.raises(ValueError, match="time-reversal"):
        bad_tr.validate()


def test_kane4_bundle_roundtrip_preserves_fingerprint(tmp_path) -> None:
    source = _toy_bundle()
    tr = np.kron(np.eye(2), np.asarray([[0.0, -1.0], [1.0, 0.0]])).astype(np.complex128)
    bundle = Kane4Bundle(**{**source.__dict__, "time_reversal_unitary": tr})
    path = tmp_path / "bundle.npz"
    metadata = save_kane4_bundle(bundle, path, metadata_path=tmp_path / "metadata.json")
    restored = load_kane4_bundle(path)
    assert metadata["bundle_fingerprint"] == bundle.fingerprint()
    assert restored.fingerprint() == bundle.fingerprint()
    np.testing.assert_allclose(restored.micro_wavefunctions, bundle.micro_wavefunctions)
    np.testing.assert_allclose(restored.time_reversal_unitary, tr)
    original_npz = path.read_bytes()
    original_metadata = (tmp_path / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError, match="must not already exist"):
        save_kane4_bundle(bundle, path, metadata_path=tmp_path / "metadata.json")
    assert path.read_bytes() == original_npz
    assert (tmp_path / "metadata.json").read_bytes() == original_metadata


def test_bundle_fingerprint_is_canonical_across_storage_dtypes(tmp_path) -> None:
    source = _two_layer_scalar_bundle(1.0)
    reduced_precision = Kane4Bundle(
        k_cart_nm_inv=source.k_cart_nm_inv.astype(np.float32),
        weights_nm2=source.weights_nm2.astype(np.float32),
        z_nm=source.z_nm.astype(np.float32),
        z_weights_nm=source.z_weights_nm.astype(np.float32),
        h0_mev=source.h0_mev.astype(np.complex64),
        micro_wavefunctions=source.micro_wavefunctions.astype(np.complex64),
        provenance=source.provenance,
    )
    path = tmp_path / "float32_bundle.npz"
    save_kane4_bundle(reduced_precision, path)
    restored = load_kane4_bundle(path)
    assert restored.fingerprint() == reduced_precision.fingerprint()


def test_polar_lift_preserves_spectrum_and_fixed_reference_frame() -> None:
    rng = np.random.default_rng(9)
    nz, nmicro, n = 2, 5, 4
    raw = rng.normal(size=(nz * nmicro, n)) + 1j * rng.normal(size=(nz * nmicro, n))
    reference_flat, _ = np.linalg.qr(raw)
    reference = reference_flat.reshape(nz, nmicro, n)
    active_rotation = _unitary(n, rng)
    active = np.einsum("zma,ab->zmb", reference, active_rotation)
    energies = np.array([-2.0, -1.0, 1.0, 2.0])
    lifted = lift_active_frame_to_reference(reference, active, energies, np.ones(nz))
    np.testing.assert_allclose(lifted.micro_wavefunctions, reference, atol=2e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(lifted.h0_mev), energies, atol=2e-12)
    np.testing.assert_allclose(lifted.principal_cos2, 1.0, atol=2e-12)
    assert lifted.spectrum_error_mev < 2e-12


def test_axial_expansion_preserves_spectrum_and_time_reversal() -> None:
    nr = 2
    phi = np.zeros((nr, 1, 8, 4), dtype=np.complex128)
    phi[:, 0, 0, 0] = 1.0
    phi[:, 0, 1, 1] = 1.0
    phi[:, 0, 2, 2] = 1.0
    phi[:, 0, 5, 3] = 1.0
    h0 = np.repeat(np.diag([-1.0, -1.0, 1.0, 1.0])[:, :, None], nr, axis=2).astype(np.complex128)
    tr = np.kron(np.eye(2), np.asarray([[0.0, -1.0], [1.0, 0.0]])).astype(np.complex128)
    radial = Kane4Bundle(
        k_cart_nm_inv=np.array([[0.02, 0.0], [0.04, 0.0]]),
        weights_nm2=np.array([0.1, 0.1]),
        z_nm=np.array([0.0, 1.0]),
        z_weights_nm=np.array([0.5, 0.5]),
        h0_mev=h0,
        micro_wavefunctions=np.repeat(phi, 2, axis=1),
        time_reversal_unitary=tr,
        provenance={
            "source": "analytic-axial-test",
            "nphi": 2,
            "finite_k_time_reversal_sewing_verified": True,
        },
    )
    with pytest.raises(ValueError, match="even integer"):
        expand_axial_radial_bundle(radial, nphi=4.5)
    expanded = expand_axial_radial_bundle(radial, nphi=4)
    expanded.validate()
    assert expanded.provenance["nphi"] == 4
    assert expanded.provenance["finite_k_time_reversal_sewing_verified"] is False
    np.testing.assert_allclose(expanded.weights_nm2, 0.025)
    for ik in range(expanded.nk):
        np.testing.assert_allclose(np.linalg.eigvalsh(expanded.h0_mev[:, :, ik]), [-1.0, -1.0, 1.0, 1.0])
    residuals = axial_time_reversal_residuals(expanded)
    assert residuals["h0_time_reversal_error_mev"] < 1e-12
    assert residuals["frame_time_reversal_error_nm_minus_half"] < 1e-12
    seed = axial_eh_seed_hamiltonian(expanded, 1j * np.eye(2), amplitude_mev=0.1)
    assert matrix_time_reversal_error(seed, expanded) < 1e-12
    rng = np.random.default_rng(10)
    random_matrices = _random_hermitian_density(4, expanded.nk, rng)
    projected = project_axial_time_reversal_matrices(random_matrices, expanded)
    assert matrix_time_reversal_error(projected, expanded) < 1e-12


def test_projected_fock_is_gauge_covariant_and_energy_invariant() -> None:
    rng = np.random.default_rng(11)
    bundle = _toy_bundle()
    local = local_density_vertices(bundle.micro_wavefunctions)
    q = np.array([[0.03, 0.02, 0.04], [0.02, 0.035, 0.025], [0.04, 0.025, 0.05]])
    green = uniform_dielectric_green_mev_nm2(q, bundle.z_nm, epsilon_r=15.0)
    operator = ProjectedFockOperator(
        local_vertices=local,
        green_mev_nm2=green,
        k_weights_nm2=bundle.weights_nm2,
        z_weights_nm=bundle.z_weights_nm,
        bundle_fingerprint=bundle.fingerprint(),
        self_cell_description="analytic positive q matrix",
    )
    density = _random_hermitian_density(4, bundle.nk, rng)
    sigma = operator(density)
    energy = operator.energy_density_mev_nm2(density)

    gauges = np.asarray([_unitary(4, rng) for _ in range(bundle.nk)])
    local_rotated = gauge_transform_local_vertices(local, gauges)
    density_rotated = gauge_transform_k_matrices(density, gauges)
    rotated_operator = ProjectedFockOperator(
        local_vertices=local_rotated,
        green_mev_nm2=green,
        k_weights_nm2=bundle.weights_nm2,
        z_weights_nm=bundle.z_weights_nm,
        bundle_fingerprint=bundle.fingerprint(),
        self_cell_description="analytic positive q matrix",
    )
    sigma_rotated = rotated_operator(density_rotated)
    expected_sigma = gauge_transform_k_matrices(sigma, gauges)
    np.testing.assert_allclose(sigma_rotated, expected_sigma, atol=2e-10, rtol=2e-11)
    rotated_energy = fock_energy_density_mev_nm2(
        density_rotated,
        sigma_rotated,
        bundle.weights_nm2,
    )
    assert abs(rotated_energy - energy) < 1e-10


def test_precomputed_exchange_tensor_matches_direct_fock_action() -> None:
    rng = np.random.default_rng(17)
    bundle = _toy_bundle()
    q = np.array([[0.03, 0.02, 0.04], [0.02, 0.035, 0.025], [0.04, 0.025, 0.05]])
    green = uniform_dielectric_green_mev_nm2(q, bundle.z_nm, epsilon_r=15.0)
    direct = ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="analytic compression parity test",
    )
    compressed = ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="analytic compression parity test",
        precompute_exchange_tensor=True,
    )
    density = _random_hermitian_density(4, bundle.nk, rng)
    np.testing.assert_allclose(compressed(density), direct(density), rtol=2e-12, atol=2e-10)
    assert compressed.fingerprint() == direct.fingerprint()
    assert compressed.action_storage_fingerprint() != direct.action_storage_fingerprint()
    assert not compressed.exchange_tensor_mev_nm2.flags.writeable
    saved_green = direct.green_mev_nm2.copy()
    green *= 2.0
    np.testing.assert_allclose(direct.green_mev_nm2, saved_green)
    writable_base = saved_green.copy()
    readonly_view = writable_base.view()
    readonly_view.setflags(write=False)
    isolated = ProjectedFockOperator.from_bundle(
        bundle,
        readonly_view,
        self_cell_description="read-only view ownership regression",
    )
    writable_base *= 3.0
    np.testing.assert_allclose(isolated.green_mev_nm2, saved_green)
    compressed.validate_against_bundle(bundle)


def test_fock_energy_directional_derivative_matches_self_energy() -> None:
    rng = np.random.default_rng(19)
    bundle = _toy_bundle()
    local = local_density_vertices(bundle.micro_wavefunctions)
    q = np.array([[0.03, 0.02, 0.04], [0.02, 0.035, 0.025], [0.04, 0.025, 0.05]])
    green = uniform_dielectric_green_mev_nm2(q, bundle.z_nm, epsilon_r=14.8)
    density = _random_hermitian_density(4, bundle.nk, rng)
    direction = _random_hermitian_density(4, bundle.nk, rng)

    def energy(values: np.ndarray) -> float:
        sigma = projected_fock_self_energy(
            values,
            local,
            green,
            bundle.weights_nm2,
            bundle.z_weights_nm,
        )
        return fock_energy_density_mev_nm2(values, sigma, bundle.weights_nm2)

    sigma = projected_fock_self_energy(
        density,
        local,
        green,
        bundle.weights_nm2,
        bundle.z_weights_nm,
    )
    analytic = np.einsum(
        "k,abk,bak->",
        bundle.weights_nm2,
        sigma,
        direction,
        optimize=True,
    ).real
    step = 1e-7
    finite_difference = (energy(density + step * direction) - energy(density - step * direction)) / (2 * step)
    np.testing.assert_allclose(finite_difference, analytic, atol=2e-5, rtol=2e-7)


def test_mesh_coulomb_self_cell_matches_equal_area_disk_average() -> None:
    k = np.array([[0.01, 0.0], [0.03, 0.0]])
    weights = np.array([0.002, 0.003])
    z = np.array([0.0, 2.0])
    epsilon = 15.0
    green = uniform_dielectric_green_on_mesh_mev_nm2(k, weights, z, epsilon_r=epsilon)
    prefactor = 2.0 * np.pi * 1439.96448 / epsilon
    for ik, weight in enumerate(weights):
        q_cell = np.sqrt((2.0 * np.pi) ** 2 * weight / np.pi)
        np.testing.assert_allclose(green[ik, ik, 0, 0], prefactor * 2.0 / q_cell)
        expected_cross_z = prefactor * 2.0 * (1.0 - np.exp(-2.0 * q_cell)) / (q_cell**2 * 2.0)
        np.testing.assert_allclose(green[ik, ik, 0, 1], expected_cross_z)
    np.testing.assert_allclose(green, np.swapaxes(np.swapaxes(green, 0, 1), 2, 3))


def test_reference_subtracted_charge_sum_rule_and_gauge_covariance() -> None:
    bundle = _toy_bundle()
    rng = np.random.default_rng(43)
    density = _random_hermitian_density(4, bundle.nk, rng)
    charge = reference_subtracted_charge_density(bundle, density)
    expected_number = float(
        np.einsum("k,aak->", bundle.weights_nm2, density, optimize=True).real
    )
    np.testing.assert_allclose(charge.active_number_change_nm2, expected_number, atol=1e-12)
    np.testing.assert_allclose(charge.integrated_profile_nm2, expected_number, atol=1e-12)
    assert abs(charge.sum_rule_error_nm2) < 1e-12

    gauge = np.stack([_unitary(4, rng) for _ in range(bundle.nk)])
    rotated_density = gauge_transform_k_matrices(density, gauge)
    rotated_phi = np.einsum(
        "kzma,kab->kzmb", bundle.micro_wavefunctions, gauge, optimize=True
    )
    rotated_bundle = Kane4Bundle(
        **{
            **bundle.__dict__,
            "h0_mev": gauge_transform_k_matrices(bundle.h0_mev, gauge),
            "micro_wavefunctions": rotated_phi,
            "provenance": {"source": "analytic-test-rotated"},
        }
    )
    rotated_charge = reference_subtracted_charge_density(rotated_bundle, rotated_density)
    np.testing.assert_allclose(
        rotated_charge.electron_density_delta_nm3,
        charge.electron_density_delta_nm3,
        atol=2e-12,
    )


def test_layered_green_matches_uniform_discrete_oracles_and_q0_limits() -> None:
    z = (np.arange(32) - 15.5) * 0.2
    epsilon = np.full(z.size, 15.0)
    q = 0.27
    periodic_matrix = periodic_layered_poisson_matrix_nm2(q, z, epsilon)
    assert np.min(np.linalg.eigvalsh(periodic_matrix)) > 0.0
    periodic_numerical = periodic_layered_green_mev_nm2(q, z, epsilon)
    periodic_exact = uniform_periodic_discrete_green_mev_nm2(q, z, epsilon_r=15.0)
    np.testing.assert_allclose(periodic_numerical, periodic_exact, rtol=2e-12, atol=2e-10)

    open_matrix = open_layered_poisson_matrix_nm2(q, z, epsilon)
    assert np.min(np.linalg.eigvalsh(open_matrix)) > 0.0
    open_numerical = open_layered_green_mev_nm2(q, z, epsilon)
    open_exact = uniform_open_discrete_green_mev_nm2(q, z, epsilon_r=15.0)
    np.testing.assert_allclose(open_numerical, open_exact, rtol=2e-12, atol=2e-10)
    with pytest.raises(ValueError, match="endpoint homogeneous plateaus"):
        open_layered_green_mev_nm2(q, z, epsilon, epsilon_left=14.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        open_layered_green_mev_nm2(q, z[::-1], epsilon[::-1])

    infrared_q = 1e-4
    periodic_ir = periodic_layered_green_mev_nm2(infrared_q, z, epsilon)
    expected_periodic_ir = 4.0 * np.pi * 1439.96448 / (z.size * (z[1] - z[0]) * 15.0)
    np.testing.assert_allclose(
        infrared_q**2 * periodic_ir[0, 7], expected_periodic_ir, rtol=2e-7
    )
    open_ir = open_layered_green_mev_nm2(infrared_q, z, epsilon)
    expected_open_ir = 2.0 * np.pi * 1439.96448 / 15.0
    np.testing.assert_allclose(infrared_q * open_ir[0, 7], expected_open_ir, rtol=4e-4)

    one_point = open_layered_green_on_mesh_mev_nm2(
        np.zeros((1, 2)), np.array([0.0013]), z, epsilon,
        self_cell_quadrature_order=24,
    )
    q_cell = np.sqrt(4.0 * np.pi * 0.0013)
    exact_self_diagonal = (
        8.0 * np.pi * 1439.96448
        / (15.0 * (z[1] - z[0]) * q_cell**2)
        * np.arcsinh((z[1] - z[0]) * q_cell / 2.0)
    )
    np.testing.assert_allclose(
        np.diag(one_point[0, 0]), exact_self_diagonal, rtol=2e-12, atol=2e-10
    )

    density = 1e-4 * np.sin(2.0 * np.pi * np.arange(z.size) / z.size)
    periodic_q0 = periodic_poisson_electron_energy_mev(z, density, epsilon)
    periodic_small_q = (z[1] - z[0]) * (
        periodic_layered_green_mev_nm2(1e-4, z, epsilon) @ density
    )
    periodic_small_q -= np.mean(periodic_small_q)
    np.testing.assert_allclose(periodic_small_q, periodic_q0, rtol=2e-7, atol=2e-8)

    open_q0 = open_layered_poisson_electron_energy_mev(z, density, epsilon)
    open_errors = []
    for small_q in (1e-3, 1e-4, 1e-5):
        open_small_q = (z[1] - z[0]) * (
            open_layered_green_mev_nm2(small_q, z, epsilon) @ density
        )
        open_small_q -= np.mean(open_small_q)
        open_errors.append(float(np.max(np.abs(open_small_q - open_q0))))
    assert open_errors[2] < open_errors[1] < open_errors[0]
    np.testing.assert_allclose(open_small_q, open_q0, rtol=5e-5, atol=2e-6)


def test_open_layered_heterostructure_q0_limit_for_neutral_sources() -> None:
    z = (np.arange(40) - 19.5) * 0.2
    epsilon = np.where(z < -1.0, 14.4, np.where(z < 1.0, 14.55, 15.69))
    rng = np.random.default_rng(47)
    for _ in range(3):
        density = rng.normal(size=z.size)
        density -= np.mean(density)
        density *= 1e-5
        q0 = open_layered_poisson_electron_energy_mev(z, density, epsilon)
        epsilon_plus = 2.0 * epsilon[:-1] * epsilon[1:] / (
            epsilon[:-1] + epsilon[1:]
        )
        flux = 0.0
        integrated = np.zeros_like(z)
        for i in range(z.size - 1):
            flux -= 4.0 * np.pi * 1439.96448 * density[i] * (z[1] - z[0])
            integrated[i + 1] = (
                integrated[i] + flux * (z[1] - z[0]) / epsilon_plus[i]
            )
        integrated -= np.mean(integrated)
        np.testing.assert_allclose(q0, integrated, rtol=2e-12, atol=2e-11)
        errors = []
        for q in (2e-3, 5e-4, 1e-4):
            finite_q = (z[1] - z[0]) * (
                open_layered_green_mev_nm2(q, z, epsilon) @ density
            )
            finite_q -= np.mean(finite_q)
            errors.append(float(np.max(np.abs(finite_q - q0))))
        assert errors[2] < errors[1] < errors[0]


def test_open_layered_mesh_green_self_cell_converges_and_is_reciprocal() -> None:
    z = (np.arange(16) - 7.5) * 0.25
    epsilon = np.where(z < 0.0, 14.4, 15.69)
    k = np.array([[0.0, 0.0], [0.035, 0.0]])
    weights = np.array([0.0012, 0.0017])
    coarse = open_layered_green_on_mesh_mev_nm2(
        k, weights, z, epsilon, self_cell_quadrature_order=8
    )
    fine = open_layered_green_on_mesh_mev_nm2(
        k, weights, z, epsilon, self_cell_quadrature_order=24
    )
    np.testing.assert_allclose(coarse[0, 1], fine[0, 1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(coarse[0, 0], fine[0, 0], rtol=2e-8, atol=2e-7)
    np.testing.assert_allclose(coarse[1, 1], fine[1, 1], rtol=2e-8, atol=2e-7)
    reciprocity = np.swapaxes(np.swapaxes(fine, 0, 1), 2, 3)
    np.testing.assert_allclose(fine, reciprocity, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(fine[0, 1])) > 0.0


def test_one_sided_gate_uniform_discrete_oracle_and_q0_poisson() -> None:
    z = (np.arange(28) - 13.5) * 0.25
    epsilon = np.full(z.size, 15.0)
    for side in ("left", "right"):
        for q in (0.0, 0.03, 0.31):
            matrix = one_sided_gated_layered_poisson_matrix_nm2(
                q, z, epsilon, gate_side=side
            )
            assert np.min(np.linalg.eigvalsh(matrix)) > 0.0
            numerical = one_sided_gated_layered_green_mev_nm2(
                q, z, epsilon, gate_side=side
            )
            exact = uniform_one_sided_gated_discrete_green_mev_nm2(
                q, z, epsilon_r=15.0, gate_side=side
            )
            np.testing.assert_allclose(numerical, exact, rtol=3e-12, atol=3e-10)

        zero_density = np.zeros(z.size)
        gate_energy = 7.25
        constant = one_sided_gated_poisson_electron_energy_mev(
            z,
            zero_density,
            epsilon,
            gate_side=side,
            gate_electron_energy_mev=gate_energy,
        )
        np.testing.assert_allclose(constant, gate_energy, rtol=0.0, atol=2e-12)

        rng = np.random.default_rng(61 if side == "left" else 62)
        density = 1e-4 * rng.normal(size=z.size)
        response = one_sided_gated_poisson_electron_energy_mev(
            z, density, epsilon, gate_side=side
        )
        green0 = one_sided_gated_layered_green_mev_nm2(
            0.0, z, epsilon, gate_side=side
        )
        np.testing.assert_allclose(
            response,
            (z[1] - z[0]) * (green0 @ density),
            rtol=2e-12,
            atol=2e-11,
        )

    left = one_sided_gated_layered_green_mev_nm2(
        0.07, z, epsilon, gate_side="left"
    )
    right = one_sided_gated_layered_green_mev_nm2(
        0.07, z, epsilon, gate_side="right"
    )
    np.testing.assert_allclose(left, right[::-1, ::-1], rtol=2e-12, atol=2e-10)


def test_one_sided_gate_layered_q0_independent_flux_oracle() -> None:
    z = (np.arange(36) - 17.5) * 0.2
    dz = z[1] - z[0]
    epsilon = np.where(z < -1.0, 14.4, np.where(z < 1.0, 14.55, 15.69))
    epsilon_plus = 2.0 * epsilon[:-1] * epsilon[1:] / (
        epsilon[:-1] + epsilon[1:]
    )
    rng = np.random.default_rng(67)
    for _ in range(3):
        density = 1e-5 * rng.normal(size=z.size)
        numerical = one_sided_gated_poisson_electron_energy_mev(
            z, density, epsilon, gate_side="left"
        )
        right_flux = 0.0
        internal_flux = np.zeros(z.size - 1)
        for i in range(z.size - 1, -1, -1):
            left_flux = right_flux + 4.0 * np.pi * 1439.96448 * density[i] * dz
            if i > 0:
                internal_flux[i - 1] = left_flux
            right_flux = left_flux
        integrated = np.empty_like(z)
        integrated[0] = right_flux * dz / (2.0 * epsilon[0])
        for i in range(z.size - 1):
            integrated[i + 1] = integrated[i] + internal_flux[i] * dz / epsilon_plus[i]
        np.testing.assert_allclose(numerical, integrated, rtol=3e-12, atol=3e-11)


def test_one_sided_gate_mesh_self_cell_and_common_builder() -> None:
    z = (np.arange(18) - 8.5) * 0.25
    epsilon = np.where(z < 0.0, 14.4, 15.69)
    k = np.array([[0.0, 0.0], [0.04, 0.0]])
    weights = np.array([0.0011, 0.0016])
    coarse = one_sided_gated_layered_green_on_mesh_mev_nm2(
        k,
        weights,
        z,
        epsilon,
        gate_side="left",
        self_cell_quadrature_order=8,
    )
    fine = one_sided_gated_layered_green_on_mesh_mev_nm2(
        k,
        weights,
        z,
        epsilon,
        gate_side="left",
        self_cell_quadrature_order=24,
    )
    np.testing.assert_allclose(coarse[0, 1], fine[0, 1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(coarse[0, 0], fine[0, 0], rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(coarse[1, 1], fine[1, 1], rtol=2e-9, atol=2e-8)
    reciprocity = np.swapaxes(np.swapaxes(fine, 0, 1), 2, 3)
    np.testing.assert_allclose(fine, reciprocity, atol=1e-10)

    uniform_z = (np.arange(14) - 6.5) * 0.2
    uniform_epsilon = np.full(uniform_z.size, 15.0)
    uniform_weight = 0.0013
    q_cell = np.sqrt(4.0 * np.pi * uniform_weight)
    nodes, gauss_weights = np.polynomial.legendre.leggauss(96)
    q_nodes = 0.5 * q_cell * (nodes + 1.0)
    q_weights = 0.5 * q_cell * gauss_weights
    for side in ("left", "right"):
        numerical_cell = one_sided_gated_layered_green_on_mesh_mev_nm2(
            np.zeros((1, 2)),
            np.array([uniform_weight]),
            uniform_z,
            uniform_epsilon,
            gate_side=side,
            self_cell_quadrature_order=24,
        )[0, 0]
        analytic_cell = np.zeros_like(numerical_cell)
        for q, weight_q in zip(q_nodes, q_weights):
            analytic_cell += float(weight_q * q) * (
                uniform_one_sided_gated_discrete_green_mev_nm2(
                    float(q),
                    uniform_z,
                    epsilon_r=15.0,
                    gate_side=side,
                )
            )
        analytic_cell *= 2.0 / q_cell**2
        np.testing.assert_allclose(
            numerical_cell, analytic_cell, rtol=3e-12, atol=3e-10
        )

    bundle = _two_layer_hartree_bundle()
    bundle_epsilon = wafer_b_dielectric_profile(bundle.z_nm)
    left = ProjectedHartreeFockOperator.from_bundle_one_sided_gated(
        bundle,
        bundle_epsilon,
        gate_side="left",
        epsilon_open=float(bundle_epsilon[-1] + 5e-13),
        self_cell_quadrature_order=8,
    )
    right = ProjectedHartreeFockOperator.from_bundle_one_sided_gated(
        bundle,
        bundle_epsilon,
        gate_side="right",
        self_cell_quadrature_order=8,
    )
    compressed_left = ProjectedHartreeFockOperator.from_bundle_one_sided_gated(
        bundle,
        bundle_epsilon,
        gate_side="left",
        self_cell_quadrature_order=8,
        precompute_exchange_tensor=True,
    )
    left.validate_against_bundle(bundle)
    compressed_left.validate_against_bundle(bundle)
    assert left.fock.fingerprint() == compressed_left.fock.fingerprint()
    assert (
        left.fock.action_storage_fingerprint()
        != compressed_left.fock.action_storage_fingerprint()
    )
    assert left.hartree.boundary_condition == "one_sided_gate_left"
    assert left.hartree.electrostatics_token is left.fock.electrostatics_token
    assert left.hartree.electrostatics_fingerprint != right.hartree.electrostatics_fingerprint
    assert "one-sided metal gate" in left.fock.self_cell_description
    density = np.zeros_like(bundle.h0_mev)
    density[:, :, 0] = np.diag([0.1, 0.1, -0.1, -0.1])
    sigma_h, sigma_f = left.components(density)
    np.testing.assert_allclose(sigma_h, np.swapaxes(sigma_h.conj(), 0, 1), atol=1e-12)
    np.testing.assert_allclose(sigma_f, np.swapaxes(sigma_f.conj(), 0, 1), atol=1e-12)
    assert left.hartree.energy_density_mev_nm2(density) > 0.0


def test_common_open_layered_hartree_fock_builder_uses_one_boundary_problem() -> None:
    bundle = _two_layer_hartree_bundle()
    epsilon = wafer_b_dielectric_profile(bundle.z_nm)
    combined = ProjectedHartreeFockOperator.from_bundle_open_layered(
        bundle,
        epsilon,
        self_cell_quadrature_order=8,
    )
    combined.validate_against_bundle(bundle)
    assert combined.hartree.boundary_condition == "open_zero_field"
    assert combined.hartree.electrostatics_fingerprint == combined.fock.electrostatics_fingerprint
    assert combined.hartree.electrostatics_token is combined.fock.electrostatics_token
    assert not combined.hartree.epsilon_r.flags.writeable
    assert not combined.fock.green_mev_nm2.flags.writeable
    assert "open layered dielectric" in combined.fock.self_cell_description
    reverse = np.swapaxes(
        np.swapaxes(combined.fock.green_mev_nm2, 0, 1), 2, 3
    )
    np.testing.assert_allclose(combined.fock.green_mev_nm2, reverse, atol=1e-10)
    density = np.zeros_like(bundle.h0_mev)
    density[:, :, 0] = np.diag([0.1, 0.1, -0.1, -0.1])
    sigma_h, sigma_f = combined.components(density)
    np.testing.assert_allclose(sigma_h, np.swapaxes(sigma_h.conj(), 0, 1), atol=1e-12)
    np.testing.assert_allclose(sigma_f, np.swapaxes(sigma_f.conj(), 0, 1), atol=1e-12)
    assert combined.hartree.energy_density_mev_nm2(density) > 0.0
    certified_epsilon = combined.hartree.epsilon_r.copy()
    epsilon[:] = 99.0
    np.testing.assert_allclose(combined.hartree.epsilon_r, certified_epsilon)

    legacy_green = uniform_dielectric_green_on_mesh_mev_nm2(
        bundle.k_cart_nm_inv,
        bundle.weights_nm2,
        bundle.z_nm,
        epsilon_r=15.0,
    )
    legacy_fock = ProjectedFockOperator.from_bundle(
        bundle,
        legacy_green,
        self_cell_description="uncertified uniform open-space diagnostic",
    )
    with pytest.raises(
        ValueError, match="one builder-attested electrostatics specification"
    ):
        ProjectedHartreeFockOperator(hartree=combined.hartree, fock=legacy_fock)
    falsely_labeled_fock = ProjectedFockOperator.from_bundle(
        bundle,
        2.0 * combined.fock.green_mev_nm2,
        self_cell_description="adversarial two-times-Coulomb label test",
        electrostatics_fingerprint=combined.hartree.electrostatics_fingerprint,
    )
    with pytest.raises(
        ValueError, match="one builder-attested electrostatics specification"
    ):
        ProjectedHartreeFockOperator(
            hartree=combined.hartree,
            fock=falsely_labeled_fock,
        )
    replayed_token_fock = ProjectedFockOperator.from_bundle(
        bundle,
        2.0 * combined.fock.green_mev_nm2,
        self_cell_description="adversarial replayed-token test",
        electrostatics_fingerprint=combined.hartree.electrostatics_fingerprint,
        _electrostatics_token=combined.hartree.electrostatics_token,
    )
    with pytest.raises(
        ValueError, match="one builder-attested electrostatics specification"
    ):
        ProjectedHartreeFockOperator(
            hartree=combined.hartree,
            fock=replayed_token_fock,
        )
    zero_fock = _zero_fock(bundle)
    zero_green_nonzero_tensor = ProjectedFockOperator(
        local_vertices=zero_fock.local_vertices,
        green_mev_nm2=zero_fock.green_mev_nm2,
        k_weights_nm2=zero_fock.k_weights_nm2,
        z_weights_nm=zero_fock.z_weights_nm,
        bundle_fingerprint=zero_fock.bundle_fingerprint,
        self_cell_description="adversarial zero-Green nonzero-tensor test",
        exchange_tensor_mev_nm2=np.ones(
            (bundle.nk, bundle.nk) + (bundle.h0_mev.shape[0],) * 4,
            dtype=np.complex128,
        ),
    )
    with pytest.raises(
        ValueError, match="one builder-attested electrostatics specification"
    ):
        ProjectedHartreeFockOperator(
            hartree=combined.hartree,
            fock=zero_green_nonzero_tensor,
        )
    acknowledged = ProjectedHartreeFockOperator(
        hartree=combined.hartree,
        fock=legacy_fock,
        allow_mixed_electrostatics=True,
    )
    assert acknowledged.allow_mixed_electrostatics

    different_bundle = Kane4Bundle(
        **{**bundle.__dict__, "provenance": {"source": "different-bundle"}}
    )
    with pytest.raises(ValueError, match="different bundles"):
        ProjectedHartreeFockOperator(
            hartree=combined.hartree,
            fock=_zero_fock(different_bundle),
        )

    mutable_bundle = _two_layer_hartree_bundle()
    protected = ProjectedHartreeFockOperator.from_bundle_open_layered(
        mutable_bundle,
        wafer_b_dielectric_profile(mutable_bundle.z_nm),
        self_cell_quadrature_order=8,
    )
    saved_weight = float(protected.hartree.k_weights_nm2[0])
    mutable_bundle.weights_nm2[0] *= 2.0
    assert protected.hartree.k_weights_nm2[0] == saved_weight
    assert protected.fock.k_weights_nm2[0] == saved_weight
    with pytest.raises(ValueError, match="fingerprints"):
        protected.validate_against_bundle(mutable_bundle)


def test_projected_hartree_covariance_energy_invariance_and_derivative() -> None:
    bundle = _two_layer_hartree_bundle()
    epsilon = wafer_b_dielectric_profile(bundle.z_nm)
    hartree = ProjectedHartreeOperator.from_bundle(bundle, epsilon)
    density = np.zeros_like(bundle.h0_mev)
    density[:, :, 0] = np.diag([0.1, 0.1, -0.1, -0.1])
    rng = np.random.default_rng(51)
    direction_block = _random_hermitian_density(4, 1, rng)[:, :, 0]
    direction_block -= np.trace(direction_block).real * np.eye(4) / 4.0
    direction = direction_block[:, :, None]

    sigma = hartree(density)
    analytic = float(
        np.einsum("k,abk,bak->", bundle.weights_nm2, sigma, direction, optimize=True).real
    )
    step = 1e-7
    finite_difference = (
        hartree.energy_density_mev_nm2(density + step * direction)
        - hartree.energy_density_mev_nm2(density - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite_difference, analytic, rtol=2e-8, atol=2e-9)

    gauge = _unitary(4, rng)[None, :, :]
    rotated_density = gauge_transform_k_matrices(density, gauge)
    rotated_bundle = Kane4Bundle(
        **{
            **bundle.__dict__,
            "h0_mev": gauge_transform_k_matrices(bundle.h0_mev, gauge),
            "micro_wavefunctions": np.einsum(
                "kzma,kab->kzmb", bundle.micro_wavefunctions, gauge, optimize=True
            ),
            "provenance": {"source": "analytic-periodic-hartree-rotated"},
        }
    )
    rotated_hartree = ProjectedHartreeOperator.from_bundle(rotated_bundle, epsilon)
    rotated_sigma = rotated_hartree(rotated_density)
    np.testing.assert_allclose(
        rotated_sigma,
        gauge_transform_k_matrices(sigma, gauge),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        rotated_hartree.energy_density_mev_nm2(rotated_density),
        hartree.energy_density_mev_nm2(density),
        rtol=2e-12,
        atol=2e-12,
    )


def test_parent_hartree_double_count_is_rejected() -> None:
    source = _two_layer_hartree_bundle()
    bundle = Kane4Bundle(
        **{
            **source.__dict__,
            "h0_mev": -source.h0_mev,
            "provenance": {
                **source.provenance,
                "parent_contains_selfconsistent_hartree": True,
            },
        }
    )
    combined = ProjectedHartreeFockOperator(
        hartree=ProjectedHartreeOperator.from_bundle(
            bundle, np.full(bundle.nz, 15.0)
        ),
        fock=_zero_fock(bundle),
    )
    with pytest.raises(ValueError, match="double count"):
        solve_reference_subtracted_matrix_ei(
            bundle,
            combined,
            config=MatrixEIConfig(
                momentum_policy="analytic_or_legacy_diagnostic",
                reference_policy="normal_ordered_hartree_fock",
            ),
        )


def test_mixed_electrostatics_solver_metadata_is_not_silently_promoted() -> None:
    source = _two_layer_hartree_bundle()
    bundle = Kane4Bundle(**{**source.__dict__, "h0_mev": -source.h0_mev})
    hartree = ProjectedHartreeOperator.from_bundle(
        bundle, np.full(bundle.nz, 15.0)
    )
    mixed = ProjectedHartreeFockOperator(
        hartree=hartree,
        fock=_zero_fock(bundle),
        allow_mixed_electrostatics=True,
    )
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        mixed,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=5,
            reference_policy="normal_ordered_hartree_fock",
        ),
    )
    assert result.run.converged
    assert result.electrostatic_ensemble.startswith("mixed_electrostatics_scaffold")
    assert "mixed_electrostatics_scaffold" in result.physics_status
    assert result.interaction_kind == "projected_hartree_fock"
    assert result.interaction_model_label == "normal-ordered Hartree-Fock"
    assert result.interaction_authority == "explicit_mixed_electrostatics_scaffold"


def test_projected_periodic_hartree_penalizes_layer_charge_transfer() -> None:
    bundle = _two_layer_hartree_bundle()
    hartree = ProjectedHartreeOperator.from_bundle(bundle, wafer_b_dielectric_profile(bundle.z_nm))
    density = np.zeros_like(bundle.h0_mev)
    density[:, :, 0] = np.diag([0.1, 0.1, -0.1, -0.1])
    sigma, profile, potential = hartree.components(density)
    assert abs(np.sum(profile * bundle.z_weights_nm)) < 1e-12
    assert hartree.energy_density_mev_nm2(density) > 0.0
    np.testing.assert_allclose(sigma, np.swapaxes(sigma.conj(), 0, 1), atol=1e-12)
    assert np.max(potential) > np.min(potential)

    combined = ProjectedHartreeFockOperator(hartree=hartree, fock=_zero_fock(bundle))
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        combined,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            reference_policy="normal_ordered_hartree_fock",
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
    )
    assert result.run.converged
    np.testing.assert_allclose(result.sigma_hartree_mev, 0.0, atol=1e-13)
    assert result.hartree_internal_energy_density_mev_nm2 == 0.0


def test_fock_operator_rejects_nonfinite_vertices() -> None:
    bundle = _toy_bundle()
    local = local_density_vertices(bundle.micro_wavefunctions)
    local[0, 0, 0, 0, 0] = np.nan
    green = np.zeros((bundle.nk, bundle.nk, bundle.nz, bundle.nz))
    with pytest.raises(ValueError, match="finite"):
        ProjectedFockOperator(
            local_vertices=local,
            green_mev_nm2=green,
            k_weights_nm2=bundle.weights_nm2,
            z_weights_nm=bundle.z_weights_nm,
            bundle_fingerprint=bundle.fingerprint(),
            self_cell_description="nonfinite rejection test",
        )


def test_streamed_uniform_toeplitz_compressed_fock_matches_dense_action(
    tmp_path,
) -> None:
    bundle = _toy_bundle()
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        bundle.k_cart_nm_inv,
        bundle.weights_nm2,
        bundle.z_nm,
        epsilon_r=15.0,
    )
    direct = ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="dense equal-area self-cell oracle",
    )
    tensor_path = tmp_path / "streamed_exchange.npy"
    tensor = precompute_uniform_toeplitz_exchange_tensor_mev_nm2(
        bundle,
        epsilon_r=15.0,
        workers=2,
        output_npy=tensor_path,
    )
    compressed = CompressedProjectedFockOperator.from_bundle_and_tensor(
        bundle,
        tensor,
        self_cell_description="streamed uniform dielectric equal-area self cell",
    )
    rng = np.random.default_rng(20260827)
    density = rng.normal(size=bundle.h0_mev.shape) + 1j * rng.normal(
        size=bundle.h0_mev.shape
    )
    density = 0.5 * (density + np.swapaxes(density.conj(), 0, 1))
    np.testing.assert_allclose(compressed(density), direct(density), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        compressed.energy_density_mev_nm2(density),
        direct.energy_density_mev_nm2(density),
        rtol=2e-12,
        atol=2e-12,
    )
    compressed.validate_against_bundle(bundle)
    bound = bind_matrix_ei_interaction(bundle, compressed)
    bound_hartree, bound_fock = bound.components(density)
    assert bound.spec.authority == "explicit_precomputed_tensor"
    np.testing.assert_array_equal(bound_hartree, np.zeros_like(bound_fock))
    np.testing.assert_array_equal(bound_fock, compressed(density))
    reciprocity_error = max(
        float(
            np.max(
                np.abs(
                    tensor[ip, ik]
                    - tensor[ik, ip].transpose(3, 2, 1, 0)
                )
            )
        )
        for ik in range(bundle.nk)
        for ip in range(bundle.nk)
    )
    assert reciprocity_error < 2e-11


def test_asymmetric_coulomb_kernel_is_rejected() -> None:
    bundle = _toy_bundle()
    q = np.full((bundle.nk, bundle.nk), 0.03)
    q[0, 1] = 0.02
    q[1, 0] = 0.04
    with pytest.raises(ValueError, match="symmetric"):
        uniform_dielectric_green_mev_nm2(q, bundle.z_nm, epsilon_r=15.0)


def test_ordinary_electron_mu_enters_charge_not_pair_channel() -> None:
    conduction = np.array([112.0, 119.0, 131.0])
    valence = np.array([116.0, 113.0, 109.0])
    mu_electron = 113.8688022886634
    xi, eta = ordinary_electron_pair_channels_mev(
        conduction, valence, mu_electron
    )
    np.testing.assert_allclose(xi, conduction - valence, atol=1e-14)
    np.testing.assert_allclose(
        eta, conduction + valence - 2.0 * mu_electron, atol=1e-14
    )

    energy_shift = 47.25
    shifted_xi, shifted_eta = ordinary_electron_pair_channels_mev(
        conduction + energy_shift,
        valence + energy_shift,
        mu_electron + energy_shift,
    )
    np.testing.assert_allclose(shifted_xi, xi, atol=1e-14)
    np.testing.assert_allclose(shifted_eta, eta, atol=1e-14)

    mu_shift = 0.37
    moved_xi, moved_eta = ordinary_electron_pair_channels_mev(
        conduction, valence, mu_electron + mu_shift
    )
    np.testing.assert_allclose(moved_xi, xi, atol=1e-14)
    np.testing.assert_allclose(moved_eta, eta - 2.0 * mu_shift, atol=1e-14)


def test_fixed_mu_density_is_grand_canonical_and_energy_shift_covariant() -> None:
    nk = 3
    energies = np.array(
        [
            [-2.0, -1.8, -1.6],
            [-0.7, -0.5, -0.3],
            [0.4, 0.6, 0.8],
            [1.5, 1.7, 1.9],
        ]
    )
    h0 = np.zeros((4, 4, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag(energies[:, ik])
    weights = np.array([0.03, 0.07, 0.11])
    temperature = 0.4
    mu = 0.15
    update = fixed_mu_fermi_density(
        h0,
        weights,
        temperature_K=temperature,
        mu_mev=mu,
    )
    expected_occupation = 1.0 / (
        1.0 + np.exp((energies - mu) / (0.08617333262 * temperature))
    )
    for ik in range(nk):
        np.testing.assert_allclose(
            update.density[:, :, ik],
            np.diag(expected_occupation[:, ik]),
            rtol=2e-13,
            atol=2e-13,
        )
    expected_number = float(np.einsum("k,nk->", weights, expected_occupation))
    np.testing.assert_allclose(
        update.observables["active_number_density_nm2"], expected_number, atol=1e-14
    )
    assert update.mu == mu
    assert "target_occupation_per_k" not in update.observables
    assert "number_residual_nm2" not in update.observables

    shift = 7.25
    shifted = fixed_mu_fermi_density(
        h0 + shift * np.eye(4, dtype=np.complex128)[:, :, None],
        weights,
        temperature_K=temperature,
        mu_mev=mu + shift,
    )
    np.testing.assert_allclose(shifted.density, update.density, atol=2e-13)
    np.testing.assert_allclose(shifted.energies, update.energies + shift, atol=2e-13)


def test_matrix_ei_exposes_restart_checkpoint_callbacks() -> None:
    bundle = _toy_bundle()
    steps: list[tuple[int, np.ndarray, float]] = []
    finals: list[tuple[np.ndarray, np.ndarray, float]] = []

    def save_step(state, step) -> None:
        steps.append((step.iteration, state.density.copy(), step.norm_raw))
        np.testing.assert_allclose(state.density, step.mixed_density, atol=0.0)

    def save_final(state, update) -> None:
        finals.append((state.density.copy(), update.density.copy(), state.mu))

    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            precision=1e-12,
            max_iter=3,
            search_mode="normal_reference",
            reference_policy="normal_ordered_exchange_only",
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=0.0,
            normal_ordering_reference_policy="electron_hole_vacuum",
        ),
        step_callback=save_step,
        final_state_callback=save_final,
    )
    assert result.run.converged
    assert len(steps) == result.run.iterations == 1
    assert steps[0][0] == 1
    assert len(finals) == 1
    np.testing.assert_allclose(finals[0][0], result.density_delta, atol=0.0)
    np.testing.assert_allclose(finals[0][1], result.density_delta, atol=1e-14)
    assert finals[0][2] == 0.0


def test_fixed_mu_config_is_fail_closed_and_solver_does_not_reroot() -> None:
    with pytest.raises(ValueError, match="requires a finite fixed_mu_mev"):
        MatrixEIConfig(constraint_policy="kane_poisson_fixed_mu")
    with pytest.raises(ValueError, match="only valid"):
        MatrixEIConfig(fixed_mu_mev=0.0)

    nk = 4
    h0 = np.zeros((4, 4, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag([-2.0 + 0.1 * ik, -1.0, 1.0, 2.0 + 0.2 * ik])
    weights = np.array([0.03, 0.07, 0.11, 0.19])
    fixed_mu = -1.25
    bundle = _bundle_from_h0(h0, weights)
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=fixed_mu,
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
    )
    assert result.run.converged
    assert result.reference_mu_mev == fixed_mu
    assert result.mu_mev == fixed_mu
    assert result.run.state.diagnostics["fixed_mu_mev"] == fixed_mu
    assert abs(result.run.state.diagnostics["reference_occupation_per_k"] - 2.0) > 0.1
    np.testing.assert_allclose(result.density_delta, 0.0, atol=1e-13)
    assert abs(result.canonical_free_energy_difference_mev_nm2) < 1e-13
    assert abs(result.grand_potential_difference_mev_nm2) < 1e-13
    assert "fixed Kane-Poisson ordinary-electron mu" in result.classification

    broken_gap_h0 = np.zeros_like(h0)
    for ik in range(nk):
        broken_gap_h0[:, :, ik] = np.diag([-2.0, -1.0, 1.0, 2.0])
    broken_gap = _bundle_from_h0(broken_gap_h0, weights)
    carrier_result = solve_reference_subtracted_matrix_ei(
        broken_gap,
        _zero_fock(broken_gap),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.02,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=0.0,
            normal_ordering_reference_policy="electron_hole_vacuum",
        ),
    )
    assert carrier_result.run.converged
    assert carrier_result.reference_mu_mev is None
    assert carrier_result.noninteracting_mu_mev == 0.0
    expected_vacuum = np.repeat(
        broken_gap.basis.h1_electron_projector[:, :, None], nk, axis=2
    )
    expected_physical = np.repeat(
        broken_gap.basis.electron_projector[:, :, None], nk, axis=2
    )
    np.testing.assert_allclose(carrier_result.reference_density, expected_vacuum, atol=1e-14)
    np.testing.assert_allclose(
        carrier_result.noninteracting_density, expected_physical, atol=1e-14
    )
    np.testing.assert_allclose(
        carrier_result.density_delta,
        expected_physical - expected_vacuum,
        atol=1e-14,
    )
    np.testing.assert_allclose(carrier_result.total_density, expected_physical, atol=1e-14)
    np.testing.assert_allclose(carrier_result.sigma_hartree_mev, 0.0, atol=1e-14)
    np.testing.assert_allclose(carrier_result.sigma_fock_mev, 0.0, atol=1e-14)
    assert carrier_result.physics_status == "derived_reference_subtracted_ordinary_electron_hf"
    assert carrier_result.ordinary_electron_order is not None
    np.testing.assert_allclose(
        carrier_result.ordinary_electron_order.delta_du_eh_mev, 0.0, atol=1e-14
    )
    assert abs(carrier_result.run.state.diagnostics["active_number_change_nm2"]) < 1e-13
    assert "E1-empty/H1-filled normal ordering" in carrier_result.classification


def test_weighted_density_and_interaction_off_reference_limit() -> None:
    nk = 4
    h0 = np.zeros((4, 4, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag([-2.0 + 0.1 * ik, -1.0, 1.0, 2.0 + 0.2 * ik])
    unequal_weights = np.array([0.03, 0.07, 0.11, 0.19])
    update = weighted_fermi_density(
        h0,
        unequal_weights,
        temperature_K=0.2,
        target_occupation_per_k=2.0,
    )
    assert abs(update.observables["achieved_occupation_per_k"] - 2.0) < 1e-11
    assert abs(update.observables["number_residual_nm2"]) < 1e-12
    n_e, n_h = active_electron_hole_areal_densities(update.density, unequal_weights)
    assert abs(n_e - n_h) < 1e-12

    bundle = _bundle_from_h0(h0, unequal_weights)
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
    )
    assert result.run.converged
    np.testing.assert_allclose(result.density_delta, 0.0, atol=1e-13)
    np.testing.assert_allclose(result.hamiltonian_mev, h0, atol=1e-13)
    np.testing.assert_allclose(result.excitonic_singular_values_mev, 0.0, atol=1e-13)
    assert abs(result.canonical_free_energy_difference_mev_nm2) < 1e-13
    assert abs(result.run.state.diagnostics["active_number_change_nm2"]) < 1e-13
    assert abs(result.run.state.diagnostics["charge_sum_rule_error_nm2"]) < 1e-13

    resumed_result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            search_mode="seeded_ei",
            normal_ordering_reference_policy="noninteracting_fermi_state",
        ),
        initial_density_delta=np.zeros_like(h0),
    )
    assert resumed_result.run.converged
    np.testing.assert_allclose(resumed_result.density_delta, 0.0, atol=1e-13)


def test_signed_radial_band_carriers_match_linear_simplex_counting(tmp_path) -> None:
    spectrum = SignedRadialBandSpectrum(
        k_nm_inv=np.array([0.0, 1.0, 2.0]),
        energies_mev=np.array([[1.0, -1.0], [0.0, 0.0], [-3.0, 3.0]]),
        band_indices=(-1, 1),
        band_labels=("H1", "E1"),
        source="analytic",
    )
    density = signed_band_carrier_density(spectrum)
    expected = 1.0 / (4.0 * np.pi)
    assert abs(density.fermi_energy_mev) < 1e-12
    np.testing.assert_allclose(density.electron_density_nm2, expected, atol=1e-12)
    np.testing.assert_allclose(density.hole_density_nm2, expected, atol=1e-12)
    assert density.outer_interval_electron_by_band_nm2[1] == 0.0
    assert density.outer_interval_hole_by_band_nm2[-1] == 0.0

    xml = tmp_path / "signed.xml"
    xml.write_text(
        "<root><dispersion>"
        '<momentum k="0"><energies>1 -1</energies><bandindex>-1 1</bandindex>'
        "<characters>H1 E1</characters></momentum>"
        '<momentum k="1"><energies>0 0</energies><bandindex>-1 1</bandindex></momentum>'
        '<momentum k="2"><energies>-3 3</energies><bandindex>-1 1</bandindex></momentum>'
        "</dispersion></root>"
    )
    parsed = SignedRadialBandSpectrum.from_kdotpy_xml(xml)
    np.testing.assert_allclose(parsed.energies_mev, spectrum.energies_mev)
    assert parsed.band_indices == spectrum.band_indices
    assert parsed.band_labels == spectrum.band_labels

    insulator = SignedRadialBandSpectrum(
        k_nm_inv=np.array([0.0, 1.0]),
        energies_mev=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        band_indices=(-1, 1),
        band_labels=("H", "E"),
        source="analytic-gap",
    )
    plateau = signed_band_carrier_density(insulator)
    assert plateau.fermi_energy_mev == 0.0
    assert plateau.electron_density_nm2 == 0.0
    assert plateau.hole_density_nm2 == 0.0


def test_microscopic_carrier_neutrality_uses_normal_reference_owner() -> None:
    bundle = _two_layer_scalar_bundle(1.0)
    carriers = KaneCarrierProjectors.from_bundle(bundle)
    neutral = charge_neutral_fermi_density(
        bundle.h0_mev,
        bundle.weights_nm2,
        carriers,
        temperature_K=0.2,
    )
    assert abs(neutral.charge_imbalance_nm2) < 1e-12

    result = solve_reference_subtracted_matrix_ei(
        bundle,
        _zero_fock(bundle),
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.2,
            mixing=0.3,
            precision=1e-11,
            max_iter=10,
            constraint_policy="microscopic_charge_neutrality",
        ),
    )
    assert result.run.converged
    final_carriers = KaneCarrierProjectors.from_bundle(bundle)
    n_e, n_h = final_carriers.densities_nm2(
        result.total_density,
        bundle.weights_nm2,
    )
    assert abs(n_e - n_h) < 1e-12


def test_remove_normal_e1_h1_hybridization_is_typed_and_block_covariant() -> None:
    h0 = np.zeros((4, 4, 2), dtype=np.complex128)
    for ik in range(2):
        h0[:, :, ik] = np.array(
            [
                [-1.0 + 0.1 * ik, 0.2j, 0.3 + 0.1j, -0.2j],
                [-0.2j, -0.6, 0.15j, 0.25 - 0.05j],
                [0.3 - 0.1j, -0.15j, 0.7, -0.1j],
                [0.2j, 0.25 + 0.05j, 0.1j, 1.2 - 0.1 * ik],
            ],
            dtype=np.complex128,
        )
    base = _bundle_from_h0(h0, np.array([0.1, 0.2]))
    derivative = np.stack((0.4 * h0, -0.3 * h0), axis=0)
    bundle = Kane4Bundle(**{**base.__dict__, "dhdk_mev_nm": derivative})
    reduced = remove_normal_e1_h1_hybridization(bundle)
    np.testing.assert_array_equal(reduced.h0_mev[:2, :2], h0[:2, :2])
    np.testing.assert_array_equal(reduced.h0_mev[2:, 2:], h0[2:, 2:])
    np.testing.assert_array_equal(reduced.h0_mev[:2, 2:], 0.0)
    np.testing.assert_array_equal(reduced.h0_mev[2:, :2], 0.0)
    np.testing.assert_array_equal(reduced.dhdk_mev_nm[:, :2, 2:], 0.0)
    np.testing.assert_array_equal(reduced.micro_wavefunctions, bundle.micro_wavefunctions)
    assert reduced.fingerprint() != bundle.fingerprint()
    receipt = reduced.provenance["normal_e1_h1_hybridization_removal"]
    assert receipt["parent_bundle_fingerprint"] == bundle.fingerprint()
    assert receipt["maximum_removed_h0_block_mev"] > 0.0
    with pytest.raises(ValueError, match="already removed"):
        remove_normal_e1_h1_hybridization(reduced)

    rng = np.random.default_rng(113)
    gauge = np.zeros((4, 4), dtype=np.complex128)
    gauge[:2, :2] = _unitary(2, rng)
    gauge[2:, 2:] = _unitary(2, rng)
    gauged_h0 = np.einsum(
        "ab,bck,cd->adk", gauge.conj().T, h0, gauge, optimize=True
    )
    gauged_phi = np.einsum(
        "kzma,ab->kzmb", bundle.micro_wavefunctions, gauge, optimize=True
    )
    gauged_derivative = np.einsum(
        "ab,xbck,cd->xadk",
        gauge.conj().T,
        derivative,
        gauge,
        optimize=True,
    )
    gauged = Kane4Bundle(
        **{
            **bundle.__dict__,
            "h0_mev": gauged_h0,
            "micro_wavefunctions": gauged_phi,
            "dhdk_mev_nm": gauged_derivative,
            "provenance": {"source": "block-gauge-test"},
        }
    )
    gauged_reduced = remove_normal_e1_h1_hybridization(gauged)
    expected = np.einsum(
        "ab,bck,cd->adk",
        gauge.conj().T,
        reduced.h0_mev,
        gauge,
        optimize=True,
    )
    np.testing.assert_allclose(gauged_reduced.h0_mev, expected, atol=1e-13)


def test_matrix_ei_rejects_forged_vertex_source_label() -> None:
    bundle = _toy_bundle()
    altered_phi = bundle.micro_wavefunctions.copy()
    altered_phi[0] = np.einsum("zma,ab->zmb", altered_phi[0], _unitary(4, np.random.default_rng(31)))
    forged_local = local_density_vertices(altered_phi)
    forged = ProjectedFockOperator(
        local_vertices=forged_local,
        green_mev_nm2=np.zeros((bundle.nk, bundle.nk, bundle.nz, bundle.nz)),
        k_weights_nm2=bundle.weights_nm2,
        z_weights_nm=bundle.z_weights_nm,
        bundle_fingerprint=bundle.fingerprint(),
        self_cell_description="forged-label rejection test",
    )
    with pytest.raises(ValueError, match="vertices"):
        solve_reference_subtracted_matrix_ei(bundle, forged)


def test_matrix_ei_rejects_mixed_source_fingerprints() -> None:
    h0 = np.diag([-1.0, -0.5, 0.5, 1.0]).astype(np.complex128)[:, :, None]
    bundle_a = _bundle_from_h0(h0, np.ones(1))
    bundle_b = Kane4Bundle(
        **{**bundle_a.__dict__, "provenance": {"source": "different"}}
    )
    with pytest.raises(ValueError, match="fingerprints"):
        solve_reference_subtracted_matrix_ei(bundle_b, _zero_fock(bundle_a))


def test_two_kramers_copy_scalar_limit_supports_matrix_ei_branch() -> None:
    xi = 0.2
    coupling = 2.0
    bundle = _two_layer_scalar_bundle(xi)
    green = np.zeros((1, 1, 2, 2), dtype=float)
    green[0, 0, 0, 1] = coupling
    green[0, 0, 1, 0] = coupling
    fock = ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="analytic interlayer-only scalar-limit kernel",
    )

    seed = np.zeros_like(bundle.h0_mev)
    seed[:2, 2:, 0] = -0.05 * np.eye(2)
    seed[2:, :2, 0] = seed[:2, 2:, 0].conj().T
    result = solve_reference_subtracted_matrix_ei(
        bundle,
        fock,
        seed_hamiltonian_mev=seed,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.01,
            mixing=0.2,
            precision=2e-10,
            max_iter=600,
            search_mode="seeded_ei",
        ),
    )
    assert result.run.converged
    assert result.physics_mode == "derived_ordinary_electron_hf_bcs"
    assert result.physics_status == "derived_reference_subtracted_ordinary_electron_hf"
    assert result.electrostatic_ensemble == "hartree_disabled_exchange_only"
    assert result.interaction_kind == "exchange_only"
    assert result.interaction_model_label == "normal-ordered exchange-only"
    assert result.interaction_authority == "source_bound_explicit_kernel"
    assert result.ordinary_electron_order is not None
    expected = np.sqrt((coupling / 2.0) ** 2 - xi**2)
    np.testing.assert_allclose(result.excitonic_singular_values_mev[0], expected, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(
        result.ordinary_electron_order.delta_du_singular_values_mev[:, 0],
        2.0 * expected,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        result.ordinary_electron_order.delta_du_eh_mev,
        -2.0 * result.ordinary_electron_order.sigma_fock_mev[:2, 2:],
        atol=1e-13,
    )
    np.testing.assert_allclose(result.ordinary_electron_order.h0_eh_mev, 0.0, atol=1e-13)
    assert result.canonical_free_energy_difference_mev_nm2 < 0.0

    fixed_mu_result = solve_reference_subtracted_matrix_ei(
        bundle,
        fock,
        seed_hamiltonian_mev=seed,
        config=MatrixEIConfig(
            momentum_policy="analytic_or_legacy_diagnostic",
            temperature_K=0.01,
            mixing=0.2,
            precision=2e-10,
            max_iter=600,
            search_mode="seeded_ei",
            constraint_policy="kane_poisson_fixed_mu",
            fixed_mu_mev=0.0,
            normal_ordering_reference_policy="electron_hole_vacuum",
        ),
    )
    assert fixed_mu_result.run.converged
    np.testing.assert_allclose(
        fixed_mu_result.excitonic_singular_values_mev[0],
        expected,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        fixed_mu_result.density_delta,
        result.density_delta,
        rtol=2e-8,
        atol=2e-8,
    )
    assert fixed_mu_result.grand_potential_difference_mev_nm2 < 0.0
