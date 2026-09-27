from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from mean_field.systems.tpt import (
    OpenPatchKeldyshFock,
    RectangularMomentumMesh,
    TPTSoc8HFConfig,
    TPTSoc8Model,
    edge_projected_seed_hamiltonian,
    keldysh_kernel_eV_angstrom2,
    rectangular_keldysh_self_cell_average_eV_angstrom2,
    solve_tpt_soc8_keldysh_hf,
)
from mean_field.systems.tpt import soc8_keldysh_hf as module


def _hermitian_field(rng: np.random.Generator, dimension: int, nk: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension, nk))
    raw = raw + 1j * rng.normal(size=raw.shape)
    return 0.5 * (raw + np.swapaxes(raw.conj(), 0, 1))


def test_rectangular_mesh_contains_gamma_and_has_continuum_weights() -> None:
    mesh = RectangularMomentumMesh.centered_midpoint(
        kx_limit_inv_angstrom=0.03,
        ky_limit_inv_angstrom=0.02,
        nx=5,
        ny=7,
    )
    assert np.min(np.linalg.norm(mesh.k_cart_inv_angstrom, axis=1)) == 0.0
    assert np.isclose(
        np.sum(mesh.weights_inv_angstrom2),
        (0.06 * 0.04) / (2.0 * np.pi) ** 2,
    )
    assert np.isclose(mesh.kx_inv_angstrom[0] - 0.5 * mesh.dkx_inv_angstrom, -0.03)
    assert np.isclose(mesh.ky_inv_angstrom[-1] + 0.5 * mesh.dky_inv_angstrom, 0.02)


def _duffy_rectangle_integral(
    dx: float,
    dy: float,
    *,
    alpha: float,
    kappa: float,
    coulomb: float,
    order: int = 96,
) -> float:
    """Independent tensor-Gauss Duffy quadrature over a centered rectangle."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ww = np.multiply.outer(w, w)
    a = 0.5 * dx
    b = 0.5 * dy
    # Two origin-sharing triangles cover one positive quadrant.  The Duffy
    # Jacobian a*b*u cancels the integrable 1/r singularity.
    q1 = np.hypot(a * uu, b * uu * vv)
    q2 = np.hypot(a * uu * vv, b * uu)
    v1 = keldysh_kernel_eV_angstrom2(
        q1, alpha_2d_angstrom=alpha, kappa=kappa, coulomb_eV_angstrom=coulomb
    )
    v2 = keldysh_kernel_eV_angstrom2(
        q2, alpha_2d_angstrom=alpha, kappa=kappa, coulomb_eV_angstrom=coulomb
    )
    positive_quadrant = float(np.sum(ww * a * b * uu * (v1 + v2)))
    return 4.0 * positive_quadrant


def test_rectangular_self_cell_matches_independent_duffy_quadrature() -> None:
    dx = 0.0031
    dy = 0.0012
    alpha = 17.854
    kappa = 3.25
    coulomb = 14.3996454784255
    actual = rectangular_keldysh_self_cell_average_eV_angstrom2(
        dx,
        dy,
        alpha_2d_angstrom=alpha,
        kappa=kappa,
        coulomb_eV_angstrom=coulomb,
    )
    oracle = _duffy_rectangle_integral(
        dx,
        dy,
        alpha=alpha,
        kappa=kappa,
        coulomb=coulomb,
    ) / (dx * dy)
    assert np.isclose(actual, oracle, rtol=2.0e-12, atol=1.0e-8)


def test_fft_open_convolution_matches_direct_oracle_without_wrapping() -> None:
    mesh = RectangularMomentumMesh.centered_midpoint(
        kx_limit_inv_angstrom=0.02,
        ky_limit_inv_angstrom=0.01,
        nx=3,
        ny=5,
    )
    operator = OpenPatchKeldyshFock(mesh, kappa=4.0)
    density = _hermitian_field(np.random.default_rng(12), 3, mesh.nx * mesh.ny)
    fft_result = operator.self_energy(density)
    direct_result = operator.direct_self_energy(density)
    assert np.allclose(fft_result, direct_result, rtol=2.0e-13, atol=2.0e-13)

    # A corner source must not reappear at the opposite corner through torus
    # wrapping.  The open result uses the largest lag, not a nearest-neighbor lag.
    corner = np.zeros((1, 1, mesh.nx * mesh.ny), dtype=np.complex128)
    corner[0, 0, 0] = 1.0
    result = operator.self_energy(corner)[0, 0].reshape(mesh.nx, mesh.ny)
    kernel = operator.lag_kernel_eV_angstrom2
    weight = mesh.weights_inv_angstrom2[0]
    assert np.isclose(result[-1, -1], -weight * kernel[-1, -1])
    assert not np.isclose(result[-1, -1], -weight * kernel[mesh.nx, mesh.ny])


def test_open_fock_builds_executable_linearity_and_self_adjointness_certificate() -> None:
    mesh = RectangularMomentumMesh.centered_midpoint(
        kx_limit_inv_angstrom=0.01,
        ky_limit_inv_angstrom=0.006,
        nx=3,
        ny=3,
    )
    operator = OpenPatchKeldyshFock(mesh, kappa=2.0)
    functional = operator.certified_functional(dimension=2)
    assert functional.certificate.accepted
    assert functional.certificate.operator_fingerprint == operator.fingerprint
    assert functional.certificate.residuals.maximum_relative_error < 2.0e-10


def test_source_bound_model_loader_and_quadratic_evaluator(tmp_path) -> None:
    theta = np.zeros((8, 8), dtype=np.complex128)
    for start in range(0, 8, 2):
        theta[start, start + 1] = 1.0
        theta[start + 1, start] = -1.0
    pair_energies = np.asarray([-0.8, -0.4, -0.53, -0.49])
    h_gamma = np.diag(np.repeat(pair_energies, 2)).astype(np.complex128)
    pi = np.zeros((2, 8, 8), dtype=np.complex128)
    w = np.zeros((2, 2, 8, 8), dtype=np.complex128)
    for index in range(8):
        w[0, 0, index, index] = 0.5
        w[1, 1, index, index] = -0.25
    path = tmp_path / "MODEL_SOC8.npz"
    np.savez(
        path,
        schema=np.asarray(module.TPT_SOC8_SCHEMA),
        active_ranks_1based=module.TPT_SOC8_ACTIVE_RANKS,
        target_ranks_1based=module.TPT_SOC8_TARGET_RANKS,
        tensor_pair_order_ranks_1based=module.TPT_SOC8_PAIR_ORDER,
        fermi_energy_eV=np.asarray(-0.52),
        lattice_angstrom=np.diag([3.7, 18.6, 24.2]),
        canonical_theta=theta,
        H_Gamma_eV=h_gamma,
        Pi_eV_angstrom=pi,
        Lowdin_W_eV_angstrom2=w,
    )
    digest = sha256(path.read_bytes()).hexdigest()
    model = TPTSoc8Model.from_npz(path, expected_sha256=digest)
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    values = model.hamiltonian(points)
    assert values.shape == (8, 8, 2)
    assert np.allclose(np.diag(values[:, :, 0]), np.repeat(pair_energies, 2) + 0.52)
    expected_shift = 0.5 * (0.5 * 0.2**2 - 0.25 * 0.1**2)
    assert np.allclose(np.diag(values[:, :, 1] - values[:, :, 0]), expected_shift)


def test_edge_projected_seed_is_hermitian_and_has_declared_norm() -> None:
    pair_energies = np.asarray([-0.8, -0.4, -0.53, -0.49])
    h0 = np.repeat(np.diag(np.repeat(pair_energies, 2))[:, :, None], 4, axis=2)
    for channel in (
        "edge_scalar_real",
        "edge_scalar_imaginary",
        "edge_kramers_cross",
        "edge_random_complex",
    ):
        seed, diagnostics = edge_projected_seed_hamiltonian(
            h0,
            channel=channel,
            amplitude_eV=0.003,
            random_seed=7,
        )
        assert np.max(np.abs(seed - np.swapaxes(seed.conj(), 0, 1))) < 1.0e-14
        assert np.allclose(
            [np.linalg.norm(seed[:, :, ik]) for ik in range(seed.shape[2])],
            0.003,
        )
        assert diagnostics["maximum_normalized_seed_norm_error_eV"] < 1.0e-14


def _synthetic_soc8_model() -> TPTSoc8Model:
    theta = np.zeros((8, 8), dtype=np.complex128)
    for start in range(0, 8, 2):
        theta[start, start + 1] = 1.0
        theta[start + 1, start] = -1.0
    pair_energies = np.asarray([-0.8, -0.4, -0.53, -0.49])
    return TPTSoc8Model(
        source_path=Path("synthetic.npz"),
        source_sha256="synthetic",
        fermi_energy_eV=-0.52,
        lattice_angstrom=np.diag([3.7, 18.6, 24.2]),
        theta=theta,
        h_gamma_eV=np.diag(np.repeat(pair_energies, 2)).astype(np.complex128),
        pi_eV_angstrom=np.zeros((2, 8, 8), dtype=np.complex128),
        lowdin_w_eV_angstrom2=np.zeros((2, 2, 8, 8), dtype=np.complex128),
    )


def test_adapter_replays_reference_and_removes_initialization_source() -> None:
    model = _synthetic_soc8_model()
    mesh = RectangularMomentumMesh.centered_midpoint(
        kx_limit_inv_angstrom=0.002,
        ky_limit_inv_angstrom=0.001,
        nx=3,
        ny=3,
    )
    negligible_interaction = OpenPatchKeldyshFock(mesh, kappa=1.0e12)
    normal, _ = solve_tpt_soc8_keldysh_hf(
        model,
        mesh,
        interaction=negligible_interaction,
        config=TPTSoc8HFConfig(mixing=1.0, precision=1.0e-10, max_iter=10),
    )
    assert normal.converged
    assert normal.run.state.diagnostics["final_raw_norm"] <= 1.0e-10
    assert np.max(np.abs(normal.density_delta)) < 1.0e-13

    seeded, _ = solve_tpt_soc8_keldysh_hf(
        model,
        mesh,
        interaction=negligible_interaction,
        config=TPTSoc8HFConfig(
            mixing=1.0,
            precision=1.0e-10,
            max_iter=10,
            seed_channel="edge_scalar_imaginary",
            seed_amplitude_eV=0.001,
        ),
    )
    assert seeded.converged
    assert seeded.run.state.diagnostics["final_raw_norm"] <= 1.0e-10
    # The 1 meV source initializes P only; the reported physical Hamiltonian
    # contains H0 plus the tiny source-free kappa=1e12 Fock field.
    h0 = model.hamiltonian(mesh.k_cart_inv_angstrom)
    assert np.allclose(
        seeded.hamiltonian - h0,
        seeded.interaction_hamiltonian,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.max(np.abs(seeded.interaction_hamiltonian)) < 1.0e-10
    mean_occupation = float(
        np.einsum(
            "k,aak->", mesh.weights_inv_angstrom2, seeded.total_density
        ).real
        / np.sum(mesh.weights_inv_angstrom2)
    )
    assert np.isclose(mean_occupation, 4.0, rtol=0.0, atol=1.0e-11)
