"""Layered-dielectric Coulomb Green functions on a uniform z grid.

Three boundary families are explicit:

- ``periodic_*`` is the repeated-supercell finite-q operator. Its uniform z
  mode behaves as 1/q^2, so its momentum self cell is logarithmically divergent
  and it must not be used as a projected-exchange mesh kernel.
- ``open_*`` uses discrete transparent boundaries into semi-infinite exterior
  dielectrics. Its finite-q kernel has the physical 1/q infrared behaviour and
  admits a finite circular momentum-cell average. The q=0 neutral limit gives
  the corresponding zero-exterior-field Hartree problem.
- ``one_sided_gated_*`` places a homogeneous Dirichlet metal plane half a
  grid cell beyond one endpoint and a transparent semi-infinite dielectric at
  the other. Its q=0 response is finite and can accept nonneutral charge by
  balancing it with induced gate charge.

These are response kernels. A nonzero applied gate potential belongs in the
one-body Kane-Poisson source Hamiltonian.
"""

from __future__ import annotations

import hashlib

import numpy as np


COULOMB_MEV_NM = 1439.96448


def layered_electrostatics_fingerprint(
    z_nm: np.ndarray,
    z_weights_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    boundary_family: str,
    epsilon_left: float | None = None,
    epsilon_right: float | None = None,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> str:
    """Fingerprint a declared layered electrostatic response problem."""

    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    weights = np.asarray(z_weights_nm, dtype=float)
    if weights.shape != z.shape or not np.allclose(weights, dz, rtol=0.0, atol=1e-12):
        raise ValueError("electrostatic z weights must equal the uniform grid spacing")
    supported_boundaries = {
        "periodic_zero_mean",
        "open_zero_field",
        "one_sided_gate_left",
        "one_sided_gate_right",
    }
    if boundary_family not in supported_boundaries:
        raise ValueError("unsupported layered electrostatic boundary family")
    eps_left = float(epsilon[0] if epsilon_left is None else epsilon_left)
    eps_right = float(epsilon[-1] if epsilon_right is None else epsilon_right)
    if boundary_family == "open_zero_field" and (
        not np.isclose(eps_left, epsilon[0], rtol=0.0, atol=1e-12)
        or not np.isclose(eps_right, epsilon[-1], rtol=0.0, atol=1e-12)
    ):
        raise ValueError("open electrostatic fingerprint requires homogeneous endpoint plateaus")
    if boundary_family == "one_sided_gate_left" and not np.isclose(
        eps_right, epsilon[-1], rtol=0.0, atol=1e-12
    ):
        raise ValueError("right-open one-sided gate requires a homogeneous right endpoint plateau")
    if boundary_family == "one_sided_gate_right" and not np.isclose(
        eps_left, epsilon[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError("left-open one-sided gate requires a homogeneous left endpoint plateau")
    if boundary_family in {"open_zero_field", "one_sided_gate_right"}:
        eps_left = float(epsilon[0])
    if boundary_family in {"open_zero_field", "one_sided_gate_left"}:
        eps_right = float(epsilon[-1])
    digest = hashlib.sha256()
    digest.update(b"layered-electrostatics-v2")
    digest.update(boundary_family.encode())
    digest.update(b"node-centered-harmonic-bonds")
    if boundary_family.startswith("one_sided_gate_"):
        digest.update(b"half-cell-dirichlet-metal-gate-opposite-transparent")
    else:
        digest.update(b"no-metallic-gates")
    digest.update(np.float64(eps_left).tobytes())
    digest.update(np.float64(eps_right).tobytes())
    digest.update(np.float64(coulomb_mev_nm).tobytes())
    for array in (z, weights, epsilon):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _validated_grid(z_nm: np.ndarray, epsilon_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.asarray(z_nm, dtype=float)
    epsilon = np.asarray(epsilon_r, dtype=float)
    if z.ndim != 1 or z.size < 4 or epsilon.shape != z.shape:
        raise ValueError("z and epsilon must be matching one-dimensional arrays")
    spacing = np.diff(z)
    if spacing.size == 0 or spacing[0] <= 0.0 or not np.allclose(
        spacing, spacing[0], rtol=0.0, atol=1e-12
    ):
        raise ValueError("layered Green function requires a strictly increasing uniform z grid")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(epsilon)) or np.any(epsilon <= 0.0):
        raise ValueError("z and epsilon_r must be finite, with positive epsilon_r")
    return z, epsilon, float(spacing[0])


def _interface_dielectric(epsilon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    epsilon_plus = 2.0 * epsilon * np.roll(epsilon, -1) / (
        epsilon + np.roll(epsilon, -1)
    )
    return epsilon_plus, np.roll(epsilon_plus, 1)


def periodic_layered_poisson_matrix_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
) -> np.ndarray:
    """Return the positive finite-q periodic Poisson operator in nm^-2."""

    q = float(q_nm_inv)
    if q <= 0.0 or not np.isfinite(q):
        raise ValueError("q_nm_inv must be finite and positive")
    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    epsilon_plus, epsilon_minus = _interface_dielectric(epsilon)
    n = z.size
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i, i] = (
            (epsilon_plus[i] + epsilon_minus[i]) / dz**2
            + epsilon[i] * q**2
        )
        matrix[i, (i + 1) % n] = -epsilon_plus[i] / dz**2
        matrix[i, (i - 1) % n] = -epsilon_minus[i] / dz**2
    return matrix


def periodic_layered_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Solve the finite-q repeated-supercell layered Green matrix."""

    z, _epsilon, dz = _validated_grid(z_nm, epsilon_r)
    matrix = periodic_layered_poisson_matrix_nm2(q_nm_inv, z, epsilon_r)
    source = (4.0 * np.pi * float(coulomb_mev_nm) / dz) * np.eye(z.size)
    green = np.linalg.solve(matrix, source)
    return 0.5 * (green + green.T)


def uniform_periodic_discrete_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Exact Fourier result for the uniform periodic finite-difference grid."""

    q = float(q_nm_inv)
    if q <= 0.0 or epsilon_r <= 0.0:
        raise ValueError("q_nm_inv and epsilon_r must be positive")
    z, _epsilon, dz = _validated_grid(z_nm, np.full(np.asarray(z_nm).shape, epsilon_r))
    n = z.size
    modes = 2.0 * np.pi * np.fft.fftfreq(n, d=dz)
    denominator = float(epsilon_r) * (
        q**2 + 4.0 * np.sin(0.5 * modes * dz) ** 2 / dz**2
    )
    first_column = (
        4.0 * np.pi * float(coulomb_mev_nm) / dz
        * np.fft.ifft(1.0 / denominator).real
    )
    indices = np.arange(n)
    return first_column[(indices[:, None] - indices[None, :]) % n]


def _discrete_decay_parameters(q_nm_inv: float, dz_nm: float) -> tuple[float, float]:
    scaled_q = float(q_nm_inv) * float(dz_nm)
    discriminant = scaled_q * np.sqrt(4.0 + scaled_q**2)
    denominator = 2.0 + scaled_q**2 + discriminant
    decay = 2.0 / denominator
    one_minus_decay = (scaled_q**2 + discriminant) / denominator
    return float(decay), float(one_minus_decay)


def open_layered_poisson_matrix_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    epsilon_left: float | None = None,
    epsilon_right: float | None = None,
) -> np.ndarray:
    """Finite-q operator with discrete transparent exterior boundaries."""

    q = float(q_nm_inv)
    if q <= 0.0 or not np.isfinite(q):
        raise ValueError("q_nm_inv must be finite and positive")
    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    eps_left = float(epsilon[0] if epsilon_left is None else epsilon_left)
    eps_right = float(epsilon[-1] if epsilon_right is None else epsilon_right)
    if eps_left <= 0.0 or eps_right <= 0.0:
        raise ValueError("exterior dielectric constants must be positive")
    if not np.isclose(eps_left, epsilon[0], rtol=0.0, atol=1e-12) or not np.isclose(
        eps_right, epsilon[-1], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "the current transparent-boundary discretization requires exterior "
            "dielectrics equal to the endpoint homogeneous plateaus"
        )
    eps_left = float(epsilon[0])
    eps_right = float(epsilon[-1])
    epsilon_plus, epsilon_minus = _interface_dielectric(epsilon)
    n = z.size
    matrix = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        matrix[i, i] = (
            (epsilon_plus[i] + epsilon_minus[i]) / dz**2
            + epsilon[i] * q**2
        )
        matrix[i, i + 1] = -epsilon_plus[i] / dz**2
        matrix[i, i - 1] = -epsilon_minus[i] / dz**2
    _decay, one_minus_decay = _discrete_decay_parameters(q, dz)
    matrix[0, 0] = (
        epsilon_plus[0] / dz**2
        + eps_left * one_minus_decay / dz**2
        + epsilon[0] * q**2
    )
    matrix[0, 1] = -epsilon_plus[0] / dz**2
    matrix[-1, -1] = (
        epsilon_minus[-1] / dz**2
        + eps_right * one_minus_decay / dz**2
        + epsilon[-1] * q**2
    )
    matrix[-1, -2] = -epsilon_minus[-1] / dz**2
    return matrix


def open_layered_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    epsilon_left: float | None = None,
    epsilon_right: float | None = None,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Solve the open-boundary finite-q layered Coulomb Green matrix."""

    z, _epsilon, dz = _validated_grid(z_nm, epsilon_r)
    matrix = open_layered_poisson_matrix_nm2(
        q_nm_inv,
        z,
        epsilon_r,
        epsilon_left=epsilon_left,
        epsilon_right=epsilon_right,
    )
    source = (4.0 * np.pi * float(coulomb_mev_nm) / dz) * np.eye(z.size)
    green_raw = np.linalg.solve(matrix, source)
    residual = matrix @ green_raw - source
    residual_scale = max(1.0, float(np.max(np.abs(source))))
    if float(np.max(np.abs(residual))) > 1e-7 * residual_scale:
        raise ValueError("open layered Green solve failed its operator residual")
    reciprocity_error = float(np.max(np.abs(green_raw - green_raw.T)))
    if reciprocity_error > 1e-8 * max(1.0, float(np.max(np.abs(green_raw)))):
        raise ValueError("open layered Green solve failed reciprocity before symmetrization")
    green = 0.5 * (green_raw + green_raw.T)
    if not np.all(np.isfinite(green)):
        raise ValueError("open layered Green matrix is nonfinite")
    return green


def uniform_open_discrete_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Exact infinite-grid Green matrix for a uniform discrete z operator."""

    q = float(q_nm_inv)
    if q <= 0.0 or epsilon_r <= 0.0:
        raise ValueError("q_nm_inv and epsilon_r must be positive")
    z, _epsilon, dz = _validated_grid(z_nm, np.full(np.asarray(z_nm).shape, epsilon_r))
    scaled_q = q * dz
    discriminant = scaled_q * np.sqrt(4.0 + scaled_q**2)
    decay = 2.0 / (2.0 + scaled_q**2 + discriminant)
    separation = np.abs(np.arange(z.size)[:, None] - np.arange(z.size)[None, :])
    return (
        4.0 * np.pi * float(coulomb_mev_nm) * dz
        / (float(epsilon_r) * discriminant)
        * decay**separation
    )


def _validated_gate_side(gate_side: str) -> str:
    side = str(gate_side).lower()
    if side not in {"left", "right"}:
        raise ValueError("gate_side must be 'left' or 'right'")
    return side


def one_sided_gated_layered_poisson_matrix_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    gate_side: str,
    epsilon_open: float | None = None,
) -> np.ndarray:
    """Layered Poisson operator with one metal gate and one open exterior.

    Grid points are cell centers. The grounded response gate lies half a grid
    spacing outside the endpoint on ``gate_side``; the opposite endpoint is a
    discrete transparent boundary into a homogeneous semi-infinite dielectric.
    ``q_nm_inv=0`` is allowed because the Dirichlet gate removes the constant
    null mode.
    """

    q = float(q_nm_inv)
    if q < 0.0 or not np.isfinite(q):
        raise ValueError("q_nm_inv must be finite and nonnegative")
    side = _validated_gate_side(gate_side)
    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    epsilon_plus, epsilon_minus = _interface_dielectric(epsilon)
    open_endpoint = epsilon[-1] if side == "left" else epsilon[0]
    eps_open = float(open_endpoint if epsilon_open is None else epsilon_open)
    if eps_open <= 0.0 or not np.isclose(
        eps_open, open_endpoint, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "the open exterior dielectric must equal its homogeneous endpoint plateau"
        )
    eps_open = float(open_endpoint)

    n = z.size
    matrix = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        matrix[i, i] = (
            (epsilon_plus[i] + epsilon_minus[i]) / dz**2
            + epsilon[i] * q**2
        )
        matrix[i, i + 1] = -epsilon_plus[i] / dz**2
        matrix[i, i - 1] = -epsilon_minus[i] / dz**2
    _decay, one_minus_decay = _discrete_decay_parameters(q, dz)
    if side == "left":
        matrix[0, 0] = (
            epsilon_plus[0] / dz**2
            + 2.0 * epsilon[0] / dz**2
            + epsilon[0] * q**2
        )
        matrix[0, 1] = -epsilon_plus[0] / dz**2
        matrix[-1, -1] = (
            epsilon_minus[-1] / dz**2
            + eps_open * one_minus_decay / dz**2
            + epsilon[-1] * q**2
        )
        matrix[-1, -2] = -epsilon_minus[-1] / dz**2
    else:
        matrix[0, 0] = (
            epsilon_plus[0] / dz**2
            + eps_open * one_minus_decay / dz**2
            + epsilon[0] * q**2
        )
        matrix[0, 1] = -epsilon_plus[0] / dz**2
        matrix[-1, -1] = (
            epsilon_minus[-1] / dz**2
            + 2.0 * epsilon[-1] / dz**2
            + epsilon[-1] * q**2
        )
        matrix[-1, -2] = -epsilon_minus[-1] / dz**2
    return matrix


def one_sided_gated_layered_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    gate_side: str,
    epsilon_open: float | None = None,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Solve the one-sided-gate/open layered Coulomb response Green matrix."""

    z, _epsilon, dz = _validated_grid(z_nm, epsilon_r)
    matrix = one_sided_gated_layered_poisson_matrix_nm2(
        q_nm_inv,
        z,
        epsilon_r,
        gate_side=gate_side,
        epsilon_open=epsilon_open,
    )
    source = (4.0 * np.pi * float(coulomb_mev_nm) / dz) * np.eye(z.size)
    green_raw = np.linalg.solve(matrix, source)
    residual = matrix @ green_raw - source
    residual_scale = max(1.0, float(np.max(np.abs(source))))
    if float(np.max(np.abs(residual))) > 1e-7 * residual_scale:
        raise ValueError("one-sided gated Green solve failed its operator residual")
    reciprocity_error = float(np.max(np.abs(green_raw - green_raw.T)))
    if reciprocity_error > 1e-8 * max(1.0, float(np.max(np.abs(green_raw)))):
        raise ValueError("one-sided gated Green solve failed reciprocity")
    green = 0.5 * (green_raw + green_raw.T)
    if not np.all(np.isfinite(green)):
        raise ValueError("one-sided gated Green matrix is nonfinite")
    return green


def uniform_one_sided_gated_discrete_green_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    *,
    epsilon_r: float,
    gate_side: str,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Exact uniform discrete image-charge oracle for one gate and one open side."""

    q = float(q_nm_inv)
    if q < 0.0 or epsilon_r <= 0.0:
        raise ValueError("q_nm_inv must be nonnegative and epsilon_r positive")
    side = _validated_gate_side(gate_side)
    z, _epsilon, dz = _validated_grid(
        z_nm, np.full(np.asarray(z_nm).shape, epsilon_r)
    )
    index = np.arange(z.size)
    if side == "right":
        index = index[::-1]
    i = index[:, None]
    j = index[None, :]
    if q == 0.0:
        return (
            4.0
            * np.pi
            * float(coulomb_mev_nm)
            * dz
            / float(epsilon_r)
            * (np.minimum(i, j) + 0.5)
        )
    scaled_q = q * dz
    discriminant = scaled_q * np.sqrt(4.0 + scaled_q**2)
    decay, _one_minus_decay = _discrete_decay_parameters(q, dz)
    direct_power = np.abs(i - j)
    image_power = i + j + 1
    log_decay = np.log(decay)
    image_difference = (
        decay**direct_power
        * -np.expm1((image_power - direct_power) * log_decay)
    )
    return (
        4.0
        * np.pi
        * float(coulomb_mev_nm)
        * dz
        / (float(epsilon_r) * discriminant)
        * image_difference
    )


def one_sided_gated_poisson_electron_energy_mev(
    z_nm: np.ndarray,
    electron_density_nm3: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    gate_side: str,
    gate_electron_energy_mev: float = 0.0,
    epsilon_open: float | None = None,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Solve q=0 Poisson with fixed gate energy and zero field at the open side.

    ``gate_electron_energy_mev`` is the electron potential energy at the metal
    plane, not a voltage. For an electrostatic voltage ``V`` in volts the
    electron energy is ``-1000*V`` meV. The response Hartree operator uses zero.
    Nonneutral density is allowed because induced metal charge closes Gauss law.
    """

    side = _validated_gate_side(gate_side)
    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    density = np.asarray(electron_density_nm3, dtype=float)
    if density.shape != z.shape or not np.all(np.isfinite(density)):
        raise ValueError("electron density must be finite and match z")
    gate_energy = float(gate_electron_energy_mev)
    if not np.isfinite(gate_energy):
        raise ValueError("gate_electron_energy_mev must be finite")
    matrix = one_sided_gated_layered_poisson_matrix_nm2(
        0.0,
        z,
        epsilon,
        gate_side=side,
        epsilon_open=epsilon_open,
    )
    source = 4.0 * np.pi * float(coulomb_mev_nm) * density
    if side == "left":
        source[0] += 2.0 * epsilon[0] * gate_energy / dz**2
    else:
        source[-1] += 2.0 * epsilon[-1] * gate_energy / dz**2
    potential = np.linalg.solve(matrix, source)
    if not np.all(np.isfinite(potential)):
        raise ValueError("one-sided gated Poisson potential is nonfinite")
    return np.asarray(potential, dtype=float)


def open_layered_poisson_electron_energy_mev(
    z_nm: np.ndarray,
    electron_density_delta_nm3: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Solve the neutral q=0 open problem with zero exterior fields."""

    z, epsilon, dz = _validated_grid(z_nm, epsilon_r)
    density = np.asarray(electron_density_delta_nm3, dtype=float)
    if density.shape != z.shape or not np.all(np.isfinite(density)):
        raise ValueError("electron density must be finite and match z")
    integrated = float(np.sum(density) * dz)
    scale = max(1.0, float(np.sum(np.abs(density)) * dz))
    if abs(integrated) > 1e-9 * scale:
        raise ValueError(f"open Poisson source is not neutral: {integrated:.3e} nm^-2")
    density = density - np.mean(density)
    epsilon_plus, epsilon_minus = _interface_dielectric(epsilon)
    n = z.size
    matrix = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        matrix[i, i] = (epsilon_plus[i] + epsilon_minus[i]) / dz**2
        matrix[i, i + 1] = -epsilon_plus[i] / dz**2
        matrix[i, i - 1] = -epsilon_minus[i] / dz**2
    matrix[0, 0] = epsilon_plus[0] / dz**2
    matrix[0, 1] = -epsilon_plus[0] / dz**2
    matrix[-1, -1] = epsilon_minus[-1] / dz**2
    matrix[-1, -2] = -epsilon_minus[-1] / dz**2
    source = 4.0 * np.pi * float(coulomb_mev_nm) * density
    matrix[-1, :] = 1.0 / n
    source[-1] = 0.0
    potential = np.linalg.solve(matrix, source)
    potential -= np.mean(potential)
    return potential


def open_layered_green_on_mesh_mev_nm2(
    k_cart_nm_inv: np.ndarray,
    k_weights_nm2: np.ndarray,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    epsilon_left: float | None = None,
    epsilon_right: float | None = None,
    self_cell_quadrature_order: int = 12,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Build the open layered kernel with circular momentum self cells."""

    k = np.asarray(k_cart_nm_inv, dtype=float)
    weights = np.asarray(k_weights_nm2, dtype=float)
    z, epsilon, _dz = _validated_grid(z_nm, epsilon_r)
    if k.ndim != 2 or k.shape[1] != 2 or weights.shape != (k.shape[0],):
        raise ValueError("k mesh must have shape (nk,2) with matching weights")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("k mesh and positive weights must be finite")
    order = int(self_cell_quadrature_order)
    if order < 4:
        raise ValueError("self_cell_quadrature_order must be at least four")

    q_matrix = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=2)
    nk, nz = k.shape[0], z.size
    green = np.empty((nk, nk, nz, nz), dtype=float)
    cache: dict[float, np.ndarray] = {}

    def green_at(q: float) -> np.ndarray:
        key = float(q)
        if key <= 0.0:
            raise ValueError("off-diagonal momentum points must be distinct")
        if key not in cache:
            cache[key] = open_layered_green_mev_nm2(
                key,
                z,
                epsilon,
                epsilon_left=epsilon_left,
                epsilon_right=epsilon_right,
                coulomb_mev_nm=coulomb_mev_nm,
            )
        return cache[key]

    for ik in range(nk):
        for ip in range(ik + 1, nk):
            block = green_at(q_matrix[ik, ip])
            green[ik, ip] = block
            green[ip, ik] = block.T

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(order)
    for ik, weight in enumerate(weights):
        q_cell = np.sqrt((2.0 * np.pi) ** 2 * float(weight) / np.pi)
        q_nodes = 0.5 * q_cell * (nodes + 1.0)
        q_weights = 0.5 * q_cell * quadrature_weights
        averaged = np.zeros((nz, nz), dtype=float)
        for q, wq in zip(q_nodes, q_weights):
            averaged += float(wq * q) * green_at(float(q))
        green[ik, ik] = (2.0 / q_cell**2) * averaged

    reciprocity = np.swapaxes(np.swapaxes(green, 0, 1), 2, 3)
    error = float(np.max(np.abs(green - reciprocity)))
    if error > 1e-8:
        raise ValueError(f"open layered Green function violates reciprocity: {error:.3e}")
    return green


def one_sided_gated_layered_green_on_mesh_mev_nm2(
    k_cart_nm_inv: np.ndarray,
    k_weights_nm2: np.ndarray,
    z_nm: np.ndarray,
    epsilon_r: np.ndarray,
    *,
    gate_side: str,
    epsilon_open: float | None = None,
    self_cell_quadrature_order: int = 12,
    coulomb_mev_nm: float = COULOMB_MEV_NM,
) -> np.ndarray:
    """Build the one-gate/open layered kernel with circular momentum cells."""

    side = _validated_gate_side(gate_side)
    k = np.asarray(k_cart_nm_inv, dtype=float)
    weights = np.asarray(k_weights_nm2, dtype=float)
    z, epsilon, _dz = _validated_grid(z_nm, epsilon_r)
    if k.ndim != 2 or k.shape[1] != 2 or weights.shape != (k.shape[0],):
        raise ValueError("k mesh must have shape (nk,2) with matching weights")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(weights)) or np.any(
        weights <= 0.0
    ):
        raise ValueError("k mesh and positive weights must be finite")
    order = int(self_cell_quadrature_order)
    if order < 4:
        raise ValueError("self_cell_quadrature_order must be at least four")

    q_matrix = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=2)
    nk, nz = k.shape[0], z.size
    green = np.empty((nk, nk, nz, nz), dtype=float)
    cache: dict[float, np.ndarray] = {}

    def green_at(q: float) -> np.ndarray:
        key = float(q)
        if key < 0.0:
            raise ValueError("momentum transfer cannot be negative")
        if key not in cache:
            cache[key] = one_sided_gated_layered_green_mev_nm2(
                key,
                z,
                epsilon,
                gate_side=side,
                epsilon_open=epsilon_open,
                coulomb_mev_nm=coulomb_mev_nm,
            )
        return cache[key]

    for ik in range(nk):
        for ip in range(ik + 1, nk):
            if q_matrix[ik, ip] == 0.0:
                raise ValueError("off-diagonal momentum points must be distinct")
            block = green_at(q_matrix[ik, ip])
            green[ik, ip] = block
            green[ip, ik] = block.T

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(order)
    for ik, weight in enumerate(weights):
        q_cell = np.sqrt((2.0 * np.pi) ** 2 * float(weight) / np.pi)
        q_nodes = 0.5 * q_cell * (nodes + 1.0)
        q_weights = 0.5 * q_cell * quadrature_weights
        averaged = np.zeros((nz, nz), dtype=float)
        for q, wq in zip(q_nodes, q_weights):
            averaged += float(wq * q) * green_at(float(q))
        green[ik, ik] = (2.0 / q_cell**2) * averaged

    reciprocity = np.swapaxes(np.swapaxes(green, 0, 1), 2, 3)
    error = float(np.max(np.abs(green - reciprocity)))
    if error > 1e-8:
        raise ValueError(
            f"one-sided gated layered Green function violates reciprocity: {error:.3e}"
        )
    return green
