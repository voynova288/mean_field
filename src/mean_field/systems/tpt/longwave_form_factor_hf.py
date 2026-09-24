"""Complete-Wannier-space TPT HF under a declared monopole density vertex.

The model uses ``Lambda_W(q) = I`` in the complete orthonormal Wannier space.
This is a phenomenological full-BZ extrapolation of a long-wavelength vertex,
not a microscopic TPT interaction.  Working in Wannier space is exactly
basis-equivalent to band-space form factors ``U_m(k)^dagger U_n(k')`` while
avoiding a gauge-dependent band truncation.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mean_field.core.hf.engine import DensityUpdateResult

COULOMB_EV_ANGSTROM = 14.3996454784255


def hermitize_last_two(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.complex128)
    return 0.5 * (values + values.conj().swapaxes(-1, -2))


def rectangular_coulomb_self_cell_average(
    dx: float,
    dy: float,
    *,
    coulomb_ev_angstrom: float = COULOMB_EV_ANGSTROM,
) -> float:
    """Average ``2*pi*C/q`` over a centered rectangular q cell."""
    dx_value, dy_value = float(dx), float(dy)
    if dx_value <= 0.0 or dy_value <= 0.0:
        raise ValueError("cell widths must be positive")
    a, b = 0.5 * dx_value, 0.5 * dy_value
    integral_one_over_q = 4.0 * (
        a * np.arcsinh(b / a) + b * np.arcsinh(a / b)
    )
    return float(
        2.0
        * np.pi
        * float(coulomb_ev_angstrom)
        * integral_one_over_q
        / (dx_value * dy_value)
    )


def build_periodic_scalar_2d_coulomb_kernel(
    mesh_shape: tuple[int, int],
    reciprocal_vectors_angstrom_inv: np.ndarray,
    *,
    coulomb_ev_angstrom: float = COULOMB_EV_ANGSTROM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``V[(k-p) mod mesh]`` and its minimum-image q norms.

    The analytic self cell is rectangular, so the two reciprocal vectors must
    be orthogonal.  Nonorthogonal cells require a different singular-cell
    integral and are rejected rather than approximated silently.
    """
    n1, n2 = (int(mesh_shape[0]), int(mesh_shape[1]))
    reciprocal = np.asarray(reciprocal_vectors_angstrom_inv, dtype=float)
    if n1 <= 1 or n2 <= 1 or reciprocal.shape != (2, 2):
        raise ValueError("mesh and reciprocal-vector shapes are invalid")
    dot = float(np.dot(reciprocal[0], reciprocal[1]))
    scale = float(np.linalg.norm(reciprocal[0]) * np.linalg.norm(reciprocal[1]))
    if abs(dot) > 1.0e-12 * scale:
        raise ValueError("rectangular self-cell integration requires orthogonal reciprocal vectors")
    q_vectors = np.empty((n1, n2, 2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            candidates = []
            for wrap_i in (-1, 0, 1):
                for wrap_j in (-1, 0, 1):
                    coefficient_i = i + wrap_i * n1
                    coefficient_j = j + wrap_j * n2
                    candidates.append(
                        coefficient_i * reciprocal[0] / n1
                        + coefficient_j * reciprocal[1] / n2
                    )
            norms = np.asarray([np.linalg.norm(vector) for vector in candidates])
            q_vectors[i, j] = candidates[int(np.argmin(norms))]
    q_norm = np.linalg.norm(q_vectors, axis=2)
    kernel = np.empty_like(q_norm)
    nonzero = q_norm > 1.0e-14
    kernel[nonzero] = 2.0 * np.pi * float(coulomb_ev_angstrom) / q_norm[nonzero]
    dx = float(np.linalg.norm(reciprocal[0]) / n1)
    dy = float(np.linalg.norm(reciprocal[1]) / n2)
    kernel[~nonzero] = rectangular_coulomb_self_cell_average(
        dx,
        dy,
        coulomb_ev_angstrom=coulomb_ev_angstrom,
    )
    # For even meshes, opposite Nyquist representatives can be degenerate.
    # The scalar kernel must nevertheless satisfy V(d)=V(-d).
    inverse_i = (-np.arange(n1)) % n1
    inverse_j = (-np.arange(n2)) % n2
    inverse_kernel = kernel[np.ix_(inverse_i, inverse_j)]
    kernel = 0.5 * (kernel + inverse_kernel)
    return kernel, q_norm


def reconstruct_complete_wannier_h0_and_reference(
    eigenvalues_ev: np.ndarray,
    eigenvectors_ket: np.ndarray,
    *,
    occupied_bands: int,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``H0`` and its occupied projector in the complete basis.

    Returns ``H0`` in repository ``(a,b,k)`` Hamiltonian orientation and
    ``P0`` in ket-projector ``(k,a,b)`` orientation.
    """
    energies = np.asarray(eigenvalues_ev, dtype=float)
    vectors = np.asarray(eigenvectors_ket, dtype=np.complex128)
    if energies.ndim != 2 or vectors.shape != (energies.shape[0], energies.shape[1], energies.shape[1]):
        raise ValueError("eigenvalue/eigenvector shape mismatch")
    nk, n_wannier = energies.shape
    n_occupied = int(occupied_bands)
    if not (0 < n_occupied < n_wannier):
        raise ValueError("occupied_bands must lie inside the complete basis")
    h0_ket = np.empty((nk, n_wannier, n_wannier), dtype=np.complex128)
    reference_ket = np.empty_like(h0_ket)

    def reconstruct(index: int):
        states = vectors[index]
        hamiltonian = (states * energies[index][None, :]) @ states.conj().T
        occupied = states[:, :n_occupied]
        reference = occupied @ occupied.conj().T
        return index, hamiltonian, reference

    iterator = map(reconstruct, range(nk)) if executor is None else executor.map(reconstruct, range(nk))
    for index, hamiltonian, reference in iterator:
        h0_ket[index] = hamiltonian
        reference_ket[index] = reference
    return h0_ket.transpose(1, 2, 0), reference_ket


def apply_periodic_wannier_fock_fft(
    density_stored: np.ndarray,
    coulomb_kernel_eps1: np.ndarray,
    *,
    mesh_shape: tuple[int, int],
    quadrature_weight_angstrom_minus2: float,
    epsilon: float,
    workers: int = 1,
) -> np.ndarray:
    """Apply ``-w sum_p V(k-p) D_W(p)`` by periodic FFT convolution."""
    from scipy import fft

    density = np.asarray(density_stored, dtype=np.complex128)
    n_wannier, second, nk = density.shape
    n1, n2 = int(mesh_shape[0]), int(mesh_shape[1])
    if second != n_wannier or nk != n1 * n2:
        raise ValueError("stored density and mesh shapes are inconsistent")
    kernel = np.asarray(coulomb_kernel_eps1, dtype=float)
    if kernel.shape != (n1, n2):
        raise ValueError("Coulomb kernel shape mismatch")
    epsilon_value = float(epsilon)
    if epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive")
    density_ket = np.swapaxes(density, 0, 1).transpose(2, 0, 1).reshape(n1, n2, n_wannier, n_wannier)
    transformed_density = fft.fftn(density_ket, axes=(0, 1), workers=int(workers))
    transformed_kernel = fft.fftn(kernel, axes=(0, 1), workers=int(workers))
    convolution = fft.ifftn(
        transformed_density * transformed_kernel[:, :, None, None],
        axes=(0, 1),
        workers=int(workers),
    )
    sigma_ket = (
        -float(quadrature_weight_angstrom_minus2)
        * convolution.reshape(nk, n_wannier, n_wannier)
        / epsilon_value
    )
    return sigma_ket.transpose(1, 2, 0)


def direct_periodic_wannier_fock_elements(
    density_ket_grid: np.ndarray,
    coulomb_kernel_eps1: np.ndarray,
    targets: list[tuple[int, int, int, int]],
    *,
    quadrature_weight_angstrom_minus2: float,
    epsilon: float,
) -> np.ndarray:
    """Direct selected-element oracle for the periodic FFT contraction."""
    density = np.asarray(density_ket_grid, dtype=np.complex128)
    kernel = np.asarray(coulomb_kernel_eps1, dtype=float)
    if density.ndim != 4 or kernel.shape != density.shape[:2]:
        raise ValueError("density grid and kernel shapes are inconsistent")
    n1, n2 = kernel.shape
    values = []
    for target_i, target_j, orbital_a, orbital_b in targets:
        total = 0.0j
        for source_i in range(n1):
            for source_j in range(n2):
                delta_i = (target_i - source_i) % n1
                delta_j = (target_j - source_j) % n2
                total += kernel[delta_i, delta_j] * density[source_i, source_j, orbital_a, orbital_b]
        values.append(
            -float(quadrature_weight_angstrom_minus2) * total / float(epsilon)
        )
    return np.asarray(values, dtype=np.complex128)


def reference_relative_energy(
    interaction_h: np.ndarray,
    h0: np.ndarray,
    density_stored: np.ndarray,
) -> float:
    """ODA-consistent average energy per represented k point."""
    density = np.asarray(density_stored, dtype=np.complex128)
    nk = density.shape[2]
    one_body = np.einsum("abk,abk->", density, h0, optimize=True)
    exchange = np.einsum("abk,abk->", density, interaction_h, optimize=True)
    return float((one_body + 0.5 * exchange).real / nk)


def build_fixed_rank_density_builder(
    reference_projector_ket: np.ndarray,
    *,
    occupied_bands: int,
    executor: ThreadPoolExecutor | None = None,
    direct_gap_tolerance_ev: float = 1.0e-11,
) -> Callable[[np.ndarray], DensityUpdateResult]:
    """Build a fixed-rank-per-k zero-temperature density update.

    A nonpositive direct rank boundary is rejected.  The indirect gap is
    reported separately so callers can reject a fixed-rank insulating ansatz
    if it becomes metallic.
    """
    reference = np.asarray(reference_projector_ket, dtype=np.complex128)
    if reference.ndim != 3 or reference.shape[1] != reference.shape[2]:
        raise ValueError("reference projector must have shape (nk,nw,nw)")
    nk, n_wannier, _ = reference.shape
    n_occupied = int(occupied_bands)
    if not (0 < n_occupied < n_wannier):
        raise ValueError("occupied_bands must lie inside the represented spectrum")

    def density_update(hamiltonian_abk: np.ndarray) -> DensityUpdateResult:
        h_ket = np.asarray(hamiltonian_abk, dtype=np.complex128).transpose(2, 0, 1)
        if h_ket.shape != (nk, n_wannier, n_wannier):
            raise ValueError(f"Hamiltonian shape mismatch: {h_ket.shape}")
        energies = np.empty((nk, n_wannier), dtype=np.float64)
        projector = np.empty_like(h_ket)

        def diagonalize_and_project(index: int):
            values, states = np.linalg.eigh(hermitize_last_two(h_ket[index]))
            occupied = states[:, :n_occupied]
            return index, values, occupied @ occupied.conj().T

        iterator = (
            map(diagonalize_and_project, range(nk))
            if executor is None
            else executor.map(diagonalize_and_project, range(nk))
        )
        for index, values, occupied_projector in iterator:
            energies[index] = values
            projector[index] = occupied_projector
        direct_gaps = energies[:, n_occupied] - energies[:, n_occupied - 1]
        minimum_direct_gap = float(np.min(direct_gaps))
        if minimum_direct_gap <= float(direct_gap_tolerance_ev):
            raise RuntimeError(
                f"closed fixed-rank occupation boundary: minimum direct gap={minimum_direct_gap:.16e} eV"
            )
        d_ket = projector - reference
        d_stored = np.swapaxes(d_ket, 1, 2).transpose(1, 2, 0)
        highest_occupied = float(np.max(energies[:, n_occupied - 1]))
        lowest_empty = float(np.min(energies[:, n_occupied]))
        indirect_gap = lowest_empty - highest_occupied
        return DensityUpdateResult(
            density=d_stored,
            energies=energies.T,
            mu=float(0.5 * (highest_occupied + lowest_empty)),
            observables={
                "minimum_direct_gap_ev": minimum_direct_gap,
                "indirect_gap_ev": indirect_gap,
            },
        )

    return density_update
