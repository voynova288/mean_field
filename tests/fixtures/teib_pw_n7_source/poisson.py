"""Fixed-density Kane-Poisson helpers for the noninteracting stage.

The source paper does not specify its Poisson boundary conditions or detailed
numerics.  This module therefore uses a periodic cell, enforces charge
neutrality, and fixes the electrostatic gauge by requiring zero mean voltage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .kane8 import (
    PeriodicProfile,
    default_heterostructure,
    kane_hamiltonian,
    plane_wave_indices,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0_F_PER_M = 8.8541878128e-12
DEFAULT_SHEET_DENSITY_CM2 = 5.5e10
MIN_WELL_LOCALIZATION = 0.5


def fermi_function(argument: np.ndarray | float) -> np.ndarray | float:
    """Evaluate ``1 / (exp(argument) + 1)`` without overflow."""

    values = np.asarray(argument, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    decaying = np.exp(-values[positive])
    result[positive] = decaying / (1.0 + decaying)
    growing = np.exp(values[~positive])
    result[~positive] = 1.0 / (1.0 + growing)
    if result.ndim == 0:
        return float(result)
    return result


def cm2_to_nm2(density_cm2: np.ndarray | float) -> np.ndarray | float:
    """Convert an areal density from cm^-2 to nm^-2."""

    density = np.asarray(density_cm2, dtype=float)
    if not np.isfinite(density).all() or np.any(density < 0.0):
        raise ValueError("density must be finite and non-negative")
    converted = density * 1.0e-14
    if converted.ndim == 0:
        return float(converted)
    return converted


def radial_fermi_disk_weights(
    k_nm_inv: np.ndarray,
    density_nm2: float,
    degeneracy: int = 2,
) -> tuple[np.ndarray, float]:
    """Return annular density weights truncated exactly at the Fermi radius."""

    k = np.asarray(k_nm_inv, dtype=float)
    density = float(density_nm2)
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k grid must be one-dimensional with at least two points")
    if not np.isfinite(k).all() or k[0] != 0.0 or np.any(np.diff(k) <= 0.0):
        raise ValueError("k grid must start at zero and increase strictly")
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("density must be finite and non-negative")
    if not isinstance(degeneracy, (int, np.integer)) or degeneracy <= 0:
        raise ValueError("degeneracy must be a positive integer")

    kf = np.sqrt(4.0 * np.pi * density / degeneracy)
    edges = np.empty(k.size + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (k[:-1] + k[1:])
    edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])
    if kf > edges[-1] + 10.0 * np.finfo(float).eps * max(1.0, kf):
        raise ValueError("k grid does not cover the Fermi disk")

    inner = np.minimum(edges[:-1], kf)
    outer = np.minimum(edges[1:], kf)
    weights = degeneracy * np.maximum(outer**2 - inner**2, 0.0) / (4.0 * np.pi)
    occupied = np.flatnonzero(weights > 0.0)
    if occupied.size:
        weights[occupied[-1]] += density - np.sum(weights)
    return weights, float(kf)


def _uniform_period(z_nm: np.ndarray) -> tuple[np.ndarray, float, float]:
    z = np.asarray(z_nm, dtype=float)
    if z.ndim != 1 or z.size < 3 or not np.isfinite(z).all():
        raise ValueError("z grid must contain at least three finite points")
    spacings = np.diff(z)
    if np.any(spacings <= 0.0) or not np.allclose(
        spacings, spacings[0], rtol=1.0e-12, atol=1.0e-14
    ):
        raise ValueError("z grid must be uniformly increasing")
    dz_nm = float(spacings[0])
    return z, dz_nm, dz_nm * z.size


def reconstruct_envelopes(
    eigenvectors: np.ndarray,
    z_nm: np.ndarray,
    period_nm: float,
    N: int,
) -> np.ndarray:
    """Reconstruct sampled normalized envelopes in ``(state, component, z)`` order."""

    z = np.asarray(z_nm, dtype=float)
    period = float(period_nm)
    indices = plane_wave_indices(N)
    n_pw = indices.size
    vectors = np.asarray(eigenvectors, dtype=complex)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if vectors.ndim != 2 or vectors.shape[0] != 8 * n_pw:
        raise ValueError("eigenvectors must have shape (8 * (2N + 1), n_states)")
    if z.ndim != 1 or z.size == 0 or period <= 0.0:
        raise ValueError("z grid and period must define a positive periodic cell")

    coefficients = vectors.reshape(8, n_pw, vectors.shape[1])
    phases = np.exp(2j * np.pi * np.outer(indices, z) / period) / np.sqrt(period)
    envelopes = np.einsum("apj,pz->jaz", coefficients, phases, optimize=True)
    norms = np.sum(np.abs(envelopes) ** 2, axis=(1, 2)) * period / z.size
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("eigenvectors must have finite nonzero norm")
    return envelopes / np.sqrt(norms)[:, None, None]


def solve_periodic_poisson(
    z_nm: np.ndarray,
    charge_density_e_nm3: np.ndarray,
    epsilon_r: np.ndarray | float,
    neutrality_tol_e_nm2: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve periodic ``-d(epsilon d phi/dz)/dz = rho`` in SI units.

    ``charge_density_e_nm3`` is signed positive-charge density in elementary
    charges per nm^3.  The returned electron potential energy is numerically
    ``-phi`` in eV.
    """

    z, dz_nm, _ = _uniform_period(z_nm)
    charge = np.asarray(charge_density_e_nm3, dtype=float)
    if charge.shape != z.shape or not np.isfinite(charge).all():
        raise ValueError("charge density must be a finite array matching z")
    epsilon = np.asarray(epsilon_r, dtype=float)
    if epsilon.ndim == 0:
        epsilon = np.full(z.size, float(epsilon))
    if epsilon.shape != z.shape or not np.isfinite(epsilon).all() or np.any(epsilon <= 0.0):
        raise ValueError("epsilon_r must be positive and match z")
    net_charge_e_nm2 = float(np.sum(charge) * dz_nm)
    if abs(net_charge_e_nm2) > neutrality_tol_e_nm2:
        raise ValueError("periodic Poisson charge must be neutral")
    if np.all(charge == 0.0):
        zeros = np.zeros_like(charge)
        return zeros, zeros.copy()

    epsilon_next = np.roll(epsilon, -1)
    epsilon_face = 2.0 * epsilon * epsilon_next / (epsilon + epsilon_next)
    dz_m = dz_nm * 1.0e-9
    face_coefficient = EPSILON_0_F_PER_M * epsilon_face / dz_m**2
    matrix = np.zeros((z.size, z.size), dtype=float)
    for row in range(z.size):
        previous = (row - 1) % z.size
        following = (row + 1) % z.size
        left = face_coefficient[previous]
        right = face_coefficient[row]
        matrix[row, row] = left + right
        matrix[row, previous] = -left
        matrix[row, following] = -right

    rhs = charge * ELEMENTARY_CHARGE_C * 1.0e27
    scale = float(np.max(np.diag(matrix)))
    augmented = np.zeros((z.size + 1, z.size + 1), dtype=float)
    augmented[:-1, :-1] = matrix / scale
    augmented[:-1, -1] = 1.0
    augmented[-1, :-1] = 1.0
    augmented_rhs = np.zeros(z.size + 1)
    augmented_rhs[:-1] = rhs / scale
    phi_v = np.linalg.solve(augmented, augmented_rhs)[:-1]
    phi_v -= np.mean(phi_v)
    return phi_v, -phi_v


def plane_wave_potential_matrix(
    z_nm: np.ndarray,
    potential_ev: np.ndarray,
    N: int,
) -> np.ndarray:
    """Convert a sampled real periodic potential to a Hermitian PW matrix."""

    z, _, period_nm = _uniform_period(z_nm)
    potential = np.asarray(potential_ev)
    if potential.shape != z.shape or not np.isfinite(potential).all():
        raise ValueError("potential must be a finite array matching z")
    if np.iscomplexobj(potential) and np.max(np.abs(np.imag(potential))) > 1.0e-13:
        raise ValueError("sampled potential must be real")
    potential = np.asarray(np.real(potential), dtype=float)
    indices = plane_wave_indices(N)
    harmonics = indices[:, None] - indices[None, :]
    coefficients: dict[int, complex] = {}
    for harmonic in np.unique(harmonics):
        coefficients[int(harmonic)] = np.mean(
            potential * np.exp(-2j * np.pi * harmonic * z / period_nm)
        )
    matrix = np.vectorize(coefficients.__getitem__, otypes=[complex])(harmonics)
    return (matrix + matrix.conj().T) / 2.0


def add_potential_to_kane(
    hamiltonian: np.ndarray, potential_matrix: np.ndarray
) -> np.ndarray:
    """Add one scalar plane-wave potential to every Kane component."""

    bare = np.asarray(hamiltonian, dtype=complex)
    potential = np.asarray(potential_matrix, dtype=complex)
    if potential.ndim != 2 or potential.shape[0] != potential.shape[1]:
        raise ValueError("potential matrix must be square")
    expected = 8 * potential.shape[0]
    if bare.shape != (expected, expected):
        raise ValueError("Hamiltonian and potential dimensions are inconsistent")
    if not np.allclose(potential, potential.conj().T, atol=1.0e-12):
        raise ValueError("potential matrix must be Hermitian")
    return bare + np.kron(np.eye(8), potential)


def _classification_weights(
    profile: PeriodicProfile,
    N: int,
    eigenvectors: np.ndarray,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pw = 2 * N + 1
    components = eigenvectors.reshape(8, n_pw, -1)
    conduction = np.sum(np.abs(components[:2]) ** 2, axis=(0, 1))
    valence = 1.0 - conduction
    z_nm = np.arange(nz, dtype=float) * profile.period_nm / nz
    envelopes = reconstruct_envelopes(
        eigenvectors, z_nm=z_nm, period_nm=profile.period_nm, N=N
    )
    density = np.sum(np.abs(envelopes) ** 2, axis=1)
    inas = np.array([profile.material_at(z).name == "InAs" for z in z_nm])
    gasb = np.array([profile.material_at(z).name == "GaSb" for z in z_nm])
    dz_nm = profile.period_nm / nz
    inas_weight = np.sum(density[:, inas], axis=1) * dz_nm
    gasb_weight = np.sum(density[:, gasb], axis=1) * dz_nm
    return conduction, valence, inas_weight, gasb_weight


def _initial_branch_index(
    energies: np.ndarray,
    component_weight: np.ndarray,
    layer_weight: np.ndarray,
    in_window: np.ndarray,
    electron: bool,
) -> int:
    candidates = np.flatnonzero(
        in_window
        & (component_weight >= 0.5)
        & (layer_weight >= MIN_WELL_LOCALIZATION)
    )
    if candidates.size == 0:
        kind = "electron" if electron else "hole"
        raise ValueError(
            f"energy window contains no {kind}-like branch localized in its well"
        )
    classification = component_weight + layer_weight
    if electron:
        order = sorted(
            candidates,
            key=lambda index: (
                round(float(energies[index]), 12),
                -float(classification[index]),
                int(index),
            ),
        )
    else:
        order = sorted(
            candidates,
            key=lambda index: (
                -round(float(energies[index]), 12),
                -float(classification[index]),
                int(index),
            ),
        )
    return int(order[0])


def _continued_branch_index(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    previous_state: np.ndarray,
    component_weight: np.ndarray,
    layer_weight: np.ndarray,
    in_window: np.ndarray,
    electron: bool,
) -> tuple[int, float]:
    candidates = np.flatnonzero(
        in_window
        & (component_weight >= 0.5)
        & (layer_weight >= MIN_WELL_LOCALIZATION)
    )
    if candidates.size == 0:
        kind = "electron" if electron else "hole"
        raise ValueError(
            f"energy window contains no {kind}-like branch localized in its well"
        )
    overlaps = np.abs(eigenvectors[:, candidates].conj().T @ previous_state) ** 2
    candidate_energies = energies[candidates]
    span = float(np.ptp(candidate_energies))
    if span == 0.0:
        energy_preference = np.ones(candidates.size)
    elif electron:
        energy_preference = 1.0 - (
            candidate_energies - np.min(candidate_energies)
        ) / span
    else:
        energy_preference = (
            candidate_energies - np.min(candidate_energies)
        ) / span
    score = (
        0.70 * overlaps
        + 0.15 * component_weight[candidates]
        + 0.10 * layer_weight[candidates]
        + 0.05 * energy_preference
    )
    selected_position = int(np.argmax(score))
    return int(candidates[selected_position]), float(overlaps[selected_position])


def track_kane_branches(
    profile: PeriodicProfile,
    N: int,
    k_nm_inv: np.ndarray,
    potential_matrix: np.ndarray | None = None,
    energy_window_ev: tuple[float, float] = (-1.0, 0.9),
    classification_nz: int = 128,
) -> dict[str, Any]:
    """Track electron-InAs and hole-GaSb branches along a radial k grid."""

    k = np.asarray(k_nm_inv, dtype=float)
    if k.ndim != 1 or k.size == 0 or np.any(np.diff(k) < 0.0):
        raise ValueError("k grid must be a nondecreasing one-dimensional array")
    if classification_nz < 2 * N + 2:
        raise ValueError("classification_nz is too small for the plane-wave basis")
    low, high = map(float, energy_window_ev)
    if not low < high:
        raise ValueError("energy window must be increasing")
    dimension = 8 * (2 * N + 1)
    if potential_matrix is not None:
        potential_matrix = np.asarray(potential_matrix, dtype=complex)

    electron_energies = np.empty(k.size)
    hole_energies = np.empty(k.size)
    electron_states = np.empty((k.size, dimension), dtype=complex)
    hole_states = np.empty((k.size, dimension), dtype=complex)
    diagnostic_names = (
        "electron_component_weight",
        "hole_component_weight",
        "electron_inas_weight",
        "hole_gasb_weight",
        "electron_overlap",
        "hole_overlap",
        "electron_classification_score",
        "hole_classification_score",
        "electron_index",
        "hole_index",
    )
    diagnostics = {name: np.empty(k.size) for name in diagnostic_names}

    for position, radial_k in enumerate(k):
        hamiltonian = kane_hamiltonian(
            profile, N=N, kx=float(radial_k), ky=0.0
        )
        if potential_matrix is not None:
            hamiltonian = add_potential_to_kane(hamiltonian, potential_matrix)
        energies, eigenvectors = np.linalg.eigh(hamiltonian)
        conduction, valence, inas_weight, gasb_weight = _classification_weights(
            profile, N, eigenvectors, classification_nz
        )
        in_window = (energies >= low) & (energies <= high)
        if position == 0:
            electron_index = _initial_branch_index(
                energies, conduction, inas_weight, in_window, electron=True
            )
            hole_index = _initial_branch_index(
                energies, valence, gasb_weight, in_window, electron=False
            )
            electron_overlap = 1.0
            hole_overlap = 1.0
        else:
            electron_index, electron_overlap = _continued_branch_index(
                energies,
                eigenvectors,
                electron_states[position - 1],
                conduction,
                inas_weight,
                in_window,
                electron=True,
            )
            hole_index, hole_overlap = _continued_branch_index(
                energies,
                eigenvectors,
                hole_states[position - 1],
                valence,
                gasb_weight,
                in_window,
                electron=False,
            )

        electron_energies[position] = energies[electron_index]
        hole_energies[position] = energies[hole_index]
        electron_states[position] = eigenvectors[:, electron_index]
        hole_states[position] = eigenvectors[:, hole_index]
        diagnostics["electron_component_weight"][position] = conduction[electron_index]
        diagnostics["hole_component_weight"][position] = valence[hole_index]
        diagnostics["electron_inas_weight"][position] = inas_weight[electron_index]
        diagnostics["hole_gasb_weight"][position] = gasb_weight[hole_index]
        diagnostics["electron_overlap"][position] = np.clip(electron_overlap, 0.0, 1.0)
        diagnostics["hole_overlap"][position] = np.clip(hole_overlap, 0.0, 1.0)
        diagnostics["electron_classification_score"][position] = (
            conduction[electron_index] + inas_weight[electron_index]
        )
        diagnostics["hole_classification_score"][position] = (
            valence[hole_index] + gasb_weight[hole_index]
        )
        diagnostics["electron_index"][position] = electron_index
        diagnostics["hole_index"][position] = hole_index

    return {
        "electron_energies_ev": electron_energies,
        "hole_energies_ev": hole_energies,
        "electron_states": electron_states,
        "hole_states": hole_states,
        "diagnostics": diagnostics,
    }


def _profile_density(
    states: np.ndarray,
    radial_weights_nm2: np.ndarray,
    z_nm: np.ndarray,
    period_nm: float,
    N: int,
    requested_density_nm2: float,
) -> np.ndarray:
    envelopes = reconstruct_envelopes(
        states.T, z_nm=z_nm, period_nm=period_nm, N=N
    )
    state_densities = np.sum(np.abs(envelopes) ** 2, axis=1)
    density_nm3 = np.tensordot(radial_weights_nm2, state_densities, axes=1)
    integral_nm2 = float(np.sum(density_nm3) * period_nm / z_nm.size)
    if integral_nm2 <= 0.0:
        raise ValueError("occupied branch has zero envelope density")
    return density_nm3 * requested_density_nm2 / integral_nm2


def _densities_from_branches(
    branches: dict[str, Any],
    weights_nm2: np.ndarray,
    z_nm: np.ndarray,
    period_nm: float,
    N: int,
    density_nm2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    electron_density = _profile_density(
        branches["electron_states"],
        weights_nm2,
        z_nm,
        period_nm,
        N,
        density_nm2,
    )
    hole_density = _profile_density(
        branches["hole_states"],
        weights_nm2,
        z_nm,
        period_nm,
        N,
        density_nm2,
    )
    return electron_density, hole_density, hole_density - electron_density


def solve_kane_poisson(
    *,
    max_iter: int = 40,
    tol: float = 1.0e-6,
    mix: float = 0.25,
    N: int = 7,
    nk: int = 81,
    nz: int = 256,
    kmax: float = 0.12,
    density_cm2: float = DEFAULT_SHEET_DENSITY_CM2,
    profile: PeriodicProfile | None = None,
    energy_window_ev: tuple[float, float] = (-1.0, 0.9),
) -> dict[str, Any]:
    """Run a fixed-density radial Kane-Poisson iteration.

    Electron and hole Fermi disks each carry the requested sheet density with
    Kramers degeneracy two.  This is a transparent approximation because the
    paper omits the occupation and electrostatic boundary details.
    """

    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(nk, (int, np.integer)) or nk < 2:
        raise ValueError("nk must be at least two")
    if not isinstance(nz, (int, np.integer)) or nz < max(8, 2 * N + 2):
        raise ValueError("nz is too small for the selected plane-wave basis")
    if not 0.0 < mix <= 1.0:
        raise ValueError("mix must lie in (0, 1]")
    if tol < 0.0 or kmax <= 0.0:
        raise ValueError("tol must be non-negative and kmax positive")

    profile = default_heterostructure() if profile is None else profile
    density_nm2 = float(cm2_to_nm2(density_cm2))
    k_nm_inv = np.linspace(0.0, float(kmax), int(nk))
    weights_nm2, kf_nm_inv = radial_fermi_disk_weights(
        k_nm_inv, density_nm2, degeneracy=2
    )
    z_nm = np.arange(nz, dtype=float) * profile.period_nm / nz
    epsilon_r = np.array(
        [profile.material_at(z).epsilon_r for z in z_nm], dtype=float
    )
    mixed_potential_ev = np.zeros(nz)
    residual_history: list[float] = []

    for _ in range(max_iter):
        potential_matrix = plane_wave_potential_matrix(z_nm, mixed_potential_ev, N)
        branches = track_kane_branches(
            profile,
            N=N,
            k_nm_inv=k_nm_inv,
            potential_matrix=potential_matrix,
            energy_window_ev=energy_window_ev,
            classification_nz=max(64, 2 * N + 2),
        )
        electron_density, hole_density, charge_density = _densities_from_branches(
            branches,
            weights_nm2,
            z_nm,
            profile.period_nm,
            N,
            density_nm2,
        )
        _, target_potential_ev = solve_periodic_poisson(
            z_nm, charge_density, epsilon_r
        )
        residual = float(np.max(np.abs(target_potential_ev - mixed_potential_ev)))
        residual_history.append(residual)
        if residual <= tol:
            break
        mixed_potential_ev = (
            (1.0 - mix) * mixed_potential_ev + mix * target_potential_ev
        )
        mixed_potential_ev -= np.mean(mixed_potential_ev)

    potential_matrix = plane_wave_potential_matrix(z_nm, mixed_potential_ev, N)
    branches = track_kane_branches(
        profile,
        N=N,
        k_nm_inv=k_nm_inv,
        potential_matrix=potential_matrix,
        energy_window_ev=energy_window_ev,
        classification_nz=max(64, 2 * N + 2),
    )
    electron_density, hole_density, charge_density = _densities_from_branches(
        branches,
        weights_nm2,
        z_nm,
        profile.period_nm,
        N,
        density_nm2,
    )
    target_phi_v, target_electron_potential_ev = solve_periodic_poisson(
        z_nm, charge_density, epsilon_r
    )
    self_consistency_residual_ev = float(
        np.max(np.abs(target_electron_potential_ev - mixed_potential_ev))
    )
    electron_potential_ev = mixed_potential_ev.copy()
    phi_v = -electron_potential_ev
    converged = self_consistency_residual_ev <= tol
    metadata = {
        "poisson_boundary": (
            "periodic zero-mean electrostatic gauge; the paper's exact "
            "Poisson boundary and numerics are omitted"
        ),
        "density_model": "fixed-density approximation with Kramers degeneracy g=2",
        "kane_parameters": (
            "inferred EP = 22.5 eV and corrected GaSb gamma2 = 0.08 "
            "(printed as 8.18)"
        ),
        "branch_projection": (
            "Kane-to-BCS branch projection by component weights, layer weights, "
            "energy window, and adjacent-k overlap"
        ),
        "minimum_well_localization": MIN_WELL_LOCALIZATION,
        "scientific_caveat": (
            "Reduced grids test numerical plumbing, not plane-wave or real-space "
            "convergence; occupations impose equal electron and hole Fermi disks."
        ),
        "period_nm": float(profile.period_nm),
        "density_cm2": float(density_cm2),
        "density_nm2": density_nm2,
        "fermi_wavevector_nm_inv": kf_nm_inv,
        "N": int(N),
        "nk": int(nk),
        "nz": int(nz),
        "kmax_nm_inv": float(kmax),
        "max_iter": int(max_iter),
        "tol_ev": float(tol),
        "mix": float(mix),
        "energy_window_ev": [float(energy_window_ev[0]), float(energy_window_ev[1])],
        "iterations": len(residual_history),
        "converged": bool(converged),
        "residual_definition": "max_z abs(Ve_Poisson - Ve_mixed) in eV",
        "final_self_consistency_residual_ev": self_consistency_residual_ev,
    }
    return {
        "z_nm": z_nm,
        "k_nm_inv": k_nm_inv,
        "epsilon_r": epsilon_r,
        "radial_weights_nm2": weights_nm2,
        "electron_energies_ev": branches["electron_energies_ev"],
        "hole_energies_ev": branches["hole_energies_ev"],
        "electron_states": branches["electron_states"],
        "hole_states": branches["hole_states"],
        "electron_density_nm3": electron_density,
        "hole_density_nm3": hole_density,
        "charge_density_e_nm3": charge_density,
        "phi_v": phi_v,
        "electron_potential_ev": electron_potential_ev,
        "mixed_electron_potential_ev": mixed_potential_ev,
        "target_phi_v": target_phi_v,
        "target_electron_potential_ev": target_electron_potential_ev,
        "self_consistency_residual_ev": np.asarray(self_consistency_residual_ev),
        "residual_history_ev": np.asarray(residual_history),
        "diagnostics": branches["diagnostics"],
        "metadata": metadata,
    }


def nonint_output_path(
    N: int, nk: int, data_dir: str | Path = "data"
) -> Path:
    """Return the deterministic Task 3 NPZ output path."""

    return Path(data_dir) / f"nonint_kane_npw{int(N)}_nk{int(nk)}.npz"


def save_poisson_result(path: str | Path, result: dict[str, Any]) -> Path:
    """Save a solver result without object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if key in {"metadata", "diagnostics"}:
            continue
        payload[key] = np.asarray(value)
    payload["metadata_json"] = np.asarray(
        json.dumps(result["metadata"], sort_keys=True)
    )
    for key, value in result["diagnostics"].items():
        payload[f"diagnostic__{key}"] = np.asarray(value)
    np.savez_compressed(destination, **payload)
    return destination


def load_poisson_result(path: str | Path) -> dict[str, Any]:
    """Reload a result written by :func:`save_poisson_result`."""

    result: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for key in archive.files:
            if key == "metadata_json":
                result["metadata"] = json.loads(str(archive[key].item()))
            elif key.startswith("diagnostic__"):
                diagnostics[key.removeprefix("diagnostic__")] = archive[key].copy()
            else:
                result[key] = archive[key].copy()
    result["diagnostics"] = diagnostics
    return result


__all__ = [
    "DEFAULT_SHEET_DENSITY_CM2",
    "ELEMENTARY_CHARGE_C",
    "EPSILON_0_F_PER_M",
    "MIN_WELL_LOCALIZATION",
    "add_potential_to_kane",
    "cm2_to_nm2",
    "fermi_function",
    "load_poisson_result",
    "nonint_output_path",
    "plane_wave_potential_matrix",
    "radial_fermi_disk_weights",
    "reconstruct_envelopes",
    "save_poisson_result",
    "solve_kane_poisson",
    "solve_periodic_poisson",
    "track_kane_branches",
]
