from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def find_chemical_potential(energies: np.ndarray, filling_fraction: float) -> float:
    flattened = np.sort(np.ravel(energies))
    occupancies = np.arange(1, flattened.size + 1, dtype=float) / float(flattened.size)
    idx = 0
    while idx < flattened.size - 1 and filling_fraction > occupancies[idx]:
        idx += 1
    if idx < flattened.size - 1:
        return float((flattened[idx + 1] + flattened[idx]) / 2.0)
    return float(flattened[idx])


def occupied_state_linear_indices(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    flattened = np.ravel(np.asarray(energies, dtype=float), order="F")
    if total_occupied <= 0:
        return np.empty(0, dtype=int)
    if total_occupied >= flattened.size:
        return np.arange(flattened.size, dtype=int)
    # Match Julia's column-major `sortperm` tie-breaking for near-degenerate occupancies.
    return np.argsort(flattened, kind="stable")[:total_occupied]


def occupied_state_mask(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    occupied = occupied_state_linear_indices(energies, total_occupied)
    mask = np.zeros(energies.size, dtype=bool)
    mask[occupied] = True
    return mask.reshape(energies.shape, order="F")


def calculate_norm_convergence(updated_density: np.ndarray, previous_density: np.ndarray) -> float:
    numerator = float(np.linalg.norm(previous_density - updated_density))
    denominator = float(np.linalg.norm(updated_density))
    if denominator < 1e-15:
        return 0.0 if numerator < 1e-15 else float("inf")
    return numerator / denominator


def _reference_diagonal_array(reference_diagonal: np.ndarray | float | None, nb: int) -> np.ndarray:
    if reference_diagonal is None:
        return np.zeros((nb,), dtype=float)
    values = np.asarray(reference_diagonal, dtype=float)
    if values.ndim == 0:
        return np.full((nb,), float(values), dtype=float)
    if values.shape != (nb,):
        raise ValueError(f"Expected reference_diagonal shape {(nb,)}, got {values.shape}")
    return values


def flat_sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    layout = np.arange(int(n_spin) * int(n_eta) * int(nb), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(nb)),
        order="F",
    )
    return np.asarray(layout[int(ispin), int(ieta), :], dtype=int)


def flatten_sector_blocks(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=np.complex128)
    if arr.ndim != 5:
        raise ValueError(f"Expected blocks shape (n_spin, n_eta, nb, nb, nk), got {arr.shape}")
    n_spin, n_eta, nb, nb_col, nk = arr.shape
    if nb_col != nb:
        raise ValueError(f"Expected square sector blocks, got {arr.shape}")
    flat = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)
    k_indices = np.arange(nk)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            flat[np.ix_(idx, idx, k_indices)] = arr[ispin, ieta]
    return flat


def unflatten_sector_blocks(flat: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.complex128)
    expected = int(n_spin) * int(n_eta) * int(nb)
    if arr.ndim != 3 or arr.shape[0] != expected or arr.shape[1] != expected:
        raise ValueError(f"Expected flat blocks shape {(expected, expected, 'nk')}, got {arr.shape}")
    nk = arr.shape[2]
    blocks = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), nk), dtype=np.complex128)
    k_indices = np.arange(nk)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            blocks[ispin, ieta] = arr[np.ix_(idx, idx, k_indices)]
    return blocks


def unflatten_sector_energies(flat_energies: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    arr = np.asarray(flat_energies, dtype=float)
    expected = int(n_spin) * int(n_eta) * int(nb)
    if arr.ndim != 2 or arr.shape[0] != expected:
        raise ValueError(f"Expected flat energy shape {(expected, 'nk')}, got {arr.shape}")
    nk = arr.shape[1]
    energies = np.zeros((int(n_spin), int(n_eta), int(nb), nk), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            energies[ispin, ieta] = arr[idx, :]
    return energies


def density_from_fixed_sector_occupations(
    hamiltonian_blocks: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(hamiltonian_blocks, dtype=np.complex128)
    if blocks.ndim != 5:
        raise ValueError(f"Expected hamiltonian_blocks shape (n_spin, n_eta, nb, nb, nk), got {blocks.shape}")
    n_spin, n_eta, nb, nb_col, nk = blocks.shape
    if nb_col != nb:
        raise ValueError(f"Expected square sector blocks, got {blocks.shape}")
    counts = np.asarray(occupation_counts, dtype=int)
    if counts.shape != (n_spin, n_eta):
        raise ValueError(f"Expected occupation_counts shape {(n_spin, n_eta)}, got {counts.shape}")
    ref = _reference_diagonal_array(reference_diagonal, nb)
    density = np.zeros_like(blocks, dtype=np.complex128)
    energies = np.zeros((n_spin, n_eta, nb, nk), dtype=float)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            n_occ = int(counts[ispin, ieta])
            if n_occ < 0 or n_occ > nb:
                raise ValueError(f"occupation count {n_occ} is outside [0, {nb}]")
            for ik in range(nk):
                evals, evecs = eigh(blocks[ispin, ieta, :, :, ik])
                energies[ispin, ieta, :, ik] = evals
                if n_occ:
                    projector = evecs[:, :n_occ] @ evecs[:, :n_occ].conjugate().T
                else:
                    projector = np.zeros((nb, nb), dtype=np.complex128)
                density[ispin, ieta, :, :, ik] = projector - np.diag(ref)
    return density, energies


def conventional_projector_to_stored(projector: np.ndarray) -> np.ndarray:
    arr = np.asarray(projector, dtype=np.complex128)
    if arr.ndim < 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected projector with square leading matrix axes, got {arr.shape}")
    return np.swapaxes(arr, 0, 1).copy()


def stored_projector_to_conventional(stored: np.ndarray) -> np.ndarray:
    return conventional_projector_to_stored(stored)


def random_unitary_from_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    sampled = rng.standard_normal((int(dim), int(dim))) + 1j * rng.standard_normal((int(dim), int(dim)))
    hermitian = sampled + sampled.conjugate().T
    _, vecs = np.linalg.eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)


def apply_random_projector_rotation(
    density: np.ndarray,
    *,
    reference_density: np.ndarray,
    alpha: float,
    seed: int,
) -> None:
    arr = np.asarray(density, dtype=np.complex128)
    reference = np.asarray(reference_density, dtype=np.complex128)
    if arr.shape != reference.shape or arr.ndim != 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            "density and reference_density must have matching shape (nt, nt, nk); "
            f"got {arr.shape} and {reference.shape}"
        )
    rng = np.random.default_rng(seed)
    nt = arr.shape[0]
    for ik in range(arr.shape[2]):
        unitary = random_unitary_from_hermitian(nt, rng)
        projector = arr[:, :, ik] + reference[:, :, ik]
        rotated_density = unitary.conjugate().T @ projector @ unitary - reference[:, :, ik]
        arr[:, :, ik] = (1.0 - float(alpha)) * arr[:, :, ik] + float(alpha) * rotated_density
