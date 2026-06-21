from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403

def moire_cell_area_nm2(lattice: HTGLattice) -> float:
    return real_space_cell_area_nm2_from_reciprocal(lattice.b_m1, lattice.b_m2)


def _infer_htg_band_count(nt: int, *, n_spin: int = 2, n_eta: int = 2) -> int:
    n_flavor = int(n_spin) * int(n_eta)
    if int(nt) % n_flavor != 0:
        raise ValueError(f"Projected dimension nt={nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}")
    n_band = int(nt) // n_flavor
    if n_band < 2 or n_band % 2 != 0:
        raise ValueError(f"HTG projected band count must be an even integer >= 2, got {n_band}")
    return n_band


def _remote_band_count_per_side(n_band: int) -> int:
    n_band = int(n_band)
    if n_band < 2 or n_band % 2 != 0:
        raise ValueError(f"HTG projected band count must be an even integer >= 2, got {n_band}")
    return (n_band - 2) // 2


def _central_projected_band_indices(n_band: int) -> tuple[int, int]:
    lower_count = _remote_band_count_per_side(n_band)
    return int(lower_count), int(lower_count + 1)


def htg_band_reference_occupations(n_band: int) -> np.ndarray:
    """Reference occupations for HTG density matrices.

    For the central two-band model this is the usual half-filled reference.
    When remote bands are included, the physically neutral reference keeps the
    lower remote bands filled, the central pair half filled, and the upper
    remote bands empty.
    """

    n_band = int(n_band)
    lower_count = _remote_band_count_per_side(n_band)
    reference = np.zeros(n_band, dtype=float)
    reference[:lower_count] = 1.0
    reference[lower_count : lower_count + 2] = 0.5
    return reference


def _htg_reference_density_diagonal(
    nt: int,
    nk: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    n_spin = int(n_spin)
    n_eta = int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    band_reference = htg_band_reference_occupations(n_band)
    idx = np.arange(int(nt), dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    diagonal = np.zeros((int(nt), int(nk)), dtype=float)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                diagonal[int(idx[ispin, ieta, iband]), :] = float(band_reference[iband])
    return diagonal


def _htg_reference_density_blocks(
    nt: int,
    nk: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    diagonal = _htg_reference_density_diagonal(nt, nk, n_spin=n_spin, n_eta=n_eta)
    reference = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    rows = np.arange(int(nt), dtype=int)
    for ik in range(int(nk)):
        reference[rows, rows, ik] = diagonal[:, ik]
    return reference


def htg_projector_from_density(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")
    return density + _htg_reference_density_blocks(nt, nk, n_spin=n_spin, n_eta=n_eta)


def _validate_primitive_cell_integer_filling(nu: float, *, atol: float = 1.0e-9) -> int:
    """Return integer primitive-cell filling or reject fractional fillings.

    Primitive-cell HTG HF cannot represent a translation-breaking rational
    filling by spreading a fractional electron over the finite k mesh. Such
    fillings require a folded-BZ/supercell adapter with an integer number of
    occupied states per supercell k point.
    """

    raw = float(nu)
    rounded = int(round(raw))
    if abs(raw - rounded) > float(atol):
        raise ValueError(
            f"Primitive-cell HTG HF requires integer filling nu per primitive moire cell; got nu={nu}. "
            "Fractional fillings require a supercell/folded-BZ calculation."
        )
    return rounded

def htg_occupied_state_count(
    nu: float,
    nt: int,
    nk: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    integer_nu = _validate_primitive_cell_integer_filling(nu)
    n_flavor = int(n_spin) * int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    lower_remote_per_flavor = _remote_band_count_per_side(n_band)
    occupied = (int(lower_remote_per_flavor) * n_flavor + int(integer_nu) + n_flavor) * int(nk)
    if occupied < 0 or occupied > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied-state count {occupied} outside [0, {int(nt) * int(nk)}]")
    return int(occupied)


def htg_occupied_bands_per_k(
    nu: float,
    nt: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    integer_nu = _validate_primitive_cell_integer_filling(nu)
    n_flavor = int(n_spin) * int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    lower_remote_per_flavor = _remote_band_count_per_side(n_band)
    occupied = int(lower_remote_per_flavor) * n_flavor + int(integer_nu) + n_flavor
    if occupied < 0 or occupied > int(nt):
        raise ValueError(f"Filling nu={nu} gives per-k occupation {occupied} outside [0, {int(nt)}]")
    return int(occupied)


def htg_filling_from_density(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    n_flavor = int(n_spin) * int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    lower_remote_per_flavor = _remote_band_count_per_side(n_band)
    projector = htg_projector_from_density(density, n_spin=n_spin, n_eta=n_eta)
    total_particles = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    particles_per_k = total_particles / float(nk)
    central_particles_per_k = particles_per_k - float(lower_remote_per_flavor) * n_flavor
    return float(central_particles_per_k - float(n_flavor))


def projector_idempotency_residual(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    projector = htg_projector_from_density(density, n_spin=n_spin, n_eta=n_eta)
    residual = 0.0
    for ik in range(nk):
        projector_block = projector[:, :, ik]
        residual = max(residual, float(np.max(np.abs(projector_block @ projector_block - projector_block))))
    return float(residual)


def hermitian_residual(blocks: np.ndarray) -> float:
    blocks = np.asarray(blocks, dtype=np.complex128)
    residual = 0.0
    for ik in range(blocks.shape[2]):
        residual = max(residual, float(np.max(np.abs(blocks[:, :, ik] - blocks[:, :, ik].conjugate().T))))
    return float(residual)


def htg_gap_estimate(energies: np.ndarray, nu: float) -> float:
    total_occupied = htg_occupied_state_count(nu, energies.shape[0], energies.shape[1])
    sorted_energies = np.sort(np.asarray(energies, dtype=float), axis=None)
    if total_occupied <= 0 or total_occupied >= sorted_energies.size:
        return float("nan")
    return float(sorted_energies[total_occupied] - sorted_energies[total_occupied - 1])


def htg_gap_from_occupation_mask(energies: np.ndarray, occupation_mask: np.ndarray) -> float:
    energies = np.asarray(energies, dtype=float)
    occupied = np.asarray(occupation_mask, dtype=bool)
    if occupied.shape != energies.shape:
        raise ValueError(f"Expected occupation mask shape {energies.shape}, got {occupied.shape}")
    if not np.any(occupied) or np.all(occupied):
        return float("nan")
    return float(np.min(energies[~occupied]) - np.max(energies[occupied]))


def htg_occupation_mask_from_density(
    density: np.ndarray,
    *,
    threshold: float = 0.0,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")
    projector = htg_projector_from_density(density, n_spin=n_spin, n_eta=n_eta)
    mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        occupations = np.linalg.eigvalsh(projector[:, :, ik]).real
        mask[:, ik] = occupations > float(threshold)
    return mask

__all__ = [name for name in globals() if not name.startswith('__')]
