from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.linalg import eigh

from ...core.hf import (
    DensityUpdateResult,
    FlavorBandData,
    HFOverlapBlockSet,
    HartreeFockRun,
    HartreeFockStepResult,
    ProjectedWavefunctionBasis,
    build_flavor_band_data,
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    find_chemical_potential,
    occupied_state_mask,
    real_space_cell_area_nm2_from_reciprocal,
    run_projected_hartree_fock,
    screened_coulomb_matrix,
)
from .hamiltonian import build_hamiltonian, centered_band_indices
from .lattice import HTGLattice, KPath, build_moire_k_grid
from .model import HTGModel
from .params import HTGParams, InteractionParams
from .topology import sublattice_sigma_z


VALLEY_SEQUENCE = (1, -1)


@dataclass(frozen=True)
class HTGProjectedBasisData:
    model: HTGModel
    interaction: InteractionParams
    mesh_size: int
    kvec: np.ndarray
    k_grid_frac: np.ndarray
    basis: ProjectedWavefunctionBasis
    h0: np.ndarray
    sigma_z: np.ndarray
    band_sigma_z: np.ndarray
    central_band_indices: tuple[int, int]
    projected_band_indices: tuple[int, ...]
    reciprocal_grid_shape: tuple[int, int]
    reciprocal_grid_origin: tuple[int, int]
    moire_cell_area_nm2: float

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])


@dataclass(frozen=True)
class HTGHartreeFockRun(HartreeFockRun):
    state: "HTGHartreeFockState"
    overlap_blocks: HFOverlapBlockSet
    basis_data: HTGProjectedBasisData


@dataclass(frozen=True)
class HTGGroundStateScan:
    runs: tuple[HTGHartreeFockRun, ...]

    @property
    def best_run(self) -> HTGHartreeFockRun:
        if not self.runs:
            raise ValueError("No HTG HF runs are available")
        return min(self.runs, key=lambda run: float(run.state.diagnostics.get("hf_energy", np.inf)))


@dataclass(frozen=True)
class HTGInteractionComponents:
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    hartree_eigenvalues: np.ndarray
    fock_eigenvalues: np.ndarray


@dataclass(frozen=True)
class HTGInteractionPathResult:
    path: KPath
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    hartree_diagonal_ev: np.ndarray
    fock_diagonal_ev: np.ndarray
    total_diagonal_ev: np.ndarray
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass(frozen=True)
class HTGHFPathResult:
    path: KPath
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_z_expectation: np.ndarray
    sigma_z_operator: np.ndarray
    band_data: FlavorBandData
    mu: float
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass
class HTGHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_z: np.ndarray
    nu: float
    v0: float
    mu: float = float("nan")
    precision: float = 1.0e-6
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2
    occupation_counts: tuple[int, ...] | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_projected_basis(
        cls,
        basis_data: HTGProjectedBasisData,
        *,
        nu: float,
        precision: float = 1.0e-6,
        occupation_counts: tuple[int, ...] | None = None,
    ) -> "HTGHartreeFockState":
        h0 = np.asarray(basis_data.h0, dtype=np.complex128).copy()
        nt, _, nk = h0.shape
        return cls(
            h0=h0,
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_z=np.asarray(basis_data.sigma_z, dtype=np.complex128).copy(),
            nu=float(nu),
            v0=1.0 / float(basis_data.moire_cell_area_nm2),
            precision=float(precision),
            n_spin=int(basis_data.basis.n_spin),
            n_eta=int(basis_data.basis.n_flavor),
            n_band=int(basis_data.basis.n_band),
            occupation_counts=occupation_counts,
        )


@dataclass(frozen=True)
class HTGInitializer:
    initial_density: np.ndarray | None = None

    def __call__(self, state: HTGHartreeFockState, *, init_mode: str, seed: int) -> None:
        if self.initial_density is not None:
            density = np.asarray(self.initial_density, dtype=np.complex128)
            if density.shape != state.density.shape:
                raise ValueError(f"Expected initial_density shape {state.density.shape}, got {density.shape}")
            state.density[:, :, :] = density
        else:
            state.density[:, :, :] = initialize_htg_density(
                state.h0,
                nu=state.nu,
                init_mode=init_mode,
                seed=seed,
                n_spin=state.n_spin,
                n_eta=state.n_eta,
                n_band=state.n_band,
            )
        _update_htg_diagnostics_from_density(state)


@dataclass(frozen=True)
class HTGDensityBuilder:
    nu: float
    sigma_z: np.ndarray | None = None
    occupation_counts: tuple[int, ...] | None = None
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, sigma_z_expectation, mu, occupation_mask = build_htg_density_from_hamiltonian(
            hamiltonian,
            nu=self.nu,
            sigma_z=self.sigma_z,
            occupation_counts=self.occupation_counts,
            n_spin=self.n_spin,
            n_eta=self.n_eta,
            n_band=self.n_band,
        )
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables={
                "sigma_z": sigma_z_expectation,
                "occupation_mask": occupation_mask,
            },
        )


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


def htg_occupied_state_count(
    nu: float,
    nt: int,
    nk: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    n_flavor = int(n_spin) * int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    lower_remote_per_flavor = _remote_band_count_per_side(n_band)
    raw = (float(lower_remote_per_flavor) * n_flavor + float(nu) + float(n_flavor)) * int(nk)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer occupied-state count {raw}")
    if rounded < 0 or rounded > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied-state count {rounded} outside [0, {int(nt) * int(nk)}]")
    return rounded


def htg_occupied_bands_per_k(
    nu: float,
    nt: int,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    n_flavor = int(n_spin) * int(n_eta)
    n_band = _infer_htg_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    lower_remote_per_flavor = _remote_band_count_per_side(n_band)
    raw = float(lower_remote_per_flavor) * n_flavor + float(nu) + float(n_flavor)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer per-k occupation {raw}")
    if rounded < 0 or rounded > int(nt):
        raise ValueError(f"Filling nu={nu} gives per-k occupation {rounded} outside [0, {int(nt)}]")
    return rounded


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


def normalize_htg_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "noninteracting": "bm",
        "random": "random",
        "diag_random": "diag_random",
        "flavor": "flavor",
        "fb": "fb",
        "d3a": "fb",
        "fb_d3a": "fb",
        "fb_d2a2": "fb",
        "d2a2": "fb",
        "d3b": "sublattice",
        "fb_d3b": "sublattice",
        "fb_d2b2": "sublattice",
        "d2b2": "sublattice",
        "fi": "fi",
        "fi_d3": "fi",
        "d3": "fi",
        "vp": "vp",
        "sp": "sp",
        "chern": "chern",
        "sublattice": "sublattice",
        "perturbed": "perturbed",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported HTG HF init mode: {init_mode}. "
            "Supported modes: bm, random, diag_random, flavor, fb/d3a, fi, vp, sp, chern, "
            "sublattice/d3b, perturbed"
        )
    return aliases[normalized]


def _flavor_priority(flag: str, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flavors = ((0, 0), (0, 1), (1, 0), (1, 1))
    n_band = int(idx.shape[2])
    lower_count = _remote_band_count_per_side(n_band)
    lower_bands = tuple(range(lower_count))
    central_a, central_b = _central_projected_band_indices(n_band)
    upper_bands = tuple(range(central_b + 1, n_band))

    def by_bands(bands: tuple[int, ...], flavor_order=flavors) -> list[int]:
        return [int(idx[ispin, ieta, iband]) for iband in bands for ispin, ieta in flavor_order]

    def by_flavors(bands: tuple[int, ...], flavor_order=flavors) -> list[int]:
        return [int(idx[ispin, ieta, iband]) for ispin, ieta in flavor_order for iband in bands]

    lower_states = by_bands(lower_bands)
    upper_states = by_bands(upper_bands)
    if flag in {"flavor", "fb", "chern"}:
        ordered = lower_states + by_bands((central_a, central_b)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "fi":
        ordered = lower_states + by_flavors((central_a, central_b)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "vp":
        flavor_order = tuple((ispin, ieta) for ieta in range(idx.shape[1]) for ispin in range(idx.shape[0]))
        ordered = by_bands(lower_bands, flavor_order) + by_flavors((central_a, central_b), flavor_order) + by_bands(upper_bands, flavor_order)
        return np.asarray(ordered, dtype=int)
    if flag == "sp":
        flavor_order = tuple((ispin, ieta) for ispin in range(idx.shape[0]) for ieta in range(idx.shape[1]))
        ordered = by_bands(lower_bands, flavor_order) + by_flavors((central_a, central_b), flavor_order) + by_bands(upper_bands, flavor_order)
        return np.asarray(ordered, dtype=int)
    if flag == "sublattice":
        ordered = lower_states + by_bands((central_b, central_a)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "random":
        return rng.permutation(idx.ravel(order="F"))
    raise ValueError(f"Unsupported HTG flavor priority flag: {flag}")


def htg_flavor_occupation_counts_for_init_mode(
    init_mode: str,
    *,
    nu: float,
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[int, ...] | None:
    """Return per-flavor occupation constraints implied by a strong-coupling seed.

    The tuple is flattened in ``(spin, valley)`` C-order and each entry gives
    how many Chern-sublattice bands are occupied in that flavor at every k.
    Stochastic/noninteracting seeds return ``None`` because they are intended
    to explore the unconstrained variational problem.
    """

    normalized = normalize_htg_init_mode(init_mode)
    if normalized in {"bm", "random", "diag_random", "perturbed"}:
        return None

    nt = int(n_spin) * int(n_eta) * int(n_band)
    occupied_per_k = htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta)
    rng = np.random.default_rng(seed)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    order = _flavor_priority(normalized, idx, rng)
    if occupied_per_k > order.size:
        raise ValueError(f"Filling nu={nu} requires {occupied_per_k} states per k, but only {order.size} are available")

    counts = np.zeros((n_spin, n_eta), dtype=int)
    reverse: dict[int, tuple[int, int, int]] = {}
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                reverse[int(idx[ispin, ieta, iband])] = (ispin, ieta, iband)
    for state_index in order[:occupied_per_k]:
        ispin, ieta, _ = reverse[int(state_index)]
        counts[ispin, ieta] += 1
    if np.any(counts < 0) or np.any(counts > n_band):
        raise ValueError(f"Invalid flavor occupation counts for init_mode={init_mode}, nu={nu}: {counts}")
    return tuple(int(value) for value in counts.reshape(-1, order="C"))


def _random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    sampled = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    hermitian = sampled + sampled.conjugate().T
    _, vecs = np.linalg.eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)


def _apply_random_rotation(
    density: np.ndarray,
    *,
    reference_density: np.ndarray,
    alpha: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    nt = density.shape[0]
    for ik in range(density.shape[2]):
        unitary = _random_unitary(nt, rng)
        projector = density[:, :, ik] + reference_density[:, :, ik]
        rotated_density = unitary.conjugate().T @ projector @ unitary - reference_density[:, :, ik]
        density[:, :, ik] = (1.0 - alpha) * density[:, :, ik] + alpha * rotated_density


def initialize_htg_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_htg_init_mode(init_mode)
    h0 = np.asarray(h0, dtype=np.complex128)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    if init_mode == "bm":
        return build_htg_density_from_hamiltonian(
            h0,
            nu=nu,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )[0]
    if init_mode == "diag_random":
        init_mode = "random"

    rng = np.random.default_rng(seed)
    reference_density = _htg_reference_density_blocks(nt, nk, n_spin=n_spin, n_eta=n_eta)
    density = np.zeros_like(h0)
    total_occupied = htg_occupied_state_count(nu, nt, nk, n_spin=n_spin, n_eta=n_eta)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")

    if init_mode == "random":
        random_energies = rng.standard_normal((nt, nk))
        occ_mask = occupied_state_mask(random_energies, total_occupied)
        for ik in range(nk):
            unitary = _random_unitary(nt, rng)
            occupied = np.flatnonzero(occ_mask[:, ik])
            if occupied.size == 0:
                density[:, :, ik] = -reference_density[:, :, ik]
            else:
                occupied_vecs = unitary[:, occupied]
                density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density

    flag = "flavor" if init_mode == "perturbed" else init_mode
    order = _flavor_priority(flag, idx, rng)
    full_states = total_occupied // nk
    partial_count = total_occupied % nk
    if full_states > order.size:
        raise ValueError(f"Filling nu={nu} requires {full_states} full states, but only {order.size} are available")
    for state_index in order[:full_states]:
        density[int(state_index), int(state_index), :] = 1.0
    if partial_count:
        state_index = int(order[full_states])
        occupied_k = rng.permutation(nk)[:partial_count]
        density[state_index, state_index, occupied_k] = 1.0
    density -= reference_density

    if init_mode == "perturbed":
        _apply_random_rotation(density, reference_density=reference_density, alpha=0.05, seed=seed)
    return density


def build_htg_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    nu: float,
    sigma_z: np.ndarray | None = None,
    occupation_counts: tuple[int, ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    if sigma_z is not None and np.asarray(sigma_z).shape != hamiltonian.shape:
        raise ValueError(f"Expected sigma_z shape {hamiltonian.shape}, got {np.asarray(sigma_z).shape}")

    energies = np.zeros((nt, nk), dtype=float)
    sigma_z_expectation = np.zeros((nt, nk), dtype=float)
    density = np.zeros_like(hamiltonian)
    reference_density = _htg_reference_density_blocks(nt, nk, n_spin=n_spin, n_eta=n_eta)

    if occupation_counts is not None:
        counts = np.asarray(occupation_counts, dtype=int).reshape(-1)
        if counts.size != int(n_spin) * int(n_eta):
            raise ValueError(
                f"Expected {int(n_spin) * int(n_eta)} flavor occupation counts, got {counts.size}"
            )
        if nt != int(n_spin) * int(n_eta) * int(n_band):
            raise ValueError(
                f"Hamiltonian dimension {nt} is incompatible with "
                f"n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}"
            )
        if np.any(counts < 0) or np.any(counts > int(n_band)):
            raise ValueError(f"Flavor occupation counts must lie in [0, {int(n_band)}], got {counts.tolist()}")
        if int(np.sum(counts)) != htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta):
            raise ValueError(
                f"Flavor occupation counts sum to {int(np.sum(counts))}, "
                f"but nu={nu} requires {htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta)} occupied bands per k"
            )

        idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
        counts_2d = counts.reshape((n_spin, n_eta), order="C")
        occ_mask = np.zeros((nt, nk), dtype=bool)
        for ik in range(nk):
            density[:, :, ik] = -reference_density[:, :, ik]
            for ispin in range(n_spin):
                for ieta in range(n_eta):
                    block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    reference_block = reference_density[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
                    if sigma_z is not None:
                        sigma_block = sigma_z[:, :, ik][np.ix_(block_indices, block_indices)]
                        sigma_z_expectation[block_indices, ik] = np.real(
                            np.diag(eigvecs.conjugate().T @ sigma_block @ eigvecs)
                        )
                    n_occ = int(counts_2d[ispin, ieta])
                    if n_occ > 0:
                        occupied_vecs = eigvecs[:, :n_occ]
                        density[:, :, ik][np.ix_(block_indices, block_indices)] = (
                            occupied_vecs.conjugate() @ occupied_vecs.T - reference_block
                        )
                        occ_mask[block_indices[:n_occ], ik] = True

        if np.any(occ_mask) and not np.all(occ_mask):
            mu = 0.5 * (float(np.max(energies[occ_mask])) + float(np.min(energies[~occ_mask])))
        else:
            mu = float(np.mean(energies))
        return density, energies, sigma_z_expectation, float(mu), occ_mask

    vecs = np.zeros_like(hamiltonian)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs
        if sigma_z is not None:
            sigma_z_expectation[:, ik] = np.real(np.diag(eigvecs.conjugate().T @ sigma_z[:, :, ik] @ eigvecs))

    total_occupied = htg_occupied_state_count(nu, nt, nk, n_spin=n_spin, n_eta=n_eta)
    occ_mask = occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, float(total_occupied) / float(energies.size))

    for ik in range(nk):
        occupied = np.flatnonzero(occ_mask[:, ik])
        if occupied.size == 0:
            density[:, :, ik] = -reference_density[:, :, ik]
            continue
        occupied_vecs = vecs[:, occupied, ik]
        density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]

    return density, energies, sigma_z_expectation, float(mu), occ_mask


def _layer_potential_operator(lattice: HTGLattice, U_ev: float) -> np.ndarray:
    diagonal = np.zeros(lattice.matrix_dim, dtype=float)
    layer_values = (float(U_ev), 0.0, -float(U_ev))
    for ig in range(lattice.n_g):
        for layer_index, value in enumerate(layer_values):
            start = 6 * ig + 2 * layer_index
            diagonal[start : start + 2] = value
    return np.diag(diagonal).astype(np.complex128)


def _rectangular_g_embedding(lattice: HTGLattice) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    mins = np.min(lattice.g_indices, axis=0)
    maxs = np.max(lattice.g_indices, axis=0)
    grid_shape = (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1))
    origin = (int(mins[0]), int(mins[1]))
    positions = {
        (int(n1), int(n2)): (int(n1 - mins[0]), int(n2 - mins[1]))
        for n1, n2 in np.asarray(lattice.g_indices, dtype=int)
    }
    return grid_shape, origin, positions


def _central_chern_basis_at_k(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    interaction: InteractionParams,
    *,
    valley: int,
    central_pair: tuple[int, int],
    sigma_z_operator: np.ndarray,
    layer_potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hmat = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if interaction.U_ev != 0.0:
        hmat = hmat + layer_potential
    subset = (int(central_pair[0]), int(central_pair[1]))
    central_evals, central_evecs = eigh(hmat, subset_by_index=subset, driver="evr")
    central_evals = np.asarray(central_evals, dtype=float)
    central_evecs = np.asarray(central_evecs, dtype=np.complex128)

    projected_sigma = central_evecs.conjugate().T @ sigma_z_operator @ central_evecs
    sigma_eigs, sigma_rot = np.linalg.eigh(projected_sigma)
    # Return positive-sigma (A-like) then negative-sigma (B-like).
    order = np.asarray([int(np.argmax(sigma_eigs)), int(np.argmin(sigma_eigs))], dtype=int)
    rot = np.asarray(sigma_rot[:, order], dtype=np.complex128)
    wavefunctions = central_evecs @ rot
    h_projected = rot.conjugate().T @ np.diag(central_evals) @ rot
    sigma_projected = rot.conjugate().T @ projected_sigma @ rot
    return wavefunctions, h_projected, sigma_projected, sigma_eigs[order]


def _hybrid_projected_basis_at_k(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    interaction: InteractionParams,
    *,
    valley: int,
    projected_indices: tuple[int, ...],
    central_pair: tuple[int, int],
    sigma_z_operator: np.ndarray,
    layer_potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected_indices_array = np.asarray(projected_indices, dtype=int)
    central_pair_array = np.asarray(central_pair, dtype=int)
    if projected_indices_array.ndim != 1 or projected_indices_array.size < 2:
        raise ValueError(f"Expected at least two projected band indices, got {projected_indices}")
    if not set(int(index) for index in central_pair).issubset(set(int(index) for index in projected_indices)):
        raise ValueError(f"Projected indices {projected_indices} must contain central pair {central_pair}")

    hmat = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if interaction.U_ev != 0.0:
        hmat = hmat + layer_potential
    subset = (int(np.min(projected_indices_array)), int(np.max(projected_indices_array)))
    evals_subset, evecs_subset = eigh(hmat, subset_by_index=subset, driver="evr")
    evals_subset = np.asarray(evals_subset, dtype=float)
    evecs_subset = np.asarray(evecs_subset, dtype=np.complex128)
    local_position = {int(index): int(index - subset[0]) for index in projected_indices_array}

    central_evals = np.asarray([evals_subset[local_position[int(index)]] for index in central_pair_array], dtype=float)
    central_evecs = np.column_stack(
        [evecs_subset[:, local_position[int(index)]] for index in central_pair_array]
    ).astype(np.complex128, copy=False)
    projected_sigma = central_evecs.conjugate().T @ sigma_z_operator @ central_evecs
    sigma_eigs, sigma_rot = np.linalg.eigh(projected_sigma)
    central_order = np.asarray([int(np.argmax(sigma_eigs)), int(np.argmin(sigma_eigs))], dtype=int)
    central_rot = np.asarray(sigma_rot[:, central_order], dtype=np.complex128)
    central_wavefunctions = central_evecs @ central_rot
    central_h = central_rot.conjugate().T @ np.diag(central_evals) @ central_rot

    lower_indices = tuple(int(index) for index in projected_indices if int(index) < int(central_pair[0]))
    upper_indices = tuple(int(index) for index in projected_indices if int(index) > int(central_pair[1]))
    ordered_vectors: list[np.ndarray] = []
    for index in lower_indices:
        ordered_vectors.append(np.asarray(evecs_subset[:, local_position[index]], dtype=np.complex128))
    for col in range(2):
        ordered_vectors.append(np.asarray(central_wavefunctions[:, col], dtype=np.complex128))
    for index in upper_indices:
        ordered_vectors.append(np.asarray(evecs_subset[:, local_position[index]], dtype=np.complex128))

    wavefunctions = np.column_stack(ordered_vectors).astype(np.complex128, copy=False)
    n_projected = wavefunctions.shape[1]
    h_projected = np.zeros((n_projected, n_projected), dtype=np.complex128)
    for out_pos, index in enumerate(lower_indices):
        h_projected[out_pos, out_pos] = float(evals_subset[local_position[index]])
    central_start = len(lower_indices)
    h_projected[central_start : central_start + 2, central_start : central_start + 2] = central_h
    for offset, index in enumerate(upper_indices):
        pos = central_start + 2 + offset
        h_projected[pos, pos] = float(evals_subset[local_position[index]])

    sigma_projected = wavefunctions.conjugate().T @ sigma_z_operator @ wavefunctions
    sigma_diagonal = np.real(np.diag(sigma_projected))
    return wavefunctions, h_projected, sigma_projected, sigma_diagonal


def centered_projection_band_indices(matrix_dim: int, projected_band_count: int) -> tuple[int, ...]:
    projected_band_count = int(projected_band_count)
    if projected_band_count < 2 or projected_band_count % 2 != 0:
        raise ValueError(f"projected_band_count must be an even integer >= 2, got {projected_band_count}")
    return tuple(int(index) for index in centered_band_indices(int(matrix_dim), projected_band_count))


def _build_htg_projected_basis_from_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    mesh_size: int,
    k_grid_frac: np.ndarray,
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    lattice = model.lattice
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    projected_indices = centered_projection_band_indices(lattice.matrix_dim, projected_band_count)
    n_projected = len(projected_indices)
    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)

    grid_shape, origin, positions = _rectangular_g_embedding(lattice)
    nx, ny = grid_shape
    embedded = np.zeros((6, nx, ny, n_projected, 2, kvec.size), dtype=np.complex128)
    h_projected = np.zeros((n_projected, n_projected, 2, kvec.size), dtype=np.complex128)
    sigma_projected = np.zeros_like(h_projected)
    band_sigma_z = np.zeros((n_projected, 2, kvec.size), dtype=float)
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)

    for iflavor, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(kvec):
            wavefunctions, h_block, sigma_block, sigma_values = _hybrid_projected_basis_at_k(
                complex(kval),
                lattice,
                model.params,
                interaction,
                valley=valley,
                projected_indices=projected_indices,
                central_pair=central_pair,
                sigma_z_operator=sigma_z_operator,
                layer_potential=layer_potential,
            )
            for source_g_index, pair in enumerate(lattice.g_indices):
                ix, iy = positions[(int(pair[0]), int(pair[1]))]
                start = 6 * source_g_index
                embedded[:, ix, iy, :, iflavor, ik] = wavefunctions[start : start + 6, :]
            h_projected[:, :, iflavor, ik] = h_block
            sigma_projected[:, :, iflavor, ik] = sigma_block
            band_sigma_z[:, iflavor, ik] = np.real(sigma_values)

    wavefunction_array = embedded.reshape((6 * nx * ny, n_projected, 2, kvec.size), order="F")
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=6,
        name="htg_chern_sublattice",
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    sigma_z = np.zeros_like(h0)
    idx = np.arange(basis.nt, dtype=int).reshape((2, 2, n_projected), order="F")
    for ik in range(basis.nk):
        for ispin in range(2):
            for iflavor in range(2):
                block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                h0[:, :, ik][np.ix_(block_indices, block_indices)] = h_projected[:, :, iflavor, ik]
                sigma_z[:, :, ik][np.ix_(block_indices, block_indices)] = sigma_projected[:, :, iflavor, ik]

    return HTGProjectedBasisData(
        model=model,
        interaction=interaction,
        mesh_size=int(mesh_size),
        kvec=kvec,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        sigma_z=sigma_z,
        band_sigma_z=band_sigma_z,
        central_band_indices=central_pair,
        projected_band_indices=projected_indices,
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_cell_area_nm2=moire_cell_area_nm2(lattice),
    )


def build_htg_projected_basis(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    mesh_size: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_mesh = resolved_interaction.n_k if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError("mesh_size must be positive")

    k_grid_frac, kvec_grid = build_moire_k_grid(model.lattice, resolved_mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid.reshape(-1), dtype=np.complex128)
    return _build_htg_projected_basis_from_kvec(
        model,
        resolved_interaction,
        kvec,
        mesh_size=resolved_mesh,
        k_grid_frac=k_grid_frac,
        projected_band_count=projected_band_count,
    )


def build_htg_projected_basis_for_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    kvec_array = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    return _build_htg_projected_basis_from_kvec(
        model,
        interaction,
        kvec_array,
        mesh_size=0,
        k_grid_frac=np.zeros((kvec_array.size, 2), dtype=float),
        projected_band_count=projected_band_count,
    )


def reciprocal_shift_labels(g_shells: int) -> tuple[int, ...]:
    g_shells = int(g_shells)
    if g_shells < 0:
        raise ValueError("g_shells must be non-negative")
    return tuple(range(-g_shells, g_shells + 1))


def _infer_g_shells_from_overlap_blocks(overlap_blocks: HFOverlapBlockSet) -> int:
    if not overlap_blocks.shifts:
        return 0
    return int(max(max(abs(int(m)), abs(int(n))) for m, n in overlap_blocks.shifts))


def build_htg_overlap_blocks(
    basis_data: HTGProjectedBasisData,
    *,
    g_shells: int | None = None,
) -> HFOverlapBlockSet:
    interaction = basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray(
        [m * basis_data.model.lattice.b_m1 + n * basis_data.model.lattice.b_m2 for m, n in shifts],
        dtype=np.complex128,
    )
    overlaps = {
        shift: calculate_projected_overlap_between(basis_data.basis, basis_data.basis, shift[0], shift[1])
        for shift in shifts
    }
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        overlap = overlaps[shift]
        diagonal_overlaps[shift] = np.diagonal(overlap, axis1=1, axis2=3)
        hartree_screening[shift] = float(screened_coulomb_matrix(np.asarray(gvec), interaction))
        qvals = basis_data.kvec[None, :] - basis_data.kvec[:, None] + complex(gvec)
        fock_screening[shift] = screened_coulomb_matrix(qvals, interaction)

    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_htg_overlap_blocks_between(
    target_basis_data: HTGProjectedBasisData,
    source_basis_data: HTGProjectedBasisData,
    *,
    g_shells: int | None = None,
    include_hartree: bool = True,
) -> HFOverlapBlockSet:
    if target_basis_data.model.lattice is not source_basis_data.model.lattice:
        target_lattice = target_basis_data.model.lattice
        source_lattice = source_basis_data.model.lattice
        if not np.array_equal(target_lattice.g_indices, source_lattice.g_indices):
            raise ValueError("Target and source HTG bases must use the same plane-wave G-index set")
    if target_basis_data.basis.grid_shape != source_basis_data.basis.grid_shape:
        raise ValueError("Target and source HTG projected bases must use the same reciprocal embedding grid")

    interaction = target_basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray(
        [m * target_basis_data.model.lattice.b_m1 + n * target_basis_data.model.lattice.b_m2 for m, n in shifts],
        dtype=np.complex128,
    )
    overlaps = {
        shift: calculate_projected_overlap_between(
            target_basis_data.basis,
            source_basis_data.basis,
            shift[0],
            shift[1],
        )
        for shift in shifts
    }
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if target_basis_data.nk == source_basis_data.nk:
            diagonal_overlaps[shift] = np.diagonal(overlaps[shift], axis1=1, axis2=3)
        if include_hartree:
            hartree_screening[shift] = float(screened_coulomb_matrix(np.asarray(gvec), interaction))
        qvals = source_basis_data.kvec[None, :] - target_basis_data.kvec[:, None] + complex(gvec)
        fock_screening[shift] = screened_coulomb_matrix(qvals, interaction)

    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_htg_interaction_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGInteractionComponents:
    hartree_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
    )
    fock_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        fock_screening=overlap_blocks.fock_screening,
    )
    hartree = build_projected_interaction_hamiltonian(
        density,
        hartree_blocks,
        v0=v0,
        beta=beta,
        use_numba=use_numba,
    )
    fock = build_projected_interaction_hamiltonian(
        density,
        fock_blocks,
        v0=v0,
        beta=beta,
        use_numba=use_numba,
    )
    total = hartree + fock
    hartree_eigs = np.zeros((hartree.shape[0], hartree.shape[2]), dtype=float)
    fock_eigs = np.zeros_like(hartree_eigs)
    for ik in range(hartree.shape[2]):
        hartree_eigs[:, ik] = np.linalg.eigvalsh(hartree[:, :, ik])
        fock_eigs[:, ik] = np.linalg.eigvalsh(fock[:, :, ik])
    return HTGInteractionComponents(
        hartree=hartree,
        fock=fock,
        total=total,
        hartree_eigenvalues=hartree_eigs,
        fock_eigenvalues=fock_eigs,
    )


def _hartree_only_blocks(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
    )


def _fock_only_blocks(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        fock_screening=overlap_blocks.fock_screening,
    )


def _flavor_band_diagonal(matrix: np.ndarray, *, n_spin: int, n_eta: int, n_band: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    nt, nt_rhs, nk = matrix.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square matrix blocks, got {matrix.shape}")
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"Matrix dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    diagonal = np.zeros((n_spin, n_eta, n_band, nk), dtype=float)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                row = int(idx[ispin, ieta, iband])
                diagonal[ispin, ieta, iband, :] = np.real(matrix[row, row, :])
    return diagonal


def evaluate_htg_interaction_path(
    hf_run: HTGHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGInteractionPathResult:
    source_basis_data = hf_run.basis_data
    resolved_g_shells = _infer_g_shells_from_overlap_blocks(hf_run.overlap_blocks) if g_shells is None else int(g_shells)
    resolved_path = (
        source_basis_data.model.paper_hf_kpath(points_per_segment=points_per_segment)
        if path is None
        else path
    )
    path_basis_data = build_htg_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        projected_band_count=hf_run.state.n_band,
    )
    source_overlap_blocks = hf_run.overlap_blocks
    target_overlap_blocks = build_htg_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    zero_base = np.zeros_like(path_basis_data.h0)
    hartree = build_projected_target_hamiltonian(
        zero_base,
        hf_run.state.density,
        source_overlap_blocks=_hartree_only_blocks(source_overlap_blocks),
        target_overlap_blocks=_hartree_only_blocks(target_overlap_blocks),
        target_source_overlap_blocks=HFOverlapBlockSet(
            shifts=target_source_overlap_blocks.shifts,
            gvecs=target_source_overlap_blocks.gvecs,
            overlaps=target_source_overlap_blocks.overlaps,
        ),
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    fock = build_projected_target_hamiltonian(
        zero_base,
        hf_run.state.density,
        source_overlap_blocks=HFOverlapBlockSet(
            shifts=source_overlap_blocks.shifts,
            gvecs=source_overlap_blocks.gvecs,
            overlaps=source_overlap_blocks.overlaps,
        ),
        target_overlap_blocks=HFOverlapBlockSet(
            shifts=target_overlap_blocks.shifts,
            gvecs=target_overlap_blocks.gvecs,
            overlaps=target_overlap_blocks.overlaps,
        ),
        target_source_overlap_blocks=_fock_only_blocks(target_source_overlap_blocks),
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    total = hartree + fock
    return HTGInteractionPathResult(
        path=resolved_path,
        hartree=hartree,
        fock=fock,
        total=total,
        hartree_diagonal_ev=_flavor_band_diagonal(
            hartree,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        fock_diagonal_ev=_flavor_band_diagonal(
            fock,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        total_diagonal_ev=_flavor_band_diagonal(
            total,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )


def _diagonalize_path_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nt, _, nk = hamiltonian.shape
    energies = np.zeros((nk, nt), dtype=float)
    sigma_z_expectation = np.zeros((nk, nt), dtype=float)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[ik, :] = eigvals
        sigma_z_expectation[ik, :] = np.real(np.diag(eigvecs.conjugate().T @ sigma_z[:, :, ik] @ eigvecs))
    return energies, sigma_z_expectation


def evaluate_htg_hf_path(
    hf_run: HTGHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGHFPathResult:
    source_basis_data = hf_run.basis_data
    resolved_g_shells = _infer_g_shells_from_overlap_blocks(hf_run.overlap_blocks) if g_shells is None else int(g_shells)
    resolved_path = (
        source_basis_data.model.paper_hf_kpath(points_per_segment=points_per_segment)
        if path is None
        else path
    )
    path_basis_data = build_htg_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        projected_band_count=hf_run.state.n_band,
    )
    source_overlap_blocks = hf_run.overlap_blocks
    target_overlap_blocks = build_htg_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    h_path = build_projected_target_hamiltonian(
        path_basis_data.h0,
        hf_run.state.density,
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    energies, sigma_z_expectation = _diagonalize_path_hamiltonian(h_path, path_basis_data.sigma_z)
    band_data = build_flavor_band_data(
        h_path,
        n_spin=hf_run.state.n_spin,
        n_eta=hf_run.state.n_eta,
        n_band=hf_run.state.n_band,
    )
    return HTGHFPathResult(
        path=resolved_path,
        hamiltonian=h_path,
        energies=energies,
        sigma_z_expectation=sigma_z_expectation,
        sigma_z_operator=path_basis_data.sigma_z,
        band_data=band_data,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )


def compute_background_density(diagonal_overlap: np.ndarray) -> complex:
    diagonal_overlap = np.asarray(diagonal_overlap, dtype=np.complex128)
    if diagonal_overlap.ndim != 3 or diagonal_overlap.shape[0] != diagonal_overlap.shape[1]:
        raise ValueError(f"Expected diagonal overlap shape (nt, nt, nk), got {diagonal_overlap.shape}")
    nt, _, nk = diagonal_overlap.shape
    return complex(np.trace(diagonal_overlap, axis1=0, axis2=1).sum() / float(nt * nk))


def compute_background_densities(overlap_blocks: HFOverlapBlockSet) -> dict[tuple[int, int], complex]:
    return {
        shift: compute_background_density(diagonal)
        for shift, diagonal in overlap_blocks.diagonal_overlaps.items()
    }


def _update_htg_diagnostics_from_density(state: HTGHartreeFockState) -> None:
    state.diagnostics["filling"] = htg_filling_from_density(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    state.diagnostics["projector_idempotency_residual"] = projector_idempotency_residual(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


def _update_htg_hf_density_update_state(state: HTGHartreeFockState, density_update: DensityUpdateResult) -> None:
    _update_htg_diagnostics_from_density(state)
    occupation_mask = density_update.observables.get("occupation_mask")
    if occupation_mask is not None:
        sector_gap = htg_gap_from_occupation_mask(
            state.energies,
            np.asarray(occupation_mask, dtype=bool),
        )
        state.diagnostics["sector_gap"] = sector_gap
        state.diagnostics["hf_gap"] = sector_gap
    else:
        state.diagnostics["hf_gap"] = htg_gap_estimate(state.energies, state.nu)
    state.diagnostics["hamiltonian_hermitian_residual"] = hermitian_residual(state.hamiltonian)
    sigma_z = density_update.observables.get("sigma_z")
    if sigma_z is not None:
        occupied = (
            np.asarray(occupation_mask, dtype=bool)
            if occupation_mask is not None
            else occupied_state_mask(
                state.energies,
                htg_occupied_state_count(state.nu, state.nt, state.nk, n_spin=state.n_spin, n_eta=state.n_eta),
            )
        )
        if np.any(occupied):
            state.diagnostics["occupied_sigma_z_mean"] = float(np.mean(np.asarray(sigma_z, dtype=float)[occupied]))


def _update_htg_hf_step_state(state: HTGHartreeFockState, step: HartreeFockStepResult) -> None:
    _update_htg_hf_density_update_state(state, step.density_update)


def build_htg_hf_kernel(
    state: HTGHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    use_numba: bool | None = None,
):
    return build_projected_hf_kernel(
        state,
        overlap_blocks,
        density_builder=HTGDensityBuilder(
            state.nu,
            sigma_z=state.sigma_z,
            occupation_counts=state.occupation_counts,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        ),
        energy_functional=compute_hf_energy,
        oda_parameterizer="default",
        step_callback=_update_htg_hf_step_state,
        final_state_callback=_update_htg_hf_density_update_state,
        convergence_rule="raw",
        v0=state.v0,
        beta=beta,
        use_numba=use_numba,
    )


def run_htg_hf(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    g_shells: int | None = None,
    projected_band_count: int = 2,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> HTGHartreeFockRun:
    normalized_init_mode = normalize_htg_init_mode(init_mode)
    basis_data = build_htg_projected_basis(
        model,
        interaction,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    occupation_counts = htg_flavor_occupation_counts_for_init_mode(
        normalized_init_mode,
        nu=nu,
        seed=seed,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    state = HTGHartreeFockState.from_projected_basis(
        basis_data,
        nu=nu,
        precision=precision,
        occupation_counts=occupation_counts,
    )
    overlap_blocks = build_htg_overlap_blocks(basis_data, g_shells=g_shells)
    base_run = run_projected_hartree_fock(
        state,
        initializer=HTGInitializer(initial_density=initial_density),
        density_builder=HTGDensityBuilder(
            nu,
            sigma_z=state.sigma_z,
            occupation_counts=occupation_counts,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        ),
        overlap_blocks=overlap_blocks,
        init_mode=normalized_init_mode,
        seed=seed,
        v0=state.v0,
        beta=beta,
        energy_functional=compute_hf_energy,
        oda_parameterizer="default",
        step_callback=_update_htg_hf_step_state,
        final_state_callback=_update_htg_hf_density_update_state,
        convergence_rule="raw",
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        use_numba=use_numba,
    )
    return HTGHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def scan_htg_ground_state(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    nu: float,
    init_modes: Iterable[str] = ("fb", "fi", "flavor", "vp", "sp", "bm", "perturbed", "random"),
    seeds: Iterable[int] = tuple(range(1, 9)),
    beta: float = 1.0,
    max_iter: int = 300,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    g_shells: int | None = None,
    projected_band_count: int = 2,
    use_numba: bool | None = None,
) -> HTGGroundStateScan:
    basis_data = build_htg_projected_basis(
        model,
        interaction,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    overlap_blocks = build_htg_overlap_blocks(basis_data, g_shells=g_shells)
    runs: list[HTGHartreeFockRun] = []
    for init_mode in init_modes:
        normalized = normalize_htg_init_mode(init_mode)
        for seed in seeds:
            occupation_counts = htg_flavor_occupation_counts_for_init_mode(
                normalized,
                nu=nu,
                seed=int(seed),
                n_spin=basis_data.basis.n_spin,
                n_eta=basis_data.basis.n_flavor,
                n_band=basis_data.basis.n_band,
            )
            state = HTGHartreeFockState.from_projected_basis(
                basis_data,
                nu=nu,
                precision=precision,
                occupation_counts=occupation_counts,
            )
            base_run = run_projected_hartree_fock(
                state,
                initializer=HTGInitializer(),
                density_builder=HTGDensityBuilder(
                    nu,
                    sigma_z=state.sigma_z,
                    occupation_counts=occupation_counts,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                    n_band=state.n_band,
                ),
                overlap_blocks=overlap_blocks,
                init_mode=normalized,
                seed=int(seed),
                v0=state.v0,
                beta=beta,
                energy_functional=compute_hf_energy,
                oda_parameterizer="default",
                step_callback=_update_htg_hf_step_state,
                final_state_callback=_update_htg_hf_density_update_state,
                convergence_rule="raw",
                max_iter=max_iter,
                oda_stall_threshold=oda_stall_threshold,
                use_numba=use_numba,
            )
            runs.append(
                HTGHartreeFockRun(
                    state=state,
                    overlap_blocks=overlap_blocks,
                    basis_data=basis_data,
                    iter_energy=base_run.iter_energy,
                    iter_err=base_run.iter_err,
                    iter_oda=base_run.iter_oda,
                    init_mode=base_run.init_mode,
                    seed=base_run.seed,
                    converged=base_run.converged,
                    exit_reason=base_run.exit_reason,
                )
            )
    return HTGGroundStateScan(runs=tuple(runs))
