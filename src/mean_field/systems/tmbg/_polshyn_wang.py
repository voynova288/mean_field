from __future__ import annotations

from ._polshyn_shared import *  # noqa: F401,F403
from ._polshyn_types import *  # noqa: F401,F403
from ._polshyn_filling import *  # noqa: F401,F403

def wang_stored_density_from_sector_blocks(density_blocks: np.ndarray) -> np.ndarray:
    """Convert conventional sector density blocks to Wang/Xiaoyu stored layout."""

    return flatten_sector_blocks(np.conj(np.asarray(density_blocks, dtype=np.complex128)))


def scaled_overlap_blocks(
    overlap_blocks: HFOverlapBlockSet,
    *,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
) -> HFOverlapBlockSet:
    """Return an overlap table with Hartree/Fock kernels rescaled for diagnostics."""

    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={tuple(shift): float(hartree_scale) * float(value) for shift, value in overlap_blocks.hartree_screening.items()},
        fock_screening={tuple(shift): float(fock_scale) * np.asarray(value, dtype=float) for shift, value in overlap_blocks.fock_screening.items()},
    )


def overlap_blocks_with_hartree_q0_zeroed(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    """Return an overlap table with only the uniform Hartree shift removed."""

    hartree = {
        tuple(shift): (0.0 if tuple(shift) == (0, 0) else float(value))
        for shift, value in overlap_blocks.hartree_screening.items()
    }
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=hartree,
        fock_screening=overlap_blocks.fock_screening,
    )





def wang_projected_wavefunction_basis(basis: PolshynProjectedBasis) -> ProjectedWavefunctionBasis:
    """View a Polshyn projected basis in the generic Wang/Xiaoyu HF layout."""

    return ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(basis.wavefunctions, dtype=np.complex128),
        grid_shape=tuple(basis.embedding_shape),
        n_spin=int(basis.n_spin),
        local_basis_size=int(basis.local_basis_size),
        name="polshyn_doubled",
        boundary_mode="zero_fill",
    )


def _flat_sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    return _core_flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)

def flatten_sector_blocks(blocks: np.ndarray) -> np.ndarray:
    """Flatten (spin, valley, band, band, k) blocks to Wang nt x nt x nk layout."""

    return _core_flatten_sector_blocks(blocks)

def unflatten_sector_blocks(flat: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_blocks(flat, n_spin=n_spin, n_eta=n_eta, nb=nb)

def unflatten_sector_energies(flat_energies: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_energies(flat_energies, n_spin=n_spin, n_eta=n_eta, nb=nb)

def wang_density_from_fixed_sector_occupations(
    hamiltonian_flat: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> DensityUpdateResult:
    """Fixed-sector density update in Wang/Xiaoyu's stored-projector convention.

    The generic validated HF kernel stores the transpose/conjugate projector
    convention used by the original Wang/Xiaoyu code, namely ``P_store = P*``.
    This differs from the legacy Polshyn helper, which stored the conventional
    density matrix.  Keeping this builder separate prevents convention mixing.
    """

    h = np.asarray(hamiltonian_flat, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square flattened Hamiltonian, got {h.shape}")
    if nt != int(n_spin) * int(n_eta) * int(nb):
        raise ValueError(f"Flattened dimension {nt} incompatible with {(n_spin, n_eta, nb)}")
    occ = np.asarray(occupation_counts, dtype=int)
    if occ.shape != (int(n_spin), int(n_eta)):
        raise ValueError(f"occupation_counts shape {occ.shape} incompatible with {(n_spin, n_eta)}")
    reference = np.asarray(reference_diagonal, dtype=float)
    if reference.shape != (int(nb),):
        raise ValueError(f"reference_diagonal shape {reference.shape} incompatible with nb={nb}")
    ref_mat = np.diag(reference).astype(np.complex128)
    density = np.zeros_like(h)
    energies = np.zeros((nt, int(nk)), dtype=float)
    sector_energies = np.zeros((int(n_spin), int(n_eta), int(nb), int(nk)), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = _flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            n_occ = int(occ[ispin, ieta])
            for ik in range(int(nk)):
                block_h = h[idx[:, None], idx[None, :], ik]
                block_h = 0.5 * (block_h + block_h.conjugate().T)
                evals, evecs = np.linalg.eigh(block_h)
                energies[idx, ik] = evals
                sector_energies[ispin, ieta, :, ik] = evals
                if n_occ == 0:
                    projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
                elif n_occ == int(nb):
                    projector = np.eye(int(nb), dtype=np.complex128)
                else:
                    vecs = evecs[:, :n_occ]
                    projector = vecs.conj() @ vecs.T
                density[idx[:, None], idx[None, :], ik] = projector - ref_mat
    mu = estimate_fermi_level_from_sector_energies(sector_energies, occ)
    return DensityUpdateResult(density=density, energies=energies, mu=float(mu))


def build_wang_overlap_blocks(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shifts: Iterable[tuple[int, int]],
    gvecs: np.ndarray,
    *,
    epsilon_r: float,
    d_sc_nm: float,
    include_hartree: bool = True,
    include_fock: bool = True,
    progress_prefix: str | None = None,
) -> HFOverlapBlockSet:
    """Build dense generic HF overlap blocks for the Wang/Xiaoyu engine."""

    target_core = wang_projected_wavefunction_basis(target)
    source_core = wang_projected_wavefunction_basis(source)
    shift_tuple = tuple(tuple(shift) for shift in shifts)
    gvec_array = np.asarray(gvecs, dtype=np.complex128)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for ishift, (shift, gvec) in enumerate(zip(shift_tuple, gvec_array, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shift_tuple) or ishift % 10 == 0):
            print(f"{progress_prefix} wang overlap {ishift}/{len(shift_tuple)} shift={shift}", flush=True)
        overlap = calculate_projected_overlap_between(target_core, source_core, int(shift[0]), int(shift[1]))
        overlaps[shift] = overlap
        if target.nk == source.nk:
            diagonal_overlaps[shift] = diagonal_overlap_blocks(overlap, nt=target_core.nt, nk=target.nk)
        if include_hartree:
            hartree_screening[shift] = screened_coulomb(complex(gvec), epsilon_r=float(epsilon_r), d_sc_nm=float(d_sc_nm))
        if include_fock:
            fock_screening[shift] = screened_coulomb_matrix(
                source.kvec[None, :] - target.kvec[:, None] + complex(gvec),
                epsilon_r=float(epsilon_r),
                d_sc_nm=float(d_sc_nm),
            )
    return HFOverlapBlockSet(
        shifts=shift_tuple,
        gvecs=gvec_array,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_wang_hf_problem(
    state: PolshynWangHFState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> HartreeFockProblem:
    """Build the shared core-HF problem wrapper for Wang/Xiaoyu tMBG HF."""

    def interaction_builder(density_flat_in: np.ndarray) -> np.ndarray:
        return build_projected_interaction_hamiltonian(
            density_flat_in,
            overlap_blocks,
            v0=float(state.v0),
            beta=1.0,
        )

    def density_builder(hamiltonian_flat: np.ndarray) -> DensityUpdateResult:
        return wang_density_from_fixed_sector_occupations(
            hamiltonian_flat,
            occupation_counts,
            reference_diagonal,
            n_spin=n_spin,
            n_eta=n_eta,
            nb=nb,
        )

    return HartreeFockProblem(
        initializer=lambda _state, *, init_mode, seed: None,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=compute_hf_energy,
            oda_delta_interaction_builder=interaction_builder,
            convergence_rule="mixed",
        ),
    )


def run_projected_hf_scf_wang(
    basis: PolshynProjectedBasis,
    *,
    occupation_counts: np.ndarray,
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    max_iter: int = 80,
    precision: float = 1e-6,
    initial_density_blocks: np.ndarray | None = None,
    oda_stall_threshold: float = 1.0e-4,
    progress_prefix: str | None = None,
    overlap_blocks: HFOverlapBlockSet | None = None,
    seed: int = 0,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
    zero_hartree_q0: bool = False,
) -> tuple[PolshynWangHFState, HFOverlapBlockSet, dict[str, Any]]:
    """Run Polshyn projected HF through the generic Wang/Xiaoyu ODA engine."""

    if overlap_blocks is None:
        overlap_blocks = build_wang_overlap_blocks(
            basis,
            basis,
            shifts,
            gvecs,
            epsilon_r=epsilon_r,
            d_sc_nm=d_sc_nm,
            include_hartree=True,
            include_fock=True,
            progress_prefix=progress_prefix,
        )
    if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
        overlap_blocks = scaled_overlap_blocks(
            overlap_blocks,
            hartree_scale=float(hartree_scale),
            fock_scale=float(fock_scale),
        )
    if bool(zero_hartree_q0):
        overlap_blocks = overlap_blocks_with_hartree_q0_zeroed(overlap_blocks)
    h0_flat = flatten_sector_blocks(basis.h0_blocks)
    if initial_density_blocks is None:
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        density_flat = init_update.density
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "bm_wang"
    else:
        # Existing initializers create conventional Hermitian density matrices;
        # Wang/Xiaoyu's kernel stores P* instead.
        density_flat = flatten_sector_blocks(np.conj(np.asarray(initial_density_blocks, dtype=np.complex128)))
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "provided_wang"

    state = PolshynWangHFState(
        h0=h0_flat.copy(),
        density=density_flat.copy(),
        hamiltonian=h0_flat.copy(),
        energies=np.asarray(energies, dtype=float).copy(),
        mu=float(mu),
        precision=float(precision),
        v0=float(v0),
        diagnostics={},
    )

    problem = build_wang_hf_problem(
        state,
        overlap_blocks,
        occupation_counts=occupation_counts,
        reference_diagonal=basis.reference_diagonal,
        n_spin=basis.n_spin,
        n_eta=basis.n_eta,
        nb=basis.nb,
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
    )
    iteration_history = [
        {
            "iteration": int(idx + 1),
            "energy": float(run.iter_energy[idx]) if idx < len(run.iter_energy) else None,
            "error": float(run.iter_err[idx]) if idx < len(run.iter_err) else None,
            "oda_lambda": float(run.iter_oda[idx]) if idx < len(run.iter_oda) else None,
        }
        for idx in range(max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda)))
    ]
    info: dict[str, Any] = {
        "mode": "polshyn_projected_hf_wang",
        "iterations": int(run.iterations),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "final_raw_norm": float(state.diagnostics.get("final_raw_norm", float("nan"))),
        "init_mode": init_mode,
        "seed": int(seed),
        "precision": float(precision),
        "oda_stall_threshold": float(oda_stall_threshold),
        "final_interaction_norm_ev": float(np.linalg.norm(state.hamiltonian - state.h0)),
        "hf_energy": float(state.diagnostics.get("hf_energy", float("nan"))),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
        "zero_hartree_q0": bool(zero_hartree_q0),
        "iteration_history": iteration_history,
    }
    if run.iter_oda.size:
        info["last_oda_lambda"] = float(run.iter_oda[-1])
        info["min_oda_lambda"] = float(np.min(run.iter_oda))
    return state, overlap_blocks, info


def wang_sector_density_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    """Return conventional sector density blocks from a Wang/Xiaoyu HF state."""

    return np.conj(unflatten_sector_blocks(state.density, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb))


def wang_sector_hamiltonian_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_blocks(state.hamiltonian, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)


def wang_sector_energy_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_energies(state.energies, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)




def translation_order_parameters(
    density_blocks: np.ndarray,
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    spin_index: int = 0,
    valley_index: int = 0,
) -> dict[str, np.ndarray | float]:
    """Fold-off-diagonal CDW order diagnostic for the doubled supercell.

    In the area-2 folded basis, fold 0 and fold 1 differ by the paper's
    translation-breaking wavevector Q=B1.  The quantity analogous to Polshyn
    Eq. (1) is therefore the norm of density-matrix elements connecting even
    and odd folded copies at the same supercell k.  A single target-band state
    (|fold0>+|fold1>)/sqrt(2) has |rho_01|=1/2, so the reported ``*_x2``
    values are normalized to have maximal target-band order near one.
    """

    density = np.asarray(density_blocks, dtype=np.complex128)
    projected_indices = tuple(int(index) for index in projected_indices)
    target_pos = projected_indices.index(int(target_band_index))
    fold0 = np.asarray([2 * iprim for iprim in range(len(projected_indices))], dtype=int)
    fold1 = fold0 + 1
    sector = density[int(spin_index), int(valley_index)]
    target_raw = np.abs(sector[2 * target_pos, 2 * target_pos + 1, :])
    all_raw = np.sqrt(np.sum(np.abs(sector[np.ix_(fold0, fold1, np.arange(sector.shape[-1]))]) ** 2, axis=(0, 1)))
    return {
        "target_raw": np.asarray(target_raw, dtype=float),
        "all_raw": np.asarray(all_raw, dtype=float),
        "target_x2": np.asarray(2.0 * target_raw, dtype=float),
        "all_x2": np.asarray(2.0 * all_raw, dtype=float),
        "target_x2_min": float(np.min(2.0 * target_raw)),
        "target_x2_mean": float(np.mean(2.0 * target_raw)),
        "target_x2_max": float(np.max(2.0 * target_raw)),
        "all_x2_min": float(np.min(2.0 * all_raw)),
        "all_x2_mean": float(np.mean(2.0 * all_raw)),
        "all_x2_max": float(np.max(2.0 * all_raw)),
    }


def estimate_fermi_level_from_sector_energies(energies: np.ndarray, occupation_counts: np.ndarray) -> float:
    vals = np.asarray(energies, dtype=float)
    occ = np.asarray(occupation_counts, dtype=int)
    occupied_max: list[float] = []
    empty_min: list[float] = []
    for ispin in range(vals.shape[0]):
        for ieta in range(vals.shape[1]):
            n_occ = int(occ[ispin, ieta])
            if n_occ > 0:
                occupied_max.append(float(np.max(vals[ispin, ieta, n_occ - 1, :])))
            if n_occ < vals.shape[2]:
                empty_min.append(float(np.min(vals[ispin, ieta, n_occ, :])))
    if occupied_max and empty_min:
        return 0.5 * (max(occupied_max) + min(empty_min))
    if occupied_max:
        return max(occupied_max)
    if empty_min:
        return min(empty_min)
    return 0.0


def moire_cell_area_nm2(lattice: TMBGLattice, *, area_ratio: int = 1) -> float:
    primitive_area = real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)
    return float(area_ratio) * float(primitive_area)

__all__ = [name for name in globals() if not name.startswith('__')]
