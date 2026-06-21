from __future__ import annotations

from ._supercell_shared import *  # noqa: F401,F403
from ._supercell_types import *  # noqa: F401,F403
from ._supercell_geometry import *  # noqa: F401,F403
from ._supercell_basis import *  # noqa: F401,F403

def _density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    primitive_nu: float,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> DensityUpdateResult:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    reference_density = _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=reference_diagonal,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    n_occ = htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=reference_diagonal,
        area_ratio=area_ratio,
        n_sector=int(n_spin) * int(n_eta),
    )
    density = np.zeros_like(hamiltonian)
    energies = np.zeros((nt, nk), dtype=float)
    occ_mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        if n_occ > 0:
            occupied_vecs = eigvecs[:, :n_occ]
            density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
            occ_mask[:n_occ, ik] = True
        else:
            density[:, :, ik] = -reference_density[:, :, ik]
    mu = find_chemical_potential(energies, float(n_occ) / float(nt))
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=float(mu),
        observables={"occupation_mask": occ_mask},
    )


@dataclass(frozen=True)
class HTGSupercellDensityBuilder:
    primitive_nu: float
    reference_diagonal: np.ndarray
    area_ratio: int
    n_spin: int = 2
    n_eta: int = 2

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        return _density_from_hamiltonian(
            hamiltonian,
            primitive_nu=self.primitive_nu,
            reference_diagonal=self.reference_diagonal,
            area_ratio=self.area_ratio,
            n_spin=self.n_spin,
            n_eta=self.n_eta,
        )


@dataclass(frozen=True)
class HTGSupercellInitializer:
    initial_density: np.ndarray | None = None

    def __call__(self, state: HTGSupercellHartreeFockState, *, init_mode: str, seed: int) -> None:
        if self.initial_density is not None:
            density = np.asarray(self.initial_density, dtype=np.complex128)
            if density.shape != state.density.shape:
                raise ValueError(f"Expected initial_density shape {state.density.shape}, got {density.shape}")
            state.density[:, :, :] = density
        else:
            state.density[:, :, :] = initialize_htg_supercell_density(state, init_mode=init_mode, seed=seed)
        _update_supercell_diagnostics_from_density(state)


def initialize_htg_supercell_density(
    state: HTGSupercellHartreeFockState,
    *,
    init_mode: str = "bm",
    seed: int = 1,
) -> np.ndarray:
    mode = str(init_mode).strip().lower().replace("-", "_")
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    if mode in {"bm", "noninteracting", "fermi"}:
        return _density_from_hamiltonian(
            state.h0,
            primitive_nu=state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ).density

    rng = np.random.default_rng(seed)
    nt, _, nk = state.h0.shape
    reference_density = _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=state.reference_diagonal,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    n_occ = htg_supercell_occupied_count_per_k(
        state.nu,
        reference_diagonal=state.reference_diagonal,
        area_ratio=area_ratio,
        n_sector=state.n_spin * state.n_eta,
    )
    if mode in {"random", "diag_random"}:
        density = np.zeros_like(state.h0)
        for ik in range(nk):
            unitary = random_unitary_from_hermitian(nt, rng)
            occupied_vecs = unitary[:, :n_occ]
            density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density
    if mode in {"perturbed", "cdw", "fold_random"}:
        perturbed = np.asarray(state.h0, dtype=np.complex128).copy()
        scale = 1.0e-3
        for ik in range(nk):
            noise = rng.standard_normal((nt, nt)) + 1j * rng.standard_normal((nt, nt))
            noise = 0.5 * (noise + noise.conjugate().T)
            perturbed[:, :, ik] += scale * noise
        return _density_from_hamiltonian(
            perturbed,
            primitive_nu=state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ).density
    raise ValueError(f"Unsupported HTG supercell init_mode={init_mode!r}; use bm, perturbed, or random")


def _infer_primitive_band_count_from_reference(reference_diagonal: np.ndarray) -> int:
    # The current HTG supercell adapter folds each primitive projected band into
    # the same number of copies and stores them contiguously.  The default and
    # production path use the central two-band window; use the shortest even
    # primitive window consistent with a repeated reference pattern.
    reference = np.asarray(reference_diagonal, dtype=float).reshape(-1)
    for primitive_count in range(2, reference.size + 1, 2):
        if reference.size % primitive_count != 0:
            continue
        repeats = reference.size // primitive_count
        compressed = reference[::repeats]
        if np.allclose(reference, np.repeat(compressed, repeats)):
            return int(primitive_count)
    raise ValueError("Could not infer primitive projected band count from supercell reference")


def _supercell_gap_from_mask(energies: np.ndarray, occupation_mask: np.ndarray) -> float:
    energies = np.asarray(energies, dtype=float)
    occupied = np.asarray(occupation_mask, dtype=bool)
    if not np.any(occupied) or np.all(occupied):
        return float("nan")
    return float(np.min(energies[~occupied]) - np.max(energies[occupied]))


def _update_supercell_diagnostics_from_density(state: HTGSupercellHartreeFockState) -> None:
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    state.diagnostics["filling"] = htg_supercell_filling_from_density(
        state.density,
        reference_diagonal=state.reference_diagonal,
        area_ratio=area_ratio,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    projector = state.density + _supercell_reference_density_blocks(
        state.nt,
        state.nk,
        reference_diagonal=state.reference_diagonal,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    residual = 0.0
    for ik in range(state.nk):
        block = projector[:, :, ik]
        residual = max(residual, float(np.max(np.abs(block @ block - block))))
    state.diagnostics["projector_idempotency_residual"] = float(residual)


def _update_supercell_density_update_state(state: HTGSupercellHartreeFockState, density_update: DensityUpdateResult) -> None:
    _update_supercell_diagnostics_from_density(state)
    occupation_mask = density_update.observables.get("occupation_mask")
    if occupation_mask is not None:
        state.diagnostics["hf_gap"] = _supercell_gap_from_mask(state.energies, np.asarray(occupation_mask, dtype=bool))
    state.diagnostics["hamiltonian_hermitian_residual"] = hermitian_residual(state.hamiltonian)


def _update_supercell_step_state(state: HTGSupercellHartreeFockState, step) -> None:
    _update_supercell_density_update_state(state, step.density_update)


def build_htg_supercell_hf_kernel(
    state: HTGSupercellHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    use_numba: bool | None = None,
):
    primitive_band_count = _infer_primitive_band_count_from_reference(state.reference_diagonal)
    area_ratio = int(state.n_band // primitive_band_count)
    return build_projected_hf_kernel(
        state,
        overlap_blocks,
        density_builder=HTGSupercellDensityBuilder(
            state.nu,
            reference_diagonal=state.reference_diagonal,
            area_ratio=area_ratio,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ),
        energy_functional=compute_hf_energy,
        oda_parameterizer="default",
        step_callback=_update_supercell_step_state,
        final_state_callback=_update_supercell_density_update_state,
        convergence_rule="raw",
        v0=state.v0,
        beta=beta,
        use_numba=use_numba,
    )


def build_htg_supercell_hf_problem(
    state: HTGSupercellHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> HartreeFockProblem:
    return build_projected_hf_problem(
        initializer=HTGSupercellInitializer(initial_density=initial_density),
        kernel=build_htg_supercell_hf_kernel(state, overlap_blocks, beta=beta, use_numba=use_numba),
    )


def run_htg_supercell_hf(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    primitive_nu: float,
    supercell: HTGSupercell | None = None,
    init_mode: str = "perturbed",
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
) -> HTGSupercellHartreeFockRun:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_minimal_fractional_supercell(primitive_nu) if supercell is None else supercell
    basis_data = build_htg_supercell_projected_basis(
        model,
        resolved_interaction,
        supercell=resolved_supercell,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    # Validate the requested filling before any SCF work.
    htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=basis_data.reference_diagonal,
        area_ratio=basis_data.supercell.area_ratio,
        n_sector=basis_data.basis.n_spin * basis_data.basis.n_flavor,
    )
    state = HTGSupercellHartreeFockState.from_projected_basis(basis_data, nu=primitive_nu, precision=precision)
    overlap_blocks = build_htg_supercell_overlap_blocks(basis_data, g_shells=g_shells)
    problem = build_htg_supercell_hf_problem(
        state,
        overlap_blocks,
        beta=beta,
        initial_density=initial_density,
        use_numba=use_numba,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=str(init_mode),
        seed=int(seed),
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
    )
    return HTGSupercellHartreeFockRun(
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


def scan_htg_supercell_ground_state(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    primitive_nu: float,
    supercell: HTGSupercell | None = None,
    init_modes: Iterable[str] = ("perturbed", "random", "bm"),
    seeds: Iterable[int] = (1, 2, 3),
    beta: float = 1.0,
    max_iter: int = 300,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    g_shells: int | None = None,
    projected_band_count: int = 2,
    use_numba: bool | None = None,
) -> HTGSupercellGroundStateScan:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_minimal_fractional_supercell(primitive_nu) if supercell is None else supercell
    basis_data = build_htg_supercell_projected_basis(
        model,
        resolved_interaction,
        supercell=resolved_supercell,
        mesh_size=mesh_size,
        projected_band_count=projected_band_count,
    )
    htg_supercell_occupied_count_per_k(
        primitive_nu,
        reference_diagonal=basis_data.reference_diagonal,
        area_ratio=basis_data.supercell.area_ratio,
        n_sector=basis_data.basis.n_spin * basis_data.basis.n_flavor,
    )
    overlap_blocks = build_htg_supercell_overlap_blocks(basis_data, g_shells=g_shells)
    runs: list[HTGSupercellHartreeFockRun] = []
    for init_mode in init_modes:
        for seed in seeds:
            state = HTGSupercellHartreeFockState.from_projected_basis(
                basis_data,
                nu=primitive_nu,
                precision=precision,
            )
            problem = build_htg_supercell_hf_problem(state, overlap_blocks, beta=beta, use_numba=use_numba)
            base_run = run_hartree_fock_problem(
                state,
                problem,
                init_mode=str(init_mode),
                seed=int(seed),
                max_iter=int(max_iter),
                oda_stall_threshold=float(oda_stall_threshold),
            )
            runs.append(
                HTGSupercellHartreeFockRun(
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
    return HTGSupercellGroundStateScan(runs=tuple(runs))

__all__ = [name for name in globals() if not name.startswith('__')]
