from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_initialization import *  # noqa: F401,F403
from ._hf_basis import *  # noqa: F401,F403
from ._hf_interaction_path import *  # noqa: F401,F403

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
) -> HartreeFockKernel:
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


def build_htg_hf_problem(
    state: HTGHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> HartreeFockProblem:
    """Build the shared core-HF problem wrapper for an HTG projected state."""

    return build_projected_hf_problem(
        initializer=HTGInitializer(initial_density=initial_density),
        kernel=build_htg_hf_kernel(
            state,
            overlap_blocks,
            beta=beta,
            use_numba=use_numba,
        ),
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
    _validate_primitive_cell_integer_filling(nu)
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
    problem = build_htg_hf_problem(
        state,
        overlap_blocks,
        beta=beta,
        initial_density=initial_density,
        use_numba=use_numba,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=normalized_init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
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
    _validate_primitive_cell_integer_filling(nu)
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
            problem = build_htg_hf_problem(
                state,
                overlap_blocks,
                beta=beta,
                use_numba=use_numba,
            )
            base_run = run_hartree_fock_problem(
                state,
                problem,
                init_mode=normalized,
                seed=int(seed),
                max_iter=max_iter,
                oda_stall_threshold=oda_stall_threshold,
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

__all__ = [name for name in globals() if not name.startswith('__')]
