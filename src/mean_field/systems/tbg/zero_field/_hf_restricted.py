from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_basis_overlap import *  # noqa: F401,F403
from ._hf_diagnostics import occupied_sigma_mean, offdiag_flavor_norm, restricted_gap_estimate

def oda_parametrization_restricted(
    state: RestrictedHartreeFockState,
    delta_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> float:
    return compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: build_interaction_hamiltonian(
            density,
            overlap_blocks,
            lattice_kvec,
            params,
            state.v0,
            beta=beta,
        ),
    )


def _restricted_density_update_result(state: RestrictedHartreeFockState, hamiltonian: np.ndarray) -> DensityUpdateResult:
    density, energies, sigma_ztauz, mu = build_restricted_density_from_hamiltonian(
        hamiltonian,
        state.sigma_z,
        nu=state.nu,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=mu,
        observables={"sigma_ztauz": sigma_ztauz},
    )


def _full_density_update_result(state: RestrictedHartreeFockState, hamiltonian: np.ndarray) -> DensityUpdateResult:
    density, energies, sigma_ztauz, mu = build_full_density_from_hamiltonian(
        hamiltonian,
        state.sigma_z,
        nu=state.nu,
    )
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=mu,
        observables={"sigma_ztauz": sigma_ztauz},
    )


def _update_tbg_hf_density_update_state(state: RestrictedHartreeFockState, density_update: DensityUpdateResult) -> None:
    sigma_ztauz = np.asarray(density_update.observables["sigma_ztauz"], dtype=float)
    state.sigma_ztauz[:, :] = sigma_ztauz
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)
    state.diagnostics["occupied_sigma_mean"] = occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu)


def _update_tbg_hf_step_state(state: RestrictedHartreeFockState, step) -> None:
    _update_tbg_hf_density_update_state(state, step.density_update)


def _flavor_diagonal_projector(state: RestrictedHartreeFockState):
    return lambda matrix: project_to_flavor_diagonal_inplace(
        matrix,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )


def build_restricted_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockKernel:
    flavor_projector = _flavor_diagonal_projector(state)
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
    )
    return build_projected_hf_kernel(
        state,
        screened_overlap_blocks,
        density_builder=lambda hamiltonian: _restricted_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: oda_parametrization_restricted(
            state_obj,
            delta_density,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
        hamiltonian_postprocessor=flavor_projector,
        density_postprocessor=flavor_projector,
        step_callback=_update_tbg_hf_step_state,
        final_state_callback=_update_tbg_hf_density_update_state,
        convergence_rule="raw",
        v0=state.v0,
        beta=beta,
    )


def build_restricted_hf_problem(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_restricted_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
        ),
        kernel=build_restricted_hf_kernel(
            state,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
    )


def initialize_restricted_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "educated",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_restricted_init_mode(init_mode)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    density = np.zeros_like(h0)
    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    full_id = identity_block(nt)
    sectors = flavor_block_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band)

    if init_mode == "bm":
        energies = np.zeros((nt, nk), dtype=float)
        for ik in range(nk):
            energies[:, ik] = np.diag(h0[:, :, ik]).real
        occupied = np.argsort(energies.ravel(order="F"))[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")
        for ik in range(nk):
            block = density[:, :, ik]
            block[np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
            block -= 0.5 * full_id
    elif is_canonical_restricted_init(init_mode):
        occupied_per_k = restricted_occupied_bands_per_k(nu, nt)
        if occupied_per_k < 0 or occupied_per_k > n_spin * n_eta:
            raise ValueError(f"Canonical restricted init only supports 0 <= occupied_per_k <= {n_spin * n_eta}, got {occupied_per_k}")
        for ispin, ieta in canonical_fig6_flavor_sequence(init_mode)[:occupied_per_k]:
            lower_band = int(idx[ispin, ieta, 0])
            density[lower_band, lower_band, :] = 1.0
        for ik in range(nk):
            density[:, :, ik] -= 0.5 * full_id
    elif init_mode == "random":
        rng = np.random.default_rng(seed)
        evals = np.zeros((nt, nk), dtype=float)
        vecs = np.zeros_like(h0)
        for ik in range(nk):
            vecs_k = vecs[:, :, ik]
            for inds in sectors:
                block_inds = np.asarray(inds, dtype=int)
                block_h = rng.standard_normal((block_inds.size, block_inds.size)) + 1j * rng.standard_normal((block_inds.size, block_inds.size))
                block_h = block_h + block_h.conj().T
                eigvals, eigvecs = eigh(block_h)
                evals[block_inds, ik] = eigvals
                vecs_k[np.ix_(block_inds, block_inds)] = eigvecs

        occupied = np.argsort(evals.ravel(order="F"))[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")

        for ik in range(nk):
            block_density = density[:, :, ik]
            vecs_k = vecs[:, :, ik]
            for inds in sectors:
                block_inds = np.asarray(inds, dtype=int)
                block_id = identity_block(block_inds.size)
                occ_local = np.flatnonzero(occ_mask[block_inds, ik])
                if occ_local.size == 0:
                    block_density[np.ix_(block_inds, block_inds)] = -0.5 * block_id
                    continue
                occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
                block_density[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id
    else:
        raise ValueError(f"Unsupported restricted init mode after normalization: {init_mode}")

    project_to_flavor_diagonal_inplace(density, sectors=sectors)
    return density


def initialize_restricted_state(
    state: RestrictedHartreeFockState,
    *,
    init_mode: str = "educated",
    seed: int = 1,
) -> float:
    state.density[:, :, :] = initialize_restricted_density(
        state.h0,
        nu=state.nu,
        init_mode=init_mode,
        seed=seed,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    filling = restricted_filling(state.density)
    state.diagnostics["filling"] = filling
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    return filling


def build_restricted_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
    *,
    nu: float,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nt, _, nk = hamiltonian.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(
            f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}"
        )

    sectors = flavor_block_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    energies = np.zeros((nt, nk), dtype=float)
    sigma_ztauz = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros_like(hamiltonian)

    for ik in range(nk):
        vecs_k = vecs[:, :, ik]
        h_k = hamiltonian[:, :, ik]
        sigma_k = sigma_z[:, :, ik]
        for inds in sectors:
            block_inds = np.asarray(inds, dtype=int)
            block_h = h_k[np.ix_(block_inds, block_inds)]
            block_sigma = sigma_k[np.ix_(block_inds, block_inds)]
            eigvals, eigvecs = eigh(block_h)
            energies[block_inds, ik] = eigvals
            vecs_k[np.ix_(block_inds, block_inds)] = eigvecs
            sigma_ztauz[block_inds, ik] = np.real(np.diag(eigvecs.conj().T @ block_sigma @ eigvecs))

    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    occ_mask = _occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, (nu + 4.0) / 8.0)

    density = np.zeros_like(hamiltonian)
    for ik in range(nk):
        block_density = density[:, :, ik]
        vecs_k = vecs[:, :, ik]
        for inds in sectors:
            block_inds = np.asarray(inds, dtype=int)
            block_id = identity_block(block_inds.size)
            occ_local = np.flatnonzero(occ_mask[block_inds, ik])
            if occ_local.size == 0:
                block_density[np.ix_(block_inds, block_inds)] = -0.5 * block_id
                continue
            occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
            block_density[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id

    project_to_flavor_diagonal_inplace(density, sectors=sectors)
    return density, energies, sigma_ztauz, mu


def update_restricted_density(
    state: RestrictedHartreeFockState,
    *,
    mixing_parameter: float = 1.0,
) -> tuple[float, float]:
    if mixing_parameter < 0.0 or mixing_parameter > 1.0:
        raise ValueError(f"mixing_parameter must lie in [0, 1], got {mixing_parameter}")

    old_density = state.density.copy()
    density_new, energies, sigma_ztauz, mu = build_restricted_density_from_hamiltonian(
        state.hamiltonian,
        state.sigma_z,
        nu=state.nu,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    mixed_density = mixing_parameter * density_new + (1.0 - mixing_parameter) * old_density
    norm_convergence = calculate_norm_convergence(mixed_density, old_density)

    state.density[:, :, :] = mixed_density
    project_to_flavor_diagonal_inplace(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    state.energies[:, :] = energies
    state.sigma_ztauz[:, :] = sigma_ztauz
    state.mu = float(mu)
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)
    state.diagnostics["occupied_sigma_mean"] = occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu)
    return norm_convergence, float(mixing_parameter)


def run_restricted_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    init_mode: str = "educated",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
) -> RestrictedHartreeFockRun:
    normalized_init_mode = normalize_restricted_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    problem = build_restricted_hf_problem(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        beta=beta,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=normalized_init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def run_restricted_hf_from_bm_solution(
    solution: BMSolution,
    *,
    nu: float,
    init_mode: str = "educated",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    overlap_lg: int | None = None,
    precision: float = 1e-5,
    oda_stall_threshold: float = 1e-3,
) -> RestrictedHartreeFockRun:
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=nu, precision=precision)
    resolved_overlap_lg = solution.lg if overlap_lg is None else int(overlap_lg)
    state.diagnostics["overlap_lg"] = float(resolved_overlap_lg)
    overlap_blocks = build_overlap_block_set(solution, lg=resolved_overlap_lg)
    return run_restricted_hartree_fock(
        state,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        seed=seed,
        beta=beta,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
