from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_basis_overlap import *  # noqa: F401,F403
from ._hf_restricted import _update_tbg_hf_density_update_state, _update_tbg_hf_step_state

def normalize_full_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    supported = {
        "random",
        "diag_random",
        "educated",
        "tivc",
        "kivc",
        "bm",
        "vp",
        "sp",
        "chern",
        "flavor",
        "sublattice",
    }
    if normalized not in supported:
        raise ValueError(
            f"Unsupported full HF init mode: {init_mode}. "
            "Supported modes: random, diag_random, educated, tivc, kivc, bm, vp, sp, chern, flavor"
        )
    return normalized


def canonical_fig6_state_sequence(*, n_spin: int = 2, n_eta: int = 2, n_band: int = 2) -> tuple[tuple[int, int, int], ...]:
    if n_spin != 2 or n_eta != 2:
        raise ValueError("The canonical Fig.6 full-HF ordering is only defined for n_spin=2, n_eta=2.")
    flavor_order = ((1, 0), (0, 0), (1, 1), (0, 1))
    return tuple((ispin, ieta, iband) for iband in range(n_band) for ispin, ieta in flavor_order)


def _full_flavor_priority(flag: str, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if flag == "random":
        return rng.permutation(idx.ravel(order="F"))
    if flag == "vp":
        return np.transpose(idx, (0, 2, 1)).ravel(order="F")
    if flag == "sp":
        return np.transpose(idx, (2, 1, 0)).ravel(order="F")
    if flag == "chern":
        return idx.ravel(order="F")
    if flag == "sublattice":
        swapped = idx.copy()
        swapped[:, 0, 0], swapped[:, 0, 1] = idx[:, 0, 1].copy(), idx[:, 0, 0].copy()
        return swapped.ravel(order="F")
    raise ValueError(f"Unsupported full flavor-polarization flag: {flag}")


def _random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    # Match the Julia full-HF initializers more closely: they build the
    # rotation from `eigvecs(Hermitian(rand(ComplexF64, ...)))`, i.e. a
    # Hermitian view over a uniformly sampled complex matrix rather than a
    # symmetrized Gaussian draw.
    sampled = rng.random((dim, dim)) + 1j * rng.random((dim, dim))
    hermitian = np.triu(sampled).astype(np.complex128, copy=True)
    hermitian += np.triu(sampled, k=1).conj().T
    diag = np.real(np.diag(sampled))
    hermitian[np.diag_indices(dim)] = diag
    _, vecs = eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)


def _apply_full_valley_rotation(
    density: np.ndarray,
    *,
    alpha: float,
    seed: int,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> None:
    rng = np.random.default_rng(seed)
    idx = np.arange(density.shape[0], dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    for ik in range(density.shape[2]):
        for ispin in range(n_spin):
            block_inds = np.asarray(idx[ispin, :, :].ravel(order="F"), dtype=int)
            unitary = _random_unitary(block_inds.size, rng)
            block = density[np.ix_(block_inds, block_inds, [ik])][:, :, 0]
            density[np.ix_(block_inds, block_inds, [ik])] = (
                (1.0 - alpha) * block + alpha * (unitary.conj().T @ block @ unitary)
            )[:, :, None]


def _apply_full_ivc_rotation(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> None:
    if n_eta != 2:
        raise ValueError("IVC rotation currently expects exactly two valleys.")
    idx = np.arange(density.shape[0], dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    mat = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    for ik in range(density.shape[2]):
        for ispin in range(n_spin):
            for iband in range(n_band):
                block_inds = np.asarray(idx[ispin, :, iband], dtype=int)
                block = density[np.ix_(block_inds, block_inds, [ik])][:, :, 0]
                density[np.ix_(block_inds, block_inds, [ik])] = (mat @ block @ mat)[:, :, None]


def _apply_full_random_rotation(density: np.ndarray, *, alpha: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    nt = density.shape[0]
    for ik in range(density.shape[2]):
        unitary = _random_unitary(nt, rng)
        block = density[:, :, ik]
        density[:, :, ik] = (1.0 - alpha) * block + alpha * (unitary.conj().T @ block @ unitary)


def initialize_full_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_full_init_mode(init_mode)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    density = np.zeros_like(h0)
    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    full_id = identity_block(nt)
    rng = np.random.default_rng(seed)
    valley_rotation_alpha = 0.0
    random_rotation_alpha = 0.0

    if init_mode == "random":
        occupied = rng.permutation(nt * nk)[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")
        for ik in range(nk):
            density[:, :, ik][np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
            density[:, :, ik] -= 0.5 * full_id
        valley_rotation_alpha = 1.0
        random_rotation_alpha = 1.0
    elif init_mode == "diag_random":
        return initialize_restricted_density(
            h0,
            nu=nu,
            init_mode="random",
            seed=seed,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
    elif init_mode == "educated":
        occupied_per_k = restricted_occupied_bands_per_k(nu, nt)
        ordered_states = canonical_fig6_state_sequence(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        for ispin, ieta, iband in ordered_states[:occupied_per_k]:
            density[int(idx[ispin, ieta, iband]), int(idx[ispin, ieta, iband]), :] = 1.0
        for ik in range(nk):
            density[:, :, ik] -= 0.5 * full_id
    elif init_mode == "tivc":
        density = initialize_full_density(h0, nu=nu, init_mode="vp", seed=seed, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        _apply_full_ivc_rotation(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        return density
    elif init_mode == "kivc":
        density = initialize_full_density(h0, nu=nu, init_mode="sublattice", seed=seed, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        _apply_full_ivc_rotation(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        return density
    else:
        flag = "random" if init_mode == "flavor" else init_mode
        if flag not in {"vp", "sp", "chern", "random", "sublattice", "bm"}:
            raise ValueError(f"Unsupported full flavor init flag: {flag}")

        if flag == "bm":
            energies = np.zeros((nt, nk), dtype=float)
            for ik in range(nk):
                energies[:, ik] = np.diag(h0[:, :, ik]).real
            occupied = np.argsort(energies.ravel(order="F"))[:total_occupied]
            occ_mask = np.zeros(nt * nk, dtype=bool)
            occ_mask[occupied] = True
            occ_mask = occ_mask.reshape((nt, nk), order="F")
            for ik in range(nk):
                density[:, :, ik][np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
                density[:, :, ik] -= 0.5 * full_id
            valley_rotation_alpha = 0.05
        else:
            n_per_flavor = nk
            num_full_flavors = total_occupied // n_per_flavor
            num_partial_flavors = 0 if total_occupied % n_per_flavor == 0 else 1
            flavor_order = _full_flavor_priority(flag, idx, rng)
            selected = np.asarray(flavor_order[: num_full_flavors + num_partial_flavors], dtype=int)
            for ifl in selected[: max(0, selected.size - num_partial_flavors)]:
                density[ifl, ifl, :] = 1.0
            if num_partial_flavors:
                ifl = int(selected[-1])
                remaining = total_occupied - (selected.size - 1) * n_per_flavor
                occupied_k = rng.permutation(n_per_flavor)[:remaining]
                density[ifl, ifl, occupied_k] = 1.0
            for ik in range(nk):
                density[:, :, ik] -= 0.5 * full_id
            valley_rotation_alpha = 0.05

    if valley_rotation_alpha > 0.0:
        _apply_full_valley_rotation(
            density,
            alpha=valley_rotation_alpha,
            seed=seed,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
    if random_rotation_alpha > 0.0:
        _apply_full_random_rotation(density, alpha=random_rotation_alpha, seed=seed)

    return density


def initialize_full_state(
    state: RestrictedHartreeFockState,
    *,
    init_mode: str = "flavor",
    seed: int = 1,
    initial_density: np.ndarray | None = None,
) -> float:
    if initial_density is None:
        state.density[:, :, :] = initialize_full_density(
            state.h0,
            nu=state.nu,
            init_mode=init_mode,
            seed=seed,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        )
    else:
        initial_density = np.asarray(initial_density, dtype=np.complex128)
        if initial_density.shape != state.density.shape:
            raise ValueError(f"Expected initial_density shape {state.density.shape}, got {initial_density.shape}")
        state.density[:, :, :] = initial_density
    filling = restricted_filling(state.density)
    state.diagnostics["filling"] = filling
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    return filling


def build_full_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
    *,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nt, _, nk = hamiltonian.shape
    energies = np.zeros((nt, nk), dtype=float)
    sigma_ztauz = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros_like(hamiltonian)

    for ik in range(nk):
        # Use the same dense Hermitian eigensolver family as the Julia
        # reference to reduce cross-language drift when frontier states become
        # nearly degenerate deep into the SCF iterations.
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik], UPLO="U")
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs
        sigma_ztauz[:, ik] = np.real(np.diag(eigvecs.conj().T @ sigma_z[:, :, ik] @ eigvecs))

    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    occ_mask = _occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, (nu + 4.0) / 8.0)

    density = np.zeros_like(hamiltonian)
    full_id = identity_block(nt)
    for ik in range(nk):
        occ_local = np.flatnonzero(occ_mask[:, ik])
        if occ_local.size == 0:
            density[:, :, ik] = -0.5 * full_id
            continue
        occupied_vecs = vecs[:, occ_local, ik]
        # Keep the current Julia full-HF projector convention for benchmark parity.
        density[:, :, ik] = occupied_vecs.conj() @ occupied_vecs.T - 0.5 * full_id

    return density, energies, sigma_ztauz, mu


def build_full_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockKernel:
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
    )
    return build_projected_hf_kernel(
        state,
        screened_overlap_blocks,
        density_builder=lambda hamiltonian: _full_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: oda_parametrization_restricted(
            state_obj,
            delta_density,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
        step_callback=_update_tbg_hf_step_state,
        final_state_callback=_update_tbg_hf_density_update_state,
        convergence_rule="mixed",
        v0=state.v0,
        beta=beta,
    )


def build_full_hf_problem(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_full_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
            initial_density=initial_density,
        ),
        kernel=build_full_hf_kernel(
            state,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
    )


def run_full_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    normalized_init_mode = normalize_full_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    problem = build_full_hf_problem(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        beta=beta,
        initial_density=initial_density,
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


def run_full_hf_from_bm_solution(
    solution: BMSolution,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    overlap_lg: int | None = None,
    precision: float = 1e-5,
    oda_stall_threshold: float = 1e-3,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=nu, precision=precision)
    resolved_overlap_lg = solution.lg if overlap_lg is None else int(overlap_lg)
    state.diagnostics["overlap_lg"] = float(resolved_overlap_lg)
    overlap_blocks = build_overlap_block_set(solution, lg=resolved_overlap_lg)
    return run_full_hartree_fock(
        state,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        seed=seed,
        beta=beta,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        initial_density=initial_density,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
