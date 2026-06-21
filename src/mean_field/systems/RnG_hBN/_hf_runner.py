from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_types import *  # noqa: F401,F403
from ._hf_basis import *  # noqa: F401,F403
from ._hf_interaction_path import *  # noqa: F401,F403

def compute_rlg_hbn_oda_parameter(
    state: RLGhBNHartreeFockState,
    delta_density: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    beta: float = 1.0,
) -> float:
    return compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: build_rlg_hbn_hf_interaction_hamiltonian(
            density,
            overlap_blocks,
            v0=state.v0,
            beta=beta,
        ),
    )


def _hermitize_blocks_inplace(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)


def rlg_hbn_projector_idempotency_residual(density_delta: np.ndarray, reference_density: np.ndarray) -> float:
    projector = rlg_hbn_projector_from_density(density_delta, reference_density)
    residual = 0.0
    for ik in range(projector.shape[2]):
        block = projector[:, :, ik]
        residual = max(residual, float(np.max(np.abs(block @ block - block))))
    return float(residual)


def rlg_hbn_hermitian_residual(blocks: np.ndarray) -> float:
    blocks = np.asarray(blocks, dtype=np.complex128)
    residual = 0.0
    for ik in range(blocks.shape[2]):
        residual = max(residual, float(np.max(np.abs(blocks[:, :, ik] - blocks[:, :, ik].conjugate().T))))
    return float(residual)


def rlg_hbn_gap_estimate(
    energies: np.ndarray,
    nu: float,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    energies = np.asarray(energies, dtype=float)
    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        energies.shape[0],
        energies.shape[1],
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    sorted_energies = np.sort(energies, axis=None)
    if total_occupied <= 0 or total_occupied >= sorted_energies.size:
        return float("nan")
    return float(sorted_energies[total_occupied] - sorted_energies[total_occupied - 1])


def _update_rlg_hbn_diagnostics_from_density(state: RLGhBNHartreeFockState) -> None:
    state.diagnostics["filling"] = rlg_hbn_filling_from_density(
        state.density,
        state.reference_density,
        active_valence_bands=state.active_valence_bands,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    state.diagnostics["projector_idempotency_residual"] = rlg_hbn_projector_idempotency_residual(
        state.density,
        state.reference_density,
    )
    state.diagnostics["density_hermitian_residual"] = rlg_hbn_hermitian_residual(state.density)
    state.diagnostics["hamiltonian_hermitian_residual"] = rlg_hbn_hermitian_residual(state.hamiltonian)
    state.diagnostics["hf_gap"] = rlg_hbn_gap_estimate(
        state.energies,
        state.nu,
        active_valence_bands=state.active_valence_bands,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


def normalize_rlg_hbn_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "sp": "bm",
        "noninteracting": "bm",
        "random": "random",
        "diag_random": "random",
        "flavor": "flavor",
        "polarized": "flavor",
        "polarized_k_up": "flavor",
        "perturbed": "perturbed",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported RLG/hBN HF init mode: {init_mode}. "
            "Supported modes: bm, random, diag_random, flavor, polarized, polarized_k_up, perturbed"
        )
    return aliases[normalized]


def rlg_hbn_flavor_occupation_counts_for_init_mode(
    init_mode: str,
    *,
    nu: float,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
    seed: int | None = None,
) -> tuple[int, ...] | None:
    normalized = normalize_rlg_hbn_init_mode(init_mode)
    if normalized in {"bm", "random", "perturbed"}:
        return None

    integer_nu = int(round(float(nu)))
    if abs(float(nu) - float(integer_nu)) > 1.0e-9:
        return None
    n_spin = int(n_spin)
    n_eta = int(n_eta)
    n_band = int(n_band)
    n_valence = int(active_valence_bands)
    if n_valence < 0 or n_valence > n_band:
        raise ValueError(f"active_valence_bands must lie in [0, {n_band}], got {active_valence_bands}")

    counts = np.full((n_spin, n_eta), n_valence, dtype=int)
    flavor_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    flavor_order = [(s, e) for s, e in flavor_order if s < n_spin and e < n_eta]
    flavor_order.extend(
        (s, e)
        for s in range(n_spin)
        for e in range(n_eta)
        if (s, e) not in flavor_order
    )
    if seed is not None and flavor_order:
        start = (int(seed) - 1) % len(flavor_order)
        flavor_order = flavor_order[start:] + flavor_order[:start]
    if integer_nu > 0:
        if integer_nu > len(flavor_order):
            raise ValueError(f"Positive integer filling nu={nu} exceeds available flavors {len(flavor_order)}")
        for ispin, ieta in flavor_order[:integer_nu]:
            counts[ispin, ieta] += 1
    elif integer_nu < 0:
        if abs(integer_nu) > len(flavor_order):
            raise ValueError(f"Negative integer filling nu={nu} exceeds available flavors {len(flavor_order)}")
        for ispin, ieta in reversed(flavor_order[-abs(integer_nu) :]):
            counts[ispin, ieta] -= 1

    if np.any(counts < 0) or np.any(counts > n_band):
        raise ValueError(f"Invalid RLG/hBN flavor counts for nu={nu}: {counts}")
    return tuple(int(value) for value in counts.reshape(-1, order="C"))


def initialize_rlg_hbn_density(
    h0: np.ndarray,
    *,
    nu: float,
    reference_density: np.ndarray,
    active_valence_bands: int,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_rlg_hbn_init_mode(init_mode)
    h0 = np.asarray(h0, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    nt, _, nk = h0.shape
    if reference_density.shape != h0.shape:
        raise ValueError(f"Expected reference_density shape {h0.shape}, got {reference_density.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    if init_mode == "bm":
        return build_rlg_hbn_density_from_hamiltonian(
            h0,
            nu=nu,
            reference_density=reference_density,
            active_valence_bands=active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )[0]

    rng = np.random.default_rng(seed)
    density = np.zeros_like(h0)
    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        nt,
        nk,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    idx = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")

    if init_mode == "random":
        random_energies = rng.standard_normal((nt, nk))
        occ_mask = occupied_state_mask(random_energies, total_occupied)
        for ik in range(nk):
            unitary = random_unitary_from_hermitian(nt, rng)
            occupied = np.flatnonzero(occ_mask[:, ik])
            if occupied.size == 0:
                density[:, :, ik] = -reference_density[:, :, ik]
            else:
                occupied_vecs = unitary[:, occupied]
                density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density

    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=nu,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        seed=seed if init_mode == "flavor" else None,
    )
    if counts is None:
        raise ValueError(f"init_mode={init_mode!r} requires integer flavor occupation counts for nu={nu}")
    counts_2d = np.asarray(counts, dtype=int).reshape((int(n_spin), int(n_eta)), order="C")
    for ik in range(nk):
        density[:, :, ik] = -reference_density[:, :, ik]
        for ispin in range(int(n_spin)):
            for ieta in range(int(n_eta)):
                n_occ = int(counts_2d[ispin, ieta])
                if n_occ <= 0:
                    continue
                block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                occupied = block_indices[:n_occ]
                density[:, :, ik][np.ix_(occupied, occupied)] = (
                    np.eye(n_occ, dtype=np.complex128)
                    - reference_density[:, :, ik][np.ix_(occupied, occupied)]
                )

    if init_mode == "perturbed":
        apply_random_projector_rotation(
            density,
            reference_density=reference_density,
            alpha=0.05,
            seed=seed,
        )
    return density


def build_rlg_hbn_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    nu: float,
    reference_density: np.ndarray,
    active_valence_bands: int,
    occupation_counts: tuple[int, ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    if reference_density.shape != hamiltonian.shape:
        raise ValueError(f"Expected reference_density shape {hamiltonian.shape}, got {reference_density.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    energies = np.zeros((nt, nk), dtype=float)
    density = np.zeros_like(hamiltonian)

    if occupation_counts is not None:
        counts = np.asarray(occupation_counts, dtype=int).reshape(-1)
        if counts.size != int(n_spin) * int(n_eta):
            raise ValueError(f"Expected {int(n_spin) * int(n_eta)} flavor occupation counts, got {counts.size}")
        if np.any(counts < 0) or np.any(counts > int(n_band)):
            raise ValueError(f"Flavor occupation counts must lie in [0, {int(n_band)}], got {counts.tolist()}")
        if int(np.sum(counts)) != rlg_hbn_occupied_bands_per_k(
            nu,
            nt,
            active_valence_bands=active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
        ):
            raise ValueError("Flavor occupation counts do not match the requested filling")

        idx = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
        counts_2d = counts.reshape((int(n_spin), int(n_eta)), order="C")
        occ_mask = np.zeros((nt, nk), dtype=bool)
        for ik in range(nk):
            density[:, :, ik] = -reference_density[:, :, ik]
            for ispin in range(int(n_spin)):
                for ieta in range(int(n_eta)):
                    block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    reference_block = reference_density[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
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
        return density, energies, float(mu), occ_mask

    vecs = np.zeros_like(hamiltonian)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs

    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        nt,
        nk,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    occ_mask = occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, float(total_occupied) / float(energies.size))

    for ik in range(nk):
        occupied = np.flatnonzero(occ_mask[:, ik])
        if occupied.size == 0:
            density[:, :, ik] = -reference_density[:, :, ik]
            continue
        occupied_vecs = vecs[:, occupied, ik]
        density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]

    return density, energies, float(mu), occ_mask


def build_rlg_hbn_hf_problem(
    state: RLGhBNHartreeFockState,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
    step_callback: Callable[[RLGhBNHartreeFockState, HartreeFockStepResult], None] | None = None,
) -> HartreeFockProblem:
    """Build the reusable core-HF problem wrapper for an RLG/hBN state.

    The RLG/hBN system layer still owns the projected basis, layer-resolved
    Coulomb tables, filling convention, and ODA functional.  This adapter only
    packages those system-specific callables behind the shared
    :class:`mean_field.core.hf.HartreeFockProblem` interface.
    """

    def initialize_state(state_obj: RLGhBNHartreeFockState, *, init_mode: str, seed: int) -> None:
        if initial_density is not None:
            density = np.asarray(initial_density, dtype=np.complex128)
            if density.shape != state_obj.density.shape:
                raise ValueError(f"Expected initial_density shape {state_obj.density.shape}, got {density.shape}")
            state_obj.density[:, :, :] = density
        else:
            state_obj.density[:, :, :] = initialize_rlg_hbn_density(
                state_obj.h0,
                nu=state_obj.nu,
                reference_density=state_obj.reference_density,
                active_valence_bands=state_obj.active_valence_bands,
                init_mode=init_mode,
                seed=seed,
                n_spin=state_obj.n_spin,
                n_eta=state_obj.n_eta,
                n_band=state_obj.n_band,
            )
        _hermitize_blocks_inplace(state_obj.density)
        _update_rlg_hbn_diagnostics_from_density(state_obj)

    def build_density(hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occupation_mask = build_rlg_hbn_density_from_hamiltonian(
            hamiltonian,
            nu=state.nu,
            reference_density=state.reference_density,
            active_valence_bands=state.active_valence_bands,
            occupation_counts=state.occupation_counts,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        )
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables={"occupation_mask": occupation_mask},
        )

    kernel = HartreeFockKernel(
        interaction_builder=lambda density: build_rlg_hbn_hf_interaction_hamiltonian(
            density,
            overlap_blocks,
            v0=state.v0,
            beta=beta,
        ),
        density_builder=build_density,
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: compute_rlg_hbn_oda_parameter(
            state_obj,  # type: ignore[arg-type]
            delta_density,
            overlap_blocks,
            beta=beta,
        ),
        hamiltonian_postprocessor=_hermitize_blocks_inplace,
        density_postprocessor=_hermitize_blocks_inplace,
        step_callback=step_callback,  # type: ignore[arg-type]
        convergence_rule="raw",
    )
    return HartreeFockProblem(
        initializer=initialize_state,
        kernel=kernel,
    )


def run_rlg_hbn_hartree_fock(
    basis_data: RLGhBNProjectedBasisData,
    *,
    overlap_blocks: RLGhBNLayerOverlapBlockSet | None = None,
    nu: float = 1.0,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 80,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    max_oda_lambda: float | None = None,
    occupation_counts: tuple[int, ...] | None = None,
    initial_density: np.ndarray | None = None,
    step_callback: Callable[[RLGhBNHartreeFockState, HartreeFockStepResult], None] | None = None,
) -> RLGhBNHartreeFockRun:
    resolved_counts = occupation_counts
    if resolved_counts is None:
        resolved_counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
            init_mode,
            nu=nu,
            active_valence_bands=basis_data.interaction.active_valence_bands,
            n_spin=basis_data.basis.n_spin,
            n_eta=basis_data.basis.n_flavor,
            n_band=basis_data.basis.n_band,
            seed=seed,
        )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=nu,
        precision=precision,
        occupation_counts=resolved_counts,
    )
    resolved_blocks = overlap_blocks if overlap_blocks is not None else build_rlg_hbn_layer_overlap_blocks(basis_data)
    problem = build_rlg_hbn_hf_problem(
        state,
        resolved_blocks,
        beta=beta,
        initial_density=initial_density,
        step_callback=step_callback,
    )
    core_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        max_oda_lambda=max_oda_lambda,
    )
    _update_rlg_hbn_diagnostics_from_density(state)
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=core_run.iter_energy,
        iter_err=core_run.iter_err,
        iter_oda=core_run.iter_oda,
        init_mode=core_run.init_mode,
        seed=core_run.seed,
        converged=core_run.converged,
        exit_reason=core_run.exit_reason,
        overlap_blocks=resolved_blocks,
        basis_data=basis_data,
    )


def scan_rlg_hbn_ground_state(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    nu: float = 1.0,
    init_modes: tuple[str, ...] = ("flavor", "bm", "perturbed"),
    seeds: tuple[int, ...] = (1,),
    beta: float = 1.0,
    max_iter: int = 80,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    max_oda_lambda: float | None = None,
    mesh_size: int | None = None,
    screening_mesh_size: int | None = None,
    run_callback: Callable[[RLGhBNHartreeFockRun], None] | None = None,
) -> RLGhBNGroundStateScan:
    basis_data = build_rlg_hbn_projected_basis(
        model,
        interaction,
        mesh_size=mesh_size,
        screening_mesh_size=screening_mesh_size,
    )
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data)
    runs: list[RLGhBNHartreeFockRun] = []
    for init_mode in init_modes:
        for seed in seeds:
            run = run_rlg_hbn_hartree_fock(
                basis_data,
                overlap_blocks=overlap_blocks,
                nu=nu,
                init_mode=init_mode,
                seed=int(seed),
                beta=beta,
                max_iter=max_iter,
                precision=precision,
                oda_stall_threshold=oda_stall_threshold,
                max_oda_lambda=max_oda_lambda,
            )
            runs.append(run)
            if run_callback is not None:
                run_callback(run)
    return RLGhBNGroundStateScan(runs=tuple(runs))

__all__ = [name for name in globals() if not name.startswith('__')]
