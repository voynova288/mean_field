from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_basis_overlap import *  # noqa: F401,F403
from ._hf_basis_overlap import _issue_tbg_zero_field_typed_hf_run
from ._hf_diagnostics import occupied_sigma_mean, offdiag_flavor_norm, restricted_gap_estimate

def oda_parametrization_restricted(
    state: RestrictedHartreeFockState,
    delta_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
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
            interaction_spec=interaction_spec,
            legacy_untyped=legacy_untyped,
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
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    _screened_overlap_blocks: HFOverlapBlockSet | None = None,
) -> HartreeFockKernel:
    flavor_projector = _flavor_diagonal_projector(state)
    screened_overlap_blocks = (
        _with_tbg_overlap_screening(
            overlap_blocks,
            lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
            params=params,
            interaction_spec=interaction_spec,
            legacy_untyped=legacy_untyped,
        )
        if _screened_overlap_blocks is None
        else _screened_overlap_blocks
    )
    return build_projected_hf_kernel(
        state,
        screened_overlap_blocks,
        density_builder=lambda hamiltonian: _restricted_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: compute_oda_parameter(
            state_obj,
            delta_density,
            interaction_builder=lambda density: build_projected_interaction_hamiltonian(
                density,
                screened_overlap_blocks,
                v0=state_obj.v0,
                beta=beta,
            ),
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
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    initial_density: np.ndarray | None = None,
    _screened_overlap_blocks: HFOverlapBlockSet | None = None,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_restricted_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
            initial_density=initial_density,
        ),
        kernel=build_restricted_hf_kernel(
            state,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
            interaction_spec=interaction_spec,
            legacy_untyped=legacy_untyped,
            _screened_overlap_blocks=_screened_overlap_blocks,
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

    conventional_projector = np.zeros_like(h0)
    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
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
            conventional_projector[:, :, ik][np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
    elif is_canonical_restricted_init(init_mode):
        occupied_per_k = restricted_occupied_bands_per_k(nu, nt)
        if occupied_per_k < 0 or occupied_per_k > n_spin * n_eta:
            raise ValueError(f"Canonical restricted init only supports 0 <= occupied_per_k <= {n_spin * n_eta}, got {occupied_per_k}")
        for ispin, ieta in canonical_fig6_flavor_sequence(init_mode)[:occupied_per_k]:
            lower_band = int(idx[ispin, ieta, 0])
            conventional_projector[lower_band, lower_band, :] = 1.0
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
            block_projector = conventional_projector[:, :, ik]
            vecs_k = vecs[:, :, ik]
            for inds in sectors:
                block_inds = np.asarray(inds, dtype=int)
                occ_local = np.flatnonzero(occ_mask[block_inds, ik])
                if occ_local.size == 0:
                    continue
                occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
                block_projector[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T
    else:
        raise ValueError(f"Unsupported restricted init mode after normalization: {init_mode}")

    density = conventional_projector_to_stored_density(conventional_projector)
    project_to_flavor_diagonal_inplace(density, sectors=sectors)
    return density


def initialize_restricted_state(
    state: RestrictedHartreeFockState,
    *,
    init_mode: str = "educated",
    seed: int = 1,
    initial_density: np.ndarray | None = None,
) -> float:
    if initial_density is None:
        state.density[:, :, :] = initialize_restricted_density(
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
            raise ValueError(
                f"Expected initial_density shape {state.density.shape}, got {initial_density.shape}"
            )
        state.density[:, :, :] = initial_density
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

    conventional_projector = np.zeros_like(hamiltonian)
    for ik in range(nk):
        block_projector = conventional_projector[:, :, ik]
        vecs_k = vecs[:, :, ik]
        for inds in sectors:
            block_inds = np.asarray(inds, dtype=int)
            occ_local = np.flatnonzero(occ_mask[block_inds, ik])
            if occ_local.size == 0:
                continue
            occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
            block_projector[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T

    density = conventional_projector_to_stored_density(conventional_projector)
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
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    source_solution: BMSolution | None = None,
    screened_block_bundle: TBGZeroFieldScreenedBlockBundle | None = None,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    resolved_seed = validate_tbg_zero_field_seed(seed)
    _validate_initial_density_override_policy(
        initial_density,
        legacy_untyped=legacy_untyped,
        hf_mode="restricted",
    )
    normalized_init_mode = normalize_restricted_init_mode(init_mode)
    typed_run_requested = screened_block_bundle is not None or interaction_spec is not None
    if typed_run_requested:
        state.nu = validate_tbg_zero_field_primitive_cell_nu(state.nu)
    resolved_max_iter = (
        validate_tbg_zero_field_typed_max_iter(max_iter)
        if typed_run_requested
        else max_iter
    )
    resolved_lattice_kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    if screened_block_bundle is not None:
        if legacy_untyped:
            raise ValueError("screened_block_bundle and legacy_untyped=True are mutually exclusive")
        if source_solution is None:
            raise ValueError("Typed restricted HF requires source_solution with a carried torus mesh")
        bundle_spec = screened_block_bundle.interaction_spec
        if interaction_spec is not None and interaction_spec.fingerprint != bundle_spec.fingerprint:
            raise ValueError("interaction_spec does not match screened_block_bundle")
        validate_tbg_zero_field_typed_hf_source(
            state,
            source_solution,
            screened_block_bundle,
            overlap_blocks=overlap_blocks,
            lattice_kvec=resolved_lattice_kvec,
            params=params,
        )
        interaction_spec = bundle_spec
        screened_overlap_blocks = screened_block_bundle.screened_blocks
    else:
        if interaction_spec is not None:
            raise ValueError(
                "Typed restricted HF requires build_tbg_zero_field_screened_block_bundle; "
                "arbitrary overlap blocks cannot be relabelled typed"
            )
        if not legacy_untyped:
            raise ValueError(
                "Restricted HF requires either a typed screened_block_bundle or explicit legacy_untyped=True"
            )
        screened_overlap_blocks = _with_tbg_overlap_screening(
            overlap_blocks,
            lattice_kvec=resolved_lattice_kvec,
            params=params,
            legacy_untyped=True,
        )

    if interaction_spec is not None:
        if state.interaction_spec is not None and state.interaction_spec.fingerprint != interaction_spec.fingerprint:
            raise ValueError("HF state is already bound to a different typed interaction_spec")
        state.interaction_spec = interaction_spec
        state.diagnostics["relative_permittivity"] = float(interaction_spec.epsr)
        state.diagnostics["screening_lm"] = float(interaction_spec.screening_lm)
        state.diagnostics["finite_zero_limit"] = float(interaction_spec.finite_zero_limit)
        state.diagnostics["zero_cutoff"] = float(interaction_spec.zero_cutoff)
        state.diagnostics["dsc_nm"] = float(interaction_spec.dsc_nm)
    elif state.interaction_spec is not None:
        raise ValueError("Cannot run legacy_untyped HF from a state already bound to a typed interaction_spec")
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    state.diagnostics["requested_max_iterations"] = float(resolved_max_iter)
    if screened_block_bundle is not None:
        state.hf_source_receipt = build_tbg_zero_field_hf_source_receipt(
            hf_mode="restricted",
            beta=beta,
            v0=state.v0,
            solution=source_solution,
            screened_block_bundle=screened_block_bundle,
        )
    else:
        state.hf_source_receipt = build_tbg_zero_field_diagnostic_hf_source_receipt(
            hf_mode="restricted",
            beta=beta,
            v0=state.v0,
            lattice_kvec=resolved_lattice_kvec,
            overlap_blocks=screened_overlap_blocks,
        )
    problem = build_restricted_hf_problem(
        state,
        screened_overlap_blocks,
        resolved_lattice_kvec,
        params,
        beta=beta,
        interaction_spec=interaction_spec,
        legacy_untyped=legacy_untyped,
        initial_density=initial_density,
        _screened_overlap_blocks=screened_overlap_blocks,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=normalized_init_mode,
        seed=resolved_seed,
        max_iter=resolved_max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
    if screened_block_bundle is not None:
        return _issue_tbg_zero_field_typed_hf_run(
            hf_mode="restricted",
            state=state,
            overlap_blocks=screened_overlap_blocks,
            screened_block_bundle=screened_block_bundle,
            base_run=base_run,
            beta=beta,
            oda_stall_threshold=oda_stall_threshold,
            requested_max_iterations=resolved_max_iter,
        )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=screened_overlap_blocks,
        screened_block_bundle=None,
        provenance=None,
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
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    resolved_seed = validate_tbg_zero_field_seed(seed)
    _validate_initial_density_override_policy(
        initial_density,
        legacy_untyped=legacy_untyped,
        hf_mode="restricted",
    )
    resolved_max_iter = (
        validate_tbg_zero_field_typed_max_iter(max_iter)
        if interaction_spec is not None
        else max_iter
    )
    resolved_nu = (
        validate_tbg_zero_field_primitive_cell_nu(nu)
        if interaction_spec is not None
        else float(nu)
    )
    state = RestrictedHartreeFockState.from_bm_solution(
        solution,
        nu=resolved_nu,
        precision=precision,
    )
    raw_overlap_lg = solution.lg if overlap_lg is None else overlap_lg
    resolved_overlap_lg = (
        validate_tbg_zero_field_typed_overlap_lg(raw_overlap_lg)
        if interaction_spec is not None
        else int(raw_overlap_lg)
    )
    state.diagnostics["overlap_lg"] = float(resolved_overlap_lg)
    if interaction_spec is not None:
        if legacy_untyped:
            raise ValueError("interaction_spec and legacy_untyped=True are mutually exclusive")
        screened_block_bundle = build_tbg_zero_field_screened_block_bundle(
            solution,
            interaction_spec=interaction_spec,
            overlap_lg=resolved_overlap_lg,
        )
        overlap_blocks = screened_block_bundle.screened_blocks
    else:
        screened_block_bundle = None
        overlap_blocks = build_overlap_block_set(
            solution,
            lg=resolved_overlap_lg,
            legacy_untyped=legacy_untyped,
        )
    return run_restricted_hartree_fock(
        state,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        seed=resolved_seed,
        beta=beta,
        max_iter=resolved_max_iter,
        oda_stall_threshold=oda_stall_threshold,
        interaction_spec=interaction_spec,
        legacy_untyped=legacy_untyped,
        source_solution=solution,
        screened_block_bundle=screened_block_bundle,
        initial_density=initial_density,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
