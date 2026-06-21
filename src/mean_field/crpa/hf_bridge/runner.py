from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from .density import *  # noqa: F401,F403
from .split_scheme import *  # noqa: F401,F403
from .kernels import *  # noqa: F401,F403
from .energy import *  # noqa: F401,F403

def _full_crpa_density_update_result(state: RestrictedHartreeFockState, hamiltonian: np.ndarray) -> DensityUpdateResult:
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


def _update_full_crpa_density_update_state(
    state: RestrictedHartreeFockState,
    density_update: DensityUpdateResult,
) -> None:
    sigma_ztauz = np.asarray(density_update.observables["sigma_ztauz"], dtype=float)
    state.sigma_ztauz[:, :] = sigma_ztauz
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)
    state.diagnostics["occupied_sigma_mean"] = occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu)


def build_full_crpa_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    beta: float = 1.0,
    relative_permittivity: float | None = None,
    screening_lm: float | None = None,
    finite_zero_limit: bool | None = None,
    zero_cutoff: float | None = None,
    fock_interpolation: str = "matrix_diagonal",
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Build a full-HF kernel using Zhang's remote-bare plus active-cRPA split."""

    coulomb = crpa_screening.result.coulomb_params
    resolved_relative_permittivity = (
        float(coulomb.epsilon_bn) if relative_permittivity is None else float(relative_permittivity)
    )
    resolved_screening_lm = float(coulomb.screening_lm) if screening_lm is None else float(screening_lm)
    resolved_finite_zero_limit = bool(coulomb.finite_zero_limit) if finite_zero_limit is None else bool(finite_zero_limit)
    resolved_zero_cutoff = float(coulomb.zero_cutoff) if zero_cutoff is None else float(zero_cutoff)
    screened_overlap_blocks = build_fock_screened_overlap_blocks(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
        crpa_screening=crpa_screening,
        fock_interpolation=fock_interpolation,
        relative_permittivity=resolved_relative_permittivity,
        screening_lm=resolved_screening_lm,
        finite_zero_limit=resolved_finite_zero_limit,
        zero_cutoff=resolved_zero_cutoff,
    )
    split_mode = crpa_split_mode()
    remote_scale = crpa_remote_bare_scale()

    density_template = np.asarray(state.density, dtype=np.complex128)
    if crpa_split_uses_remote_bare(split_mode):
        remote_reference_delta = half_reference_delta_like(density_template)
        remote_hartree, remote_fock = build_bare_projected_interaction_components(
            remote_reference_delta,
            overlap_blocks,
            v0=state.v0,
            beta=beta,
            use_numba=use_numba,
        )
        remote_bare_hamiltonian = select_remote_reference_components(remote_hartree, remote_fock, split_mode)
        remote_bare_hamiltonian *= remote_scale
        if bool(state.diagnostics.get("crpa_remote_bare_added", 0.0)):
            raise RuntimeError("Refusing to add the bare remote-band cRPA reference term twice to state.h0.")
        state.h0[:, :, :] += remote_bare_hamiltonian
        state.diagnostics["crpa_remote_bare_added"] = 1.0
        state.diagnostics["crpa_remote_bare_scale"] = float(remote_scale)
        state.diagnostics["crpa_remote_bare_hartree_fro_norm"] = float(np.linalg.norm(remote_hartree))
        state.diagnostics["crpa_remote_bare_fock_fro_norm"] = float(np.linalg.norm(remote_fock))
        state.diagnostics["crpa_remote_bare_fro_norm"] = float(np.linalg.norm(remote_bare_hamiltonian))
        state.diagnostics["crpa_remote_bare_max_abs"] = float(np.max(np.abs(remote_bare_hamiltonian)))
    else:
        state.diagnostics["crpa_remote_bare_added"] = 0.0
        state.diagnostics["crpa_remote_bare_scale"] = 0.0
        state.diagnostics["crpa_remote_bare_fro_norm"] = 0.0
        state.diagnostics["crpa_remote_bare_max_abs"] = 0.0

    if crpa_split_uses_active_cnp_reference(split_mode):
        active_cnp_projector = active_lower_flat_projector_like(
            density_template,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        )
        active_cnp_hartree, active_cnp_fock = build_crpa_projected_interaction_components(
            active_cnp_projector,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )
        active_cnp_reference = select_active_cnp_reference_components(
            active_cnp_hartree,
            active_cnp_fock,
            split_mode,
        )
        if bool(state.diagnostics.get("crpa_active_cnp_reference_added", 0.0)):
            raise RuntimeError("Refusing to add the active CNP cRPA reference term twice to state.h0.")
        state.h0[:, :, :] += active_cnp_reference
        state.diagnostics["crpa_active_cnp_reference_added"] = 1.0
        state.diagnostics["crpa_active_cnp_reference_hartree_fro_norm"] = float(np.linalg.norm(active_cnp_hartree))
        state.diagnostics["crpa_active_cnp_reference_fock_fro_norm"] = float(np.linalg.norm(active_cnp_fock))
        state.diagnostics["crpa_active_cnp_reference_fro_norm"] = float(np.linalg.norm(active_cnp_reference))
        state.diagnostics["crpa_active_cnp_reference_max_abs"] = float(np.max(np.abs(active_cnp_reference)))
    else:
        state.diagnostics["crpa_active_cnp_reference_added"] = 0.0
        state.diagnostics["crpa_active_cnp_reference_fro_norm"] = 0.0
        state.diagnostics["crpa_active_cnp_reference_max_abs"] = 0.0

    def crpa_dynamic_builder(projector_or_delta: np.ndarray) -> np.ndarray:
        return build_crpa_projected_interaction_hamiltonian(
            projector_or_delta,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )

    def hartree_delta_fock_projector_components(density_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return build_crpa_hartree_delta_fock_projector_components(
            density_delta,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )

    def delta_interaction_components(delta_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return build_crpa_projected_interaction_components_from_densities(
            delta_density,
            delta_density,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )

    def interaction_builder(density_delta: np.ndarray) -> np.ndarray:
        if crpa_split_uses_hartree_delta_fock_projector(split_mode):
            hartree_h, fock_h = hartree_delta_fock_projector_components(density_delta)
            return hartree_h + fock_h
        return crpa_dynamic_builder(crpa_active_density_from_delta(density_delta, split_mode))

    def oda_delta_interaction_builder(delta_density: np.ndarray) -> np.ndarray:
        if crpa_split_uses_hartree_delta_fock_projector(split_mode):
            hartree_h, fock_h = delta_interaction_components(delta_density)
            return hartree_h + fock_h
        return crpa_dynamic_builder(delta_density)

    def oda_parameterizer(state_obj, delta_density: np.ndarray) -> float:
        if crpa_split_uses_hartree_delta_fock_projector(split_mode):
            delta_hartree_h, delta_fock_h = delta_interaction_components(delta_density)
            return hartree_delta_fock_projector_oda_parameter(
                state_obj,
                delta_density,
                delta_hartree_h=delta_hartree_h,
                delta_fock_h=delta_fock_h,
                interaction_h=state_obj.hamiltonian - state_obj.h0,
            )
        delta_h = oda_delta_interaction_builder(delta_density)
        interaction_h = state_obj.hamiltonian - state_obj.h0
        if not crpa_split_uses_projector(split_mode):
            return compute_oda_parameter(
                state_obj,
                delta_density,
                delta_h=delta_h,
                interaction_h=interaction_h,
            )
        return split_oda_parameter(
            state_obj,
            delta_density,
            delta_h=delta_h,
            interaction_h=interaction_h,
        )

    def crpa_energy_functional(interaction_h: np.ndarray, h0: np.ndarray, density_delta: np.ndarray) -> float:
        if crpa_split_uses_hartree_delta_fock_projector(split_mode):
            hartree_h, fock_h = hartree_delta_fock_projector_components(density_delta)
            components = crpa_hartree_delta_fock_projector_energy_components(
                h0,
                density_delta,
                hartree_h,
                fock_h,
            )
            return float(components["E_total"])
        return crpa_split_energy_functional(interaction_h, h0, density_delta)

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=lambda hamiltonian: _full_crpa_density_update_result(state, hamiltonian),
        energy_functional=crpa_energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=None,
        step_callback=lambda state_obj, step: _update_full_crpa_density_update_state(state_obj, step.density_update),
        final_state_callback=_update_full_crpa_density_update_state,
        convergence_rule="mixed",
    )


def build_bare_split_full_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Build Zhang's no-cRPA split kernel: h_BM + Sigma_bare[-0.5I] + Sigma_bare[P].

    This is algebraically equivalent to Wang/Xiaoyu's stored-density kernel
    h_BM + Sigma_bare[D] when D = P - 0.5 I.  Keeping it as a first-class
    kernel gives the cRPA workflow a production-size bare-limit gate.
    """

    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
    )

    density_template = np.asarray(state.density, dtype=np.complex128)
    remote_reference_delta = half_reference_delta_like(density_template)
    remote_bare_hamiltonian = build_projected_interaction_hamiltonian(
        remote_reference_delta,
        screened_overlap_blocks,
        v0=state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    if bool(state.diagnostics.get("bare_split_remote_bare_added", 0.0)):
        raise RuntimeError("Refusing to add the bare split reference term twice to state.h0.")
    state.h0[:, :, :] += remote_bare_hamiltonian
    state.diagnostics["bare_split_remote_bare_added"] = 1.0
    state.diagnostics["bare_split_remote_bare_fro_norm"] = float(np.linalg.norm(remote_bare_hamiltonian))
    state.diagnostics["bare_split_remote_bare_max_abs"] = float(np.max(np.abs(remote_bare_hamiltonian)))

    def active_builder(projector_or_delta: np.ndarray) -> np.ndarray:
        return build_projected_interaction_hamiltonian(
            projector_or_delta,
            screened_overlap_blocks,
            v0=state.v0,
            beta=beta,
            use_numba=use_numba,
        )

    def interaction_builder(density_delta: np.ndarray) -> np.ndarray:
        return active_builder(physical_projector_from_delta(density_delta))

    def oda_parameterizer(state_obj, delta_density: np.ndarray) -> float:
        delta_h = active_builder(delta_density)
        interaction_h = state_obj.hamiltonian - state_obj.h0
        return split_oda_parameter(
            state_obj,
            delta_density,
            delta_h=delta_h,
            interaction_h=interaction_h,
        )

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=lambda hamiltonian: _full_crpa_density_update_result(state, hamiltonian),
        energy_functional=crpa_split_energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=None,
        step_callback=lambda state_obj, step: _update_full_crpa_density_update_state(state_obj, step.density_update),
        final_state_callback=_update_full_crpa_density_update_state,
        convergence_rule="mixed",
    )


def run_bare_split_full_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1.0e-3,
    initial_density: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> RestrictedHartreeFockRun:
    """Run the no-cRPA Zhang bare-split framework with the full-HF updater."""

    normalized_init_mode = normalize_full_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    state.diagnostics["interaction_model"] = "zhang_bare_split"
    kernel = build_bare_split_full_hf_kernel(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        beta=beta,
        use_numba=use_numba,
    )
    base_run = run_hartree_fock_problem(
        state,
        HartreeFockProblem(
            initializer=lambda state_obj, *, init_mode, seed: initialize_full_state(
                state_obj,
                init_mode=init_mode,
                seed=seed,
                initial_density=initial_density,
            ),
            kernel=kernel,
        ),
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


def run_full_crpa_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1.0e-3,
    initial_density: np.ndarray | None = None,
    relative_permittivity: float | None = None,
    screening_lm: float | None = None,
    finite_zero_limit: bool | None = None,
    zero_cutoff: float | None = None,
    fock_interpolation: str = "matrix_diagonal",
    use_numba: bool | None = None,
) -> RestrictedHartreeFockRun:
    normalized_init_mode = normalize_full_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    state.diagnostics["interaction_model"] = "zhang_crpa_screened"
    kernel = build_full_crpa_hf_kernel(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        crpa_screening=crpa_screening,
        beta=beta,
        relative_permittivity=relative_permittivity,
        screening_lm=screening_lm,
        finite_zero_limit=finite_zero_limit,
        zero_cutoff=zero_cutoff,
        fock_interpolation=fock_interpolation,
        use_numba=use_numba,
    )
    base_run = run_hartree_fock_problem(
        state,
        HartreeFockProblem(
            initializer=lambda state_obj, *, init_mode, seed: initialize_full_state(
                state_obj,
                init_mode=init_mode,
                seed=seed,
                initial_density=initial_density,
            ),
            kernel=kernel,
        ),
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


__all__ = [name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None) == __name__ and not name.startswith('_')]
