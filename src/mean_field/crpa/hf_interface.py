from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    compute_hf_energy,
    compute_oda_parameter,
    flavor_block_indices,
    run_hartree_fock_problem,
)
from ..core.hf.overlap import compute_density_overlap_trace_from_diagonal, contract_fock_term_from_overlap
from ..systems.tbg.params import TBGParameters
from ..systems.tbg.zero_field.hf import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    _screened_coulomb_matrix,
    build_full_density_from_hamiltonian,
    coulomb_unit,
    initialize_full_state,
    normalize_full_init_mode,
    occupied_sigma_mean,
    offdiag_flavor_norm,
    restricted_filling,
    restricted_gap_estimate,
)
from .screened_coulomb import CRPAScreenedCoulomb


def build_fock_screened_overlap_blocks(
    overlap_blocks: HFOverlapBlockSet,
    *,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    crpa_screening: CRPAScreenedCoulomb,
    relative_permittivity: float = 4.0,
    screening_lm: float,
    finite_zero_limit: bool = True,
    zero_cutoff: float = 1.0e-6,
) -> HFOverlapBlockSet:
    """Return overlap blocks with Fock kernels divided by cRPA epsilon.

    The current core HF Hartree path accepts one scalar kernel per reciprocal
    shift, so the full non-diagonal Hartree cRPA matrix is intentionally not
    injected here. Use this helper for Fock-only smoke checks or as the Fock
    half of a custom HF+cRPA interaction builder.
    """

    kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        bare = _screened_coulomb_matrix(
            kvec[None, :] - kvec[:, None] + complex(gvec),
            screening_lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
        eps = np.vectorize(crpa_screening.nearest_fock_epsilon, otypes=[float])(
            kvec[None, :] - kvec[:, None] + complex(gvec)
        )
        fock_screening[shift] = bare / eps

    return replace(overlap_blocks, fock_screening=fock_screening)


def build_crpa_projected_interaction_hamiltonian(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Build projected HF interaction with Zhang cRPA screening.

    Hartree uses the full non-diagonal ``Vbar_cRPA(q_tilde=0)`` matrix in the
    Q basis. Fock expects ``overlap_blocks.fock_screening`` to have already
    been replaced by ``V(q)/epsilon(q)`` through
    ``build_fock_screened_overlap_blocks``.
    """

    rho = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = rho.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho.shape}")

    interaction = np.zeros_like(rho)
    if len(overlap_blocks.shifts) == 0:
        return interaction

    v0 = coulomb_unit(params)
    scale = float(beta) * float(v0) / float(nk)
    crpa_q_shifts = [tuple(int(v) for v in row) for row in crpa_screening.result.q_shifts.tolist()]
    crpa_shift_to_index = {shift: idx for idx, shift in enumerate(crpa_q_shifts)}
    hartree_dimless = np.asarray(crpa_screening.get_hartree_screened_v(), dtype=np.complex128) / float(v0)

    hartree_traces: dict[tuple[int, int], complex] = {}
    for shift in overlap_blocks.shifts:
        if shift not in crpa_shift_to_index:
            continue
        diagonal = overlap_blocks.diagonal_overlaps.get(shift)
        if diagonal is None:
            continue
        hartree_traces[shift] = compute_density_overlap_trace_from_diagonal(rho, diagonal, use_numba=use_numba)

    for q2_shift, q2_index in crpa_shift_to_index.items():
        diagonal_q2 = overlap_blocks.diagonal_overlaps.get(q2_shift)
        if diagonal_q2 is None:
            continue
        coeff = 0.0 + 0.0j
        for q1_shift, trace_q1 in hartree_traces.items():
            q1_index = crpa_shift_to_index[q1_shift]
            coeff += hartree_dimless[q1_index, q2_index] * trace_q1
        if coeff != 0.0:
            interaction += scale * coeff * diagonal_q2

    for shift in overlap_blocks.shifts:
        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is None:
            continue
        overlap = overlap_blocks.overlaps[shift]
        if fock_kernel.shape != (nk, nk):
            raise ValueError(f"Expected fock kernel shape {(nk, nk)}, got {fock_kernel.shape} for shift {shift}")
        interaction -= contract_fock_term_from_overlap(
            overlap,
            rho,
            scale * fock_kernel,
            use_numba=use_numba,
        )

    return interaction


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
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Build a full-HF kernel using Zhang cRPA screened interactions."""

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
        relative_permittivity=resolved_relative_permittivity,
        screening_lm=resolved_screening_lm,
        finite_zero_limit=resolved_finite_zero_limit,
        zero_cutoff=resolved_zero_cutoff,
    )

    def interaction_builder(density: np.ndarray) -> np.ndarray:
        return build_crpa_projected_interaction_hamiltonian(
            density,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=lambda hamiltonian: _full_crpa_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: compute_oda_parameter(
            state_obj,
            delta_density,
            interaction_builder=interaction_builder,
        ),
        step_callback=lambda state_obj, step: _update_full_crpa_density_update_state(state_obj, step.density_update),
        final_state_callback=_update_full_crpa_density_update_state,
        convergence_rule="mixed",
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
