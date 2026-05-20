from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    build_projected_interaction_hamiltonian,
    flavor_block_indices,
    run_hartree_fock_problem,
)
from ..core.hf.overlap import compute_density_overlap_trace_from_diagonal, contract_fock_term_from_overlap
from ..systems.tbg.params import TBGParameters
from ..systems.tbg.zero_field.hf import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    _screened_coulomb_matrix,
    _with_tbg_overlap_screening,
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


def half_reference_delta_like(density: np.ndarray) -> np.ndarray:
    """Return the stored-density reference term D_ref = -0.5 I."""

    template = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = template.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {template.shape}")
    out = np.zeros_like(template, dtype=np.complex128)
    diagonal = np.arange(nt)
    out[diagonal, diagonal, :] = -0.5
    if out.shape[2] != nk:
        raise RuntimeError("Internal density-reference construction changed k dimension unexpectedly.")
    return out


def physical_projector_from_delta(density_delta: np.ndarray) -> np.ndarray:
    """Convert the stored full-HF density D = P - 0.5 I to the physical projector P."""

    projector = np.asarray(density_delta, dtype=np.complex128).copy()
    nt, nt_rhs, _nk = projector.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {projector.shape}")
    diagonal = np.arange(nt)
    projector[diagonal, diagonal, :] += 0.5
    return projector


def crpa_split_energy_functional(interaction_hamiltonian: np.ndarray, h0: np.ndarray, density_delta: np.ndarray) -> float:
    """Energy functional for H = h_BM + DeltaH_I^bare + Sigma_cRPA[P]."""

    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    total = np.einsum("abk,abk->", h0, projector, optimize=True)
    total += 0.5 * np.einsum("abk,abk->", interaction_hamiltonian, projector, optimize=True)
    return float(total.real / float(nk))


def split_oda_parameter(
    state_obj,
    delta_density: np.ndarray,
    *,
    delta_h: np.ndarray,
    interaction_h: np.ndarray | None = None,
) -> float:
    """ODA parameter for split Hamiltonians using D = P - 0.5 I storage.

    The split Zhang-style functional is quadratic in the physical projector
    P, while the solver stores the shifted density D.  The last bilinear term
    must therefore contract ``delta_h`` with P, not with D.  Using the generic
    Wang ODA formula with a split ``h0`` would miss the +0.5 I reference term
    and can make the no-cRPA Zhang/Wang trajectories diverge.
    """

    delta = np.asarray(delta_density, dtype=np.complex128)
    delta_interaction = np.asarray(delta_h, dtype=np.complex128)
    active_interaction = (
        np.asarray(state_obj.hamiltonian - state_obj.h0, dtype=np.complex128)
        if interaction_h is None
        else np.asarray(interaction_h, dtype=np.complex128)
    )
    active_projector = physical_projector_from_delta(state_obj.density)

    a = np.einsum("abk,abk->", delta, delta_interaction, optimize=True)
    b = np.einsum("abk,abk->", delta, state_obj.h0, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", delta, active_interaction, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", active_projector, delta_interaction, optimize=True)
    a = float(a.real / state_obj.nk)
    b = float(b.real / state_obj.nk)

    if abs(a) < 1e-15:
        return 1.0 if b < 0.0 else 0.0
    lambda0 = -b / a
    if a > 0.0:
        if lambda0 <= 0.0:
            return 0.0
        if lambda0 < 1.0:
            return float(lambda0)
        return 1.0
    if lambda0 <= 0.5:
        return 1.0
    return 0.0


def build_fock_screened_overlap_blocks(
    overlap_blocks: HFOverlapBlockSet,
    *,
    lattice_kvec: np.ndarray | None = None,
    target_kvec: np.ndarray | None = None,
    source_kvec: np.ndarray | None = None,
    params: TBGParameters,
    crpa_screening: CRPAScreenedCoulomb,
    fock_interpolation: str = "matrix_diagonal",
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

    ``fock_interpolation="matrix_diagonal"`` is the production SCF mode: each
    physical transfer vector is decomposed as ``q_tilde + Q`` and the stored
    dielectric matrix diagonal is used directly. ``linear``/``nearest`` are
    retained for off-grid diagnostic path plots.
    """

    if lattice_kvec is None and (target_kvec is None or source_kvec is None):
        raise ValueError("Pass lattice_kvec for square SCF blocks or both target_kvec and source_kvec.")
    if lattice_kvec is not None:
        target = np.asarray(lattice_kvec, dtype=np.complex128)
        source = target
    else:
        target = np.asarray(target_kvec, dtype=np.complex128)
        source = np.asarray(source_kvec, dtype=np.complex128)

    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        qvals = source[None, :] - target[:, None] + complex(gvec)
        V_bare_with_BN = _screened_coulomb_matrix(
            qvals,
            screening_lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
        eps_crpa = crpa_screening.fock_epsilon_array(
            qvals,
            method=fock_interpolation,
        )
        V_screened_crpa = V_bare_with_BN / eps_crpa
        fock_screening[shift] = V_screened_crpa

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

    hartree, fock = build_crpa_projected_interaction_components(
        density,
        overlap_blocks,
        crpa_screening=crpa_screening,
        params=params,
        beta=beta,
        use_numba=use_numba,
    )
    return hartree + fock


def build_crpa_projected_interaction_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate Hartree and Fock HF potentials for diagnostics.

    ``overlap_blocks`` must already contain the cRPA-screened Fock kernels
    produced by ``build_fock_screened_overlap_blocks``.
    """

    rho = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = rho.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho.shape}")

    hartree = np.zeros_like(rho)
    fock = np.zeros_like(rho)
    if len(overlap_blocks.shifts) == 0:
        return hartree, fock

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
            coeff += hartree_dimless[q2_index, q1_index] * trace_q1
        if coeff != 0.0:
            hartree += scale * coeff * diagonal_q2

    for shift in overlap_blocks.shifts:
        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is None:
            continue
        overlap = overlap_blocks.overlaps[shift]
        if fock_kernel.shape != (nk, nk):
            raise ValueError(f"Expected fock kernel shape {(nk, nk)}, got {fock_kernel.shape} for shift {shift}")
        fock -= contract_fock_term_from_overlap(
            overlap,
            rho,
            scale * fock_kernel,
            use_numba=use_numba,
        )

    return hartree, fock


def crpa_hf_energy_components(
    h0: np.ndarray,
    density_delta: np.ndarray,
    hartree_hamiltonian: np.ndarray,
    fock_hamiltonian: np.ndarray,
) -> dict[str, float]:
    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    e_band = np.einsum("abk,abk->", h0, projector, optimize=True).real / float(nk)
    e_hartree = 0.5 * np.einsum("abk,abk->", hartree_hamiltonian, projector, optimize=True).real / float(nk)
    e_fock = 0.5 * np.einsum("abk,abk->", fock_hamiltonian, projector, optimize=True).real / float(nk)
    return {
        "E_band": float(e_band),
        "E_Hartree": float(e_hartree),
        "E_Fock": float(e_fock),
        "E_total": float(e_band + e_hartree + e_fock),
    }


def build_crpa_projected_target_hamiltonian(
    base_hamiltonian: np.ndarray,
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet,
    target_overlap_blocks: HFOverlapBlockSet,
    target_source_overlap_blocks: HFOverlapBlockSet,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Build an off-grid/path Hamiltonian using the same HF+cRPA interaction.

    This is the cRPA analogue of ``build_projected_target_hamiltonian``:
    source/source densities stay on the self-consistent k mesh, while the
    target Hamiltonian can live on a path mesh. Hartree uses the non-diagonal
    cRPA q=0 matrix and Fock expects target-source blocks whose
    ``fock_screening`` has already been replaced by ``V(q)/epsilon(q)``.
    """

    target_hamiltonian = np.asarray(base_hamiltonian, dtype=np.complex128).copy()
    rho = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk_source = rho.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho.shape}")
    if target_hamiltonian.shape[0] != nt or target_hamiltonian.shape[1] != nt:
        raise ValueError(f"Expected target Hamiltonian flavor dimension {nt}, got {target_hamiltonian.shape}")

    nk_target = target_hamiltonian.shape[2]
    v0 = coulomb_unit(params)
    scale = float(beta) * float(v0) / float(nk_source)
    crpa_q_shifts = [tuple(int(v) for v in row) for row in crpa_screening.result.q_shifts.tolist()]
    crpa_shift_to_index = {shift: idx for idx, shift in enumerate(crpa_q_shifts)}
    hartree_dimless = np.asarray(crpa_screening.get_hartree_screened_v(), dtype=np.complex128) / float(v0)

    hartree_traces: dict[tuple[int, int], complex] = {}
    for shift in source_overlap_blocks.shifts:
        if shift not in crpa_shift_to_index:
            continue
        source_diagonal = source_overlap_blocks.diagonal_overlaps.get(shift)
        if source_diagonal is None:
            continue
        hartree_traces[shift] = compute_density_overlap_trace_from_diagonal(rho, source_diagonal, use_numba=use_numba)

    for q2_shift, q2_index in crpa_shift_to_index.items():
        target_diagonal = target_overlap_blocks.diagonal_overlaps.get(q2_shift)
        if target_diagonal is None:
            continue
        coeff = 0.0 + 0.0j
        for q1_shift, trace_q1 in hartree_traces.items():
            q1_index = crpa_shift_to_index[q1_shift]
            coeff += hartree_dimless[q2_index, q1_index] * trace_q1
        if coeff != 0.0:
            target_hamiltonian += scale * coeff * target_diagonal

    for shift in target_source_overlap_blocks.shifts:
        fock_kernel = target_source_overlap_blocks.fock_screening.get(shift)
        if fock_kernel is None:
            continue
        if fock_kernel.shape != (nk_target, nk_source):
            raise ValueError(f"Expected fock kernel shape {(nk_target, nk_source)}, got {fock_kernel.shape} for shift {shift}")
        target_source_overlap = target_source_overlap_blocks.overlaps[shift]
        target_hamiltonian -= contract_fock_term_from_overlap(
            target_source_overlap,
            rho,
            scale * fock_kernel,
            use_numba=use_numba,
        )

    return target_hamiltonian


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

    density_template = np.asarray(state.density, dtype=np.complex128)
    remote_reference_delta = half_reference_delta_like(density_template)
    remote_bare_hamiltonian = build_projected_interaction_hamiltonian(
        remote_reference_delta,
        overlap_blocks,
        v0=state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    if bool(state.diagnostics.get("crpa_remote_bare_added", 0.0)):
        raise RuntimeError("Refusing to add the bare remote-band cRPA reference term twice to state.h0.")
    state.h0[:, :, :] += remote_bare_hamiltonian
    state.diagnostics["crpa_remote_bare_added"] = 1.0
    state.diagnostics["crpa_remote_bare_fro_norm"] = float(np.linalg.norm(remote_bare_hamiltonian))
    state.diagnostics["crpa_remote_bare_max_abs"] = float(np.max(np.abs(remote_bare_hamiltonian)))

    def crpa_dynamic_builder(projector_or_delta: np.ndarray) -> np.ndarray:
        return build_crpa_projected_interaction_hamiltonian(
            projector_or_delta,
            screened_overlap_blocks,
            crpa_screening=crpa_screening,
            params=params,
            beta=beta,
            use_numba=use_numba,
        )

    def interaction_builder(density_delta: np.ndarray) -> np.ndarray:
        return crpa_dynamic_builder(physical_projector_from_delta(density_delta))

    def oda_delta_interaction_builder(delta_density: np.ndarray) -> np.ndarray:
        return crpa_dynamic_builder(delta_density)

    def oda_parameterizer(state_obj, delta_density: np.ndarray) -> float:
        delta_h = oda_delta_interaction_builder(delta_density)
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
