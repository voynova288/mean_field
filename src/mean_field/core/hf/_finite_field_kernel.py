from __future__ import annotations

from ._finite_field_shared import *  # noqa: F401,F403
from ._finite_field_types import *  # noqa: F401,F403
from ._finite_field_initialization import *  # noqa: F401,F403
from ._finite_field_interaction import *  # noqa: F401,F403

def summarize_finite_field_hartree_fock(
    state: FiniteFieldHartreeFockState,
    run: HartreeFockRun | None = None,
) -> FiniteFieldHartreeFockSummary:
    """Return a compact no-I/O summary for checkpoint comparisons."""

    n_occ = finite_field_occupied_state_count(state.nu, state.nt, state.nk)
    flat_energies = np.sort(np.asarray(state.energies, dtype=float).reshape(-1))
    if 0 < n_occ < flat_energies.size:
        gap = float(flat_energies[n_occ] - flat_energies[n_occ - 1])
    else:
        gap = float("nan")
    energy = float(state.diagnostics.get("hf_energy", np.nan))
    final_raw_norm = float(state.diagnostics.get("final_raw_norm", np.nan))
    if run is None:
        iterations = int(round(float(state.diagnostics.get("iterations", 0.0))))
        converged = bool(final_raw_norm <= state.precision) if np.isfinite(final_raw_norm) else False
        exit_reason = "unknown"
    else:
        iterations = int(run.iterations)
        converged = bool(run.converged)
        exit_reason = str(run.exit_reason)
    return FiniteFieldHartreeFockSummary(
        filling=finite_field_filling(state.density),
        energy_per_muc=energy,
        mu=float(state.mu),
        single_particle_gap=gap,
        final_raw_norm=final_raw_norm,
        iterations=iterations,
        converged=converged,
        exit_reason=exit_reason,
    )

def calculate_valley_spin_order_parameters(
    hamiltonian: Array,
    energies: Array,
    mu: float,
    *,
    q: int,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> dict[str, float]:
    """Return ``s_i eta_j`` order parameters in the convention of the Julia code."""

    pauli = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    sub = int(n_band) * int(q)
    identity_band = np.eye(sub, dtype=np.complex128)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    eps = np.asarray(energies, dtype=float)
    out: dict[str, float] = {}
    nk = h.shape[2]
    for ispin, spin_mat in enumerate(pauli):
        for ieta, eta_mat in enumerate(pauli):
            op = np.kron(spin_mat, np.kron(eta_mat, identity_band))
            values = np.zeros_like(eps)
            for ik in range(nk):
                _vals, vecs = np.linalg.eigh(h[:, :, ik])
                values[:, ik] = np.diag(vecs.conj().T @ op @ vecs).real
            out[f"s{ispin}_eta{ieta}"] = float(values[eps <= float(mu)].sum() / eps.size * 8.0)
    return out

def _finite_field_density_builder_for_state(
    state: FiniteFieldHartreeFockState,
) -> Callable[[Array], DensityUpdateResult]:
    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        result = density_update_from_hamiltonian(hamiltonian, nu=state.nu, sigma_z=state.sigma_z)
        sigma_obs = result.observables.get("sigma_ztauz")
        if isinstance(sigma_obs, np.ndarray) and sigma_obs.shape == state.sigma_ztauz.shape:
            state.sigma_ztauz[:, :] = sigma_obs
        return result

    return density_builder


def build_finite_field_hf_kernel(
    state: FiniteFieldHartreeFockState,
    overlap_data: MagneticOverlapData,
    *,
    k_vectors: Array,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    kvec = np.asarray(k_vectors, dtype=np.complex128)
    if kvec.shape != (state.nk,):
        raise ValueError(f"Expected k_vectors shape {(state.nk,)}, got {kvec.shape}")
    if int(normalization_count) <= 0:
        raise ValueError("normalization_count must be positive")
    overlap_blocks = build_magnetic_hf_overlap_block_set(
        overlap_data,
        k_vectors=kvec,
        screening_lm=screening_lm,
        relative_permittivity=relative_permittivity,
    )
    effective_v0 = float(state.v0) * float(state.nk) / float(normalization_count)
    return build_projected_hf_kernel(
        state,
        overlap_blocks,
        density_builder=_finite_field_density_builder_for_state(state),
        v0=effective_v0,
        beta=beta,
        energy_functional=compute_finite_field_hf_energy,
        convergence_rule="mixed",
        use_numba=use_numba,
    )

def build_tl_symmetric_finite_field_hf_kernel(
    state: FiniteFieldHartreeFockState,
    overlap_data: MagneticOverlapData,
    *,
    full_k_vectors: Array,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    interaction_builder = lambda density: build_tl_symmetric_magnetic_interaction_hamiltonian(
        density,
        overlap_data,
        full_k_vectors=full_k_vectors,
        flux=state.flux,
        nq=state.nq,
        v0=state.v0,
        normalization_count=normalization_count,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=_finite_field_density_builder_for_state(state),
        energy_functional=compute_finite_field_hf_energy,
        oda_delta_interaction_builder=interaction_builder,
        convergence_rule="mixed",
    )

def build_finite_field_hf_problem(
    kernel: HartreeFockKernel,
    *,
    initializer: Callable[[FiniteFieldHartreeFockState, str, int], None] | None = None,
) -> HartreeFockProblem:
    def default_initializer(state, *, init_mode: str, seed: int) -> None:
        if initializer is None:
            initialize_density_from_h0(state, init_mode=init_mode, seed=seed)
        else:
            initializer(state, init_mode, seed)

    return HartreeFockProblem(initializer=default_initializer, kernel=kernel)

def run_finite_field_hartree_fock(
    state: FiniteFieldHartreeFockState,
    kernel: HartreeFockKernel,
    *,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
) -> HartreeFockRun:
    problem = build_finite_field_hf_problem(kernel)
    return run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )

def build_finite_field_hf_kernel_from_inputs(
    inputs: FiniteFieldHartreeFockInputBundle,
    *,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Build an HF kernel from any assembled finite-B input bundle.

    Full magnetic-BZ bundles dispatch to :func:`build_finite_field_hf_kernel`.
    Reduced tL-symmetric/IKS bundles dispatch to
    :func:`build_tl_symmetric_finite_field_hf_kernel` and use ``phi`` for the
    IKS phase.  This keeps workflow code on one API while preserving the two
    physics contractions internally.
    """

    if isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs):
        return build_tl_symmetric_finite_field_hf_kernel(
            inputs.state,
            inputs.overlap_data,
            full_k_vectors=inputs.full_k_vectors,
            normalization_count=inputs.normalization_count,
            screening_lm=screening_lm,
            beta=beta,
            phi=phi,
            relative_permittivity=relative_permittivity,
            use_numba=use_numba,
        )
    if isinstance(inputs, FiniteFieldHartreeFockInputs):
        return build_finite_field_hf_kernel(
            inputs.state,
            inputs.overlap_data,
            k_vectors=inputs.k_vectors,
            normalization_count=inputs.normalization_count,
            screening_lm=screening_lm,
            beta=beta,
            relative_permittivity=relative_permittivity,
            use_numba=use_numba,
        )
    raise TypeError(f"Unsupported finite-field HF input bundle type: {type(inputs).__name__}")

def run_finite_field_hartree_fock_from_inputs(
    inputs: FiniteFieldHartreeFockInputBundle,
    *,
    screening_lm: float,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockRun:
    """Run finite-B HF from a full or reduced no-I/O input bundle.

    The generic SCF/ODA loop still lives in :mod:`mean_field.core.hf`; this
    adapter only builds the correct finite-B TBG kernel for the provided bundle.
    """

    kernel = build_finite_field_hf_kernel_from_inputs(
        inputs,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )
    return run_finite_field_hartree_fock(
        inputs.state,
        kernel,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )

def build_tl_symmetric_finite_field_hf_kernel_from_inputs(
    inputs: FiniteFieldTLSymmetricHartreeFockInputs,
    *,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF kernels."""

    return build_finite_field_hf_kernel_from_inputs(
        inputs,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )

def run_tl_symmetric_finite_field_hartree_fock_from_inputs(
    inputs: FiniteFieldTLSymmetricHartreeFockInputs,
    *,
    screening_lm: float,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockRun:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF runs."""

    return run_finite_field_hartree_fock_from_inputs(
        inputs,
        screening_lm=screening_lm,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
