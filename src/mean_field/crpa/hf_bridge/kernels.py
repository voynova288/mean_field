from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from .density import physical_projector_from_delta
from .split_scheme import crpa_remote_bare_scale

def build_bare_projected_interaction_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate bare Hartree and Fock potentials in the core HF convention."""

    rho = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = rho.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho.shape}")

    hartree = np.zeros_like(rho)
    fock = np.zeros_like(rho)
    scale = float(beta) * float(v0) / float(nk)
    for shift in overlap_blocks.shifts:
        overlap = overlap_blocks.overlaps[shift]
        diagonal = overlap_blocks.diagonal_overlaps.get(shift)
        hartree_kernel = overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            if diagonal is None:
                raise ValueError(f"Missing diagonal overlap for active Hartree shift {shift}")
            trace = compute_density_overlap_trace_from_diagonal(rho, diagonal, use_numba=use_numba)
            hartree += scale * float(hartree_kernel) * trace * diagonal

        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            if fock_kernel.shape != (nk, nk):
                raise ValueError(f"Expected fock kernel shape {(nk, nk)}, got {fock_kernel.shape} for shift {shift}")
            fock -= contract_fock_term_from_overlap(
                overlap,
                rho,
                scale * fock_kernel,
                use_numba=use_numba,
            )
    return hartree, fock


def build_bare_projected_target_components(
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet,
    target_overlap_blocks: HFOverlapBlockSet,
    target_source_overlap_blocks: HFOverlapBlockSet,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate bare Hartree and Fock path potentials."""

    rho = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk_source = rho.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho.shape}")
    if len(target_source_overlap_blocks.shifts) == 0:
        raise ValueError("Target-source overlap blocks are empty; cannot infer target k-count.")
    first_shift = target_source_overlap_blocks.shifts[0]
    first_overlap = target_source_overlap_blocks.overlaps[first_shift]
    nk_target = int(first_overlap.shape[1])
    hartree = np.zeros((nt, nt, nk_target), dtype=np.complex128)
    fock = np.zeros_like(hartree)
    scale = float(beta) * float(v0) / float(nk_source)

    for shift in target_source_overlap_blocks.shifts:
        hartree_kernel = source_overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            source_diagonal = source_overlap_blocks.diagonal_overlaps.get(shift)
            target_diagonal = target_overlap_blocks.diagonal_overlaps.get(shift)
            if source_diagonal is None or target_diagonal is None:
                raise ValueError(f"Missing source/target diagonal overlap for active Hartree shift {shift}")
            trace = compute_density_overlap_trace_from_diagonal(rho, source_diagonal, use_numba=use_numba)
            hartree += scale * float(hartree_kernel) * trace * target_diagonal

        fock_kernel = target_source_overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            if fock_kernel.shape != (nk_target, nk_source):
                raise ValueError(f"Expected fock kernel shape {(nk_target, nk_source)}, got {fock_kernel.shape}")
            fock -= contract_fock_term_from_overlap(
                target_source_overlap_blocks.overlaps[shift],
                rho,
                scale * fock_kernel,
                use_numba=use_numba,
            )
    return hartree, fock


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
    physical transfer vector is decomposed as ``q_tilde + Q`` and the scalar
    divisor implied by ``diag(V) @ epsilon_inv`` is used. ``linear``/``nearest``
    are retained for off-grid diagnostic path plots.
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
    # cRPA replaces V(k' - k + Q) on the same reciprocal-transfer shell that
    # the validated bare HF kernel uses.  Do not promote inactive square-grid
    # corner shifts into the Fock sum.
    active_fock_shifts = set(overlap_blocks.fock_screening)
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        if shift not in active_fock_shifts:
            continue
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

    return build_crpa_projected_interaction_components_from_densities(
        density,
        density,
        overlap_blocks,
        crpa_screening=crpa_screening,
        params=params,
        beta=beta,
        use_numba=use_numba,
    )


def build_crpa_projected_interaction_components_from_densities(
    hartree_density: np.ndarray,
    fock_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cRPA Hartree and Fock potentials from possibly distinct densities.

    This is a diagnostic hook for background-convention checks.  Production
    HF+cRPA uses the same physical projector for both terms through
    ``build_crpa_projected_interaction_components``.
    """

    rho_hartree = np.asarray(hartree_density, dtype=np.complex128)
    rho_fock = np.asarray(fock_density, dtype=np.complex128)
    if rho_hartree.shape != rho_fock.shape:
        raise ValueError(f"Expected matching Hartree/Fock density shapes, got {rho_hartree.shape} and {rho_fock.shape}")
    nt, nt_rhs, nk = rho_hartree.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho_hartree.shape}")

    hartree = np.zeros_like(rho_hartree)
    fock = np.zeros_like(rho_hartree)
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
        hartree_traces[shift] = compute_density_overlap_trace_from_diagonal(rho_hartree, diagonal, use_numba=use_numba)

    for q2_shift, q2_index in crpa_shift_to_index.items():
        diagonal_q2 = overlap_blocks.diagonal_overlaps.get(q2_shift)
        if diagonal_q2 is None:
            continue
        coeff = 0.0 + 0.0j
        for q1_shift, trace_q1 in hartree_traces.items():
            q1_index = crpa_shift_to_index[q1_shift]
            # trace_q1 is the conjugate density component R_Q=<rho_Q>^*,
            # while diagonal_q2 is the output operator S_Q.  The screened
            # interaction energy is R_Q1 W[Q1,Q2] S_Q2, so the Hamiltonian
            # coefficient for output Q2 uses W[input, output].
            coeff += hartree_dimless[q1_index, q2_index] * trace_q1
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
            rho_fock,
            scale * fock_kernel,
            use_numba=use_numba,
        )

    return hartree, fock


def build_crpa_hartree_delta_fock_projector_components(
    density_delta: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic active cRPA split: Hartree from D, Fock from P = D + 0.5 I."""

    return build_crpa_projected_interaction_components_from_densities(
        density_delta,
        physical_projector_from_delta(density_delta),
        overlap_blocks,
        crpa_screening=crpa_screening,
        params=params,
        beta=beta,
        use_numba=use_numba,
    )



def build_crpa_projected_target_components(
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet,
    target_overlap_blocks: HFOverlapBlockSet,
    target_source_overlap_blocks: HFOverlapBlockSet,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate off-grid/path cRPA Hartree and Fock potentials."""

    return build_crpa_projected_target_components_from_densities(
        density,
        density,
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        crpa_screening=crpa_screening,
        params=params,
        beta=beta,
        use_numba=use_numba,
    )


def build_crpa_projected_target_components_from_densities(
    hartree_density: np.ndarray,
    fock_density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet,
    target_overlap_blocks: HFOverlapBlockSet,
    target_source_overlap_blocks: HFOverlapBlockSet,
    crpa_screening: CRPAScreenedCoulomb,
    params: TBGParameters,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return off-grid/path cRPA potentials from distinct Hartree/Fock densities."""

    rho_hartree = np.asarray(hartree_density, dtype=np.complex128)
    rho_fock = np.asarray(fock_density, dtype=np.complex128)
    if rho_hartree.shape != rho_fock.shape:
        raise ValueError(f"Expected matching Hartree/Fock density shapes, got {rho_hartree.shape} and {rho_fock.shape}")
    nt, nt_rhs, nk_source = rho_hartree.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {rho_hartree.shape}")
    if len(target_source_overlap_blocks.shifts) == 0:
        raise ValueError("Target-source overlap blocks are empty; cannot infer target k-count.")
    first_shift = target_source_overlap_blocks.shifts[0]
    first_overlap = target_source_overlap_blocks.overlaps[first_shift]
    nk_target = int(first_overlap.shape[1])
    if first_overlap.shape[0] != nt or first_overlap.shape[2] != nt:
        raise ValueError(f"Expected target-source flavor dimension {nt}, got {first_overlap.shape}")

    hartree = np.zeros((nt, nt, nk_target), dtype=np.complex128)
    fock = np.zeros_like(hartree)
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
        hartree_traces[shift] = compute_density_overlap_trace_from_diagonal(
            rho_hartree,
            source_diagonal,
            use_numba=use_numba,
        )

    for q2_shift, q2_index in crpa_shift_to_index.items():
        target_diagonal = target_overlap_blocks.diagonal_overlaps.get(q2_shift)
        if target_diagonal is None:
            continue
        coeff = 0.0 + 0.0j
        for q1_shift, trace_q1 in hartree_traces.items():
            q1_index = crpa_shift_to_index[q1_shift]
            coeff += hartree_dimless[q1_index, q2_index] * trace_q1
        if coeff != 0.0:
            hartree += scale * coeff * target_diagonal

    for shift in target_source_overlap_blocks.shifts:
        fock_kernel = target_source_overlap_blocks.fock_screening.get(shift)
        if fock_kernel is None:
            continue
        if fock_kernel.shape != (nk_target, nk_source):
            raise ValueError(f"Expected fock kernel shape {(nk_target, nk_source)}, got {fock_kernel.shape} for shift {shift}")
        target_source_overlap = target_source_overlap_blocks.overlaps[shift]
        fock -= contract_fock_term_from_overlap(
            target_source_overlap,
            rho_fock,
            scale * fock_kernel,
            use_numba=use_numba,
        )

    return hartree, fock


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
    hartree, fock = build_crpa_projected_target_components(
        density,
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        crpa_screening=crpa_screening,
        params=params,
        beta=beta,
        use_numba=use_numba,
    )
    if target_hamiltonian.shape != hartree.shape:
        raise ValueError(f"Expected target Hamiltonian shape {hartree.shape}, got {target_hamiltonian.shape}")
    return target_hamiltonian + hartree + fock


__all__ = [name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None) == __name__ and not name.startswith('_')]
