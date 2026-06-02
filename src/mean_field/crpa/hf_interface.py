from __future__ import annotations

from dataclasses import replace
import os

import numpy as np

from ..core.hf import (
    compute_oda_parameter,
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


_CRPA_SPLIT_MODE_ALIASES = {
    "remote_projector": "remote_projector",
    "remote_bare_projector": "remote_projector",
    "remote+p": "remote_projector",
    "legacy_remote_projector": "remote_projector",
    "production": "active_cnp_fock_reference_projector",
    "remote_delta": "remote_delta",
    "remote_bare_delta": "remote_delta",
    "remote+d": "remote_delta",
    "remote_hartree_delta_fock_projector": "remote_hartree_delta_fock_projector",
    "remote_hartree_d_fock_p": "remote_hartree_delta_fock_projector",
    "remote_hd_fp": "remote_hartree_delta_fock_projector",
    "remote_fock_projector": "remote_fock_projector",
    "remote_fock_p": "remote_fock_projector",
    "remote_fock_only_projector": "remote_fock_projector",
    "remote_fp": "remote_fock_projector",
    "remote_fock_active_cnp_fock_reference_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fock_active_cnp_fock_ref_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fock_cnp_fock_reference_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fp_cnp_fp": "remote_fock_active_cnp_fock_reference_projector",
    "active_cnp_reference_projector": "active_cnp_reference_projector",
    "active_cnp_ref_projector": "active_cnp_reference_projector",
    "cnp_reference_projector": "active_cnp_reference_projector",
    "active_cnp_fock_reference_projector": "active_cnp_fock_reference_projector",
    "active_cnp_fock_ref_projector": "active_cnp_fock_reference_projector",
    "cnp_fock_reference_projector": "active_cnp_fock_reference_projector",
    "minus_active_cnp_fock_projector": "active_cnp_fock_reference_projector",
    "active_cnp_fock_reference_hartree_delta_projector": "active_cnp_fock_reference_hartree_delta_projector",
    "active_cnp_fock_ref_hartree_delta": "active_cnp_fock_reference_hartree_delta_projector",
    "active_cnp_fock_ref_hd_fp": "active_cnp_fock_reference_hartree_delta_projector",
    "cnp_fock_ref_hd_fp": "active_cnp_fock_reference_hartree_delta_projector",
    "no_remote_projector": "no_remote_projector",
    "projector_only": "no_remote_projector",
    "p_only": "no_remote_projector",
    "no_remote_delta": "no_remote_delta",
    "delta_only": "no_remote_delta",
    "d_only": "no_remote_delta",
}


def crpa_split_mode() -> str:
    """Return the diagnostic cRPA split mode.

    The production convention is the flat-subspace cRPA HF split validated
    against Zhang's projected Eq. (17)-(20): build the active cRPA self-energy
    from the physical projector P = D + 0.5 I and subtract the CNP lower-flat
    Fock reference.  The old bare remote-projector split remains available as
    an explicit diagnostic/legacy mode.
    """

    raw = os.environ.get("MEAN_FIELD_CRPA_SPLIT_MODE", "active_cnp_fock_reference_projector")
    normalized = raw.strip().lower().replace("-", "_")
    try:
        return _CRPA_SPLIT_MODE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CRPA_SPLIT_MODE_ALIASES.values())))
        raise ValueError(
            f"Unsupported MEAN_FIELD_CRPA_SPLIT_MODE={raw!r}; allowed canonical modes: {allowed}"
        ) from exc


def crpa_split_uses_remote_bare(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_projector",
        "remote_delta",
        "remote_hartree_delta_fock_projector",
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_projector(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_projector",
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
        "active_cnp_reference_projector",
        "active_cnp_fock_reference_projector",
        "no_remote_projector",
    }


def crpa_split_uses_hartree_delta_fock_projector(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_hartree_delta_fock_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
    }


def crpa_split_uses_remote_fock_only(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_active_cnp_reference(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "active_cnp_reference_projector",
        "active_cnp_fock_reference_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_active_cnp_fock_only(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "active_cnp_fock_reference_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_remote_bare_scale() -> float:
    """Diagnostic scale factor for the bare remote/reference cRPA split term."""

    raw = os.environ.get("MEAN_FIELD_CRPA_REMOTE_BARE_SCALE", "1.0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Unsupported MEAN_FIELD_CRPA_REMOTE_BARE_SCALE={raw!r}; expected a float") from exc
    if not np.isfinite(value):
        raise ValueError(f"Unsupported MEAN_FIELD_CRPA_REMOTE_BARE_SCALE={raw!r}; expected a finite float")
    return value


def crpa_active_density_from_delta(density_delta: np.ndarray, mode: str | None = None) -> np.ndarray:
    resolved = crpa_split_mode() if mode is None else str(mode)
    if crpa_split_uses_projector(resolved):
        return physical_projector_from_delta(density_delta)
    return np.asarray(density_delta, dtype=np.complex128)


def select_remote_reference_components(
    hartree: np.ndarray,
    fock: np.ndarray,
    mode: str | None = None,
) -> np.ndarray:
    """Select the fixed remote/reference one-body term for a diagnostic split."""

    resolved = crpa_split_mode() if mode is None else str(mode)
    if not crpa_split_uses_remote_bare(resolved):
        return np.zeros_like(np.asarray(hartree, dtype=np.complex128))
    hartree_arr = np.asarray(hartree, dtype=np.complex128)
    fock_arr = np.asarray(fock, dtype=np.complex128)
    if hartree_arr.shape != fock_arr.shape:
        raise ValueError(f"Expected matching Hartree/Fock shapes, got {hartree_arr.shape} and {fock_arr.shape}")
    if crpa_split_uses_remote_fock_only(resolved):
        return fock_arr.copy()
    return hartree_arr + fock_arr


def select_active_cnp_reference_components(
    hartree: np.ndarray,
    fock: np.ndarray,
    mode: str | None = None,
) -> np.ndarray:
    """Return the fixed CNP active-reference subtraction for a split mode."""

    resolved = crpa_split_mode() if mode is None else str(mode)
    if not crpa_split_uses_active_cnp_reference(resolved):
        return np.zeros_like(np.asarray(hartree, dtype=np.complex128))
    hartree_arr = np.asarray(hartree, dtype=np.complex128)
    fock_arr = np.asarray(fock, dtype=np.complex128)
    if hartree_arr.shape != fock_arr.shape:
        raise ValueError(f"Expected matching Hartree/Fock shapes, got {hartree_arr.shape} and {fock_arr.shape}")
    if crpa_split_uses_active_cnp_fock_only(resolved):
        return -fock_arr.copy()
    return -(hartree_arr + fock_arr)


def active_lower_flat_projector_like(
    density: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> np.ndarray:
    """Projector for the CNP lower-flat active reference in current state ordering."""

    template = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = template.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {template.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(
            f"Density dimension {nt} does not match n_spin*n_eta*n_band="
            f"{int(n_spin) * int(n_eta) * int(n_band)}"
        )
    if int(n_band) < 2:
        raise ValueError(f"Expected at least two active bands, got n_band={n_band}")
    out = np.zeros_like(template, dtype=np.complex128)
    indices = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            state = int(indices[ispin, ieta, 0])
            out[state, state, :] = 1.0
    if out.shape[2] != nk:
        raise RuntimeError("Internal CNP reference construction changed k dimension unexpectedly.")
    return out


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


def crpa_hartree_delta_fock_projector_energy_components(
    h0: np.ndarray,
    density_delta: np.ndarray,
    hartree_hamiltonian: np.ndarray,
    fock_hamiltonian: np.ndarray,
) -> dict[str, float]:
    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    e_band = np.einsum("abk,abk->", h0, projector, optimize=True).real / float(nk)
    e_hartree = 0.5 * np.einsum("abk,abk->", hartree_hamiltonian, density_delta, optimize=True).real / float(nk)
    e_fock = 0.5 * np.einsum("abk,abk->", fock_hamiltonian, projector, optimize=True).real / float(nk)
    return {
        "E_band": float(e_band),
        "E_Hartree": float(e_hartree),
        "E_Fock": float(e_fock),
        "E_total": float(e_band + e_hartree + e_fock),
    }


def hartree_delta_fock_projector_oda_parameter(
    state_obj,
    delta_density: np.ndarray,
    *,
    delta_hartree_h: np.ndarray,
    delta_fock_h: np.ndarray,
    interaction_h: np.ndarray | None = None,
) -> float:
    """ODA parameter for the diagnostic Hartree[D] + Fock[P] active split."""

    delta = np.asarray(delta_density, dtype=np.complex128)
    delta_hartree = np.asarray(delta_hartree_h, dtype=np.complex128)
    delta_fock = np.asarray(delta_fock_h, dtype=np.complex128)
    delta_interaction = delta_hartree + delta_fock
    active_interaction = (
        np.asarray(state_obj.hamiltonian - state_obj.h0, dtype=np.complex128)
        if interaction_h is None
        else np.asarray(interaction_h, dtype=np.complex128)
    )
    active_projector = physical_projector_from_delta(state_obj.density)

    a = np.einsum("abk,abk->", delta, delta_interaction, optimize=True)
    b = np.einsum("abk,abk->", delta, state_obj.h0, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", delta, active_interaction, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", state_obj.density, delta_hartree, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", active_projector, delta_fock, optimize=True)
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
