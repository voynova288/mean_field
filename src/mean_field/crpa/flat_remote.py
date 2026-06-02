from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.hf import HFOverlapBlockSet, ProjectedWavefunctionBasis, calculate_projected_overlap_between
from ..core.hf.overlap import compute_density_overlap_trace_from_diagonal, contract_fock_term_from_overlap
from ..systems.tbg.zero_field.hf import (
    _hex_shell_contains,
    _precompute_overlap_screening,
    coulomb_unit,
    reciprocal_shift_labels,
)
from ..systems.tbg.zero_field.model import solve_bm_model
from .band_classifier import BandClassification
from .bm import AllBandBMSolution
from .coulomb import CRPACoulombParams


EQ19_FLAT_REMOTE_MODE = "eq19_flat_remote"
HF_ACTIVE_FLAT_MODE = "hf_active_flat"
BM_CHI0_ENERGY_MODE = "bm"

_CHI0_ENERGY_MODE_ALIASES = {
    "bm": BM_CHI0_ENERGY_MODE,
    "bare_bm": BM_CHI0_ENERGY_MODE,
    "bare": BM_CHI0_ENERGY_MODE,
    "hf_active_flat": HF_ACTIVE_FLAT_MODE,
    "hf_flat": HF_ACTIVE_FLAT_MODE,
    "c2t_flat": HF_ACTIVE_FLAT_MODE,
    "active_flat": HF_ACTIVE_FLAT_MODE,
    "eq19": EQ19_FLAT_REMOTE_MODE,
    "eq19_flat_remote": EQ19_FLAT_REMOTE_MODE,
    "eq19_flat": EQ19_FLAT_REMOTE_MODE,
    "flat_remote": EQ19_FLAT_REMOTE_MODE,
    "remote_hf_flat": EQ19_FLAT_REMOTE_MODE,
}


def normalize_chi0_energy_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    try:
        return _CHI0_ENERGY_MODE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CHI0_ENERGY_MODE_ALIASES.values())))
        raise ValueError(f"Unsupported chi0_energy_mode={mode!r}; allowed modes: {allowed}") from exc


def _constant_flat_indices(classification: BandClassification) -> np.ndarray:
    flat_indices = np.asarray(classification.flat_indices, dtype=int)
    if flat_indices.ndim != 3:
        raise ValueError(f"Expected flat_indices shape (valley, k, flat), got {flat_indices.shape}")
    if flat_indices.shape[2] != 2:
        raise ValueError(f"Eq.19 flat correction expects two flat bands, got {flat_indices.shape[2]}")
    reference = flat_indices[0, 0, :].copy()
    if not np.all(flat_indices == reference[None, None, :]):
        raise ValueError(
            "Eq.19 flat correction currently requires a fixed flat-band index pair. "
            "Use flat_method='center' for the cRPA input."
        )
    return reference


def _half_reference_delta_like(density: np.ndarray) -> np.ndarray:
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


def _bare_interaction_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _flat_overlap_block_set(
    solution: AllBandBMSolution,
    flat_indices: np.ndarray,
    *,
    coulomb_params: CRPACoulombParams,
    overlap_lg: int | None = None,
) -> HFOverlapBlockSet:
    flat = np.asarray(flat_indices, dtype=int).reshape(-1)
    basis = ProjectedWavefunctionBasis(
        np.asarray(solution.uk[:, flat, :, :], dtype=np.complex128),
        grid_shape=solution.grid_shape,
        n_spin=2,
        local_basis_size=solution.nlocal,
        name="eq19_flat_remote_chi0_basis",
    )

    shell_lg = int(solution.lg if overlap_lg is None else overlap_lg)
    labels = reciprocal_shift_labels(shell_lg)
    raw_shifts = tuple((m, n) for m in labels for n in labels)
    raw_gvecs = np.asarray(
        [m * solution.params.g1 + n * solution.params.g2 for m, n in raw_shifts],
        dtype=np.complex128,
    )
    active_pairs = [
        (shift, complex(gvec))
        for shift, gvec in zip(raw_shifts, raw_gvecs, strict=True)
        if _hex_shell_contains(solution.params, complex(gvec))
    ]
    shifts = tuple(shift for shift, _gvec in active_pairs)
    gvecs = np.asarray([gvec for _shift, gvec in active_pairs], dtype=np.complex128)
    overlaps = {
        shift: calculate_projected_overlap_between(basis, basis, int(shift[0]), int(shift[1]))
        for shift in shifts
    }
    diagonal_overlaps, hartree_screening, fock_screening = _precompute_overlap_screening(
        shifts,
        gvecs,
        overlaps,
        params=solution.params,
        target_kvec=np.asarray(solution.lattice_kvec, dtype=np.complex128),
        source_kvec=np.asarray(solution.lattice_kvec, dtype=np.complex128),
        relative_permittivity=float(coulomb_params.epsilon_bn),
        screening_lm=float(coulomb_params.screening_lm),
        finite_zero_limit=bool(coulomb_params.finite_zero_limit),
        zero_cutoff=float(coulomb_params.zero_cutoff),
    )
    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def apply_eq19_flat_remote_correction(
    solution: AllBandBMSolution,
    classification: BandClassification,
    *,
    coulomb_params: CRPACoulombParams,
    overlap_lg: int | None = None,
    beta: float = 1.0,
    remove_scalar_shift: bool = True,
    use_numba: bool | None = None,
) -> tuple[AllBandBMSolution, BandClassification, dict[str, object]]:
    """Use h_BM(flat) + Eq.19 remote HF potential as the flat-band chi0 input.

    The remote bands remain the all-band BM eigenstates.  Only the two flat
    columns are rotated within the flat pair and assigned the eigenvalues of
    the Eq.19-corrected two-band Hamiltonian at each valley/k.  The global
    scalar part of Eq.19 is a chemical-potential reference term; by default it
    is removed before using the flat energies in flat-remote chi0 denominators.
    """

    flat_indices = _constant_flat_indices(classification)
    overlap_blocks = _flat_overlap_block_set(
        solution,
        flat_indices,
        coulomb_params=coulomb_params,
        overlap_lg=overlap_lg,
    )
    zero_density = np.zeros((8, 8, solution.nk), dtype=np.complex128)
    remote_ref = _half_reference_delta_like(zero_density)
    remote_hartree, remote_fock = _bare_interaction_components(
        remote_ref,
        overlap_blocks,
        v0=coulomb_unit(solution.params),
        beta=float(beta),
        use_numba=use_numba,
    )
    remote_total = remote_hartree + remote_fock

    new_spectrum = np.asarray(solution.spectrum, dtype=float).copy()
    new_uk = np.asarray(solution.uk, dtype=np.complex128).copy()
    n_flat = int(flat_indices.size)
    state_indices = np.arange(8, dtype=int).reshape((2, solution.n_eta, n_flat), order="F")
    scalar_shift = 0.0
    scalar_count = 0
    if bool(remove_scalar_shift):
        for ieta in range(solution.n_eta):
            rows = state_indices[0, ieta, :]
            for ik in range(solution.nk):
                remote_block = remote_total[np.ix_(rows, rows, [ik])][:, :, 0]
                scalar_shift += float(np.trace(remote_block).real / float(n_flat))
                scalar_count += 1
        scalar_shift = scalar_shift / float(scalar_count)
    max_hermitian_error = 0.0
    identity_flat = np.eye(n_flat, dtype=np.complex128)
    for ieta in range(solution.n_eta):
        rows = state_indices[0, ieta, :]
        for ik in range(solution.nk):
            h_flat = np.diag(np.asarray(solution.spectrum[flat_indices, ieta, ik], dtype=float))
            h_flat = h_flat.astype(np.complex128, copy=False)
            h_flat = h_flat + remote_total[np.ix_(rows, rows, [ik])][:, :, 0]
            h_flat = h_flat - float(scalar_shift) * identity_flat
            hermitian_error = float(np.max(np.abs(h_flat - h_flat.conjugate().T)))
            max_hermitian_error = max(max_hermitian_error, hermitian_error)
            h_flat = 0.5 * (h_flat + h_flat.conjugate().T)
            evals, evecs = np.linalg.eigh(h_flat)
            new_spectrum[flat_indices, ieta, ik] = np.asarray(evals, dtype=float)
            new_uk[:, flat_indices, ieta, ik] = (
                np.asarray(solution.uk[:, flat_indices, ieta, ik], dtype=np.complex128) @ evecs
            )

    old_flat = np.asarray(solution.spectrum[flat_indices, :, :], dtype=float)
    new_flat = np.asarray(new_spectrum[flat_indices, :, :], dtype=float)
    delta = new_flat - old_flat
    metadata: dict[str, object] = {
        "chi0_energy_mode": EQ19_FLAT_REMOTE_MODE,
        "chi0_eq19_remove_scalar_shift": bool(remove_scalar_shift),
        "chi0_eq19_removed_scalar_shift_mev": float(scalar_shift),
        "chi0_eq19_flat_indices": [int(v) for v in flat_indices.tolist()],
        "chi0_eq19_overlap_lg": int(solution.lg if overlap_lg is None else overlap_lg),
        "chi0_eq19_shift_count": int(len(overlap_blocks.shifts)),
        "chi0_eq19_remote_hartree_fro_mev": float(np.linalg.norm(remote_hartree)),
        "chi0_eq19_remote_fock_fro_mev": float(np.linalg.norm(remote_fock)),
        "chi0_eq19_remote_total_max_abs_mev": float(np.max(np.abs(remote_total))),
        "chi0_eq19_flat_delta_min_mev": float(np.min(delta)),
        "chi0_eq19_flat_delta_max_mev": float(np.max(delta)),
        "chi0_eq19_flat_delta_rms_mev": float(np.sqrt(np.mean(delta * delta))),
        "chi0_eq19_flat_pair_span_before_mev": float(np.max(old_flat) - np.min(old_flat)),
        "chi0_eq19_flat_pair_span_after_mev": float(np.max(new_flat) - np.min(new_flat)),
        "chi0_eq19_max_hermitian_error_mev": float(max_hermitian_error),
    }
    corrected_solution = replace(solution, spectrum=new_spectrum, uk=new_uk)
    corrected_classification = replace(classification, energies=new_spectrum)
    return corrected_solution, corrected_classification, metadata


def apply_hf_active_flat_basis(
    solution: AllBandBMSolution,
    classification: BandClassification,
) -> tuple[AllBandBMSolution, BandClassification, dict[str, object]]:
    """Use the HF active C2T flat-band basis for flat-remote chi0 transitions.

    The all-band BM eigensolver keeps the raw ``eigh`` gauge.  The zero-field HF
    solver deliberately C2T-symmetrizes the two active flat bands before building
    overlap blocks.  For an HF-compatible cRPA artifact, flat-band legs in
    flat-remote transitions must use the same active basis as the HF problem.
    Remote-band columns remain the all-band BM eigenvectors.
    """

    flat_indices = _constant_flat_indices(classification)
    active = solve_bm_model(
        solution.params,
        np.asarray(solution.lattice_kvec, dtype=np.complex128),
        lg=int(solution.lg),
        sigma_rotation=bool(solution.sigma_rotation),
        calculate_chern_operator=False,
    )
    if active.uk.shape != (solution.basis_dimension, flat_indices.size, solution.n_eta, solution.nk):
        raise ValueError(
            "HF active flat basis shape does not match all-band cRPA flat slots: "
            f"{active.uk.shape} vs {(solution.basis_dimension, flat_indices.size, solution.n_eta, solution.nk)}"
        )

    old_flat = np.asarray(solution.uk[:, flat_indices, :, :], dtype=np.complex128)
    new_flat = np.asarray(active.uk, dtype=np.complex128)
    overlap_svals: list[float] = []
    max_off_subspace = 0.0
    for ieta in range(solution.n_eta):
        for ik in range(solution.nk):
            overlap = old_flat[:, :, ieta, ik].conjugate().T @ new_flat[:, :, ieta, ik]
            svals = np.linalg.svd(overlap, compute_uv=False)
            overlap_svals.extend(float(v) for v in svals)
            max_off_subspace = max(max_off_subspace, float(np.max(np.abs(overlap.conjugate().T @ overlap - np.eye(flat_indices.size)))))

    new_spectrum = np.asarray(solution.spectrum, dtype=float).copy()
    new_uk = np.asarray(solution.uk, dtype=np.complex128).copy()
    new_spectrum[flat_indices, :, :] = np.asarray(active.spectrum, dtype=float)
    new_uk[:, flat_indices, :, :] = new_flat

    flat_energy_delta = np.asarray(active.spectrum, dtype=float) - np.asarray(
        solution.spectrum[flat_indices, :, :],
        dtype=float,
    )
    metadata: dict[str, object] = {
        "chi0_energy_mode": HF_ACTIVE_FLAT_MODE,
        "chi0_hf_active_flat_indices": [int(v) for v in flat_indices.tolist()],
        "chi0_hf_active_flat_basis": "solve_bm_model C2T-symmetrized active flat columns",
        "chi0_hf_active_flat_min_subspace_singular_value": float(np.min(overlap_svals)),
        "chi0_hf_active_flat_mean_subspace_singular_value": float(np.mean(overlap_svals)),
        "chi0_hf_active_flat_max_projector_error": float(max_off_subspace),
        "chi0_hf_active_flat_energy_delta_max_abs_mev": float(np.max(np.abs(flat_energy_delta))),
    }
    corrected_solution = replace(solution, spectrum=new_spectrum, uk=new_uk)
    corrected_classification = replace(classification, energies=new_spectrum)
    return corrected_solution, corrected_classification, metadata


def apply_chi0_energy_mode(
    solution: AllBandBMSolution,
    classification: BandClassification,
    *,
    mode: str,
    coulomb_params: CRPACoulombParams,
    overlap_lg: int | None = None,
    beta: float = 1.0,
    remove_scalar_shift: bool = True,
    use_numba: bool | None = None,
) -> tuple[AllBandBMSolution, BandClassification, dict[str, object]]:
    resolved = normalize_chi0_energy_mode(mode)
    if resolved == BM_CHI0_ENERGY_MODE:
        return solution, replace(classification, energies=np.asarray(solution.spectrum, dtype=float)), {
            "chi0_energy_mode": BM_CHI0_ENERGY_MODE
        }
    if resolved == HF_ACTIVE_FLAT_MODE:
        return apply_hf_active_flat_basis(solution, classification)
    if resolved == EQ19_FLAT_REMOTE_MODE:
        return apply_eq19_flat_remote_correction(
            solution,
            classification,
            coulomb_params=coulomb_params,
            overlap_lg=overlap_lg,
            beta=float(beta),
            remove_scalar_shift=bool(remove_scalar_shift),
            use_numba=use_numba,
        )
    raise RuntimeError(f"Unhandled normalized chi0 energy mode: {resolved}")
