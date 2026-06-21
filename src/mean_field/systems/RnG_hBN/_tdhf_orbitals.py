from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403

def _flavor_block_offdiag_residual(
    hamiltonian: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> float:
    indices = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    max_residual = 0.0
    for ik in range(hamiltonian.shape[2]):
        for spin_a in range(int(n_spin)):
            for eta_a in range(int(n_eta)):
                rows = np.asarray(indices[spin_a, eta_a, :], dtype=int)
                for spin_b in range(int(n_spin)):
                    for eta_b in range(int(n_eta)):
                        if spin_a == spin_b and eta_a == eta_b:
                            continue
                        cols = np.asarray(indices[spin_b, eta_b, :], dtype=int)
                        block = hamiltonian[:, :, ik][np.ix_(rows, cols)]
                        if block.size:
                            max_residual = max(max_residual, float(np.max(np.abs(block))))
    return max_residual


def build_rlg_hbn_tdhf_orbitals(state: RLGhBNHartreeFockState) -> RLGhBNTDHFOrbitals:
    """Diagonalize the converged HF Hamiltonian in the same ordering as HF."""

    hamiltonian = np.asarray(state.hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square HF Hamiltonian blocks, got {hamiltonian.shape}")
    n_spin = int(state.n_spin)
    n_eta = int(state.n_eta)
    n_band = int(state.n_band)
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"HF dimension {nt} incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    energies = np.zeros((nt, nk), dtype=float)
    eigenvectors = np.zeros((nt, nt, nk), dtype=np.complex128)
    occ_mask = np.zeros((nt, nk), dtype=bool)

    if state.occupation_counts is not None:
        offdiag_residual = _flavor_block_offdiag_residual(
            hamiltonian,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
        if offdiag_residual > 1.0e-8:
            raise ValueError(
                "occupation_counts TDHF orbital shortcut requires a spin-valley block-diagonal HF Hamiltonian; "
                f"max off-block element is {offdiag_residual:.6e}. Use full diagonalization/occupation logic for "
                "flavor-mixed, IVC, or translation-breaking states."
            )
        counts = np.asarray(state.occupation_counts, dtype=int).reshape((n_spin, n_eta), order="C")
        indices = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
        for ik in range(nk):
            for ispin in range(n_spin):
                for ieta in range(n_eta):
                    block_indices = np.asarray(indices[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
                    eigenvectors[np.ix_(block_indices, block_indices, [ik])] = eigvecs[:, :, None]
                    n_occ = int(counts[ispin, ieta])
                    if n_occ > 0:
                        occ_mask[block_indices[:n_occ], ik] = True
    else:
        for ik in range(nk):
            eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
            energies[:, ik] = eigvals
            eigenvectors[:, :, ik] = eigvecs
        total_occupied = rlg_hbn_occupied_state_count(
            state.nu,
            nt,
            nk,
            active_valence_bands=state.active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
        )
        occ_mask[:, :] = occupied_state_mask(energies, total_occupied)

    if np.any(occ_mask) and not np.all(occ_mask):
        mu = 0.5 * (float(np.max(energies[occ_mask])) + float(np.min(energies[~occ_mask])))
    else:
        mu = float(np.mean(energies))
    return RLGhBNTDHFOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occ_mask,
        mu=mu,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )


def _canonical_hf_state_from_input(canonical_hf: ContractHFState | ContractHFRunResult) -> ContractHFState:
    if isinstance(canonical_hf, ContractHFRunResult):
        return canonical_hf.final_state
    if isinstance(canonical_hf, ContractHFState):
        return canonical_hf
    raise TypeError("canonical_hf must be a mean_field.core.contracts.HFState or HFRunResult")


def _metadata_sequence_len(value: object) -> int:
    if value is None or isinstance(value, (str, bytes)):
        return 0
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _infer_rlg_hbn_dimensions_from_canonical_state(
    state: ContractHFState,
    *,
    n_spin: int | None,
    n_eta: int | None,
    n_band: int | None,
) -> tuple[int, int, int]:
    system = str(getattr(state.basis.physical_model, "system", ""))
    basis_system = str(getattr(state.basis.basis_model, "system", ""))
    if system != "RnG_hBN" or basis_system != "RnG_hBN":
        raise ValueError(
            "RLG/hBN TDHF canonical adapter only accepts canonical HFState/HFRunResult for system "
            f"'RnG_hBN'; got physical={system!r}, basis={basis_system!r}"
        )

    metadata = dict(state.basis.metadata)
    resolved_n_eta = int(n_eta) if n_eta is not None else _metadata_sequence_len(metadata.get("valleys"))
    if resolved_n_eta <= 0:
        resolved_n_eta = 2

    if n_band is not None:
        resolved_n_band = int(n_band)
    else:
        resolved_n_band = _metadata_sequence_len(metadata.get("active_band_indices_per_band"))
        if resolved_n_band <= 0:
            resolved_n_band = int(state.basis.active_valence_bands) + int(state.basis.active_conduction_bands)
    if resolved_n_band <= 0:
        raise ValueError("Cannot infer positive RLG/hBN n_band from canonical ProjectedBasis metadata")

    nt = int(state.hamiltonian.total.shape[0])
    if n_spin is not None:
        resolved_n_spin = int(n_spin)
    else:
        denom = resolved_n_eta * resolved_n_band
        if denom <= 0 or nt % denom != 0:
            raise ValueError(
                "Cannot infer RLG/hBN n_spin from canonical dimensions: "
                f"nt={nt}, n_eta={resolved_n_eta}, n_band={resolved_n_band}"
            )
        resolved_n_spin = nt // denom
    if resolved_n_spin <= 0:
        raise ValueError(f"RLG/hBN n_spin must be positive, got {resolved_n_spin}")
    if resolved_n_spin * resolved_n_eta * resolved_n_band != nt:
        raise ValueError(
            "Canonical HFState dimension is incompatible with requested RLG/hBN dimensions: "
            f"nt={nt}, n_spin={resolved_n_spin}, n_eta={resolved_n_eta}, n_band={resolved_n_band}"
        )
    return resolved_n_spin, resolved_n_eta, resolved_n_band


def _max_abs(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _reorder_canonical_orbitals_to_rlg_hbn_flavor_order(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    occupied_mask: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    flavor_resolution_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nt, nk = int(energies.shape[0]), int(energies.shape[1])
    indices = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
    reordered_energies = np.empty_like(np.asarray(energies, dtype=float))
    reordered_vectors = np.empty_like(np.asarray(eigenvectors, dtype=np.complex128))
    reordered_occupied = np.empty_like(np.asarray(occupied_mask, dtype=bool))
    tolerance = float(flavor_resolution_tolerance)

    for ik in range(nk):
        assigned = np.zeros((nt,), dtype=bool)
        for ispin in range(int(n_spin)):
            for ieta in range(int(n_eta)):
                target_indices = np.asarray(indices[ispin, ieta, :], dtype=int)
                weights = np.sum(np.abs(eigenvectors[target_indices, :, ik]) ** 2, axis=0)
                selected = np.flatnonzero(weights >= 1.0 - tolerance)
                if selected.size != int(n_band):
                    best_weight = float(np.max(weights)) if weights.size else 0.0
                    raise ValueError(
                        "Canonical RLG/hBN TDHF orbitals are not spin-valley flavor-resolved. "
                        "The generic canonical diagonalization may mix degenerate or flavor-mixed sectors, "
                        "so RLG/hBN flavor-tagged TDHF pairs would be ambiguous; use the legacy system "
                        "orbital builder or add a system-specific flavor gauge. "
                        f"k={ik}, spin={ispin}, eta={ieta}, selected={selected.size}, "
                        f"expected={n_band}, best_block_weight={best_weight:.6e}"
                    )
                if np.any(assigned[selected]):
                    raise ValueError("Canonical RLG/hBN TDHF flavor assignment is not one-to-one")
                order = np.argsort(np.asarray(energies[selected, ik], dtype=float), kind="stable")
                selected = selected[order]
                reordered_energies[target_indices, ik] = energies[selected, ik]
                reordered_vectors[:, target_indices, ik] = eigenvectors[:, selected, ik]
                reordered_occupied[target_indices, ik] = occupied_mask[selected, ik]
                assigned[selected] = True
        if not np.all(assigned):
            missing = np.flatnonzero(~assigned)
            raise ValueError(f"Canonical RLG/hBN TDHF flavor assignment missed HF states at k={ik}: {missing.tolist()}")
    return reordered_energies, reordered_vectors, reordered_occupied


def build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    n_spin: int | None = None,
    n_eta: int | None = None,
    n_band: int | None = None,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> RLGhBNTDHFOrbitals:
    """Convert canonical HFState/HFRunResult orbitals into RLG/hBN TDHF ordering.

    The generic core boundary diagonalizes the full HF Hamiltonian.  RLG/hBN
    TDHF still assigns particle-hole flavor metadata from the local orbital
    index, so this adapter only accepts canonical orbitals that can be resolved
    unambiguously into the RLG/hBN spin-valley blocks.  Flavor-mixed or
    degenerate gauges are rejected instead of fabricating flavor-tagged TDHF
    sectors.
    """

    state = _canonical_hf_state_from_input(canonical_hf)
    resolved_n_spin, resolved_n_eta, resolved_n_band = _infer_rlg_hbn_dimensions_from_canonical_state(
        state,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )
    offdiag_residual = _flavor_block_offdiag_residual(
        np.asarray(state.hamiltonian.total, dtype=np.complex128),
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
    )
    if offdiag_residual > float(flavor_resolution_tolerance):
        raise ValueError(
            "Canonical RLG/hBN TDHF adapter requires a spin-valley block-diagonal HF Hamiltonian; "
            f"max off-block element is {offdiag_residual:.6e}. Full flavor-mixed RLG/hBN TDHF needs a "
            "separate system-specific flavor/gauge treatment."
        )

    if isinstance(canonical_hf, ContractHFRunResult):
        canonical = canonical_tdhf_orbitals_from_hf_run_result(
            canonical_hf,
            occupation_policy=occupation_policy,
            projector_tolerance=projector_tolerance,
            degeneracy_tolerance=degeneracy_tolerance,
        )
    else:
        canonical = canonical_tdhf_orbitals_from_hf_state(
            state,
            occupation_policy=occupation_policy,
            projector_tolerance=projector_tolerance,
            degeneracy_tolerance=degeneracy_tolerance,
        )
    energies, eigenvectors, occupied_mask = _reorder_canonical_orbitals_to_rlg_hbn_flavor_order(
        canonical.energies,
        canonical.eigenvectors,
        canonical.occupied_mask,
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    return RLGhBNTDHFOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occupied_mask,
        mu=float(canonical.mu),
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
    )


def _occupied_projector_from_tdhf_orbitals(orbitals: RLGhBNTDHFOrbitals, k_index: int) -> np.ndarray:
    vectors = np.asarray(orbitals.eigenvectors[:, :, int(k_index)], dtype=np.complex128)
    occupations = np.diag(np.asarray(orbitals.occupied_mask[:, int(k_index)], dtype=float))
    return vectors @ occupations @ vectors.conjugate().T


def _rlg_hbn_tdhf_orbital_parity_metrics(
    legacy: RLGhBNTDHFOrbitals,
    canonical: RLGhBNTDHFOrbitals,
) -> dict[str, float]:
    if legacy.energies.shape != canonical.energies.shape:
        raise ValueError(f"TDHF orbital energy shapes differ: {legacy.energies.shape} vs {canonical.energies.shape}")
    if legacy.eigenvectors.shape != canonical.eigenvectors.shape:
        raise ValueError(
            f"TDHF orbital eigenvector shapes differ: {legacy.eigenvectors.shape} vs {canonical.eigenvectors.shape}"
        )
    if legacy.occupied_mask.shape != canonical.occupied_mask.shape:
        raise ValueError(
            f"TDHF occupied-mask shapes differ: {legacy.occupied_mask.shape} vs {canonical.occupied_mask.shape}"
        )

    energy_residual = _max_abs(np.asarray(legacy.energies) - np.asarray(canonical.energies))
    occupied_mask_mismatches = float(np.count_nonzero(np.asarray(legacy.occupied_mask) != np.asarray(canonical.occupied_mask)))
    vector_overlap_residual = 0.0
    occupied_projector_residual = 0.0
    for ik in range(legacy.nk):
        overlap = legacy.eigenvectors[:, :, ik].conjugate().T @ canonical.eigenvectors[:, :, ik]
        diagonal_abs = np.abs(np.diag(overlap))
        offdiag = overlap.copy()
        index = np.arange(overlap.shape[0])
        offdiag[index, index] = 0.0
        vector_overlap_residual = max(
            vector_overlap_residual,
            _max_abs(diagonal_abs - 1.0),
            _max_abs(offdiag),
        )
        occupied_projector_residual = max(
            occupied_projector_residual,
            _max_abs(
                _occupied_projector_from_tdhf_orbitals(legacy, ik)
                - _occupied_projector_from_tdhf_orbitals(canonical, ik)
            ),
        )
    return {
        "energy_residual": float(energy_residual),
        "occupied_mask_mismatches": occupied_mask_mismatches,
        "vector_overlap_residual": float(vector_overlap_residual),
        "occupied_projector_residual": float(occupied_projector_residual),
    }


def _validate_rlg_hbn_tdhf_orbital_parity(
    legacy: RLGhBNTDHFOrbitals,
    canonical: RLGhBNTDHFOrbitals,
    *,
    tolerance: float,
) -> dict[str, float]:
    metrics = _rlg_hbn_tdhf_orbital_parity_metrics(legacy, canonical)
    failures = {
        key: value
        for key, value in metrics.items()
        if (key == "occupied_mask_mismatches" and value != 0.0)
        or (key != "occupied_mask_mismatches" and value > float(tolerance))
    }
    if failures:
        raise ValueError(
            "Canonical RLG/hBN TDHF orbitals do not match the existing system orbital path within "
            f"tolerance {float(tolerance):.6e}: {failures}"
        )
    return metrics


def validate_rlg_hbn_tdhf_canonical_orbital_parity(
    state: RLGhBNHartreeFockState,
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> dict[str, float]:
    """Validate canonical HFState/HFRunResult TDHF orbitals against RLG/hBN legacy orbitals."""

    legacy = build_rlg_hbn_tdhf_orbitals(state)
    canonical = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    return _validate_rlg_hbn_tdhf_orbital_parity(legacy, canonical, tolerance=tolerance)

__all__ = [name for name in globals() if not name.startswith('__')]
