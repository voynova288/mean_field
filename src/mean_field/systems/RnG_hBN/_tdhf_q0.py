from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_support import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403
from ._tdhf_pairs import *  # noqa: F401,F403

def build_rlg_hbn_tdhf_q0_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    *,
    beta: float = 1.0,
    include_direct_terms: bool = True,
    include_exchange_terms: bool = True,
    include_b_terms: bool = True,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
) -> TDHFMatrices:
    """Build dense q=0 TDHF matrices for a pre-filtered pair list.

    The ``vectorized`` path groups pairs by k and performs the layer/form-factor
    contractions with NumPy's compiled kernels.  ``generic`` is retained as a
    small-test reference because it calls the on-demand ``V_hf`` element-by-
    element in Python.
    """

    _reject_zero_literal_q0_fock_env()
    if assembly == "generic":
        interaction = build_rlg_hbn_tdhf_interaction(run, orbitals, beta=beta)
        return build_tdhf_matrices(
            orbitals.global_energies,
            pairs,
            interaction,
            include_direct_terms=include_direct_terms,
            include_exchange_terms=include_exchange_terms,
            include_b_terms=include_b_terms,
            structure_tolerance=structure_tolerance,
        )
    if assembly != "vectorized":
        raise ValueError(f"Unsupported RLG/hBN TDHF assembly mode: {assembly!r}")
    return _build_rlg_hbn_tdhf_q0_matrices_vectorized(
        run,
        orbitals,
        pairs,
        beta=beta,
        include_direct_terms=include_direct_terms,
        include_exchange_terms=include_exchange_terms,
        include_b_terms=include_b_terms,
        structure_tolerance=structure_tolerance,
    )


def _build_rlg_hbn_tdhf_q0_matrices_vectorized(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    *,
    beta: float,
    include_direct_terms: bool,
    include_exchange_terms: bool,
    include_b_terms: bool,
    structure_tolerance: float,
) -> TDHFMatrices:
    ph_pairs = tuple(pairs)
    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    if n_pairs == 0:
        L = assemble_tdhf_liouvillian(A, B)
        structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    p_local = np.empty(n_pairs, dtype=int)
    h_local = np.empty(n_pairs, dtype=int)
    pair_k = np.empty(n_pairs, dtype=int)
    for index, pair in enumerate(ph_pairs):
        p_local[index], p_k = orbitals.decode_global_index(pair.particle)
        h_local[index], h_k = orbitals.decode_global_index(pair.hole)
        if p_k != h_k:
            raise ValueError("RLG/hBN q=0 TDHF pair has particle and hole at different k")
        pair_k[index] = p_k
        A[index, index] = orbitals.global_energies[pair.particle] - orbitals.global_energies[pair.hole]

    indices_by_k = tuple(np.nonzero(pair_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)

    for shift in run.overlap_blocks.shifts:
        layer_overlap = np.asarray(run.overlap_blocks.layer_overlaps[shift], dtype=np.complex128)
        fock_kernel = np.asarray(run.overlap_blocks.fock_layer_coulomb[shift], dtype=float)
        n_layer = int(layer_overlap.shape[0])
        if include_direct_terms:
            F_ph = np.zeros((n_layer, n_pairs), dtype=np.complex128)
            F_hp = np.zeros((n_layer, n_pairs), dtype=np.complex128)
            for ik, indices in enumerate(indices_by_k):
                if indices.size == 0:
                    continue
                u_k = U[:, :, ik]
                p_idx = p_local[indices]
                h_idx = h_local[indices]
                for layer in range(n_layer):
                    full = u_k.conj().T @ layer_overlap[layer, :, ik, :, ik] @ u_k
                    F_ph[layer, indices] = full[p_idx, h_idx]
                    F_hp[layer, indices] = full[h_idx, p_idx]
            for ik, row_indices in enumerate(indices_by_k):
                if row_indices.size == 0:
                    continue
                kernel0 = fock_kernel[ik, ik]
                A[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                    "lm,li,mj->ij",
                    kernel0,
                    F_ph[:, row_indices],
                    np.conj(F_ph),
                    optimize=True,
                )
                if include_b_terms:
                    B[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                        "lm,li,mj->ij",
                        kernel0,
                        F_ph[:, row_indices],
                        np.conj(F_hp),
                        optimize=True,
                    )

        if include_exchange_terms:
            for kt, target_indices in enumerate(indices_by_k):
                if target_indices.size == 0:
                    continue
                u_target = U[:, :, kt]
                p_t = p_local[target_indices]
                h_t = h_local[target_indices]
                for ks, source_indices in enumerate(indices_by_k):
                    if source_indices.size == 0:
                        continue
                    u_source = U[:, :, ks]
                    p_s = p_local[source_indices]
                    h_s = h_local[source_indices]
                    pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                    hh = np.empty_like(pp)
                    ph = np.empty_like(pp) if include_b_terms else None
                    hp = np.empty_like(pp) if include_b_terms else None
                    for layer in range(n_layer):
                        full = u_target.conj().T @ layer_overlap[layer, :, kt, :, ks] @ u_source
                        pp[layer] = full[np.ix_(p_t, p_s)]
                        hh[layer] = full[np.ix_(h_t, h_s)]
                        if include_b_terms:
                            ph[layer] = full[np.ix_(p_t, h_s)]  # type: ignore[index]
                            hp[layer] = full[np.ix_(h_t, p_s)]  # type: ignore[index]
                    kernel = fock_kernel[kt, ks]
                    A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                        "lm,lij,mij->ij",
                        kernel,
                        pp,
                        np.conj(hh),
                        optimize=True,
                    )
                    if include_b_terms:
                        B[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                            "lm,lij,mij->ij",
                            kernel,
                            ph,
                            np.conj(hp),
                            optimize=True,
                        )

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)


def _assert_finite_q_shortcut_is_safe(
    run: RLGhBNHartreeFockRun,
    pairs: tuple[ParticleHolePair, ...],
) -> None:
    if int(run.state.active_valence_bands) != 0:
        raise ValueError("finite-q exchange shortcut requires conduction-only active space")
    if run.state.occupation_counts is None:
        raise ValueError("finite-q exchange shortcut requires saved occupation_counts metadata")
    counts = np.asarray(run.state.occupation_counts, dtype=int).reshape((int(run.state.n_spin), int(run.state.n_eta)), order="C")
    occupied_flavors = [(int(s), int(e)) for s in range(counts.shape[0]) for e in range(counts.shape[1]) if int(counts[s, e]) > 0]
    if len(occupied_flavors) != 1:
        raise ValueError(f"finite-q exchange shortcut requires exactly one occupied flavor, got {occupied_flavors}")
    for pair in pairs:
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q exchange shortcut pairs must carry SpinValleyFlavor metadata")
        if particle.spin == hole.spin and particle.valley == hole.valley:
            raise ValueError("finite-q exchange shortcut is not valid for intraflavor pairs")

__all__ = [name for name in globals() if not name.startswith('__')]
