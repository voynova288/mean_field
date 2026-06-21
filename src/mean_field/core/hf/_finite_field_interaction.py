from __future__ import annotations

from ._finite_field_shared import *  # noqa: F401,F403
from ._finite_field_types import *  # noqa: F401,F403
from ._finite_field_initialization import *  # noqa: F401,F403

def expand_valley_overlap_data_to_flavors(
    valley_k: MagneticOverlapData,
    valley_kprime: MagneticOverlapData,
    *,
    q: int,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> MagneticOverlapData:
    """Expand valley-resolved ``bmLL`` overlaps into the full HF flavor basis.

    Author ``MagneticFieldHF*.jl`` reads one metadata file per valley and copies
    each ``2q`` overlap block into both spin sectors, with no off-diagonal
    valley/spin density-overlap matrix elements.  This helper performs the same
    expansion for Python arrays.
    """

    if tuple(valley_k.shifts) != tuple(valley_kprime.shifts):
        raise ValueError("K and Kprime overlap shift tables must match")
    if np.asarray(valley_k.gvecs).shape != np.asarray(valley_kprime.gvecs).shape:
        raise ValueError("K and Kprime g-vector tables must have matching shape")
    n_sub = int(n_band) * int(q)
    nt = n_sub * int(n_eta) * int(n_spin)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    for shift in valley_k.shifts:
        block_k = np.asarray(valley_k.overlaps[shift], dtype=np.complex128)
        block_kp = np.asarray(valley_kprime.overlaps[shift], dtype=np.complex128)
        if block_k.shape != block_kp.shape:
            raise ValueError(f"K/Kprime overlap shape mismatch for {shift}: {block_k.shape} != {block_kp.shape}")
        if block_k.shape[0] != n_sub or block_k.shape[2] != n_sub:
            raise ValueError(f"Expected valley overlap flavor dimension {n_sub}, got {block_k.shape} for {shift}")
        nk_target = block_k.shape[1]
        nk_source = block_k.shape[3]
        full = np.zeros((nt, nk_target, nt, nk_source), dtype=np.complex128)
        for eta, valley_block in enumerate((block_k, block_kp)):
            for spin in range(int(n_spin)):
                indices = [state_index(band, eta, spin, subbands_per_flavor=n_sub, n_eta=n_eta) for band in range(n_sub)]
                full[np.ix_(indices, np.arange(nk_target), indices, np.arange(nk_source))] = valley_block
        overlaps[shift] = full
    return MagneticOverlapData(shifts=tuple(valley_k.shifts), gvecs=np.asarray(valley_k.gvecs, dtype=np.complex128).copy(), overlaps=overlaps)

def build_magnetic_hf_overlap_block_set(
    overlap_data: MagneticOverlapData,
    *,
    k_vectors: Array,
    screening_lm: float,
    relative_permittivity: float = 15.0,
    use_hartree: bool = True,
    use_fock: bool = True,
) -> HFOverlapBlockSet:
    """Adapt finite-B magnetic overlaps to the generic projected-HF blocks.

    The magnetic-HF full-BZ contraction has the same Hartree/Fock tensor shape
    as :func:`build_projected_interaction_hamiltonian`.  The only convention
    difference is the finite-B normalization factor, which callers supply via an
    effective ``v0`` when invoking the generic interaction builder.
    """

    kvec = np.asarray(k_vectors, dtype=np.complex128)
    if kvec.ndim != 1:
        raise ValueError(f"Expected k_vectors shape (nk,), got {kvec.shape}")
    nk = int(kvec.size)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(overlap_data.shifts, overlap_data.gvecs, strict=True):
        block = np.asarray(overlap_data.overlaps[shift], dtype=np.complex128)
        if block.ndim != 4 or block.shape[1] != nk or block.shape[3] != nk:
            raise ValueError(f"Expected magnetic overlap block with nk={nk}, got {block.shape} for shift {shift}")
        overlaps[shift] = block
        if use_hartree:
            diagonal_overlaps[shift] = diagonal_overlap_blocks(block, nt=block.shape[0], nk=nk)
            hartree_screening[shift] = screened_coulomb_finite_b(
                gvec,
                screening_lm,
                relative_permittivity=relative_permittivity,
            )
        if use_fock:
            qvals = kvec.reshape(1, nk) - kvec.reshape(nk, 1) + gvec  # target ik rows, source ip cols.
            fock_screening[shift] = _screened_coulomb_finite_b_array(
                qvals,
                screening_lm,
                relative_permittivity=relative_permittivity,
            )
    return HFOverlapBlockSet(
        shifts=tuple(overlap_data.shifts),
        gvecs=np.asarray(overlap_data.gvecs, dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_magnetic_interaction_hamiltonian(
    density: Array,
    overlap_data: MagneticOverlapData,
    *,
    k_vectors: Array,
    v0: float,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    relative_permittivity: float = 15.0,
    use_hartree: bool = True,
    use_fock: bool = True,
    use_numba: bool | None = None,
) -> Array:
    """Build full finite-B ``Hartree - Fock`` interaction Hamiltonian.

    This is the direct Python counterpart of ``add_HartreeFock`` in
    ``MagneticFieldHF.jl`` / ``MagneticFieldHF_IKS.jl``.  The finite-B
    normalization is ``1/hf.latt.nk = 1/(q*nq)^2``, not ``1/number_of_MBZ_k``.
    """

    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected density shape (nt,nt,nk), got {density.shape}")
    kvec = np.asarray(k_vectors, dtype=np.complex128)
    if kvec.shape != (nk,):
        raise ValueError(f"Expected k_vectors shape {(nk,)}, got {kvec.shape}")
    if int(normalization_count) <= 0:
        raise ValueError("normalization_count must be positive")
    overlap_blocks = build_magnetic_hf_overlap_block_set(
        overlap_data,
        k_vectors=kvec,
        screening_lm=screening_lm,
        relative_permittivity=relative_permittivity,
        use_hartree=use_hartree,
        use_fock=use_fock,
    )
    effective_v0 = float(v0) * float(nk) / float(normalization_count)
    return build_projected_interaction_hamiltonian(
        density,
        overlap_blocks,
        v0=effective_v0,
        beta=beta,
        use_numba=use_numba,
    )

def apply_iks_phase_to_transposed_density(
    transposed_density: Array,
    *,
    q: int,
    rp_position: int,
    phi: float,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> Array:
    """Apply Eq. S81/S78 IKS phase to ``transpose(P)`` for one tL2 orbit point."""

    mat = np.asarray(transposed_density, dtype=np.complex128).copy()
    if abs(float(phi)) < 1e-15:
        return mat
    n_sub = int(n_band) * int(q)
    expected = n_sub * int(n_eta) * int(n_spin)
    if mat.shape != (expected, expected):
        raise ValueError(f"Expected matrix shape {(expected, expected)}, got {mat.shape}")
    view = mat.reshape((n_sub, n_eta, n_spin, n_sub, n_eta, n_spin), order="F")
    phase = np.exp(1j * float(phi) * int(rp_position))
    view[:, 0, :, :, 1, :] *= np.conj(phase)
    view[:, 1, :, :, 0, :] *= phase
    return mat

def _expanded_iks_transposed_source_density(
    density: Array,
    *,
    indices: Array,
    rps: Array,
    q: int,
    phi: float,
    n_eta: int,
    n_spin: int,
    n_band: int,
) -> Array:
    """Expand reduced IKS ``P.T`` slices onto the full magnetic-orbit source grid."""

    nt, _, nk_reduced = density.shape
    expanded = np.empty((nt, nt, int(q) * nk_reduced), dtype=np.complex128)
    for ip in range(nk_reduced):
        for rp in range(int(q)):
            full_source = int(indices[rp, ip])
            if abs(float(phi)) < 1e-15:
                expanded[:, :, full_source] = density[:, :, ip].T
            else:
                expanded[:, :, full_source] = apply_iks_phase_to_transposed_density(
                    density[:, :, ip].T,
                    q=q,
                    rp_position=int(rps[rp]),
                    phi=phi,
                    n_eta=n_eta,
                    n_spin=n_spin,
                    n_band=n_band,
                )
    return expanded

def build_tl_symmetric_magnetic_interaction_hamiltonian(
    density: Array,
    overlap_data: MagneticOverlapData,
    *,
    full_k_vectors: Array,
    flux: MagneticFlux,
    nq: int,
    v0: float,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
    use_numba: bool | None = None,
) -> Array:
    """Build the magnetic-translation-symmetric / IKS-reduced HF Hamiltonian.

    This ports ``MagneticFieldHF_tLSymmetric*_IKS*.jl``: the density is defined
    on ``nq**2`` reduced momenta, while Fock sums over the ``q``-point orbit
    generated by ``t_L2``.  Hartree terms survive only for ``n % q == 0`` and
    carry the author-code factor ``q``.
    """

    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk_reduced = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected density shape (nt,nt,nk), got {density.shape}")
    q = int(flux.q)
    nq = int(nq)
    if nk_reduced != nq * nq:
        raise ValueError(f"Expected reduced nk=nq**2={nq*nq}, got {nk_reduced}")
    indices = magnetic_orbit_indices(q, nq)
    full_k = np.asarray(full_k_vectors, dtype=np.complex128)
    if full_k.shape != (q * nk_reduced,):
        raise ValueError(f"Expected full_k_vectors shape {(q * nk_reduced,)}, got {full_k.shape}")
    rps = magnetic_r_orbit_positions(flux.p, flux.q)
    out = np.zeros_like(density)
    prefactor = float(beta) * float(v0) / float(normalization_count)
    reduced_targets = indices[0, :].astype(int, copy=False)
    source_density_t = _expanded_iks_transposed_source_density(
        density,
        indices=indices,
        rps=rps,
        q=q,
        phi=phi,
        n_eta=n_eta,
        n_spin=n_spin,
        n_band=n_band,
    )

    for shift, gvec in zip(overlap_data.shifts, overlap_data.gvecs, strict=True):
        m, n = shift
        _ = m
        overlap = np.asarray(overlap_data.overlaps[shift], dtype=np.complex128)
        expected_shape = (nt, q * nk_reduced, nt, q * nk_reduced)
        if overlap.shape != expected_shape:
            raise ValueError(f"Expected tL overlap shape {expected_shape}, got {overlap.shape} for shift {shift}")
        kernel_g = screened_coulomb_finite_b(gvec, screening_lm, relative_permittivity=relative_permittivity)
        if n % q == 0 and kernel_g != 0.0:
            diagonal = np.stack([overlap[:, int(full_ik), :, int(full_ik)] for full_ik in reduced_targets], axis=2).astype(np.complex128, copy=False)
            tr_pg = compute_density_overlap_trace_from_diagonal(density, diagonal, use_numba=use_numba)
            out += prefactor * kernel_g * tr_pg * q * diagonal

        target_k = full_k[reduced_targets]
        qvals = full_k.reshape(1, q * nk_reduced) - target_k.reshape(nk_reduced, 1) + gvec
        fock_kernel = prefactor * _screened_coulomb_finite_b_array(qvals, screening_lm, relative_permittivity=relative_permittivity)
        for ik, full_target in enumerate(reduced_targets):
            tmp_fock = np.zeros((nt, nt), dtype=np.complex128)
            for full_source in range(q * nk_reduced):
                coeff = fock_kernel[ik, full_source]
                if coeff == 0.0:
                    continue
                lam = overlap[:, int(full_target), :, full_source]
                tmp_fock += coeff * (lam @ source_density_t[:, :, full_source] @ lam.conj().T)
            out[:, :, ik] -= tmp_fock
    return out

def compute_finite_field_hf_energy(interaction_hamiltonian: Array, h0: Array, density: Array) -> float:
    """Author-code energy per moire unit cell, ``8/(nt*nk)`` times the trace."""

    interaction = np.asarray(interaction_hamiltonian, dtype=np.complex128)
    h0_arr = np.asarray(h0, dtype=np.complex128)
    p = np.asarray(density, dtype=np.complex128)
    total = np.einsum("abk,abk->", interaction, p, optimize=True) / 2.0
    total += np.einsum("abk,abk->", h0_arr, p, optimize=True)
    return float((8.0 * total.real) / (h0_arr.shape[0] * h0_arr.shape[2]))

__all__ = [name for name in globals() if not name.startswith('__')]
