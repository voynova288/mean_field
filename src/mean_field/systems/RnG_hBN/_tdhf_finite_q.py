from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_support import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403
from ._tdhf_pairs import *  # noqa: F401,F403
from ._tdhf_q0 import *  # noqa: F401,F403

def build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    beta: float = 1.0,
    structure_tolerance: float = 1.0e-6,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
) -> TDHFMatrices:
    """Build finite-q TDHF matrices for flavor-flip shortcut channels.

    This is the first finite-q production path needed for Fig. S45 spin and
    valley dispersions.  It intentionally implements only the conduction-only,
    fully polarized shortcut case where direct and B terms vanish and the A
    block contains the one-body term plus exchange.  Intra-flavor finite-q RPA
    requires the full Eq. D19 X/Y q/-q bookkeeping and is deliberately not
    hidden behind this shortcut helper.

    Periodic wrapping is handled by treating the loop variable as the *physical*
    Umklapp ``G``.  For a form factor whose target/source momenta have integer
    reciprocal wraps ``W_target`` and ``W_source``, the cached overlap shift is
    ``G + W_target - W_source``.  If the overlap block set has been augmented
    with extra closure keys, pass the original Coulomb-cutoff keys through
    ``physical_shifts`` so they are used only as cached form factors, not as
    extra physical G terms in the sum.
    """

    _reject_zero_literal_q0_fock_env()
    _assert_finite_q_shortcut_is_safe(run, tuple(pairs))
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))
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
    h_k = np.empty(n_pairs, dtype=int)
    p_plus_k = np.empty(n_pairs, dtype=int)
    wrap_plus = np.empty((n_pairs, 2), dtype=int)
    for index, pair in enumerate(ph_pairs):
        p_local[index], particle_k = orbitals.decode_global_index(pair.particle)
        h_local[index], hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, wrap = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if particle_k != expected_particle_k:
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        h_k[index] = hole_k
        p_plus_k[index] = particle_k
        wrap_plus[index] = wrap
        A[index, index] = orbitals.energies[p_local[index], particle_k] - orbitals.energies[h_local[index], hole_k]

    indices_by_hole_k = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    overlap_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.layer_overlaps.items()}
    kernel_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.fock_layer_coulomb.items()}
    missing_shifts: set[tuple[int, int]] = set()
    resolved_physical_shifts = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )

    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        hh_overlap = overlap_by_shift.get(g0)
        if hh_overlap is None:
            missing_shifts.add(g0)
            continue
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            p_t_k = int(p_plus_k[target_indices[0]])
            wrap_t = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_k]
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                p_s_k = int(p_plus_k[source_indices[0]])
                wrap_s = tuple(int(v) for v in wrap_plus[source_indices[0]])
                pp_shift = _add_shift(g0, _sub_shift(wrap_t, wrap_s))
                pp_overlap = overlap_by_shift.get(pp_shift)
                pp_kernel = kernel_by_shift.get(pp_shift)
                if pp_overlap is None or pp_kernel is None:
                    missing_shifts.add(pp_shift)
                    continue
                u_p_source = U[:, :, p_s_k]
                u_h_source = U[:, :, ks]
                kernel = np.asarray(pp_kernel[p_t_k, p_s_k], dtype=float)
                n_layer = int(pp_overlap.shape[0])
                pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hh = np.empty_like(pp)
                for layer in range(n_layer):
                    pp_full = u_p_target.conj().T @ pp_overlap[layer, :, p_t_k, :, p_s_k] @ u_p_source
                    hh_full = u_h_target.conj().T @ hh_overlap[layer, :, kt, :, ks] @ u_h_source
                    pp[layer] = pp_full[np.ix_(p_t, p_s)]
                    hh[layer] = hh_full[np.ix_(h_t, h_s)]
                A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    pp,
                    np.conj(hh),
                    optimize=True,
                )
    if missing_shifts and require_complete_umklapp:
        preview = sorted(missing_shifts)[:10]
        raise ValueError(
            f"Finite-q exchange assembly requires cached overlap shifts not present in this HF run: {preview}"
        )
    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)


def _assert_finite_q_intraflavor_pairs(pairs: tuple[ParticleHolePair, ...]) -> None:
    for pair in pairs:
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q intraflavor pairs must carry SpinValleyFlavor metadata")
        if particle.spin != hole.spin or particle.valley != hole.valley:
            raise ValueError("finite-q intraflavor full TDHF requires particle and hole in the same flavor")

def build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    beta: float = 1.0,
    structure_tolerance: float = 1.0e-6,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
    _build_partner: bool = True,
) -> TDHFMatrices:
    """Build full finite-q intraflavor TDHF matrices using paper Eq. D19.

    Pair labels are the X-sector operators ``d†_{k+q,p} d_{k,h}``.  The B block
    columns use the corresponding Y-sector partner ``d†_{k,h} d_{k-q,p}`` with
    the same base hole momentum ``k`` and local HF band labels.  At ``q=0`` this
    reduces exactly to the existing q=0 direct/exchange/B assembly.  For nonzero
    ``q``, the returned Liouvillian is the Eq. D19 partner block
    ``[[A(q), B(q)], [-B(-q)*, -A(-q)*]]``; correspondingly the reported B
    residual checks ``B(q)=B(-q)^T`` rather than the q=0-only ``B(q)=B(q)^T``.
    """

    _reject_zero_literal_q0_fock_env()
    ph_pairs = tuple(pairs)
    _assert_finite_q_intraflavor_pairs(ph_pairs)
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))

    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    if n_pairs == 0:
        L = assemble_tdhf_liouvillian(A, B)
        structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    p_local = np.empty(n_pairs, dtype=int)
    h_local = np.empty(n_pairs, dtype=int)
    h_k = np.empty(n_pairs, dtype=int)
    p_plus_k = np.empty(n_pairs, dtype=int)
    p_minus_k = np.empty(n_pairs, dtype=int)
    wrap_plus = np.empty((n_pairs, 2), dtype=int)
    wrap_minus = np.empty((n_pairs, 2), dtype=int)
    minus_shift = (-shift[0], -shift[1])
    for index, pair in enumerate(ph_pairs):
        p_local[index], particle_k = orbitals.decode_global_index(pair.particle)
        h_local[index], hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, plus_wrap = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if int(particle_k) != int(expected_particle_k):
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        minus_k, minus_wrap = _shift_k_index_with_wrap(hole_k, minus_shift, mesh_shape)
        if orbitals.occupied_mask[p_local[index], minus_k]:
            raise ValueError(
                "finite-q intraflavor Eq. D19 requires the Y-sector particle at k-q to be unoccupied; "
                f"local={p_local[index]} k_minus={minus_k} is occupied"
            )
        h_k[index] = int(hole_k)
        p_plus_k[index] = int(particle_k)
        p_minus_k[index] = int(minus_k)
        wrap_plus[index] = plus_wrap
        wrap_minus[index] = minus_wrap
        A[index, index] = orbitals.energies[p_local[index], particle_k] - orbitals.energies[h_local[index], hole_k]

    indices_by_hole_k = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    overlap_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.layer_overlaps.items()}
    kernel_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.fock_layer_coulomb.items()}
    resolved_physical_shifts = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )
    missing_shifts: set[tuple[int, int]] = set()

    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        hh_overlap = overlap_by_shift.get(g0)
        if hh_overlap is None:
            missing_shifts.add(g0)
            continue
        n_layer = int(hh_overlap.shape[0])

        # Direct A/B terms: physical transfer q + G.  The X form factor uses
        # k+q -> k, while the Y partner uses k -> k-q.
        plus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        minus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        direct_kernel_by_k: dict[int, np.ndarray] = {}
        for ik, indices in enumerate(indices_by_hole_k):
            if indices.size == 0:
                continue
            plus_key = _add_shift(g0, tuple(int(v) for v in wrap_plus[indices[0]]))
            minus_key = _sub_shift(g0, tuple(int(v) for v in wrap_minus[indices[0]]))
            plus_overlap = overlap_by_shift.get(plus_key)
            minus_overlap = overlap_by_shift.get(minus_key)
            plus_kernel = kernel_by_shift.get(plus_key)
            if plus_overlap is None or plus_kernel is None:
                missing_shifts.add(plus_key)
                continue
            if minus_overlap is None:
                missing_shifts.add(minus_key)
                continue
            p_plus = int(p_plus_k[indices[0]])
            p_minus = int(p_minus_k[indices[0]])
            u_h = U[:, :, ik]
            u_p_plus = U[:, :, p_plus]
            u_p_minus = U[:, :, p_minus]
            p_idx = p_local[indices]
            h_idx = h_local[indices]
            direct_kernel_by_k[int(ik)] = np.asarray(plus_kernel[p_plus, ik], dtype=float)
            for layer in range(n_layer):
                plus_full = u_p_plus.conj().T @ plus_overlap[layer, :, p_plus, :, ik] @ u_h
                minus_full = u_h.conj().T @ minus_overlap[layer, :, ik, :, p_minus] @ u_p_minus
                plus_direct[layer, indices] = plus_full[p_idx, h_idx]
                minus_direct[layer, indices] = minus_full[h_idx, p_idx]
        for ik, row_indices in enumerate(indices_by_hole_k):
            if row_indices.size == 0 or int(ik) not in direct_kernel_by_k:
                continue
            kernel = direct_kernel_by_k[int(ik)]
            A[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(plus_direct),
                optimize=True,
            )
            B[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(minus_direct),
                optimize=True,
            )

        # A-exchange: V[p(k+q), h'(k'), p'(k'+q), h(k)].
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            p_t_plus = int(p_plus_k[target_indices[0]])
            wrap_t_plus = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_plus]
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s_plus = int(p_plus_k[source_indices[0]])
                wrap_s_plus = tuple(int(v) for v in wrap_plus[source_indices[0]])
                pp_shift = _add_shift(g0, _sub_shift(wrap_t_plus, wrap_s_plus))
                pp_overlap = overlap_by_shift.get(pp_shift)
                pp_kernel = kernel_by_shift.get(pp_shift)
                if pp_overlap is None or pp_kernel is None:
                    missing_shifts.add(pp_shift)
                    continue
                u_p_source = U[:, :, p_s_plus]
                u_h_source = U[:, :, ks]
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                kernel = np.asarray(pp_kernel[p_t_plus, p_s_plus], dtype=float)
                pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hh = np.empty_like(pp)
                for layer in range(n_layer):
                    pp_full = u_p_target.conj().T @ pp_overlap[layer, :, p_t_plus, :, p_s_plus] @ u_p_source
                    hh_full = u_h_target.conj().T @ hh_overlap[layer, :, kt, :, ks] @ u_h_source
                    pp[layer] = pp_full[np.ix_(p_t, p_s)]
                    hh[layer] = hh_full[np.ix_(h_t, h_s)]
                A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    pp,
                    np.conj(hh),
                    optimize=True,
                )

        # B-exchange: V[p(k+q), p'(k'-q), h'(k'), h(k)].
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            p_t_plus = int(p_plus_k[target_indices[0]])
            wrap_t_plus = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_plus]
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            left_shift = _add_shift(g0, wrap_t_plus)
            left_overlap = overlap_by_shift.get(left_shift)
            left_kernel = kernel_by_shift.get(left_shift)
            if left_overlap is None or left_kernel is None:
                missing_shifts.add(left_shift)
                continue
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s_minus = int(p_minus_k[source_indices[0]])
                wrap_s_minus = tuple(int(v) for v in wrap_minus[source_indices[0]])
                right_shift = _sub_shift(g0, wrap_s_minus)
                right_overlap = overlap_by_shift.get(right_shift)
                if right_overlap is None:
                    missing_shifts.add(right_shift)
                    continue
                u_h_source = U[:, :, ks]
                u_p_minus_source = U[:, :, p_s_minus]
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                kernel = np.asarray(left_kernel[p_t_plus, ks], dtype=float)
                ph = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hp = np.empty_like(ph)
                for layer in range(n_layer):
                    ph_full = u_p_target.conj().T @ left_overlap[layer, :, p_t_plus, :, ks] @ u_h_source
                    hp_full = u_h_target.conj().T @ right_overlap[layer, :, kt, :, p_s_minus] @ u_p_minus_source
                    ph[layer] = ph_full[np.ix_(p_t, h_s)]
                    hp[layer] = hp_full[np.ix_(h_t, p_s)]
                B[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    ph,
                    np.conj(hp),
                    optimize=True,
                )

    if missing_shifts and require_complete_umklapp:
        preview = sorted(missing_shifts)[:10]
        raise ValueError(
            f"Finite-q intraflavor assembly requires cached overlap shifts not present in this HF run: {preview}"
        )

    if _build_partner and shift != (0, 0):
        minus_q_pairs_all = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, minus_shift)
        minus_q_pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(minus_q_pairs_all, "intraflavor")
        if len(minus_q_pairs) != n_pairs:
            raise ValueError(
                "finite-q intraflavor +q and -q pair spaces have different sizes: "
                f"{n_pairs} vs {len(minus_q_pairs)}"
            )
        for plus_pair, minus_pair in zip(ph_pairs, minus_q_pairs, strict=True):
            plus_p_local, _plus_p_k = orbitals.decode_global_index(plus_pair.particle)
            plus_h_local, plus_h_k = orbitals.decode_global_index(plus_pair.hole)
            minus_p_local, _minus_p_k = orbitals.decode_global_index(minus_pair.particle)
            minus_h_local, minus_h_k = orbitals.decode_global_index(minus_pair.hole)
            if (plus_p_local, plus_h_local, plus_h_k) != (minus_p_local, minus_h_local, minus_h_k):
                raise ValueError("finite-q intraflavor +q/-q pair order mismatch")
        minus_matrices = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            minus_q_pairs,
            minus_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
            require_complete_umklapp=require_complete_umklapp,
            physical_shifts=physical_shifts,
            _build_partner=False,
        )
        L = np.block(
            [
                [A, B],
                [-np.conj(minus_matrices.B), -np.conj(minus_matrices.A)],
            ]
        )
        a_residual = max(
            float(np.max(np.abs(A - np.conj(A.T)))) if A.size else 0.0,
            float(np.max(np.abs(minus_matrices.A - np.conj(minus_matrices.A.T)))) if minus_matrices.A.size else 0.0,
        )
        b_residual = float(np.max(np.abs(B - minus_matrices.B.T))) if B.size else 0.0
        structure = TDHFStructureResiduals(
            a_hermitian=a_residual,
            b_symmetric=b_residual,
            particle_hole_symmetry=0.0,
            tolerance=float(structure_tolerance),
        )
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

def _filter_rlg_hbn_tdhf_finite_q_pairs(
    all_pairs: Sequence[ParticleHolePair],
    channel: str,
) -> tuple[ParticleHolePair, ...]:
    ph_pairs = tuple(all_pairs)
    groups: dict[str, list[int]] = {"intraflavor": [], "intervalley": [], "interspin": [], "inter_spin_valley": []}
    for index, pair in enumerate(ph_pairs):
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q pairs must carry SpinValleyFlavor metadata")
        same_spin = particle.spin == hole.spin
        same_valley = particle.valley == hole.valley
        if same_spin and same_valley:
            groups["intraflavor"].append(index)
        elif same_spin and not same_valley:
            groups["intervalley"].append(index)
        elif not same_spin and same_valley:
            groups["interspin"].append(index)
        elif not same_spin and not same_valley:
            groups["inter_spin_valley"].append(index)
    if channel not in groups:
        raise ValueError(f"finite-q channel must be one of {tuple(groups)}, got {channel!r}")
    return tuple(ph_pairs[index] for index in groups[str(channel)])

def _filter_rlg_hbn_tdhf_finite_q_shortcut_pairs(
    all_pairs: Sequence[ParticleHolePair],
    channel: str,
) -> tuple[ParticleHolePair, ...]:
    ph_pairs = tuple(all_pairs)
    if channel not in {"intervalley", "interspin", "inter_spin_valley"}:
        raise ValueError(f"finite-q shortcut channel must be a flavor-flip channel, got {channel!r}")
    groups: dict[str, list[int]] = {"intervalley": [], "interspin": [], "inter_spin_valley": []}
    for index, pair in enumerate(ph_pairs):
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q pairs must carry SpinValleyFlavor metadata")
        same_spin = particle.spin == hole.spin
        same_valley = particle.valley == hole.valley
        if same_spin and not same_valley:
            groups["intervalley"].append(index)
        elif not same_spin and same_valley:
            groups["interspin"].append(index)
        elif not same_spin and not same_valley:
            groups["inter_spin_valley"].append(index)
    return tuple(ph_pairs[index] for index in groups[str(channel)])

__all__ = [name for name in globals() if not name.startswith('__')]
