from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ._hf_types import RLGhBNHartreeFockRun
from ._tdhf_pairs import (
    _add_shift,
    _mesh_shape_from_k_grid_frac,
    _shift_k_index_with_wrap,
    _sub_shift,
)
from ._tdhf_types import RLGhBNTDHFOrbitals


TDHFFiniteQTerms = dict[str, np.ndarray]


def _wigner_seitz_wraps(
    lattice: object,
    fractional_delta: np.ndarray,
    *,
    tie_atol: float = 1.0e-12,
) -> tuple[tuple[tuple[int, int], float], ...]:
    delta = np.asarray(fractional_delta, dtype=float).reshape(2)
    nearest = np.rint(delta).astype(int)
    candidates: list[tuple[float, tuple[int, int]]] = []
    for dm in range(int(nearest[0]) - 2, int(nearest[0]) + 3):
        for dn in range(int(nearest[1]) - 2, int(nearest[1]) + 3):
            wrap = (int(dm), int(dn))
            residual = delta - np.asarray(wrap, dtype=float)
            reciprocal = float(residual[0]) * lattice.g_m1 + float(residual[1]) * lattice.g_m2
            candidates.append((float(abs(reciprocal)), wrap))
    best = min(value for value, _ in candidates)
    wraps = sorted(
        {wrap for value, wrap in candidates if abs(value - best) <= float(tie_atol)}
    )
    weight = 1.0 / float(len(wraps))
    return tuple((wrap, weight) for wrap in wraps)


def _decode_pair_arrays(
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    mesh_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ph_pairs = tuple(pairs)
    n_pairs = len(ph_pairs)
    p_local = np.empty(n_pairs, dtype=int)
    h_local = np.empty(n_pairs, dtype=int)
    h_k = np.empty(n_pairs, dtype=int)
    p_plus_k = np.empty(n_pairs, dtype=int)
    p_minus_k = np.empty(n_pairs, dtype=int)
    wrap_plus = np.empty((n_pairs, 2), dtype=int)
    wrap_minus = np.empty((n_pairs, 2), dtype=int)
    minus_shift = (-int(q_shift[0]), -int(q_shift[1]))
    for index, pair in enumerate(ph_pairs):
        p_local[index], p_plus_k[index] = orbitals.decode_global_index(pair.particle)
        h_local[index], h_k[index] = orbitals.decode_global_index(pair.hole)
        expected, plus_wrap = _shift_k_index_with_wrap(
            int(h_k[index]),
            q_shift,
            mesh_shape,
        )
        if int(expected) != int(p_plus_k[index]):
            raise ValueError(
                "finite-q pair particle momentum mismatch: "
                f"pair={index}, actual={p_plus_k[index]}, expected={expected}, q={q_shift}"
            )
        p_minus_k[index], minus_wrap = _shift_k_index_with_wrap(
            int(h_k[index]),
            minus_shift,
            mesh_shape,
        )
        if orbitals.occupied_mask[p_local[index], p_minus_k[index]]:
            raise ValueError(
                "finite-q D19 requires the Y-sector particle at k-q to be unoccupied: "
                f"pair={index}, local={p_local[index]}, k_minus={p_minus_k[index]}"
            )
        wrap_plus[index] = plus_wrap
        wrap_minus[index] = minus_wrap
    return p_local, h_local, h_k, p_plus_k, p_minus_k, wrap_plus, wrap_minus


def required_rlg_hbn_tdhf_ws_overlap_shifts(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs_by_shift: Mapping[tuple[int, int], Sequence[object]],
    physical_shifts: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return direct and actual-node WS exchange overlap keys for q sectors."""

    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    frac = np.asarray(run.basis_data.k_grid_frac, dtype=float)
    lattice = run.basis_data.basis_model.lattice
    nx, ny = mesh_shape
    resolved_physical = tuple((int(g[0]), int(g[1])) for g in physical_shifts)
    keys: set[tuple[int, int]] = set(resolved_physical)
    for q_shift, pairs in pairs_by_shift.items():
        (
            _p_local,
            _h_local,
            h_k,
            p_plus_k,
            _p_minus_k,
            wrap_plus,
            wrap_minus,
        ) = _decode_pair_arrays(orbitals, pairs, q_shift, mesh_shape)
        indices_by_hole = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
        q_frac = np.asarray(
            [float(q_shift[0]) / float(nx), float(q_shift[1]) / float(ny)],
            dtype=float,
        )
        for g0 in resolved_physical:
            for index in range(len(tuple(pairs))):
                keys.add(_add_shift(g0, tuple(int(x) for x in wrap_plus[index])))
                keys.add(_sub_shift(g0, tuple(int(x) for x in wrap_minus[index])))
            for kt, target_indices in enumerate(indices_by_hole):
                if target_indices.size == 0:
                    continue
                wrap_t_plus = tuple(int(x) for x in wrap_plus[target_indices[0]])
                for ks, source_indices in enumerate(indices_by_hole):
                    if source_indices.size == 0:
                        continue
                    wrap_s_plus = tuple(int(x) for x in wrap_plus[source_indices[0]])
                    wrap_s_minus = tuple(int(x) for x in wrap_minus[source_indices[0]])
                    for wrap, _weight in _wigner_seitz_wraps(lattice, frac[kt] - frac[ks]):
                        base_shift = _sub_shift(g0, wrap)
                        keys.add(base_shift)
                        keys.add(_add_shift(base_shift, _sub_shift(wrap_t_plus, wrap_s_plus)))
                    for wrap, _weight in _wigner_seitz_wraps(
                        lattice,
                        frac[kt] + q_frac - frac[ks],
                    ):
                        base_shift = _sub_shift(g0, wrap)
                        keys.add(_add_shift(base_shift, wrap_t_plus))
                        keys.add(_sub_shift(base_shift, wrap_s_minus))
    return tuple(sorted(keys))


def build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    *,
    beta: float = 1.0,
    physical_shifts: Iterable[tuple[int, int]] | None = None,
    require_complete_umklapp: bool = True,
) -> TDHFFiniteQTerms:
    """Build the five Eq. D19 terms with actual-node WS exchange folding."""

    ph_pairs = tuple(pairs)
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    shift = (int(q_shift[0]), int(q_shift[1]))
    n_pairs = len(ph_pairs)
    terms = {
        name: np.zeros((n_pairs, n_pairs), dtype=np.complex128)
        for name in ("A0", "A_direct", "A_exchange", "B_direct", "B_exchange")
    }
    if n_pairs == 0:
        return terms
    (
        p_local,
        h_local,
        h_k,
        p_plus_k,
        p_minus_k,
        wrap_plus,
        wrap_minus,
    ) = _decode_pair_arrays(orbitals, ph_pairs, shift, mesh_shape)
    for index in range(n_pairs):
        terms["A0"][index, index] = (
            orbitals.energies[p_local[index], p_plus_k[index]]
            - orbitals.energies[h_local[index], h_k[index]]
        )

    indices_by_hole = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    eigenvectors = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    overlap_by_shift = {
        tuple(int(v) for v in key): value
        for key, value in run.overlap_blocks.layer_overlaps.items()
    }
    kernel_by_shift = {
        tuple(int(v) for v in key): value
        for key, value in run.overlap_blocks.fock_layer_coulomb.items()
    }
    resolved_physical = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )
    missing: set[tuple[int, int]] = set()

    # Direct terms retain the physical q+G transfer and endpoint wrap keys.
    for g0 in resolved_physical:
        sample = overlap_by_shift.get(g0)
        if sample is None:
            missing.add(g0)
            continue
        n_layer = int(sample.shape[0])
        plus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        minus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        direct_kernel_by_k: dict[int, np.ndarray] = {}
        for hole_index, indices in enumerate(indices_by_hole):
            if indices.size == 0:
                continue
            plus_key = _add_shift(g0, tuple(int(x) for x in wrap_plus[indices[0]]))
            minus_key = _sub_shift(g0, tuple(int(x) for x in wrap_minus[indices[0]]))
            plus_overlap = overlap_by_shift.get(plus_key)
            minus_overlap = overlap_by_shift.get(minus_key)
            plus_kernel = kernel_by_shift.get(plus_key)
            if plus_overlap is None or plus_kernel is None:
                missing.add(plus_key)
                continue
            if minus_overlap is None:
                missing.add(minus_key)
                continue
            p_plus = int(p_plus_k[indices[0]])
            p_minus = int(p_minus_k[indices[0]])
            u_h = eigenvectors[:, :, hole_index]
            u_p_plus = eigenvectors[:, :, p_plus]
            u_p_minus = eigenvectors[:, :, p_minus]
            p_indices = p_local[indices]
            h_indices = h_local[indices]
            direct_kernel_by_k[hole_index] = np.asarray(
                plus_kernel[p_plus, hole_index],
                dtype=float,
            )
            for layer in range(n_layer):
                plus_full = (
                    u_p_plus.conj().T
                    @ plus_overlap[layer, :, p_plus, :, hole_index]
                    @ u_h
                )
                minus_full = (
                    u_h.conj().T
                    @ minus_overlap[layer, :, hole_index, :, p_minus]
                    @ u_p_minus
                )
                plus_direct[layer, indices] = plus_full[p_indices, h_indices]
                minus_direct[layer, indices] = minus_full[h_indices, p_indices]
        for hole_index, row_indices in enumerate(indices_by_hole):
            if row_indices.size == 0 or hole_index not in direct_kernel_by_k:
                continue
            kernel = direct_kernel_by_k[hole_index]
            terms["A_direct"][np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(plus_direct),
                optimize=True,
            )
            terms["B_direct"][np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(minus_direct),
                optimize=True,
            )

    # Both exchange terms use their actual expanded-node transfer, WS folded
    # before adding each physical G. Exact boundary ties are averaged.
    frac = np.asarray(run.basis_data.k_grid_frac, dtype=float)
    lattice = run.basis_data.basis_model.lattice
    nx, ny = mesh_shape
    q_frac = np.asarray([float(shift[0]) / float(nx), float(shift[1]) / float(ny)])
    for g0 in resolved_physical:
        for kt, target_indices in enumerate(indices_by_hole):
            if target_indices.size == 0:
                continue
            p_t_plus = int(p_plus_k[target_indices[0]])
            wrap_t_plus = tuple(int(x) for x in wrap_plus[target_indices[0]])
            u_p_target = eigenvectors[:, :, p_t_plus]
            u_h_target = eigenvectors[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            for ks, source_indices in enumerate(indices_by_hole):
                if source_indices.size == 0:
                    continue
                p_s_plus = int(p_plus_k[source_indices[0]])
                p_s_minus = int(p_minus_k[source_indices[0]])
                wrap_s_plus = tuple(int(x) for x in wrap_plus[source_indices[0]])
                wrap_s_minus = tuple(int(x) for x in wrap_minus[source_indices[0]])
                u_p_source = eigenvectors[:, :, p_s_plus]
                u_h_source = eigenvectors[:, :, ks]
                u_p_minus_source = eigenvectors[:, :, p_s_minus]
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]

                for wrap, weight in _wigner_seitz_wraps(lattice, frac[kt] - frac[ks]):
                    base_shift = _sub_shift(g0, wrap)
                    hh_overlap = overlap_by_shift.get(base_shift)
                    pp_shift = _add_shift(
                        base_shift,
                        _sub_shift(wrap_t_plus, wrap_s_plus),
                    )
                    pp_overlap = overlap_by_shift.get(pp_shift)
                    pp_kernel = kernel_by_shift.get(pp_shift)
                    if hh_overlap is None:
                        missing.add(base_shift)
                        continue
                    if pp_overlap is None or pp_kernel is None:
                        missing.add(pp_shift)
                        continue
                    n_layer = int(hh_overlap.shape[0])
                    kernel = np.asarray(pp_kernel[p_t_plus, p_s_plus], dtype=float)
                    pp = np.empty(
                        (n_layer, target_indices.size, source_indices.size),
                        dtype=np.complex128,
                    )
                    hh = np.empty_like(pp)
                    for layer in range(n_layer):
                        pp_full = (
                            u_p_target.conj().T
                            @ pp_overlap[layer, :, p_t_plus, :, p_s_plus]
                            @ u_p_source
                        )
                        hh_full = (
                            u_h_target.conj().T
                            @ hh_overlap[layer, :, kt, :, ks]
                            @ u_h_source
                        )
                        pp[layer] = pp_full[np.ix_(p_t, p_s)]
                        hh[layer] = hh_full[np.ix_(h_t, h_s)]
                    terms["A_exchange"][np.ix_(target_indices, source_indices)] -= (
                        scale
                        * float(weight)
                        * np.einsum(
                            "lm,lij,mij->ij",
                            kernel,
                            pp,
                            np.conj(hh),
                            optimize=True,
                        )
                    )

                for wrap, weight in _wigner_seitz_wraps(
                    lattice,
                    frac[kt] + q_frac - frac[ks],
                ):
                    base_shift = _sub_shift(g0, wrap)
                    left_shift = _add_shift(base_shift, wrap_t_plus)
                    right_shift = _sub_shift(base_shift, wrap_s_minus)
                    left_overlap = overlap_by_shift.get(left_shift)
                    right_overlap = overlap_by_shift.get(right_shift)
                    left_kernel = kernel_by_shift.get(left_shift)
                    if left_overlap is None or left_kernel is None:
                        missing.add(left_shift)
                        continue
                    if right_overlap is None:
                        missing.add(right_shift)
                        continue
                    n_layer = int(left_overlap.shape[0])
                    kernel = np.asarray(left_kernel[p_t_plus, ks], dtype=float)
                    ph = np.empty(
                        (n_layer, target_indices.size, source_indices.size),
                        dtype=np.complex128,
                    )
                    hp = np.empty_like(ph)
                    for layer in range(n_layer):
                        ph_full = (
                            u_p_target.conj().T
                            @ left_overlap[layer, :, p_t_plus, :, ks]
                            @ u_h_source
                        )
                        hp_full = (
                            u_h_target.conj().T
                            @ right_overlap[layer, :, kt, :, p_s_minus]
                            @ u_p_minus_source
                        )
                        ph[layer] = ph_full[np.ix_(p_t, h_s)]
                        hp[layer] = hp_full[np.ix_(h_t, p_s)]
                    terms["B_exchange"][np.ix_(target_indices, source_indices)] -= (
                        scale
                        * float(weight)
                        * np.einsum(
                            "lm,lij,mij->ij",
                            kernel,
                            ph,
                            np.conj(hp),
                            optimize=True,
                        )
                    )

    if missing and require_complete_umklapp:
        raise ValueError(
            "finite-q WS term assembly requires missing overlap shifts: "
            f"{sorted(missing)[:20]}"
        )
    return terms


def sum_finite_q_terms(
    terms: Mapping[str, np.ndarray],
    names: Iterable[str],
) -> np.ndarray:
    selected = tuple(names)
    if not selected:
        first = next(iter(terms.values()))
        return np.zeros_like(first)
    result = np.zeros_like(terms[selected[0]])
    for name in selected:
        result += terms[name]
    return result
