from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._hf_basis import (
    _build_projected_basis_for_indices,
    _c3_fixed_representative_shift_orbit,
    _c3_mesh_pair_and_representative_shift,
    _c3_reciprocal_shift,
    _c3_transform_raw_components,
    _k_index_to_pair,
    _pair_to_k_index,
    _raw_overlap_shift_for_physical_g,
    _raw_pair_from_canonical_pair,
    _rlg_hbn_layer_local_indices,
)
from ._hf_shared import shift_wavefunction_grid
from ._hf_types import RLGhBNHartreeFockRun, RLGhBNProjectedBasisData
from ._tdhf_pairs import _mesh_shape_from_k_grid_frac, _shift_k_index_with_wrap
from ._tdhf_types import RLGhBNTDHFOrbitals
from .interaction import layer_coulomb_matrix_mev_nm2


ComplexEvaluator = Callable[[int, int], complex]


@dataclass(frozen=True)
class RLGhBNTDHFFixedTermEvaluators:
    """Canonical microscopic evaluators used by the fixed quotient transport.

    Each callback must evaluate one interaction entry directly from microscopic
    states/form factors.  Passing a preassembled source matrix through a lookup
    callback violates this provider contract.
    """

    a_direct: ComplexEvaluator
    a_exchange: ComplexEvaluator
    b_direct: ComplexEvaluator
    b_exchange: ComplexEvaluator


@dataclass(frozen=True)
class RLGhBNTDHFFixedTransportResult:
    terms: dict[str, np.ndarray]
    touched_a_entries: int
    touched_b_entries: int
    max_left_support: int
    max_right_support: int
    partner_terms: dict[str, np.ndarray] | None = None


def c3_reciprocal_index(shift: tuple[int, int]) -> tuple[int, int]:
    """Apply the RLG/hBN reciprocal C3 map ``(m,n)->(-n,m-n)``."""

    m, n = (int(shift[0]), int(shift[1]))
    return (-n, m - n)


def c3_repeated_zone_offset(
    source_shift: tuple[int, int],
    target_shift: tuple[int, int],
    mesh_size: int,
) -> tuple[int, int]:
    """Return the reciprocal offset between a repeated-zone target and C3 source.

    The result ``R`` is defined by ``target = C3(source) + mesh_size * R``.
    A nonintegral result means that the two shifts are not in the same C3 orbit.
    """

    mesh = int(mesh_size)
    if mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    rotated = c3_reciprocal_index(source_shift)
    delta = (int(target_shift[0]) - rotated[0], int(target_shift[1]) - rotated[1])
    if delta[0] % mesh != 0 or delta[1] % mesh != 0:
        raise ValueError(
            "target shift is not a repeated-zone representative of C3(source): "
            f"source={source_shift}, target={target_shift}, C3(source)={rotated}, mesh={mesh}"
        )
    return (delta[0] // mesh, delta[1] // mesh)


def c3_direct_physical_shell(
    physical_shifts: Sequence[tuple[int, int]],
    *,
    repeated_zone_offset: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Map a direct-term G shell through one repeated-zone C3 edge.

    For ``q_target = C3(q_source) + mesh * R``, each microscopic shell label
    transforms as ``G_target = C3(G_source) - R``. A complete canonical
    cutoff shell is C3 invariant as a set, but retaining the explicit C3 map
    is essential when this helper is composed across more than one edge.
    Exchange terms do not use this helper; they require endpoint-resolved WS
    folding.
    """

    offset = (int(repeated_zone_offset[0]), int(repeated_zone_offset[1]))
    return tuple(
        (
            c3_reciprocal_index((int(g[0]), int(g[1])))[0] - offset[0],
            c3_reciprocal_index((int(g[0]), int(g[1])))[1] - offset[1],
        )
        for g in physical_shifts
    )


def c3_composed_direct_physical_shell(
    physical_shifts: Sequence[tuple[int, int]],
    *,
    repeated_zone_offsets: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Transport a direct shell recursively through repeated-zone C3 edges."""

    shell = tuple((int(g[0]), int(g[1])) for g in physical_shifts)
    for offset in repeated_zone_offsets:
        shell = c3_direct_physical_shell(
            shell,
            repeated_zone_offset=(int(offset[0]), int(offset[1])),
        )
    return shell


def physical_minus_particle_hole(
    *,
    bra: object,
    ket: object,
) -> tuple[object, object]:
    """Convert Y/minus operator order into physical particle/hole order.

    ``endpoint_nodes(role="minus")`` represents ``d†_h d_{p(k-q)}``, so its
    bra is the hole and its ket is the particle.  The returned tuple is always
    ``(particle, hole)``.
    """

    return ket, bra


def _validated_inputs(
    base_terms: Mapping[str, np.ndarray],
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    target_x_fixed: np.ndarray,
    target_y_fixed: np.ndarray,
) -> tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("A_direct", "A_exchange", "B_direct", "B_exchange")
    missing = [name for name in required if name not in base_terms]
    if missing:
        raise KeyError(f"missing fixed-transport terms: {missing}")
    terms = {name: np.array(base_terms[name], dtype=np.complex128, copy=True) for name in required}
    shape = terms[required[0]].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"term matrices must be square, got {shape}")
    n = int(shape[0])
    for name, value in terms.items():
        if value.shape != (n, n):
            raise ValueError(f"{name} has shape {value.shape}, expected {(n, n)}")
    left = np.asarray(left_transform, dtype=np.complex128)
    right = np.asarray(right_transform, dtype=np.complex128)
    if left.shape[0] != n or right.shape[0] != n:
        raise ValueError(
            f"target transform rows must equal matrix size {n}, got left={left.shape}, right={right.shape}"
        )
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"left/right source dimensions differ: {left.shape} vs {right.shape}")
    x_fixed = np.asarray(target_x_fixed, dtype=bool)
    y_fixed = np.asarray(target_y_fixed, dtype=bool)
    if x_fixed.shape != (n,) or y_fixed.shape != (n,):
        raise ValueError(f"fixed masks must have shape {(n,)}, got X={x_fixed.shape}, Y={y_fixed.shape}")
    return n, terms, left, right, x_fixed, y_fixed


def _supports(transform: np.ndarray, tolerance: float) -> list[np.ndarray]:
    supports = [np.flatnonzero(np.abs(transform[row]) > tolerance) for row in range(transform.shape[0])]
    empty = [row for row, support in enumerate(supports) if support.size == 0]
    if empty:
        raise ValueError(f"sewing transform has empty target-row support: {empty[:8]}")
    return supports


def transport_fixed_terms_from_canonical_form_factors(
    base_target_terms: Mapping[str, np.ndarray],
    *,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    target_x_fixed: np.ndarray,
    target_y_fixed: np.ndarray,
    evaluators: RLGhBNTDHFFixedTermEvaluators,
    support_tolerance: float = 1.0e-12,
    partner_target_terms: Mapping[str, np.ndarray] | None = None,
) -> RLGhBNTDHFFixedTransportResult:
    """Assemble fixed target terms from canonical microscopic evaluators.

    This routine performs component-level quotient transport. It never accepts
    or rotates a preassembled canonical interaction matrix. A terms use
    ``E A E†`` component weights, while B terms use ``E_q B E_-q^T`` weights.
    Only entries selected by the role-specific masks are replaced.
    """

    tolerance = float(support_tolerance)
    if tolerance < 0.0:
        raise ValueError(f"support_tolerance must be nonnegative, got {tolerance}")
    n, terms, left, right, x_fixed, y_fixed = _validated_inputs(
        base_target_terms,
        left_transform,
        right_transform,
        target_x_fixed,
        target_y_fixed,
    )
    partner_terms = None
    if partner_target_terms is not None:
        partner_terms = {
            name: np.array(partner_target_terms[name], dtype=np.complex128, copy=True)
            for name in ("A_direct", "A_exchange", "B_direct", "B_exchange")
        }
        if any(value.shape != (n, n) for value in partner_terms.values()):
            raise ValueError("partner target term matrices must match the target matrix shape")
    left_support = _supports(left, tolerance)
    right_support = _supports(right, tolerance)
    touched_a = 0
    touched_b = 0
    for i in range(n):
        for j in range(n):
            if bool(x_fixed[i]) or bool(x_fixed[j]):
                touched_a += 1
                a_direct = 0.0 + 0.0j
                a_exchange = 0.0 + 0.0j
                for source_i_raw in left_support[i]:
                    source_i = int(source_i_raw)
                    left_weight = complex(left[i, source_i])
                    for source_j_raw in left_support[j]:
                        source_j = int(source_j_raw)
                        weight = left_weight * np.conj(complex(left[j, source_j]))
                        a_direct += weight * complex(evaluators.a_direct(source_i, source_j))
                        a_exchange += weight * complex(evaluators.a_exchange(source_i, source_j))
                terms["A_direct"][i, j] = a_direct
                terms["A_exchange"][i, j] = a_exchange
            if bool(x_fixed[i]) or bool(y_fixed[j]):
                touched_b += 1
                b_direct = 0.0 + 0.0j
                b_exchange = 0.0 + 0.0j
                for source_i_raw in left_support[i]:
                    source_i = int(source_i_raw)
                    left_weight = complex(left[i, source_i])
                    for source_j_raw in right_support[j]:
                        source_j = int(source_j_raw)
                        weight = left_weight * complex(right[j, source_j])
                        b_direct += weight * complex(evaluators.b_direct(source_i, source_j))
                        b_exchange += weight * complex(evaluators.b_exchange(source_i, source_j))
                terms["B_direct"][i, j] = b_direct
                terms["B_exchange"][i, j] = b_exchange
                if partner_terms is not None:
                    partner_terms["B_direct"][j, i] = b_direct
                    partner_terms["B_exchange"][j, i] = b_exchange
    return RLGhBNTDHFFixedTransportResult(
        terms=terms,
        touched_a_entries=touched_a,
        touched_b_entries=touched_b,
        max_left_support=max(int(value.size) for value in left_support),
        max_right_support=max(int(value.size) for value in right_support),
        partner_terms=partner_terms,
    )


def populate_shared_b_partner_entries(
    q_direct: np.ndarray,
    q_exchange: np.ndarray,
    minus_q_direct: np.ndarray,
    minus_q_exchange: np.ndarray,
    *,
    q_x_fixed: np.ndarray,
    q_y_fixed: np.ndarray,
    direct_evaluator: ComplexEvaluator,
    exchange_evaluator: ComplexEvaluator,
) -> int:
    """Populate q/-q fixed B entries from one canonical microscopic cache path.

    Each canonical ``(i,j)`` evaluator is called once. The resulting provider
    scalar is exposed as both ``B(q)[i,j]`` and ``B(-q)[j,i]`` as required by
    D19. This happens during provider assembly, not as matrix symmetrization.
    """

    qd = np.asarray(q_direct)
    qx = np.asarray(q_exchange)
    md = np.asarray(minus_q_direct)
    mx = np.asarray(minus_q_exchange)
    if qd.ndim != 2 or qd.shape[0] != qd.shape[1]:
        raise ValueError(f"B matrices must be square, got {qd.shape}")
    shape = qd.shape
    if any(value.shape != shape for value in (qx, md, mx)):
        raise ValueError("all q/-q B matrices must have the same square shape")
    n = int(shape[0])
    x_fixed = np.asarray(q_x_fixed, dtype=bool)
    y_fixed = np.asarray(q_y_fixed, dtype=bool)
    if x_fixed.shape != (n,) or y_fixed.shape != (n,):
        raise ValueError(f"fixed masks must have shape {(n,)}")
    count = 0
    for i in range(n):
        for j in range(n):
            if not (bool(x_fixed[i]) or bool(y_fixed[j])):
                continue
            direct = complex(direct_evaluator(i, j))
            exchange = complex(exchange_evaluator(i, j))
            qd[i, j] = direct
            qx[i, j] = exchange
            md[j, i] = direct
            mx[j, i] = exchange
            count += 1
    return count


@dataclass(frozen=True)
class RLGhBNTDHFExpandedNodeKey:
    stored_k: int
    wrap: tuple[int, int]
    reciprocal_shift: tuple[int, int]


@dataclass(frozen=True)
class RLGhBNTDHFDecodedPairGeometry:
    p_local: np.ndarray
    h_local: np.ndarray
    h_k: np.ndarray
    p_plus_k: np.ndarray
    p_minus_k: np.ndarray
    wrap_plus: np.ndarray
    wrap_minus: np.ndarray


class RLGhBNTDHFExpandedNodeBuilder:
    """Request only expanded representative nodes touched by one q sector."""

    def __init__(self, run: RLGhBNHartreeFockRun) -> None:
        self.run = run
        self.base = run.basis_data
        self.mesh_size = int(self.base.mesh_size)
        self.fixed_pairs = {
            (int(pair[0]), int(pair[1]))
            for pair in self.base.c3_fixed_representative_pairs
        }
        self.node_index: dict[RLGhBNTDHFExpandedNodeKey, int] = {}
        self.node_keys: list[RLGhBNTDHFExpandedNodeKey] = []

    def representative_shifts(
        self,
        stored_k: int,
        wrap: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        pair = _k_index_to_pair(int(stored_k), self.mesh_size)
        if pair in self.fixed_pairs:
            c3_pair, representative_shift = _c3_mesh_pair_and_representative_shift(
                pair,
                self.mesh_size,
            )
            if c3_pair != pair:
                raise RuntimeError(f"fixed pair bookkeeping error: {pair} maps to {c3_pair}")
            representatives = _c3_fixed_representative_shift_orbit(representative_shift)
        else:
            if self.base.periodic_reciprocal_shifts is None:
                raise RuntimeError("projected basis has no periodic_reciprocal_shifts")
            representatives = (
                tuple(int(x) for x in self.base.periodic_reciprocal_shifts[int(stored_k)]),
            )
        return tuple(
            (int(rep[0]) + int(wrap[0]), int(rep[1]) + int(wrap[1]))
            for rep in representatives
        )

    def nodes_for_leg(self, stored_k: int, wrap: tuple[int, int]) -> tuple[int, ...]:
        normalized_wrap = (int(wrap[0]), int(wrap[1]))
        nodes: list[int] = []
        for reciprocal_shift in self.representative_shifts(stored_k, normalized_wrap):
            key = RLGhBNTDHFExpandedNodeKey(
                stored_k=int(stored_k),
                wrap=normalized_wrap,
                reciprocal_shift=(int(reciprocal_shift[0]), int(reciprocal_shift[1])),
            )
            if key not in self.node_index:
                self.node_index[key] = len(self.node_keys)
                self.node_keys.append(key)
            nodes.append(self.node_index[key])
        return tuple(nodes)


def decode_finite_q_pair_geometry(
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    mesh_shape: tuple[int, int],
) -> RLGhBNTDHFDecodedPairGeometry:
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
        wrap_plus[index] = plus_wrap
        wrap_minus[index] = minus_wrap
    return RLGhBNTDHFDecodedPairGeometry(
        p_local=p_local,
        h_local=h_local,
        h_k=h_k,
        p_plus_k=p_plus_k,
        p_minus_k=p_minus_k,
        wrap_plus=wrap_plus,
        wrap_minus=wrap_minus,
    )


def required_expanded_periodic_gauge_padding(
    basis_data: RLGhBNProjectedBasisData,
    reciprocal_shifts: Sequence[tuple[int, int]],
) -> int:
    """Return the smallest embedding padding that contains all relabelled G support."""

    g_indices = np.asarray(basis_data.basis_model.lattice.g_indices, dtype=int)
    mins = np.min(g_indices, axis=0)
    maxs = np.max(g_indices, axis=0)
    required = 0
    for reciprocal_shift in reciprocal_shifts:
        shift = (int(reciprocal_shift[0]), int(reciprocal_shift[1]))
        for valley in basis_data.valleys:
            for pair in g_indices:
                raw = _raw_pair_from_canonical_pair(pair, shift, valley=int(valley))
                required = max(
                    required,
                    int(mins[0] - raw[0]),
                    int(mins[1] - raw[1]),
                    int(raw[0] - maxs[0]),
                    int(raw[1] - maxs[1]),
                )
    return max(0, int(required))


@dataclass(frozen=True)
class RLGhBNTDHFSparseFixedContext:
    geometry: RLGhBNTDHFDecodedPairGeometry
    builder: RLGhBNTDHFExpandedNodeBuilder
    basis_data: RLGhBNProjectedBasisData
    periodic_gauge_padding: int


def _sparse_context_geometry_and_builder(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
) -> tuple[RLGhBNTDHFDecodedPairGeometry, RLGhBNTDHFExpandedNodeBuilder]:
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    geometry = decode_finite_q_pair_geometry(orbitals, pairs, q_shift, mesh_shape)
    builder = RLGhBNTDHFExpandedNodeBuilder(run)
    for index in range(len(tuple(pairs))):
        builder.nodes_for_leg(int(geometry.h_k[index]), (0, 0))
        builder.nodes_for_leg(
            int(geometry.p_plus_k[index]),
            tuple(int(x) for x in geometry.wrap_plus[index]),
        )
        builder.nodes_for_leg(
            int(geometry.p_minus_k[index]),
            tuple(int(x) for x in geometry.wrap_minus[index]),
        )
    if not builder.node_keys:
        raise RuntimeError("no sparse expanded nodes requested")
    return geometry, builder


def required_sparse_fixed_context_padding(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
) -> int:
    """Return the exact local padding required by one finite-q source context."""

    _geometry, builder = _sparse_context_geometry_and_builder(
        run,
        orbitals,
        pairs,
        q_shift,
    )
    reciprocal_shifts = tuple(
        (int(key.reciprocal_shift[0]), int(key.reciprocal_shift[1]))
        for key in builder.node_keys
    )
    return required_expanded_periodic_gauge_padding(
        run.basis_data,
        reciprocal_shifts,
    )


def build_sparse_fixed_context(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    *,
    periodic_gauge_padding: int | None = None,
) -> RLGhBNTDHFSparseFixedContext:
    """Build the basis-only expanded context used by the sparse fixed provider."""

    geometry, builder = _sparse_context_geometry_and_builder(
        run,
        orbitals,
        pairs,
        q_shift,
    )

    base = run.basis_data
    lattice = base.basis_model.lattice
    kvec: list[complex] = []
    frac: list[np.ndarray] = []
    reciprocal_shifts: list[tuple[int, int]] = []
    for key in builder.node_keys:
        wrap_vec = int(key.wrap[0]) * lattice.g_m1 + int(key.wrap[1]) * lattice.g_m2
        kvec.append(complex(base.kvec[int(key.stored_k)] + wrap_vec))
        frac.append(
            np.asarray(base.k_grid_frac[int(key.stored_k)], dtype=float)
            + np.asarray(key.wrap, dtype=float)
        )
        reciprocal_shifts.append(
            (int(key.reciprocal_shift[0]), int(key.reciprocal_shift[1]))
        )
    minimum_padding = required_expanded_periodic_gauge_padding(base, reciprocal_shifts)
    resolved_padding = minimum_padding if periodic_gauge_padding is None else int(periodic_gauge_padding)
    if resolved_padding < minimum_padding:
        raise ValueError(
            "periodic_gauge_padding is too small for the requested expanded nodes: "
            f"requested={resolved_padding}, required={minimum_padding}, q={q_shift}"
        )
    basis_data = _build_projected_basis_for_indices(
        physical_model=base.model,
        basis_model=base.basis_model,
        interaction=base.interaction,
        kvec=np.asarray(kvec, dtype=np.complex128),
        band_indices=base.active_band_indices,
        valleys=base.valleys,
        mesh_size=base.mesh_size,
        k_grid_frac=np.asarray(frac, dtype=float),
        screening=base.screening,
        name="rlg_hbn_tdhf_sparse_fixed_nodes",
        build_h0=False,
        reciprocal_shifts=tuple(reciprocal_shifts),
        c3_fixed_representative_pairs=base.c3_fixed_representative_pairs,
        periodic_gauge_padding=resolved_padding,
    )
    return RLGhBNTDHFSparseFixedContext(
        geometry=geometry,
        builder=builder,
        basis_data=basis_data,
        periodic_gauge_padding=resolved_padding,
    )


def _local_to_spin_flavor_band(
    local_index: int,
    *,
    n_spin: int,
    n_flavor: int,
) -> tuple[int, int, int]:
    local = int(local_index)
    spin = local % int(n_spin)
    flavor = (local // int(n_spin)) % int(n_flavor)
    band = local // (int(n_spin) * int(n_flavor))
    return spin, flavor, band


def sparse_fixed_leg_vector(
    context: RLGhBNTDHFSparseFixedContext,
    orbitals: RLGhBNTDHFOrbitals,
    *,
    local_index: int,
    node: int,
) -> np.ndarray:
    key = context.builder.node_keys[int(node)]
    basis_data = context.basis_data
    n_spin = int(orbitals.n_spin)
    n_flavor = int(orbitals.n_eta)
    component_dim = int(basis_data.basis.wavefunctions.shape[0])
    coefficients = np.asarray(
        orbitals.eigenvectors[:, int(local_index), int(key.stored_k)],
        dtype=np.complex128,
    )
    result = np.zeros(n_spin * n_flavor * component_dim, dtype=np.complex128)
    for basis_local, coefficient in enumerate(coefficients):
        if coefficient == 0.0:
            continue
        spin, flavor, band = _local_to_spin_flavor_band(
            basis_local,
            n_spin=n_spin,
            n_flavor=n_flavor,
        )
        offset = (spin * n_flavor + flavor) * component_dim
        result[offset : offset + component_dim] += coefficient * np.asarray(
            basis_data.basis.wavefunctions[:, band, flavor, int(node)],
            dtype=np.complex128,
        )
    return result


def sparse_layer_form_factor(
    basis_data: RLGhBNProjectedBasisData,
    bra_vector: np.ndarray,
    ket_vector: np.ndarray,
    reciprocal_shift: tuple[int, int],
) -> np.ndarray:
    basis = basis_data.basis
    nx, ny = basis.grid_shape
    component_dim = int(basis.wavefunctions.shape[0])
    n_spin = int(basis.n_spin)
    n_flavor = int(basis.n_flavor)
    layer_count = int(basis_data.basis_model.params.layer_count)
    bra_blocks = np.asarray(bra_vector, dtype=np.complex128).reshape(
        n_spin * n_flavor,
        component_dim,
    )
    ket_blocks = np.asarray(ket_vector, dtype=np.complex128).reshape(
        n_spin * n_flavor,
        component_dim,
    )
    result = np.zeros(layer_count, dtype=np.complex128)
    for spin in range(n_spin):
        for flavor in range(n_flavor):
            block = spin * n_flavor + flavor
            valley = int(basis_data.valleys[flavor])
            raw_m, raw_n = _raw_overlap_shift_for_physical_g(
                reciprocal_shift,
                valley=valley,
            )
            bra_grid = bra_blocks[block].reshape(
                (basis.local_basis_size, nx, ny),
                order="F",
            )
            ket_grid = ket_blocks[block].reshape(
                (basis.local_basis_size, nx, ny),
                order="F",
            )
            shifted = shift_wavefunction_grid(
                ket_grid,
                -raw_m,
                -raw_n,
                boundary_mode="zero_fill",
                grid_axes=(1, 2),
            )
            for layer in range(layer_count):
                local_indices = _rlg_hbn_layer_local_indices(
                    basis,
                    layer,
                    layer_count=layer_count,
                )
                result[layer] += np.sum(
                    np.conj(bra_grid[local_indices]) * shifted[local_indices]
                )
    return result


def _choose_representative_node(nodes: tuple[int, ...], fixed_copy: int) -> int:
    if len(nodes) == 1:
        return int(nodes[0])
    if len(nodes) == 3:
        return int(nodes[int(fixed_copy) % 3])
    raise ValueError(f"unexpected expanded endpoint nodes: {nodes}")


def _endpoint_nodes(
    context: RLGhBNTDHFSparseFixedContext,
    *,
    role: str,
    pair_index: int,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    geometry = context.geometry
    builder = context.builder
    index = int(pair_index)
    if role == "plus":
        bra_nodes = builder.nodes_for_leg(
            int(geometry.p_plus_k[index]),
            tuple(int(x) for x in geometry.wrap_plus[index]),
        )
        ket_nodes = builder.nodes_for_leg(int(geometry.h_k[index]), (0, 0))
        return (
            int(geometry.p_local[index]),
            int(geometry.h_local[index]),
            bra_nodes,
            ket_nodes,
        )
    if role == "minus":
        bra_nodes = builder.nodes_for_leg(int(geometry.h_k[index]), (0, 0))
        ket_nodes = builder.nodes_for_leg(
            int(geometry.p_minus_k[index]),
            tuple(int(x) for x in geometry.wrap_minus[index]),
        )
        return (
            int(geometry.h_local[index]),
            int(geometry.p_local[index]),
            bra_nodes,
            ket_nodes,
        )
    raise ValueError(f"role must be 'plus' or 'minus', got {role!r}")


@dataclass(frozen=True)
class RLGhBNTDHFSparsePairState:
    p_tag: tuple[object, ...]
    h_tag: tuple[object, ...]
    p_vector: np.ndarray
    h_vector: np.ndarray
    p_node: int
    h_node: int


def build_sparse_role_pair_states(
    context: RLGhBNTDHFSparseFixedContext,
    orbitals: RLGhBNTDHFOrbitals,
    *,
    role: str,
    fixed_copy: int,
    vector_cache: dict[tuple[int, int], np.ndarray],
) -> tuple[RLGhBNTDHFSparsePairState, ...]:
    states: list[RLGhBNTDHFSparsePairState] = []
    pair_count = int(context.geometry.h_k.size)
    for index in range(pair_count):
        bra_local, ket_local, bra_nodes, ket_nodes = _endpoint_nodes(
            context,
            role=role,
            pair_index=index,
        )
        bra_node = _choose_representative_node(bra_nodes, fixed_copy)
        ket_node = _choose_representative_node(ket_nodes, fixed_copy)
        if role == "plus":
            p_local, p_node = bra_local, bra_node
            h_local, h_node = ket_local, ket_node
        else:
            p_local, p_node = ket_local, ket_node
            h_local, h_node = bra_local, bra_node
        p_key = (int(p_local), int(p_node))
        h_key = (int(h_local), int(h_node))
        if p_key not in vector_cache:
            vector_cache[p_key] = sparse_fixed_leg_vector(
                context,
                orbitals,
                local_index=p_key[0],
                node=p_key[1],
            )
        if h_key not in vector_cache:
            vector_cache[h_key] = sparse_fixed_leg_vector(
                context,
                orbitals,
                local_index=h_key[0],
                node=h_key[1],
            )
        states.append(
            RLGhBNTDHFSparsePairState(
                p_tag=(role, "p", p_key[0], p_key[1]),
                h_tag=(role, "h", h_key[0], h_key[1]),
                p_vector=vector_cache[p_key],
                h_vector=vector_cache[h_key],
                p_node=p_key[1],
                h_node=h_key[1],
            )
        )
    return tuple(states)


@dataclass(frozen=True)
class RLGhBNTDHFInteractionEndpoint:
    tag: tuple[object, ...]
    vector: np.ndarray
    node: int


def _pair_state_endpoint(
    state: RLGhBNTDHFSparsePairState,
    leg: str,
) -> RLGhBNTDHFInteractionEndpoint:
    if leg == "p":
        return RLGhBNTDHFInteractionEndpoint(state.p_tag, state.p_vector, state.p_node)
    if leg == "h":
        return RLGhBNTDHFInteractionEndpoint(state.h_tag, state.h_vector, state.h_node)
    raise ValueError(f"leg must be 'p' or 'h', got {leg!r}")


def _ws_wraps(
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


class RLGhBNTDHFSparseMicroscopicEvaluator:
    """On-demand fixed-leg interaction evaluator with actual-node WS exchange."""

    def __init__(
        self,
        run: RLGhBNHartreeFockRun,
        context: RLGhBNTDHFSparseFixedContext,
        physical_shifts: Sequence[tuple[int, int]],
        *,
        beta: float = 1.0,
        momentum_tolerance: float = 1.0e-10,
        transfer_tolerance: float = 1.0e-12,
    ) -> None:
        self.context = context
        self.physical_shifts = tuple(
            (int(shift[0]), int(shift[1])) for shift in physical_shifts
        )
        self.scale = (
            float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
        )
        self.basis_data = context.basis_data
        self.frac = np.asarray(self.basis_data.k_grid_frac, dtype=float)
        self.lattice = self.basis_data.basis_model.lattice
        self.layer_count = int(self.basis_data.basis_model.params.layer_count)
        self.layer_spacing_nm = float(
            self.basis_data.basis_model.params.layer_spacing_nm
        )
        self.momentum_tolerance = float(momentum_tolerance)
        self.transfer_tolerance = float(transfer_tolerance)
        self.form_factor_cache: dict[
            tuple[tuple[object, ...], tuple[object, ...], tuple[int, int]],
            np.ndarray,
        ] = {}
        self.kernel_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray] = {}
        self.ws_wrap_cache: dict[
            tuple[int, int],
            tuple[tuple[tuple[int, int], float], ...],
        ] = {}
        self.momentum_rejected_count = 0

    def form_factor(
        self,
        left: RLGhBNTDHFInteractionEndpoint,
        right: RLGhBNTDHFInteractionEndpoint,
        shift: tuple[int, int],
    ) -> np.ndarray:
        normalized_shift = (int(shift[0]), int(shift[1]))
        key = (left.tag, right.tag, normalized_shift)
        value = self.form_factor_cache.get(key)
        if value is None:
            value = sparse_layer_form_factor(
                self.basis_data,
                left.vector,
                right.vector,
                normalized_shift,
            )
            self.form_factor_cache[key] = value
        return value

    def kernel(
        self,
        left_node: int,
        right_node: int,
        shift: tuple[int, int],
    ) -> np.ndarray:
        key = (
            int(left_node),
            int(right_node),
            (int(shift[0]), int(shift[1])),
        )
        value = self.kernel_cache.get(key)
        if value is None:
            g_vector = key[2][0] * self.lattice.g_m1 + key[2][1] * self.lattice.g_m2
            q_value = complex(
                self.basis_data.kvec[key[0]]
                - self.basis_data.kvec[key[1]]
                + g_vector
            )
            value = layer_coulomb_matrix_mev_nm2(
                abs(q_value),
                self.layer_count,
                self.basis_data.interaction,
                layer_spacing_nm=self.layer_spacing_nm,
            )
            self.kernel_cache[key] = value
        return value

    def wraps(
        self,
        left_node: int,
        right_node: int,
    ) -> tuple[tuple[tuple[int, int], float], ...]:
        key = (int(left_node), int(right_node))
        value = self.ws_wrap_cache.get(key)
        if value is None:
            value = _ws_wraps(
                self.lattice,
                self.frac[key[0]] - self.frac[key[1]],
            )
            self.ws_wrap_cache[key] = value
        return value

    def generic(
        self,
        *,
        a: RLGhBNTDHFInteractionEndpoint,
        b: RLGhBNTDHFInteractionEndpoint,
        c: RLGhBNTDHFInteractionEndpoint,
        d: RLGhBNTDHFInteractionEndpoint,
        sign: complex = 1.0 + 0.0j,
    ) -> complex:
        momentum_residual = complex(
            self.basis_data.kvec[int(a.node)]
            + self.basis_data.kvec[int(b.node)]
            - self.basis_data.kvec[int(c.node)]
            - self.basis_data.kvec[int(d.node)]
        )
        if abs(momentum_residual) > self.momentum_tolerance:
            self.momentum_rejected_count += 1
            return 0.0 + 0.0j
        total = 0.0 + 0.0j
        for shift in self.physical_shifts:
            left = self.form_factor(a, c, shift)
            right = self.form_factor(d, b, shift)
            total += self.scale * np.einsum(
                "lm,l,m->",
                self.kernel(a.node, c.node, shift),
                left,
                np.conj(right),
                optimize=True,
            )
        return complex(sign * total)

    def ws_generic(
        self,
        *,
        a: RLGhBNTDHFInteractionEndpoint,
        b: RLGhBNTDHFInteractionEndpoint,
        c: RLGhBNTDHFInteractionEndpoint,
        d: RLGhBNTDHFInteractionEndpoint,
        sign: complex = 1.0 + 0.0j,
    ) -> complex:
        delta_left = self.frac[int(a.node)] - self.frac[int(c.node)]
        delta_right = self.frac[int(d.node)] - self.frac[int(b.node)]
        if float(np.max(np.abs(delta_left - delta_right))) > self.transfer_tolerance:
            raise RuntimeError(
                "expanded-node exchange transfer mismatch: "
                f"left={delta_left.tolist()}, right={delta_right.tolist()}"
            )
        total = 0.0 + 0.0j
        for physical_shift in self.physical_shifts:
            for wrap, weight in self.wraps(a.node, c.node):
                shift = (
                    int(physical_shift[0]) - int(wrap[0]),
                    int(physical_shift[1]) - int(wrap[1]),
                )
                left = self.form_factor(a, c, shift)
                right = self.form_factor(d, b, shift)
                total += self.scale * float(weight) * np.einsum(
                    "lm,l,m->",
                    self.kernel(a.node, c.node, shift),
                    left,
                    np.conj(right),
                    optimize=True,
                )
        return complex(sign * total)

    def a_direct(
        self,
        left_state: RLGhBNTDHFSparsePairState,
        right_state: RLGhBNTDHFSparsePairState,
    ) -> complex:
        return self.generic(
            a=_pair_state_endpoint(left_state, "p"),
            b=_pair_state_endpoint(right_state, "h"),
            c=_pair_state_endpoint(left_state, "h"),
            d=_pair_state_endpoint(right_state, "p"),
        )

    def a_exchange(
        self,
        left_state: RLGhBNTDHFSparsePairState,
        right_state: RLGhBNTDHFSparsePairState,
    ) -> complex:
        return self.ws_generic(
            a=_pair_state_endpoint(left_state, "p"),
            b=_pair_state_endpoint(right_state, "h"),
            c=_pair_state_endpoint(right_state, "p"),
            d=_pair_state_endpoint(left_state, "h"),
            sign=-1.0 + 0.0j,
        )

    def b_direct(
        self,
        plus_state: RLGhBNTDHFSparsePairState,
        minus_state: RLGhBNTDHFSparsePairState,
    ) -> complex:
        return self.generic(
            a=_pair_state_endpoint(plus_state, "p"),
            b=_pair_state_endpoint(minus_state, "p"),
            c=_pair_state_endpoint(plus_state, "h"),
            d=_pair_state_endpoint(minus_state, "h"),
        )

    def b_exchange(
        self,
        plus_state: RLGhBNTDHFSparsePairState,
        minus_state: RLGhBNTDHFSparsePairState,
    ) -> complex:
        return self.ws_generic(
            a=_pair_state_endpoint(plus_state, "p"),
            b=_pair_state_endpoint(minus_state, "p"),
            c=_pair_state_endpoint(minus_state, "h"),
            d=_pair_state_endpoint(plus_state, "h"),
            sign=-1.0 + 0.0j,
        )

    def metadata(self) -> dict[str, int]:
        return {
            "form_factor_cache_count": len(self.form_factor_cache),
            "kernel_cache_count": len(self.kernel_cache),
            "ws_wrap_cache_count": len(self.ws_wrap_cache),
            "momentum_rejected_count": int(self.momentum_rejected_count),
        }


@dataclass(frozen=True)
class RLGhBNTDHFSparseFixedSource:
    context: RLGhBNTDHFSparseFixedContext
    plus_states: tuple[RLGhBNTDHFSparsePairState, ...]
    minus_states: tuple[RLGhBNTDHFSparsePairState, ...]
    x_fixed: np.ndarray
    y_fixed: np.ndarray
    evaluator: RLGhBNTDHFSparseMicroscopicEvaluator
    term_evaluators: RLGhBNTDHFFixedTermEvaluators
    leg_vector_cache_count: int


def fixed_role_masks(
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    fixed_indices: set[int],
    mesh_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[bool] = []
    y_values: list[bool] = []
    mesh_shape = (int(mesh_size), int(mesh_size))
    minus_shift = (-int(q_shift[0]), -int(q_shift[1]))
    for pair in pairs:
        _, p_plus_k = orbitals.decode_global_index(pair.particle)
        _, h_k = orbitals.decode_global_index(pair.hole)
        p_minus_k, _ = _shift_k_index_with_wrap(int(h_k), minus_shift, mesh_shape)
        x_values.append(int(h_k) in fixed_indices or int(p_plus_k) in fixed_indices)
        y_values.append(int(h_k) in fixed_indices or int(p_minus_k) in fixed_indices)
    return np.asarray(x_values, dtype=bool), np.asarray(y_values, dtype=bool)


def build_sparse_fixed_source(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
    q_shift: tuple[int, int],
    physical_shifts: Sequence[tuple[int, int]],
    *,
    fixed_copy: int = 0,
    periodic_gauge_padding: int | None = None,
    beta: float = 1.0,
) -> RLGhBNTDHFSparseFixedSource:
    context = build_sparse_fixed_context(
        run,
        orbitals,
        pairs,
        q_shift,
        periodic_gauge_padding=periodic_gauge_padding,
    )
    vector_cache: dict[tuple[int, int], np.ndarray] = {}
    plus_states = build_sparse_role_pair_states(
        context,
        orbitals,
        role="plus",
        fixed_copy=fixed_copy,
        vector_cache=vector_cache,
    )
    minus_states = build_sparse_role_pair_states(
        context,
        orbitals,
        role="minus",
        fixed_copy=fixed_copy,
        vector_cache=vector_cache,
    )
    fixed_indices = {
        int(pair[0]) * int(run.basis_data.mesh_size) + int(pair[1])
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    x_fixed, y_fixed = fixed_role_masks(
        orbitals,
        pairs,
        q_shift,
        fixed_indices,
        int(run.basis_data.mesh_size),
    )
    evaluator = RLGhBNTDHFSparseMicroscopicEvaluator(
        run,
        context,
        physical_shifts,
        beta=beta,
    )
    term_evaluators = RLGhBNTDHFFixedTermEvaluators(
        a_direct=lambda i, j: evaluator.a_direct(plus_states[int(i)], plus_states[int(j)]),
        a_exchange=lambda i, j: evaluator.a_exchange(plus_states[int(i)], plus_states[int(j)]),
        b_direct=lambda i, j: evaluator.b_direct(plus_states[int(i)], minus_states[int(j)]),
        b_exchange=lambda i, j: evaluator.b_exchange(plus_states[int(i)], minus_states[int(j)]),
    )
    return RLGhBNTDHFSparseFixedSource(
        context=context,
        plus_states=plus_states,
        minus_states=minus_states,
        x_fixed=x_fixed,
        y_fixed=y_fixed,
        evaluator=evaluator,
        term_evaluators=term_evaluators,
        leg_vector_cache_count=len(vector_cache),
    )


def build_periodic_gauge_basis_view(
    run: RLGhBNHartreeFockRun,
    *,
    periodic_gauge_padding: int,
    name: str = "rlg_hbn_tdhf_periodic_gauge_view",
) -> RLGhBNProjectedBasisData:
    """Rebuild microscopic wavefunctions at the saved torus k points with local padding."""

    base = run.basis_data
    if base.periodic_reciprocal_shifts is None:
        raise RuntimeError("projected basis has no periodic reciprocal shifts")
    return _build_projected_basis_for_indices(
        physical_model=base.model,
        basis_model=base.basis_model,
        interaction=base.interaction,
        kvec=np.asarray(base.kvec, dtype=np.complex128),
        band_indices=base.active_band_indices,
        valleys=base.valleys,
        mesh_size=base.mesh_size,
        k_grid_frac=np.asarray(base.k_grid_frac, dtype=float),
        screening=base.screening,
        name=name,
        build_h0=False,
        reciprocal_shifts=base.periodic_reciprocal_shifts,
        c3_fixed_representative_pairs=base.c3_fixed_representative_pairs,
        periodic_gauge_padding=int(periodic_gauge_padding),
    )


def _shift_rectangular_components(
    vector: np.ndarray,
    basis_data: RLGhBNProjectedBasisData,
    *,
    shift: tuple[int, int],
) -> np.ndarray:
    basis = basis_data.basis
    nx, ny = basis.grid_shape
    local_size = int(basis.local_basis_size)
    sx, sy = (int(shift[0]), int(shift[1]))
    source = np.asarray(vector, dtype=np.complex128).reshape(
        (local_size, nx, ny),
        order="F",
    )
    target = np.zeros_like(source)
    for ix in range(nx):
        target_x = ix + sx
        if target_x < 0 or target_x >= nx:
            continue
        for iy in range(ny):
            target_y = iy + sy
            if target_y < 0 or target_y >= ny:
                continue
            target[:, target_x, target_y] = source[:, ix, iy]
    return target.reshape(np.asarray(vector).shape, order="F")


def _hf_full_vector_in_periodic_gauge(
    basis_data: RLGhBNProjectedBasisData,
    orbitals: RLGhBNTDHFOrbitals,
    *,
    local_index: int,
    k_index: int,
    wrap: tuple[int, int],
) -> np.ndarray:
    n_spin = int(orbitals.n_spin)
    n_flavor = int(orbitals.n_eta)
    component_dim = int(basis_data.basis.wavefunctions.shape[0])
    coefficients = np.asarray(
        orbitals.eigenvectors[:, int(local_index), int(k_index)],
        dtype=np.complex128,
    )
    result = np.zeros(n_spin * n_flavor * component_dim, dtype=np.complex128)
    for basis_local, coefficient in enumerate(coefficients):
        if coefficient == 0.0:
            continue
        spin, flavor, band = _local_to_spin_flavor_band(
            basis_local,
            n_spin=n_spin,
            n_flavor=n_flavor,
        )
        block = np.asarray(
            basis_data.basis.wavefunctions[:, band, flavor, int(k_index)],
            dtype=np.complex128,
        )
        valley = int(basis_data.valleys[flavor])
        shifted = _shift_rectangular_components(
            block,
            basis_data,
            shift=(-valley * int(wrap[0]), -valley * int(wrap[1])),
        )
        offset = (spin * n_flavor + flavor) * component_dim
        result[offset : offset + component_dim] += coefficient * shifted
    return result


def _c3_transform_full_vector(
    basis_data: RLGhBNProjectedBasisData,
    vector: np.ndarray,
    *,
    source_total_shift: tuple[int, int],
    target_total_shift: tuple[int, int],
) -> np.ndarray:
    n_spin = int(basis_data.basis.n_spin)
    n_flavor = int(basis_data.basis.n_flavor)
    component_dim = int(basis_data.basis.wavefunctions.shape[0])
    source = np.asarray(vector, dtype=np.complex128).reshape(
        n_spin * n_flavor,
        component_dim,
    )
    target = np.zeros_like(source)
    for spin in range(n_spin):
        for flavor in range(n_flavor):
            block = spin * n_flavor + flavor
            target[block] = _c3_transform_raw_components(
                source[block],
                basis_data,
                valley=int(basis_data.valleys[flavor]),
                source_total_shift=source_total_shift,
                target_total_shift=target_total_shift,
            )
    return target.reshape(np.asarray(vector).shape)


def _total_periodic_shift(
    basis_data: RLGhBNProjectedBasisData,
    k_index: int,
    wrap: tuple[int, int],
) -> tuple[int, int]:
    if basis_data.periodic_reciprocal_shifts is None:
        raise RuntimeError("projected basis has no periodic reciprocal shifts")
    base = basis_data.periodic_reciprocal_shifts[int(k_index)]
    return (int(base[0]) + int(wrap[0]), int(base[1]) + int(wrap[1]))


def _single_leg_c3_matrix(
    basis_data: RLGhBNProjectedBasisData,
    orbitals: RLGhBNTDHFOrbitals,
    vector_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray],
    *,
    source_k: int,
    source_wrap: tuple[int, int],
    target_k: int,
    target_wrap: tuple[int, int],
) -> np.ndarray:
    source_total = _total_periodic_shift(basis_data, source_k, source_wrap)
    target_total = _total_periodic_shift(basis_data, target_k, target_wrap)
    transformed: dict[int, np.ndarray] = {}
    for source_local in range(orbitals.nt):
        source_key = (
            int(source_local),
            int(source_k),
            (int(source_wrap[0]), int(source_wrap[1])),
        )
        if source_key not in vector_cache:
            vector_cache[source_key] = _hf_full_vector_in_periodic_gauge(
                basis_data,
                orbitals,
                local_index=source_local,
                k_index=source_k,
                wrap=source_wrap,
            )
        transformed[source_local] = _c3_transform_full_vector(
            basis_data,
            vector_cache[source_key],
            source_total_shift=source_total,
            target_total_shift=target_total,
        )
    result = np.zeros((orbitals.nt, orbitals.nt), dtype=np.complex128)
    for target_local in range(orbitals.nt):
        target_key = (
            int(target_local),
            int(target_k),
            (int(target_wrap[0]), int(target_wrap[1])),
        )
        if target_key not in vector_cache:
            vector_cache[target_key] = _hf_full_vector_in_periodic_gauge(
                basis_data,
                orbitals,
                local_index=target_local,
                k_index=target_k,
                wrap=target_wrap,
            )
        target_vector = vector_cache[target_key]
        for source_local in range(orbitals.nt):
            result[target_local, source_local] = np.vdot(
                target_vector,
                transformed[source_local],
            )
    return result


def _pair_label(
    orbitals: RLGhBNTDHFOrbitals,
    pair: object,
) -> tuple[int, int, int]:
    p_local, _ = orbitals.decode_global_index(pair.particle)
    h_local, h_k = orbitals.decode_global_index(pair.hole)
    return int(p_local), int(h_local), int(h_k)


def _c3_k_index(index: int, mesh_size: int) -> int:
    pair = _k_index_to_pair(int(index), int(mesh_size))
    rotated = c3_reciprocal_index(pair)
    target_pair = (rotated[0] % int(mesh_size), rotated[1] % int(mesh_size))
    return _pair_to_k_index(target_pair, int(mesh_size))


def build_raw_pair_c3_sewing(
    basis_data: RLGhBNProjectedBasisData,
    orbitals: RLGhBNTDHFOrbitals,
    *,
    source_pairs: Sequence[object],
    source_shift: tuple[int, int],
    target_pairs: Sequence[object],
    target_shift: tuple[int, int],
    vector_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray] | None = None,
) -> np.ndarray:
    """Build the projected X-sector pair sewing before fixed energy assignment."""

    mesh = int(basis_data.mesh_size)
    mesh_shape = (mesh, mesh)
    cache = {} if vector_cache is None else vector_cache
    target_by_label = {
        _pair_label(orbitals, pair): index
        for index, pair in enumerate(target_pairs)
    }
    source_geometry = decode_finite_q_pair_geometry(
        orbitals,
        source_pairs,
        source_shift,
        mesh_shape,
    )
    result = np.zeros(
        (len(tuple(target_pairs)), len(tuple(source_pairs))),
        dtype=np.complex128,
    )
    leg_cache: dict[
        tuple[int, tuple[int, int], int, tuple[int, int]],
        np.ndarray,
    ] = {}
    for source_index in range(len(tuple(source_pairs))):
        source_h_k = int(source_geometry.h_k[source_index])
        target_h_k = _c3_k_index(source_h_k, mesh)
        target_p_k, target_p_wrap = _shift_k_index_with_wrap(
            target_h_k,
            target_shift,
            mesh_shape,
        )
        source_p_k = int(source_geometry.p_plus_k[source_index])
        source_p_wrap = tuple(int(x) for x in source_geometry.wrap_plus[source_index])
        h_key = (source_h_k, (0, 0), target_h_k, (0, 0))
        p_key = (
            source_p_k,
            source_p_wrap,
            int(target_p_k),
            (int(target_p_wrap[0]), int(target_p_wrap[1])),
        )
        if h_key not in leg_cache:
            leg_cache[h_key] = _single_leg_c3_matrix(
                basis_data,
                orbitals,
                cache,
                source_k=source_h_k,
                source_wrap=(0, 0),
                target_k=target_h_k,
                target_wrap=(0, 0),
            )
        if p_key not in leg_cache:
            leg_cache[p_key] = _single_leg_c3_matrix(
                basis_data,
                orbitals,
                cache,
                source_k=source_p_k,
                source_wrap=source_p_wrap,
                target_k=int(target_p_k),
                target_wrap=(int(target_p_wrap[0]), int(target_p_wrap[1])),
            )
        hole_sewing = leg_cache[h_key]
        particle_sewing = leg_cache[p_key]
        for target_p_local in range(orbitals.nt):
            for target_h_local in range(orbitals.nt):
                target_index = target_by_label.get(
                    (target_p_local, target_h_local, target_h_k)
                )
                if target_index is None:
                    continue
                result[target_index, source_index] += (
                    particle_sewing[
                        target_p_local,
                        int(source_geometry.p_local[source_index]),
                    ]
                    * np.conj(
                        hole_sewing[
                            target_h_local,
                            int(source_geometry.h_local[source_index]),
                        ]
                    )
                )
    return result


@dataclass(frozen=True)
class RLGhBNTDHFEnergySewing:
    matrix: np.ndarray
    assignment_max_energy_delta: float
    assignment_mean_energy_delta: float
    unitarity_residual_max: float
    condition_number: float


def build_energy_assigned_c3_sewing(
    raw_sewing: np.ndarray,
    *,
    source_fixed: np.ndarray,
    target_fixed: np.ndarray,
    source_energies: np.ndarray,
    target_energies: np.ndarray,
    raw_phase_tolerance: float = 1.0e-14,
) -> RLGhBNTDHFEnergySewing:
    """Replace the fixed block by an A0-compatible energy assignment."""

    result = np.array(raw_sewing, dtype=np.complex128, copy=True)
    source_mask = np.asarray(source_fixed, dtype=bool)
    target_mask = np.asarray(target_fixed, dtype=bool)
    source_indices = np.flatnonzero(source_mask)
    target_indices = np.flatnonzero(target_mask)
    if source_indices.size != target_indices.size:
        raise ValueError(
            "source/target fixed pair counts differ: "
            f"{source_indices.size} != {target_indices.size}"
        )
    result[np.ix_(target_mask, ~source_mask)] = 0.0
    result[np.ix_(~target_mask, source_mask)] = 0.0
    result[np.ix_(target_mask, source_mask)] = 0.0
    cost = np.abs(
        np.asarray(target_energies, dtype=float)[target_indices, None]
        - np.asarray(source_energies, dtype=float)[source_indices][None, :]
    )
    row_assignment, col_assignment = linear_sum_assignment(cost)
    for target_row, source_col in zip(row_assignment, col_assignment, strict=True):
        target_index = int(target_indices[int(target_row)])
        source_index = int(source_indices[int(source_col)])
        phase = 1.0 + 0.0j
        raw_value = complex(raw_sewing[target_index, source_index])
        if abs(raw_value) > float(raw_phase_tolerance):
            phase = raw_value / abs(raw_value)
        result[target_index, source_index] = phase
    gram = result.conj().T @ result
    return RLGhBNTDHFEnergySewing(
        matrix=result,
        assignment_max_energy_delta=(
            float(np.max(cost[row_assignment, col_assignment]))
            if row_assignment.size
            else 0.0
        ),
        assignment_mean_energy_delta=(
            float(np.mean(cost[row_assignment, col_assignment]))
            if row_assignment.size
            else 0.0
        ),
        unitarity_residual_max=(
            float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
            if gram.size
            else 0.0
        ),
        condition_number=float(np.linalg.cond(result)) if result.size else 0.0,
    )


def pair_excitation_energies(
    orbitals: RLGhBNTDHFOrbitals,
    pairs: Sequence[object],
) -> np.ndarray:
    return np.asarray(
        [
            orbitals.global_energies[pair.particle]
            - orbitals.global_energies[pair.hole]
            for pair in pairs
        ],
        dtype=float,
    )
