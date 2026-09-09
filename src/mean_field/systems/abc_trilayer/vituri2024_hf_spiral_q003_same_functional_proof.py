"""Candidate-only q003 transition/adjoint and same-functional proof layer.

This module closes a *candidate* proof layer left deliberately open by
``vituri2024_hf_spiral_q003_same_functional_bridge``.  It independently
streams every occupied-to-virtual exact-integer tuple, assigns its unique
signed sector and canonical ``{q,-q}`` root, and checks the hypotheses needed
for the real-inner-product injection/extraction theorem.  The full-q003
receipt path never calls production
``Vituri2024HFSpiralFullHessianContext._build_embedding``; a separately
size-gated reduced diagnostic does call it and compares its exact
order/extraction, including one-empty conjugate orbits.

The term table binds exact production and scalar callables and records a
candidate term-by-term derivation.  Only independently executed geometry and
reduced component comparisons are described as computed facts.  Source fingerprints are provenance only: neither a hash
match nor this candidate derivation is treated as proof authority.  Detached
theorem review and a separate certifier are still required.

The public factory is an honest-caller API boundary, not a hostile Python
sandbox or an unforgeability claim.  Factory products are identity-registered
so copying a closure token into a directly constructed record does not make it
validate.  The receipt is immutable and array-free.  This module exports no
operator, action, LinearOperator, eigensolver, stability, production, or paper
authority surface.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from typing import Final, Iterator

import numpy as np

from . import vituri2024_hf_fft as _fft_module
from . import vituri2024_hf_spiral_full_hessian as _hessian_module
from . import vituri2024_hf_spiral_full_response as _response_module
from . import vituri2024_hf_spiral_full_stability as _inventory_module
from . import vituri2024_hf_spiral_q003_same_functional_bridge as _bridge_module
from . import vituri2024_tdhf_exact_integer_signed_scalar as _scalar_module
from ...core.hf import zero_temperature_sector_stability as _core_module

Array = np.ndarray

VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION: Final[str] = (
    "vituri2024_q003_exact_integer_transition_adjoint_candidate_proof.v1"
)
VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE: Final[str] = (
    "both_pinned_q003_sources_full_selected_spin_fixed_rank_exact_integer_inventory"
)
VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY: Final[str] = (
    "candidate_only_independently_computed_transition_geometry_with_recorded_"
    "formula_and_signed_lane_covariance_derivations_pending_detached_review_"
    "without_production_J_scalar_hessian_action_eigensolver_stability_"
    "scientific_production_or_paper_authority"
)
VITURI2024_Q003_TRANSITION_ORDER: Final[str] = (
    "particle_orbital_id_then_hole_orbital_id_exact_integer_lexicographic"
)
VITURI2024_Q003_CANONICAL_ROOT: Final[str] = (
    "min_lexicographic((dx,dy,chi),(-dx,-dy,-chi))"
)
VITURI2024_Q003_REAL_ADJOINT_THEOREM: Final[str] = (
    "candidate_theorem_hypotheses:[H1_each_nonempty_signed_lane_has_exact_"
    "ordered_particle_hole_embedding_E_s;H2_E_s_vertices_are_injective_and_"
    "opposite_lane_vertices_are_disjoint_by_occupied_virtual_complementarity;"
    "H3_W_r=E_r*x+E_-r*y^dagger_and_W_-r=W_r^dagger;H4_signed_Hilbert_inner_"
    "product_is_half_Re_sum_s_Tr(W_s^dagger*Z_s);H5_kernel_K_is_real_even_"
    "K(delta)=K(-delta),area_is_real_positive,and_spinor_form_factors_on_"
    "opposite_supports_are_complex_conjugates;H6_direct_term_D_s=+Fout_s*"
    "K(-d_s)*sum(Fsrc_s*W_s)_for_chi0_and_zero_otherwise_so_H3_H5_imply_"
    "D_-s[W_s^dagger]=D_s[W_s]^dagger_term_by_term;H7_exchange_term_X_s=\n"
    "-M_s^dagger*C_K[M_s*W_s],and_real_even_C_K_plus_opposite_support_"
    "conjugation_imply_X_-s[W_s^dagger]=X_s[W_s]^dagger_term_by_term;"
    "H8_Sigma_s=(D_s+X_s)/area_therefore_has_the_same_signed_lane_covariance]."
    "Then_J(x,y)=(W_r,W_-r),Jdagger(C_r,C_-r)=(E_r^dagger*C_r,"
    "conj(E_-r^dagger*C_r))_when_C_-r=C_r^dagger;hence_"
    "<J(x,y),C>_H=Re(x^dagger*Jdagger_x+y^dagger*Jdagger_y)."
    "This_is_a_recorded_candidate_derivation_not_an_established_production_"
    "J_identity_or_scalar_Hessian_theorem"
)
VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE: Final[str] = (
    "H_prod=2Jdagger(D_prod+L_prod)J=2Jdagger(D_F+Sigma)J=H_E_"
    "candidate_derivation_pending_detached_review"
)
VITURI2024_Q003_NORMALIZATION: Final[str] = (
    "raw_total_energy_half_pair_real_inner_product_no_Nk_division_"
    "no_extra_interaction_half_real_hessian_factor_two"
)
VITURI2024_Q003_COMMON_FORMULA_INPUTS: Final[str] = (
    "same_exact_selected_spinors_integer_labels_fft_plan_kernel_area_flavor_order_"
    "signed_displacement_valley_charge_and_signed_density_block"
)
VITURI2024_Q003_STREAM_PARTICLE_CHUNK: Final[int] = 16
VITURI2024_Q003_REDUCED_ORACLE_MAX_TRANSITIONS: Final[int] = 10_000

_EXPECTED_OCCUPIED = _bridge_module.VITURI2024_Q003_SELECTED_OCCUPIED_COUNT
_EXPECTED_VIRTUAL = _bridge_module.VITURI2024_Q003_SELECTED_VIRTUAL_COUNT
_EXPECTED_NONEMPTY = _bridge_module.VITURI2024_Q003_NONEMPTY_SECTOR_COUNT
_EXPECTED_ORBITS = _bridge_module.VITURI2024_Q003_CANONICAL_ORBIT_COUNT
_EXPECTED_COMPLEX = _bridge_module.VITURI2024_Q003_COMPLEX_DIMENSION
_EXPECTED_REAL = _bridge_module.VITURI2024_Q003_REAL_DIMENSION

_SHA256 = sha256
_JSON_DUMPS = json.dumps


def _make_factory_boundary():
    """Return an honest-caller factory and identity-product predicate."""

    token = object()
    products: list[object] = []

    def construct(owner: type, /, **kwargs: object):
        result = owner(_factory_token=token, **kwargs)
        products.append(result)
        validator = getattr(result, "validate_live_state", None)
        if validator is not None:
            validator()
        return result

    def require(candidate: object, role: str) -> None:
        if candidate is not token:
            raise TypeError(f"{role} are factory-only")

    def is_product(candidate: object, owner: type) -> bool:
        # Registration, rather than token possession, is the validation gate.
        return type(candidate) is owner and any(candidate is item for item in products)

    return construct, require, is_product


_construct_receipt, _require_factory, _is_factory_product = _make_factory_boundary()


class _OneShotBindingGuard:
    """Capture the original transitive guard without a rebasable module lookup."""

    def __init__(self) -> None:
        self.__callback = None
        self.__namespace = None

    def bind(self, callback: object, namespace: dict[str, object]) -> None:
        if (
            self.__callback is not None
            or not callable(callback)
            or type(namespace) is not dict
        ):
            raise RuntimeError("candidate proof binding guard initialization drifted")
        self.__callback = callback
        self.__namespace = namespace

    def __call__(self) -> None:
        if self.__callback is None or self.__namespace is None:
            raise RuntimeError("candidate proof binding guard is not initialized")
        if self.__namespace.get("_validate_bindings") is not self.__callback:
            raise RuntimeError("candidate proof live-state guard binding drifted")
        self.__callback()


_LIVE_BINDING_GUARD = _OneShotBindingGuard()


def _canonical(value: object) -> bytes:
    return _JSON_DUMPS(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return _SHA256(_canonical(value)).hexdigest()


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    return _SHA256(
        str(array.dtype).encode()
        + b"\0"
        + _JSON_DUMPS(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    ).hexdigest()


def _source_sha256(value: object) -> str:
    return _SHA256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _module_sha256(module: object) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError("candidate proof source owner has no source file")
    return _SHA256(Path(source).resolve().read_bytes()).hexdigest()


def _json_native(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_native(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, tuple):
        return tuple(_json_native(item) for item in value)
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"candidate proof receipt contains non-JSON value {type(value)!r}")


def _strict_centered_inputs(
    inventory: object, *, inventory_already_validated: bool = False
) -> tuple[Array, Array, int, int]:
    if type(inventory) is not _inventory_module.Vituri2024HFSpiralFullSectorInventory:
        raise TypeError("transition proof requires the exact full-sector inventory")
    if type(inventory_already_validated) is not bool:
        raise TypeError("inventory validation state must be an exact bool")
    if not inventory_already_validated:
        inventory.validate_live_state()
    size = inventory.mesh_size
    nk = inventory.nk
    half = size // 2
    labels = inventory.integer_mesh_labels
    expected = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    occupations = inventory.selected_occupations
    if (
        size < 3
        or size % 2 != 1
        or nk != size * size
        or type(labels) is not np.ndarray
        or labels.dtype != np.dtype(np.int64)
        or labels.shape != (nk, 2)
        or labels.flags.writeable
        or not np.array_equal(labels, expected)
        or type(occupations) is not np.ndarray
        or occupations.dtype != np.dtype(np.bool_)
        or occupations.shape != (2, nk)
        or occupations.flags.writeable
    ):
        raise ValueError("transition proof requires immutable exact centered-square inputs")
    return labels, occupations, size, nk


def _root_mask(dx: Array, dy: Array, charge: Array) -> Array:
    """Return whether each signed key is its lexicographic conjugate-pair root."""

    return (
        (dx < -dx)
        | ((dx == -dx) & (dy < -dy))
        | ((dx == -dx) & (dy == -dy) & (charge <= -charge))
    )


def _transition_chunks(
    inventory: object,
    *,
    particle_chunk_size: int = VITURI2024_Q003_STREAM_PARTICLE_CHUNK,
    _validated_inputs: tuple[Array, Array, int, int] | None = None,
) -> Iterator[Array]:
    """Stream canonical transition rows without production embedding helpers.

    Columns are ``particle_slot, particle_k, hole_slot, hole_k, dx, dy, chi,
    root_dx, root_dy, root_chi, lane_side, vertex_row, vertex_col, vertex_base``.
    Rows are rooted in exact ``particle_id, hole_id`` lexicographic order.
    """

    if _validated_inputs is None:
        labels, occupations, _size, nk = _strict_centered_inputs(inventory)
    else:
        if type(_validated_inputs) is not tuple or len(_validated_inputs) != 4:
            raise TypeError("validated transition inputs must be an exact four-tuple")
        labels, occupations, _size, nk = _validated_inputs
    if type(particle_chunk_size) is not int or particle_chunk_size <= 0:
        raise ValueError("particle chunk size must be an exact positive int")
    occupied_ids = np.flatnonzero(occupations.reshape(-1)).astype(np.int64)
    virtual_ids = np.flatnonzero(~occupations.reshape(-1)).astype(np.int64)
    if not np.all(occupied_ids[1:] > occupied_ids[:-1]) or not np.all(
        virtual_ids[1:] > virtual_ids[:-1]
    ):
        raise RuntimeError("selected orbital ids are not strictly ordered")
    hole_slots = occupied_ids // nk
    hole_k = occupied_ids % nk
    selected_valleys = np.asarray(inventory.selected_valleys, dtype=np.int64)
    for start in range(0, int(virtual_ids.size), particle_chunk_size):
        particle_ids_chunk = virtual_ids[start : start + particle_chunk_size]
        particle_ids = np.repeat(particle_ids_chunk, occupied_ids.size)
        hole_ids = np.tile(occupied_ids, particle_ids_chunk.size)
        particle_slots = particle_ids // nk
        particle_k = particle_ids % nk
        tiled_hole_slots = np.tile(hole_slots, particle_ids_chunk.size)
        tiled_hole_k = np.tile(hole_k, particle_ids_chunk.size)
        displacement = labels[particle_k] - labels[tiled_hole_k]
        dx = displacement[:, 0]
        dy = displacement[:, 1]
        charge = selected_valleys[particle_slots] - selected_valleys[tiled_hole_slots]
        is_root = _root_mask(dx, dy, charge)
        root_dx = np.where(is_root, dx, -dx)
        root_dy = np.where(is_root, dy, -dy)
        root_charge = np.where(is_root, charge, -charge)
        lane_side = (~is_root).astype(np.int64)
        vertex_row = np.where(is_root, particle_slots, tiled_hole_slots)
        vertex_col = np.where(is_root, tiled_hole_slots, particle_slots)
        vertex_base = np.where(is_root, tiled_hole_k, particle_k)
        rows = np.stack(
            (
                particle_slots,
                particle_k,
                tiled_hole_slots,
                tiled_hole_k,
                dx,
                dy,
                charge,
                root_dx,
                root_dy,
                root_charge,
                lane_side,
                vertex_row,
                vertex_col,
                vertex_base,
            ),
            axis=1,
        ).astype(np.int64, copy=False)
        yield rows


@dataclass(frozen=True, slots=True)
class _TransitionAudit:
    selected_occupied_count: int
    selected_virtual_count: int
    transition_count: int
    nonempty_signed_lane_count: int
    canonical_orbit_count: int
    nonempty_signed_lanes_checked: int
    zero_self_conjugate_transition_count: int
    sector_dimension_mismatch_count: int
    occupation_mismatch_count: int
    inverse_sector_mismatch_count: int
    inverse_vertex_mismatch_count: int
    reverse_occupation_collision_count: int
    tuple_order_mismatch_count: int
    canonical_tuple_stream_sha256: str
    canonical_vertex_stream_sha256: str


def _audit_transition_geometry(
    inventory: object, *, inventory_already_validated: bool = False
) -> _TransitionAudit:
    """Compute all transition/J hypotheses by a particle-hole pair stream."""

    labels, occupations, size, nk = _strict_centered_inputs(
        inventory, inventory_already_validated=inventory_already_validated
    )
    width = 2 * size - 1
    offset = size - 1
    valley_to_charge_index = {
        int(charge): index for index, charge in enumerate(inventory.valley_charges)
    }
    sector_counts = np.zeros((3, width, width), dtype=np.int64)
    transition_hash = _SHA256()
    vertex_hash = _SHA256()
    transition_count = 0
    occupation_mismatches = 0
    inverse_sector_mismatches = 0
    inverse_vertex_mismatches = 0
    reverse_collisions = 0
    tuple_order_mismatches = 0
    self_conjugate = 0
    previous_particle_id = -1
    previous_hole_id = -1

    for rows in _transition_chunks(
        inventory,
        _validated_inputs=(labels, occupations, size, nk),
    ):
        ps, pk, hs, hk = (rows[:, index] for index in range(4))
        dx, dy, charge = (rows[:, index] for index in range(4, 7))
        root_dx, root_dy, root_charge = (rows[:, index] for index in range(7, 10))
        side, vertex_row, vertex_col, vertex_base = (
            rows[:, index] for index in range(10, 14)
        )
        particle_ids = ps * nk + pk
        hole_ids = hs * nk + hk
        if rows.shape[0]:
            pair_order_bad = (particle_ids[1:] < particle_ids[:-1]) | (
                (particle_ids[1:] == particle_ids[:-1])
                & (hole_ids[1:] <= hole_ids[:-1])
            )
            tuple_order_mismatches += int(np.count_nonzero(pair_order_bad))
            if (
                int(particle_ids[0]) < previous_particle_id
                or (
                    int(particle_ids[0]) == previous_particle_id
                    and int(hole_ids[0]) <= previous_hole_id
                )
            ):
                tuple_order_mismatches += 1
            previous_particle_id = int(particle_ids[-1])
            previous_hole_id = int(hole_ids[-1])

        occupation_mismatches += int(
            np.count_nonzero(occupations[ps, pk] | ~occupations[hs, hk])
        )
        inverse_displacement = labels[pk] - labels[hk]
        inverse_charge = (
            np.asarray(inventory.selected_valleys, dtype=np.int64)[ps]
            - np.asarray(inventory.selected_valleys, dtype=np.int64)[hs]
        )
        inverse_sector_mismatches += int(
            np.count_nonzero(
                (inverse_displacement[:, 0] != dx)
                | (inverse_displacement[:, 1] != dy)
                | (inverse_charge != charge)
            )
        )
        self_conjugate += int(
            np.count_nonzero((dx == 0) & (dy == 0) & (charge == 0))
        )

        reconstructed_ps = np.where(side == 0, vertex_row, vertex_col)
        reconstructed_hs = np.where(side == 0, vertex_col, vertex_row)
        reconstructed_pk = np.where(
            side == 0,
            (vertex_base // size + root_dy) * size
            + vertex_base % size
            + root_dx,
            vertex_base,
        )
        reconstructed_hk = np.where(
            side == 0,
            vertex_base,
            (vertex_base // size + root_dy) * size
            + vertex_base % size
            + root_dx,
        )
        inverse_vertex_mismatches += int(
            np.count_nonzero(
                (reconstructed_ps != ps)
                | (reconstructed_hs != hs)
                | (reconstructed_pk != pk)
                | (reconstructed_hk != hk)
            )
        )
        # A first/second-lane vertex collision would require the reversed
        # transition (h,p).  It cannot be occupied-to-virtual because p is
        # virtual and h is occupied; compute that contradiction for every row.
        reverse_collisions += int(
            np.count_nonzero(occupations[ps, pk] & ~occupations[hs, hk])
        )

        charge_indices = np.where(charge == -2, 0, np.where(charge == 0, 1, 2))
        if np.any((charge != -2) & (charge != 0) & (charge != 2)):
            raise ValueError("transition proof encountered an unsupported valley charge")
        flat_sector = (
            (charge_indices * width + dy + offset) * width + dx + offset
        )
        sector_counts += np.bincount(
            flat_sector, minlength=sector_counts.size
        ).reshape(sector_counts.shape)
        canonical_pairs = np.stack((particle_ids, hole_ids), axis=1).astype(
            np.int64, copy=False
        )
        canonical_vertices = np.stack(
            (
                root_dx,
                root_dy,
                root_charge,
                side,
                vertex_row,
                vertex_col,
                vertex_base,
            ),
            axis=1,
        ).astype(np.int64, copy=False)
        transition_hash.update(canonical_pairs.tobytes(order="C"))
        vertex_hash.update(canonical_vertices.tobytes(order="C"))
        transition_count += int(rows.shape[0])

    production_counts = np.asarray(
        inventory.charge_sector_complex_dimensions, dtype=np.int64
    )
    sector_mismatches = int(np.count_nonzero(sector_counts != production_counts))
    nonempty = int(np.count_nonzero(sector_counts))
    orbit_count = 0
    signed_slots = 0
    charges = tuple(int(value) for value in inventory.valley_charges)
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            for charge in charges:
                key = (dx, dy, charge)
                partner = (-dx, -dy, -charge)
                if key > partner:
                    continue
                first = sector_counts[valley_to_charge_index[charge], dy + offset, dx + offset]
                second = sector_counts[
                    valley_to_charge_index[-charge], -dy + offset, -dx + offset
                ]
                if int(first) + int(second):
                    orbit_count += 1
                    # Count only nonempty signed lanes.  A one-empty orbit is
                    # one checked lane, never two nominal partner slots.
                    signed_slots += int(first > 0) + int(second > 0)

    return _TransitionAudit(
        selected_occupied_count=int(np.count_nonzero(occupations)),
        selected_virtual_count=int(np.count_nonzero(~occupations)),
        transition_count=transition_count,
        nonempty_signed_lane_count=nonempty,
        canonical_orbit_count=orbit_count,
        nonempty_signed_lanes_checked=signed_slots,
        zero_self_conjugate_transition_count=self_conjugate,
        sector_dimension_mismatch_count=sector_mismatches,
        occupation_mismatch_count=occupation_mismatches,
        inverse_sector_mismatch_count=inverse_sector_mismatches,
        inverse_vertex_mismatch_count=inverse_vertex_mismatches,
        reverse_occupation_collision_count=reverse_collisions,
        tuple_order_mismatch_count=tuple_order_mismatches,
        canonical_tuple_stream_sha256=transition_hash.hexdigest(),
        canonical_vertex_stream_sha256=vertex_hash.hexdigest(),
    )


def _reduced_exhaustive_canonical_vertex_oracle(inventory: object) -> int:
    """Compare the stream to an independent scalar-loop canonical-vertex table."""

    labels, occupations, _size, nk = _strict_centered_inputs(inventory)
    expected_count = int(np.count_nonzero(occupations) * np.count_nonzero(~occupations))
    if expected_count > VITURI2024_Q003_REDUCED_ORACLE_MAX_TRANSITIONS:
        raise ValueError("canonical-vertex oracle is restricted to reduced inventories")
    streamed = np.concatenate(tuple(_transition_chunks(inventory)), axis=0)
    brute: list[tuple[int, ...]] = []
    valleys = inventory.selected_valleys
    for particle_id in np.flatnonzero((~occupations).reshape(-1)).tolist():
        ps, pk = divmod(int(particle_id), nk)
        for hole_id in np.flatnonzero(occupations.reshape(-1)).tolist():
            hs, hk = divmod(int(hole_id), nk)
            dx, dy = (int(value) for value in (labels[pk] - labels[hk]))
            charge = int(valleys[ps] - valleys[hs])
            key = (dx, dy, charge)
            root = min(key, (-dx, -dy, -charge))
            side = int(key != root)
            if side == 0:
                vertex = (ps, hs, hk)
            else:
                vertex = (hs, ps, pk)
            brute.append((ps, pk, hs, hk, dx, dy, charge, *root, side, *vertex))
    expected = np.asarray(brute, dtype=np.int64)
    if streamed.shape != expected.shape or not np.array_equal(streamed, expected):
        raise AssertionError("streaming canonical vertices disagree with reduced oracle")
    encoded = (
        (((streamed[:, 7] + nk) * (2 * nk + 1) + (streamed[:, 8] + nk)) * 5
         + (streamed[:, 9] + 2))
        * (4 * nk)
        + (streamed[:, 11] * 2 + streamed[:, 12]) * nk
        + streamed[:, 13]
    )
    if np.unique(encoded).size != encoded.size:
        raise AssertionError("reduced canonical-root vertices collide")
    return int(expected.shape[0])


@dataclass(frozen=True, slots=True)
class _ReducedEmbeddingComparison:
    nonempty_signed_lanes_checked: int
    empty_partner_lanes_skipped: int
    one_empty_orbits_checked: int
    transition_dimensions_summed: int
    support_mismatch_count: int
    flavor_block_mismatch_count: int
    order_or_extraction_mismatch_count: int


def _production_pair_order(charge: int) -> tuple[tuple[int, int], ...]:
    """Independent statement of the reviewed production flavor-loop order."""

    if charge == 0:
        return ((0, 0), (1, 1))
    if charge == 2:
        return ((1, 0),)
    if charge == -2:
        return ((0, 1),)
    raise ValueError("unsupported reduced comparison charge")


def _reduced_actual_embedding_comparison(context: object) -> _ReducedEmbeddingComparison:
    """Execute actual production embedding order/extraction on a reduced source.

    This function is deliberately size-gated.  Its result must not be used to
    claim that the full q003 production embedding was enumerated.
    """

    if type(context) is not _hessian_module.Vituri2024HFSpiralFullHessianContext:
        raise TypeError("reduced embedding comparison requires an exact context")
    context.validate_live_state()
    inventory = context.inventory
    transition_count = inventory.selected_occupied_count * inventory.selected_virtual_count
    if transition_count > VITURI2024_Q003_REDUCED_ORACLE_MAX_TRANSITIONS:
        raise ValueError("actual embedding comparison is restricted to reduced inventories")
    rows = np.concatenate(tuple(_transition_chunks(inventory)), axis=0)
    checked = 0
    skipped = 0
    one_empty = 0
    summed = 0
    support_mismatches = 0
    flavor_mismatches = 0
    mismatches = 0
    for orbit in inventory.iter_conjugate_orbits(include_zero_dimension=False):
        dimensions = (orbit.first_complex_dimension, orbit.second_complex_dimension)
        if (dimensions[0] == 0) != (dimensions[1] == 0):
            one_empty += 1
        for key, dimension in zip((orbit.first, orbit.second), dimensions, strict=True):
            if dimension == 0:
                skipped += 1
                continue
            action = context.action_factory.prepare_fft_action(key)
            scalar_bases, scalar_targets = _scalar_module._support(
                context.response.fft_plan,
                (key.displacement_x, key.displacement_y),
            )
            support_mismatches += int(
                not np.array_equal(action.bases, scalar_bases)
                or not np.array_equal(action.targets, scalar_targets)
            )
            flavor_mismatches += int(
                _response_module._allowed_flavor_blocks(key)
                != _scalar_module._allowed_blocks(key.valley_charge)
            )
            actual = context._build_embedding(key, action.bases, action.targets)
            selected = rows[
                (rows[:, 4] == key.displacement_x)
                & (rows[:, 5] == key.displacement_y)
                & (rows[:, 6] == key.valley_charge)
            ]
            pair_rank = {
                pair: rank for rank, pair in enumerate(_production_pair_order(key.valley_charge))
            }
            ranks = np.asarray(
                [pair_rank[(int(row[0]), int(row[2]))] for row in selected],
                dtype=np.int64,
            )
            order = np.lexsort((selected[:, 3], ranks))
            expected = selected[order]
            arrays_match = (
                actual.complex_dimension == dimension == expected.shape[0]
                and np.array_equal(actual.particle_valley_slots, expected[:, 0])
                and np.array_equal(actual.particle_k_indices, expected[:, 1])
                and np.array_equal(actual.hole_valley_slots, expected[:, 2])
                and np.array_equal(actual.hole_k_indices, expected[:, 3])
                and np.array_equal(actual.particle_orbital_ids, expected[:, 0] * context.nk + expected[:, 1])
                and np.array_equal(actual.hole_orbital_ids, expected[:, 2] * context.nk + expected[:, 3])
            )
            mismatches += int(not arrays_match)
            checked += 1
            summed += dimension
    return _ReducedEmbeddingComparison(
        checked,
        skipped,
        one_empty,
        summed,
        support_mismatches,
        flavor_mismatches,
        mismatches,
    )


def _production_direct_kernel_component(kernel: object, displacement: tuple[int, int]):
    dx, dy = displacement
    return kernel(-dx, -dy)


def _scalar_direct_kernel_component(kernel: object, displacement: tuple[int, int]):
    dx, dy = displacement
    return kernel(0 - dx, 0 - dy)


def _production_exchange_kernel_component(
    kernel: object, output: tuple[int, int], source: tuple[int, int]
):
    return -kernel(output[0] - source[0], output[1] - source[1])


def _scalar_exchange_kernel_component(
    kernel: object, output: tuple[int, int], source: tuple[int, int]
):
    delta = (output[0] - source[0], output[1] - source[1])
    return -kernel(*delta)


def _scalar_direct_component(
    response: object,
    key: object,
    block: Array,
    bases: Array,
    targets: Array,
) -> Array:
    """Independent scalar direct term, including the single area division."""

    result = np.zeros_like(block)
    if key.valley_charge != 0:
        return result
    spinors = response.selected_spinors
    plan = response.fft_plan
    kernel = plan.kernel_by_signed_displacement[
        -key.displacement_y + plan.mesh_size - 1,
        -key.displacement_x + plan.mesh_size - 1,
    ]
    charge = 0.0 + 0.0j
    for flavor in range(2):
        source = np.einsum(
            "cp,cp->p", spinors[flavor][:, bases].conj(),
            spinors[flavor][:, targets], optimize=True,
        )
        charge += np.sum(source * block[flavor, flavor, bases])
    for flavor in range(2):
        target = np.einsum(
            "cm,cm->m", spinors[flavor][:, targets].conj(),
            spinors[flavor][:, bases], optimize=True,
        )
        result[flavor, flavor, bases] = target * kernel * charge
    result /= response.area_angstrom_squared
    return result


def _production_component_action(
    response: object,
    key: object,
    block: Array,
    bases: Array,
    targets: Array,
    component: str,
) -> Array:
    if component == "direct":
        result = _response_module._direct_rank_one_numerator(
            response.selected_spinors, response.fft_plan, key, block, bases, targets
        )
    elif component == "exchange":
        result = np.zeros_like(block)
        shifted = _response_module._shifted_spinors(
            response.selected_spinors, bases, targets
        )
        _response_module._accumulate_exchange_mdagger_ck_m_numerator(
            result, response.selected_spinors, response.fft_plan, key, block, shifted
        )
    else:
        raise ValueError("unknown production component")
    result = np.asarray(result, dtype=np.complex128).copy()
    result /= response.area_angstrom_squared
    support = np.zeros(response.nk, dtype=np.bool_)
    support[bases] = True
    result[:, :, ~support] = 0.0
    return result


def _scalar_component_action(
    response: object,
    key: object,
    block: Array,
    bases: Array,
    targets: Array,
    component: str,
) -> Array:
    direct = _scalar_direct_component(response, key, block, bases, targets)
    if component == "direct":
        return direct
    total = _scalar_module.vituri2024_exact_integer_signed_scalar_fft_action(
        response.selected_spinors,
        response.fft_plan,
        response.area_angstrom_squared,
        (key.displacement_x, key.displacement_y),
        key.valley_charge,
        block,
    )
    if component == "exchange":
        return np.asarray(total) - direct
    if component == "total":
        return np.asarray(total)
    raise ValueError("unknown scalar component")


def _inject_root_block(prepared: object, first: Array, second: Array) -> Array:
    block = np.zeros((2, 2, prepared.context.nk), dtype=np.complex128)
    first_embedding = prepared.first_embedding
    block[
        first_embedding.particle_valley_slots,
        first_embedding.hole_valley_slots,
        first_embedding.hole_k_indices,
    ] = first
    second_embedding = prepared.second_embedding
    if second_embedding.complex_dimension:
        indices = (
            second_embedding.hole_valley_slots,
            second_embedding.particle_valley_slots,
            second_embedding.particle_k_indices,
        )
        if np.any(block[indices] != 0.0):
            raise RuntimeError("reduced paired injection collided across signed lanes")
        block[indices] = second.conj()
    return block


def _extract_scalar_jdagger(
    prepared: object, first: Array, second: Array, component: str
) -> tuple[Array, Array]:
    block = _inject_root_block(prepared, first, second)
    first_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.first)
    second_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.second)
    first_result = _scalar_component_action(
        prepared.context.response, prepared.orbit.first, block,
        first_action.bases, first_action.targets, component,
    )
    conjugate = np.zeros_like(block)
    conjugate[:, :, first_action.targets] = block[
        :, :, first_action.bases
    ].swapaxes(0, 1).conj()
    second_result = _scalar_component_action(
        prepared.context.response, prepared.orbit.second, conjugate,
        second_action.bases, second_action.targets, component,
    )
    first_embedding = prepared.first_embedding
    second_embedding = prepared.second_embedding
    return (
        np.asarray(first_result[
            first_embedding.particle_valley_slots,
            first_embedding.hole_valley_slots,
            first_embedding.hole_k_indices,
        ]),
        np.asarray(second_result[
            second_embedding.particle_valley_slots,
            second_embedding.hole_valley_slots,
            second_embedding.hole_k_indices,
        ]),
    )


class _ReducedComponentAction:
    def __init__(self, response: object, key: object, bases: Array, targets: Array, component: str):
        self.response = response
        self.key = key
        self.bases = bases
        self.targets = targets
        self.component = component

    def __call__(self, block: Array) -> Array:
        return _production_component_action(
            self.response, self.key, block, self.bases, self.targets, self.component
        )


def _production_paired_component(
    prepared: object, first: Array, second: Array, component: str
) -> tuple[Array, Array]:
    first_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.first)
    second_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.second)
    callback = _hessian_module._VituriPairedInteractionCallback(
        response=prepared.context.response,
        first_embedding=prepared.first_embedding,
        second_embedding=prepared.second_embedding,
        first_action=_ReducedComponentAction(
            prepared.context.response, prepared.orbit.first,
            first_action.bases, first_action.targets, component,
        ),
        second_action=_ReducedComponentAction(
            prepared.context.response, prepared.orbit.second,
            second_action.bases, second_action.targets, component,
        ),
        expected_response_fingerprint=prepared.context.response.response_fingerprint,
    )
    return callback(first, second)


def _signed_lane_covariance_residual(
    prepared: object,
    first: Array,
    second: Array,
    component: str,
    implementation: str,
) -> float:
    """Compute ``R_-s[W_s†] - R_s[W_s]†`` on one real basis vector."""

    block = _inject_root_block(prepared, first, second)
    first_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.first)
    second_action = prepared.context.action_factory.prepare_fft_action(prepared.orbit.second)
    partner_block = np.zeros_like(block)
    partner_block[:, :, first_action.targets] = block[
        :, :, first_action.bases
    ].swapaxes(0, 1).conj()

    if implementation == "production" and component == "total":
        root_output = first_action(block)
        partner_output = second_action(partner_block)
    elif implementation == "production":
        root_output = _production_component_action(
            prepared.context.response,
            prepared.orbit.first,
            block,
            first_action.bases,
            first_action.targets,
            component,
        )
        partner_output = _production_component_action(
            prepared.context.response,
            prepared.orbit.second,
            partner_block,
            second_action.bases,
            second_action.targets,
            component,
        )
    elif implementation == "scalar":
        root_output = _scalar_component_action(
            prepared.context.response,
            prepared.orbit.first,
            block,
            first_action.bases,
            first_action.targets,
            component,
        )
        partner_output = _scalar_component_action(
            prepared.context.response,
            prepared.orbit.second,
            partner_block,
            second_action.bases,
            second_action.targets,
            component,
        )
    else:
        raise ValueError("unknown signed-lane covariance implementation")

    expected_partner = np.zeros_like(block)
    expected_partner[:, :, first_action.targets] = np.asarray(root_output)[
        :, :, first_action.bases
    ].swapaxes(0, 1).conj()
    return float(
        np.max(
            np.abs(np.asarray(partner_output) - expected_partner),
            initial=0.0,
        )
    )


@dataclass(frozen=True, slots=True)
class _ReducedTermComparison:
    complete_real_basis_vectors_checked: int
    nonempty_signed_lanes_checked: int
    one_empty_orbits_checked: int
    support_exact: bool
    flavor_blocks_exact: bool
    production_embedding_order_extraction_exact: bool
    direct_component_within_tolerance: bool
    exchange_component_within_tolerance: bool
    area_normalization_within_tolerance: bool
    interaction_sign_within_tolerance: bool
    production_total_signed_lane_covariance_within_tolerance: bool
    production_direct_signed_lane_covariance_within_tolerance: bool
    production_exchange_signed_lane_covariance_within_tolerance: bool
    scalar_total_signed_lane_covariance_within_tolerance: bool
    scalar_direct_signed_lane_covariance_within_tolerance: bool
    scalar_exchange_signed_lane_covariance_within_tolerance: bool
    maximum_total_residual: float
    maximum_direct_residual: float
    maximum_exchange_residual: float
    maximum_production_total_signed_lane_covariance_residual: float
    maximum_production_direct_signed_lane_covariance_residual: float
    maximum_production_exchange_signed_lane_covariance_residual: float
    maximum_scalar_total_signed_lane_covariance_residual: float
    maximum_scalar_direct_signed_lane_covariance_residual: float
    maximum_scalar_exchange_signed_lane_covariance_residual: float


def _reduced_complete_basis_term_comparison(context: object) -> _ReducedTermComparison:
    """Compare production/scalar actions and covariance on a complete reduced basis."""

    embedding = _reduced_actual_embedding_comparison(context)
    residuals = {"total": 0.0, "direct": 0.0, "exchange": 0.0}
    covariance_residuals = {
        (implementation, component): 0.0
        for implementation in ("production", "scalar")
        for component in ("total", "direct", "exchange")
    }
    basis_count = 0
    for orbit in context.inventory.iter_conjugate_orbits(include_zero_dimension=False):
        prepared = context.build_orbit_hessian(orbit.first)
        real_dimension = prepared.real_dimension
        for index in range(real_dimension):
            direction = np.zeros(real_dimension, dtype=np.float64)
            direction[index] = 1.0
            first, second = prepared.hessian.frame.unpack_real(direction)
            actual_total = prepared.hessian._interaction_response(first, second)
            scalar_total = _extract_scalar_jdagger(prepared, first, second, "total")
            actual_total_real = prepared.hessian.frame.pack_complex(
                *actual_total, factor=2.0
            )
            scalar_total_real = prepared.hessian.frame.pack_complex(
                *scalar_total, factor=2.0
            )
            for component in ("direct", "exchange"):
                actual = _production_paired_component(prepared, first, second, component)
                scalar = _extract_scalar_jdagger(prepared, first, second, component)
                actual_real = prepared.hessian.frame.pack_complex(*actual, factor=2.0)
                scalar_real = prepared.hessian.frame.pack_complex(*scalar, factor=2.0)
                residuals[component] = max(
                    residuals[component],
                    float(np.max(np.abs(actual_real - scalar_real), initial=0.0)),
                )
            residuals["total"] = max(
                residuals["total"],
                float(np.max(np.abs(actual_total_real - scalar_total_real), initial=0.0)),
            )
            for implementation in ("production", "scalar"):
                for component in ("total", "direct", "exchange"):
                    key = (implementation, component)
                    covariance_residuals[key] = max(
                        covariance_residuals[key],
                        _signed_lane_covariance_residual(
                            prepared, first, second, component, implementation
                        ),
                    )
            basis_count += 1
    tolerance = 5.0e-12
    complete_geometry = (
        embedding.nonempty_signed_lanes_checked
        == context.inventory.nonempty_sector_count
        and embedding.transition_dimensions_summed == context.inventory.complex_dimension
    )
    all_terms_within_tolerance = all(
        residuals[component] <= tolerance
        for component in ("total", "direct", "exchange")
    )
    return _ReducedTermComparison(
        complete_real_basis_vectors_checked=basis_count,
        nonempty_signed_lanes_checked=embedding.nonempty_signed_lanes_checked,
        one_empty_orbits_checked=embedding.one_empty_orbits_checked,
        support_exact=complete_geometry and embedding.support_mismatch_count == 0,
        flavor_blocks_exact=complete_geometry and embedding.flavor_block_mismatch_count == 0,
        production_embedding_order_extraction_exact=(
            complete_geometry and embedding.order_or_extraction_mismatch_count == 0
        ),
        direct_component_within_tolerance=residuals["direct"] <= tolerance,
        exchange_component_within_tolerance=residuals["exchange"] <= tolerance,
        area_normalization_within_tolerance=all_terms_within_tolerance,
        interaction_sign_within_tolerance=all_terms_within_tolerance,
        production_total_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("production", "total")] <= tolerance
        ),
        production_direct_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("production", "direct")] <= tolerance
        ),
        production_exchange_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("production", "exchange")] <= tolerance
        ),
        scalar_total_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("scalar", "total")] <= tolerance
        ),
        scalar_direct_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("scalar", "direct")] <= tolerance
        ),
        scalar_exchange_signed_lane_covariance_within_tolerance=(
            covariance_residuals[("scalar", "exchange")] <= tolerance
        ),
        maximum_total_residual=residuals["total"],
        maximum_direct_residual=residuals["direct"],
        maximum_exchange_residual=residuals["exchange"],
        maximum_production_total_signed_lane_covariance_residual=covariance_residuals[
            ("production", "total")
        ],
        maximum_production_direct_signed_lane_covariance_residual=covariance_residuals[
            ("production", "direct")
        ],
        maximum_production_exchange_signed_lane_covariance_residual=covariance_residuals[
            ("production", "exchange")
        ],
        maximum_scalar_total_signed_lane_covariance_residual=covariance_residuals[
            ("scalar", "total")
        ],
        maximum_scalar_direct_signed_lane_covariance_residual=covariance_residuals[
            ("scalar", "direct")
        ],
        maximum_scalar_exchange_signed_lane_covariance_residual=covariance_residuals[
            ("scalar", "exchange")
        ],
    )


def _recorded_asymmetric_kernel_orientation_derivation() -> tuple[tuple[str, int], ...]:
    """Record proof-local orientation arithmetic hidden by an even kernel.

    This derivation calls only local scalar helpers.  It does not exercise the
    actual production or scalar low-level action implementation and is not an
    implementation canary.
    """

    def kernel(dx: int, dy: int) -> int:
        return 10_000 + 101 * dx + 17 * dy

    displacement = (1, -1)
    output = (0, 0)
    source = (1, -1)
    direct_expected = kernel(-1, 1)
    direct_wrong = kernel(1, -1)
    exchange_expected = -kernel(-1, 1)
    exchange_functional_orientation = -kernel(1, -1)
    production_direct = _production_direct_kernel_component(kernel, displacement)
    scalar_direct = _scalar_direct_kernel_component(kernel, displacement)
    production_exchange = _production_exchange_kernel_component(kernel, output, source)
    scalar_exchange = _scalar_exchange_kernel_component(kernel, output, source)
    if direct_expected == direct_wrong or exchange_expected == exchange_functional_orientation:
        raise RuntimeError("asymmetric-kernel orientation derivation lost discrimination")
    if production_direct != direct_expected or scalar_direct != direct_expected:
        raise RuntimeError("direct component orientation regressed")
    if production_exchange != exchange_expected or scalar_exchange != exchange_expected:
        raise RuntimeError("exchange component orientation or sign regressed")

    def even_kernel(dx: int, dy: int) -> int:
        return (kernel(dx, dy) + kernel(-dx, -dy)) // 2

    even_direct_expected = 10_000
    even_exchange_expected = -10_000
    if (
        _production_direct_kernel_component(even_kernel, displacement)
        != even_direct_expected
        or _scalar_direct_kernel_component(even_kernel, displacement)
        != even_direct_expected
        or _production_exchange_kernel_component(even_kernel, output, source)
        != even_exchange_expected
        or _scalar_exchange_kernel_component(even_kernel, output, source)
        != even_exchange_expected
        or even_direct_expected != even_kernel(*displacement)
        or even_exchange_expected != -even_kernel(
            source[0] - output[0], source[1] - output[1]
        )
    ):
        raise RuntimeError("even-kernel orientation bridge derivation failed")
    return (
        ("direct_expected_K_minus_d", direct_expected),
        ("direct_production_component", production_direct),
        ("direct_scalar_component", scalar_direct),
        ("direct_wrong_K_plus_d", direct_wrong),
        ("exchange_expected_minus_K_output_minus_source", exchange_expected),
        ("exchange_production_component", production_exchange),
        ("exchange_scalar_component", scalar_exchange),
        ("exchange_wrong_minus_K_source_minus_output", exchange_functional_orientation),
        ("evenized_direct_bridge", even_direct_expected),
        ("evenized_exchange_bridge", even_exchange_expected),
    )


_FORMULA_CALLABLE_SPECS: Final[tuple[tuple[str, object, str], ...]] = (
    ("prod_direct", _response_module, "_direct_rank_one_numerator"),
    ("prod_exchange", _response_module, "_accumulate_exchange_mdagger_ck_m_numerator"),
    ("prod_apply", _response_module.Vituri2024HFSpiralSignedDisplacementResponse, "_apply_fft_validated"),
    ("prod_support", _response_module, "_support_indices"),
    ("prod_blocks", _response_module, "_allowed_flavor_blocks"),
    ("prod_pair_injection", _hessian_module._VituriPairedInteractionCallback, "__call__"),
    ("prod_embedding_source", _hessian_module.Vituri2024HFSpiralFullHessianContext, "_build_embedding"),
    ("prod_real_pack", _core_module.PairedOrbitalTransitionFrame, "pack_complex"),
    ("prod_real_hessian", _core_module.PairedSectorOrbitalHessian, "matvec"),
    ("scalar_sigma", _scalar_module, "vituri2024_exact_integer_signed_scalar_fft_action"),
    ("scalar_support", _scalar_module, "_support"),
    ("scalar_blocks", _scalar_module, "_allowed_blocks"),
    ("scalar_pair_trace", _scalar_module, "vituri2024_exact_integer_paired_interaction_trace"),
    ("scalar_curvature", _scalar_module, "vituri2024_exact_integer_orbital_scalar_curvature"),
)
_IMPORT_FORMULA_CALLABLES = tuple(
    (label, owner, attribute, getattr(owner, attribute))
    for label, owner, attribute in _FORMULA_CALLABLE_SPECS
)
_FORMULA_CONSTANT_SPECS: Final[tuple[tuple[str, object, str], ...]] = (
    (
        "response_kernel_contract",
        _response_module,
        "VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT",
    ),
    (
        "fft_no_wrap_policy",
        _fft_module,
        "VITURI2024_TRANSLATIONAL_HF_FFT_POLICY",
    ),
    (
        "bridge_interaction_identity",
        _bridge_module,
        "VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_IDENTITY",
    ),
    (
        "bridge_interaction_derivation",
        _bridge_module,
        "VITURI2024_Q003_SAME_FUNCTIONAL_INTERACTION_DERIVATION",
    ),
    (
        "scalar_candidate_authority",
        _scalar_module,
        "VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY",
    ),
)
_IMPORT_FORMULA_CONSTANTS = tuple(
    (label, owner, attribute, getattr(owner, attribute))
    for label, owner, attribute in _FORMULA_CONSTANT_SPECS
)

_SOURCE_FRAGMENT_REQUIREMENTS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("prod_direct", ("-key.displacement_y", "-key.displacement_x", "charge += np.sum", "target_form_factor * displacement_kernel * charge")),
    ("scalar_sigma", ("-dy + size - 1", "-dx + size - 1", "charge += np.sum", "target_factor * displacement_kernel * charge", "kernel_fft[None, None, :, :] * transformed", "result[left, right].reshape(size, size)[:] -=", "shifted_grid[left].conj()", "result /= area")),
    ("prod_exchange", ("kernel_fft", "output -= np.einsum", "left_shifted.conj()", "right_base")),
    ("prod_apply", ("_direct_response", "_accumulate_exchange_mdagger_ck_m_numerator", "result /= self.area_angstrom_squared", "result[:, :, ~support] = 0.0")),
    ("prod_embedding_source", ("for particle_valley, hole_valley in _allowed_pairs", "for base in bases", "particle_k.append(target)", "hole_k.append(base_int)")),
    ("prod_pair_injection", ("block[a, b, k] = first_values", "block[hole, particle, base] = second_values.conj()", "swapaxes(0, 1).conj()")),
    ("scalar_pair_trace", ("negative[right, left, target]", "positive[left, right, base].conjugate()", "np.vdot(positive, positive_action)", "np.vdot(negative, negative_action)")),
    ("scalar_curvature", ("2.0 * np.sum", "one_body + interaction")),
    ("prod_real_hessian", ("factor=2.0",)),
)


def _callable_source_records() -> tuple[tuple[str, str, str, str], ...]:
    records = []
    for label, owner, attribute, expected in _IMPORT_FORMULA_CALLABLES:
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(f"candidate proof formula binding drifted: {label}")
        records.append(
            (
                label,
                getattr(expected, "__module__", ""),
                getattr(expected, "__qualname__", ""),
                _source_sha256(expected),
            )
        )
    return tuple(records)


def _formula_constant_records() -> tuple[tuple[str, str, str, object], ...]:
    records = []
    for label, owner, attribute, expected in _IMPORT_FORMULA_CONSTANTS:
        if getattr(owner, attribute, None) != expected:
            raise RuntimeError(f"candidate proof formula constant drifted: {label}")
        records.append((label, getattr(owner, "__name__", ""), attribute, expected))
    return tuple(records)


def _formula_correspondence_table() -> tuple[tuple[object, ...], ...]:
    sources = {label: inspect.getsource(value) for label, _o, _a, value in _IMPORT_FORMULA_CALLABLES}
    for label, fragments in _SOURCE_FRAGMENT_REQUIREMENTS:
        if any(fragment not in sources[label] for fragment in fragments):
            raise RuntimeError(f"candidate proof formula structure drifted: {label}")
    hashes = {record[0]: record[3] for record in _callable_source_records()}
    transition_stream_hash = _source_sha256(_transition_chunks)
    # Rows bind reviewed source structure and record candidate correspondence.
    # The penultimate True means only "source structure bound"; the final
    # False explicitly says executable/full formula equality is not established.
    return (
        ("support", "prod_support", hashes["prod_support"], "scalar_support", hashes["scalar_support"], "exact_centered_integer_particle_minus_hole_no_wrap", True, False),
        ("flavor_blocks", "prod_blocks", hashes["prod_blocks"], "scalar_blocks", hashes["scalar_blocks"], "chi0=(0,0),(1,1);chi+2=(1,0);chi-2=(0,1)", True, False),
        ("transition_geometry", "prod_embedding_source", hashes["prod_embedding_source"], "independent_transition_stream", transition_stream_hash, "full_q003_independent_canonical_tuple_stream;production_order_source_reviewed_only;production_embedding_not_called", True, False),
        ("direct_rank_one", "prod_direct", hashes["prod_direct"], "scalar_sigma", hashes["scalar_sigma"], "candidate_derivation:+F_output_base*K(-d)*sum(F_source_target*W);chi=0_only", True, False),
        ("exchange", "prod_exchange", hashes["prod_exchange"], "scalar_sigma", hashes["scalar_sigma"], "candidate_derivation:-Mdagger*C_K*M;orientation_bridge_requires_real_even_kernel", True, False),
        ("area", "prod_apply", hashes["prod_apply"], "scalar_sigma", hashes["scalar_sigma"], "candidate_derivation:single_division_by_same_source_bound_positive_area", True, False),
        ("pair_geometry", "prod_pair_injection", hashes["prod_pair_injection"], "scalar_pair_trace", hashes["scalar_pair_trace"], "candidate_derivation:W_root=E_first*x+E_second*conj(y);W_partner=W_root^dagger", True, False),
        ("real_hessian", "prod_real_hessian", hashes["prod_real_hessian"], "scalar_curvature", hashes["scalar_curvature"], "candidate_derivation:factor_two_no_Nk_no_extra_interaction_half", True, False),
    )


def _kernel_hypotheses(context: object) -> tuple[float, float, float]:
    plan = context.response.fft_plan
    kernel = plan.kernel_by_signed_displacement
    scale = max(1.0, float(np.max(np.abs(kernel), initial=0.0)))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    imaginary = float(np.max(np.abs(kernel.imag), initial=0.0))
    even = float(np.max(np.abs(kernel - kernel[::-1, ::-1]), initial=0.0))
    if imaginary > tolerance or even > tolerance:
        raise ValueError("candidate proof requires the source-bound real-even kernel")
    return imaginary, even, tolerance


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SameFunctionalProofSourceRecord:
    _factory_token: InitVar[object]
    group_label: str
    context_fingerprint: str
    inventory_fingerprint: str
    response_fingerprint: str
    selected_spinors_sha256: str
    integer_mesh_labels_sha256: str
    selected_occupations_sha256: str
    fft_plan_fingerprint: str
    kernel_sha256: str
    area_angstrom_squared: float
    selected_occupied_count: int
    selected_virtual_count: int
    transition_count: int
    complex_dimension: int
    real_dimension: int
    nonempty_signed_lane_count: int
    canonical_orbit_count: int
    nonempty_signed_lanes_checked: int
    maximum_kernel_imaginary_residual: float
    maximum_kernel_evenness_residual: float
    kernel_tolerance: float
    canonical_tuple_stream_sha256: str
    canonical_vertex_stream_sha256: str
    fingerprint: str = field(init=False)

    def __post_init__(
        self, _factory_token: object, _require: object = _require_factory
    ) -> None:
        _require(_factory_token, "candidate proof source records")
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in ("_factory_token", "fingerprint")
        }
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def validate_live_state(
        self,
        _registered: object = _is_factory_product,
        _owner_and_guard: object = None,
    ) -> None:
        if _owner_and_guard is None:
            owner = Vituri2024Q003SameFunctionalProofSourceRecord
            guard = _LIVE_BINDING_GUARD.__call__
        else:
            owner, guard = _owner_and_guard
        if not _registered(self, owner):
            raise ValueError("candidate proof source is not an identity-registered factory product")
        guard()
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in ("_factory_token", "fingerprint")
        }
        if self.fingerprint != _fingerprint(payload):
            raise ValueError("candidate proof source fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024Q003SameFunctionalCandidateProofReceipt:
    """Array-free candidate theorem receipt; every promotion remains false."""

    _factory_token: InitVar[object]
    sources: tuple[Vituri2024Q003SameFunctionalProofSourceRecord, ...]
    bridge_receipt_fingerprint: str
    callable_source_records: tuple[tuple[str, str, str, str], ...]
    formula_constant_records: tuple[tuple[str, str, str, object], ...]
    formula_correspondence_table: tuple[tuple[object, ...], ...]
    recorded_asymmetric_kernel_orientation_derivation: tuple[tuple[str, int], ...]
    implementation_fingerprint: str
    receipt_fingerprint: str = field(init=False)
    api_version: str = field(default=VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION, init=False)
    scope: str = field(default=VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE, init=False)
    authority: str = field(default=VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY, init=False)
    transition_order: str = field(default=VITURI2024_Q003_TRANSITION_ORDER, init=False)
    canonical_root: str = field(default=VITURI2024_Q003_CANONICAL_ROOT, init=False)
    real_adjoint_theorem: str = field(default=VITURI2024_Q003_REAL_ADJOINT_THEOREM, init=False)
    hessian_identity: str = field(default=VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE, init=False)
    normalization: str = field(default=VITURI2024_Q003_NORMALIZATION, init=False)
    common_formula_inputs: str = field(
        default=VITURI2024_Q003_COMMON_FORMULA_INPUTS, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    exact_integer_transition_enumeration_completed: bool = field(default=True, init=False)
    all_nonempty_signed_lanes_enumerated: bool = field(default=True, init=False)
    canonical_root_ordering_checked: bool = field(default=True, init=False)
    lane_tuple_injectivity_checked: bool = field(default=True, init=False)
    cross_lane_no_collision_checked: bool = field(default=True, init=False)
    inverse_support_checked: bool = field(default=True, init=False)
    independent_inverse_extraction_hypothesis_checked: bool = field(
        default=True, init=False
    )
    full_inventory_partition_checked: bool = field(default=True, init=False)
    independent_real_inner_product_geometry_hypothesis_checked: bool = field(
        default=True, init=False
    )
    candidate_j_jdagger_adjoint_derivation_recorded: bool = field(default=True, init=False)
    candidate_signed_lane_covariance_derivation_recorded: bool = field(default=True, init=False)
    candidate_formula_correspondence_derivation_recorded: bool = field(default=True, init=False)
    exact_formula_constants_bound: bool = field(default=True, init=False)
    candidate_common_formula_inputs_derivation_recorded: bool = field(default=True, init=False)
    implementation_canary_passed: bool = field(default=False, init=False)
    honest_caller_runtime_bindings_checked: bool = field(default=True, init=False)
    equality_claimed_from_fingerprints_alone: bool = field(default=False, init=False)
    production_build_embedding_called: bool = field(default=False, init=False)
    full_q003_production_embedding_enumeration_completed: bool = field(default=False, init=False)
    production_j_identity_computed: bool = field(default=False, init=False)
    signed_lane_covariance_established: bool = field(default=False, init=False)
    formula_correspondence_established: bool = field(default=False, init=False)
    detached_theorem_review_completed: bool = field(default=False, init=False)
    j_jdagger_adjoint_theorem_authority_established: bool = field(default=False, init=False)
    l_prod_equals_sigma_authority_established: bool = field(default=False, init=False)
    same_functional_scalar_hessian_authority_established: bool = field(default=False, init=False)
    full_inventory_exact_unitary_curvature_established: bool = field(default=False, init=False)
    literal_float_full_functional_parity_established: bool = field(default=False, init=False)
    linear_operator_authorized: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    scientific_authority_promoted: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(
        self, _factory_token: object, _require: object = _require_factory
    ) -> None:
        _require(_factory_token, "candidate proof receipts")
        payload = {
            item.name: _json_native(getattr(self, item.name))
            for item in fields(self)
            if item.name not in ("_factory_token", "receipt_fingerprint")
        }
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(payload))

    def validate_live_state(
        self,
        _registered: object = _is_factory_product,
        _guard: object = _LIVE_BINDING_GUARD.__call__,
    ) -> None:
        if not _registered(self, Vituri2024Q003SameFunctionalCandidateProofReceipt):
            raise ValueError("candidate proof receipt is not an identity-registered factory product")
        _guard()
        if (
            type(self.sources) is not tuple
            or any(
                type(source) is not Vituri2024Q003SameFunctionalProofSourceRecord
                for source in self.sources
            )
            or tuple(source.group_label for source in self.sources) != ("3307", "40fd")
        ):
            raise ValueError("candidate proof must contain both exact q003 sources once")
        if (
            type(self.bridge_receipt_fingerprint) is not str
            or len(self.bridge_receipt_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.bridge_receipt_fingerprint
            )
        ):
            raise ValueError("candidate proof bridge fingerprint is invalid")
        for source in self.sources:
            source.validate_live_state()
            source_payload = {
                item.name: getattr(source, item.name)
                for item in fields(source)
                if item.name not in ("_factory_token", "fingerprint")
            }
            digests = (
                source.context_fingerprint,
                source.inventory_fingerprint,
                source.response_fingerprint,
                source.selected_spinors_sha256,
                source.integer_mesh_labels_sha256,
                source.selected_occupations_sha256,
                source.fft_plan_fingerprint,
                source.kernel_sha256,
                source.canonical_tuple_stream_sha256,
                source.canonical_vertex_stream_sha256,
                source.fingerprint,
            )
            source_locked = (
                source.fingerprint == _fingerprint(source_payload),
                all(
                    type(digest) is str
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                    for digest in digests
                ),
                type(source.area_angstrom_squared) is float,
                bool(np.isfinite(source.area_angstrom_squared)),
                source.area_angstrom_squared > 0.0,
                source.selected_occupied_count == _EXPECTED_OCCUPIED,
                source.selected_virtual_count == _EXPECTED_VIRTUAL,
                source.transition_count == _EXPECTED_COMPLEX,
                source.complex_dimension == _EXPECTED_COMPLEX,
                source.real_dimension == _EXPECTED_REAL,
                source.real_dimension == 2 * source.complex_dimension,
                source.nonempty_signed_lane_count == _EXPECTED_NONEMPTY,
                source.canonical_orbit_count == _EXPECTED_ORBITS,
                source.nonempty_signed_lane_count == 38_259,
                source.nonempty_signed_lanes_checked == 38_259,
                source.maximum_kernel_imaginary_residual <= source.kernel_tolerance,
                source.maximum_kernel_evenness_residual <= source.kernel_tolerance,
                source.kernel_tolerance > 0.0,
            )
            if not all(source_locked):
                raise ValueError("candidate proof source record content drifted")
        payload = {
            item.name: _json_native(getattr(self, item.name))
            for item in fields(self)
            if item.name not in ("_factory_token", "receipt_fingerprint")
        }
        locked = (
            self.receipt_fingerprint == _fingerprint(payload),
            self.callable_source_records == _callable_source_records(),
            self.formula_constant_records == _formula_constant_records(),
            self.formula_correspondence_table == _formula_correspondence_table(),
            self.recorded_asymmetric_kernel_orientation_derivation
            == _recorded_asymmetric_kernel_orientation_derivation(),
            self.implementation_fingerprint == vituri2024_q003_same_functional_proof_implementation_fingerprint(),
            self.api_version == VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION,
            self.scope == VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE,
            self.authority == VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY,
            self.transition_order == VITURI2024_Q003_TRANSITION_ORDER,
            self.canonical_root == VITURI2024_Q003_CANONICAL_ROOT,
            self.real_adjoint_theorem == VITURI2024_Q003_REAL_ADJOINT_THEOREM,
            self.hessian_identity == VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE,
            self.normalization == VITURI2024_Q003_NORMALIZATION,
            self.common_formula_inputs == VITURI2024_Q003_COMMON_FORMULA_INPUTS,
            self.candidate_only is True,
            self.exact_integer_transition_enumeration_completed is True,
            self.all_nonempty_signed_lanes_enumerated is True,
            self.canonical_root_ordering_checked is True,
            self.lane_tuple_injectivity_checked is True,
            self.cross_lane_no_collision_checked is True,
            self.inverse_support_checked is True,
            self.independent_inverse_extraction_hypothesis_checked is True,
            self.full_inventory_partition_checked is True,
            self.independent_real_inner_product_geometry_hypothesis_checked is True,
            self.candidate_j_jdagger_adjoint_derivation_recorded is True,
            self.candidate_signed_lane_covariance_derivation_recorded is True,
            self.candidate_formula_correspondence_derivation_recorded is True,
            self.exact_formula_constants_bound is True,
            self.candidate_common_formula_inputs_derivation_recorded is True,
            self.implementation_canary_passed is False,
            self.honest_caller_runtime_bindings_checked is True,
            self.equality_claimed_from_fingerprints_alone is False,
            self.production_build_embedding_called is False,
            self.full_q003_production_embedding_enumeration_completed is False,
            self.production_j_identity_computed is False,
            self.signed_lane_covariance_established is False,
            self.formula_correspondence_established is False,
            self.detached_theorem_review_completed is False,
            self.j_jdagger_adjoint_theorem_authority_established is False,
            self.l_prod_equals_sigma_authority_established is False,
            self.same_functional_scalar_hessian_authority_established is False,
            self.full_inventory_exact_unitary_curvature_established is False,
            self.literal_float_full_functional_parity_established is False,
            self.linear_operator_authorized is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.scientific_authority_promoted is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("candidate proof authority or source binding drifted")


def _bind_live_validator(
    implementation: object,
    registered: object,
    second_argument: object,
):
    """Remove caller-overridable guard parameters while retaining closures."""

    def validate_live_state(self) -> None:
        implementation(self, registered, second_argument)

    return validate_live_state


Vituri2024Q003SameFunctionalProofSourceRecord.validate_live_state = _bind_live_validator(
    Vituri2024Q003SameFunctionalProofSourceRecord.validate_live_state,
    _is_factory_product,
    (
        Vituri2024Q003SameFunctionalProofSourceRecord,
        _LIVE_BINDING_GUARD.__call__,
    ),
)
Vituri2024Q003SameFunctionalCandidateProofReceipt.validate_live_state = _bind_live_validator(
    Vituri2024Q003SameFunctionalCandidateProofReceipt.validate_live_state,
    _is_factory_product,
    _LIVE_BINDING_GUARD.__call__,
)


def _validate_audit(audit: _TransitionAudit, *, enforce_q003: bool) -> None:
    hypotheses = (
        audit.transition_count
        == audit.selected_occupied_count * audit.selected_virtual_count,
        audit.zero_self_conjugate_transition_count == 0,
        audit.sector_dimension_mismatch_count == 0,
        audit.occupation_mismatch_count == 0,
        audit.inverse_sector_mismatch_count == 0,
        audit.inverse_vertex_mismatch_count == 0,
        audit.reverse_occupation_collision_count == 0,
        audit.tuple_order_mismatch_count == 0,
        audit.nonempty_signed_lanes_checked == audit.nonempty_signed_lane_count,
    )
    if not all(hypotheses):
        raise ValueError(f"candidate transition/J hypotheses failed: {audit!r}")
    if enforce_q003 and (
        audit.selected_occupied_count != _EXPECTED_OCCUPIED
        or audit.selected_virtual_count != _EXPECTED_VIRTUAL
        or audit.transition_count != _EXPECTED_COMPLEX
        or _EXPECTED_NONEMPTY != 38_259
        or audit.nonempty_signed_lane_count != 38_259
        or audit.nonempty_signed_lanes_checked != 38_259
        or audit.canonical_orbit_count != _EXPECTED_ORBITS
        or 2 * audit.transition_count != _EXPECTED_REAL
    ):
        raise ValueError("candidate proof inventory is not the complete pinned q003 inventory")


def _build_source_record(
    context: object,
    group_label: str,
    bridge_source: object,
    factory: object,
) -> Vituri2024Q003SameFunctionalProofSourceRecord:
    if type(context) is not _hessian_module.Vituri2024HFSpiralFullHessianContext:
        raise TypeError("candidate proof requires exact full-Hessian contexts")
    context.validate_live_state()
    if (
        bridge_source.group_label != group_label
        or bridge_source.context_fingerprint != context.context_fingerprint
        or bridge_source.inventory_fingerprint != context.inventory.inventory_fingerprint
        or bridge_source.response_fingerprint != context.response.response_fingerprint
    ):
        raise ValueError("candidate proof context is not bound to its bridge source")
    audit = _audit_transition_geometry(
        context.inventory, inventory_already_validated=True
    )
    _validate_audit(audit, enforce_q003=True)
    imaginary, even, tolerance = _kernel_hypotheses(context)
    response = context.response
    plan = response.fft_plan
    return factory(
        Vituri2024Q003SameFunctionalProofSourceRecord,
        group_label=group_label,
        context_fingerprint=context.context_fingerprint,
        inventory_fingerprint=context.inventory.inventory_fingerprint,
        response_fingerprint=response.response_fingerprint,
        selected_spinors_sha256=_array_sha256(response.selected_spinors),
        integer_mesh_labels_sha256=_array_sha256(context.inventory.integer_mesh_labels),
        selected_occupations_sha256=_array_sha256(context.inventory.selected_occupations),
        fft_plan_fingerprint=plan.fingerprint,
        kernel_sha256=_array_sha256(plan.kernel_by_signed_displacement),
        area_angstrom_squared=response.area_angstrom_squared,
        selected_occupied_count=audit.selected_occupied_count,
        selected_virtual_count=audit.selected_virtual_count,
        transition_count=audit.transition_count,
        complex_dimension=audit.transition_count,
        real_dimension=2 * audit.transition_count,
        nonempty_signed_lane_count=audit.nonempty_signed_lane_count,
        canonical_orbit_count=audit.canonical_orbit_count,
        nonempty_signed_lanes_checked=audit.nonempty_signed_lanes_checked,
        maximum_kernel_imaginary_residual=imaginary,
        maximum_kernel_evenness_residual=even,
        kernel_tolerance=tolerance,
        canonical_tuple_stream_sha256=audit.canonical_tuple_stream_sha256,
        canonical_vertex_stream_sha256=audit.canonical_vertex_stream_sha256,
    )


def _build_candidate_proof_impl(
    *,
    context_3307: object,
    context_40fd: object,
    same_functional_bridge_receipt: object,
    _factory: object,
    _bridge_receipt_type: type,
) -> Vituri2024Q003SameFunctionalCandidateProofReceipt:
    _validate_bindings()
    if type(same_functional_bridge_receipt) is not _bridge_receipt_type:
        raise TypeError("candidate proof requires the exact same-functional bridge receipt")
    same_functional_bridge_receipt.validate_live_state()
    if (
        same_functional_bridge_receipt.scalar_hessian_authority_established is not False
        or same_functional_bridge_receipt.linear_operator_authorized is not False
        or same_functional_bridge_receipt.hermitian_eigensolver_authorized is not False
        or same_functional_bridge_receipt.scientific_authority_promoted is not False
    ):
        raise ValueError("candidate proof requires an unpromoted structural bridge")
    by_label = {source.group_label: source for source in same_functional_bridge_receipt.sources}
    if tuple(sorted(by_label)) != ("3307", "40fd"):
        raise ValueError("same-functional bridge source inventory is incomplete")
    records = (
        _build_source_record(context_3307, "3307", by_label["3307"], _factory),
        _build_source_record(context_40fd, "40fd", by_label["40fd"], _factory),
    )
    receipt = _factory(
        Vituri2024Q003SameFunctionalCandidateProofReceipt,
        sources=records,
        bridge_receipt_fingerprint=same_functional_bridge_receipt.receipt_fingerprint,
        callable_source_records=_callable_source_records(),
        formula_constant_records=_formula_constant_records(),
        formula_correspondence_table=_formula_correspondence_table(),
        recorded_asymmetric_kernel_orientation_derivation=(
            _recorded_asymmetric_kernel_orientation_derivation()
        ),
        implementation_fingerprint=vituri2024_q003_same_functional_proof_implementation_fingerprint(),
    )
    _validate_bindings()
    return receipt


_IMPORT_LOCAL_ALIASES = (
    ("np", np),
    ("inspect", inspect),
    ("json", json),
    ("Path", Path),
    ("sys", sys),
    ("sha256", sha256),
    ("fields", fields),
    ("is_dataclass", is_dataclass),
    ("Array", Array),
    ("_require_factory", _require_factory),
    ("_is_factory_product", _is_factory_product),
    ("_LIVE_BINDING_GUARD", _LIVE_BINDING_GUARD),
    ("_TransitionAudit", _TransitionAudit),
    ("_ReducedEmbeddingComparison", _ReducedEmbeddingComparison),
    ("_ReducedTermComparison", _ReducedTermComparison),
    ("_ReducedComponentAction", _ReducedComponentAction),
    (
        "Vituri2024Q003SameFunctionalProofSourceRecord",
        Vituri2024Q003SameFunctionalProofSourceRecord,
    ),
    (
        "Vituri2024Q003SameFunctionalCandidateProofReceipt",
        Vituri2024Q003SameFunctionalCandidateProofReceipt,
    ),
    ("_fft_module", _fft_module),
    ("_hessian_module", _hessian_module),
    ("_response_module", _response_module),
    ("_inventory_module", _inventory_module),
    ("_bridge_module", _bridge_module),
    ("_scalar_module", _scalar_module),
    ("_core_module", _core_module),
)
_IMPORT_PRIMITIVES = (
    (np, "ndarray", np.ndarray),
    (np, "ascontiguousarray", np.ascontiguousarray),
    (np, "asarray", np.asarray),
    (np, "flatnonzero", np.flatnonzero),
    (np, "repeat", np.repeat),
    (np, "tile", np.tile),
    (np, "where", np.where),
    (np, "stack", np.stack),
    (np, "zeros", np.zeros),
    (np, "bincount", np.bincount),
    (np, "count_nonzero", np.count_nonzero),
    (np, "array_equal", np.array_equal),
    (np, "any", np.any),
    (np, "all", np.all),
    (np, "isfinite", np.isfinite),
    (np, "max", np.max),
    (np, "abs", np.abs),
    (np, "unique", np.unique),
    (inspect, "getsource", inspect.getsource),
    (inspect, "getsourcefile", inspect.getsourcefile),
    (json, "dumps", json.dumps),
    (Path, "read_bytes", Path.read_bytes),
    (Path, "resolve", Path.resolve),
    (sys, "modules", sys.modules),
)
_IMPORT_MODULE_BINDINGS = tuple(
    (module.__name__, module)
    for module in (
        sys.modules[__name__],
        _fft_module,
        _hessian_module,
        _response_module,
        _inventory_module,
        _bridge_module,
        _scalar_module,
        _core_module,
    )
)
_IMPORT_TRANSITIVE_CALLABLES = (
    (
        _hessian_module.Vituri2024HFSpiralFullHessianContext,
        "validate_live_state",
        _hessian_module.Vituri2024HFSpiralFullHessianContext.validate_live_state,
    ),
    (
        _inventory_module.Vituri2024HFSpiralFullSectorInventory,
        "validate_live_state",
        _inventory_module.Vituri2024HFSpiralFullSectorInventory.validate_live_state,
    ),
    (
        _response_module.Vituri2024HFSpiralValidatedResponseActionFactory,
        "prepare_fft_action",
        _response_module.Vituri2024HFSpiralValidatedResponseActionFactory.prepare_fft_action,
    ),
    (
        _bridge_module.Vituri2024Q003SameFunctionalStructuralReceipt,
        "validate_live_state",
        _bridge_module.Vituri2024Q003SameFunctionalStructuralReceipt.validate_live_state,
    ),
    (
        _bridge_module,
        "vituri2024_q003_same_functional_formula_implementation_fingerprint",
        _bridge_module.vituri2024_q003_same_functional_formula_implementation_fingerprint,
    ),
    (
        _scalar_module,
        "vituri2024_exact_integer_signed_scalar_implementation_fingerprint",
        _scalar_module.vituri2024_exact_integer_signed_scalar_implementation_fingerprint,
    ),
)
_IMPORT_CONSTANTS = (
    ("VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION", VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION),
    ("VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE", VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE),
    ("VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY", VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY),
    ("VITURI2024_Q003_TRANSITION_ORDER", VITURI2024_Q003_TRANSITION_ORDER),
    ("VITURI2024_Q003_CANONICAL_ROOT", VITURI2024_Q003_CANONICAL_ROOT),
    ("VITURI2024_Q003_REAL_ADJOINT_THEOREM", VITURI2024_Q003_REAL_ADJOINT_THEOREM),
    ("VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE", VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE),
    ("VITURI2024_Q003_NORMALIZATION", VITURI2024_Q003_NORMALIZATION),
    ("VITURI2024_Q003_COMMON_FORMULA_INPUTS", VITURI2024_Q003_COMMON_FORMULA_INPUTS),
    ("_EXPECTED_OCCUPIED", _EXPECTED_OCCUPIED),
    ("_EXPECTED_VIRTUAL", _EXPECTED_VIRTUAL),
    ("_EXPECTED_NONEMPTY", _EXPECTED_NONEMPTY),
    ("_EXPECTED_ORBITS", _EXPECTED_ORBITS),
    ("_EXPECTED_COMPLEX", _EXPECTED_COMPLEX),
    ("_EXPECTED_REAL", _EXPECTED_REAL),
)


def _current_implementation_fingerprint() -> str:
    return _fingerprint(
        {
            "api_version": VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION,
            "module_files": tuple(
                (getattr(module, "__name__", ""), _module_sha256(module))
                for module in (
                    sys.modules[__name__],
                    _bridge_module,
                    _hessian_module,
                    _response_module,
                    _inventory_module,
                    _scalar_module,
                    _core_module,
                    _fft_module,
                )
            ),
            "callables": _callable_source_records(),
            "constants": _formula_constant_records(),
            "formula_table": _formula_correspondence_table(),
            "orientation_derivation": (
                _recorded_asymmetric_kernel_orientation_derivation()
            ),
            "numpy_version": np.__version__,
        }
    )


_IMPORT_IMPLEMENTATION_FINGERPRINT = _current_implementation_fingerprint()


def vituri2024_q003_same_functional_proof_implementation_fingerprint() -> str:
    _validate_bindings()
    current = _current_implementation_fingerprint()
    if current != _IMPORT_IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("candidate proof implementation drifted after import")
    return current


_RECEIPT_METHOD_SPECS = (
    (Vituri2024Q003SameFunctionalProofSourceRecord, "__post_init__"),
    (Vituri2024Q003SameFunctionalProofSourceRecord, "validate_live_state"),
    (Vituri2024Q003SameFunctionalCandidateProofReceipt, "__post_init__"),
    (Vituri2024Q003SameFunctionalCandidateProofReceipt, "validate_live_state"),
)
_IMPORT_RECEIPT_METHODS = tuple(
    (owner, name, getattr(owner, name)) for owner, name in _RECEIPT_METHOD_SPECS
)
_IMPORT_LOCAL_FUNCTIONS = tuple(
    (name, globals()[name])
    for name in (
        "_make_factory_boundary",
        "_canonical",
        "_fingerprint",
        "_array_sha256",
        "_source_sha256",
        "_module_sha256",
        "_json_native",
        "_strict_centered_inputs",
        "_root_mask",
        "_transition_chunks",
        "_audit_transition_geometry",
        "_reduced_exhaustive_canonical_vertex_oracle",
        "_production_pair_order",
        "_reduced_actual_embedding_comparison",
        "_production_direct_kernel_component",
        "_scalar_direct_kernel_component",
        "_production_exchange_kernel_component",
        "_scalar_exchange_kernel_component",
        "_scalar_direct_component",
        "_production_component_action",
        "_scalar_component_action",
        "_inject_root_block",
        "_extract_scalar_jdagger",
        "_production_paired_component",
        "_signed_lane_covariance_residual",
        "_reduced_complete_basis_term_comparison",
        "_recorded_asymmetric_kernel_orientation_derivation",
        "_callable_source_records",
        "_formula_constant_records",
        "_formula_correspondence_table",
        "_kernel_hypotheses",
        "_bind_live_validator",
        "_validate_audit",
        "_build_source_record",
        "_build_candidate_proof_impl",
        "_current_implementation_fingerprint",
        "vituri2024_q003_same_functional_proof_implementation_fingerprint",
    )
)


def _validate_bindings(
    _namespace: dict[str, object] = globals(),
    _local_aliases: tuple[tuple[str, object], ...] = _IMPORT_LOCAL_ALIASES,
    _primitives: tuple[tuple[object, str, object], ...] = _IMPORT_PRIMITIVES,
    _modules: tuple[tuple[str, object], ...] = _IMPORT_MODULE_BINDINGS,
    _transitive: tuple[tuple[object, str, object], ...] = _IMPORT_TRANSITIVE_CALLABLES,
    _constants: tuple[tuple[str, object], ...] = _IMPORT_CONSTANTS,
    _formula_specs: tuple[tuple[str, object, str], ...] = _FORMULA_CALLABLE_SPECS,
    _formula_callables: tuple[tuple[str, object, str, object], ...] = _IMPORT_FORMULA_CALLABLES,
    _constant_specs: tuple[tuple[str, object, str], ...] = _FORMULA_CONSTANT_SPECS,
    _formula_constants: tuple[tuple[str, object, str, object], ...] = _IMPORT_FORMULA_CONSTANTS,
    _receipt_methods: tuple[tuple[type, str, object], ...] = _IMPORT_RECEIPT_METHODS,
    _local_functions: tuple[tuple[str, object], ...] = _IMPORT_LOCAL_FUNCTIONS,
    _callable_records: object = _callable_source_records,
    _constant_records: object = _formula_constant_records,
) -> None:
    table_bindings = (
        ("_IMPORT_LOCAL_ALIASES", _local_aliases),
        ("_IMPORT_PRIMITIVES", _primitives),
        ("_IMPORT_MODULE_BINDINGS", _modules),
        ("_IMPORT_TRANSITIVE_CALLABLES", _transitive),
        ("_IMPORT_CONSTANTS", _constants),
        ("_IMPORT_FORMULA_CALLABLES", _formula_callables),
        ("_IMPORT_FORMULA_CONSTANTS", _formula_constants),
        ("_IMPORT_RECEIPT_METHODS", _receipt_methods),
        ("_IMPORT_LOCAL_FUNCTIONS", _local_functions),
    )
    for name, expected in table_bindings:
        if _namespace.get(name) is not expected:
            raise RuntimeError(f"candidate proof binding table drifted: {name}")
    if _namespace.get("_SHA256") is not sha256 or _namespace.get("_JSON_DUMPS") is not json.dumps:
        raise RuntimeError("candidate proof hash/serialization primitive drifted")
    for name, expected in _local_aliases:
        if _namespace.get(name) is not expected:
            raise RuntimeError(f"candidate proof local alias drifted: {name}")
    for owner, attribute, expected in _primitives:
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(f"candidate proof primitive binding drifted: {attribute}")
    for module_name, expected in _modules:
        if sys.modules.get(module_name) is not expected:
            raise RuntimeError(f"candidate proof module binding drifted: {module_name}")
    for owner, attribute, expected in _transitive:
        if getattr(owner, attribute, None) is not expected:
            raise RuntimeError(f"candidate proof transitive binding drifted: {attribute}")
    for name, expected in _constants:
        if _namespace.get(name) != expected:
            raise RuntimeError(f"candidate proof constant drifted: {name}")
    if _namespace.get("_FORMULA_CALLABLE_SPECS") != _formula_specs or _formula_specs != tuple(
        (label, owner, attribute)
        for label, owner, attribute, _value in _formula_callables
    ):
        raise RuntimeError("candidate proof formula inventory drifted")
    if _namespace.get("_FORMULA_CONSTANT_SPECS") != _constant_specs or _constant_specs != tuple(
        (label, owner, attribute)
        for label, owner, attribute, _value in _formula_constants
    ):
        raise RuntimeError("candidate proof formula constant inventory drifted")
    _callable_records()
    _constant_records()
    for owner, name, expected in _receipt_methods:
        if getattr(owner, name, None) is not expected:
            raise RuntimeError(f"candidate proof receipt method drifted: {owner.__name__}.{name}")
    for name, expected in _local_functions:
        if _namespace.get(name) is not expected:
            raise RuntimeError(f"candidate proof function binding drifted: {name}")
    _transitive[-2][2]()
    _transitive[-1][2]()


_LIVE_BINDING_GUARD.bind(_validate_bindings, globals())


def _make_public_builder(
    implementation: object,
    guard: object,
    local_aliases: tuple[tuple[str, object], ...],
    primitives: tuple[tuple[object, str, object], ...],
    constants: tuple[tuple[str, object], ...],
    formula_callables: tuple[tuple[str, object, str, object], ...],
    formula_constants: tuple[tuple[str, object, str, object], ...],
    local_functions: tuple[tuple[str, object], ...],
    receipt_methods: tuple[tuple[type, str, object], ...],
    bridge_module: object,
    bridge_receipt_type: type,
    implementation_fingerprint: str,
    factory: object,
    namespace: dict[str, object],
):
    def immutable_guard() -> None:
        if namespace.get("_IMPORT_LOCAL_ALIASES") is not local_aliases:
            raise RuntimeError("candidate proof local-alias table drifted")
        if namespace.get("_IMPORT_PRIMITIVES") is not primitives:
            raise RuntimeError("candidate proof primitive table drifted")
        if namespace.get("_IMPORT_CONSTANTS") is not constants:
            raise RuntimeError("candidate proof constant table drifted")
        if namespace.get("_IMPORT_FORMULA_CALLABLES") is not formula_callables:
            raise RuntimeError("candidate proof formula table drifted")
        if namespace.get("_IMPORT_FORMULA_CONSTANTS") is not formula_constants:
            raise RuntimeError("candidate proof formula-constant table drifted")
        if namespace.get("_IMPORT_LOCAL_FUNCTIONS") is not local_functions:
            raise RuntimeError("candidate proof function table drifted")
        if namespace.get("_IMPORT_RECEIPT_METHODS") is not receipt_methods:
            raise RuntimeError("candidate proof receipt-method table drifted")
        if namespace.get("_IMPORT_IMPLEMENTATION_FINGERPRINT") != implementation_fingerprint:
            raise RuntimeError("candidate proof import fingerprint drifted")
        if (
            getattr(
                bridge_module,
                "Vituri2024Q003SameFunctionalStructuralReceipt",
                None,
            )
            is not bridge_receipt_type
        ):
            raise RuntimeError("candidate proof bridge receipt type binding drifted")
        for name, expected in local_aliases:
            if namespace.get(name) is not expected:
                raise RuntimeError(f"candidate proof local alias drifted: {name}")
        for owner, attribute, expected in primitives:
            if getattr(owner, attribute, None) is not expected:
                raise RuntimeError(f"candidate proof primitive binding drifted: {attribute}")
        for name, expected in constants:
            if namespace.get(name) != expected:
                raise RuntimeError(f"candidate proof constant drifted: {name}")
        for _label, owner, attribute, expected in formula_callables:
            if getattr(owner, attribute, None) is not expected:
                raise RuntimeError(f"candidate proof formula binding drifted: {attribute}")
        for _label, owner, attribute, expected in formula_constants:
            if getattr(owner, attribute, None) != expected:
                raise RuntimeError(f"candidate proof formula constant drifted: {attribute}")
        for name, expected in local_functions:
            if namespace.get(name) is not expected:
                raise RuntimeError(f"candidate proof function binding drifted: {name}")
        for owner, name, expected in receipt_methods:
            if getattr(owner, name, None) is not expected:
                raise RuntimeError(f"candidate proof receipt method drifted: {owner.__name__}.{name}")
        if namespace.get("_validate_bindings") is not guard:
            raise RuntimeError("candidate proof guard binding drifted")
        if namespace.get("build_vituri2024_q003_same_functional_candidate_proof_receipt") is not public_builder:
            raise RuntimeError("candidate proof public builder binding drifted")
        guard()

    def public_builder(
        *,
        context_3307: object,
        context_40fd: object,
        same_functional_bridge_receipt: object,
    ) -> Vituri2024Q003SameFunctionalCandidateProofReceipt:
        immutable_guard()
        result = implementation(
            context_3307=context_3307,
            context_40fd=context_40fd,
            same_functional_bridge_receipt=same_functional_bridge_receipt,
            _factory=factory,
            _bridge_receipt_type=bridge_receipt_type,
        )
        immutable_guard()
        return result

    return public_builder


build_vituri2024_q003_same_functional_candidate_proof_receipt = _make_public_builder(
    _build_candidate_proof_impl,
    _validate_bindings,
    _IMPORT_LOCAL_ALIASES,
    _IMPORT_PRIMITIVES,
    _IMPORT_CONSTANTS,
    _IMPORT_FORMULA_CALLABLES,
    _IMPORT_FORMULA_CONSTANTS,
    _IMPORT_LOCAL_FUNCTIONS,
    _IMPORT_RECEIPT_METHODS,
    _bridge_module,
    _bridge_module.Vituri2024Q003SameFunctionalStructuralReceipt,
    _IMPORT_IMPLEMENTATION_FINGERPRINT,
    _construct_receipt,
    globals(),
)
del _construct_receipt


__all__ = [
    "VITURI2024_Q003_CANONICAL_ROOT",
    "VITURI2024_Q003_COMMON_FORMULA_INPUTS",
    "VITURI2024_Q003_HESSIAN_IDENTITY_CANDIDATE",
    "VITURI2024_Q003_NORMALIZATION",
    "VITURI2024_Q003_REAL_ADJOINT_THEOREM",
    "VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_API_VERSION",
    "VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_AUTHORITY",
    "VITURI2024_Q003_SAME_FUNCTIONAL_PROOF_SCOPE",
    "VITURI2024_Q003_TRANSITION_ORDER",
    "build_vituri2024_q003_same_functional_candidate_proof_receipt",
    "vituri2024_q003_same_functional_proof_implementation_fingerprint",
]
