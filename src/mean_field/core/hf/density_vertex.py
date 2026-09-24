"""System-agnostic sparse density-vertex Hartree--Fock algebra.

The canonical input density is the ket-oriented block matrix
``P[k, a, b] = <c_b^dagger c_a>``.  Interaction vertices are sparse one-body
operators ``Gamma_L`` between momentum blocks.  The optimized Hartree/Fock
routines contract these vertices directly and never assemble a dense
four-index interaction tensor.

The literal four-index routines are deliberately bounded test oracles.  They
exist to validate the sparse contractions on tiny systems, not as a production
implementation.  Physical systems remain responsible for constructing their
vertices, labels, weights, screening choices, and reference provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from numbers import Integral
import threading
from typing import Literal, Sequence
import weakref

import numpy as np

Array = np.ndarray
ReferencePolicy = Literal["absolute", "subtract_explicit"]
ZeroModePolicy = Literal["include_supplied", "background_removed"]

LITERAL_WICK_MAX_ORBITALS = 16


@dataclass(frozen=True)
class SparseDensityVertex:
    """Sparse ``Gamma_L`` on momentum blocks with a supplied coefficient."""

    label: tuple[int, int]
    target_k: Array
    source_k: Array
    rows: Array
    columns: Array
    values: Array
    weight: float
    provenance: str

    def __post_init__(self) -> None:
        if (
            len(self.label) != 2
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in self.label
            )
        ):
            raise TypeError("density vertex label must contain two exact integers")
        raw_indices = (self.target_k, self.source_k, self.rows, self.columns)
        for raw in raw_indices:
            probe = np.asarray(raw)
            if probe.dtype.kind not in "iu" or probe.dtype.kind == "b":
                raise TypeError("density vertex indices must be exact integer arrays")
        arrays = [np.asarray(value, dtype=np.int64).reshape(-1) for value in raw_indices]
        values = np.asarray(self.values, dtype=np.complex128).reshape(-1)
        if (
            not arrays[0].size
            or any(array.size != arrays[0].size for array in arrays[1:])
            or values.size != arrays[0].size
        ):
            raise ValueError("density vertex arrays must be nonempty and equally sized")
        records = np.column_stack(arrays)
        if np.unique(records, axis=0).shape[0] != records.shape[0]:
            raise ValueError("duplicate Gamma_L entries are not permitted")
        if not np.all(np.isfinite(values)) or not math.isfinite(float(self.weight)):
            raise ValueError("density vertex values and weight must be finite")
        if not str(self.provenance).strip():
            raise ValueError("density vertex provenance must be explicit")
        object.__setattr__(self, "label", (int(self.label[0]), int(self.label[1])))
        for name, value in zip(
            ("target_k", "source_k", "rows", "columns"), arrays, strict=True
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "provenance", str(self.provenance))

    @property
    def reverse_label(self) -> tuple[int, int]:
        return -self.label[0], -self.label[1]


def identity_density_vertex(
    *, nk: int, dimension: int, weight: float, provenance: str
) -> SparseDensityVertex:
    """Build the exact identity vertex ``Gamma_0`` on every momentum block."""

    k = np.repeat(np.arange(int(nk), dtype=np.int64), int(dimension))
    orbital = np.tile(np.arange(int(dimension), dtype=np.int64), int(nk))
    return SparseDensityVertex(
        label=(0, 0),
        target_k=k,
        source_k=k.copy(),
        rows=orbital,
        columns=orbital.copy(),
        values=np.ones(k.size, dtype=np.complex128),
        weight=float(weight),
        provenance=provenance,
    )


def reverse_density_vertex(
    vertex: SparseDensityVertex, *, provenance: str
) -> SparseDensityVertex:
    """Build ``Gamma_-L = Gamma_L^dagger`` with the same weight."""

    return SparseDensityVertex(
        label=vertex.reverse_label,
        target_k=vertex.source_k,
        source_k=vertex.target_k,
        rows=vertex.columns,
        columns=vertex.rows,
        values=vertex.values.conjugate(),
        weight=vertex.weight,
        provenance=provenance,
    )


def _sorted_vertex_records(
    vertex: SparseDensityVertex,
    *,
    nk: int,
    dimension: int,
    reverse_adjoint: bool,
) -> tuple[Array, Array]:
    """Return packed record keys and aligned values without Python tuples."""

    if reverse_adjoint:
        target_k = vertex.source_k
        source_k = vertex.target_k
        rows = vertex.columns
        columns = vertex.rows
        values = vertex.values.conjugate()
    else:
        target_k = vertex.target_k
        source_k = vertex.source_k
        rows = vertex.rows
        columns = vertex.columns
        values = vertex.values
    keys = (
        ((target_k * int(nk) + source_k) * int(dimension) + rows)
        * int(dimension)
        + columns
    )
    order = np.argsort(keys)
    return keys[order], values[order]


def validate_density_vertex_family(
    vertices: Sequence[SparseDensityVertex],
    *,
    nk: int,
    dimension: int,
    closure_tolerance: float,
    zero_mode_policy: ZeroModePolicy,
) -> None:
    """Require ``Gamma_0=I``, reverse closure, and injective observed K mappings.

    Physical-system adapters must additionally validate label-to-momentum
    semantics and completeness of the projected vertex inventory.
    """

    items = tuple(vertices)
    by_label = {vertex.label: vertex for vertex in items}
    if len(by_label) != len(items):
        raise ValueError("Gamma_L labels must be unique")
    if (0, 0) not in by_label:
        raise ValueError("density vertex family must contain Gamma0=I")
    tolerance = float(closure_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("closure_tolerance must be finite and nonnegative")
    if zero_mode_policy not in {"include_supplied", "background_removed"}:
        raise ValueError("unsupported zero-mode policy")

    if isinstance(nk, (bool, np.bool_)) or not isinstance(nk, Integral) or nk <= 0:
        raise TypeError("nk must be an exact positive integer")
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, Integral)
        or dimension <= 0
    ):
        raise TypeError("dimension must be an exact positive integer")
    nk = int(nk)
    dimension = int(dimension)
    for vertex in items:
        if np.any(vertex.target_k < 0) or np.any(vertex.target_k >= nk):
            raise ValueError("Gamma_L target K index is outside the mesh")
        if np.any(vertex.source_k < 0) or np.any(vertex.source_k >= nk):
            raise ValueError("Gamma_L source K index is outside the mesh")
        if np.any(vertex.rows < 0) or np.any(vertex.rows >= dimension):
            raise ValueError("Gamma_L row index is outside the basis")
        if np.any(vertex.columns < 0) or np.any(vertex.columns >= dimension):
            raise ValueError("Gamma_L column index is outside the basis")
        diagonal_flags = vertex.target_k == vertex.source_k
        if np.any(diagonal_flags) and not np.all(diagonal_flags):
            raise ValueError(
                "one Gamma_L label cannot mix diagonal and off-diagonal K mappings"
            )
        pair_codes = np.unique(vertex.target_k * nk + vertex.source_k)
        pair_targets = pair_codes // nk
        pair_sources = pair_codes % nk
        if np.unique(pair_sources).size != pair_sources.size:
            raise ValueError("one Gamma_L source K maps to multiple target K blocks")
        if np.unique(pair_targets).size != pair_targets.size:
            raise ValueError(
                "one Gamma_L target K receives multiple source K blocks"
            )
        reverse = by_label.get(vertex.reverse_label)
        if reverse is None:
            raise ValueError(f"Gamma_L family lacks reverse label {vertex.reverse_label}")
        if abs(vertex.weight - reverse.weight) > tolerance:
            raise ValueError("reverse Gamma_L weights differ")
        if vertex.label > vertex.reverse_label:
            continue
        expected_keys, expected_values = _sorted_vertex_records(
            vertex,
            nk=nk,
            dimension=dimension,
            reverse_adjoint=True,
        )
        actual_keys, actual_values = _sorted_vertex_records(
            reverse,
            nk=nk,
            dimension=dimension,
            reverse_adjoint=False,
        )
        if (
            not np.array_equal(expected_keys, actual_keys)
            or np.any(np.abs(expected_values - actual_values) > tolerance)
        ):
            raise ValueError("Gamma_-L is not Gamma_L^dagger")

    identity = by_label[(0, 0)]
    identity_orbitals = identity.target_k * dimension + identity.rows
    if (
        identity.values.size != nk * dimension
        or not np.array_equal(identity.target_k, identity.source_k)
        or not np.array_equal(identity.rows, identity.columns)
        or not np.array_equal(identity.values, np.ones(identity.values.size))
        or not np.array_equal(
            np.sort(identity_orbitals), np.arange(nk * dimension)
        )
    ):
        raise ValueError("Gamma0 must equal the exact identity on every K block")
    if zero_mode_policy == "background_removed" and by_label[(0, 0)].weight != 0.0:
        raise ValueError("background_removed requires an explicit zero Gamma0 weight")


def _immutable_array_copy(array: Array, *, dtype: np.dtype) -> Array:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )


def _immutable_vertex_copy(vertex: SparseDensityVertex) -> SparseDensityVertex:
    return SparseDensityVertex(
        label=vertex.label,
        target_k=_immutable_array_copy(vertex.target_k, dtype=np.dtype(np.int64)),
        source_k=_immutable_array_copy(vertex.source_k, dtype=np.dtype(np.int64)),
        rows=_immutable_array_copy(vertex.rows, dtype=np.dtype(np.int64)),
        columns=_immutable_array_copy(vertex.columns, dtype=np.dtype(np.int64)),
        values=_immutable_array_copy(vertex.values, dtype=np.dtype(np.complex128)),
        weight=vertex.weight,
        provenance=vertex.provenance,
    )


@dataclass(frozen=True)
class _PreparedFockGroup:
    """Immutable one-source/one-target sparse Fock contraction plan."""

    vertex_index: int
    target_k: int
    source_k: int
    entry_indices: Array
    left_entry_indices_by_row_stripe: tuple[Array, ...]
    weight: float
    rows_unique: bool


_PREPARED_DENSITY_VERTEX_FACTORY_TOKEN = object()
_PREPARED_DENSITY_VERTEX_REGISTRY_LOCK = threading.RLock()
_PREPARED_DENSITY_VERTEX_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType["PreparedDensityVertexFamily"],
        "PreparedDensityVertexFamily",
    ],
] = {}


@dataclass(frozen=True, init=False)
class PreparedDensityVertexFamily:
    """Validated immutable vertex family; construct via the preparation helper."""

    vertices: tuple[SparseDensityVertex, ...]
    nk: int
    dimension: int
    closure_tolerance: float
    zero_mode_policy: ZeroModePolicy
    reverse_vertex_indices: tuple[int, ...]
    diagonal_entry_indices: tuple[Array, ...]
    fock_groups_by_target: tuple[tuple[_PreparedFockGroup, ...], ...]
    fock_output_row_stripes: int
    fock_work_unit_count: int
    storage_nbytes: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "PreparedDensityVertexFamily must be created by "
            "prepare_density_vertex_family"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("PreparedDensityVertexFamily does not permit subclasses")

    def __copy__(self) -> "PreparedDensityVertexFamily":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "PreparedDensityVertexFamily":
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("PreparedDensityVertexFamily does not permit pickling")


def prepare_density_vertex_family(
    vertices: Sequence[SparseDensityVertex],
    *,
    nk: int,
    dimension: int,
    closure_tolerance: float,
    zero_mode_policy: ZeroModePolicy,
    fock_output_row_stripes: int = 1,
) -> PreparedDensityVertexFamily:
    """Validate, defensively copy, and freeze sparse arrays for repeated SCF use."""

    if isinstance(nk, (bool, np.bool_)) or not isinstance(nk, Integral) or nk <= 0:
        raise TypeError("nk must be an exact positive integer")
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, Integral)
        or dimension <= 0
    ):
        raise TypeError("dimension must be an exact positive integer")
    tolerance = float(closure_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("closure_tolerance must be finite and nonnegative")
    if (
        type(fock_output_row_stripes) is not int
        or fock_output_row_stripes <= 0
        or fock_output_row_stripes > int(dimension)
    ):
        raise TypeError(
            "fock_output_row_stripes must be an exact positive integer "
            "not exceeding dimension"
        )
    source_items = tuple(vertices)
    items = tuple(_immutable_vertex_copy(vertex) for vertex in source_items)
    validate_density_vertex_family(
        items,
        nk=int(nk),
        dimension=int(dimension),
        closure_tolerance=tolerance,
        zero_mode_policy=zero_mode_policy,
    )
    by_label = {vertex.label: index for index, vertex in enumerate(items)}
    reverse_vertex_indices = tuple(
        by_label[vertex.reverse_label] for vertex in items
    )
    diagonal_entry_indices = tuple(
        _immutable_array_copy(
            np.flatnonzero(vertex.target_k == vertex.source_k),
            dtype=np.dtype(np.int64),
        )
        for vertex in items
    )
    groups_by_target: list[list[_PreparedFockGroup]] = [
        [] for _ in range(int(nk))
    ]
    for vertex_index, vertex in enumerate(items):
        pair_codes = np.unique(vertex.target_k * int(nk) + vertex.source_k)
        for pair_code in pair_codes:
            target = int(pair_code // int(nk))
            source = int(pair_code % int(nk))
            selected = _immutable_array_copy(
                np.flatnonzero(
                    (vertex.target_k == target) & (vertex.source_k == source)
                ),
                dtype=np.dtype(np.int64),
            )
            rows = vertex.rows[selected]
            stripe_ids = np.minimum(
                rows * int(fock_output_row_stripes) // int(dimension),
                int(fock_output_row_stripes) - 1,
            )
            if fock_output_row_stripes == 1:
                left_by_stripe = (selected,)
            else:
                left_by_stripe = tuple(
                    _immutable_array_copy(
                        selected[stripe_ids == stripe],
                        dtype=np.dtype(np.int64),
                    )
                    for stripe in range(int(fock_output_row_stripes))
                )
            groups_by_target[int(target)].append(
                _PreparedFockGroup(
                    vertex_index=vertex_index,
                    target_k=int(target),
                    source_k=int(source),
                    entry_indices=selected,
                    left_entry_indices_by_row_stripe=left_by_stripe,
                    weight=float(vertex.weight),
                    rows_unique=np.unique(rows).size == rows.size,
                )
            )
    prepared = object.__new__(PreparedDensityVertexFamily)
    object.__setattr__(
        prepared,
        "_factory_token",
        _PREPARED_DENSITY_VERTEX_FACTORY_TOKEN,
    )
    object.__setattr__(prepared, "vertices", items)
    object.__setattr__(prepared, "nk", int(nk))
    object.__setattr__(prepared, "dimension", int(dimension))
    object.__setattr__(prepared, "closure_tolerance", tolerance)
    object.__setattr__(prepared, "zero_mode_policy", zero_mode_policy)
    object.__setattr__(
        prepared, "reverse_vertex_indices", reverse_vertex_indices
    )
    object.__setattr__(
        prepared, "diagonal_entry_indices", diagonal_entry_indices
    )
    frozen_groups = tuple(tuple(groups) for groups in groups_by_target)
    object.__setattr__(prepared, "fock_groups_by_target", frozen_groups)
    object.__setattr__(
        prepared, "fock_output_row_stripes", int(fock_output_row_stripes)
    )
    object.__setattr__(
        prepared,
        "fock_work_unit_count",
        int(nk) * int(fock_output_row_stripes),
    )
    vertex_storage = sum(
        array.nbytes
        for vertex in items
        for array in (
            vertex.target_k,
            vertex.source_k,
            vertex.rows,
            vertex.columns,
            vertex.values,
        )
    )
    index_arrays = [*diagonal_entry_indices]
    for groups in frozen_groups:
        for group in groups:
            index_arrays.append(group.entry_indices)
            index_arrays.extend(group.left_entry_indices_by_row_stripe)
    seen_index_arrays: set[int] = set()
    index_storage = 0
    for array in index_arrays:
        identity = id(array)
        if identity not in seen_index_arrays:
            seen_index_arrays.add(identity)
            index_storage += array.nbytes
    object.__setattr__(prepared, "storage_nbytes", vertex_storage + index_storage)

    # Keep the execution plan outside the caller-visible object.  Frozen
    # dataclasses can still be modified through object.__setattr__; execution
    # therefore resolves to this private shadow rather than rereading fields
    # from the public capability object.  The public inspection shadow must not
    # share nested vertex/group instances with the trusted plan, because those
    # frozen instances can also be modified through object.__setattr__.
    public_values = dict(vars(prepared))
    public_values["vertices"] = tuple(
        _immutable_vertex_copy(vertex) for vertex in prepared.vertices
    )
    public_values["diagonal_entry_indices"] = tuple(
        _immutable_array_copy(indices, dtype=np.dtype(np.int64))
        for indices in prepared.diagonal_entry_indices
    )
    public_values["fock_groups_by_target"] = tuple(
        tuple(
            _PreparedFockGroup(
                vertex_index=group.vertex_index,
                target_k=group.target_k,
                source_k=group.source_k,
                entry_indices=_immutable_array_copy(
                    group.entry_indices, dtype=np.dtype(np.int64)
                ),
                left_entry_indices_by_row_stripe=tuple(
                    _immutable_array_copy(indices, dtype=np.dtype(np.int64))
                    for indices in group.left_entry_indices_by_row_stripe
                ),
                weight=group.weight,
                rows_unique=group.rows_unique,
            )
            for group in groups
        )
        for groups in prepared.fock_groups_by_target
    )
    public = object.__new__(PreparedDensityVertexFamily)
    for name, value in public_values.items():
        object.__setattr__(public, name, value)
    registry_key = id(public)

    def remove_registry_entry(reference: object, *, key: int = registry_key) -> None:
        with _PREPARED_DENSITY_VERTEX_REGISTRY_LOCK:
            current = _PREPARED_DENSITY_VERTEX_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _PREPARED_DENSITY_VERTEX_REGISTRY.pop(key, None)

    reference = weakref.ref(public, remove_registry_entry)
    with _PREPARED_DENSITY_VERTEX_REGISTRY_LOCK:
        _PREPARED_DENSITY_VERTEX_REGISTRY[registry_key] = (reference, prepared)
    return public


@dataclass(frozen=True)
class MicroscopicInteractionSpec:
    """Explicit Hartree/Fock and normal-order choices, with no defaults."""

    include_hartree: bool
    include_fock: bool
    hartree_reference_policy: ReferencePolicy
    fock_reference_policy: ReferencePolicy
    zero_mode_policy: ZeroModePolicy
    closure_tolerance: float
    normalization_provenance: str
    unresolved_physical_choices: tuple[str, ...]

    def validate(self, *, require_resolved: bool) -> None:
        if type(self.include_hartree) is not bool or type(self.include_fock) is not bool:
            raise TypeError("include_hartree/include_fock must be explicit bools")
        if self.hartree_reference_policy not in {"absolute", "subtract_explicit"}:
            raise ValueError("unsupported Hartree reference policy")
        if self.fock_reference_policy not in {"absolute", "subtract_explicit"}:
            raise ValueError("unsupported Fock reference policy")
        if self.zero_mode_policy not in {"include_supplied", "background_removed"}:
            raise ValueError("unsupported zero-mode policy")
        if (
            not math.isfinite(float(self.closure_tolerance))
            or float(self.closure_tolerance) < 0.0
        ):
            raise ValueError("closure_tolerance must be finite and nonnegative")
        if not str(self.normalization_provenance).strip():
            raise ValueError("interaction normalization provenance must be explicit")
        if require_resolved and self.unresolved_physical_choices:
            raise ValueError(
                "interaction spec has unresolved physical choices: "
                + "; ".join(self.unresolved_physical_choices)
            )


@dataclass(frozen=True)
class InteractionReferences:
    """References required or forbidden according to each explicit policy."""

    hartree: Array | None
    fock: Array | None
    provenance: str


@dataclass(frozen=True)
class InteractionAction:
    """Separated and total sparse density-vertex mean-field actions."""

    hartree: Array
    fock: Array
    total: Array


@dataclass(frozen=True)
class InteractionEnergyEvaluation:
    """One evaluated action and the matching reference-aware scalar energy."""

    action: InteractionAction
    energy: float


# Descriptive aliases for callers that prefer density-vertex-specific names.
DensityVertexInteractionSpec = MicroscopicInteractionSpec
DensityVertexReferenceSpec = InteractionReferences
DensityVertexHartreeFockAction = InteractionAction
DensityVertexInteractionEnergyEvaluation = InteractionEnergyEvaluation


def _resolve_reference(
    projector: Array,
    reference: Array | None,
    *,
    policy: ReferencePolicy,
    name: str,
) -> Array:
    if policy == "absolute":
        if reference is not None:
            raise ValueError(f"{name} reference must be None for policy='absolute'")
        return projector
    if reference is None:
        raise ValueError(f"{name} reference is required for subtract_explicit")
    resolved = np.asarray(reference, dtype=np.complex128)
    if resolved.shape != projector.shape or not np.all(np.isfinite(resolved)):
        raise ValueError(f"{name} reference must be finite and match projector shape")
    return projector - resolved


def _vertex_trace(vertex: SparseDensityVertex, density: Array) -> complex:
    total = 0.0 + 0.0j
    diagonal = vertex.target_k == vertex.source_k
    for k, row, column, value in zip(
        vertex.target_k[diagonal],
        vertex.rows[diagonal],
        vertex.columns[diagonal],
        vertex.values[diagonal],
        strict=True,
    ):
        total += value * density[int(k), int(column), int(row)]
    return complex(total)


def _add_sparse_block(
    out: Array, vertex: SparseDensityVertex, coefficient: complex
) -> None:
    diagonal = vertex.target_k == vertex.source_k
    np.add.at(
        out,
        (
            vertex.target_k[diagonal],
            vertex.rows[diagonal],
            vertex.columns[diagonal],
        ),
        coefficient * vertex.values[diagonal],
    )


def _add_fock(
    out: Array, vertex: SparseDensityVertex, density: Array, coefficient: float
) -> None:
    pairs = sorted(
        set(zip(vertex.target_k.tolist(), vertex.source_k.tolist(), strict=True))
    )
    for target, source in pairs:
        selected = (vertex.target_k == target) & (vertex.source_k == source)
        rows = vertex.rows[selected]
        columns = vertex.columns[selected]
        values = vertex.values[selected]
        contribution = (
            values[:, None]
            * density[int(source)][columns[:, None], columns[None, :]]
            * values.conjugate()[None, :]
        )
        np.add.at(
            out[int(target)],
            (rows[:, None], rows[None, :]),
            float(coefficient) * contribution,
        )


def _vertex_trace_prepared(
    vertex: SparseDensityVertex, diagonal_indices: Array, density: Array
) -> complex:
    total = 0.0 + 0.0j
    for index in diagonal_indices:
        position = int(index)
        total += vertex.values[position] * density[
            int(vertex.target_k[position]),
            int(vertex.columns[position]),
            int(vertex.rows[position]),
        ]
    return complex(total)


def _add_sparse_block_prepared(
    out: Array,
    vertex: SparseDensityVertex,
    diagonal_indices: Array,
    coefficient: complex,
) -> None:
    np.add.at(
        out,
        (
            vertex.target_k[diagonal_indices],
            vertex.rows[diagonal_indices],
            vertex.columns[diagonal_indices],
        ),
        coefficient * vertex.values[diagonal_indices],
    )


def _add_prepared_fock_group(
    out_block: Array,
    vertex: SparseDensityVertex,
    group: _PreparedFockGroup,
    density: Array,
    *,
    left_positions: Array,
    tile_rows: int,
) -> None:
    if left_positions.size == 0:
        return
    right_positions = group.entry_indices
    right_rows = vertex.rows[right_positions]
    right_columns = vertex.columns[right_positions]
    right_conjugate_values = vertex.values[right_positions].conjugate()
    left_rows = vertex.rows[left_positions]
    left_columns = vertex.columns[left_positions]
    left_values = vertex.values[left_positions]
    for start in range(0, left_rows.size, tile_rows):
        stop = min(start + tile_rows, left_rows.size)
        contribution = (
            left_values[start:stop, None]
            * density[group.source_k][
                left_columns[start:stop, None], right_columns[None, :]
            ]
            * right_conjugate_values[None, :]
        )
        scaled = -group.weight * contribution
        index = (left_rows[start:stop, None], right_rows[None, :])
        if group.rows_unique:
            out_block[index] += scaled
        else:
            np.add.at(out_block, index, scaled)


def _add_prepared_fock_target_row_stripe(
    out: Array,
    target_k: int,
    row_stripe: int,
    groups: tuple[_PreparedFockGroup, ...],
    items: tuple[SparseDensityVertex, ...],
    density: Array,
    *,
    tile_rows: int,
) -> None:
    out_block = out[target_k]
    for group in groups:
        _add_prepared_fock_group(
            out_block,
            items[group.vertex_index],
            group,
            density,
            left_positions=group.left_entry_indices_by_row_stripe[row_stripe],
            tile_rows=tile_rows,
        )

def _validate_projector(projector_ket: Array) -> Array:
    projector = np.asarray(projector_ket, dtype=np.complex128)
    if (
        projector.ndim != 3
        or projector.shape[1] != projector.shape[2]
        or not np.all(np.isfinite(projector))
    ):
        raise ValueError("projector_ket must be finite with shape (nk,nb,nb)")
    return projector


def _resolve_enabled_references(
    projector: Array,
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
) -> tuple[Array, Array]:
    """Resolve only enabled terms; disabled references are deliberately unused."""

    q_h = (
        _resolve_reference(
            projector,
            references.hartree,
            policy=spec.hartree_reference_policy,
            name="Hartree",
        )
        if spec.include_hartree
        else np.zeros_like(projector)
    )
    q_f = (
        _resolve_reference(
            projector,
            references.fock,
            policy=spec.fock_reference_policy,
            name="Fock",
        )
        if spec.include_fock
        else np.zeros_like(projector)
    )
    return q_h, q_f


def _resolve_vertex_execution(
    vertices: Sequence[SparseDensityVertex] | PreparedDensityVertexFamily,
    *,
    nk: int,
    dimension: int,
    spec: MicroscopicInteractionSpec,
) -> tuple[tuple[SparseDensityVertex, ...], PreparedDensityVertexFamily | None]:
    if type(vertices) is PreparedDensityVertexFamily:
        with _PREPARED_DENSITY_VERTEX_REGISTRY_LOCK:
            registered = _PREPARED_DENSITY_VERTEX_REGISTRY.get(id(vertices))
        if registered is None or registered[0]() is not vertices:
            raise TypeError(
                "prepared density-vertex family must be exact and factory-built"
            )
        trusted = registered[1]
        if trusted.nk != nk or trusted.dimension != dimension:
            raise ValueError("prepared density-vertex family shape does not match projector")
        if trusted.zero_mode_policy != spec.zero_mode_policy:
            raise ValueError("prepared density-vertex zero-mode policy mismatch")
        if trusted.closure_tolerance != float(spec.closure_tolerance):
            raise ValueError("prepared density-vertex closure tolerance mismatch")
        return trusted.vertices, trusted

    # Preserve the direct sequence execution path and its contribution order,
    # but close validation/use TOCTOU by validating a defensive immutable
    # snapshot rather than caller-owned arrays.
    items = tuple(_immutable_vertex_copy(vertex) for vertex in tuple(vertices))
    validate_density_vertex_family(
        items,
        nk=nk,
        dimension=dimension,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
    )
    return items, None


def _apply_hartree_fock_action(
    projector: Array,
    items: tuple[SparseDensityVertex, ...],
    prepared: PreparedDensityVertexFamily | None,
    spec: MicroscopicInteractionSpec,
    q_h: Array,
    q_f: Array,
    *,
    target_workers: int,
    fock_tile_rows: int,
) -> InteractionAction:
    hartree = np.zeros_like(projector)
    fock = np.zeros_like(projector)
    if prepared is None:
        by_label = {vertex.label: vertex for vertex in items}
        for vertex in items:
            if spec.include_hartree:
                reverse_charge = _vertex_trace(by_label[vertex.reverse_label], q_h)
                _add_sparse_block(hartree, vertex, vertex.weight * reverse_charge)
            if spec.include_fock:
                _add_fock(fock, vertex, q_f, -vertex.weight)
    else:
        if spec.include_hartree:
            for index, vertex in enumerate(items):
                reverse_index = prepared.reverse_vertex_indices[index]
                reverse_charge = _vertex_trace_prepared(
                    items[reverse_index],
                    prepared.diagonal_entry_indices[reverse_index],
                    q_h,
                )
                _add_sparse_block_prepared(
                    hartree,
                    vertex,
                    prepared.diagonal_entry_indices[index],
                    vertex.weight * reverse_charge,
                )
        if spec.include_fock:
            work_units = tuple(
                (target, row_stripe)
                for target in range(prepared.nk)
                for row_stripe in range(prepared.fock_output_row_stripes)
            )
            if target_workers == 1:
                for target, row_stripe in work_units:
                    _add_prepared_fock_target_row_stripe(
                        fock,
                        target,
                        row_stripe,
                        prepared.fock_groups_by_target[target],
                        items,
                        q_f,
                        tile_rows=fock_tile_rows,
                    )
            else:
                with ThreadPoolExecutor(
                    max_workers=min(target_workers, len(work_units))
                ) as executor:
                    futures = tuple(
                        executor.submit(
                            _add_prepared_fock_target_row_stripe,
                            fock,
                            target,
                            row_stripe,
                            prepared.fock_groups_by_target[target],
                            items,
                            q_f,
                            tile_rows=fock_tile_rows,
                        )
                        for target, row_stripe in work_units
                    )
                    for future in futures:
                        future.result()
    return InteractionAction(hartree=hartree, fock=fock, total=hartree + fock)


def hartree_fock_action(
    projector_ket: Array,
    vertices: Sequence[SparseDensityVertex] | PreparedDensityVertexFamily,
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> InteractionAction:
    """Apply sparse-vertex Hartree/Fock maps without a dense two-body tensor."""

    spec.validate(require_resolved=True)
    projector = _validate_projector(projector_ket)
    if not str(references.provenance).strip():
        raise ValueError("reference provenance must be explicit")
    if type(target_workers) is not int or target_workers <= 0:
        raise TypeError("target_workers must be an exact positive integer")
    if type(fock_tile_rows) is not int or fock_tile_rows <= 0:
        raise TypeError("fock_tile_rows must be an exact positive integer")
    nk, dimension, _ = projector.shape
    items, prepared = _resolve_vertex_execution(
        vertices, nk=nk, dimension=dimension, spec=spec
    )
    q_h, q_f = _resolve_enabled_references(projector, spec, references)
    return _apply_hartree_fock_action(
        projector,
        items,
        prepared,
        spec,
        q_h,
        q_f,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    )


def _block_trace(left: Array, right: Array) -> complex:
    return complex(np.einsum("kab,kba->", left, right, optimize=True))


def hartree_fock_action_and_energy(
    projector_ket: Array,
    h0: Array,
    vertices: Sequence[SparseDensityVertex] | PreparedDensityVertexFamily,
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> InteractionEnergyEvaluation:
    """Evaluate one action and its matching reference-aware scalar energy.

    This fused public path prevents SCF adapters from repeating the expensive
    sparse Hartree/Fock contraction merely to evaluate the scalar functional.
    The separated Hartree and Fock actions are retained because their reference
    policies may differ.
    """

    spec.validate(require_resolved=True)
    projector = _validate_projector(projector_ket)
    bare = np.asarray(h0, dtype=np.complex128)
    if bare.shape != projector.shape:
        raise ValueError("h0 and projector_ket must have the same K-block shape")
    if not str(references.provenance).strip():
        raise ValueError("reference provenance must be explicit")
    if type(target_workers) is not int or target_workers <= 0:
        raise TypeError("target_workers must be an exact positive integer")
    if type(fock_tile_rows) is not int or fock_tile_rows <= 0:
        raise TypeError("fock_tile_rows must be an exact positive integer")
    nk, dimension, _ = projector.shape
    items, prepared = _resolve_vertex_execution(
        vertices, nk=nk, dimension=dimension, spec=spec
    )
    q_h, q_f = _resolve_enabled_references(projector, spec, references)
    action = _apply_hartree_fock_action(
        projector,
        items,
        prepared,
        spec,
        q_h,
        q_f,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    )
    value = _block_trace(bare, projector)
    value += 0.5 * _block_trace(action.hartree, q_h)
    value += 0.5 * _block_trace(action.fock, q_f)
    imaginary_bound = 256.0 * np.finfo(float).eps * max(1.0, abs(value))
    if abs(value.imag) > imaginary_bound:
        raise RuntimeError("Hartree/Fock energy has a non-roundoff imaginary part")
    return InteractionEnergyEvaluation(action=action, energy=float(value.real))


def hartree_fock_energy(
    projector_ket: Array,
    h0: Array,
    vertices: Sequence[SparseDensityVertex] | PreparedDensityVertexFamily,
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> float:
    """Return the scalar whose derivative is ``h0 + Hartree + Fock``."""

    return hartree_fock_action_and_energy(
        projector_ket,
        h0,
        vertices,
        spec,
        references,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    ).energy


def _dense_global_vertex_for_oracle(
    vertex: SparseDensityVertex, *, nk: int, dimension: int
) -> Array:
    out = np.zeros((nk * dimension, nk * dimension), dtype=np.complex128)
    rows = vertex.target_k * dimension + vertex.rows
    columns = vertex.source_k * dimension + vertex.columns
    out[rows, columns] = vertex.values
    return out


def _dense_global_block_diagonal(blocks: Array) -> Array:
    nk, dimension, _ = blocks.shape
    out = np.zeros((nk * dimension, nk * dimension), dtype=np.complex128)
    for k in range(nk):
        sl = slice(k * dimension, (k + 1) * dimension)
        out[sl, sl] = blocks[k]
    return out


def literal_wick_four_index_action(
    projector_ket: Array,
    vertices: Sequence[SparseDensityVertex],
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
    *,
    max_orbitals: int,
) -> InteractionAction:
    """Literal ``i,b,g,j`` Wick-action oracle for tiny systems only."""

    projector = np.asarray(projector_ket, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[1] != projector.shape[2]:
        raise ValueError("projector_ket must have shape (nk,nb,nb)")
    nk, dimension, _ = projector.shape
    total_dimension = nk * dimension
    bound = int(max_orbitals)
    if bound <= 0 or bound > LITERAL_WICK_MAX_ORBITALS:
        raise ValueError("literal oracle bound must lie in [1,16]")
    if total_dimension > bound:
        raise ValueError("literal four-index action exceeds its explicit oracle bound")
    spec.validate(require_resolved=True)
    if not str(references.provenance).strip():
        raise ValueError("reference provenance must be explicit")
    validate_density_vertex_family(
        vertices,
        nk=nk,
        dimension=dimension,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
    )
    q_h_blocks, q_f_blocks = _resolve_enabled_references(
        projector, spec, references
    )
    q_h = _dense_global_block_diagonal(q_h_blocks)
    q_f = _dense_global_block_diagonal(q_f_blocks)
    dense = {
        vertex.label: _dense_global_vertex_for_oracle(
            vertex, nk=nk, dimension=dimension
        )
        for vertex in vertices
    }
    h_h = np.zeros_like(q_h)
    h_f = np.zeros_like(q_f)
    for vertex in vertices:
        left = dense[vertex.label]
        right = dense[vertex.reverse_label]
        for i in range(total_dimension):
            for j in range(total_dimension):
                for b in range(total_dimension):
                    for g in range(total_dimension):
                        if spec.include_hartree:
                            h_h[i, j] += (
                                vertex.weight
                                * left[i, j]
                                * right[b, g]
                                * q_h[g, b]
                            )
                        if spec.include_fock:
                            h_f[i, j] -= (
                                vertex.weight
                                * left[i, g]
                                * right[b, j]
                                * q_f[g, b]
                            )
    block_h = np.empty_like(projector)
    block_f = np.empty_like(projector)
    for k in range(nk):
        sl = slice(k * dimension, (k + 1) * dimension)
        block_h[k] = h_h[sl, sl]
        block_f[k] = h_f[sl, sl]
    offdiag = h_h + h_f - _dense_global_block_diagonal(block_h + block_f)
    if np.max(np.abs(offdiag), initial=0.0) > spec.closure_tolerance:
        raise RuntimeError("literal oracle produced off-diagonal K blocks")
    return InteractionAction(
        hartree=block_h,
        fock=block_f,
        total=block_h + block_f,
    )


def literal_wick_four_index_energy(
    projector_ket: Array,
    vertices: Sequence[SparseDensityVertex],
    spec: MicroscopicInteractionSpec,
    references: InteractionReferences,
    *,
    max_orbitals: int,
) -> float:
    """Literal ``a,b,c,d`` direct-minus-exchange Wick scalar oracle."""

    projector = _validate_projector(projector_ket)
    nk, dimension, _ = projector.shape
    n = nk * dimension
    bound = int(max_orbitals)
    if bound <= 0 or bound > LITERAL_WICK_MAX_ORBITALS:
        raise ValueError("literal oracle bound must lie in [1,16]")
    if n > bound:
        raise ValueError("literal four-index energy exceeds its explicit oracle bound")
    spec.validate(require_resolved=True)
    if not str(references.provenance).strip():
        raise ValueError("reference provenance must be explicit")
    validate_density_vertex_family(
        vertices,
        nk=nk,
        dimension=dimension,
        closure_tolerance=spec.closure_tolerance,
        zero_mode_policy=spec.zero_mode_policy,
    )
    q_h_blocks, q_f_blocks = _resolve_enabled_references(
        projector, spec, references
    )
    q_h = _dense_global_block_diagonal(q_h_blocks)
    q_f = _dense_global_block_diagonal(q_f_blocks)
    dense = {
        vertex.label: _dense_global_vertex_for_oracle(
            vertex, nk=nk, dimension=dimension
        )
        for vertex in vertices
    }
    value = 0.0 + 0.0j
    for vertex in vertices:
        left = dense[vertex.label]
        right = dense[vertex.reverse_label]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        coefficient = (
                            0.5 * vertex.weight * left[a, b] * right[c, d]
                        )
                        if spec.include_hartree:
                            value += coefficient * q_h[b, a] * q_h[d, c]
                        if spec.include_fock:
                            value -= coefficient * q_f[d, a] * q_f[b, c]
    imaginary_bound = 256.0 * np.finfo(float).eps * max(1.0, abs(value))
    if abs(value.imag) > imaginary_bound:
        raise RuntimeError("literal Wick energy has a non-roundoff imaginary part")
    return float(value.real)


__all__ = [
    "DensityVertexHartreeFockAction",
    "DensityVertexInteractionEnergyEvaluation",
    "DensityVertexInteractionSpec",
    "DensityVertexReferenceSpec",
    "LITERAL_WICK_MAX_ORBITALS",
    "PreparedDensityVertexFamily",
    "ReferencePolicy",
    "SparseDensityVertex",
    "ZeroModePolicy",
    "hartree_fock_action",
    "hartree_fock_action_and_energy",
    "hartree_fock_energy",
    "identity_density_vertex",
    "literal_wick_four_index_action",
    "literal_wick_four_index_energy",
    "prepare_density_vertex_family",
    "reverse_density_vertex",
    "validate_density_vertex_family",
]
