"""Draft full-microscopic HTQG 4x1 supercell primitives (Batch A).

This module fixes only the algebraic conventions requested for the small
``4 x 1`` qualification system.  It does *not* choose a Coulomb kernel, a
zero-mode/background prescription, a normal-order reference, a filling, or a
wall profile.  Those inputs have no defaults and must carry provenance.

The canonical ket density is ``P_ab = <c_b^dagger c_a>``.  The existing HF
repository stores its matrix transpose.  The system-agnostic Hartree/Fock
contractions are imported from ``mean_field.core.hf.density_vertex`` and
re-exported here for compatibility.  This module constructs the HTQG physical
plane-wave vertices and retains all basis and integer Q-label conventions.

The direct wall Hamiltonian follows the source-qualified ``wall_operator``
ordering:

``H34 = sum_c (M_c A_c + A_c^dagger M_c^dagger)``.

Here ``A_c`` is the no-displacement layer-4 -> layer-3 edge.  K-prime is built
only as ``H_K(-k)^*``.  The legacy formula
``n1=m+eta*N*g1, n2=eta*N*g2`` defines only the K seed-shell labels; full
Batch-A authority is the fixed reduced-k/valley-resolved support satisfying
``n_T=-n-tr_label_wrap[k]`` exactly.

Evidence inspected for this first draft:
``results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/``
``source_capsule_v13r3_full_parent_wall_spectrum_solver_qualification_``
``mesh18_regular6430/wall_operator.py`` and ``FORMULA_CONTRACT.json``.

The source-bound screened interaction/reference/filling constructor lives in
``microscopic_hf.py``.  This basis/H0 module still has no hidden physical
defaults, and the wall profile remains explicit.  It is infrastructure, not a
self-consistent or paper-result claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import math
import threading
from typing import Literal, Sequence
import weakref

import numpy as np

from mean_field.core.hf.occupations import (
    GlobalOccupationResult,
    global_canonical_occupations,
)
from mean_field.core.hf.density_vertex import (
    DensityVertexHartreeFockAction as InteractionAction,
    DensityVertexInteractionSpec as MicroscopicInteractionSpec,
    DensityVertexReferenceSpec as InteractionReferences,
    LITERAL_WICK_MAX_ORBITALS,
    SparseDensityVertex,
    hartree_fock_action,
    hartree_fock_energy,
    identity_density_vertex,
    literal_wick_four_index_action,
    literal_wick_four_index_energy,
    reverse_density_vertex,
    validate_density_vertex_family,
)
from mean_field.core.supercell import IntegerSupercell

from .hamiltonian import (
    _mdt_factor,
    build_coupling_table,
    build_hamiltonian,
    moire_coupling_matrix,
)
from .lattice import HTQGLattice, dot_2d, hex_shell_indices
from .params import HTQGParams

Array = np.ndarray
FourierConvention = Literal["numpy_fft_exp_minus_iqx"]

BATCH_A_PRIMITIVE_MESH = (8, 4)
BATCH_A_REDUCED_MESH = (2, 4)
BATCH_A_PERIOD = 4
VALLEY_ORDER = (1, -1)
_DENSITY_VERTEX_PAIR_CHUNK_ENTRIES = 262_144


def batch_a_supercell() -> IntegerSupercell:
    """Return the fixed real-space ``diag(4, 1)`` IntegerSupercell."""

    return IntegerSupercell(n11=4, n12=0, n21=0, n22=1)


def _immutable_array_copy(values: Array, *, dtype: np.dtype) -> Array:
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _immutable_int64_copy(values: Array) -> Array:
    return _immutable_array_copy(values, dtype=np.dtype(np.int64))


def _exact_python_index(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python integer")
    return value


def _validate_batch_a_supercell(supercell: IntegerSupercell) -> None:
    if type(supercell) is not IntegerSupercell or any(
        type(getattr(supercell, name)) is not int
        for name in ("n11", "n12", "n21", "n22")
    ):
        raise TypeError(
            "Batch-A supercell entries must be exact Python integers"
        )
    if supercell != batch_a_supercell():
        raise ValueError("Batch-A requires IntegerSupercell diag(4,1)")


def _exact_batch_a_fold_arrays() -> tuple[Array, Array, Array]:
    forward = np.empty((2, 4, 4, 2), dtype=np.int64)
    inverse_reduced = np.full((8, 4, 2), -1, dtype=np.int64)
    inverse_fold = np.full((8, 4), -1, dtype=np.int64)
    for r1 in range(2):
        for r2 in range(4):
            for fold in range(4):
                pair = ((r1 + 2 * fold) % 8, r2)
                forward[r1, r2, fold] = pair
                inverse_reduced[pair] = (r1, r2)
                inverse_fold[pair] = fold
    return forward, inverse_reduced, inverse_fold


@dataclass(frozen=True)
class BatchAFoldMap:
    """Exact ``(K1,K2,fold) <-> (k1,k2)`` finite-torus bijection."""

    supercell: IntegerSupercell
    primitive_mesh: tuple[int, int]
    reduced_mesh: tuple[int, int]
    reduced_fold_to_primitive: Array
    primitive_to_reduced: Array
    primitive_to_fold: Array

    def __post_init__(self) -> None:
        _validate_batch_a_supercell(self.supercell)
        if any(
            type(value) is not int
            for value in tuple(self.primitive_mesh) + tuple(self.reduced_mesh)
        ):
            raise TypeError("Batch-A mesh entries must be exact Python integers")
        if tuple(self.primitive_mesh) != BATCH_A_PRIMITIVE_MESH:
            raise ValueError("Batch-A fold map requires primitive mesh (8,4)")
        if tuple(self.reduced_mesh) != BATCH_A_REDUCED_MESH:
            raise ValueError("Batch-A fold map requires reduced mesh (2,4)")
        expected = _exact_batch_a_fold_arrays()
        supplied = (
            self.reduced_fold_to_primitive,
            self.primitive_to_reduced,
            self.primitive_to_fold,
        )
        names = (
            "reduced_fold_to_primitive",
            "primitive_to_reduced",
            "primitive_to_fold",
        )
        for name, values, canonical in zip(names, supplied, expected, strict=True):
            array = np.asarray(values)
            if array.dtype.kind not in "iu" or not np.array_equal(array, canonical):
                raise ValueError(f"{name} is not the canonical Batch-A fold map")
            object.__setattr__(self, name, _immutable_int64_copy(array))
        object.__setattr__(self, "primitive_mesh", BATCH_A_PRIMITIVE_MESH)
        object.__setattr__(self, "reduced_mesh", BATCH_A_REDUCED_MESH)

    @property
    def area_ratio(self) -> int:
        return int(self.supercell.area_ratio)

    @property
    def primitive_nk(self) -> int:
        return int(np.prod(self.primitive_mesh))

    @property
    def reduced_nk(self) -> int:
        return int(np.prod(self.reduced_mesh))

    def primitive_coordinate(
        self, reduced_coordinate: tuple[int, int], fold: int
    ) -> tuple[int, int]:
        i = _exact_python_index(reduced_coordinate[0], name="reduced_coordinate[0]")
        j = _exact_python_index(reduced_coordinate[1], name="reduced_coordinate[1]")
        fold = _exact_python_index(fold, name="fold")
        if not (0 <= i < self.reduced_mesh[0] and 0 <= j < self.reduced_mesh[1]):
            raise IndexError("reduced coordinate is outside the Batch-A mesh")
        if not 0 <= fold < self.area_ratio:
            raise IndexError("fold is outside the Batch-A fold range")
        pair = self.reduced_fold_to_primitive[i, j, fold]
        return int(pair[0]), int(pair[1])

    def reduced_coordinate_and_fold(
        self, primitive_coordinate: tuple[int, int]
    ) -> tuple[tuple[int, int], int]:
        i = _exact_python_index(primitive_coordinate[0], name="primitive_coordinate[0]")
        j = _exact_python_index(primitive_coordinate[1], name="primitive_coordinate[1]")
        if not (0 <= i < self.primitive_mesh[0] and 0 <= j < self.primitive_mesh[1]):
            raise IndexError("primitive coordinate is outside the Batch-A mesh")
        reduced = self.primitive_to_reduced[i, j]
        return (int(reduced[0]), int(reduced[1])), int(self.primitive_to_fold[i, j])


def _validate_batch_a_fold_map(fold_map: BatchAFoldMap) -> None:
    if type(fold_map) is not BatchAFoldMap:
        raise TypeError("Batch-A consumers require an exact BatchAFoldMap")
    _validate_batch_a_supercell(fold_map.supercell)
    if (
        fold_map.supercell != batch_a_supercell()
        or fold_map.primitive_mesh != BATCH_A_PRIMITIVE_MESH
        or fold_map.reduced_mesh != BATCH_A_REDUCED_MESH
    ):
        raise ValueError("fold map does not carry the exact Batch-A geometry")
    for name, canonical in zip(
        (
            "reduced_fold_to_primitive",
            "primitive_to_reduced",
            "primitive_to_fold",
        ),
        _exact_batch_a_fold_arrays(),
        strict=True,
    ):
        if not np.array_equal(getattr(fold_map, name), canonical):
            raise ValueError(f"{name} is not the canonical Batch-A fold map")


def build_batch_a_fold_map(
    *,
    supercell: IntegerSupercell,
    primitive_mesh: tuple[int, int],
    reduced_mesh: tuple[int, int],
) -> BatchAFoldMap:
    """Build the fixed Batch-A bijection and reject nearby guessed geometries.

    For ``S=diag(4,1)``, ``K=(r1/8)b1+(r2/4)b2`` and fold ``m`` maps to
    primitive mesh coordinate ``(r1+2m mod 8, r2)``.
    """

    _validate_batch_a_supercell(supercell)
    if any(type(value) is not int for value in primitive_mesh + reduced_mesh):
        raise TypeError("Batch-A mesh entries must be exact Python integers")
    primitive = tuple(primitive_mesh)
    reduced = tuple(reduced_mesh)
    if primitive != BATCH_A_PRIMITIVE_MESH:
        raise ValueError("Batch A requires primitive mesh (8,4) exactly")
    if reduced != BATCH_A_REDUCED_MESH:
        raise ValueError("Batch A requires reduced mesh (2,4) exactly")

    forward, inverse_reduced, inverse_fold = _exact_batch_a_fold_arrays()
    return BatchAFoldMap(
        supercell=supercell,
        primitive_mesh=primitive,
        reduced_mesh=reduced,
        reduced_fold_to_primitive=forward,
        primitive_to_reduced=inverse_reduced,
        primitive_to_fold=inverse_fold,
    )


@dataclass(frozen=True)
class MicroscopicBasisLayout:
    """Canonical C-order ``(fold,g,layer,sublattice,valley,spin)`` layout."""

    g_indices: Array
    valley_order: tuple[int, int]
    spin_order: tuple[str, str]
    period_cells: int

    def __post_init__(self) -> None:
        g = np.asarray(self.g_indices)
        if g.dtype.kind not in "iu" or g.ndim != 2 or g.shape[1] != 2 or g.shape[0] == 0:
            raise TypeError("g_indices must be a nonempty integer array of shape (ng,2)")
        g = np.asarray(g, dtype=np.int64)
        if np.unique(g, axis=0).shape[0] != g.shape[0]:
            raise ValueError("g_indices must not contain duplicate reciprocal labels")
        if any(type(value) is not int for value in self.valley_order):
            raise TypeError("valley_order entries must be exact Python integers")
        if tuple(self.valley_order) != VALLEY_ORDER:
            raise ValueError("canonical valley_order must be physical labels (+1,-1)")
        if tuple(self.spin_order) != ("up", "down") or any(
            type(value) is not str for value in self.spin_order
        ):
            raise ValueError("canonical spin_order must be exactly ('up','down')")
        spins = tuple(self.spin_order)
        if type(self.period_cells) is not int:
            raise TypeError("period_cells must be an exact Python integer")
        if self.period_cells != BATCH_A_PERIOD:
            raise ValueError("Batch-A canonical basis requires period_cells=4")
        object.__setattr__(self, "g_indices", _immutable_int64_copy(g))
        object.__setattr__(self, "valley_order", VALLEY_ORDER)
        object.__setattr__(self, "spin_order", spins)
        object.__setattr__(self, "period_cells", BATCH_A_PERIOD)

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        return (BATCH_A_PERIOD, int(self.g_indices.shape[0]), 4, 2, 2, 2)

    @property
    def dimension(self) -> int:
        return int(np.prod(self.shape))

    @property
    def no_fold_dimension(self) -> int:
        return self.dimension // BATCH_A_PERIOD

    def flat_index(
        self,
        fold: int,
        g: int,
        layer: int,
        sublattice: int,
        valley: int,
        spin: int,
    ) -> int:
        labels = tuple(
            _exact_python_index(value, name=name)
            for name, value in (
                ("fold", fold),
                ("g", g),
                ("layer", layer),
                ("sublattice", sublattice),
                ("valley", valley),
                ("spin", spin),
            )
        )
        if any(value < 0 or value >= size for value, size in zip(labels, self.shape, strict=True)):
            raise IndexError(f"basis coordinate {labels} is outside shape {self.shape}")
        return int(np.ravel_multi_index(labels, self.shape, order="C"))

    def unravel_index(self, index: int) -> tuple[int, int, int, int, int, int]:
        index = _exact_python_index(index, name="index")
        if not 0 <= index < self.dimension:
            raise IndexError("flat basis index is outside the canonical layout")
        return tuple(
            int(value)
            for value in np.unravel_index(index, self.shape, order="C")
        )  # type: ignore[return-value]

    def legacy_seed_plane_wave_label(
        self, *, fold: int, g: int, valley: int
    ) -> tuple[int, int]:
        """Return the k-independent seed-shell label only.

        This helper is not the physical-label authority for the reduced-k
        supercell problem.  Consumers must use
        :class:`BatchAPhysicalLabelSupport` with an explicit reduced-k index.
        """

        fold = _exact_python_index(fold, name="fold")
        g = _exact_python_index(g, name="g")
        valley = _exact_python_index(valley, name="valley")
        if not 0 <= fold < BATCH_A_PERIOD:
            raise IndexError("fold is outside [0,4)")
        if not 0 <= g < self.g_indices.shape[0]:
            raise IndexError("g is outside the supplied physical shell")
        if not 0 <= valley < 2:
            raise IndexError("valley index is outside [0,2)")
        eta = int(self.valley_order[valley])
        g1, g2 = (int(value) for value in self.g_indices[g])
        return int(fold) + eta * BATCH_A_PERIOD * g1, eta * BATCH_A_PERIOD * g2


def _exact_batch_a_tr_support_arrays(seed_g_indices: Array) -> tuple[Array, Array, Array]:
    """Return the unique canonical Batch-A labels, TR partners, and wraps."""

    seed = np.asarray(seed_g_indices, dtype=np.int64)
    ng = int(seed.shape[0])
    labels = np.empty((8, 2, 4, ng, 2), dtype=np.int64)
    partner = np.empty(8, dtype=np.int64)
    wrap = np.empty((8, 2), dtype=np.int64)

    for r1 in range(2):
        for r2 in range(4):
            k = r1 * 4 + r2
            rt1 = (-r1) % 2
            rt2 = (-r2) % 4
            partner[k] = rt1 * 4 + rt2
            wrap[k] = ((r1 + rt1) // 2, r2 + rt2)
            for fold in range(4):
                labels[k, 0, fold, :, 0] = fold + 4 * seed[:, 0]
                labels[k, 0, fold, :, 1] = 4 * seed[:, 1]

    for k in range(8):
        kt = int(partner[k])
        transformed = -labels[k, 0].reshape(4 * ng, 2) - wrap[k]
        buckets: list[list[tuple[tuple[int, int], Array]]] = [[] for _ in range(4)]
        for n in transformed:
            fold = int(n[0] % 4)
            parent_label = ((fold - int(n[0])) // 4, -int(n[1]) // 4)
            buckets[fold].append((parent_label, n.copy()))
        for fold, bucket in enumerate(buckets):
            if len(bucket) != ng:
                raise RuntimeError("TR support does not preserve per-fold cardinality")
            bucket.sort(key=lambda item: item[0])
            labels[kt, 1, fold] = np.asarray(
                [item[1] for item in bucket], dtype=np.int64
            )
    return labels, partner, wrap


@dataclass(frozen=True)
class BatchAPhysicalLabelSupport:
    """Fixed-cardinality, reduced-k/valley-resolved physical PW support.

    ``labels`` has shape ``(8,2,4,ng,2)`` in canonical dense slot order.
    ``tr_pw_permutation[k,valley,a]`` maps flattened plane-wave slot ``a`` at
    ``(k,valley)`` to its exact time-reversed slot at
    ``(tr_partner_k[k],1-valley)``. The wrap is in physical-label units and
    enforces ``n_T = -n - tr_label_wrap[k]`` exactly.
    """

    seed_g_indices: Array
    labels: Array
    tr_partner_k: Array
    tr_label_wrap: Array
    tr_pw_permutation: Array
    provenance: str

    def __post_init__(self) -> None:
        seed = np.asarray(self.seed_g_indices)
        labels = np.asarray(self.labels)
        partner = np.asarray(self.tr_partner_k)
        wrap = np.asarray(self.tr_label_wrap)
        permutation = np.asarray(self.tr_pw_permutation)
        if seed.dtype.kind not in "iu" or seed.ndim != 2 or seed.shape[1] != 2:
            raise TypeError("support seed_g_indices must be integer shape (ng,2)")
        ng = int(seed.shape[0])
        if ng == 0 or np.unique(seed, axis=0).shape[0] != ng:
            raise ValueError("support seed_g_indices must be nonempty and unique")
        if labels.dtype.kind not in "iu" or labels.shape != (8, 2, 4, ng, 2):
            raise TypeError("support labels must be integer shape (8,2,4,ng,2)")
        if partner.dtype.kind not in "iu" or partner.shape != (8,):
            raise TypeError("support tr_partner_k must be integer shape (8,)")
        if wrap.dtype.kind not in "iu" or wrap.shape != (8, 2):
            raise TypeError("support tr_label_wrap must be integer shape (8,2)")
        if permutation.dtype.kind not in "iu" or permutation.shape != (8, 2, 4 * ng):
            raise TypeError("support TR permutation must be integer shape (8,2,4*ng)")
        if not str(self.provenance).strip():
            raise ValueError("support provenance must be explicit")

        for k in range(8):
            kt = int(partner[k])
            if not 0 <= kt < 8 or int(partner[kt]) != k:
                raise ValueError("support TR k map must be an involution")
            for valley in range(2):
                flat_labels = labels[k, valley].reshape(4 * ng, 2)
                if np.unique(flat_labels, axis=0).shape[0] != 4 * ng:
                    raise ValueError("support physical labels must be unique per k/valley")
                for fold in range(4):
                    if np.any(np.mod(labels[k, valley, fold, :, 0], 4) != fold):
                        raise ValueError("support fold slots must equal n1 modulo four")
                    if np.any(np.mod(labels[k, valley, fold, :, 1], 4) != 0):
                        raise ValueError("support physical n2 labels must be multiples of four")
                perm = permutation[k, valley]
                if (
                    np.unique(perm).size != 4 * ng
                    or np.any(perm < 0)
                    or np.any(perm >= 4 * ng)
                ):
                    raise ValueError("support TR plane-wave map must be a permutation")
                target = labels[kt, 1 - valley].reshape(4 * ng, 2)[perm]
                if not np.array_equal(target, -flat_labels - wrap[k]):
                    raise ValueError("support labels violate exact time-reversal sewing")
                inverse = permutation[kt, 1 - valley][perm]
                if not np.array_equal(inverse, np.arange(4 * ng, dtype=np.int64)):
                    raise ValueError("support TR plane-wave permutation is not involutive")
                if not np.array_equal(wrap[kt], wrap[k]):
                    raise ValueError("support TR label wrap must agree on partner points")

                parent = np.empty((4, ng, 2), dtype=np.int64)
                eta = VALLEY_ORDER[valley]
                for fold in range(4):
                    parent[fold, :, 0] = (
                        eta * (labels[k, valley, fold, :, 0] - fold) // 4
                    )
                    parent[fold, :, 1] = eta * labels[k, valley, fold, :, 1] // 4
                seed_set = {tuple(int(value) for value in item) for item in seed}
                for fold in range(4):
                    block = parent[fold]
                    sums = np.sum(block, axis=0) - np.sum(seed, axis=0)
                    if np.any(np.mod(sums, ng) != 0):
                        raise ValueError(
                            "support parent block is not an integer translate of the seed"
                        )
                    translation = sums // ng
                    translated = {
                        (g1 + int(translation[0]), g2 + int(translation[1]))
                        for g1, g2 in seed_set
                    }
                    if translated != {
                        tuple(int(value) for value in item) for item in block
                    }:
                        raise ValueError(
                            "support parent block is not a translate of the seed support"
                        )

        expected_labels, expected_partner, expected_wrap = (
            _exact_batch_a_tr_support_arrays(seed)
        )
        if (
            not np.array_equal(partner, expected_partner)
            or not np.array_equal(wrap, expected_wrap)
            or not np.array_equal(labels, expected_labels)
        ):
            raise ValueError(
                "support is not bound to the exact canonical Batch-A reduced-k/TR geometry"
            )

        for name, array in (
            ("seed_g_indices", seed),
            ("labels", labels),
            ("tr_partner_k", partner),
            ("tr_label_wrap", wrap),
            ("tr_pw_permutation", permutation),
        ):
            object.__setattr__(self, name, _immutable_int64_copy(array))
        object.__setattr__(self, "provenance", str(self.provenance))

    @property
    def ng(self) -> int:
        return int(self.seed_g_indices.shape[0])

    def physical_plane_wave_label(
        self, *, reduced_k: int, valley: int, fold: int, g: int
    ) -> tuple[int, int]:
        reduced_k = _exact_python_index(reduced_k, name="reduced_k")
        valley = _exact_python_index(valley, name="valley")
        fold = _exact_python_index(fold, name="fold")
        g = _exact_python_index(g, name="g")
        if not 0 <= reduced_k < 8:
            raise IndexError("reduced_k is outside the Batch-A mesh")
        if not 0 <= valley < 2:
            raise IndexError("valley index is outside [0,2)")
        if not 0 <= fold < 4 or not 0 <= g < self.ng:
            raise IndexError("plane-wave slot is outside the support")
        pair = self.labels[reduced_k, valley, fold, g]
        return int(pair[0]), int(pair[1])

    def parent_g_label(
        self, *, reduced_k: int, valley: int, fold: int, g: int
    ) -> tuple[int, int]:
        """Decode the unbarred parent reciprocal label for one dense slot."""

        n1, n2 = self.physical_plane_wave_label(
            reduced_k=reduced_k, valley=valley, fold=fold, g=g
        )
        eta = VALLEY_ORDER[int(valley)]
        if n1 % 4 != int(fold) or n2 % 4:
            raise RuntimeError("support label is not exactly parent-decodable")
        return eta * ((n1 - int(fold)) // 4), eta * (n2 // 4)

    def parent_g_indices(self, *, reduced_k: int, valley: int, fold: int) -> Array:
        return np.asarray(
            [
                self.parent_g_label(
                    reduced_k=reduced_k, valley=valley, fold=fold, g=g
                )
                for g in range(self.ng)
            ],
            dtype=np.int64,
        )


    @property
    def support_sha256(self) -> str:
        """Deterministic identity of every support-defining field."""

        return batch_a_physical_label_support_sha256(self)

    def parent_translation(
        self, *, reduced_k: int, valley: int, fold: int
    ) -> tuple[int, int]:
        """Return the exact parent-shell translation for one dense block."""

        parent = self.parent_g_indices(
            reduced_k=reduced_k, valley=valley, fold=fold
        )
        sums = np.sum(parent, axis=0) - np.sum(self.seed_g_indices, axis=0)
        if np.any(np.mod(sums, self.ng) != 0):
            raise RuntimeError("support parent translation is not integral")
        translation = sums // self.ng
        return int(translation[0]), int(translation[1])


def _validate_batch_a_physical_label_support(
    support: BatchAPhysicalLabelSupport,
) -> None:
    if type(support) is not BatchAPhysicalLabelSupport:
        raise TypeError(
            "Batch-A consumers require an exact BatchAPhysicalLabelSupport"
        )
    labels, partner, wrap = _exact_batch_a_tr_support_arrays(
        support.seed_g_indices
    )
    permutation = _batch_a_tr_permutation(labels, partner, wrap)
    for name, canonical in (
        ("labels", labels),
        ("tr_partner_k", partner),
        ("tr_label_wrap", wrap),
        ("tr_pw_permutation", permutation),
    ):
        if not np.array_equal(getattr(support, name), canonical):
            raise ValueError(f"support {name} is not canonical for seed_g_indices")


def _validate_batch_a_layout_support(
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
) -> None:
    if type(layout) is not MicroscopicBasisLayout:
        raise TypeError("Batch-A consumers require an exact MicroscopicBasisLayout")
    _validate_batch_a_physical_label_support(support)
    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("physical-label support seed does not match layout.g_indices")


def validate_batch_a_physical_label_support(
    support: BatchAPhysicalLabelSupport,
) -> None:
    """Public fail-closed validation of canonical Batch-A support authority."""

    _validate_batch_a_physical_label_support(support)


def validate_batch_a_layout_support(
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
) -> None:
    """Public fail-closed validation of one canonical layout/support pair."""

    _validate_batch_a_layout_support(layout, support)


def batch_a_physical_label_support_sha256(
    support: BatchAPhysicalLabelSupport,
) -> str:
    """Hash canonical support arrays plus explicit provenance."""

    _validate_batch_a_physical_label_support(support)
    digest = hashlib.sha256()
    for name in (
        "seed_g_indices",
        "labels",
        "tr_partner_k",
        "tr_label_wrap",
        "tr_pw_permutation",
    ):
        values = np.asarray(getattr(support, name), dtype="<i8", order="C")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes(order="C"))
        digest.update(values.tobytes(order="C"))
    digest.update(b"provenance\0")
    digest.update(support.provenance.encode("utf-8"))
    return digest.hexdigest()


def _batch_a_tr_permutation(
    labels: Array,
    partner: Array,
    wrap: Array,
) -> Array:
    ng = int(labels.shape[3])
    permutation = np.empty((8, 2, 4 * ng), dtype=np.int64)
    for k in range(8):
        kt = int(partner[k])
        for valley in range(2):
            lookup = {
                tuple(int(value) for value in n): slot
                for slot, n in enumerate(labels[kt, 1 - valley].reshape(4 * ng, 2))
            }
            for slot, n in enumerate(labels[k, valley].reshape(4 * ng, 2)):
                target = tuple(int(value) for value in (-n - wrap[k]))
                if target not in lookup:
                    raise RuntimeError("constructed support is not closed under time reversal")
                permutation[k, valley, slot] = lookup[target]
    return permutation


def build_batch_a_tr_covariant_support(
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    *,
    provenance: str,
) -> BatchAPhysicalLabelSupport:
    """Build the minimal exact-TR support without enlarging dense HF blocks."""

    _validate_batch_a_fold_map(fold_map)
    if type(layout) is not MicroscopicBasisLayout:
        raise TypeError("TR support requires an exact MicroscopicBasisLayout")
    if not str(provenance).strip():
        raise ValueError("support provenance must be explicit")
    labels, partner, wrap = _exact_batch_a_tr_support_arrays(layout.g_indices)
    permutation = _batch_a_tr_permutation(labels, partner, wrap)

    return BatchAPhysicalLabelSupport(
        seed_g_indices=layout.g_indices,
        labels=labels,
        tr_partner_k=partner,
        tr_label_wrap=wrap,
        tr_pw_permutation=permutation,
        provenance=str(provenance),
    )

@dataclass(frozen=True)
class ProfileFourierInput:
    """Explicit periodic profile samples; no wall shape is assumed here."""

    sample_positions_cells: Array
    values_by_valley: Array
    period_cells: int
    convention: FourierConvention
    provenance: str

    def __post_init__(self) -> None:
        x = np.asarray(self.sample_positions_cells, dtype=np.float64)
        values = np.asarray(self.values_by_valley, dtype=np.complex128)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("profile sample positions must be a nonempty 1D array")
        if values.shape != (2, x.size):
            raise ValueError("values_by_valley must have shape (2, nsample)")
        if type(self.period_cells) is not int:
            raise TypeError("profile period_cells must be an exact Python integer")
        if self.period_cells != BATCH_A_PERIOD:
            raise ValueError("Batch-A profile period must be four cells")
        expected = np.arange(x.size, dtype=float) * BATCH_A_PERIOD / float(x.size)
        if not np.array_equal(x, expected):
            raise ValueError("profile samples must use exact x_j=j*4/Nsample ordering")
        if self.convention != "numpy_fft_exp_minus_iqx":
            raise ValueError("unsupported Fourier convention")
        if not np.all(np.isfinite(values)):
            raise ValueError("profile samples must be finite")
        if not str(self.provenance).strip():
            raise ValueError("profile provenance must be explicit")
        object.__setattr__(
            self,
            "sample_positions_cells",
            _immutable_array_copy(x, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "values_by_valley",
            _immutable_array_copy(values, dtype=np.dtype(np.complex128)),
        )
        object.__setattr__(self, "period_cells", BATCH_A_PERIOD)
        object.__setattr__(self, "provenance", str(self.provenance))


def build_periodic_smoothstep_profile(
    *,
    wall_width_cells: float,
    nsample: int,
    provenance: str,
) -> ProfileFourierInput:
    """Sample the authorized four-cell periodic ABA|ABG double wall.

    ``wall_width_cells`` is the full width of each transition in the dimensionless
    coordinate ``u in [0,4)``.  The ABA plateau (``s_ABG=0``) is centered at
    ``u=1`` and the ABG plateau (``s_ABG=1``) at ``u=3``.  Smoothstep walls are
    centered at ``u=0`` (the periodic seam) and ``u=2``.  The function only
    samples the continuous profile; Fourier alias/convergence checks remain the
    caller's responsibility.
    """

    width = float(wall_width_cells)
    if not math.isfinite(width) or not (0.0 < width < BATCH_A_PERIOD / 2.0):
        raise ValueError("wall_width_cells must lie strictly between 0 and N/2=2")
    if type(nsample) is not int or nsample < 8 or nsample % 2:
        raise ValueError("nsample must be an even exact integer of at least 8")
    if not str(provenance).strip():
        raise ValueError("smoothstep wall provenance must be explicit")

    u = np.arange(nsample, dtype=np.float64) * BATCH_A_PERIOD / float(nsample)

    def smoothstep(t: Array) -> Array:
        clipped = np.clip(t, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    # Start from exact ABA/ABG plateaus and replace only the two transition
    # intervals.  The seam coordinate is the unique representative in [-2,2).
    values = np.where(u < BATCH_A_PERIOD / 2.0, 0.0, 1.0)
    middle_t = (u - (BATCH_A_PERIOD / 2.0 - width / 2.0)) / width
    middle_mask = np.abs(u - BATCH_A_PERIOD / 2.0) <= width / 2.0
    values[middle_mask] = smoothstep(middle_t[middle_mask])

    seam = (u + BATCH_A_PERIOD / 2.0) % BATCH_A_PERIOD - BATCH_A_PERIOD / 2.0
    seam_mask = np.abs(seam) <= width / 2.0
    values[seam_mask] = 1.0 - smoothstep(seam[seam_mask] / width + 0.5)

    return ProfileFourierInput(
        sample_positions_cells=u,
        values_by_valley=np.asarray((values, values), dtype=np.complex128),
        period_cells=BATCH_A_PERIOD,
        convention="numpy_fft_exp_minus_iqx",
        provenance=(
            f"user-authorized N=4 periodic smoothstep double wall; "
            f"full wall_width_cells={width:.17g}; nsample={nsample}; "
            + str(provenance)
        ),
    )


@dataclass(frozen=True)
class SparseOneBodyOperator:
    """COO-like square one-body operator with duplicate-free entries."""

    dimension: int
    rows: Array
    columns: Array
    values: Array
    provenance: str

    def __post_init__(self) -> None:
        if type(self.dimension) is not int:
            raise TypeError("sparse one-body dimension must be an exact Python integer")
        dimension = self.dimension
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        columns = np.asarray(self.columns, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.complex128).reshape(-1)
        if dimension <= 0 or rows.size != columns.size or rows.size != values.size:
            raise ValueError("invalid sparse one-body operator dimensions")
        if np.any(rows < 0) or np.any(rows >= dimension) or np.any(columns < 0) or np.any(columns >= dimension):
            raise ValueError("sparse one-body index is out of range")
        pairs = np.column_stack((rows, columns))
        if np.unique(pairs, axis=0).shape[0] != rows.size:
            raise ValueError("duplicate sparse one-body entries are not permitted")
        if not np.all(np.isfinite(values)):
            raise ValueError("sparse one-body values must be finite")
        if not str(self.provenance).strip():
            raise ValueError("sparse one-body provenance must be explicit")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "rows", _immutable_int64_copy(rows))
        object.__setattr__(self, "columns", _immutable_int64_copy(columns))
        object.__setattr__(
            self,
            "values",
            _immutable_array_copy(values, dtype=np.dtype(np.complex128)),
        )
        object.__setattr__(self, "provenance", str(self.provenance))

    def apply(self, vectors: Array) -> Array:
        x = np.asarray(vectors, dtype=np.complex128)
        one = x.ndim == 1
        matrix = x[:, None] if one else x
        if matrix.ndim != 2 or matrix.shape[0] != self.dimension:
            raise ValueError("one-body operator input has incompatible shape")
        out = np.zeros_like(matrix)
        np.add.at(out, self.rows, self.values[:, None] * matrix[self.columns])
        return out[:, 0] if one else out

    def left_multiply_dense(self, matrix: Array) -> Array:
        dense = np.asarray(matrix, dtype=np.complex128)
        if dense.ndim != 2 or dense.shape[0] != self.dimension:
            raise ValueError("dense right factor has incompatible shape")
        out = np.zeros_like(dense)
        for row, column, value in zip(self.rows, self.columns, self.values, strict=True):
            out[int(row)] += value * dense[int(column)]
        return out

    def to_dense_for_test(self, *, max_dimension: int) -> Array:
        if self.dimension > int(max_dimension):
            raise ValueError("dense one-body materialization exceeds the explicit test bound")
        out = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        out[self.rows, self.columns] = self.values
        return out


def build_profile_fourier_multiplier(
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    profile: ProfileFourierInput,
    *,
    reduced_k: int,
) -> SparseOneBodyOperator:
    """Build projected multiplication using the wall ``(g2,n)`` injection."""

    _validate_batch_a_layout_support(layout, support)
    if profile.period_cells != layout.period_cells:
        raise ValueError("profile and canonical basis periods differ")
    nsample = int(profile.sample_positions_cells.size)
    maximum_span = 0
    for valley in range(2):
        by_g2: dict[int, list[int]] = {}
        for fold in range(BATCH_A_PERIOD):
            for g, (_g1, g2) in enumerate(layout.g_indices):
                n1, n2 = support.physical_plane_wave_label(
                    reduced_k=reduced_k, fold=fold, g=g, valley=valley
                )
                by_g2.setdefault(int(n2), []).append(n1)
        maximum_span = max(
            maximum_span,
            *(max(values) - min(values) for values in by_g2.values()),
        )
    if nsample < 2 * maximum_span + 1:
        raise ValueError(
            "profile sampling aliases physical Fourier differences; require "
            f"at least {2 * maximum_span + 1} samples"
        )
    coefficients = np.fft.fft(profile.values_by_valley, axis=1) / float(nsample)
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []

    # Multiplication preserves g2, layer, sublattice, valley, and spin.  It may
    # couple fold/g1 labels through q=n_out-n_in.
    for valley in range(2):
        states_by_g2: dict[int, list[tuple[int, int, int]]] = {}
        for fold in range(BATCH_A_PERIOD):
            for g, (_g1, g2) in enumerate(layout.g_indices):
                n1, n2 = support.physical_plane_wave_label(
                    reduced_k=reduced_k, fold=fold, g=g, valley=valley
                )
                states_by_g2.setdefault(int(n2), []).append((fold, g, n1))
        for states in states_by_g2.values():
            if len({n1 for _fold, _g, n1 in states}) != len(states):
                raise RuntimeError("profile (g2,n) injection is not one-to-one")
            for layer in range(4):
                for sublattice in range(2):
                    for spin in range(2):
                        for fold_out, g_out, n_out in states:
                            row = layout.flat_index(
                                fold_out, g_out, layer, sublattice, valley, spin
                            )
                            for fold_in, g_in, n_in in states:
                                column = layout.flat_index(
                                    fold_in, g_in, layer, sublattice, valley, spin
                                )
                                value = coefficients[valley, (n_out - n_in) % nsample]
                                if value != 0.0:
                                    rows.append(row)
                                    columns.append(column)
                                    values.append(complex(value))
    return SparseOneBodyOperator(
        dimension=layout.dimension,
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        values=np.asarray(values, dtype=np.complex128),
        provenance=f"profile Fourier multiplier: {profile.provenance}",
    )


def _direct_dft_profile_entries(
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    profile: ProfileFourierInput,
    *,
    reduced_k: int,
) -> tuple[Array, Array, Array]:
    """Independent direct-DFT matrix-element oracle for a profile multiplier."""

    _validate_batch_a_layout_support(layout, support)
    if profile.period_cells != layout.period_cells:
        raise ValueError("profile and canonical basis periods differ")
    nsample = int(profile.sample_positions_cells.size)
    sample_indices = np.arange(nsample, dtype=float)
    coefficient_cache: dict[tuple[int, int], complex] = {}

    def coefficient(valley: int, delta_n: int) -> complex:
        key = (int(valley), int(delta_n))
        if key not in coefficient_cache:
            phase = np.exp(-2.0j * np.pi * sample_indices * delta_n / nsample)
            coefficient_cache[key] = complex(
                np.dot(profile.values_by_valley[valley], phase) / nsample
            )
        return coefficient_cache[key]

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for row in range(layout.dimension):
        fold_out, g_out, layer_out, sub_out, valley_out, spin_out = (
            layout.unravel_index(row)
        )
        n_out, n2_out = support.physical_plane_wave_label(
            reduced_k=reduced_k, fold=fold_out, g=g_out, valley=valley_out
        )
        for column in range(layout.dimension):
            fold_in, g_in, layer_in, sub_in, valley_in, spin_in = (
                layout.unravel_index(column)
            )
            if (
                layer_out != layer_in
                or sub_out != sub_in
                or valley_out != valley_in
                or spin_out != spin_in
            ):
                continue
            n_in, n2_in = support.physical_plane_wave_label(
                reduced_k=reduced_k, fold=fold_in, g=g_in, valley=valley_in
            )
            if n2_out != n2_in:
                continue
            value = coefficient(valley_out, n_out - n_in)
            if value != 0.0:
                rows.append(row)
                columns.append(column)
                values.append(value)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(values, dtype=np.complex128),
    )


def _multiply_profile_entries(
    matrix: Array,
    rows: Array,
    columns: Array,
    values: Array,
    *,
    ordering: Literal["left", "right"],
) -> Array:
    """Apply independently generated profile entries without sparse helper reuse."""

    operand = np.asarray(matrix, dtype=np.complex128)
    out = np.zeros_like(operand)
    if ordering == "left":
        for row, column, value in zip(rows, columns, values, strict=True):
            out[int(row)] += value * operand[int(column)]
        return out
    if ordering == "right":
        for row, column, value in zip(rows, columns, values, strict=True):
            out[:, int(column)] += operand[:, int(row)] * value
        return out
    raise ValueError("ordering must be 'left' or 'right'")


def ket_blocks_to_repository_stored(projector_ket: Array) -> Array:
    """Map ``(K,a,b)`` ket ``P_ab`` to stored ``(b,a,K)`` transpose."""

    ket = np.asarray(projector_ket, dtype=np.complex128)
    if ket.ndim != 3 or ket.shape[1] != ket.shape[2]:
        raise ValueError("ket projector must have shape (nk,nb,nb)")
    return np.transpose(ket, (2, 1, 0)).copy()


def repository_stored_to_ket_blocks(stored: Array) -> Array:
    """Inverse of :func:`ket_blocks_to_repository_stored` (no conjugation)."""

    values = np.asarray(stored, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("repository stored density must have shape (nb,nb,nk)")
    return np.transpose(values, (2, 1, 0)).copy()


def ket_hamiltonian_to_repository_blocks(hamiltonian_ket: Array) -> Array:
    """Move K last without transposing Hamiltonian matrix indices."""

    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[1] != hamiltonian.shape[2]:
        raise ValueError("ket Hamiltonian must have shape (nk,nb,nb)")
    return np.transpose(hamiltonian, (1, 2, 0)).copy()


def repository_hamiltonian_to_ket_blocks(hamiltonian: Array) -> Array:
    """Move a framework Hamiltonian ``(nb,nb,nk)`` to K-block layout."""

    values = np.asarray(hamiltonian, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("repository Hamiltonian must have shape (nb,nb,nk)")
    return np.transpose(values, (2, 0, 1)).copy()


@dataclass(frozen=True)
class MicroscopicH0Spec:
    """All direct-H0 inputs; every physically variable item is explicit."""

    lattice: HTQGLattice
    params: HTQGParams
    fold_map: BatchAFoldMap
    layout: MicroscopicBasisLayout
    support: BatchAPhysicalLabelSupport
    profile_s_abg: ProfileFourierInput
    uniform_replay_tolerance: float
    evidence_paths: tuple[str, ...]
    unresolved_physical_choices: tuple[str, ...]

    @property
    def support_sha256(self) -> str:
        """Identity that must also bind vertices, references, and checkpoints."""

        return self.support.support_sha256

    def validate(self, *, require_resolved: bool) -> None:
        _validate_batch_a_fold_map(self.fold_map)
        _validate_batch_a_layout_support(self.layout, self.support)
        if self.fold_map.supercell != batch_a_supercell():
            raise ValueError("H0 spec does not use the Batch-A supercell")
        if not np.array_equal(self.layout.g_indices, self.lattice.g_indices):
            raise ValueError("layout g_indices must exactly replay lattice.g_indices")
        if not np.array_equal(self.support.seed_g_indices, self.layout.g_indices):
            raise ValueError("physical-label support seed must replay layout.g_indices")
        if self.support.labels.shape[:4] != (
            self.fold_map.reduced_nk,
            2,
            BATCH_A_PERIOD,
            self.layout.g_indices.shape[0],
        ):
            raise ValueError("physical-label support does not match H0 basis/mesh")
        s = self.profile_s_abg.values_by_valley
        if float(np.max(np.abs(s.imag))) != 0.0:
            raise ValueError("s_ABG profile must be explicitly real")
        if not np.array_equal(s[0], s[1]):
            raise ValueError("s_ABG itself must be valley independent")
        if np.any(s.real < 0.0) or np.any(s.real > 1.0):
            raise ValueError("s_ABG profile values must lie in [0,1]")
        tolerance = float(self.uniform_replay_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("uniform replay tolerance must be finite and nonnegative")
        if not self.evidence_paths or any(not str(path).strip() for path in self.evidence_paths):
            raise ValueError("at least one explicit H0 evidence path is required")
        if require_resolved and self.unresolved_physical_choices:
            raise ValueError(
                "H0 spec has unresolved physical choices: "
                + "; ".join(self.unresolved_physical_choices)
            )


_MICROSCOPIC_H0_ARTIFACT_REGISTRY_LOCK = threading.RLock()
_MICROSCOPIC_H0_ARTIFACT_REGISTRY: dict[
    int,
    tuple[weakref.ReferenceType["MicroscopicH0Artifact"], tuple[object, ...]],
] = {}


@dataclass(frozen=True, init=False)
class MicroscopicH0Artifact:
    """Factory-sealed direct-H0 result with derivation and support lineage."""

    h0_ket: Array
    h0_sha256: str
    support_sha256: str
    evidence_paths: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "MicroscopicH0Artifact must be created by assemble_direct_h0_artifact"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("MicroscopicH0Artifact does not permit subclasses")

    def __copy__(self) -> "MicroscopicH0Artifact":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "MicroscopicH0Artifact":
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("MicroscopicH0Artifact does not permit pickling")


def validate_microscopic_h0_artifact(
    artifact: MicroscopicH0Artifact,
) -> MicroscopicH0Artifact:
    """Require an unmodified exact artifact registered by the H0 assembler."""

    if type(artifact) is not MicroscopicH0Artifact:
        raise TypeError("h0_artifact must be an exact factory-built artifact")
    with _MICROSCOPIC_H0_ARTIFACT_REGISTRY_LOCK:
        registered = _MICROSCOPIC_H0_ARTIFACT_REGISTRY.get(id(artifact))
    h0 = getattr(artifact, "h0_ket", None)
    if not isinstance(h0, np.ndarray):
        raise TypeError("h0_artifact must be an exact factory-built artifact")
    current = (
        getattr(artifact, "h0_sha256", None),
        getattr(artifact, "support_sha256", None),
        getattr(artifact, "evidence_paths", None),
        hashlib.sha256(
            np.ascontiguousarray(h0, dtype=np.dtype("<c16")).tobytes(order="C")
        ).hexdigest(),
    )
    if (
        registered is None
        or registered[0]() is not artifact
        or current != registered[1]
    ):
        raise TypeError("h0_artifact must be an exact factory-built artifact")
    return artifact


def build_batch_a_reduced_k_grid(
    lattice: HTQGLattice, fold_map: BatchAFoldMap
) -> Array:
    _validate_batch_a_fold_map(fold_map)
    if fold_map.primitive_mesh != BATCH_A_PRIMITIVE_MESH or fold_map.reduced_mesh != BATCH_A_REDUCED_MESH:
        raise ValueError("not the exact Batch-A fold map")
    out = np.empty((2, 4), dtype=np.complex128)
    for r1 in range(2):
        for r2 in range(4):
            out[r1, r2] = (r1 / 8.0) * lattice.b_m1 + (r2 / 4.0) * lattice.b_m2
    return out


def _embed_parent_block(
    out: Array,
    block: Array,
    *,
    layout: MicroscopicBasisLayout,
    fold: int,
    valley: int,
    spin: int,
) -> None:
    ng = layout.g_indices.shape[0]
    expected = 8 * ng
    if block.shape != (expected, expected):
        raise ValueError("parent HTQG block has an incompatible shell dimension")
    indices = np.asarray(
        [
            layout.flat_index(fold, g, layer, sub, valley, spin)
            for g in range(ng)
            for layer in range(4)
            for sub in range(2)
        ],
        dtype=np.int64,
    )
    out[np.ix_(indices, indices)] += block


def _lattice_on_parent_g_indices(
    lattice: HTQGLattice,
    parent_g_indices: Array,
) -> HTQGLattice:
    """Return an immutable lattice view on one support-resolved parent shell."""

    labels = np.asarray(parent_g_indices, dtype=np.int64)
    if labels.ndim != 2 or labels.shape != lattice.g_indices.shape:
        raise ValueError("support parent labels have an incompatible shell shape")
    if np.unique(labels, axis=0).shape[0] != labels.shape[0]:
        raise ValueError("support parent labels must be unique")
    vectors = np.asarray(
        [int(g1) * lattice.b_m1 + int(g2) * lattice.b_m2 for g1, g2 in labels],
        dtype=np.complex128,
    )
    return replace(
        lattice,
        g_indices=labels.copy(),
        g_vectors=vectors,
        matrix_dim=8 * int(labels.shape[0]),
    )


def _support_parent_lattice(
    spec: MicroscopicH0Spec,
    *,
    reduced_k: int,
    valley: int,
    fold: int,
) -> HTQGLattice:
    return _lattice_on_parent_g_indices(
        spec.lattice,
        spec.support.parent_g_indices(
            reduced_k=reduced_k, valley=valley, fold=fold
        ),
    )


_T34_PARENT_INTEGER_SHIFTS = ((0, 0), (1, 0), (0, 1))


def _one_way_t34_parent(
    *,
    k_tilde: complex,
    valley_eta: int,
    channel: int,
    lattice: HTQGLattice,
    params: HTQGParams,
    entries: Sequence[object],
) -> Array:
    """No-displacement layer-4 -> layer-3 edge ``A_c`` in parent order."""

    ng = lattice.n_g
    out = np.zeros((8 * ng, 8 * ng), dtype=np.complex128)
    base_k = complex(k_tilde) if int(valley_eta) == 1 else -complex(k_tilde)
    for entry in entries:
        if int(getattr(entry, "channel")) != int(channel):
            continue
        source = int(getattr(entry, "source_index"))
        target = int(getattr(entry, "target_index"))
        block = moire_coupling_matrix(channel, params) * _mdt_factor(
            channel=channel,
            k_tilde=base_k,
            source_g=complex(lattice.g_vectors[source]),
            target_g=complex(lattice.g_vectors[target]),
            source_layer=4,
            target_layer=3,
            lattice=lattice,
            params=params,
        )
        if int(valley_eta) == -1:
            block = block.conjugate()
        row = slice(8 * target + 4, 8 * target + 6)
        column = slice(8 * source + 6, 8 * source + 8)
        out[row, column] += block
    return out


def _one_way_t34_edge_block(
    *,
    k_tilde: complex,
    valley_eta: int,
    channel: int,
    source_g_label: tuple[int, int],
    lattice: HTQGLattice,
    params: HTQGParams,
) -> tuple[tuple[int, int], Array]:
    """Return one unprojected ``4 -> 3`` edge and its parent target label.

    The target reciprocal label may lie outside ``lattice.g_indices``.  This is
    required for the once-projected local product ``P M_c A_c P``: the
    intermediate plane wave is an algebraic label, not an added basis state.
    K-prime follows ``H_K'(k) = H_K(-k)^*`` and therefore passes the unbarred
    parent labels to the MDT factor before conjugating the block.
    """

    channel = int(channel)
    if not 0 <= channel < len(_T34_PARENT_INTEGER_SHIFTS):
        raise IndexError("T34 channel must lie in [0,3)")
    eta = int(valley_eta)
    if eta not in VALLEY_ORDER:
        raise ValueError("valley_eta must be the physical label +1 or -1")
    source_label = (int(source_g_label[0]), int(source_g_label[1]))
    shift = _T34_PARENT_INTEGER_SHIFTS[channel]
    target_label = (source_label[0] + shift[0], source_label[1] + shift[1])
    source_g = source_label[0] * lattice.b_m1 + source_label[1] * lattice.b_m2
    target_g = target_label[0] * lattice.b_m1 + target_label[1] * lattice.b_m2
    base_k = complex(k_tilde) if eta == 1 else -complex(k_tilde)
    block = moire_coupling_matrix(channel, params) * _mdt_factor(
        channel=channel,
        k_tilde=base_k,
        source_g=complex(source_g),
        target_g=complex(target_g),
        source_layer=4,
        target_layer=3,
        lattice=lattice,
        params=params,
    )
    if eta == -1:
        block = block.conjugate()
    return target_label, np.asarray(block, dtype=np.complex128)


def _projected_profile_t34_product(
    *,
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    profile: ProfileFourierInput,
    channel: int,
    reduced_k_index: int,
    reduced_k: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
) -> Array:
    """Construct ``P M_c A_c P`` without projecting the intermediate state."""

    _validate_batch_a_layout_support(layout, support)
    if profile.period_cells != layout.period_cells:
        raise ValueError("profile and canonical basis periods differ")
    nsample = int(profile.sample_positions_cells.size)
    coefficients = np.fft.fft(profile.values_by_valley, axis=1) / float(nsample)
    output_by_valley_n2: list[dict[int, list[tuple[int, int, int]]]] = []
    for valley in range(2):
        by_n2: dict[int, list[tuple[int, int, int]]] = {}
        for fold in range(BATCH_A_PERIOD):
            for g in range(layout.g_indices.shape[0]):
                n1, n2 = support.physical_plane_wave_label(
                    reduced_k=reduced_k_index,
                    fold=fold,
                    g=g,
                    valley=valley,
                )
                by_n2.setdefault(int(n2), []).append((fold, g, int(n1)))
        output_by_valley_n2.append(by_n2)

    maximum_delta_n1 = 0
    virtual_edges: list[
        tuple[int, int, int, tuple[int, int], Array, list[tuple[int, int, int]]]
    ] = []
    for valley, eta in enumerate(layout.valley_order):
        by_n2 = output_by_valley_n2[valley]
        for fold in range(BATCH_A_PERIOD):
            primitive_k = complex(
                reduced_k + (fold / BATCH_A_PERIOD) * lattice.b_m1
            )
            for g in range(layout.g_indices.shape[0]):
                source_label = support.parent_g_label(
                    reduced_k=reduced_k_index,
                    valley=valley,
                    fold=fold,
                    g=g,
                )
                target_label, block = _one_way_t34_edge_block(
                    k_tilde=primitive_k,
                    valley_eta=int(eta),
                    channel=channel,
                    source_g_label=source_label,
                    lattice=lattice,
                    params=params,
                )
                target_n1 = int(fold) + int(eta) * BATCH_A_PERIOD * target_label[0]
                target_n2 = int(eta) * BATCH_A_PERIOD * target_label[1]
                outputs = by_n2.get(target_n2, [])
                for _fold_out, _g_out, output_n1 in outputs:
                    maximum_delta_n1 = max(
                        maximum_delta_n1, abs(int(output_n1) - target_n1)
                    )
                virtual_edges.append(
                    (valley, fold, g, (target_n1, target_n2), block, outputs)
                )
    if nsample < 2 * maximum_delta_n1 + 1:
        raise ValueError(
            "profile sampling aliases the virtual T34 intermediate; require "
            f"at least {2 * maximum_delta_n1 + 1} samples"
        )

    out = np.zeros((layout.dimension, layout.dimension), dtype=np.complex128)
    for valley, fold_in, g_in, target_n, block, outputs in virtual_edges:
        target_n1, _target_n2 = target_n
        for fold_out, g_out, output_n1 in outputs:
            coefficient = coefficients[
                valley, (int(output_n1) - int(target_n1)) % nsample
            ]
            if coefficient == 0.0:
                continue
            for spin in range(2):
                for sub_out in range(2):
                    row = layout.flat_index(
                        fold_out, g_out, 2, sub_out, valley, spin
                    )
                    for sub_in in range(2):
                        column = layout.flat_index(
                            fold_in, g_in, 3, sub_in, valley, spin
                        )
                        out[row, column] += coefficient * block[sub_out, sub_in]
    return out


def _direct_dft_projected_profile_t34_product(
    *,
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    profile: ProfileFourierInput,
    channel: int,
    reduced_k_index: int,
    reduced_k: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
) -> Array:
    """Padded-parent, literal-sample oracle for ``P M_c A_c P``.

    Each ``(k,valley,fold)`` block receives its own translated one-shell-padded
    parent lattice.  The coupling-table implementation determines the edge;
    cropping back to the retained support occurs only after profile
    multiplication.  Physical labels are decoded directly from the canonical
    arrays here, without the production ``parent_g_*`` helpers.
    """

    _validate_batch_a_layout_support(layout, support)

    def oracle_parent_indices(*, valley: int, fold: int) -> Array:
        eta = int(layout.valley_order[valley])
        physical = np.asarray(
            support.labels[reduced_k_index, valley, fold], dtype=np.int64
        )
        if np.any(np.mod(physical[:, 0], 4) != fold) or np.any(
            np.mod(physical[:, 1], 4) != 0
        ):
            raise RuntimeError("oracle support labels are not parent-decodable")
        return np.column_stack(
            (
                eta * (physical[:, 0] - fold) // 4,
                eta * physical[:, 1] // 4,
            )
        ).astype(np.int64, copy=False)

    def oracle_translation(parent: Array) -> Array:
        sums = np.sum(parent, axis=0) - np.sum(
            support.seed_g_indices, axis=0
        )
        if np.any(np.mod(sums, support.ng) != 0):
            raise RuntimeError("oracle support translation is not integral")
        return np.asarray(sums // support.ng, dtype=np.int64)

    if profile.period_cells != layout.period_cells:
        raise ValueError("profile and canonical basis periods differ")
    channel = int(channel)
    if not 0 <= channel < 3:
        raise IndexError("T34 channel must lie in [0,3)")

    nsample = int(profile.sample_positions_cells.size)
    sample_indices = np.arange(nsample, dtype=np.float64)
    out = np.zeros((layout.dimension, layout.dimension), dtype=np.complex128)
    maximum_delta_n1 = 0
    for valley, eta in enumerate(layout.valley_order):
        for fold_in in range(BATCH_A_PERIOD):
            retained_labels = oracle_parent_indices(valley=valley, fold=fold_in)
            translation = oracle_translation(retained_labels)
            padded_labels = hex_shell_indices(int(lattice.n_shells) + 1) + translation
            padded_vectors = np.asarray(
                [
                    int(g1) * lattice.b_m1 + int(g2) * lattice.b_m2
                    for g1, g2 in padded_labels
                ],
                dtype=np.complex128,
            )
            padded_entries = build_coupling_table(
                padded_vectors, lattice.q_vectors
            )
            retained_lookup = {
                tuple(int(value) for value in label): g
                for g, label in enumerate(retained_labels)
            }
            padded_edges: list[tuple[int, tuple[int, int], complex, complex]] = []
            for entry in padded_entries:
                if int(entry.channel) != channel:
                    continue
                source_index = int(entry.source_index)
                target_index = int(entry.target_index)
                source_label = tuple(
                    int(value) for value in padded_labels[source_index]
                )
                if source_label not in retained_lookup:
                    continue
                target_label = tuple(
                    int(value) for value in padded_labels[target_index]
                )
                padded_edges.append(
                    (
                        retained_lookup[source_label],
                        target_label,
                        complex(padded_vectors[source_index]),
                        complex(padded_vectors[target_index]),
                    )
                )
            if len(padded_edges) != layout.g_indices.shape[0] or len(
                {edge[0] for edge in padded_edges}
            ) != layout.g_indices.shape[0]:
                raise RuntimeError(
                    "support-resolved padded parent lacks one T34 edge per source"
                )

            primitive_k = complex(
                reduced_k + (fold_in / BATCH_A_PERIOD) * lattice.b_m1
            )
            base_k = primitive_k if int(eta) == 1 else -primitive_k
            for g_in, target_label, source_g, target_g in padded_edges:
                block = moire_coupling_matrix(channel, params) * _mdt_factor(
                    channel=channel,
                    k_tilde=base_k,
                    source_g=source_g,
                    target_g=target_g,
                    source_layer=4,
                    target_layer=3,
                    lattice=lattice,
                    params=params,
                )
                if int(eta) == -1:
                    block = block.conjugate()
                target_n1 = (
                    fold_in + int(eta) * BATCH_A_PERIOD * int(target_label[0])
                )
                target_n2 = int(eta) * BATCH_A_PERIOD * int(target_label[1])
                for fold_out in range(BATCH_A_PERIOD):
                    for g_out in range(layout.g_indices.shape[0]):
                        output_pair = support.labels[
                            reduced_k_index, valley, fold_out, g_out
                        ]
                        output_n1 = int(output_pair[0])
                        output_n2 = int(output_pair[1])
                        if int(output_n2) != target_n2:
                            continue
                        delta_n1 = int(output_n1) - target_n1
                        maximum_delta_n1 = max(maximum_delta_n1, abs(delta_n1))
                        phase = np.exp(
                            -2.0j * np.pi * sample_indices * delta_n1 / nsample
                        )
                        coefficient = complex(
                            np.dot(profile.values_by_valley[valley], phase) / nsample
                        )
                        if coefficient == 0.0:
                            continue
                        for spin in range(2):
                            for sub_out in range(2):
                                row = layout.flat_index(
                                    fold_out, g_out, 2, sub_out, valley, spin
                                )
                                for sub_in in range(2):
                                    column = layout.flat_index(
                                        fold_in, g_in, 3, sub_in, valley, spin
                                    )
                                    out[row, column] += (
                                        coefficient * block[sub_out, sub_in]
                                    )
    if nsample < 2 * maximum_delta_n1 + 1:
        raise ValueError(
            "profile sampling aliases the virtual T34 intermediate; require "
            f"at least {2 * maximum_delta_n1 + 1} samples"
        )
    return out


def _assemble_direct_h0_unchecked(spec: MicroscopicH0Spec) -> Array:
    layout = spec.layout
    lattice = spec.lattice
    reduced_k = build_batch_a_reduced_k_grid(lattice, spec.fold_map).reshape(-1)
    result = np.zeros(
        (spec.fold_map.reduced_nk, layout.dimension, layout.dimension),
        dtype=np.complex128,
    )
    dba = complex(lattice.d_ba)

    for reduced_index, kval in enumerate(reduced_k):
        hrest = np.zeros((layout.dimension, layout.dimension), dtype=np.complex128)
        for fold in range(BATCH_A_PERIOD):
            primitive_k = complex(kval + (fold / BATCH_A_PERIOD) * lattice.b_m1)
            for valley_index, eta in enumerate(layout.valley_order):
                parent_lattice = _support_parent_lattice(
                    spec,
                    reduced_k=reduced_index,
                    valley=valley_index,
                    fold=fold,
                )
                parent_entries = build_coupling_table(
                    parent_lattice.g_vectors, parent_lattice.q_vectors
                )
                for spin in range(2):
                    endpoint = build_hamiltonian(
                        primitive_k,
                        parent_lattice,
                        spec.params,
                        domain="alpha_beta_alpha",
                        valley=int(eta),
                        coupling_table=parent_entries,
                    )
                    endpoint_without_t34 = endpoint.copy()
                    for channel in range(3):
                        parent_a = _one_way_t34_parent(
                            k_tilde=primitive_k,
                            valley_eta=int(eta),
                            channel=channel,
                            lattice=parent_lattice,
                            params=spec.params,
                            entries=parent_entries,
                        )
                        qdotd = dot_2d(complex(lattice.q_vectors[channel]), dba)
                        # ABA has d34=-dBA.  Existing HTQG assembly uses
                        # exp[-i q.d34], hence p_ABA=exp[+i q.dBA] in K.
                        p_aba_k = np.exp(1j * qdotd)
                        p_aba = p_aba_k if int(eta) == 1 else p_aba_k.conjugate()
                        endpoint_without_t34 -= p_aba * parent_a
                        endpoint_without_t34 -= (p_aba * parent_a).conj().T
                    _embed_parent_block(
                        hrest,
                        endpoint_without_t34,
                        layout=layout,
                        fold=fold,
                        valley=valley_index,
                        spin=spin,
                    )

        total = hrest
        s_values = spec.profile_s_abg.values_by_valley[0].real
        for channel in range(3):
            qdotd = dot_2d(complex(lattice.q_vectors[channel]), dba)
            p_k = np.exp(-1j * qdotd * (2.0 * s_values - 1.0))
            phase_profile = ProfileFourierInput(
                sample_positions_cells=spec.profile_s_abg.sample_positions_cells,
                values_by_valley=np.asarray((p_k, p_k.conjugate())),
                period_cells=BATCH_A_PERIOD,
                convention="numpy_fft_exp_minus_iqx",
                provenance=(
                    f"ordered T34 phase channel {channel}; "
                    + spec.profile_s_abg.provenance
                ),
            )
            ma = _projected_profile_t34_product(
                layout=layout,
                support=spec.support,
                profile=phase_profile,
                channel=channel,
                reduced_k_index=reduced_index,
                reduced_k=complex(kval),
                lattice=lattice,
                params=spec.params,
            )
            # Project only the final ordered product: P M A P + h.c.  The
            # virtual A target is not truncated before multiplication by M.
            total = total + ma + ma.conj().T
        scale = max(1.0, float(np.max(np.abs(total))))
        residual = float(np.max(np.abs(total - total.conj().T)))
        if residual > 64.0 * np.finfo(float).eps * scale:
            raise RuntimeError("ordered direct H0 assembly is not Hermitian")
        result[reduced_index] = total
    return result


def assemble_direct_h0_dft_oracle(
    spec: MicroscopicH0Spec,
    *,
    ordering: Literal["left", "right"],
    max_dimension: int,
) -> Array:
    """Independent direct-DFT oracle for nonuniform ordered H0 assembly.

    The ``left`` path directly enumerates the once-projected ``P M A P``
    matrix elements and evaluates every profile coefficient by a literal sample
    sum.  It does not call the production product helper.  The ``right`` path
    intentionally retains the projected-factor ``(P A P)(P M P)`` product as a
    wrong-order discriminator.  Endpoint subtraction is shared because exact
    uniform ABA/ABG replay covers that separately.
    """

    spec.validate(require_resolved=True)
    if spec.layout.dimension > int(max_dimension):
        raise ValueError("direct-DFT H0 oracle exceeds explicit dimension bound")
    if ordering not in {"left", "right"}:
        raise ValueError("ordering must be 'left' or 'right'")

    layout = spec.layout
    lattice = spec.lattice
    reduced_k = build_batch_a_reduced_k_grid(lattice, spec.fold_map).reshape(-1)
    result = np.zeros(
        (spec.fold_map.reduced_nk, layout.dimension, layout.dimension),
        dtype=np.complex128,
    )
    dba = complex(lattice.d_ba)
    phase_profiles: list[ProfileFourierInput] = []
    s_values = spec.profile_s_abg.values_by_valley[0].real
    for channel in range(3):
        qdotd = dot_2d(complex(lattice.q_vectors[channel]), dba)
        p_k = np.exp(-1j * qdotd * (2.0 * s_values - 1.0))
        phase_profile = ProfileFourierInput(
            sample_positions_cells=spec.profile_s_abg.sample_positions_cells,
            values_by_valley=np.asarray((p_k, p_k.conjugate())),
            period_cells=BATCH_A_PERIOD,
            convention="numpy_fft_exp_minus_iqx",
            provenance=(
                f"independent direct-DFT T34 phase channel {channel}; "
                + spec.profile_s_abg.provenance
            ),
        )
        phase_profiles.append(phase_profile)

    for reduced_index, kval in enumerate(reduced_k):
        profile_entries = [
            _direct_dft_profile_entries(
                layout,
                spec.support,
                phase_profile,
                reduced_k=reduced_index,
            )
            for phase_profile in phase_profiles
        ]
        hrest = np.zeros((layout.dimension, layout.dimension), dtype=np.complex128)
        a_by_channel = [np.zeros_like(hrest) for _channel in range(3)]
        for fold in range(BATCH_A_PERIOD):
            primitive_k = complex(kval + (fold / BATCH_A_PERIOD) * lattice.b_m1)
            for valley_index, eta in enumerate(layout.valley_order):
                parent_lattice = _support_parent_lattice(
                    spec,
                    reduced_k=reduced_index,
                    valley=valley_index,
                    fold=fold,
                )
                parent_entries = build_coupling_table(
                    parent_lattice.g_vectors, parent_lattice.q_vectors
                )
                for spin in range(2):
                    endpoint = build_hamiltonian(
                        primitive_k,
                        parent_lattice,
                        spec.params,
                        domain="alpha_beta_alpha",
                        valley=int(eta),
                        coupling_table=parent_entries,
                    )
                    endpoint_without_t34 = endpoint.copy()
                    for channel in range(3):
                        parent_a = _one_way_t34_parent(
                            k_tilde=primitive_k,
                            valley_eta=int(eta),
                            channel=channel,
                            lattice=parent_lattice,
                            params=spec.params,
                            entries=parent_entries,
                        )
                        qdotd = dot_2d(complex(lattice.q_vectors[channel]), dba)
                        p_aba_k = np.exp(1j * qdotd)
                        p_aba = (
                            p_aba_k if int(eta) == 1 else p_aba_k.conjugate()
                        )
                        endpoint_without_t34 -= p_aba * parent_a
                        endpoint_without_t34 -= (p_aba * parent_a).conj().T
                        _embed_parent_block(
                            a_by_channel[channel],
                            parent_a,
                            layout=layout,
                            fold=fold,
                            valley=valley_index,
                            spin=spin,
                        )
                    _embed_parent_block(
                        hrest,
                        endpoint_without_t34,
                        layout=layout,
                        fold=fold,
                        valley=valley_index,
                        spin=spin,
                    )
        total = hrest
        for channel, a_operator in enumerate(a_by_channel):
            if ordering == "left":
                product = _direct_dft_projected_profile_t34_product(
                    layout=layout,
                    support=spec.support,
                    profile=phase_profiles[channel],
                    channel=channel,
                    reduced_k_index=reduced_index,
                    reduced_k=complex(kval),
                    lattice=lattice,
                    params=spec.params,
                )
            else:
                rows, columns, values = profile_entries[channel]
                product = _multiply_profile_entries(
                    a_operator,
                    rows,
                    columns,
                    values,
                    ordering="right",
                )
            total = total + product + product.conj().T
        result[reduced_index] = total
    return result


def assemble_uniform_endpoint_h0(
    spec: MicroscopicH0Spec, *, endpoint: Literal["ABA", "ABG"]
) -> Array:
    """Fold exact primitive endpoint blocks without wall interpolation."""

    spec.validate(require_resolved=True)
    domain = "alpha_beta_alpha" if endpoint == "ABA" else "alpha_beta_gamma"
    reduced_k = build_batch_a_reduced_k_grid(spec.lattice, spec.fold_map).reshape(-1)
    out = np.zeros(
        (spec.fold_map.reduced_nk, spec.layout.dimension, spec.layout.dimension),
        dtype=np.complex128,
    )
    for reduced_index, kval in enumerate(reduced_k):
        for fold in range(BATCH_A_PERIOD):
            primitive_k = complex(kval + (fold / BATCH_A_PERIOD) * spec.lattice.b_m1)
            for valley_index, eta in enumerate(spec.layout.valley_order):
                parent_lattice = _support_parent_lattice(
                    spec,
                    reduced_k=reduced_index,
                    valley=valley_index,
                    fold=fold,
                )
                parent_entries = build_coupling_table(
                    parent_lattice.g_vectors, parent_lattice.q_vectors
                )
                parent = build_hamiltonian(
                    primitive_k,
                    parent_lattice,
                    spec.params,
                    domain=domain,
                    valley=int(eta),
                    coupling_table=parent_entries,
                )
                for spin in range(2):
                    _embed_parent_block(
                        out[reduced_index],
                        parent,
                        layout=spec.layout,
                        fold=fold,
                        valley=valley_index,
                        spin=spin,
                    )
    return out


def uniform_h0_replay_residuals(spec: MicroscopicH0Spec) -> dict[str, float]:
    """Compare ordered assembly at ``s=0/1`` with exact ABA/ABG blocks."""

    spec.validate(require_resolved=True)
    residuals: dict[str, float] = {}
    for endpoint, value in (("ABA", 0.0), ("ABG", 1.0)):
        samples = np.full_like(spec.profile_s_abg.values_by_valley, value)
        endpoint_profile = ProfileFourierInput(
            sample_positions_cells=spec.profile_s_abg.sample_positions_cells,
            values_by_valley=samples,
            period_cells=BATCH_A_PERIOD,
            convention="numpy_fft_exp_minus_iqx",
            provenance=f"uniform {endpoint} replay",
        )
        ordered = _assemble_direct_h0_unchecked(
            replace(spec, profile_s_abg=endpoint_profile)
        )
        exact = assemble_uniform_endpoint_h0(spec, endpoint=endpoint)  # type: ignore[arg-type]
        residuals[endpoint] = float(np.max(np.abs(ordered - exact)))
    return residuals


def assemble_direct_h0(
    spec: MicroscopicH0Spec, *, verify_uniform_replay: bool
) -> Array:
    """Assemble full microscopic H0; optionally fail closed on both endpoints."""

    spec.validate(require_resolved=True)
    if type(verify_uniform_replay) is not bool:
        raise TypeError("verify_uniform_replay must be an explicit bool")
    if verify_uniform_replay:
        residuals = uniform_h0_replay_residuals(spec)
        failed = {
            name: value
            for name, value in residuals.items()
            if value > float(spec.uniform_replay_tolerance)
        }
        if failed:
            raise RuntimeError(f"uniform ABA/ABG H0 replay failed: {failed}")
    return _assemble_direct_h0_unchecked(spec)


def assemble_direct_h0_artifact(
    spec: MicroscopicH0Spec, *, verify_uniform_replay: bool
) -> MicroscopicH0Artifact:
    """Assemble and seal one direct H0 with its validated typed derivation."""

    h0 = _immutable_array_copy(
        assemble_direct_h0(spec, verify_uniform_replay=verify_uniform_replay),
        dtype=np.dtype(np.complex128),
    )
    h0_sha256 = hashlib.sha256(
        np.ascontiguousarray(h0, dtype=np.dtype("<c16")).tobytes(order="C")
    ).hexdigest()
    artifact = object.__new__(MicroscopicH0Artifact)
    object.__setattr__(artifact, "h0_ket", h0)
    object.__setattr__(artifact, "h0_sha256", h0_sha256)
    object.__setattr__(artifact, "support_sha256", spec.support_sha256)
    object.__setattr__(
        artifact,
        "evidence_paths",
        tuple(str(path) for path in spec.evidence_paths),
    )
    expected = (
        h0_sha256,
        spec.support_sha256,
        tuple(str(path) for path in spec.evidence_paths),
        h0_sha256,
    )
    registry_key = id(artifact)

    def remove_registry_entry(reference: object, *, key: int = registry_key) -> None:
        with _MICROSCOPIC_H0_ARTIFACT_REGISTRY_LOCK:
            current = _MICROSCOPIC_H0_ARTIFACT_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _MICROSCOPIC_H0_ARTIFACT_REGISTRY.pop(key, None)

    reference = weakref.ref(artifact, remove_registry_entry)
    with _MICROSCOPIC_H0_ARTIFACT_REGISTRY_LOCK:
        _MICROSCOPIC_H0_ARTIFACT_REGISTRY[registry_key] = (reference, expected)
    return artifact


def _batch_a_plane_wave_state_tables(
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
) -> tuple[Array, Array, Array, Array]:
    """Return support-resolved ``(K,fold,g)`` state and momentum tables."""

    _validate_batch_a_fold_map(fold_map)
    _validate_batch_a_layout_support(layout, support)
    ng = int(layout.g_indices.shape[0])
    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("density support seed must replay layout.g_indices")
    if support.labels.shape != (fold_map.reduced_nk, 2, 4, ng, 2):
        raise ValueError("density support does not match the Batch-A layout/mesh")
    states_per_species = fold_map.reduced_nk * BATCH_A_PERIOD * ng
    state_k = np.repeat(
        np.arange(fold_map.reduced_nk, dtype=np.int64), BATCH_A_PERIOD * ng
    )
    folds = np.tile(
        np.repeat(np.arange(BATCH_A_PERIOD, dtype=np.int64), ng),
        fold_map.reduced_nk,
    )
    g_indices = np.tile(
        np.arange(ng, dtype=np.int64), fold_map.reduced_nk * BATCH_A_PERIOD
    )
    if state_k.size != states_per_species:
        raise RuntimeError("Batch-A plane-wave state-table size mismatch")

    # Physical common-grid momenta are (r1+2*n1, r2+n2), where ``n`` is the
    # support authority. Keep the arithmetic exact and reject int64 overflow.
    int64_limit = int(np.iinfo(np.int64).max)
    max_abs_n1 = max(
        abs(int(np.min(support.labels[..., 0]))),
        abs(int(np.max(support.labels[..., 0]))),
    )
    max_abs_n2 = max(
        abs(int(np.min(support.labels[..., 1]))),
        abs(int(np.max(support.labels[..., 1]))),
    )
    if max_abs_n1 > (int64_limit - 1) // 2 or max_abs_n2 > int64_limit - 3:
        raise OverflowError("support plane-wave labels exceed exact int64 momentum range")

    reduced_r1 = state_k // BATCH_A_REDUCED_MESH[1]
    reduced_r2 = state_k % BATCH_A_REDUCED_MESH[1]
    momenta = np.empty((2, states_per_species, 2), dtype=np.int64)
    for valley in range(2):
        physical_labels = support.labels[state_k, valley, folds, g_indices]
        momenta[valley, :, 0] = reduced_r1 + 2 * physical_labels[:, 0]
        momenta[valley, :, 1] = reduced_r2 + physical_labels[:, 1]
    return state_k, folds, g_indices, momenta


def _batch_a_plane_wave_label_counts(
    momenta: Array,
) -> tuple[Array, int, int]:
    """Count all same-valley transfer labels in bounded typed chunks."""

    q1_min = min(
        int(np.min(valley[:, 0])) - int(np.max(valley[:, 0]))
        for valley in momenta
    )
    q1_max = max(
        int(np.max(valley[:, 0])) - int(np.min(valley[:, 0]))
        for valley in momenta
    )
    q2_min = min(
        int(np.min(valley[:, 1])) - int(np.max(valley[:, 1]))
        for valley in momenta
    )
    q2_max = max(
        int(np.max(valley[:, 1])) - int(np.min(valley[:, 1]))
        for valley in momenta
    )
    width1 = q1_max - q1_min + 1
    width2 = q2_max - q2_min + 1
    label_slots = width1 * width2
    if label_slots > int(np.iinfo(np.intp).max):
        raise OverflowError("plane-wave transfer-label table exceeds platform limits")

    counts = np.zeros((2, label_slots), dtype=np.int64)
    states_per_species = int(momenta.shape[1])
    target_chunk = max(
        1, _DENSITY_VERTEX_PAIR_CHUNK_ENTRIES // states_per_species
    )
    for valley in range(2):
        momentum = momenta[valley]
        for start in range(0, states_per_species, target_chunk):
            stop = min(start + target_chunk, states_per_species)
            linear_labels = (
                momentum[start:stop, 0, None]
                - momentum[None, :, 0]
                - q1_min
            ) * width2
            linear_labels += (
                momentum[start:stop, 1, None]
                - momentum[None, :, 1]
                - q2_min
            )
            counts[valley] += np.bincount(
                linear_labels.reshape(-1), minlength=label_slots
            )
        if int(np.sum(counts[valley], dtype=np.int64)) != states_per_species**2:
            raise RuntimeError("plane-wave transfer-label count pass is incomplete")
    return counts, q1_min, q2_min


def validate_batch_a_plane_wave_density_vertices(
    vertices: Sequence[SparseDensityVertex],
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
) -> None:
    """Validate physical labels and the exact retained inventory without sets."""

    _validate_batch_a_fold_map(fold_map)
    _validate_batch_a_layout_support(layout, support)

    items = tuple(vertices)
    labels = sorted(vertex.label for vertex in items)
    if any(labels[index - 1] == labels[index] for index in range(1, len(labels))):
        raise ValueError("plane-wave density-vertex labels must be unique")

    ng = int(layout.g_indices.shape[0])
    states_per_species = fold_map.reduced_nk * BATCH_A_PERIOD * ng
    species_count = int(np.prod(layout.shape[2:]))
    expected_entries = species_count * states_per_species**2
    actual_entries = sum(int(vertex.values.size) for vertex in items)
    track_inventory = actual_entries == expected_entries
    seen = np.zeros(expected_entries, dtype=np.bool_) if track_inventory else None

    # Each physical pair has the unique typed rank
    # ``(species, target-(K,fold,g), source-(K,fold,g))``.  Streaming these
    # ranks proves exact inventory without materializing expected/actual tuple
    # sets.  Chunk-local uniqueness plus the bit inventory is fail-closed even
    # if a caller mutates a frozen dataclass's underlying arrays after init.
    for vertex in items:
        label0, label1 = vertex.label
        size = int(vertex.values.size)
        if any(
            int(array.size) != size
            for array in (
                vertex.target_k,
                vertex.source_k,
                vertex.rows,
                vertex.columns,
            )
        ):
            raise ValueError("plane-wave density-vertex arrays must be equally sized")
        for start in range(0, size, _DENSITY_VERTEX_PAIR_CHUNK_ENTRIES):
            stop = min(start + _DENSITY_VERTEX_PAIR_CHUNK_ENTRIES, size)
            target_k = vertex.target_k[start:stop]
            source_k = vertex.source_k[start:stop]
            rows = vertex.rows[start:stop]
            columns = vertex.columns[start:stop]
            if (
                np.any(target_k < 0)
                or np.any(target_k >= fold_map.reduced_nk)
                or np.any(source_k < 0)
                or np.any(source_k >= fold_map.reduced_nk)
                or np.any(rows < 0)
                or np.any(rows >= layout.dimension)
                or np.any(columns < 0)
                or np.any(columns >= layout.dimension)
            ):
                raise ValueError("plane-wave density-vertex index is outside its basis")

            target_basis = np.unravel_index(rows, layout.shape, order="C")
            source_basis = np.unravel_index(columns, layout.shape, order="C")
            if any(
                np.any(target_basis[axis] != source_basis[axis])
                for axis in range(2, 6)
            ):
                raise ValueError(
                    "plane-wave density vertices must conserve layer, sublattice, "
                    "valley, and spin"
                )

            target_r1 = target_k // BATCH_A_REDUCED_MESH[1]
            target_r2 = target_k % BATCH_A_REDUCED_MESH[1]
            source_r1 = source_k // BATCH_A_REDUCED_MESH[1]
            source_r2 = source_k % BATCH_A_REDUCED_MESH[1]
            if np.any(
                target_r1
                != (source_r1 + int(label0)) % BATCH_A_REDUCED_MESH[0]
            ) or np.any(
                target_r2
                != (source_r2 + int(label1)) % BATCH_A_REDUCED_MESH[1]
            ):
                raise ValueError(
                    f"Gamma_{vertex.label} is not one fixed reduced-K translation"
                )

            valley = target_basis[4]
            target_labels = support.labels[
                target_k, valley, target_basis[0], target_basis[1]
            ]
            source_labels = support.labels[
                source_k, valley, source_basis[0], source_basis[1]
            ]
            actual_q1 = (
                target_r1
                + 2 * target_labels[:, 0]
                - source_r1
                - 2 * source_labels[:, 0]
            )
            actual_q2 = (
                target_r2
                + target_labels[:, 1]
                - source_r2
                - source_labels[:, 1]
            )
            if np.any(actual_q1 != int(label0)) or np.any(
                actual_q2 != int(label1)
            ):
                raise ValueError(
                    f"Gamma_{vertex.label} entry has a different physical transfer"
                )

            if seen is not None:
                species = (
                    ((target_basis[2] * 2 + target_basis[3]) * 2 + valley) * 2
                    + target_basis[5]
                )
                target_state = (
                    (target_k * BATCH_A_PERIOD + target_basis[0]) * ng
                    + target_basis[1]
                )
                source_state = (
                    (source_k * BATCH_A_PERIOD + source_basis[0]) * ng
                    + source_basis[1]
                )
                pair_rank = (
                    (species * states_per_species + target_state)
                    * states_per_species
                    + source_state
                )
                unique_ranks = np.unique(pair_rank)
                if unique_ranks.size != pair_rank.size or np.any(seen[unique_ranks]):
                    raise ValueError(
                        "plane-wave density-vertex matrix-element inventory is incomplete"
                    )
                seen[unique_ranks] = True

    if not track_inventory:
        raise ValueError("plane-wave density-vertex label inventory is incomplete")
    if seen is None or not np.all(seen):
        raise ValueError("plane-wave density-vertex label inventory is incomplete")


def build_primitive_plane_wave_density_vertices(
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
    *,
    b_m1: complex,
    b_m2: complex,
    weight_for_transfer: Callable[[tuple[int, int], complex], float],
    provenance: str,
    max_entries: int,
) -> tuple[SparseDensityVertex, ...]:
    """Build the primitive ``(8,4)`` vertices paired with the 4x1 inventory.

    Primitive one-body order is ``(g,layer,sublattice,valley,spin)``. Labels
    use the same common units as the supercell vertices:
    ``Q=(q1/8)b_m1+(q2/4)b_m2``.
    """

    _validate_batch_a_fold_map(fold_map)
    _validate_batch_a_layout_support(layout, support)
    if not callable(weight_for_transfer):
        raise TypeError("weight_for_transfer must be callable")
    if not str(provenance).strip():
        raise ValueError("primitive density-vertex provenance must be explicit")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be an explicit positive integer")
    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("primitive density support seed must replay layout.g_indices")
    if support.labels.shape[:4] != (
        fold_map.reduced_nk,
        2,
        BATCH_A_PERIOD,
        layout.g_indices.shape[0],
    ):
        raise ValueError("primitive density support does not match layout/mesh")

    primitive_shape = layout.shape[1:]
    primitive_dimension = layout.no_fold_dimension
    by_species: dict[
        tuple[int, int, int, int], list[tuple[int, int, tuple[int, int]]]
    ] = {}
    for p1 in range(BATCH_A_PRIMITIVE_MESH[0]):
        for p2 in range(BATCH_A_PRIMITIVE_MESH[1]):
            k = p1 * BATCH_A_PRIMITIVE_MESH[1] + p2
            (r1, r2), fold = fold_map.reduced_coordinate_and_fold((p1, p2))
            reduced_k = r1 * BATCH_A_REDUCED_MESH[1] + r2
            for orbital in range(primitive_dimension):
                g, layer, sublattice, valley, spin = tuple(
                    int(value)
                    for value in np.unravel_index(orbital, primitive_shape, order="C")
                )
                eta = int(layout.valley_order[valley])
                g1, g2 = support.parent_g_label(
                    reduced_k=reduced_k,
                    valley=valley,
                    fold=fold,
                    g=g,
                )
                momentum = (p1 + 8 * eta * g1, p2 + 4 * eta * g2)
                by_species.setdefault(
                    (layer, sublattice, valley, spin), []
                ).append((k, orbital, momentum))

    grouped: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    count = 0
    for records in by_species.values():
        for target_k, row, target_momentum in records:
            for source_k, column, source_momentum in records:
                label = (
                    target_momentum[0] - source_momentum[0],
                    target_momentum[1] - source_momentum[1],
                )
                grouped.setdefault(label, []).append(
                    (target_k, source_k, row, column)
                )
                count += 1
                if count > max_entries:
                    raise ValueError(
                        "primitive density-vertex inventory exceeds max_entries"
                    )

    vertices: list[SparseDensityVertex] = []
    for label in sorted(grouped):
        records = np.asarray(grouped[label], dtype=np.int64)
        qvec = (
            (label[0] / 8.0) * complex(b_m1)
            + (label[1] / 4.0) * complex(b_m2)
        )
        weight = float(weight_for_transfer(label, qvec))
        if not math.isfinite(weight):
            raise ValueError(f"non-finite supplied weight for transfer {label}")
        vertices.append(
            SparseDensityVertex(
                label=label,
                target_k=records[:, 0],
                source_k=records[:, 1],
                rows=records[:, 2],
                columns=records[:, 3],
                values=np.ones(records.shape[0], dtype=np.complex128),
                weight=weight,
                provenance=f"{provenance}; primitive Q-label={label}",
            )
        )
    result = tuple(vertices)
    validate_density_vertex_family(
        result,
        nk=fold_map.primitive_nk,
        dimension=primitive_dimension,
        closure_tolerance=0.0,
        zero_mode_policy="include_supplied",
    )
    return result


def validate_primitive_supercell_density_vertex_equivalence(
    primitive_vertices: Sequence[SparseDensityVertex],
    supercell_vertices: Sequence[SparseDensityVertex],
    *,
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    tolerance: float,
) -> None:
    """Certify exact entrywise folding of the primitive physical vertices."""

    _validate_batch_a_fold_map(fold_map)
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if (
        fold_map.supercell != batch_a_supercell()
        or fold_map.primitive_mesh != BATCH_A_PRIMITIVE_MESH
        or fold_map.reduced_mesh != BATCH_A_REDUCED_MESH
    ):
        raise ValueError("vertex equivalence requires exact Batch-A geometry")
    primitive_items = tuple(primitive_vertices)
    supercell_items = tuple(supercell_vertices)
    primitive = {vertex.label: vertex for vertex in primitive_items}
    supercell = {vertex.label: vertex for vertex in supercell_items}
    if len(primitive) != len(primitive_items) or len(supercell) != len(supercell_items):
        raise ValueError("density-vertex labels must be unique")
    if primitive.keys() != supercell.keys():
        raise ValueError("primitive/supercell density-vertex label sets differ")

    k_map = np.empty(fold_map.primitive_nk, dtype=np.int64)
    fold_by_k = np.empty(fold_map.primitive_nk, dtype=np.int64)
    for p1 in range(BATCH_A_PRIMITIVE_MESH[0]):
        for p2 in range(BATCH_A_PRIMITIVE_MESH[1]):
            primitive_k = p1 * BATCH_A_PRIMITIVE_MESH[1] + p2
            reduced_1 = p1 % BATCH_A_REDUCED_MESH[0]
            fold = p1 // BATCH_A_REDUCED_MESH[0]
            k_map[primitive_k] = reduced_1 * BATCH_A_REDUCED_MESH[1] + p2
            fold_by_k[primitive_k] = fold

    orbital_map = np.empty(
        (fold_map.supercell.area_ratio, layout.no_fold_dimension), dtype=np.int64
    )
    primitive_shape = layout.shape[1:]
    for fold in range(fold_map.supercell.area_ratio):
        for orbital in range(layout.no_fold_dimension):
            g, layer, sublattice, valley, spin = tuple(
                int(value)
                for value in np.unravel_index(orbital, primitive_shape, order="C")
            )
            orbital_map[fold, orbital] = layout.flat_index(
                fold, g, layer, sublattice, valley, spin
            )

    def sorted_records(
        target_k: Array,
        source_k: Array,
        rows: Array,
        columns: Array,
        values: Array,
    ) -> tuple[Array, Array]:
        records = np.column_stack((target_k, source_k, rows, columns)).astype(
            np.int64, copy=False
        )
        order = np.lexsort(
            (records[:, 3], records[:, 2], records[:, 1], records[:, 0])
        )
        return records[order], np.asarray(values, dtype=np.complex128)[order]

    for label in primitive:
        source = primitive[label]
        target = supercell[label]
        if abs(source.weight - target.weight) > tol:
            raise ValueError(f"primitive/supercell weight mismatch for Q={label}")
        mapped_target_k = k_map[source.target_k]
        mapped_source_k = k_map[source.source_k]
        mapped_rows = orbital_map[fold_by_k[source.target_k], source.rows]
        mapped_columns = orbital_map[fold_by_k[source.source_k], source.columns]
        expected_records, expected_values = sorted_records(
            mapped_target_k,
            mapped_source_k,
            mapped_rows,
            mapped_columns,
            source.values,
        )
        actual_records, actual_values = sorted_records(
            target.target_k,
            target.source_k,
            target.rows,
            target.columns,
            target.values,
        )
        if not np.array_equal(expected_records, actual_records):
            raise ValueError(f"primitive/supercell entry mismatch for Q={label}")
        if np.max(np.abs(expected_values - actual_values), initial=0.0) > tol:
            raise ValueError(f"primitive/supercell value mismatch for Q={label}")


def build_plane_wave_density_vertices(
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
    *,
    b_m1: complex,
    b_m2: complex,
    weight_for_transfer: Callable[[tuple[int, int], complex], float],
    provenance: str,
    max_entries: int,
) -> tuple[SparseDensityVertex, ...]:
    """Build intravalley density vertices in the fixed microscopic basis.

    A transfer label ``(q1,q2)`` means
    ``Q=(q1/8)b_m1+(q2/4)b_m2`` on the Batch-A finite mesh.  Each vertex
    preserves layer, sublattice, valley, and spin.  Coulomb values and every
    normalization factor remain explicit caller-supplied weights.
    """

    _validate_batch_a_fold_map(fold_map)
    _validate_batch_a_layout_support(layout, support)
    if not callable(weight_for_transfer):
        raise TypeError("weight_for_transfer must be callable")
    if not str(provenance).strip():
        raise ValueError("density-vertex provenance must be explicit")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be an explicit positive integer")

    state_k, folds, g_indices, momenta = _batch_a_plane_wave_state_tables(
        layout, fold_map, support
    )
    states_per_species = int(state_k.size)
    species = list(np.ndindex(layout.shape[2:]))
    total_entries = len(species) * states_per_species**2
    if total_entries > max_entries:
        raise ValueError("microscopic density-vertex inventory exceeds max_entries")

    # Pass 1: count transfer labels with bounded typed buffers.  Counts differ
    # only by valley; layer, sublattice, and spin provide exact multiplicities.
    valley_counts, q1_min, q2_min = _batch_a_plane_wave_label_counts(momenta)
    label_slots = int(valley_counts.shape[1])
    width2 = (
        max(
            int(np.max(valley[:, 1])) - int(np.min(valley[:, 1]))
            for valley in momenta
        )
        - q2_min
        + 1
    )
    species_valleys = np.asarray([item[2] for item in species], dtype=np.int64)
    global_counts = np.sum(valley_counts[species_valleys], axis=0, dtype=np.int64)
    label_flats = np.flatnonzero(global_counts)
    label_counts = global_counts[label_flats]
    q1_labels = label_flats // width2 + q1_min
    q2_labels = label_flats % width2 + q2_min

    # Allocate exactly the final SparseDensityVertex inventory.  Label slices
    # are lexicographic because the dense linear key is q1-major, q2-minor.
    label_offsets = np.empty(label_flats.size, dtype=np.int64)
    label_offsets[0] = 0
    if label_flats.size > 1:
        label_offsets[1:] = np.cumsum(label_counts[:-1], dtype=np.int64)
    target_k_all = np.empty(total_entries, dtype=np.int64)
    source_k_all = np.empty(total_entries, dtype=np.int64)
    rows_all = np.empty(total_entries, dtype=np.int64)
    columns_all = np.empty(total_entries, dtype=np.int64)
    values_all = np.ones(total_entries, dtype=np.complex128)

    label_lookup = np.full(label_slots, -1, dtype=np.int64)
    label_lookup[label_flats] = np.arange(label_flats.size, dtype=np.int64)
    per_species_counts = valley_counts[species_valleys][:, label_flats]
    species_offsets = np.zeros_like(per_species_counts)
    if len(species) > 1:
        species_offsets[1:] = np.cumsum(
            per_species_counts[:-1], axis=0, dtype=np.int64
        )
    species_cursors = np.zeros_like(per_species_counts)

    ng = int(layout.g_indices.shape[0])
    orbital_by_species = np.empty((len(species), states_per_species), dtype=np.int64)
    for species_index, (layer, sublattice, valley, spin) in enumerate(species):
        orbital_by_species[species_index] = (
            (((((folds * ng + g_indices) * 4 + layer) * 2 + sublattice) * 2
              + valley) * 2)
            + spin
        )

    # Pass 2: stable-counting placement into preallocated label slices.  Stable
    # order within every chunk, increasing target chunks, and canonical species
    # offsets exactly preserve the former species/target/source append order.
    target_chunk = max(
        1, _DENSITY_VERTEX_PAIR_CHUNK_ENTRIES // states_per_species
    )
    for valley in range(2):
        species_indices = np.flatnonzero(species_valleys == valley)
        momentum = momenta[valley]
        for target_start in range(0, states_per_species, target_chunk):
            target_stop = min(target_start + target_chunk, states_per_species)
            linear_labels = (
                momentum[target_start:target_stop, 0, None]
                - momentum[None, :, 0]
                - q1_min
            ) * width2
            linear_labels += (
                momentum[target_start:target_stop, 1, None]
                - momentum[None, :, 1]
                - q2_min
            )
            vertex_ids = label_lookup[linear_labels.reshape(-1)]
            if np.any(vertex_ids < 0):
                raise RuntimeError("uncounted plane-wave transfer in fill pass")
            order = np.argsort(vertex_ids, kind="stable")
            sorted_vertex_ids = vertex_ids[order]
            group_starts = np.flatnonzero(
                np.r_[True, sorted_vertex_ids[1:] != sorted_vertex_ids[:-1]]
            )
            group_lengths = np.diff(np.r_[group_starts, order.size])
            group_vertex_ids = sorted_vertex_ids[group_starts]
            within_group = np.arange(order.size, dtype=np.int64)
            within_group -= np.repeat(group_starts, group_lengths)
            target_positions = target_start + order // states_per_species
            source_positions = order % states_per_species

            for species_index in species_indices:
                prior = species_cursors[species_index, group_vertex_ids]
                destinations = (
                    label_offsets[sorted_vertex_ids]
                    + species_offsets[species_index, sorted_vertex_ids]
                    + np.repeat(prior, group_lengths)
                    + within_group
                )
                target_k_all[destinations] = state_k[target_positions]
                source_k_all[destinations] = state_k[source_positions]
                rows_all[destinations] = orbital_by_species[
                    species_index, target_positions
                ]
                columns_all[destinations] = orbital_by_species[
                    species_index, source_positions
                ]
                species_cursors[species_index, group_vertex_ids] += group_lengths

    if not np.array_equal(species_cursors, per_species_counts):
        raise RuntimeError("plane-wave density-vertex fill pass is incomplete")

    vertices: list[SparseDensityVertex] = []
    for index, (q1, q2) in enumerate(zip(q1_labels, q2_labels, strict=True)):
        label = (int(q1), int(q2))
        start = int(label_offsets[index])
        stop = start + int(label_counts[index])
        qvec = (
            (label[0] / 8.0) * complex(b_m1)
            + (label[1] / 4.0) * complex(b_m2)
        )
        weight = float(weight_for_transfer(label, qvec))
        if not math.isfinite(weight):
            raise ValueError(f"non-finite supplied weight for transfer {label}")
        vertices.append(
            SparseDensityVertex(
                label=label,
                target_k=target_k_all[start:stop],
                source_k=source_k_all[start:stop],
                rows=rows_all[start:stop],
                columns=columns_all[start:stop],
                values=values_all[start:stop],
                weight=weight,
                provenance=f"{provenance}; Q-label={label}",
            )
        )
    result = tuple(vertices)
    validate_batch_a_plane_wave_density_vertices(
        result, layout, fold_map, support
    )
    return result

def assemble_primitive_endpoint_h0(
    *,
    lattice: HTQGLattice,
    params: HTQGParams,
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
    endpoint: Literal["ABA", "ABG"],
) -> Array:
    """Assemble exact primitive ``(8,4)`` H0 blocks in no-fold basis order."""

    _validate_batch_a_fold_map(fold_map)
    _validate_batch_a_layout_support(layout, support)
    if not np.array_equal(layout.g_indices, lattice.g_indices):
        raise ValueError("primitive H0 layout must exactly replay lattice.g_indices")
    if fold_map.primitive_mesh != BATCH_A_PRIMITIVE_MESH:
        raise ValueError("primitive H0 requires the exact Batch-A primitive mesh")
    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("primitive H0 support seed must replay layout.g_indices")
    domain = "alpha_beta_alpha" if endpoint == "ABA" else "alpha_beta_gamma"
    primitive_shape = layout.shape[1:]
    primitive_dimension = layout.no_fold_dimension
    out = np.zeros(
        (fold_map.primitive_nk, primitive_dimension, primitive_dimension),
        dtype=np.complex128,
    )
    for p1 in range(BATCH_A_PRIMITIVE_MESH[0]):
        for p2 in range(BATCH_A_PRIMITIVE_MESH[1]):
            k_index = p1 * BATCH_A_PRIMITIVE_MESH[1] + p2
            k_tilde = (p1 / 8.0) * lattice.b_m1 + (p2 / 4.0) * lattice.b_m2
            (r1, r2), fold = fold_map.reduced_coordinate_and_fold((p1, p2))
            reduced_k = r1 * BATCH_A_REDUCED_MESH[1] + r2
            for valley, eta in enumerate(layout.valley_order):
                parent_lattice = _lattice_on_parent_g_indices(
                    lattice,
                    support.parent_g_indices(
                        reduced_k=reduced_k,
                        valley=valley,
                        fold=fold,
                    ),
                )
                parent_entries = build_coupling_table(
                    parent_lattice.g_vectors, parent_lattice.q_vectors
                )
                parent = build_hamiltonian(
                    complex(k_tilde),
                    parent_lattice,
                    params,
                    domain=domain,
                    valley=int(eta),
                    coupling_table=parent_entries,
                )
                for spin in range(2):
                    indices = np.asarray(
                        [
                            np.ravel_multi_index(
                                (g, layer, sublattice, valley, spin),
                                primitive_shape,
                                order="C",
                            )
                            for g in range(layout.g_indices.shape[0])
                            for layer in range(4)
                            for sublattice in range(2)
                        ],
                        dtype=np.int64,
                    )
                    out[k_index][np.ix_(indices, indices)] += parent
    return out


def fold_primitive_blocks(operator: Array, fold_map: BatchAFoldMap) -> Array:
    """Fold ``(8,4,r,r)`` primitive blocks into ``(2,4,4r,4r)``."""

    _validate_batch_a_fold_map(fold_map)
    primitive = np.asarray(operator, dtype=np.complex128)
    if primitive.ndim != 4 or primitive.shape[:2] != BATCH_A_PRIMITIVE_MESH or primitive.shape[2] != primitive.shape[3]:
        raise ValueError("primitive operator must have shape (8,4,r,r)")
    rank = primitive.shape[2]
    folded = np.zeros((2, 4, 4 * rank, 4 * rank), dtype=np.complex128)
    for r1 in range(2):
        for r2 in range(4):
            for fold in range(4):
                p1, p2 = fold_map.primitive_coordinate((r1, r2), fold)
                sl = slice(fold * rank, (fold + 1) * rank)
                folded[r1, r2, sl, sl] = primitive[p1, p2]
    return folded


def unfold_translation_invariant_blocks(
    folded_operator: Array,
    fold_map: BatchAFoldMap,
    *,
    offdiagonal_tolerance: float,
) -> Array:
    """Inverse fold, failing if a fold-offdiagonal block is nonzero."""

    _validate_batch_a_fold_map(fold_map)
    folded = np.asarray(folded_operator, dtype=np.complex128)
    if folded.ndim != 4 or folded.shape[:2] != BATCH_A_REDUCED_MESH or folded.shape[2] != folded.shape[3] or folded.shape[2] % 4:
        raise ValueError("folded operator must have shape (2,4,4r,4r)")
    tolerance = float(offdiagonal_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("offdiagonal_tolerance must be finite and nonnegative")
    rank = folded.shape[2] // 4
    primitive = np.empty((8, 4, rank, rank), dtype=np.complex128)
    for r1 in range(2):
        for r2 in range(4):
            for left in range(4):
                left_sl = slice(left * rank, (left + 1) * rank)
                for right in range(4):
                    right_sl = slice(right * rank, (right + 1) * rank)
                    block = folded[r1, r2, left_sl, right_sl]
                    if left != right and np.max(np.abs(block), initial=0.0) > tolerance:
                        raise ValueError("folded operator breaks primitive translations")
                p1, p2 = fold_map.primitive_coordinate((r1, r2), left)
                primitive[p1, p2] = folded[r1, r2, left_sl, left_sl]
    return primitive


@dataclass(frozen=True)
class PrimitiveSupercellReplayScaffold:
    """Non-authorizing parity record for a caller-supplied interaction action."""

    unfolded_action: Array
    action_max_abs_residual: float
    unfolded_h0: Array
    h0_max_abs_residual: float
    evidence_paths: tuple[str, ...]
    uncertainty: str


def primitive_supercell_interaction_replay_scaffold(
    *,
    primitive_h0: Array,
    primitive_action: Array,
    folded_h0: Array,
    folded_action: Array,
    fold_map: BatchAFoldMap,
    offdiagonal_tolerance: float,
    evidence_paths: tuple[str, ...],
    uncertainty: str,
) -> PrimitiveSupercellReplayScaffold:
    """Compare supplied primitive/folded actions without asserting authority."""

    if not evidence_paths or not str(uncertainty).strip():
        raise ValueError("replay scaffold requires evidence paths and uncertainty")
    unfolded_h0 = unfold_translation_invariant_blocks(
        folded_h0, fold_map, offdiagonal_tolerance=offdiagonal_tolerance
    )
    unfolded_action = unfold_translation_invariant_blocks(
        folded_action, fold_map, offdiagonal_tolerance=offdiagonal_tolerance
    )
    primitive_h = np.asarray(primitive_h0, dtype=np.complex128)
    primitive_i = np.asarray(primitive_action, dtype=np.complex128)
    if unfolded_h0.shape != primitive_h.shape or unfolded_action.shape != primitive_i.shape:
        raise ValueError("primitive and unfolded replay shapes differ")
    return PrimitiveSupercellReplayScaffold(
        unfolded_action=unfolded_action,
        action_max_abs_residual=float(np.max(np.abs(unfolded_action - primitive_i))),
        unfolded_h0=unfolded_h0,
        h0_max_abs_residual=float(np.max(np.abs(unfolded_h0 - primitive_h))),
        evidence_paths=tuple(str(path) for path in evidence_paths),
        uncertainty=str(uncertainty),
    )


__all__ = [
    "BATCH_A_PRIMITIVE_MESH",
    "BATCH_A_REDUCED_MESH",
    "BatchAFoldMap",
    "BatchAPhysicalLabelSupport",
    "GlobalOccupationResult",
    "InteractionAction",
    "InteractionReferences",
    "LITERAL_WICK_MAX_ORBITALS",
    "MicroscopicBasisLayout",
    "MicroscopicH0Spec",
    "MicroscopicInteractionSpec",
    "PrimitiveSupercellReplayScaffold",
    "ProfileFourierInput",
    "SparseDensityVertex",
    "SparseOneBodyOperator",
    "assemble_direct_h0",
    "assemble_direct_h0_dft_oracle",
    "assemble_primitive_endpoint_h0",
    "assemble_uniform_endpoint_h0",
    "batch_a_physical_label_support_sha256",
    "batch_a_supercell",
    "build_batch_a_fold_map",
    "build_batch_a_reduced_k_grid",
    "build_batch_a_tr_covariant_support",
    "build_periodic_smoothstep_profile",
    "build_plane_wave_density_vertices",
    "build_primitive_plane_wave_density_vertices",
    "build_profile_fourier_multiplier",
    "fold_primitive_blocks",
    "global_canonical_occupations",
    "hartree_fock_action",
    "hartree_fock_energy",
    "identity_density_vertex",
    "ket_blocks_to_repository_stored",
    "ket_hamiltonian_to_repository_blocks",
    "literal_wick_four_index_action",
    "literal_wick_four_index_energy",
    "primitive_supercell_interaction_replay_scaffold",
    "repository_hamiltonian_to_ket_blocks",
    "repository_stored_to_ket_blocks",
    "reverse_density_vertex",
    "uniform_h0_replay_residuals",
    "unfold_translation_invariant_blocks",
    "validate_batch_a_plane_wave_density_vertices",
    "validate_density_vertex_family",
    "validate_primitive_supercell_density_vertex_equivalence",
]
