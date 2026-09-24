"""Integer folded-supercell Hartree-Fock adapter for PtSe2 OpenMX active spaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStepResult,
)
from mean_field.core.hf.diis import (
    FixedPointEvaluation,
    PulayDIISOptions,
    PulayDIISStep,
    PulayOperatorDIISStep,
    PulayOperatorEvaluation,
    run_pulay_diis_fixed_point,
    run_pulay_operator_diis,
)
from mean_field.core.hf.overlap import (
    HFOverlapBlockSet,
    compute_density_overlap_trace_from_diagonal,
)
from mean_field.core.hf.orbital_root import (
    OrbitalRootEvaluation,
    run_orbital_rotation_root,
)
from mean_field.core.hf.zero_temperature_stability import (
    ZeroTemperatureOrbitalHessian,
    build_zero_temperature_orbital_hessian,
)
from mean_field.core.hf.interaction import (
    build_projected_interaction_hamiltonian,
    compute_hf_energy,
    run_projected_hartree_fock,
)
from mean_field.core.hf.occupations import calculate_norm_convergence
from mean_field.core.supercell import IntegerSupercell

from .hf import (
    BOHR_INV_TO_NM_INV,
    PtSe2HFState,
    PtSe2ProjectedHFData,
    _hermitize_blocks,
    ptse2_density_from_fixed_filling,
)


@dataclass(frozen=True)
class PtSe2FoldMap:
    primitive_mesh: int
    active_rank: int
    supercell: IntegerSupercell
    fold_representatives: np.ndarray
    fold_shifts: np.ndarray
    reduced_supercell_numerators: np.ndarray
    primitive_indices: np.ndarray
    primitive_to_reduced: np.ndarray
    primitive_to_fold: np.ndarray
    reduced_mesh: int | None = None

    @property
    def primitive_nk(self) -> int:
        return int(self.primitive_mesh * self.primitive_mesh)

    @property
    def reduced_nk(self) -> int:
        return int(self.primitive_indices.shape[0])

    @property
    def area_ratio(self) -> int:
        return int(self.supercell.area_ratio)

    @property
    def folded_rank(self) -> int:
        return int(self.area_ratio * self.active_rank)


@dataclass(frozen=True)
class PtSe2SupercellOverlapCache:
    blocks: HFOverlapBlockSet
    fold_map: PtSe2FoldMap
    pair_masks: dict[tuple[int, int], np.ndarray]
    channel_q_indices: dict[tuple[int, int], np.ndarray]
    assignment_counts: dict[tuple[int, int], np.ndarray]
    source_cache: str
    source_summary: str
    cache_sha256: str
    run_id: str
    hf_authorized: bool


@dataclass(frozen=True)
class PtSe2SupercellHFData:
    primitive_data: PtSe2ProjectedHFData
    fold_map: PtSe2FoldMap
    h0: np.ndarray
    initial_density: np.ndarray
    overlap_cache: PtSe2SupercellOverlapCache
    primitive_area_nm2: float
    supercell_area_nm2: float
    supercell_filling_nu: int


@dataclass(frozen=True)
class PtSe2TranslationObservables:
    absolute_frobenius: float
    relative_frobenius: float
    maximum_fold_offdiagonal_abs: float
    sector_frobenius: dict[tuple[int, int], float]


@dataclass(frozen=True)
class PtSe2SupercellProjectorEvaluation:
    physical_projector: np.ndarray
    stored_density: np.ndarray
    interaction_hamiltonian: np.ndarray
    total_hamiltonian: np.ndarray
    raw_total_hamiltonian_hermiticity: float
    energy: float


@dataclass(frozen=True)
class PtSe2SupercellOrbitalHessianData:
    hessian: ZeroTemperatureOrbitalHessian
    physical_projector: np.ndarray
    stored_density: np.ndarray
    interaction_hamiltonian: np.ndarray
    total_hamiltonian: np.ndarray
    occupied_per_k: int
    fold_count: int


@dataclass(frozen=True)
class PtSe2FixedPointPayload:
    interaction_h: np.ndarray
    total_hamiltonian: np.ndarray
    density_update: DensityUpdateResult
    energy: float
    physical_raw_norm: float | None = None
    level_shift_ev: float = 0.0


def ptse2_2x2_supercell() -> IntegerSupercell:
    return IntegerSupercell(n11=2, n12=0, n21=0, n22=2)


def ptse2_2x1_supercell() -> IntegerSupercell:
    """Representative 2x1 stripe cell: A1=2*a1, A2=a2."""

    return IntegerSupercell(n11=2, n12=0, n21=0, n22=1)


def ptse2_sqrt3_supercell() -> IntegerSupercell:
    """sqrt(3)xsqrt(3) cell for the actual 120-degree OpenMX basis."""

    return IntegerSupercell(n11=1, n12=-1, n21=1, n22=2)


def _supercell_matrix(supercell: IntegerSupercell) -> np.ndarray:
    return np.asarray(
        ((supercell.n11, supercell.n12), (supercell.n21, supercell.n22)),
        dtype=np.int64,
    )


def _integer_row_quotient(
    vector: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray | None:
    """Return integer g with vector=g@matrix, or None if no such g exists."""

    a, b = int(matrix[0, 0]), int(matrix[0, 1])
    c, d = int(matrix[1, 0]), int(matrix[1, 1])
    determinant = a * d - b * c
    if determinant <= 0:
        raise ValueError("quotient matrix must have positive determinant")
    x, y = int(vector[0]), int(vector[1])
    numerator = (x * d - y * c, -x * b + y * a)
    if any(value % determinant != 0 for value in numerator):
        return None
    return np.asarray(
        (numerator[0] // determinant, numerator[1] // determinant),
        dtype=np.int64,
    )


def build_ptse2_fold_map(
    mesh: int,
    active_rank: int,
    supercell: IntegerSupercell,
) -> PtSe2FoldMap:
    """Build the exact image/kernel fold of Z_mesh^2 under p -> p S^T."""

    mesh = int(mesh)
    active_rank = int(active_rank)
    if mesh <= 0:
        raise ValueError("primitive mesh must be positive")
    if active_rank <= 0:
        raise ValueError("active_rank must be positive")
    matrix = _supercell_matrix(supercell)
    area_ratio = int(supercell.area_ratio)
    coordinates = np.asarray(
        [(ix, iy) for ix in range(mesh) for iy in range(mesh)], dtype=np.int64
    )
    transformed = (coordinates @ matrix.T) % mesh
    fold_shifts = np.asarray(
        coordinates[np.all(transformed == 0, axis=1)], dtype=np.int64
    )
    if fold_shifts.shape != (area_ratio, 2):
        raise ValueError(
            "primitive mesh is incompatible with the supercell Smith invariants"
        )
    fold_representatives = np.empty_like(fold_shifts)
    for fold_index, shift in enumerate(fold_shifts):
        numerator = shift @ matrix.T
        if np.any(numerator % mesh != 0):
            raise RuntimeError("supercell kernel shift has no integer reciprocal label")
        fold_representatives[fold_index] = numerator // mesh
    if tuple(fold_representatives[0]) != (0, 0):
        raise RuntimeError("fold ordering must begin with the zero representative")

    grouped: dict[tuple[int, int], list[int]] = {}
    for primitive_index, reduced_numerator in enumerate(transformed):
        key = (int(reduced_numerator[0]), int(reduced_numerator[1]))
        grouped.setdefault(key, []).append(int(primitive_index))
    ordered_keys = tuple(sorted(grouped))
    expected_reduced_nk = mesh * mesh // area_ratio
    if len(ordered_keys) != expected_reduced_nk:
        raise RuntimeError("supercell reduced image has the wrong cardinality")
    primitive_indices = np.empty(
        (expected_reduced_nk, area_ratio), dtype=np.int64
    )
    primitive_to_reduced = np.empty(mesh * mesh, dtype=np.int64)
    primitive_to_fold = np.empty(mesh * mesh, dtype=np.int64)
    for reduced_index, key in enumerate(ordered_keys):
        members = tuple(sorted(grouped[key]))
        if len(members) != area_ratio:
            raise RuntimeError("one reduced supercell fiber has the wrong size")
        member_set = set(members)
        base = coordinates[members[0]]
        for fold_index, shift in enumerate(fold_shifts):
            primitive_coordinate = (base + shift) % mesh
            primitive_index = int(
                primitive_coordinate[0] * mesh + primitive_coordinate[1]
            )
            if primitive_index not in member_set:
                raise RuntimeError("kernel shifts do not reproduce one reduced fiber")
            primitive_indices[reduced_index, fold_index] = primitive_index
            primitive_to_reduced[primitive_index] = reduced_index
            primitive_to_fold[primitive_index] = fold_index
    if np.unique(primitive_indices).size != mesh * mesh:
        raise RuntimeError("supercell fold map is not a primitive-mesh bijection")
    reduced_mesh = (
        mesh // 2
        if supercell == ptse2_2x2_supercell() and mesh % 2 == 0
        else None
    )
    return PtSe2FoldMap(
        primitive_mesh=mesh,
        active_rank=active_rank,
        supercell=supercell,
        fold_representatives=fold_representatives,
        fold_shifts=fold_shifts,
        reduced_supercell_numerators=np.asarray(ordered_keys, dtype=np.int64),
        primitive_indices=primitive_indices,
        primitive_to_reduced=primitive_to_reduced,
        primitive_to_fold=primitive_to_fold,
        reduced_mesh=reduced_mesh,
    )



def _folded_indices(
    fold_index: int,
    active_rank: int,
    area_ratio: int,
) -> np.ndarray:
    """Active-major flattened indices alpha=area_ratio*a+fold."""

    return (
        int(area_ratio) * np.arange(int(active_rank), dtype=np.int64)
        + int(fold_index)
    )


def fold_ptse2_primitive_operator(
    operator: np.ndarray,
    fold_map: PtSe2FoldMap,
) -> np.ndarray:
    primitive = np.asarray(operator, dtype=np.complex128)
    rank = int(fold_map.active_rank)
    if primitive.shape != (rank, rank, fold_map.primitive_nk):
        raise ValueError("primitive operator shape is incompatible with the fold map")
    folded = np.zeros(
        (fold_map.folded_rank, fold_map.folded_rank, fold_map.reduced_nk),
        dtype=np.complex128,
    )
    for reduced_index in range(fold_map.reduced_nk):
        for fold_index in range(fold_map.area_ratio):
            active = _folded_indices(
                fold_index, rank, fold_map.area_ratio
            )
            primitive_index = int(fold_map.primitive_indices[reduced_index, fold_index])
            folded[:, :, reduced_index][np.ix_(active, active)] = primitive[
                :, :, primitive_index
            ]
    return folded


def embed_ptse2_translation_invariant_density(
    primitive_density: np.ndarray,
    fold_map: PtSe2FoldMap,
) -> np.ndarray:
    return fold_ptse2_primitive_operator(primitive_density, fold_map)


def unfold_ptse2_translation_invariant_operator(
    folded_operator: np.ndarray,
    fold_map: PtSe2FoldMap,
    *,
    offdiagonal_tolerance: float = 1.0e-12,
) -> np.ndarray:
    folded = np.asarray(folded_operator, dtype=np.complex128)
    if folded.shape != (
        fold_map.folded_rank,
        fold_map.folded_rank,
        fold_map.reduced_nk,
    ):
        raise ValueError("folded operator shape is incompatible with the fold map")
    tolerance = float(offdiagonal_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("offdiagonal_tolerance must be finite and nonnegative")
    observables = ptse2_translation_observables(folded, fold_map)
    if observables.maximum_fold_offdiagonal_abs > tolerance:
        raise ValueError("folded operator breaks primitive translations")
    rank = int(fold_map.active_rank)
    primitive = np.empty((rank, rank, fold_map.primitive_nk), dtype=np.complex128)
    for reduced_index in range(fold_map.reduced_nk):
        for fold_index in range(fold_map.area_ratio):
            active = _folded_indices(
                fold_index, rank, fold_map.area_ratio
            )
            primitive_index = int(fold_map.primitive_indices[reduced_index, fold_index])
            primitive[:, :, primitive_index] = folded[:, :, reduced_index][
                np.ix_(active, active)
            ]
    return primitive


def _centered_supercell_steps(fold_map: PtSe2FoldMap) -> np.ndarray:
    mesh = int(fold_map.primitive_mesh)
    coordinates = np.asarray(
        fold_map.reduced_supercell_numerators, dtype=np.int64
    )
    steps = np.empty(
        (fold_map.reduced_nk, fold_map.reduced_nk, 2), dtype=np.int64
    )
    half = mesh // 2
    for target_index, target in enumerate(coordinates):
        for source_index, source in enumerate(coordinates):
            raw = target - source
            steps[target_index, source_index] = (raw + half) % mesh - half
    return steps


def build_ptse2_supercell_overlap_cache(
    primitive_data: PtSe2ProjectedHFData,
    supercell: IntegerSupercell,
) -> PtSe2SupercellOverlapCache:
    primitive_cache = primitive_data.overlap_cache
    mesh = int(round(np.sqrt(primitive_cache.blocks.overlaps[
        primitive_cache.blocks.shifts[0]
    ].shape[1])))
    if mesh * mesh != primitive_data.h0.shape[2]:
        raise ValueError("primitive overlap cache is not one square mesh")
    expected_k = np.asarray(
        [
            (ix / mesh, iy / mesh, 0.0)
            for ix in range(mesh)
            for iy in range(mesh)
        ],
        dtype=np.float64,
    )
    actual_k = np.asarray(primitive_data.k_fractional, dtype=np.float64)
    if actual_k.shape != expected_k.shape or not np.allclose(
        actual_k, expected_k, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("primitive data does not use canonical ix*mesh+iy ordering")
    rank = int(primitive_cache.active_rank)
    fold_map = build_ptse2_fold_map(mesh, rank, supercell)
    reduced_steps = _centered_supercell_steps(fold_map)
    q_numerators = np.asarray(
        primitive_cache.physical_q_numerators, dtype=np.int64
    )
    if np.any(q_numerators[:, 2] != 0):
        raise ValueError("supercell adapter currently supports in-plane physical Q only")
    matrix = _supercell_matrix(supercell)

    records: list[tuple[tuple[int, int], int, int, int, int, int, int, int]] = []
    labels: set[tuple[int, int]] = set()
    primitive_channel_count = 0
    for shift in primitive_cache.blocks.shifts:
        mask = np.asarray(primitive_cache.pair_masks[shift], dtype=bool)
        q_table = np.asarray(primitive_cache.channel_q_indices[shift], dtype=np.int64)
        for primitive_target, primitive_source in np.argwhere(mask):
            primitive_target = int(primitive_target)
            primitive_source = int(primitive_source)
            target_reduced = int(fold_map.primitive_to_reduced[primitive_target])
            source_reduced = int(fold_map.primitive_to_reduced[primitive_source])
            target_fold = int(fold_map.primitive_to_fold[primitive_target])
            source_fold = int(fold_map.primitive_to_fold[primitive_source])
            q_index = int(q_table[primitive_target, primitive_source])
            if q_index < 0:
                raise RuntimeError("admitted primitive channel has no physical-Q index")
            difference = (
                q_numerators[q_index, :2] @ matrix.T
                - reduced_steps[target_reduced, source_reduced]
            )
            if np.any(difference % mesh != 0):
                raise RuntimeError(
                    "primitive channel does not map to an integer supercell field"
                )
            label_array = difference // mesh
            label = (int(label_array[0]), int(label_array[1]))
            labels.add(label)
            records.append(
                (
                    label,
                    primitive_target,
                    primitive_source,
                    target_reduced,
                    source_reduced,
                    target_fold,
                    source_fold,
                    q_index,
                )
            )
            primitive_channel_count += 1

    ordered_labels = tuple(
        sorted(labels, key=lambda item: (item[0] * item[0] + item[1] * item[1], item))
    )
    folded_rank = int(fold_map.folded_rank)
    reduced_nk = int(fold_map.reduced_nk)
    overlaps = {
        label: np.zeros(
            (folded_rank, reduced_nk, folded_rank, reduced_nk),
            dtype=np.complex128,
        )
        for label in ordered_labels
    }
    pair_masks = {
        label: np.zeros((reduced_nk, reduced_nk), dtype=bool)
        for label in ordered_labels
    }
    channel_q_indices = {
        label: np.full((reduced_nk, reduced_nk), -1, dtype=np.int64)
        for label in ordered_labels
    }
    assignment_counts = {
        label: np.zeros((reduced_nk, reduced_nk), dtype=np.int64)
        for label in ordered_labels
    }
    fold_assignment = {
        label: np.zeros(
            (
                reduced_nk,
                reduced_nk,
                fold_map.area_ratio,
                fold_map.area_ratio,
            ),
            dtype=bool,
        )
        for label in ordered_labels
    }
    primitive_overlaps = primitive_cache.blocks.overlaps
    shift_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for shift in primitive_cache.blocks.shifts:
        mask = np.asarray(primitive_cache.pair_masks[shift], dtype=bool)
        for primitive_target, primitive_source in np.argwhere(mask):
            key = (int(primitive_target), int(primitive_source))
            q_index = int(
                primitive_cache.channel_q_indices[shift][
                    primitive_target, primitive_source
                ]
            )
            compound = (key[0], key[1], q_index)
            if compound in shift_lookup:
                raise RuntimeError("duplicate primitive pair/Q channel in local-field inventory")
            shift_lookup[compound] = shift

    assigned = 0
    for (
        label,
        primitive_target,
        primitive_source,
        target_reduced,
        source_reduced,
        target_fold,
        source_fold,
        q_index,
    ) in records:
        compound = (primitive_target, primitive_source, q_index)
        shift = shift_lookup[compound]
        if fold_assignment[label][
            target_reduced, source_reduced, target_fold, source_fold
        ]:
            raise RuntimeError("duplicate folded overlap assignment")
        existing_q = int(channel_q_indices[label][target_reduced, source_reduced])
        if existing_q not in (-1, q_index):
            raise RuntimeError("one supercell pair/local field maps to multiple physical Q")
        target_active = _folded_indices(
            target_fold, rank, fold_map.area_ratio
        )
        source_active = _folded_indices(
            source_fold, rank, fold_map.area_ratio
        )
        overlaps[label][np.ix_(
            target_active,
            [target_reduced],
            source_active,
            [source_reduced],
        )] = primitive_overlaps[shift][
            :, primitive_target, :, primitive_source
        ][:, None, :, None]
        fold_assignment[label][
            target_reduced, source_reduced, target_fold, source_fold
        ] = True
        pair_masks[label][target_reduced, source_reduced] = True
        channel_q_indices[label][target_reduced, source_reduced] = q_index
        assignment_counts[label][target_reduced, source_reduced] += 1
        assigned += 1
    if assigned != primitive_channel_count:
        raise RuntimeError("folded channel coverage differs from primitive coverage")
    for label in ordered_labels:
        active_counts = assignment_counts[label][pair_masks[label]]
        if active_counts.size and not np.all(
            active_counts == fold_map.area_ratio
        ):
            raise RuntimeError(
                "each admitted supercell pair must contain area_ratio fold assignments"
            )

    q_norm_nm_inv = (
        np.linalg.norm(primitive_cache.physical_q_bohr_inv, axis=1)
        * BOHR_INV_TO_NM_INV
    )
    # Recover W(Q) from any primitive Fock table, preserving the exact screening
    # values already qualified in the primitive bridge.
    w_by_q = np.full(q_numerators.shape[0], np.nan, dtype=np.float64)
    for shift in primitive_cache.blocks.shifts:
        mask = primitive_cache.pair_masks[shift]
        q_table = primitive_cache.channel_q_indices[shift]
        kernel = primitive_cache.blocks.fock_screening[shift]
        existing = w_by_q[q_table[mask]]
        incoming = kernel[mask]
        if np.any(np.isfinite(existing) & (np.abs(existing - incoming) > 1.0e-12)):
            raise RuntimeError("primitive screening values disagree for one physical Q")
        w_by_q[q_table[mask]] = incoming
    if np.any(~np.isfinite(w_by_q)):
        raise RuntimeError("primitive screening tables do not cover every physical Q")
    if np.any(q_norm_nm_inv < 0.0):
        raise RuntimeError("invalid physical-Q norms")

    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for label in ordered_labels:
        diagonal_overlaps[label] = np.diagonal(
            overlaps[label], axis1=1, axis2=3
        )
        q_table = channel_q_indices[label]
        mask = pair_masks[label]
        kernel = np.zeros((reduced_nk, reduced_nk), dtype=np.float64)
        kernel[mask] = w_by_q[q_table[mask]]
        fock_screening[label] = kernel
        diagonal_mask = np.diag(mask)
        if np.all(diagonal_mask):
            diagonal_q = np.diag(q_table)
            if np.unique(diagonal_q).size != 1:
                raise RuntimeError("supercell Hartree label does not use one physical Q")
            hartree_screening[label] = float(w_by_q[int(diagonal_q[0])])
        elif np.any(diagonal_mask):
            raise RuntimeError("partially admitted supercell Hartree label")

    local_fields = (
        np.asarray(ordered_labels, dtype=np.float64)
        @ np.linalg.inv(matrix).T
    )
    local_cart_bohr = np.column_stack(
        (local_fields, np.zeros(len(ordered_labels), dtype=np.float64))
    ) @ primitive_data.source.reciprocal_bohr
    local_cart_nm = local_cart_bohr * BOHR_INV_TO_NM_INV
    gvecs = local_cart_nm[:, 0] + 1j * local_cart_nm[:, 1]
    blocks = HFOverlapBlockSet(
        shifts=ordered_labels,
        gvecs=np.asarray(gvecs, dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )
    return PtSe2SupercellOverlapCache(
        blocks=blocks,
        fold_map=fold_map,
        pair_masks=pair_masks,
        channel_q_indices=channel_q_indices,
        assignment_counts=assignment_counts,
        source_cache=primitive_cache.source_cache,
        source_summary=primitive_cache.source_summary,
        cache_sha256=primitive_cache.cache_sha256,
        run_id=primitive_cache.run_id,
        hf_authorized=primitive_cache.hf_authorized,
    )



def build_ptse2_supercell_hf_data(
    primitive_data: PtSe2ProjectedHFData,
    supercell: IntegerSupercell,
    *,
    target_primitive_filling_nu: float | None = None,
) -> PtSe2SupercellHFData:
    """Build a folded HF problem at a commensurate primitive-cell filling.

    ``primitive_data`` supplies the physical h0/form-factor model.  A rational
    target filling is admissible only when multiplying it by the integer-cell
    area gives an integer supercell filling, so the folded problem never
    realizes a rational filling by splitting an arbitrary finite-mesh subset.
    """
    primitive_filling = (
        float(primitive_data.config.filling_nu)
        if target_primitive_filling_nu is None
        else float(target_primitive_filling_nu)
    )
    if not np.isfinite(primitive_filling):
        raise ValueError("target primitive filling must be finite")
    active_rank = int(primitive_data.config.active_rank)
    if not -active_rank <= primitive_filling <= 0.0:
        raise ValueError("target primitive filling must lie in [-active_rank, 0]")
    folded_filling = float(supercell.area_ratio) * primitive_filling
    supercell_filling = int(np.rint(folded_filling))
    if abs(folded_filling - supercell_filling) > 1.0e-12:
        raise ValueError(
            "target primitive filling is incommensurate with the integer supercell"
        )
    overlap_cache = build_ptse2_supercell_overlap_cache(
        primitive_data, supercell
    )
    fold_map = overlap_cache.fold_map
    h0 = fold_ptse2_primitive_operator(primitive_data.h0, fold_map)
    initial = ptse2_density_from_fixed_filling(
        h0,
        supercell_filling,
        degeneracy_tolerance_ev=primitive_data.config.occupation_degeneracy_tolerance_ev,
        degenerate_occupation_policy=primitive_data.config.degenerate_occupation_policy,
    ).density
    return PtSe2SupercellHFData(
        primitive_data=primitive_data,
        fold_map=fold_map,
        h0=h0,
        initial_density=np.asarray(initial, dtype=np.complex128),
        overlap_cache=overlap_cache,
        primitive_area_nm2=float(primitive_data.area_nm2),
        supercell_area_nm2=(
            float(fold_map.area_ratio) * float(primitive_data.area_nm2)
        ),
        supercell_filling_nu=supercell_filling,
    )



def ptse2_supercell_label_sector(
    label: tuple[int, int],
    fold_map: PtSe2FoldMap,
) -> tuple[int, int]:
    """Canonical quotient class in Z^2 / (Z^2 S^T)."""

    vector = np.asarray(label, dtype=np.int64)
    matrix_transpose = _supercell_matrix(fold_map.supercell).T
    for representative in fold_map.fold_representatives:
        if _integer_row_quotient(
            vector - representative, matrix_transpose
        ) is not None:
            return (int(representative[0]), int(representative[1]))
    raise RuntimeError("supercell label has no quotient-sector representative")


def ptse2_supercell_label_breaks_translation(
    label: tuple[int, int],
    fold_map: PtSe2FoldMap,
) -> bool:
    return ptse2_supercell_label_sector(label, fold_map) != (0, 0)


def _physical_cdw_seed_hamiltonian(
    data: PtSe2SupercellHFData,
    *,
    sectors: tuple[tuple[int, int], ...],
    amplitude_ev: float,
    seed: int,
) -> np.ndarray:
    amplitude = float(amplitude_ev)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("CDW seed amplitude must be finite and positive")
    if not sectors:
        raise ValueError("at least one CDW sector is required")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed:
        raise ValueError("CDW seed must be an integer")
    rng = np.random.default_rng(int(seed))
    perturbation = np.zeros_like(data.h0)
    for sector in sectors:
        label = (int(sector[0]), int(sector[1]))
        if not ptse2_supercell_label_breaks_translation(
            label, data.fold_map
        ):
            raise ValueError("CDW seed sector must break one primitive translation")
        if label not in data.overlap_cache.blocks.hartree_screening:
            raise ValueError(
                f"CDW seed sector {label} lacks a fully admitted static vertex"
            )
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        vertex = data.overlap_cache.blocks.diagonal_overlaps[label]
        if np.linalg.norm(vertex) <= 1.0e-14:
            raise ValueError(f"CDW seed sector {label} has a zero static vertex")
        perturbation += 0.5 * amplitude * (
            np.exp(1j * phase) * vertex
            + np.exp(-1j * phase) * np.swapaxes(vertex.conj(), 0, 1)
        )
    _hermitize_blocks(perturbation)
    return data.h0 + perturbation


def initialize_ptse2_supercell_density(
    data: PtSe2SupercellHFData,
    *,
    mode: str,
    seed: int = 1,
    amplitude_ev: float = 1.0e-3,
    primitive_density: np.ndarray | None = None,
    sectors: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"bare", "noninteracting"}:
        return np.array(data.initial_density, copy=True)
    if normalized in {"translation_invariant", "embedded"}:
        if primitive_density is None:
            raise ValueError(
                "translation-invariant initialization requires primitive_density"
            )
        return embed_ptse2_translation_invariant_density(
            primitive_density, data.fold_map
        )
    if sectors is None or not sectors:
        raise ValueError("a generic CDW initialization requires explicit sectors")
    seeded_hamiltonian = _physical_cdw_seed_hamiltonian(
        data,
        sectors=tuple((int(a), int(b)) for a, b in sectors),
        amplitude_ev=amplitude_ev,
        seed=seed,
    )
    density = ptse2_density_from_fixed_filling(
        seeded_hamiltonian,
        data.supercell_filling_nu,
        degeneracy_tolerance_ev=data.primitive_data.config.occupation_degeneracy_tolerance_ev,
        degenerate_occupation_policy=data.primitive_data.config.degenerate_occupation_policy,
    ).density
    identity = np.eye(data.fold_map.folded_rank, dtype=np.complex128)
    measured_filling = 0.0
    for reduced_index in range(data.fold_map.reduced_nk):
        projector = density[:, :, reduced_index] + identity
        eigenvalues = np.linalg.eigvalsh(projector)
        if eigenvalues[0] < -1.0e-10 or eigenvalues[-1] > 1.0 + 1.0e-10:
            raise RuntimeError("CDW seed projector is outside [0,1]")
        measured_filling += float(np.trace(density[:, :, reduced_index]).real)
    measured_filling /= data.fold_map.reduced_nk
    if abs(measured_filling - data.supercell_filling_nu) > 1.0e-10:
        raise RuntimeError("CDW seed density has the wrong filling")
    observables = ptse2_translation_observables(density, data.fold_map)
    requested_classes = {
        ptse2_supercell_label_sector(label, data.fold_map)
        for label in sectors
    }
    if any(
        observables.sector_frobenius.get(sector, 0.0) <= 1.0e-12
        for sector in requested_classes
    ):
        raise RuntimeError("CDW seed did not generate its requested translation sector")
    return density


def initialize_ptse2_2x2_density(
    data: PtSe2SupercellHFData,
    *,
    mode: str,
    seed: int = 1,
    amplitude_ev: float = 1.0e-3,
    primitive_density: np.ndarray | None = None,
) -> np.ndarray:
    if data.fold_map.supercell != ptse2_2x2_supercell():
        raise ValueError("2x2 initializer requires 2x2 supercell data")
    normalized = str(mode).strip().lower().replace("-", "_")
    sector_lookup = {
        "cdw_10": ((1, 0),),
        "cdw_01": ((0, 1),),
        "cdw_11": ((1, 1),),
        "cdw_mixed": ((1, 0), (0, 1), (1, 1)),
    }
    if normalized not in {
        "bare",
        "noninteracting",
        "translation_invariant",
        "embedded",
        *sector_lookup,
    }:
        raise ValueError(
            "unsupported 2x2 initialization mode"
        )
    return initialize_ptse2_supercell_density(
        data,
        mode=normalized,
        seed=seed,
        amplitude_ev=amplitude_ev,
        primitive_density=primitive_density,
        sectors=sector_lookup.get(normalized),
    )


def run_ptse2_supercell_projected_hf(
    data: PtSe2SupercellHFData,
    *,
    initial_density: np.ndarray,
    init_mode: str,
    seed: int,
    step_callback: Callable[[PtSe2HFState, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[PtSe2HFState, DensityUpdateResult], None] | None = None,
    max_oda_lambda: float | None = None,
) -> HartreeFockRun:
    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if density.shape != data.h0.shape:
        raise ValueError("supercell initial density has the wrong shape")
    for reduced_index in range(data.fold_map.reduced_nk):
        block = density[:, :, reduced_index]
        if np.linalg.norm(block - block.conj().T) > 1.0e-10:
            raise ValueError("supercell initial density is not Hermitian")
    if not str(init_mode).strip():
        raise ValueError("supercell init_mode must be nonempty")
    if int(seed) != seed:
        raise ValueError("supercell seed must be an integer")
    identity = np.eye(data.fold_map.folded_rank, dtype=np.complex128)
    measured_filling = 0.0
    for reduced_index in range(data.fold_map.reduced_nk):
        projector = density[:, :, reduced_index] + identity
        eigenvalues = np.linalg.eigvalsh(projector)
        if eigenvalues[0] < -1.0e-10 or eigenvalues[-1] > 1.0 + 1.0e-10:
            raise ValueError("supercell initial projector is outside [0,1]")
        measured_filling += float(np.trace(density[:, :, reduced_index]).real)
    measured_filling /= data.fold_map.reduced_nk
    if abs(measured_filling - data.supercell_filling_nu) > 1.0e-10:
        raise ValueError("supercell initial density has the wrong filling")

    state = PtSe2HFState(
        h0=np.array(data.h0, copy=True),
        density=np.array(density, copy=True),
        hamiltonian=np.array(data.h0, copy=True),
        energies=np.empty((data.fold_map.folded_rank, data.fold_map.reduced_nk)),
        precision=float(data.primitive_data.config.precision),
        v0=1.0 / float(data.supercell_area_nm2),
    )

    def initializer(target: PtSe2HFState, *, init_mode: str, seed: int) -> None:
        target.density[:, :, :] = density

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return ptse2_density_from_fixed_filling(
            hamiltonian,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=data.primitive_data.config.occupation_degeneracy_tolerance_ev,
            degenerate_occupation_policy=data.primitive_data.config.degenerate_occupation_policy,
        )

    config = data.primitive_data.config
    return run_projected_hartree_fock(
        state,
        initializer=initializer,
        density_builder=density_builder,
        overlap_blocks=data.overlap_cache.blocks,
        init_mode=str(init_mode),
        seed=int(seed),
        v0=state.v0,
        oda_parameterizer=("default" if config.oda_mode == "default" else None),
        hamiltonian_postprocessor=_hermitize_blocks,
        density_postprocessor=_hermitize_blocks,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_rule=config.convergence_rule,
        max_iter=int(config.max_iter),
        max_oda_lambda=max_oda_lambda,
        use_numba=bool(config.use_numba),
    )


def run_ptse2_supercell_diis_hf(
    data: PtSe2SupercellHFData,
    *,
    initial_density: np.ndarray,
    init_mode: str,
    seed: int,
    options: PulayDIISOptions | None = None,
    step_callback: Callable[
        [PulayDIISStep, FixedPointEvaluation[PtSe2FixedPointPayload]], None
    ]
    | None = None,
) -> HartreeFockRun:
    """Recover a zero-temperature supercell HF fixed point with Pulay DIIS."""

    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if density.shape != data.h0.shape:
        raise ValueError("supercell DIIS initial density has the wrong shape")
    target_trace = float(data.supercell_filling_nu * data.fold_map.reduced_nk)
    identity = np.eye(data.fold_map.folded_rank, dtype=np.complex128)

    def validate_iterate(candidate: np.ndarray) -> None:
        hermiticity = float(
            np.max(
                np.abs(candidate - np.swapaxes(candidate.conj(), 0, 1)),
                initial=0.0,
            )
        )
        if hermiticity > 1.0e-9:
            raise ValueError("DIIS iterate is not Hermitian")
        trace = float(
            np.trace(candidate, axis1=0, axis2=1).real.sum()
        )
        if abs(trace - target_trace) > 1.0e-8:
            raise ValueError("DIIS iterate does not preserve global filling")

    def validate_output(candidate: np.ndarray) -> None:
        validate_iterate(candidate)
        for reduced_index in range(data.fold_map.reduced_nk):
            projector = candidate[:, :, reduced_index] + identity
            eigenvalues = np.linalg.eigvalsh(projector)
            if eigenvalues[0] < -1.0e-9 or eigenvalues[-1] > 1.0 + 1.0e-9:
                raise ValueError("DIIS map output projector is outside [0,1]")

    def evaluate(candidate: np.ndarray) -> FixedPointEvaluation[PtSe2FixedPointPayload]:
        interaction_h = build_projected_interaction_hamiltonian(
            candidate,
            data.overlap_cache.blocks,
            v0=1.0 / float(data.supercell_area_nm2),
            use_numba=bool(data.primitive_data.config.use_numba),
        )
        total_hamiltonian = np.asarray(
            data.h0 + interaction_h, dtype=np.complex128
        )
        _hermitize_blocks(total_hamiltonian)
        update = ptse2_density_from_fixed_filling(
            total_hamiltonian,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=(
                data.primitive_data.config.occupation_degeneracy_tolerance_ev
            ),
            degenerate_occupation_policy=(
                data.primitive_data.config.degenerate_occupation_policy
            ),
        )
        mapped = np.asarray(update.density, dtype=np.complex128).copy()
        _hermitize_blocks(mapped)
        energy = compute_hf_energy(interaction_h, data.h0, candidate)
        return FixedPointEvaluation(
            output=mapped,
            payload=PtSe2FixedPointPayload(
                interaction_h=np.asarray(interaction_h, dtype=np.complex128),
                total_hamiltonian=total_hamiltonian,
                density_update=update,
                energy=float(energy),
            ),
        )

    iter_energy: list[float] = []

    def on_step(
        step: PulayDIISStep,
        evaluation: FixedPointEvaluation[PtSe2FixedPointPayload],
    ) -> None:
        iter_energy.append(float(evaluation.payload.energy))
        if step_callback is not None:
            step_callback(step, evaluation)

    result = run_pulay_diis_fixed_point(
        density,
        evaluate,
        precision=float(data.primitive_data.config.precision),
        max_iter=int(data.primitive_data.config.max_iter),
        options=options,
        validate_output=validate_output,
        validate_iterate=validate_iterate,
        step_callback=on_step,
    )
    projector_density = np.asarray(
        result.final_evaluation.output, dtype=np.complex128
    ).copy()
    validate_output(projector_density)
    closure = evaluate(projector_density)
    closure_norm = float(
        calculate_norm_convergence(closure.output, projector_density)
    )
    precision = float(data.primitive_data.config.precision)
    closure_converged = closure_norm <= precision
    converged = bool(result.converged and closure_converged)
    exit_reason = str(result.exit_reason)
    if result.converged and not closure_converged:
        exit_reason = "projector_closure_failed"
    elif not result.converged and closure_converged:
        converged = True
        exit_reason = "converged_projector_closure"
    final_payload = closure.payload
    state = PtSe2HFState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=projector_density,
        hamiltonian=np.asarray(
            final_payload.total_hamiltonian, dtype=np.complex128
        ).copy(),
        energies=np.asarray(
            final_payload.density_update.energies, dtype=np.float64
        ).copy(),
        precision=precision,
        v0=1.0 / float(data.supercell_area_nm2),
    )
    state.mu = float(final_payload.density_update.mu)
    state.diagnostics["hf_energy"] = float(final_payload.energy)
    state.diagnostics["final_raw_norm"] = closure_norm
    state.diagnostics["iterations"] = float(result.iterations)
    state.diagnostics["diis_restart_count"] = float(result.restart_count)
    state.diagnostics["diis_candidate_raw_norm"] = float(
        result.final_residual_norm
    )
    state.diagnostics["diis_projector_closure_raw_norm"] = closure_norm
    iter_energy.append(float(final_payload.energy))
    iter_err = np.append(result.iter_residual, closure_norm)
    return HartreeFockRun(
        state=state,
        iter_energy=np.asarray(iter_energy, dtype=np.float64),
        iter_err=np.asarray(iter_err, dtype=np.float64),
        iter_oda=np.full(iter_err.size, np.nan, dtype=np.float64),
        init_mode=str(init_mode),
        seed=int(seed),
        converged=converged,
        exit_reason=exit_reason,
    )


def evaluate_ptse2_supercell_physical_projector(
    data: PtSe2SupercellHFData,
    physical_projector: np.ndarray,
) -> PtSe2SupercellProjectorEvaluation:
    """Rebuild physical H and scalar energy on a supercell ket projector."""

    projector = np.asarray(physical_projector, dtype=np.complex128)
    if projector.shape != data.h0.shape or not np.all(np.isfinite(projector)):
        raise ValueError("physical supercell projector has invalid shape or values")
    identity = np.eye(data.fold_map.folded_rank, dtype=np.complex128)
    for k_index in range(data.fold_map.reduced_nk):
        block = projector[:, :, k_index]
        if np.linalg.norm(block - block.conj().T) > 1.0e-9:
            raise ValueError("physical supercell projector is not Hermitian")
        if np.linalg.norm(block @ block - block) > 1.0e-8:
            raise ValueError("physical supercell projector is not idempotent")
        trace = float(np.trace(block).real)
        if abs(trace - round(trace)) > 1.0e-8:
            raise ValueError("physical supercell projector has noninteger rank")
    stored_density = np.swapaxes(projector, 0, 1) - identity[:, :, None]
    interaction_h = build_projected_interaction_hamiltonian(
        stored_density,
        data.overlap_cache.blocks,
        v0=1.0 / float(data.supercell_area_nm2),
        use_numba=bool(data.primitive_data.config.use_numba),
    )
    total_hamiltonian = np.asarray(data.h0 + interaction_h, dtype=np.complex128)
    raw_hermiticity = float(
        np.max(
            np.abs(
                total_hamiltonian
                - np.swapaxes(total_hamiltonian.conj(), 0, 1)
            ),
            initial=0.0,
        )
    )
    if raw_hermiticity > 1.0e-9:
        raise ValueError("raw PtSe2 total Hamiltonian is not Hermitian")
    _hermitize_blocks(total_hamiltonian)
    energy = compute_hf_energy(interaction_h, data.h0, stored_density)
    return PtSe2SupercellProjectorEvaluation(
        physical_projector=np.asarray(projector, dtype=np.complex128),
        stored_density=np.asarray(stored_density, dtype=np.complex128),
        interaction_hamiltonian=np.asarray(interaction_h, dtype=np.complex128),
        total_hamiltonian=np.asarray(total_hamiltonian, dtype=np.complex128),
        raw_total_hamiltonian_hermiticity=raw_hermiticity,
        energy=float(energy),
    )


def build_ptse2_supercell_zero_temperature_hessian(
    data: PtSe2SupercellHFData,
    stored_density: np.ndarray,
    *,
    stationarity_tolerance_ev: float | None = None,
) -> PtSe2SupercellOrbitalHessianData:
    """Bind the generic zero-T orbital Hessian to PtSe2 stored densities."""

    density = np.asarray(stored_density, dtype=np.complex128)
    if density.shape != data.h0.shape or not np.all(np.isfinite(density)):
        raise ValueError("stored supercell density has invalid shape or values")
    n = data.fold_map.folded_rank
    nk = data.fold_map.reduced_nk
    identity = np.eye(n, dtype=np.complex128)
    physical_projector = np.empty_like(density)
    occupied_counts: list[int] = []
    for k_index in range(nk):
        physical_projector[:, :, k_index] = (
            density[:, :, k_index] + identity
        ).T
        eigenvalues = np.linalg.eigvalsh(physical_projector[:, :, k_index])
        occupied_counts.append(int(np.count_nonzero(eigenvalues > 0.5)))
    if len(set(occupied_counts)) != 1:
        raise ValueError("zero-T Hessian requires a uniform occupied rank per k")
    occupied_per_k = occupied_counts[0]
    evaluation = evaluate_ptse2_supercell_physical_projector(
        data, physical_projector
    )
    interaction_h = evaluation.interaction_hamiltonian
    total_hamiltonian = evaluation.total_hamiltonian

    def hamiltonian_response(delta_projector: np.ndarray) -> np.ndarray:
        delta_stored = np.swapaxes(
            np.asarray(delta_projector, dtype=np.complex128), 0, 1
        )
        return build_projected_interaction_hamiltonian(
            delta_stored,
            data.overlap_cache.blocks,
            v0=1.0 / float(data.supercell_area_nm2),
            use_numba=bool(data.primitive_data.config.use_numba),
        )

    hessian = build_zero_temperature_orbital_hessian(
        physical_projector,
        total_hamiltonian,
        hamiltonian_response,
        occupied_per_k=occupied_per_k,
        k_weights=np.full(nk, 1.0 / nk, dtype=np.float64),
        stationarity_tolerance_ev=stationarity_tolerance_ev,
    )
    return PtSe2SupercellOrbitalHessianData(
        hessian=hessian,
        physical_projector=np.asarray(physical_projector, dtype=np.complex128),
        stored_density=np.asarray(density, dtype=np.complex128),
        interaction_hamiltonian=np.asarray(interaction_h, dtype=np.complex128),
        total_hamiltonian=np.asarray(total_hamiltonian, dtype=np.complex128),
        occupied_per_k=occupied_per_k,
        fold_count=int(data.fold_map.area_ratio),
    )


def ptse2_translation_breaking_orbital_action(
    binding: PtSe2SupercellOrbitalHessianData,
    weighted_vector: np.ndarray,
) -> np.ndarray:
    """Project an orbital tangent onto all nontrivial fold coherences."""

    hessian = binding.hessian
    frame = hessian.frame
    vector = np.asarray(weighted_vector, dtype=np.float64)
    if vector.shape != (frame.real_size,):
        raise ValueError("weighted orbital vector has the wrong shape")
    coordinates = frame.unpack_weighted_real(vector)
    tangent = frame.tangent_projector(coordinates)
    # PtSe2 folded indices use the active-major ABI alpha = d*a + fold.
    fold_indices = np.arange(frame.n, dtype=np.int64) % binding.fold_count
    breaking_mask = fold_indices[:, None] != fold_indices[None, :]
    tangent *= breaking_mask[:, :, None]
    projected_coordinates = np.empty_like(coordinates)
    for k_index in range(frame.nk):
        occupied = frame.basis[:, : frame.nocc, k_index]
        virtual = frame.basis[:, frame.nocc :, k_index]
        projected_coordinates[:, :, k_index] = (
            virtual.conj().T @ tangent[:, :, k_index] @ occupied
        )
    return frame.pack_weighted_complex(projected_coordinates)


def ptse2_supercell_energy_from_physical_projector(
    data: PtSe2SupercellHFData,
    physical_projector: np.ndarray,
) -> float:
    """Evaluate the exact projected-HF scalar energy on a physical projector."""

    return float(
        evaluate_ptse2_supercell_physical_projector(
            data, physical_projector
        ).energy
    )


def ptse2_stored_density_commutator(
    hamiltonian: np.ndarray,
    stored_density: np.ndarray,
) -> np.ndarray:
    """Return ``[H, P_ket]`` for ``D_stored = P_ket.T - I``."""

    hvalue = np.asarray(hamiltonian, dtype=np.complex128)
    dvalue = np.asarray(stored_density, dtype=np.complex128)
    if hvalue.shape != dvalue.shape or hvalue.ndim != 3:
        raise ValueError("Hamiltonian and stored density must share shape (n,n,nk)")
    if hvalue.shape[0] != hvalue.shape[1]:
        raise ValueError("Hamiltonian blocks must be square")
    identity = np.eye(hvalue.shape[0], dtype=np.complex128)
    output = np.empty_like(hvalue)
    for k_index in range(hvalue.shape[2]):
        physical_projector = (dvalue[:, :, k_index] + identity).T
        hblock = hvalue[:, :, k_index]
        output[:, :, k_index] = (
            hblock @ physical_projector - physical_projector @ hblock
        )
    return output


def ptse2_level_shifted_hamiltonian(
    hamiltonian: np.ndarray,
    stored_density: np.ndarray,
    *,
    level_shift_ev: float,
) -> np.ndarray:
    """Raise the current physical virtual subspace by a positive energy shift."""

    hvalue = np.asarray(hamiltonian, dtype=np.complex128)
    dvalue = np.asarray(stored_density, dtype=np.complex128)
    shift = float(level_shift_ev)
    if hvalue.shape != dvalue.shape or hvalue.ndim != 3:
        raise ValueError("Hamiltonian and stored density must share shape (n,n,nk)")
    if not np.isfinite(shift) or shift < 0.0:
        raise ValueError("level_shift_ev must be finite and nonnegative")
    identity = np.eye(hvalue.shape[0], dtype=np.complex128)
    output = np.array(hvalue, copy=True)
    for k_index in range(hvalue.shape[2]):
        physical_projector = (dvalue[:, :, k_index] + identity).T
        output[:, :, k_index] += shift * (identity - physical_projector)
    _hermitize_blocks(output)
    return output


def run_ptse2_supercell_operator_diis_hf(
    data: PtSe2SupercellHFData,
    *,
    initial_density: np.ndarray,
    init_mode: str,
    seed: int,
    options: PulayDIISOptions | None = None,
    degenerate_occupation_policy: str = "fail",
    level_shift_ev: float = 0.0,
    step_callback: Callable[
        [PulayOperatorDIISStep, PulayOperatorEvaluation[PtSe2FixedPointPayload]],
        None,
    ]
    | None = None,
) -> HartreeFockRun:
    """Run physical-Hamiltonian/commutator Pulay DIIS for a supercell."""

    density = np.asarray(initial_density, dtype=np.complex128).copy()
    if degenerate_occupation_policy not in {"fail", "equal_ensemble"}:
        raise ValueError("invalid operator-DIIS degenerate occupation policy")
    if not np.isfinite(level_shift_ev) or float(level_shift_ev) < 0.0:
        raise ValueError("level_shift_ev must be finite and nonnegative")
    if density.shape != data.h0.shape:
        raise ValueError("supercell operator-DIIS initial density has the wrong shape")
    target_trace = float(data.supercell_filling_nu * data.fold_map.reduced_nk)
    identity = np.eye(data.fold_map.folded_rank, dtype=np.complex128)

    def validate_density(candidate: np.ndarray) -> None:
        hermiticity = float(
            np.max(
                np.abs(candidate - np.swapaxes(candidate.conj(), 0, 1)),
                initial=0.0,
            )
        )
        if hermiticity > 1.0e-9:
            raise ValueError("operator-DIIS density is not Hermitian")
        trace = float(np.trace(candidate, axis1=0, axis2=1).real.sum())
        if abs(trace - target_trace) > 1.0e-8:
            raise ValueError("operator-DIIS density does not preserve global filling")
        for reduced_index in range(data.fold_map.reduced_nk):
            projector = candidate[:, :, reduced_index] + identity
            eigenvalues = np.linalg.eigvalsh(projector)
            if eigenvalues[0] < -1.0e-9 or eigenvalues[-1] > 1.0 + 1.0e-9:
                raise ValueError("operator-DIIS projector is outside [0,1]")
            if np.linalg.norm(projector @ projector - projector) > 1.0e-8:
                raise ValueError("operator-DIIS density is not a Slater projector")

    def validate_hamiltonian(candidate: np.ndarray) -> None:
        residual = float(
            np.max(
                np.abs(candidate - np.swapaxes(candidate.conj(), 0, 1)),
                initial=0.0,
            )
        )
        if residual > 1.0e-9:
            raise ValueError("operator-DIIS Hamiltonian is not Hermitian")

    def map_hamiltonian(candidate: np.ndarray) -> np.ndarray:
        update = ptse2_density_from_fixed_filling(
            candidate,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=(
                data.primitive_data.config.occupation_degeneracy_tolerance_ev
            ),
            degenerate_occupation_policy=degenerate_occupation_policy,
        )
        mapped = np.asarray(update.density, dtype=np.complex128).copy()
        _hermitize_blocks(mapped)
        return mapped

    def evaluate(
        candidate: np.ndarray,
    ) -> PulayOperatorEvaluation[PtSe2FixedPointPayload]:
        interaction_h = build_projected_interaction_hamiltonian(
            candidate,
            data.overlap_cache.blocks,
            v0=1.0 / float(data.supercell_area_nm2),
            use_numba=bool(data.primitive_data.config.use_numba),
        )
        total_hamiltonian = np.asarray(
            data.h0 + interaction_h, dtype=np.complex128
        )
        _hermitize_blocks(total_hamiltonian)
        physical_update = ptse2_density_from_fixed_filling(
            total_hamiltonian,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=(
                data.primitive_data.config.occupation_degeneracy_tolerance_ev
            ),
            degenerate_occupation_policy=degenerate_occupation_policy,
        )
        shifted_hamiltonian = ptse2_level_shifted_hamiltonian(
            total_hamiltonian,
            candidate,
            level_shift_ev=float(level_shift_ev),
        )
        shifted_update = ptse2_density_from_fixed_filling(
            shifted_hamiltonian,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=(
                data.primitive_data.config.occupation_degeneracy_tolerance_ev
            ),
            degenerate_occupation_policy=degenerate_occupation_policy,
        )
        mapped = np.asarray(shifted_update.density, dtype=np.complex128).copy()
        _hermitize_blocks(mapped)
        error = ptse2_stored_density_commutator(
            total_hamiltonian, candidate
        )
        physical_raw_norm = calculate_norm_convergence(
            physical_update.density, candidate
        )
        energy = compute_hf_energy(interaction_h, data.h0, candidate)
        return PulayOperatorEvaluation(
            hamiltonian=shifted_hamiltonian,
            mapped_density=mapped,
            error=error,
            payload=PtSe2FixedPointPayload(
                interaction_h=np.asarray(interaction_h, dtype=np.complex128),
                total_hamiltonian=total_hamiltonian,
                density_update=physical_update,
                energy=float(energy),
                physical_raw_norm=float(physical_raw_norm),
                level_shift_ev=float(level_shift_ev),
            ),
        )

    iter_energy: list[float] = []
    iter_physical_err: list[float] = []

    def on_step(
        step: PulayOperatorDIISStep,
        evaluation: PulayOperatorEvaluation[PtSe2FixedPointPayload],
    ) -> None:
        iter_energy.append(float(evaluation.payload.energy))
        iter_physical_err.append(float(evaluation.payload.physical_raw_norm))
        if step_callback is not None:
            step_callback(step, evaluation)

    result = run_pulay_operator_diis(
        density,
        evaluate,
        map_hamiltonian,
        precision=float(data.primitive_data.config.precision),
        max_iter=int(data.primitive_data.config.max_iter),
        options=options,
        validate_density=validate_density,
        validate_hamiltonian=validate_hamiltonian,
        step_callback=on_step,
    )
    final_payload = result.final_evaluation.payload
    state = PtSe2HFState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.asarray(result.density, dtype=np.complex128).copy(),
        hamiltonian=np.asarray(
            final_payload.total_hamiltonian, dtype=np.complex128
        ).copy(),
        energies=np.asarray(
            final_payload.density_update.energies, dtype=np.float64
        ).copy(),
        precision=float(data.primitive_data.config.precision),
        v0=1.0 / float(data.supercell_area_nm2),
    )
    state.mu = float(final_payload.density_update.mu)
    final_physical_raw_norm = float(final_payload.physical_raw_norm)
    physical_converged = (
        final_physical_raw_norm <= float(data.primitive_data.config.precision)
    )
    exit_reason = str(result.exit_reason)
    if physical_converged and not result.converged:
        exit_reason = "converged_physical_replay"
    elif result.converged and not physical_converged:
        exit_reason = "shifted_converged_physical_replay_failed"
    state.diagnostics["hf_energy"] = float(final_payload.energy)
    state.diagnostics["final_raw_norm"] = final_physical_raw_norm
    state.diagnostics["shifted_final_raw_norm"] = float(
        result.final_residual_norm
    )
    state.diagnostics["level_shift_ev"] = float(level_shift_ev)
    state.diagnostics["iterations"] = float(result.iterations)
    state.diagnostics["diis_restart_count"] = float(result.restart_count)
    state.diagnostics["diis_commutator_rms_ev"] = float(
        result.final_error_norm
    )
    iter_err = np.asarray(iter_physical_err, dtype=np.float64)
    return HartreeFockRun(
        state=state,
        iter_energy=np.asarray(iter_energy, dtype=np.float64),
        iter_err=np.asarray(iter_err, dtype=np.float64),
        iter_oda=np.full(iter_err.size, np.nan, dtype=np.float64),
        init_mode=str(init_mode),
        seed=int(seed),
        converged=bool(physical_converged),
        exit_reason=exit_reason,
    )


def run_ptse2_supercell_orbital_root_hf(
    data: PtSe2SupercellHFData,
    *,
    initial_density: np.ndarray,
    init_mode: str,
    seed: int,
    residual_tolerance_ev: float = 1.0e-10,
    max_iter: int = 100,
    degenerate_occupation_policy: str = "fail",
) -> HartreeFockRun:
    """Solve the unrestricted supercell HF orbital-gradient root."""

    density0 = np.asarray(initial_density, dtype=np.complex128)
    if density0.shape != data.h0.shape:
        raise ValueError("orbital-root initial density has the wrong shape")
    if degenerate_occupation_policy not in {"fail", "equal_ensemble"}:
        raise ValueError("invalid orbital-root degenerate occupation policy")
    n = data.fold_map.folded_rank
    identity = np.eye(n, dtype=np.complex128)
    projector0 = np.empty_like(density0)
    occupied_counts: list[int] = []
    for k_index in range(data.fold_map.reduced_nk):
        projector0[:, :, k_index] = (density0[:, :, k_index] + identity).T
        eigenvalues = np.linalg.eigvalsh(projector0[:, :, k_index])
        occupied_counts.append(int(np.count_nonzero(eigenvalues > 0.5)))
    if len(set(occupied_counts)) != 1:
        raise ValueError("orbital-root requires a uniform occupied count per k")
    occupied_per_k = occupied_counts[0]

    def evaluate_projector(
        physical_projector: np.ndarray,
    ) -> OrbitalRootEvaluation[PtSe2FixedPointPayload]:
        stored_density = np.empty_like(physical_projector)
        for k_index in range(data.fold_map.reduced_nk):
            stored_density[:, :, k_index] = (
                physical_projector[:, :, k_index].T - identity
            )
        interaction_h = build_projected_interaction_hamiltonian(
            stored_density,
            data.overlap_cache.blocks,
            v0=1.0 / float(data.supercell_area_nm2),
            use_numba=bool(data.primitive_data.config.use_numba),
        )
        total_hamiltonian = np.asarray(
            data.h0 + interaction_h, dtype=np.complex128
        )
        _hermitize_blocks(total_hamiltonian)
        update = ptse2_density_from_fixed_filling(
            total_hamiltonian,
            data.supercell_filling_nu,
            degeneracy_tolerance_ev=(
                data.primitive_data.config.occupation_degeneracy_tolerance_ev
            ),
            degenerate_occupation_policy=degenerate_occupation_policy,
        )
        energy = compute_hf_energy(interaction_h, data.h0, stored_density)
        return OrbitalRootEvaluation(
            hamiltonian=total_hamiltonian,
            payload=PtSe2FixedPointPayload(
                interaction_h=np.asarray(interaction_h, dtype=np.complex128),
                total_hamiltonian=total_hamiltonian,
                density_update=update,
                energy=float(energy),
            ),
        )

    result = run_orbital_rotation_root(
        projector0,
        evaluate_projector,
        occupied_per_k=occupied_per_k,
        residual_tolerance=float(residual_tolerance_ev),
        max_iter=int(max_iter),
    )
    final_density = np.empty_like(result.projector)
    for k_index in range(data.fold_map.reduced_nk):
        final_density[:, :, k_index] = result.projector[:, :, k_index].T - identity
    final_payload = result.final_evaluation.payload
    final_raw_norm = calculate_norm_convergence(
        final_payload.density_update.density, final_density
    )
    physical_converged = (
        result.converged
        and final_raw_norm <= float(data.primitive_data.config.precision)
    )
    exit_reason = str(result.exit_reason)
    if result.converged and not physical_converged:
        exit_reason = "orbital_root_converged_aufbau_replay_failed"
    state = PtSe2HFState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.asarray(final_density, dtype=np.complex128),
        hamiltonian=np.asarray(
            final_payload.total_hamiltonian, dtype=np.complex128
        ).copy(),
        energies=np.asarray(
            final_payload.density_update.energies, dtype=np.float64
        ).copy(),
        precision=float(data.primitive_data.config.precision),
        v0=1.0 / float(data.supercell_area_nm2),
    )
    state.mu = float(final_payload.density_update.mu)
    state.diagnostics["hf_energy"] = float(final_payload.energy)
    state.diagnostics["final_raw_norm"] = float(final_raw_norm)
    state.diagnostics["orbital_root_residual_rms_ev"] = float(
        result.residual_rms
    )
    state.diagnostics["orbital_root_residual_max_abs_ev"] = float(
        result.residual_max_abs
    )
    state.diagnostics["orbital_root_function_evaluations"] = float(
        result.function_evaluations
    )
    state.diagnostics["iterations"] = float(result.iterations)
    iter_err = np.asarray(result.residual_history, dtype=np.float64)
    return HartreeFockRun(
        state=state,
        iter_energy=np.full(iter_err.size, np.nan, dtype=np.float64),
        iter_err=iter_err,
        iter_oda=np.full(iter_err.size, np.nan, dtype=np.float64),
        init_mode=str(init_mode),
        seed=int(seed),
        converged=bool(physical_converged),
        exit_reason=exit_reason,
    )


def run_ptse2_2x2_projected_hf(
    data: PtSe2SupercellHFData,
    *,
    initial_density: np.ndarray,
    init_mode: str,
    seed: int,
    step_callback: Callable[[PtSe2HFState, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[PtSe2HFState, DensityUpdateResult], None] | None = None,
    max_oda_lambda: float | None = None,
) -> HartreeFockRun:
    if data.fold_map.supercell != ptse2_2x2_supercell():
        raise ValueError("2x2 runner requires 2x2 supercell data")
    allowed = {
        "bare",
        "translation_invariant",
        "cdw_10",
        "cdw_01",
        "cdw_11",
        "cdw_mixed",
    }
    if str(init_mode) not in allowed:
        raise ValueError(f"unsupported 2x2 init_mode={init_mode!r}")
    return run_ptse2_supercell_projected_hf(
        data,
        initial_density=initial_density,
        init_mode=init_mode,
        seed=seed,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        max_oda_lambda=max_oda_lambda,
    )


def ptse2_translation_observables(
    density: np.ndarray,
    fold_map: PtSe2FoldMap,
) -> PtSe2TranslationObservables:
    values = np.asarray(density, dtype=np.complex128)
    if values.shape != (
        fold_map.folded_rank,
        fold_map.folded_rank,
        fold_map.reduced_nk,
    ):
        raise ValueError("density shape is incompatible with the fold map")
    rank = int(fold_map.active_rank)
    total_squared = float(np.vdot(values, values).real)
    offdiagonal_squared = 0.0
    maximum = 0.0
    sector_labels = {
        (int(row[0]), int(row[1]))
        for row in fold_map.fold_representatives
        if tuple(row) != (0, 0)
    }
    sectors = {sector: 0.0 for sector in sorted(sector_labels)}
    folds = np.asarray(fold_map.fold_representatives, dtype=np.int64)
    for target_fold in range(fold_map.area_ratio):
        target = _folded_indices(
            target_fold, rank, fold_map.area_ratio
        )
        for source_fold in range(fold_map.area_ratio):
            if target_fold == source_fold:
                continue
            source = _folded_indices(
                source_fold, rank, fold_map.area_ratio
            )
            block = values[np.ix_(
                target, source, np.arange(fold_map.reduced_nk)
            )]
            squared = float(np.vdot(block, block).real)
            offdiagonal_squared += squared
            maximum = max(
                maximum, float(np.max(np.abs(block), initial=0.0))
            )
            difference = folds[target_fold] - folds[source_fold]
            sector = ptse2_supercell_label_sector(
                (int(difference[0]), int(difference[1])), fold_map
            )
            if sector == (0, 0):
                raise RuntimeError("off-diagonal fold block mapped to trivial sector")
            sectors[sector] += squared
    normalization = float(fold_map.reduced_nk)
    absolute = float(np.sqrt(offdiagonal_squared / normalization))
    relative = float(np.sqrt(offdiagonal_squared / max(total_squared, 1.0e-300)))
    sector_frobenius = {
        sector: float(np.sqrt(value / normalization))
        for sector, value in sectors.items()
    }
    return PtSe2TranslationObservables(
        absolute_frobenius=absolute,
        relative_frobenius=relative,
        maximum_fold_offdiagonal_abs=maximum,
        sector_frobenius=sector_frobenius,
    )


def build_ptse2_literal_physical_q_interaction(
    density: np.ndarray,
    primitive_data: PtSe2ProjectedHFData,
    fold_map: PtSe2FoldMap,
) -> np.ndarray:
    """Literal physical-Q contraction independent of supercell local-field regrouping.

    This validation oracle groups the primitive cache directly by its physical-Q
    index. It is intentionally separate from the local-field regrouping path
    and is suitable for bounded parity checks, not production SCF iterations.
    """

    values = np.asarray(density, dtype=np.complex128)
    folded_rank = int(fold_map.folded_rank)
    reduced_nk = int(fold_map.reduced_nk)
    if values.shape != (folded_rank, folded_rank, reduced_nk):
        raise ValueError("density shape is incompatible with the fold map")
    primitive_cache = primitive_data.overlap_cache
    if primitive_cache.active_rank != fold_map.active_rank:
        raise ValueError("primitive cache rank differs from the fold map")
    vertices: dict[tuple[int, int, int], np.ndarray] = {}
    assignments: dict[tuple[int, int, int], np.ndarray] = {}
    w_by_q = np.full(
        primitive_cache.physical_q_numerators.shape[0], np.nan, dtype=np.float64
    )
    for shift in primitive_cache.blocks.shifts:
        mask = np.asarray(primitive_cache.pair_masks[shift], dtype=bool)
        q_table = np.asarray(
            primitive_cache.channel_q_indices[shift], dtype=np.int64
        )
        kernel = np.asarray(
            primitive_cache.blocks.fock_screening[shift], dtype=np.float64
        )
        incoming = kernel[mask]
        existing = w_by_q[q_table[mask]]
        if np.any(np.isfinite(existing) & (np.abs(existing - incoming) > 1.0e-12)):
            raise RuntimeError("primitive screening values disagree for one physical Q")
        w_by_q[q_table[mask]] = incoming
        primitive_overlap = primitive_cache.blocks.overlaps[shift]
        for primitive_target, primitive_source in np.argwhere(mask):
            primitive_target = int(primitive_target)
            primitive_source = int(primitive_source)
            target_reduced = int(fold_map.primitive_to_reduced[primitive_target])
            source_reduced = int(fold_map.primitive_to_reduced[primitive_source])
            target_fold = int(fold_map.primitive_to_fold[primitive_target])
            source_fold = int(fold_map.primitive_to_fold[primitive_source])
            q_index = int(q_table[primitive_target, primitive_source])
            key = (q_index, target_reduced, source_reduced)
            if key not in vertices:
                vertices[key] = np.zeros(
                    (folded_rank, folded_rank), dtype=np.complex128
                )
                assignments[key] = np.zeros(
                    (fold_map.area_ratio, fold_map.area_ratio), dtype=bool
                )
            if assignments[key][target_fold, source_fold]:
                raise RuntimeError("duplicate literal physical-Q fold assignment")
            target_active = _folded_indices(
                target_fold, fold_map.active_rank, fold_map.area_ratio
            )
            source_active = _folded_indices(
                source_fold, fold_map.active_rank, fold_map.area_ratio
            )
            vertices[key][np.ix_(target_active, source_active)] = primitive_overlap[
                :, primitive_target, :, primitive_source
            ]
            assignments[key][target_fold, source_fold] = True
    if any(
        np.count_nonzero(assigned_folds) != fold_map.area_ratio
        for assigned_folds in assignments.values()
    ):
        raise RuntimeError(
            "literal physical-Q vertex lacks area_ratio fold assignments"
        )
    if np.any(~np.isfinite(w_by_q)):
        raise RuntimeError("literal physical-Q oracle lacks screening values")

    interaction = np.zeros_like(values)
    scale = 1.0 / (
        float(primitive_data.area_nm2)
        * float(fold_map.area_ratio)
        * float(reduced_nk)
    )
    diagonal_by_q: dict[int, dict[int, np.ndarray]] = {}
    for (q_index, target_reduced, source_reduced), vertex in vertices.items():
        interaction[:, :, target_reduced] -= (
            scale
            * w_by_q[q_index]
            * vertex
            @ values[:, :, source_reduced].T
            @ vertex.conj().T
        )
        if target_reduced == source_reduced:
            diagonal_by_q.setdefault(q_index, {})[target_reduced] = vertex
    for q_index, diagonal_vertices in diagonal_by_q.items():
        if len(diagonal_vertices) != reduced_nk:
            raise RuntimeError("literal Hartree physical Q is only partially admitted")
        trace = 0.0 + 0.0j
        for reduced_index, vertex in diagonal_vertices.items():
            trace += np.einsum(
                "ab,ba->",
                values[:, :, reduced_index],
                np.conj(vertex),
                optimize=True,
            )
        for reduced_index, vertex in diagonal_vertices.items():
            interaction[:, :, reduced_index] += (
                scale * w_by_q[q_index] * trace * vertex
            )
    return interaction


def build_ptse2_2x2_literal_physical_q_interaction(
    density: np.ndarray,
    primitive_data: PtSe2ProjectedHFData,
    fold_map: PtSe2FoldMap,
) -> np.ndarray:
    if fold_map.supercell != ptse2_2x2_supercell():
        raise ValueError("2x2 literal wrapper requires the 2x2 fold map")
    return build_ptse2_literal_physical_q_interaction(
        density, primitive_data, fold_map
    )


def ptse2_supercell_hole_fourier(
    density: np.ndarray,
    overlap_cache: PtSe2SupercellOverlapCache,
    *,
    expected_holes_per_supercell: float | None = None,
    use_numba: bool | None = None,
) -> dict[tuple[int, int], complex]:
    """Return active-space hole coefficients relative to P_ref=I_(area_ratio*r)."""

    values = np.asarray(density, dtype=np.complex128)
    if values.shape != (
        overlap_cache.fold_map.folded_rank,
        overlap_cache.fold_map.folded_rank,
        overlap_cache.fold_map.reduced_nk,
    ):
        raise ValueError("density shape is incompatible with the supercell cache")
    coefficients: dict[tuple[int, int], complex] = {}
    for label in overlap_cache.blocks.hartree_screening:
        diagonal = overlap_cache.blocks.diagonal_overlaps[label]
        # The generic stored-projector contraction evaluates Tr[D^K Lambda_L^dagger],
        # i.e. the coefficient at -L. Rekey it explicitly before reconstruction.
        reverse_label = (-label[0], -label[1])
        value = -compute_density_overlap_trace_from_diagonal(
            values,
            diagonal,
            use_numba=use_numba,
        ) / float(overlap_cache.fold_map.reduced_nk)
        if reverse_label in coefficients and abs(coefficients[reverse_label] - value) > 1.0e-12:
            raise RuntimeError("duplicate reverse-keyed hole Fourier coefficient")
        coefficients[reverse_label] = value
    zero = coefficients.get((0, 0))
    expected_zero = (
        float(overlap_cache.fold_map.area_ratio)
        if expected_holes_per_supercell is None
        else float(expected_holes_per_supercell)
    )
    if not np.isfinite(expected_zero) or not 0.0 <= expected_zero <= values.shape[0]:
        raise ValueError("expected holes per supercell must lie in [0, folded_rank]")
    if zero is None or abs(zero - expected_zero) > 1.0e-8:
        raise ValueError("supercell hole zero mode differs from the expected filling")
    for label, value in coefficients.items():
        reverse = (-label[0], -label[1])
        if reverse not in coefficients:
            raise ValueError("supercell hole Fourier inventory is not reverse closed")
        if abs(coefficients[reverse] - value.conjugate()) > 1.0e-8:
            raise ValueError("supercell hole Fourier coefficients violate reverse reality")
    return coefficients


def reconstruct_ptse2_supercell_hole_density(
    coefficients: dict[tuple[int, int], complex],
    *,
    supercell_area_nm2: float,
    grid_size: int = 240,
    translation_breaking_only: bool = False,
    fold_map: PtSe2FoldMap | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_size = int(grid_size)
    area = float(supercell_area_nm2)
    if grid_size <= 1:
        raise ValueError("grid_size must exceed one")
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("supercell_area_nm2 must be finite and positive")
    frac = np.arange(grid_size, dtype=np.float64) / float(grid_size)
    u, v = np.meshgrid(frac, frac, indexing="ij")
    density = np.zeros_like(u, dtype=np.complex128)
    if translation_breaking_only and fold_map is None:
        raise ValueError(
            "translation-breaking reconstruction requires an explicit fold_map"
        )
    for (l1, l2), value in coefficients.items():
        if translation_breaking_only:
            preserves_translation = not ptse2_supercell_label_breaks_translation(
                (l1, l2), fold_map
            )
            if preserves_translation:
                continue
        phase = np.exp(-2j * np.pi * (l1 * u + l2 * v))
        density += complex(value) * phase
    density /= area
    maximum_imaginary = float(np.max(np.abs(density.imag)))
    if maximum_imaginary > 1.0e-9:
        raise ValueError("reconstructed supercell density is not real")
    return u, v, np.asarray(density.real, dtype=np.float64)


__all__ = [
    "PtSe2FixedPointPayload",
    "PtSe2FoldMap",
    "PtSe2SupercellHFData",
    "PtSe2SupercellOrbitalHessianData",
    "PtSe2SupercellOverlapCache",
    "PtSe2SupercellProjectorEvaluation",
    "PtSe2TranslationObservables",
    "build_ptse2_2x2_literal_physical_q_interaction",
    "build_ptse2_fold_map",
    "build_ptse2_literal_physical_q_interaction",
    "build_ptse2_supercell_hf_data",
    "build_ptse2_supercell_overlap_cache",
    "build_ptse2_supercell_zero_temperature_hessian",
    "embed_ptse2_translation_invariant_density",
    "evaluate_ptse2_supercell_physical_projector",
    "fold_ptse2_primitive_operator",
    "initialize_ptse2_2x2_density",
    "initialize_ptse2_supercell_density",
    "ptse2_2x1_supercell",
    "ptse2_2x2_supercell",
    "ptse2_sqrt3_supercell",
    "ptse2_supercell_hole_fourier",
    "ptse2_supercell_label_breaks_translation",
    "ptse2_level_shifted_hamiltonian",
    "ptse2_stored_density_commutator",
    "ptse2_supercell_energy_from_physical_projector",
    "ptse2_supercell_label_sector",
    "ptse2_translation_breaking_orbital_action",
    "ptse2_translation_observables",
    "reconstruct_ptse2_supercell_hole_density",
    "run_ptse2_2x2_projected_hf",
    "run_ptse2_supercell_diis_hf",
    "run_ptse2_supercell_operator_diis_hf",
    "run_ptse2_supercell_orbital_root_hf",
    "run_ptse2_supercell_projected_hf",
    "unfold_ptse2_translation_invariant_operator",
]
