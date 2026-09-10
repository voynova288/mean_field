"""Tolerance-qualified nested-domain density embedding for Vituri spiral HF.

A source projector on an odd centered integer-label square is copied bitwise to
matching labels of a larger odd square.  New target labels are initialized with
the exact ``4 x 4`` identity.  Hole/rank preservation is nevertheless reported
as a tolerance-qualified numerical statement, with the measured trace and
spectral-rank residuals retained in the receipt.

The factory-only receipt establishes an initializer transfer, not UV
convergence, SCF stationarity, branch identity, or paper authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .vituri2024_hf import vituri2024_native_density_to_conventional_k_diagonal
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_spiral import Vituri2024PreparedHFSpiral

Array = NDArray[np.generic]

VITURI2024_SPIRAL_DENSITY_EMBEDDING_API_VERSION: Final[str] = (
    "vituri2024_spiral_nested_square_density_embedding.v2"
)
VITURI2024_SPIRAL_DENSITY_EMBEDDING_AUTHORITY: Final[str] = (
    "fixed_density_tolerance_qualified_integer_label_initializer_transfer_only_"
    "not_uv_convergence_scf_branch_or_paper_authority"
)
VITURI2024_SPIRAL_DENSITY_EMBEDDING_RANK_TOLERANCE: Final[float] = 5.0e-12



def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _strict_finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _strict_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _bytes_backed_readonly(value: object, dtype: np.dtype, shape: tuple[int, ...], label: str) -> Array:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact ndarray")
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{label} must have exact dtype {dtype} and shape {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _is_bytes_or_readonly_memoryview_backed(value: object) -> bool:
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    owner: object = value
    seen: set[int] = set()
    while isinstance(owner, np.ndarray):
        if id(owner) in seen:
            return False
        seen.add(id(owner))
        owner = owner.base
    if isinstance(owner, bytes):
        return True
    return isinstance(owner, memoryview) and owner.readonly


def _array_sha256(value: object) -> str:
    array = np.asarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array), initial=0.0))


def _centered_square_labels(prepared: Vituri2024PreparedHFSpiral) -> tuple[Array, float]:
    nk = prepared.nk
    size = math.isqrt(nk)
    if size * size != nk or size < 3 or size % 2 != 1:
        raise ValueError("spiral domain must be an odd centered square")
    half = size // 2
    labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    positive_x = int(np.flatnonzero(np.all(labels == (1, 0), axis=1))[0])
    spacing = float(prepared.ordered_mesh[positive_x, 0])
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spiral mesh has no positive finite Cartesian spacing")
    expected_mesh = np.asarray(labels, dtype=np.float64) * spacing
    if not np.array_equal(prepared.ordered_mesh, expected_mesh):
        raise ValueError("spiral mesh is not the canonical exact centered-square label map")
    return _bytes_backed_readonly(labels, np.dtype(np.int64), (nk, 2), "integer labels"), spacing


def _validate_spiral_projector(
    prepared: Vituri2024PreparedHFSpiral,
    density_native: object,
    *,
    label: str,
) -> tuple[Array, int, float, float]:
    density = _bytes_backed_readonly(
        density_native,
        np.dtype(np.complex128),
        (4, 4, prepared.nk),
        f"{label} density",
    )
    conventional = vituri2024_native_density_to_conventional_k_diagonal(density)
    tolerance = VITURI2024_SPIRAL_DENSITY_EMBEDDING_RANK_TOLERANCE
    if _max_abs(conventional - conventional.swapaxes(0, 1).conj()) > tolerance:
        raise ValueError(f"{label} density must be Hermitian at every momentum")
    idempotency = max(
        _max_abs(conventional[:, :, k] @ conventional[:, :, k] - conventional[:, :, k])
        for k in range(prepared.nk)
    )
    if idempotency > tolerance:
        raise ValueError(f"{label} density must be a momentum-local projector")
    selected = np.asarray(
        [
            index
            for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
            if spin == prepared.choice.selected_spin
        ],
        dtype=np.int64,
    )
    spectators = np.asarray(
        [
            index
            for index, (_valley, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
            if spin == -prepared.choice.selected_spin
        ],
        dtype=np.int64,
    )
    momenta = np.arange(prepared.nk, dtype=np.int64)
    cross = conventional[np.ix_(selected, spectators, momenta)]
    spectator = conventional[np.ix_(spectators, spectators, momenta)]
    if (
        _max_abs(cross) > tolerance
        or _max_abs(
            spectator
            - np.repeat(np.eye(2, dtype=np.complex128)[:, :, None], prepared.nk, axis=2)
        )
        > tolerance
    ):
        raise ValueError(f"{label} density left the selected-rank/full-spectator sector")
    eigenvalue_rank_residual = 0.0
    measured_trace = 0.0
    for k in range(prepared.nk):
        eigenvalues = np.linalg.eigvalsh(conventional[:, :, k])
        eigenvalue_rank_residual = max(
            eigenvalue_rank_residual, _max_abs(eigenvalues - np.rint(eigenvalues))
        )
        measured_trace += float(np.trace(conventional[:, :, k]).real)
    expected_rank = 4 * prepared.nk - 2 * prepared.holes_per_valley
    trace_rank_residual = abs(measured_trace - float(expected_rank))
    if (
        eigenvalue_rank_residual > tolerance
        or trace_rank_residual > tolerance * max(1.0, float(prepared.nk))
    ):
        raise ValueError(f"{label} density tolerance-qualified rank/hole contract is invalid")
    return density, expected_rank, trace_rank_residual, eigenvalue_rank_residual


def _common_physical_contract(
    source: Vituri2024PreparedHFSpiral,
    target: Vituri2024PreparedHFSpiral,
    source_labels: Array,
    target_labels: Array,
    source_spacing: float,
    target_spacing: float,
) -> tuple[Array, Array]:
    if target.nk <= source.nk:
        raise ValueError("target centered square must be strictly larger than source")
    if source_spacing != target_spacing:
        raise ValueError("source and target Cartesian spacings differ")
    if source.holes_per_valley != target.holes_per_valley:
        raise ValueError("source and target expected hole counts differ")
    if source.choice.fingerprint != target.choice.fingerprint:
        raise ValueError("source and target finite-q spiral choices differ")
    source_mesh_receipt = source.functional.mesh_receipt
    target_mesh_receipt = target.functional.mesh_receipt
    if (
        source_mesh_receipt.area_angstrom_squared
        != target_mesh_receipt.area_angstrom_squared
        or source_mesh_receipt.uniform_weight_inverse_angstrom_squared
        != target_mesh_receipt.uniform_weight_inverse_angstrom_squared
        or source_mesh_receipt.policy != target_mesh_receipt.policy
    ):
        raise ValueError("source and target finite-area/quadrature contracts differ")
    if (
        source.functional.interaction_fingerprint
        != target.functional.interaction_fingerprint
        or source.functional.q0_choice.fingerprint
        != target.functional.q0_choice.fingerprint
        or source.backend_kind != target.backend_kind
    ):
        raise ValueError("source and target interaction/q0/backend contracts differ")

    target_join = {
        tuple(int(x) for x in label): index for index, label in enumerate(target_labels)
    }
    if len(target_join) != target.nk:
        raise RuntimeError("target integer-label inventory is not unique")
    try:
        common_target_indices = np.asarray(
            [target_join[tuple(int(x) for x in label)] for label in source_labels],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("source integer labels are not a subset of target labels") from error
    if len(set(int(x) for x in common_target_indices)) != source.nk:
        raise RuntimeError("common integer-label join is not one-to-one")
    common_set = set(int(x) for x in common_target_indices)
    annulus_target_indices = np.asarray(
        [index for index in range(target.nk) if index not in common_set], dtype=np.int64
    )
    if annulus_target_indices.size != target.nk - source.nk:
        raise RuntimeError("target annulus inventory is incomplete")

    common_pairs = (
        (source.ordered_mesh, target.ordered_mesh[common_target_indices], "mesh"),
        (
            source.shifted_momenta_by_valley,
            target.shifted_momenta_by_valley[:, common_target_indices, :],
            "shifted momenta",
        ),
        (
            source.active_band_states,
            target.active_band_states[:, :, common_target_indices],
            "active states",
        ),
        (
            source.active_band_energies_by_valley,
            target.active_band_energies_by_valley[:, common_target_indices],
            "active energies",
        ),
        (source.h0_native, target.h0_native[:, :, common_target_indices], "h0"),
        (
            source.functional.normal_order_reference_native,
            target.functional.normal_order_reference_native[:, :, common_target_indices],
            "normal-order reference",
        ),
    )
    for source_value, target_value, label in common_pairs:
        if not np.array_equal(source_value, target_value):
            raise ValueError(f"source and target common-label {label} differ")
    return (
        _bytes_backed_readonly(
            common_target_indices,
            np.dtype(np.int64),
            (source.nk,),
            "common target indices",
        ),
        _bytes_backed_readonly(
            annulus_target_indices,
            np.dtype(np.int64),
            (target.nk - source.nk,),
            "annulus target indices",
        ),
    )


@dataclass(frozen=True, slots=True)
class _Vituri2024SpiralDensityEmbeddingReceipt:
    """Private factory product for one bytes-backed nested embedding."""

    _factory_token: InitVar[object]
    source_prepared: Vituri2024PreparedHFSpiral = field(repr=False, compare=False)
    target_prepared: Vituri2024PreparedHFSpiral = field(repr=False, compare=False)
    source_density_native: Array
    embedded_density_native: Array
    source_integer_labels: Array
    target_integer_labels: Array
    common_target_indices: Array
    annulus_target_indices: Array
    source_prepared_fingerprint: str
    target_prepared_fingerprint: str
    source_density_sha256: str
    source_group_lineage_fingerprint: str
    source_group_fingerprint: str
    source_closure_digest: str
    target_density_sha256: str
    source_mesh_size: int
    target_mesh_size: int
    delta_k_inverse_angstrom: float
    area_angstrom_squared: float
    uniform_weight_inverse_angstrom_squared: float
    q_inverse_angstrom: tuple[float, float]
    holes_per_valley: int
    expected_source_total_rank: int
    expected_target_total_rank: int
    expected_source_hole_count: int
    expected_target_hole_count: int
    expected_annulus_hole_count: int
    source_trace_rank_residual: float
    target_trace_rank_residual: float
    annulus_trace_full_rank_residual: float
    source_eigenvalue_rank_residual: float
    target_eigenvalue_rank_residual: float
    rank_validation_tolerance: float
    common_label_count: int
    annulus_label_count: int
    fingerprint: str = field(init=False)
    authority: str = field(
        default=VITURI2024_SPIRAL_DENSITY_EMBEDDING_AUTHORITY, init=False
    )
    bitwise_common_label_copy: bool = field(default=True, init=False)
    exact_full_identity_annulus: bool = field(default=True, init=False)
    tolerance_qualified_hole_inventory_preserved: bool = field(default=True, init=False)
    fixed_physical_contract: bool = field(default=True, init=False)
    uv_convergence_authority: bool = field(default=False, init=False)
    scf_stationarity_authority: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if not _embedding_receipt_factory_admission(_factory_token):
            raise TypeError("spiral density embedding receipts are factory-only")
        source_size = _strict_int(self.source_mesh_size, "source_mesh_size")
        target_size = _strict_int(self.target_mesh_size, "target_mesh_size")
        source_count = source_size**2
        target_count = target_size**2
        source_density = _bytes_backed_readonly(
            self.source_density_native,
            np.dtype(np.complex128),
            (4, 4, source_count),
            "receipt source density",
        )
        density = _bytes_backed_readonly(
            self.embedded_density_native,
            np.dtype(np.complex128),
            (4, 4, target_count),
            "receipt embedded density",
        )
        source_labels = _bytes_backed_readonly(
            self.source_integer_labels,
            np.dtype(np.int64),
            (source_count, 2),
            "receipt source labels",
        )
        target_labels = _bytes_backed_readonly(
            self.target_integer_labels,
            np.dtype(np.int64),
            (target_count, 2),
            "receipt target labels",
        )
        common = _bytes_backed_readonly(
            self.common_target_indices,
            np.dtype(np.int64),
            (source_count,),
            "receipt common indices",
        )
        annulus = _bytes_backed_readonly(
            self.annulus_target_indices,
            np.dtype(np.int64),
            (target_count - source_count,),
            "receipt annulus indices",
        )
        object.__setattr__(self, "source_mesh_size", source_size)
        object.__setattr__(self, "target_mesh_size", target_size)
        object.__setattr__(self, "source_density_native", source_density)
        object.__setattr__(self, "embedded_density_native", density)
        object.__setattr__(self, "source_integer_labels", source_labels)
        object.__setattr__(self, "target_integer_labels", target_labels)
        object.__setattr__(self, "common_target_indices", common)
        object.__setattr__(self, "annulus_target_indices", annulus)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self._validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_SPIRAL_DENSITY_EMBEDDING_API_VERSION,
                "source_prepared_fingerprint": self.source_prepared_fingerprint,
                "target_prepared_fingerprint": self.target_prepared_fingerprint,
                "source_density_sha256": self.source_density_sha256,
                "source_group_lineage_fingerprint": self.source_group_lineage_fingerprint,
                "source_group_fingerprint": self.source_group_fingerprint,
                "source_closure_digest": self.source_closure_digest,
                "target_density_sha256": self.target_density_sha256,
                "source_labels_sha256": _array_sha256(self.source_integer_labels),
                "target_labels_sha256": _array_sha256(self.target_integer_labels),
                "common_target_indices_sha256": _array_sha256(self.common_target_indices),
                "annulus_target_indices_sha256": _array_sha256(self.annulus_target_indices),
                "source_mesh_size": self.source_mesh_size,
                "target_mesh_size": self.target_mesh_size,
                "delta_k_inverse_angstrom": self.delta_k_inverse_angstrom,
                "area_angstrom_squared": self.area_angstrom_squared,
                "uniform_weight_inverse_angstrom_squared": self.uniform_weight_inverse_angstrom_squared,
                "q_inverse_angstrom": self.q_inverse_angstrom,
                "holes_per_valley": self.holes_per_valley,
                "expected_source_total_rank": self.expected_source_total_rank,
                "expected_target_total_rank": self.expected_target_total_rank,
                "expected_source_hole_count": self.expected_source_hole_count,
                "expected_target_hole_count": self.expected_target_hole_count,
                "expected_annulus_hole_count": self.expected_annulus_hole_count,
                "source_trace_rank_residual": self.source_trace_rank_residual,
                "target_trace_rank_residual": self.target_trace_rank_residual,
                "annulus_trace_full_rank_residual": self.annulus_trace_full_rank_residual,
                "source_eigenvalue_rank_residual": self.source_eigenvalue_rank_residual,
                "target_eigenvalue_rank_residual": self.target_eigenvalue_rank_residual,
                "rank_validation_tolerance": self.rank_validation_tolerance,
                "common_label_count": self.common_label_count,
                "annulus_label_count": self.annulus_label_count,
                "authority": self.authority,
            }
        )

    def _validate_live_state(self) -> None:
        """Revalidate hashes, scalars, partition, labels, and both live preparations."""

        if type(self.source_prepared) is not Vituri2024PreparedHFSpiral:
            raise TypeError("embedding source preparation type drifted")
        if type(self.target_prepared) is not Vituri2024PreparedHFSpiral:
            raise TypeError("embedding target preparation type drifted")
        self.source_prepared.validate_live_state()
        self.target_prepared.validate_live_state()
        if any(
            not _is_bytes_or_readonly_memoryview_backed(value)
            for value in (
                self.source_density_native,
                self.embedded_density_native,
                self.source_integer_labels,
                self.target_integer_labels,
                self.common_target_indices,
                self.annulus_target_indices,
            )
        ):
            raise ValueError("embedding arrays must remain bytes/memoryview-backed readonly")
        source_size = _strict_int(self.source_mesh_size, "source_mesh_size")
        target_size = _strict_int(self.target_mesh_size, "target_mesh_size")
        if source_size < 3 or source_size % 2 != 1 or target_size <= source_size or target_size % 2 != 1:
            raise ValueError("embedding mesh-size scalars are invalid")
        source_count = source_size**2
        target_count = target_size**2
        exact_arrays = (
            (self.source_density_native, np.dtype(np.complex128), (4, 4, source_count), "source density"),
            (self.embedded_density_native, np.dtype(np.complex128), (4, 4, target_count), "target density"),
            (self.source_integer_labels, np.dtype(np.int64), (source_count, 2), "source labels"),
            (self.target_integer_labels, np.dtype(np.int64), (target_count, 2), "target labels"),
            (self.common_target_indices, np.dtype(np.int64), (source_count,), "common indices"),
            (self.annulus_target_indices, np.dtype(np.int64), (target_count - source_count,), "annulus indices"),
        )
        for value, dtype, shape, label in exact_arrays:
            if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
                raise ValueError(f"embedding {label} dtype/shape drifted")
            if value.flags.writeable or not np.all(np.isfinite(value)):
                raise ValueError(f"embedding {label} mutability/finiteness drifted")
        _strict_sha256(self.source_prepared_fingerprint, "source prepared fingerprint")
        _strict_sha256(self.target_prepared_fingerprint, "target prepared fingerprint")
        _strict_sha256(self.source_density_sha256, "source density hash")
        _strict_sha256(
            self.source_group_lineage_fingerprint, "source-group lineage fingerprint"
        )
        _strict_sha256(self.source_group_fingerprint, "source-group fingerprint")
        _strict_sha256(self.source_closure_digest, "source closure digest")
        _strict_sha256(self.target_density_sha256, "target density hash")
        if (
            self.source_prepared_fingerprint != self.source_prepared.fingerprint
            or self.target_prepared_fingerprint != self.target_prepared.fingerprint
            or self.source_density_sha256 != _array_sha256(self.source_density_native)
            or self.target_density_sha256 != _array_sha256(self.embedded_density_native)
        ):
            raise ValueError("embedding prepared/density hash binding drifted")
        source_labels, source_spacing = _centered_square_labels(self.source_prepared)
        target_labels, target_spacing = _centered_square_labels(self.target_prepared)
        common, annulus = _common_physical_contract(
            self.source_prepared,
            self.target_prepared,
            source_labels,
            target_labels,
            source_spacing,
            target_spacing,
        )
        if not all(
            np.array_equal(actual, expected)
            for actual, expected in (
                (self.source_integer_labels, source_labels),
                (self.target_integer_labels, target_labels),
                (self.common_target_indices, common),
                (self.annulus_target_indices, annulus),
            )
        ):
            raise ValueError("embedding label mapping/index partition drifted")
        if (
            np.unique(np.concatenate((common, annulus))).size != target_count
            or not np.array_equal(np.sort(np.concatenate((common, annulus))), np.arange(target_count))
        ):
            raise ValueError("embedding common/annulus partition is incomplete")
        if not np.array_equal(self.embedded_density_native[:, :, common], self.source_density_native):
            raise ValueError("embedded common-label density is not a bitwise source copy")
        identity = np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], annulus.size, axis=2)
        if not np.array_equal(self.embedded_density_native[:, :, annulus], identity):
            raise ValueError("embedded target annulus is not exactly full flavor identity")
        _, source_rank, source_trace_residual, source_eigen_residual = _validate_spiral_projector(
            self.source_prepared, self.source_density_native, label="source"
        )
        _, target_rank, target_trace_residual, target_eigen_residual = _validate_spiral_projector(
            self.target_prepared, self.embedded_density_native, label="target"
        )
        annulus_trace = sum(
            float(np.trace(self.embedded_density_native[:, :, int(index)]).real)
            for index in annulus
        )
        annulus_trace_residual = abs(annulus_trace - 4.0 * float(annulus.size))
        finite_scalars = (
            (self.delta_k_inverse_angstrom, "delta_k_inverse_angstrom", True),
            (self.area_angstrom_squared, "area_angstrom_squared", True),
            (self.uniform_weight_inverse_angstrom_squared, "uniform_weight_inverse_angstrom_squared", True),
            (self.source_trace_rank_residual, "source_trace_rank_residual", False),
            (self.target_trace_rank_residual, "target_trace_rank_residual", False),
            (self.annulus_trace_full_rank_residual, "annulus_trace_full_rank_residual", False),
            (self.source_eigenvalue_rank_residual, "source_eigenvalue_rank_residual", False),
            (self.target_eigenvalue_rank_residual, "target_eigenvalue_rank_residual", False),
            (self.rank_validation_tolerance, "rank_validation_tolerance", True),
        )
        for value, label, positive in finite_scalars:
            checked = _strict_finite_float(value, label, positive=positive)
            if not positive and checked < 0.0:
                raise ValueError(f"{label} must be nonnegative")
        if type(self.q_inverse_angstrom) is not tuple or len(self.q_inverse_angstrom) != 2:
            raise TypeError("q_inverse_angstrom must be an exact length-two tuple")
        q = tuple(_strict_finite_float(value, "q component") for value in self.q_inverse_angstrom)
        mesh = self.target_prepared.functional.mesh_receipt
        expected_ints = {
            "holes_per_valley": self.target_prepared.holes_per_valley,
            "expected_source_total_rank": source_rank,
            "expected_target_total_rank": target_rank,
            "expected_source_hole_count": 4 * source_count - source_rank,
            "expected_target_hole_count": 4 * target_count - target_rank,
            "expected_annulus_hole_count": 0,
            "common_label_count": source_count,
            "annulus_label_count": target_count - source_count,
        }
        for name, expected in expected_ints.items():
            if _strict_int(getattr(self, name), name) != expected:
                raise ValueError(f"embedding {name} drifted")
        expected_scalars = {
            "delta_k_inverse_angstrom": target_spacing,
            "area_angstrom_squared": mesh.area_angstrom_squared,
            "uniform_weight_inverse_angstrom_squared": mesh.uniform_weight_inverse_angstrom_squared,
            "source_trace_rank_residual": source_trace_residual,
            "target_trace_rank_residual": target_trace_residual,
            "annulus_trace_full_rank_residual": annulus_trace_residual,
            "source_eigenvalue_rank_residual": source_eigen_residual,
            "target_eigenvalue_rank_residual": target_eigen_residual,
            "rank_validation_tolerance": VITURI2024_SPIRAL_DENSITY_EMBEDDING_RANK_TOLERANCE,
        }
        if any(getattr(self, name) != expected for name, expected in expected_scalars.items()):
            raise ValueError("embedding measured/scalar contract drifted")
        if q != tuple(float(x) for x in self.target_prepared.choice.q_inverse_angstrom):
            raise ValueError("embedding q scalar binding drifted")
        if not all(
            (
                self.bitwise_common_label_copy is True,
                self.exact_full_identity_annulus is True,
                self.tolerance_qualified_hole_inventory_preserved is True,
                self.fixed_physical_contract is True,
                self.uv_convergence_authority is False,
                self.scf_stationarity_authority is False,
                self.authority == VITURI2024_SPIRAL_DENSITY_EMBEDDING_AUTHORITY,
            )
        ):
            raise ValueError("embedding receipt authority flags are invalid")
        if self._current_fingerprint() != self.fingerprint:
            raise ValueError("embedding receipt fingerprint drifted")

    @property
    def density_native(self) -> Array:
        """Compatibility alias for the immutable embedded target density."""

        return self.embedded_density_native


def _make_embedding_receipt_product_api():
    """Keep the constructor token and product-identity registry closure-private."""

    factory_token = object()
    issued_products: dict[
        int, tuple[_Vituri2024SpiralDensityEmbeddingReceipt, str]
    ] = {}

    def token_is_valid(candidate: object) -> bool:
        return candidate is factory_token

    def construct(**fields: object) -> _Vituri2024SpiralDensityEmbeddingReceipt:
        product = _Vituri2024SpiralDensityEmbeddingReceipt(
            _factory_token=factory_token, **fields
        )
        issued_state_fingerprint = product._current_fingerprint()
        if issued_state_fingerprint != product.fingerprint:
            raise RuntimeError("embedding receipt issuance fingerprint drifted")
        issued_products[id(product)] = (product, issued_state_fingerprint)
        return product

    def factory(
        source_prepared: Vituri2024PreparedHFSpiral,
        target_prepared: Vituri2024PreparedHFSpiral,
        source_density_native: object,
        source_group_lineage_receipt: object,
    ) -> _Vituri2024SpiralDensityEmbeddingReceipt:
        return _embed_vituri2024_spiral_density_nested_square_impl(
            source_prepared,
            target_prepared,
            source_density_native,
            source_group_lineage_receipt,
            _receipt_constructor=construct,
        )

    def validate(product: object) -> _Vituri2024SpiralDensityEmbeddingReceipt:
        if type(product) is not _Vituri2024SpiralDensityEmbeddingReceipt:
            raise TypeError("spiral density embedding receipt must be a factory product")
        issued = issued_products.get(id(product))
        if issued is None or issued[0] is not product:
            raise TypeError("spiral density embedding receipt identity is not registered")
        product._validate_live_state()
        if product._current_fingerprint() != issued[1]:
            raise ValueError("spiral density embedding receipt changed after factory issuance")
        return product

    return token_is_valid, factory, validate


(
    _embedding_receipt_factory_admission,
    embed_vituri2024_spiral_density_nested_square,
    validate_vituri2024_spiral_density_embedding_receipt,
) = _make_embedding_receipt_product_api()


def _embed_vituri2024_spiral_density_nested_square_impl(
    source_prepared: Vituri2024PreparedHFSpiral,
    target_prepared: Vituri2024PreparedHFSpiral,
    source_density_native: object,
    source_group_lineage_receipt: object,
    *,
    _receipt_constructor: object,
) -> _Vituri2024SpiralDensityEmbeddingReceipt:
    """Build an embedding from one complete-inventory source-group lineage."""

    # The local import avoids a module cycle: the normal-closure adapter imports
    # this embedding module, while it owns the detached source-lineage factory.
    from .vituri2024_hf_spiral_normal_closure import (
        validate_vituri2024_spiral_normal_source_lineage_receipt,
    )

    lineage = validate_vituri2024_spiral_normal_source_lineage_receipt(
        source_group_lineage_receipt, require_complete_inventory=True
    )
    if type(source_prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("source_prepared must be Vituri2024PreparedHFSpiral")
    if type(target_prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("target_prepared must be Vituri2024PreparedHFSpiral")
    source_prepared.validate_live_state()
    target_prepared.validate_live_state()
    source_density, source_rank, source_trace_residual, source_eigen_residual = (
        _validate_spiral_projector(source_prepared, source_density_native, label="source")
    )
    source_density_sha256 = _array_sha256(source_density)
    if (
        lineage.source_prepared_fingerprint != source_prepared.fingerprint
        or lineage.source_group_final_density_sha256 != source_density_sha256
        or lineage.source_group_fingerprint
        != lineage.source_stationary_group_fingerprints[lineage.source_group_index]
    ):
        raise ValueError(
            "source-group lineage does not hash-match source prepared/density/group fingerprint"
        )
    source_labels, source_spacing = _centered_square_labels(source_prepared)
    target_labels, target_spacing = _centered_square_labels(target_prepared)
    common, annulus = _common_physical_contract(
        source_prepared,
        target_prepared,
        source_labels,
        target_labels,
        source_spacing,
        target_spacing,
    )
    embedded = np.repeat(
        np.eye(4, dtype=np.complex128)[:, :, None], target_prepared.nk, axis=2
    )
    embedded[:, :, common] = source_density
    if not np.array_equal(embedded[:, :, common], source_density):
        raise RuntimeError("common-label density copy was not bitwise exact")
    identity = np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], annulus.size, axis=2)
    if not np.array_equal(embedded[:, :, annulus], identity):
        raise RuntimeError("new target labels were not initialized exactly full")
    embedded_bound, target_rank, target_trace_residual, target_eigen_residual = (
        _validate_spiral_projector(target_prepared, embedded, label="target")
    )
    annulus_trace = sum(
        float(np.trace(embedded_bound[:, :, int(index)]).real) for index in annulus
    )
    annulus_trace_residual = abs(annulus_trace - 4.0 * float(annulus.size))
    source_size = math.isqrt(source_prepared.nk)
    target_size = math.isqrt(target_prepared.nk)
    mesh_receipt = target_prepared.functional.mesh_receipt
    if not callable(_receipt_constructor):
        raise TypeError("private embedding receipt constructor must be callable")
    receipt = _receipt_constructor(
        source_prepared=source_prepared,
        target_prepared=target_prepared,
        source_density_native=source_density,
        embedded_density_native=embedded_bound,
        source_integer_labels=source_labels,
        target_integer_labels=target_labels,
        common_target_indices=common,
        annulus_target_indices=annulus,
        source_prepared_fingerprint=source_prepared.fingerprint,
        target_prepared_fingerprint=target_prepared.fingerprint,
        source_density_sha256=source_density_sha256,
        source_group_lineage_fingerprint=lineage.fingerprint,
        source_group_fingerprint=lineage.source_group_fingerprint,
        source_closure_digest=lineage.source_closure_digest,
        target_density_sha256=_array_sha256(embedded_bound),
        source_mesh_size=source_size,
        target_mesh_size=target_size,
        delta_k_inverse_angstrom=target_spacing,
        area_angstrom_squared=float(mesh_receipt.area_angstrom_squared),
        uniform_weight_inverse_angstrom_squared=float(
            mesh_receipt.uniform_weight_inverse_angstrom_squared
        ),
        q_inverse_angstrom=tuple(float(x) for x in target_prepared.choice.q_inverse_angstrom),
        holes_per_valley=target_prepared.holes_per_valley,
        expected_source_total_rank=source_rank,
        expected_target_total_rank=target_rank,
        expected_source_hole_count=4 * source_prepared.nk - source_rank,
        expected_target_hole_count=4 * target_prepared.nk - target_rank,
        expected_annulus_hole_count=0,
        source_trace_rank_residual=source_trace_residual,
        target_trace_rank_residual=target_trace_residual,
        annulus_trace_full_rank_residual=annulus_trace_residual,
        source_eigenvalue_rank_residual=source_eigen_residual,
        target_eigenvalue_rank_residual=target_eigen_residual,
        rank_validation_tolerance=VITURI2024_SPIRAL_DENSITY_EMBEDDING_RANK_TOLERANCE,
        common_label_count=source_prepared.nk,
        annulus_label_count=target_prepared.nk - source_prepared.nk,
    )
    return validate_vituri2024_spiral_density_embedding_receipt(receipt)


embed_vituri2024_nested_spiral_density = embed_vituri2024_spiral_density_nested_square


__all__ = [
    "VITURI2024_SPIRAL_DENSITY_EMBEDDING_API_VERSION",
    "VITURI2024_SPIRAL_DENSITY_EMBEDDING_AUTHORITY",
    "VITURI2024_SPIRAL_DENSITY_EMBEDDING_RANK_TOLERANCE",
    "embed_vituri2024_nested_spiral_density",
    "embed_vituri2024_spiral_density_nested_square",
    "validate_vituri2024_spiral_density_embedding_receipt",
]
