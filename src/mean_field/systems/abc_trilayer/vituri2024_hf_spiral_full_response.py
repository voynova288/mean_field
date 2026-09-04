"""Candidate integer-label off-k response for Vituri spiral stability.

The existing translational FFT functional is exact only on k-diagonal density
blocks.  This module extends the same projected-Hamiltonian algebra to one
signed off-diagonal density block

``W_d(k) = D[(a,k+d),(b,k)]``

on the same finite centered square.  Momentum conservation is imposed on the
integer mesh labels and never wraps.  The direct and exchange formulas are

``Sigma_H[a,a; k+d,k] = F_a(k+d,k) K(-d) sum_{c,p} F_c(p,p+d) W_cc(p) / A``

For the source-bound real-even kernel, ``K(-d)=K(d)``.

and

``Sigma_F[a,b; k+d,k] = -sum_p V(p-k)
    F_a(k+d,p+d) F_b(p,k) W_ab(p) / A``.

A reduced dense exchange-contraction reference and the zero-padded FFT
factorization are retained.  A separate integer-mask full projected-H oracle
belongs to qualification tests.  This is a candidate algebra surface only.
It returns the selected-output projection for frozen spectator input and does
not assert
parity with the older literal-float quartet mask, source/scalar authority,
reciprocity, Hermitian-eigensolver readiness, or local stability.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import inspect
import json
import math
from typing import Final

import numpy as np
from scipy.fft import fft2 as _FFT2, ifft2 as _IFFT2

from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from .vituri2024_tdhf_full_functional import (
    VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK,
    _bytes_backed,
    _exact_local_mask,
)
from .vituri2024_hf_spiral_full_stability import (
    Vituri2024HFSpiralFullSectorInventory,
    Vituri2024HFSpiralFullSectorKey,
)

Array = np.ndarray

VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION: Final[str] = (
    "vituri2024_hf_spiral_full_selected_spin_signed_displacement_response.v1"
)
VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY: Final[str] = (
    "candidate_integer_label_no_wrap_projected_h_signed_displacement_response_"
    "only_not_scalar_hessian_eigensolver_local_stability_or_paper_authority"
)
VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK: Final[int] = 121
VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_API_VERSION: Final[str] = (
    "vituri2024_hf_spiral_literal_mask_equivalence.v1"
)
VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE: Final[int] = 101
VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE: Final[str] = (
    "caller_supplied_common_cartesian_mesh_exact_local_mask_only_with_mesh_hash_"
    "not_flavor_resolved_shifted_momenta_or_full_functional_action_parity"
)
VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC: Final[str] = (
    "for_each_cartesian_axis_exhaust_all_scalar_quartets_using_"
    "((k_alpha+k_beta)-k_gamma)-k_delta_eq_0_exact_float64_and_compare_"
    "to_identically_ordered_integer_labels"
)
VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT: Final[str] = (
    "projected_h_uses_K_particle_minus_output_while_fft_uses_K_output_minus_"
    "particle_with_source_bound_real_even_kernel_verified"
)

_RESPONSE_TOKEN = object()
_LITERAL_MASK_RECEIPT_TOKEN = object()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _exact_local_mask_source_closure_sha256() -> str:
    source = "\n---dependency---\n".join(
        inspect.getsource(function) for function in (_exact_local_mask, _bytes_backed)
    )
    return sha256(source.encode()).hexdigest()


def _readonly_complex128(value: object, shape: tuple[int, ...], label: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.complex128)
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must be finite exact complex128 {shape}")
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.complex128
    ).reshape(shape)
    result.setflags(write=False)
    return result


def _readonly_float64(value: object, shape: tuple[int, ...], label: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must be finite exact float64 {shape}")
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float64
    ).reshape(shape)
    result.setflags(write=False)
    return result


def _selected_spinors(inventory: Vituri2024HFSpiralFullSectorInventory) -> Array:
    prepared = inventory.restricted_preparation.prepared
    selected_indices = inventory.restricted_preparation.receipt.selected_flavor_indices
    valley_to_state = {
        valley: index
        for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
    }
    selected = np.empty((2, 6, inventory.nk), dtype=np.complex128)
    for slot, flavor_index in enumerate(selected_indices):
        valley = inventory.selected_valleys[slot]
        if prepared.choice.selected_spin != INTERNAL_FLAVOR_ORDER[flavor_index][1]:
            raise ValueError("selected flavor spin drifted from the spiral choice")
        selected[slot] = prepared.active_band_states[valley_to_state[valley]]
    return _readonly_complex128(selected, selected.shape, "selected spinors")


def _support_indices(
    inventory: Vituri2024HFSpiralFullSectorInventory,
    key: Vituri2024HFSpiralFullSectorKey,
) -> tuple[Array, Array]:
    inventory._key_indices(key)
    labels = inventory.integer_mesh_labels
    offset = inventory.mesh_size // 2
    target_x = labels[:, 0] + key.displacement_x
    target_y = labels[:, 1] + key.displacement_y
    keep = (
        (target_x >= -offset)
        & (target_x <= offset)
        & (target_y >= -offset)
        & (target_y <= offset)
    )
    bases = np.flatnonzero(keep).astype(np.int64)
    targets = (
        (target_y[keep] + offset) * inventory.mesh_size
        + target_x[keep]
        + offset
    ).astype(np.int64)
    expected_displacement = np.asarray(
        (key.displacement_x, key.displacement_y), dtype=np.int64
    )
    if not np.all(
        inventory.integer_mesh_labels[targets]
        - inventory.integer_mesh_labels[bases]
        == expected_displacement[None, :]
    ):
        raise RuntimeError("signed-displacement support indexing drifted")
    readonly_bases = np.frombuffer(bases.tobytes(), dtype=np.int64)
    readonly_targets = np.frombuffer(targets.tobytes(), dtype=np.int64)
    readonly_bases.setflags(write=False)
    readonly_targets.setflags(write=False)
    return readonly_bases, readonly_targets


def _shifted_spinors(
    selected_spinors: Array, bases: Array, targets: Array
) -> Array:
    shifted = np.zeros_like(selected_spinors)
    shifted[:, :, bases] = selected_spinors[:, :, targets]
    return _readonly_complex128(shifted, shifted.shape, "shifted selected spinors")


def _allowed_flavor_blocks(key: Vituri2024HFSpiralFullSectorKey) -> tuple[tuple[int, int], ...]:
    if key.valley_charge == 0:
        return ((0, 0), (1, 1))
    if key.valley_charge == 2:
        return ((1, 0),)
    if key.valley_charge == -2:
        return ((0, 1),)
    raise RuntimeError("unreachable valley charge")


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralLiteralMaskEquivalenceReceipt:
    """Factory-only exhaustive proof for the central unshifted mesh mask."""

    _factory_token: InitVar[object]
    mesh_size: int
    nk: int
    integer_mesh_labels_sha256: str
    momentum_mesh_inverse_angstrom_sha256: str
    coordinate_table_sha256: tuple[str, str]
    scalar_quartets_checked_per_axis: int
    false_positive_counts: tuple[int, int]
    false_negative_counts: tuple[int, int]
    maximum_conserving_float_residuals: tuple[float, float]
    minimum_nonconserving_float_residuals: tuple[float, float]
    exact_local_mask_source_closure_sha256: str
    fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_API_VERSION,
        init=False,
    )
    scope: str = field(
        default=VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE,
        init=False,
    )
    arithmetic_contract: str = field(
        default=VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC,
        init=False,
    )
    exact_local_mask_policy: str = field(
        default=VITURI2024_FULL_FUNCTIONAL_EXACT_LOCAL_MASK,
        init=False,
    )
    cartesian_separability_established: bool = field(default=True, init=False)
    exhaustive_scalar_quartet_check: bool = field(default=True, init=False)
    literal_float_quartet_mask_equivalence_established: bool = field(
        default=True, init=False
    )
    flavor_resolved_shifted_momentum_mask_equivalence_established: bool = field(
        default=False, init=False
    )
    full_functional_action_parity_established: bool = field(default=False, init=False)
    scalar_hessian_authority_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _LITERAL_MASK_RECEIPT_TOKEN:
            raise TypeError("literal-mask equivalence receipt is factory-only")
        payload = self._payload()
        if (
            type(self.mesh_size) is not int
            or type(self.nk) is not int
            or self.mesh_size * self.mesh_size != self.nk
            or self.mesh_size > VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE
            or self.scalar_quartets_checked_per_axis != self.mesh_size**4
        ):
            raise ValueError("literal-mask receipt dimension inventory is invalid")
        for digest in (
            self.integer_mesh_labels_sha256,
            self.momentum_mesh_inverse_angstrom_sha256,
            *self.coordinate_table_sha256,
            self.exact_local_mask_source_closure_sha256,
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("literal-mask receipt contains an invalid SHA-256")
        if self.false_positive_counts != (0, 0):
            raise ValueError("literal-mask receipt contains false positives")
        if self.false_negative_counts != (0, 0):
            raise ValueError("literal-mask receipt contains false negatives")
        if self.maximum_conserving_float_residuals != (0.0, 0.0):
            raise ValueError("literal-mask receipt conserving residual is nonzero")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.minimum_nonconserving_float_residuals
        ):
            raise ValueError("literal-mask receipt nonconserving residual is invalid")
        object.__setattr__(self, "fingerprint", _fingerprint(payload))

    def _payload(self) -> dict[str, object]:
        return {
            "api_version": self.api_version,
            "scope": self.scope,
            "arithmetic_contract": self.arithmetic_contract,
            "exact_local_mask_policy": self.exact_local_mask_policy,
            "mesh_size": self.mesh_size,
            "nk": self.nk,
            "integer_mesh_labels_sha256": self.integer_mesh_labels_sha256,
            "momentum_mesh_inverse_angstrom_sha256": (
                self.momentum_mesh_inverse_angstrom_sha256
            ),
            "coordinate_table_sha256": self.coordinate_table_sha256,
            "scalar_quartets_checked_per_axis": self.scalar_quartets_checked_per_axis,
            "false_positive_counts": self.false_positive_counts,
            "false_negative_counts": self.false_negative_counts,
            "maximum_conserving_float_residuals": (
                self.maximum_conserving_float_residuals
            ),
            "minimum_nonconserving_float_residuals": (
                self.minimum_nonconserving_float_residuals
            ),
            "exact_local_mask_source_closure_sha256": (
                self.exact_local_mask_source_closure_sha256
            ),
            "cartesian_separability_established": (
                self.cartesian_separability_established
            ),
            "exhaustive_scalar_quartet_check": self.exhaustive_scalar_quartet_check,
            "literal_float_quartet_mask_equivalence_established": (
                self.literal_float_quartet_mask_equivalence_established
            ),
            "flavor_resolved_shifted_momentum_mask_equivalence_established": (
                self.flavor_resolved_shifted_momentum_mask_equivalence_established
            ),
            "full_functional_action_parity_established": (
                self.full_functional_action_parity_established
            ),
            "scalar_hessian_authority_established": (
                self.scalar_hessian_authority_established
            ),
            "production_ready": self.production_ready,
        }

    def validate_live_state(self) -> None:
        if self.fingerprint != _fingerprint(self._payload()):
            raise ValueError("literal-mask receipt fingerprint drifted")
        current_source_sha256 = _exact_local_mask_source_closure_sha256()
        if current_source_sha256 != self.exact_local_mask_source_closure_sha256:
            raise ValueError("literal full-functional mask implementation drifted")


def certify_vituri2024_hf_spiral_literal_mask_equivalence(
    integer_mesh_labels: Array,
    momentum_mesh_inverse_angstrom: Array,
) -> Vituri2024HFSpiralLiteralMaskEquivalenceReceipt:
    """Exhaust the exact vertex predicate without allocating an ``Nk^4`` mask."""

    if (
        type(integer_mesh_labels) is not np.ndarray
        or integer_mesh_labels.dtype != np.dtype(np.int64)
        or integer_mesh_labels.ndim != 2
        or integer_mesh_labels.shape[1] != 2
    ):
        raise TypeError("integer mesh labels must be exact int64 (Nk,2)")
    labels_snapshot = np.frombuffer(
        np.ascontiguousarray(integer_mesh_labels).tobytes(order="C"), dtype=np.int64
    ).reshape(integer_mesh_labels.shape)
    labels_snapshot.setflags(write=False)
    nk = int(labels_snapshot.shape[0])
    size = math.isqrt(nk)
    if size * size != nk or size < 3 or size % 2 != 1:
        raise ValueError("literal-mask certification requires an odd square mesh")
    if size > VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE:
        raise ValueError("literal-mask exhaustive certification exceeds reviewed size cap")
    if (
        type(momentum_mesh_inverse_angstrom) is not np.ndarray
        or momentum_mesh_inverse_angstrom.dtype != np.dtype(np.float64)
        or momentum_mesh_inverse_angstrom.shape != (nk, 2)
        or not np.all(np.isfinite(momentum_mesh_inverse_angstrom))
    ):
        raise TypeError("momentum mesh must be finite exact float64 (Nk,2)")
    mesh_snapshot = np.frombuffer(
        np.ascontiguousarray(momentum_mesh_inverse_angstrom).tobytes(order="C"),
        dtype=np.float64,
    ).reshape(momentum_mesh_inverse_angstrom.shape)
    mesh_snapshot.setflags(write=False)
    half = size // 2
    expected_labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    if not np.array_equal(labels_snapshot, expected_labels):
        raise ValueError("integer mesh labels are not the complete centered square")

    coordinate_tables: list[Array] = []
    false_positive_counts: list[int] = []
    false_negative_counts: list[int] = []
    maximum_conserving_residuals: list[float] = []
    minimum_nonconserving_residuals: list[float] = []
    integer_coordinates = np.arange(-half, half + 1, dtype=np.int64)
    for axis in range(2):
        table = np.empty(size, dtype=np.float64)
        for offset, label in enumerate(integer_coordinates):
            values = mesh_snapshot[
                labels_snapshot[:, axis] == label, axis
            ]
            if values.shape != (size,) or not np.all(values == values[0]):
                raise ValueError("momentum mesh is not exactly Cartesian-separable")
            table[offset] = values[0]
        reconstructed = table[labels_snapshot[:, axis] + half]
        if not np.array_equal(reconstructed, mesh_snapshot[:, axis]):
            raise ValueError("Cartesian coordinate-table reconstruction failed")
        false_positive = 0
        false_negative = 0
        maximum_conserving = 0.0
        minimum_nonconserving = math.inf
        for alpha in range(size):
            for beta in range(size):
                float_residual = (
                    (table[alpha] + table[beta])
                    - table[:, None]
                    - table[None, :]
                )
                integer_residual = (
                    (integer_coordinates[alpha] + integer_coordinates[beta])
                    - integer_coordinates[:, None]
                    - integer_coordinates[None, :]
                )
                literal = float_residual == 0.0
                integer = integer_residual == 0
                false_positive += int(np.count_nonzero(literal & ~integer))
                false_negative += int(np.count_nonzero(~literal & integer))
                if np.any(integer):
                    maximum_conserving = max(
                        maximum_conserving,
                        float(np.max(np.abs(float_residual[integer]), initial=0.0)),
                    )
                if np.any(~integer):
                    minimum_nonconserving = min(
                        minimum_nonconserving,
                        float(np.min(np.abs(float_residual[~integer]), initial=math.inf)),
                    )
        if false_positive or false_negative or maximum_conserving != 0.0:
            raise ValueError(
                "integer labels and literal float64 vertex arithmetic are inequivalent"
            )
        coordinate_tables.append(_readonly_float64(table, (size,), "coordinate table"))
        false_positive_counts.append(false_positive)
        false_negative_counts.append(false_negative)
        maximum_conserving_residuals.append(maximum_conserving)
        minimum_nonconserving_residuals.append(minimum_nonconserving)

    payload = {
        "mesh_size": size,
        "nk": nk,
        "integer_mesh_labels_sha256": _array_sha256(labels_snapshot),
        "momentum_mesh_inverse_angstrom_sha256": _array_sha256(mesh_snapshot),
        "coordinate_table_sha256": tuple(_array_sha256(table) for table in coordinate_tables),
        "scalar_quartets_checked_per_axis": size**4,
        "false_positive_counts": tuple(false_positive_counts),
        "false_negative_counts": tuple(false_negative_counts),
        "maximum_conserving_float_residuals": tuple(maximum_conserving_residuals),
        "minimum_nonconserving_float_residuals": tuple(minimum_nonconserving_residuals),
        "exact_local_mask_source_closure_sha256": (
            _exact_local_mask_source_closure_sha256()
        ),
    }
    receipt = Vituri2024HFSpiralLiteralMaskEquivalenceReceipt(
        _factory_token=_LITERAL_MASK_RECEIPT_TOKEN,
        **payload,
    )
    receipt.validate_live_state()
    return receipt


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralSignedDisplacementResponse:
    """Source-bound dense/FFT response for one selected-spin signed block."""

    _factory_token: InitVar[object]
    inventory: Vituri2024HFSpiralFullSectorInventory
    selected_spinors: Array
    response_fingerprint: str = field(init=False)
    api_version: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION, init=False
    )
    authority: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY, init=False
    )
    dense_max_nk: int = field(
        default=VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK, init=False
    )
    candidate_only: bool = field(default=True, init=False)
    no_wrap: bool = field(default=True, init=False)
    dense_reference_available_on_reduced_meshes: bool = field(
        default=True, init=False
    )
    fft_linear_convolution: bool = field(default=True, init=False)
    q_and_minus_q_averaged: bool = field(default=False, init=False)
    selected_output_projection_with_frozen_spectator_input: bool = field(
        default=True, init=False
    )
    integer_label_interaction_conservation_convention: bool = field(
        default=True, init=False
    )
    integer_label_interaction_conservation_established: bool = field(
        default=False, init=False
    )
    real_even_kernel_verified: bool = field(default=True, init=False)
    kernel_orientation_contract: str = field(
        default=VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT, init=False
    )
    literal_float_full_functional_parity_established: bool = field(
        default=False, init=False
    )
    dense_fft_parity_established: bool = field(default=False, init=False)
    scalar_curvature_established: bool = field(default=False, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    hermitian_eigensolver_authorized: bool = field(default=False, init=False)
    full_local_stability_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RESPONSE_TOKEN:
            raise TypeError("signed-displacement response is factory-only")
        object.__setattr__(self, "response_fingerprint", self._expected_fingerprint())
        self.validate_live_state()

    @property
    def nk(self) -> int:
        return self.inventory.nk

    @property
    def mesh_size(self) -> int:
        return self.inventory.mesh_size

    @property
    def area_angstrom_squared(self) -> float:
        return float(
            self.inventory.restricted_preparation.prepared.functional.mesh_receipt.area_angstrom_squared
        )

    @property
    def fft_plan(self):
        return self.inventory.restricted_preparation.prepared.functional.fft_plan

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.response_fingerprint

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "authority": self.authority,
                "inventory": self.inventory.inventory_fingerprint,
                "selected_spinors": _array_sha256(self.selected_spinors),
                "fft_plan": self.fft_plan.fingerprint,
                "area_angstrom_squared": self.area_angstrom_squared,
                "dense_max_nk": self.dense_max_nk,
                "flags": (
                    self.candidate_only,
                    self.no_wrap,
                    self.dense_reference_available_on_reduced_meshes,
                    self.fft_linear_convolution,
                    self.q_and_minus_q_averaged,
                    self.selected_output_projection_with_frozen_spectator_input,
                    self.integer_label_interaction_conservation_convention,
                    self.integer_label_interaction_conservation_established,
                    self.real_even_kernel_verified,
                    self.kernel_orientation_contract,
                    self.literal_float_full_functional_parity_established,
                    self.dense_fft_parity_established,
                    self.scalar_curvature_established,
                    self.reciprocity_established,
                    self.hermitian_eigensolver_authorized,
                    self.full_local_stability_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        if type(self.inventory) is not Vituri2024HFSpiralFullSectorInventory:
            raise TypeError("response inventory type drifted")
        self.inventory.validate_live_state()
        expected_spinors = _selected_spinors(self.inventory)
        if (
            type(self.selected_spinors) is not np.ndarray
            or self.selected_spinors.dtype != np.dtype(np.complex128)
            or self.selected_spinors.shape != (2, 6, self.nk)
            or self.selected_spinors.flags.writeable
            or not np.all(np.isfinite(self.selected_spinors))
            or not np.array_equal(self.selected_spinors, expected_spinors)
        ):
            raise ValueError("selected response spinors drifted")
        area = self.area_angstrom_squared
        signed_kernel = self.fft_plan.kernel_by_signed_displacement
        kernel_scale = max(1.0, float(np.max(np.abs(signed_kernel), initial=0.0)))
        kernel_tolerance = 64.0 * np.finfo(np.float64).eps * kernel_scale
        kernel_is_real_even = (
            float(np.max(np.abs(signed_kernel.imag), initial=0.0))
            <= kernel_tolerance
            and float(
                np.max(
                    np.abs(signed_kernel - signed_kernel[::-1, ::-1]),
                    initial=0.0,
                )
            )
            <= kernel_tolerance
        )
        locked = (
            self.api_version == VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION,
            self.authority == VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY,
            self.dense_max_nk == VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK,
            math.isfinite(area),
            area > 0.0,
            self.fft_plan is self.inventory.restricted_preparation.prepared.functional.fft_plan,
            self.candidate_only is True,
            self.no_wrap is True,
            self.dense_reference_available_on_reduced_meshes is True,
            self.fft_linear_convolution is True,
            self.q_and_minus_q_averaged is False,
            self.selected_output_projection_with_frozen_spectator_input is True,
            self.integer_label_interaction_conservation_convention is True,
            self.integer_label_interaction_conservation_established is False,
            self.real_even_kernel_verified is True,
            kernel_is_real_even,
            self.kernel_orientation_contract
            == VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT,
            self.literal_float_full_functional_parity_established is False,
            self.dense_fft_parity_established is False,
            self.scalar_curvature_established is False,
            self.reciprocity_established is False,
            self.hermitian_eigensolver_authorized is False,
            self.full_local_stability_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
            self._expected_fingerprint() == self.response_fingerprint,
        )
        if not all(locked):
            raise ValueError("signed-displacement response authority or binding drifted")

    def support_indices(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> tuple[Array, Array]:
        self.validate_live_state()
        return _support_indices(self.inventory, key)

    def _validate_signed_block(
        self, key: Vituri2024HFSpiralFullSectorKey, value: object
    ) -> tuple[Array, Array, Array]:
        self.inventory._key_indices(key)
        block = _readonly_complex128(
            value,
            (2, 2, self.nk),
            "signed-displacement density block",
        )
        bases, targets = _support_indices(self.inventory, key)
        support = np.zeros(self.nk, dtype=np.bool_)
        support[bases] = True
        if np.count_nonzero(block[:, :, ~support]):
            raise ValueError("signed density has nonzero entries outside no-wrap support")
        allowed = set(_allowed_flavor_blocks(key))
        for left in range(2):
            for right in range(2):
                if (left, right) not in allowed and np.count_nonzero(block[left, right]):
                    raise ValueError("signed density violates its conserved valley charge")
        return block, bases, targets

    def _make_validated_fft_action_unchecked(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction":
        bases, targets = _support_indices(self.inventory, key)
        return Vituri2024HFSpiralValidatedSignedDisplacementFFTAction(
            _factory_token=_RESPONSE_TOKEN,
            response=self,
            key=key,
            bases=bases,
            targets=targets,
            shifted_spinors=_shifted_spinors(self.selected_spinors, bases, targets),
            expected_response_fingerprint=self.response_fingerprint,
        )

    def make_validated_action_factory(
        self,
    ) -> "Vituri2024HFSpiralValidatedResponseActionFactory":
        """Validate the complete source once for repeated sector preparation."""

        self.validate_live_state()
        return Vituri2024HFSpiralValidatedResponseActionFactory(
            _factory_token=_RESPONSE_TOKEN,
            response=self,
            expected_response_fingerprint=self.response_fingerprint,
        )

    def make_validated_fft_action(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction":
        """Validate the complete source once and return one hot-path callback."""

        return self.make_validated_action_factory().prepare_fft_action(key)

    def conjugate_block(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        value: Array,
    ) -> Array:
        """Return ``W_-d(k+d)=W_d(k)^dagger`` without identifying sectors."""

        self.validate_live_state()
        block, bases, targets = self._validate_signed_block(key, value)
        partner = np.zeros_like(block)
        partner[:, :, targets] = block[:, :, bases].swapaxes(0, 1).conj()
        return _readonly_complex128(
            partner,
            partner.shape,
            "conjugate signed-displacement density block",
        )

    def _direct_response(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        block: Array,
        bases: Array,
        targets: Array,
    ) -> Array:
        result = np.zeros_like(block)
        if key.valley_charge != 0:
            return result
        # The projected-H contraction carries K(p-(p+d))=K(-d).
        # The source-bound kernel is verified real-even, but the explicit minus
        # sign keeps the defining orientation visible and regression-testable.
        displacement_kernel = self.fft_plan.kernel_by_signed_displacement[
            -key.displacement_y + self.mesh_size - 1,
            -key.displacement_x + self.mesh_size - 1,
        ]
        charge = 0.0 + 0.0j
        for flavor in range(2):
            source_form_factor = np.einsum(
                "cp,cp->p",
                self.selected_spinors[flavor][:, bases].conj(),
                self.selected_spinors[flavor][:, targets],
                optimize=True,
            )
            charge += np.sum(
                source_form_factor * block[flavor, flavor, bases]
            )
        for flavor in range(2):
            target_form_factor = np.einsum(
                "cm,cm->m",
                self.selected_spinors[flavor][:, targets].conj(),
                self.selected_spinors[flavor][:, bases],
                optimize=True,
            )
            result[flavor, flavor, bases] = (
                target_form_factor * displacement_kernel * charge
            )
        return result

    def apply_dense(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        signed_density_block: Array,
    ) -> Array:
        """Apply the exact-integer dense contraction reference on a reduced mesh."""

        self.validate_live_state()
        if self.nk > self.dense_max_nk:
            raise ValueError(
                "dense signed-displacement oracle is restricted to reduced meshes"
            )
        block, bases, targets = self._validate_signed_block(
            key, signed_density_block
        )
        result = self._direct_response(key, block, bases, targets)
        labels = self.inventory.integer_mesh_labels
        offset = self.mesh_size - 1
        difference = labels[bases][None, :, :] - labels[bases][:, None, :]
        kernel = self.fft_plan.kernel_by_signed_displacement[
            difference[..., 1] + offset,
            difference[..., 0] + offset,
        ]
        for left, right in _allowed_flavor_blocks(key):
            values = block[left, right, bases]
            if np.count_nonzero(values) == 0:
                continue
            left_factor = np.einsum(
                "cm,cp->mp",
                self.selected_spinors[left][:, targets].conj(),
                self.selected_spinors[left][:, targets],
                optimize=True,
            )
            right_factor = np.einsum(
                "cp,cm->pm",
                self.selected_spinors[right][:, bases].conj(),
                self.selected_spinors[right][:, bases],
                optimize=True,
            ).T
            result[left, right, bases] -= np.einsum(
                "mp,mp,mp,p->m",
                kernel,
                left_factor,
                right_factor,
                values,
                optimize=True,
            )
        result /= self.area_angstrom_squared
        return _readonly_complex128(result, result.shape, "dense signed response")

    def _apply_fft_validated(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        block: Array,
        bases: Array,
        targets: Array,
        shifted_spinors: Array,
    ) -> Array:
        """Hot path after complete source and signed-block validation."""

        result = self._direct_response(key, block, bases, targets)
        size = self.mesh_size
        spinors_grid = self.selected_spinors.reshape(2, 6, size, size)
        shifted_grid = shifted_spinors.reshape(2, 6, size, size)
        for left, right in _allowed_flavor_blocks(key):
            values = block[left, right].reshape(size, size)
            if np.count_nonzero(values) == 0:
                continue
            output = result[left, right].reshape(size, size)
            left_shifted = shifted_grid[left]
            right_base = spinors_grid[right]
            sources = (
                left_shifted[:, None, :, :]
                * right_base[None, :, :, :].conj()
                * values[None, None, :, :]
            )
            padded = np.zeros(
                (
                    6,
                    6,
                    self.fft_plan.padding_size,
                    self.fft_plan.padding_size,
                ),
                dtype=np.complex128,
            )
            padded[:, :, :size, :size] = sources
            transformed = _FFT2(
                padded, axes=(-2, -1), workers=self.fft_plan.fft_workers
            )
            convolution = _IFFT2(
                self.fft_plan.kernel_fft[None, None, :, :] * transformed,
                axes=(-2, -1),
                workers=self.fft_plan.fft_workers,
            )[:, :, :size, :size]
            output -= np.einsum(
                "cxy,exy,cexy->xy",
                left_shifted.conj(),
                right_base,
                convolution,
                optimize=True,
            )
        result /= self.area_angstrom_squared
        support = np.zeros(self.nk, dtype=np.bool_)
        support[bases] = True
        result[:, :, ~support] = 0.0
        return _readonly_complex128(result, result.shape, "FFT signed response")

    def apply_fft(
        self,
        key: Vituri2024HFSpiralFullSectorKey,
        signed_density_block: Array,
    ) -> Array:
        """Validate once, then apply the no-wrap FFT factorization."""

        self.validate_live_state()
        block, bases, targets = self._validate_signed_block(
            key, signed_density_block
        )
        return self._apply_fft_validated(
            key,
            block,
            bases,
            targets,
            _shifted_spinors(self.selected_spinors, bases, targets),
        )


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralValidatedResponseActionFactory:
    """Public validate-once factory for many signed-sector FFT actions."""

    _factory_token: InitVar[object]
    response: Vituri2024HFSpiralSignedDisplacementResponse
    expected_response_fingerprint: str

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RESPONSE_TOKEN:
            raise TypeError("validated response action factory is factory-only")
        if type(self.response) is not Vituri2024HFSpiralSignedDisplacementResponse:
            raise TypeError("validated response action factory type drifted")
        if self.expected_response_fingerprint != self.response.response_fingerprint:
            raise ValueError("validated response action factory fingerprint drifted")

    def prepare_fft_action(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction":
        if self.expected_response_fingerprint != self.response.response_fingerprint:
            raise ValueError("validated response action factory became stale")
        return self.response._make_validated_fft_action_unchecked(key)


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralValidatedSignedDisplacementFFTAction:
    """Validate-once callback suitable for a Krylov hot path."""

    _factory_token: InitVar[object]
    response: Vituri2024HFSpiralSignedDisplacementResponse
    key: Vituri2024HFSpiralFullSectorKey
    bases: Array
    targets: Array
    shifted_spinors: Array
    expected_response_fingerprint: str

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _RESPONSE_TOKEN:
            raise TypeError("validated FFT action is factory-only")
        if type(self.response) is not Vituri2024HFSpiralSignedDisplacementResponse:
            raise TypeError("validated FFT action response type drifted")
        if type(self.key) is not Vituri2024HFSpiralFullSectorKey:
            raise TypeError("validated FFT action key type drifted")
        expected_bases, expected_targets = _support_indices(
            self.response.inventory, self.key
        )
        for value, expected, label in (
            (self.bases, expected_bases, "bases"),
            (self.targets, expected_targets, "targets"),
        ):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.int64)
                or value.flags.writeable
                or not np.array_equal(value, expected)
            ):
                raise ValueError(f"validated FFT action {label} drifted")
        expected_shifted = _shifted_spinors(
            self.response.selected_spinors, self.bases, self.targets
        )
        if (
            type(self.shifted_spinors) is not np.ndarray
            or self.shifted_spinors.dtype != np.dtype(np.complex128)
            or self.shifted_spinors.shape != (2, 6, self.response.nk)
            or self.shifted_spinors.flags.writeable
            or not np.array_equal(self.shifted_spinors, expected_shifted)
        ):
            raise ValueError("validated FFT action shifted spinors drifted")
        if self.expected_response_fingerprint != self.response.response_fingerprint:
            raise ValueError("validated FFT action fingerprint drifted")

    def __call__(self, signed_density_block: Array) -> Array:
        if self.expected_response_fingerprint != self.response.response_fingerprint:
            raise ValueError("validated FFT action response became stale")
        block, bases, targets = self.response._validate_signed_block(
            self.key, signed_density_block
        )
        if not (
            np.array_equal(bases, self.bases)
            and np.array_equal(targets, self.targets)
        ):
            raise ValueError("validated FFT action support became stale")
        return self.response._apply_fft_validated(
            self.key,
            block,
            self.bases,
            self.targets,
            self.shifted_spinors,
        )


def build_vituri2024_hf_spiral_signed_displacement_response(
    inventory: Vituri2024HFSpiralFullSectorInventory,
) -> Vituri2024HFSpiralSignedDisplacementResponse:
    """Build the candidate dense/FFT off-k interaction-response surface."""

    if type(inventory) is not Vituri2024HFSpiralFullSectorInventory:
        raise TypeError("response requires the exact full sector inventory")
    inventory.validate_live_state()
    return Vituri2024HFSpiralSignedDisplacementResponse(
        _factory_token=_RESPONSE_TOKEN,
        inventory=inventory,
        selected_spinors=_selected_spinors(inventory),
    )


__all__ = [
    "VITURI2024_HF_SPIRAL_FULL_RESPONSE_API_VERSION",
    "VITURI2024_HF_SPIRAL_FULL_RESPONSE_AUTHORITY",
    "VITURI2024_HF_SPIRAL_FULL_RESPONSE_DENSE_MAX_NK",
    "VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT",
    "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_API_VERSION",
    "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_ARITHMETIC",
    "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_MAX_MESH_SIZE",
    "VITURI2024_HF_SPIRAL_LITERAL_MASK_EQUIVALENCE_SCOPE",
    "Vituri2024HFSpiralLiteralMaskEquivalenceReceipt",
    "Vituri2024HFSpiralSignedDisplacementResponse",
    "Vituri2024HFSpiralValidatedResponseActionFactory",
    "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction",
    "build_vituri2024_hf_spiral_signed_displacement_response",
    "certify_vituri2024_hf_spiral_literal_mask_equivalence",
]
