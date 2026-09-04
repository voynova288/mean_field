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
import json
import math
from typing import Final

import numpy as np

from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
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
VITURI2024_HF_SPIRAL_FULL_RESPONSE_KERNEL_CONTRACT: Final[str] = (
    "projected_h_uses_K_particle_minus_output_while_fft_uses_K_output_minus_"
    "particle_with_source_bound_real_even_kernel_verified"
)

_RESPONSE_TOKEN = object()


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


def _allowed_flavor_blocks(key: Vituri2024HFSpiralFullSectorKey) -> tuple[tuple[int, int], ...]:
    if key.valley_charge == 0:
        return ((0, 0), (1, 1))
    if key.valley_charge == 2:
        return ((1, 0),)
    if key.valley_charge == -2:
        return ((0, 1),)
    raise RuntimeError("unreachable valley charge")


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

    def make_validated_fft_action(
        self, key: Vituri2024HFSpiralFullSectorKey
    ) -> "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction":
        """Validate the complete source once and return a hot-path callback."""

        self.validate_live_state()
        bases, targets = _support_indices(self.inventory, key)
        return Vituri2024HFSpiralValidatedSignedDisplacementFFTAction(
            _factory_token=_RESPONSE_TOKEN,
            response=self,
            key=key,
            bases=bases,
            targets=targets,
            expected_response_fingerprint=self.response_fingerprint,
        )

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
    ) -> Array:
        """Hot path after complete source and signed-block validation."""

        result = self._direct_response(key, block, bases, targets)
        size = self.mesh_size
        shifted_spinors = np.zeros((2, 6, self.nk), dtype=np.complex128)
        shifted_spinors[:, :, bases] = self.selected_spinors[:, :, targets]
        spinors_grid = self.selected_spinors.reshape(2, 6, size, size)
        shifted_grid = shifted_spinors.reshape(2, 6, size, size)
        for left, right in _allowed_flavor_blocks(key):
            values = block[left, right].reshape(size, size)
            if np.count_nonzero(values) == 0:
                continue
            output = result[left, right].reshape(size, size)
            for left_component in range(6):
                left_shifted = shifted_grid[left, left_component]
                for right_component in range(6):
                    right_base = spinors_grid[right, right_component]
                    source = np.asarray(
                        left_shifted * right_base.conj() * values,
                        dtype=np.complex128,
                    )
                    convolution = self.fft_plan._convolve_validated(source)
                    output -= (
                        left_shifted.conj() * right_base * convolution
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
        return self._apply_fft_validated(key, block, bases, targets)


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralValidatedSignedDisplacementFFTAction:
    """Validate-once callback suitable for a Krylov hot path."""

    _factory_token: InitVar[object]
    response: Vituri2024HFSpiralSignedDisplacementResponse
    key: Vituri2024HFSpiralFullSectorKey
    bases: Array
    targets: Array
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
            self.key, block, self.bases, self.targets
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
    "Vituri2024HFSpiralSignedDisplacementResponse",
    "Vituri2024HFSpiralValidatedSignedDisplacementFFTAction",
    "build_vituri2024_hf_spiral_signed_displacement_response",
]
