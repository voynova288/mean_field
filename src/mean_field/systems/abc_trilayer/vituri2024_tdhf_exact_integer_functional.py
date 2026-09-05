"""Reduced exact-integer Vituri projected-H scalar functional candidate.

This module is intentionally distinct from :mod:`vituri2024_tdhf_full_functional`.
The older functional uses literal float64 quartet equality.  Here momentum
conservation is imposed only through complete centered integer mesh labels.
No ``Nk^4`` mask is stored or allocated.

The conventional density convention is ``P_ij = <c_j^dagger c_i>`` and

``Q = P - R``
``E[P] = Tr(h0 P) + 1/2 Tr(Q Sigma[Q])``
``F[P] = h0 + Sigma[Q]``
``dF[P;D] = Sigma[D]``.

The initial implementation is a reduced-mesh ``O(Nk^3)`` reference kernel.
It is not the scalable q003 backend and carries no source, scalar-Hessian,
reciprocity, stability, production, or paper authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
import math
from typing import Final

import numpy as np

from mean_field.core.hf.tdhf_scalar_functional import (
    TDHFFullProjectorFunctionalBinding,
    TDHFScalarFunctionalInputsManifest,
    bind_tdhf_scalar_kernel,
    make_tdhf_scalar_functional_inputs_manifest,
)

from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER

Array = np.ndarray

VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_full_projector_scalar_reference.v1"
)
VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY: Final[str] = (
    "reduced_mesh_exact_integer_reference_candidate_not_source_scalar_hessian_"
    "reciprocity_stability_production_or_paper_authority"
)
VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION: Final[str] = (
    "conventional_dense_P_ij=<c_j^dagger_c_i>; exact_centered_integer_label_"
    "conservation; no_wrap_no_torus_no_carry; raw_finite_square_total"
)
VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK: Final[int] = 121
VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE: Final[float] = 5.0e-11
VITURI2024_EXACT_INTEGER_FUNCTIONAL_INPUT_NAMES: Final[tuple[str, ...]] = (
    "area_angstrom_squared",
    "form_factors_by_flavor",
    "functional_api_version",
    "h0_full",
    "integer_mesh_labels",
    "interaction_kernel_by_mesh_pair",
    "kernel_fingerprint",
    "normal_order_reference",
    "ordered_mesh",
)


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


def _function_live_fingerprint(function: object) -> str:
    if not inspect.isfunction(function):
        raise TypeError("exact-integer implementation binding is not a function")
    payload = {
        "source": inspect.getsource(function),
        "defaults": repr(function.__defaults__),
        "kwdefaults": repr(function.__kwdefaults__),
        "module": function.__module__,
        "qualname": function.__qualname__,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_bytes_backed_readonly(value: object) -> bool:
    current = value
    while type(current) is np.ndarray:
        if current.flags.writeable or current.base is None:
            return False
        current = current.base
    if isinstance(current, bytes):
        return True
    if isinstance(current, memoryview):
        return current.readonly
    return False


def _readonly_exact(
    value: object, *, dtype: np.dtype, shape: tuple[int, ...], label: str
) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must be finite exact {dtype} {shape}")
    result = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    result = float(np.max(np.abs(array)))
    if not math.isfinite(result):
        raise ValueError("nonfinite exact-integer functional residual")
    return result


def _readonly_hermitian(value: object, dimension: int, label: str) -> Array:
    result = _readonly_exact(
        value,
        dtype=np.dtype(np.complex128),
        shape=(dimension, dimension),
        label=label,
    )
    scale = max(1.0, _max_abs(result))
    if _max_abs(result - result.conj().T) > 64.0 * np.finfo(np.float64).eps * scale:
        raise ValueError(f"{label} must be Hermitian")
    return result


def _complete_centered_labels(value: object) -> tuple[Array, int]:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.int64)
        or value.ndim != 2
        or value.shape[1] != 2
    ):
        raise TypeError("integer mesh labels must be exact int64 (Nk,2)")
    nk = int(value.shape[0])
    size = math.isqrt(nk)
    if size * size != nk or size < 3 or size % 2 != 1:
        raise ValueError("integer functional requires an odd square mesh")
    if nk > VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK:
        raise ValueError("exact-integer reference exceeds the reviewed reduced-mesh cap")
    half = size // 2
    expected = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    if not np.array_equal(value, expected):
        raise ValueError("integer labels must use centered iy-outer/ix-inner order")
    return _readonly_exact(
        value,
        dtype=np.dtype(np.int64),
        shape=(nk, 2),
        label="integer mesh labels",
    ), size


def vituri2024_exact_integer_projected_interaction_action(
    density: Array,
    *,
    integer_mesh_labels: Array,
    form_factors_by_flavor: Array,
    interaction_kernel_by_mesh_pair: Array,
    area_angstrom_squared: float,
) -> Array:
    """Apply ``Sigma`` by exact integer conservation in ``O(Nk^3)`` memory-safe form."""

    labels, size = _complete_centered_labels(integer_mesh_labels)
    nk = int(labels.shape[0])
    flavors = len(INTERNAL_FLAVOR_ORDER)
    dimension = flavors * nk
    clean = _readonly_hermitian(density, dimension, "interaction-action density")
    form_factors = _readonly_exact(
        form_factors_by_flavor,
        dtype=np.dtype(np.complex128),
        shape=(flavors, nk, nk),
        label="form factors by flavor",
    )
    kernel = _readonly_exact(
        interaction_kernel_by_mesh_pair,
        dtype=np.dtype(np.float64),
        shape=(nk, nk),
        label="interaction kernel by mesh pair",
    )
    if isinstance(area_angstrom_squared, (bool, np.bool_)):
        raise TypeError("finite area must be a strict real scalar")
    area = float(area_angstrom_squared)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("finite area must be positive")
    if _max_abs(form_factors - form_factors.swapaxes(1, 2).conj()) > 5.0e-12:
        raise ValueError("form factors violate pair conjugation")
    if _max_abs(kernel - kernel.T) > 64.0 * np.finfo(np.float64).eps * max(
        1.0, _max_abs(kernel)
    ):
        raise ValueError("interaction kernel is not symmetric")
    if np.any(kernel <= 0.0):
        raise ValueError("interaction kernel must be strictly positive")

    x = clean.reshape(flavors, nk, flavors, nk)
    charge = np.zeros((nk, nk), dtype=np.complex128)
    for flavor in range(flavors):
        charge += form_factors[flavor] * x[flavor, :, flavor, :].T

    half = size // 2
    result = np.zeros_like(x)
    all_p = np.arange(nk, dtype=np.int64)
    for output in range(nk):
        output_label = labels[output]
        for base in range(nk):
            target_labels = output_label[None, :] + labels - labels[base][None, :]
            keep = np.all((target_labels >= -half) & (target_labels <= half), axis=1)
            p = all_p[keep]
            target = target_labels[keep]
            r = (target[:, 1] + half) * size + target[:, 0] + half
            direct = np.sum(kernel[p, r] * charge[p, r])
            for flavor in range(flavors):
                result[flavor, output, flavor, base] += (
                    form_factors[flavor, output, base] * direct
                )
            for left_flavor in range(flavors):
                left = form_factors[left_flavor, output, r]
                for right_flavor in range(flavors):
                    result[left_flavor, output, right_flavor, base] -= np.sum(
                        kernel[p, base]
                        * left
                        * form_factors[right_flavor, p, base]
                        * x[left_flavor, r, right_flavor, p]
                    )

    result = result.reshape(dimension, dimension) / area
    residual = _max_abs(result - result.conj().T)
    scale = max(1.0, _max_abs(result))
    if residual > VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE * scale:
        raise ValueError("exact-integer interaction action is not Hermitian")
    readonly = np.frombuffer(
        np.ascontiguousarray(result).tobytes(order="C"), dtype=np.complex128
    ).reshape(result.shape)
    readonly.setflags(write=False)
    return readonly


@dataclass(frozen=True, slots=True)
class Vituri2024ExactIntegerFunctionalKernel:
    """Reduced dense E/F/dF reference with exact integer conservation."""

    integer_mesh_labels: Array
    ordered_mesh: Array
    form_factors_by_flavor: Array
    interaction_kernel_by_mesh_pair: Array
    h0_full: Array
    normal_order_reference: Array
    area_angstrom_squared: float
    provenance: str
    nk: int = field(init=False)
    dimension: int = field(init=False)
    mesh_size: int = field(init=False)
    component_fingerprints: tuple[tuple[str, str], ...] = field(init=False)
    implementation_fingerprint: str = field(init=False)
    _kernel_fingerprint: str = field(init=False, repr=False)
    api_version: str = field(
        default=VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION, init=False
    )
    authority: str = field(
        default=VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY, init=False
    )
    convention: str = field(
        default=VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION, init=False
    )
    exact_integer_conservation: bool = field(default=True, init=False)
    no_wrap: bool = field(default=True, init=False)
    raw_finite_square_total: bool = field(default=True, init=False)
    literal_float_mask_used: bool = field(default=False, init=False)
    source_closure_established: bool = field(default=False, init=False)
    scalar_hessian_authority_established: bool = field(default=False, init=False)
    reciprocity_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        labels, size = _complete_centered_labels(self.integer_mesh_labels)
        nk = int(labels.shape[0])
        flavors = len(INTERNAL_FLAVOR_ORDER)
        dimension = flavors * nk
        mesh = _readonly_exact(
            self.ordered_mesh,
            dtype=np.dtype(np.float64),
            shape=(nk, 2),
            label="ordered mesh",
        )
        positive_x = np.flatnonzero(
            (labels[:, 0] == 1) & (labels[:, 1] == 0)
        )
        if positive_x.shape != (1,):
            raise ValueError("ordered mesh lacks the unique +x unit label")
        delta = float(mesh[int(positive_x[0]), 0])
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("ordered mesh spacing must be positive")
        expected_mesh = np.asarray(labels, dtype=np.float64) * delta
        if not np.array_equal(mesh, expected_mesh):
            raise ValueError("ordered mesh must equal integer labels times one spacing")
        form_factors = _readonly_exact(
            self.form_factors_by_flavor,
            dtype=np.dtype(np.complex128),
            shape=(flavors, nk, nk),
            label="form factors by flavor",
        )
        kernel = _readonly_exact(
            self.interaction_kernel_by_mesh_pair,
            dtype=np.dtype(np.float64),
            shape=(nk, nk),
            label="interaction kernel by mesh pair",
        )
        h0 = _readonly_hermitian(self.h0_full, dimension, "h0_full")
        reference = _readonly_hermitian(
            self.normal_order_reference, dimension, "normal-order reference"
        )
        reference_eigenvalues = np.linalg.eigvalsh(reference)
        if (
            float(reference_eigenvalues[0]) < -5.0e-12
            or float(reference_eigenvalues[-1]) > 1.0 + 5.0e-12
        ):
            raise ValueError("normal-order reference must satisfy 0 <= R <= I")
        if isinstance(self.area_angstrom_squared, (bool, np.bool_)):
            raise TypeError("finite area must be a strict real scalar")
        area = float(self.area_angstrom_squared)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("finite area must be positive")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("exact-integer functional provenance must be explicit")
        # Validate all interaction structural contracts once at construction.
        zero = np.zeros((dimension, dimension), dtype=np.complex128)
        vituri2024_exact_integer_projected_interaction_action(
            zero,
            integer_mesh_labels=labels,
            form_factors_by_flavor=form_factors,
            interaction_kernel_by_mesh_pair=kernel,
            area_angstrom_squared=area,
        )
        components = tuple(
            (name, _array_sha256(value))
            for name, value in (
                ("integer_mesh_labels", labels),
                ("ordered_mesh", mesh),
                ("form_factors_by_flavor", form_factors),
                ("interaction_kernel_by_mesh_pair", kernel),
                ("h0_full", h0),
                ("normal_order_reference", reference),
            )
        )
        implementation_fingerprint = self._expected_implementation_fingerprint()
        object.__setattr__(self, "integer_mesh_labels", labels)
        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "form_factors_by_flavor", form_factors)
        object.__setattr__(self, "interaction_kernel_by_mesh_pair", kernel)
        object.__setattr__(self, "h0_full", h0)
        object.__setattr__(self, "normal_order_reference", reference)
        object.__setattr__(self, "area_angstrom_squared", area)
        object.__setattr__(self, "nk", nk)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "mesh_size", size)
        object.__setattr__(self, "component_fingerprints", components)
        object.__setattr__(self, "implementation_fingerprint", implementation_fingerprint)
        object.__setattr__(self, "_kernel_fingerprint", self._expected_kernel_fingerprint())
        self.validate_live_state()

    def _expected_component_fingerprints(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, _array_sha256(value))
            for name, value in (
                ("integer_mesh_labels", self.integer_mesh_labels),
                ("ordered_mesh", self.ordered_mesh),
                ("form_factors_by_flavor", self.form_factors_by_flavor),
                ("interaction_kernel_by_mesh_pair", self.interaction_kernel_by_mesh_pair),
                ("h0_full", self.h0_full),
                ("normal_order_reference", self.normal_order_reference),
            )
        )

    def _expected_implementation_fingerprint(self) -> str:
        live_functions = (
            ("interaction_action_function", vituri2024_exact_integer_projected_interaction_action),
            ("complete_centered_labels", _complete_centered_labels),
            ("readonly_exact", _readonly_exact),
            ("readonly_hermitian", _readonly_hermitian),
            ("max_abs", _max_abs),
            ("array_sha256", _array_sha256),
            ("function_live_fingerprint", _function_live_fingerprint),
            ("is_bytes_backed_readonly", _is_bytes_backed_readonly),
            ("kernel_method___post_init__", getattr(type(self), "__post_init__")),
            (
                "kernel_property_kernel_fingerprint",
                getattr(type(self), "kernel_fingerprint").fget,
            ),
            *tuple(
                (f"kernel_method_{name}", getattr(type(self), name))
                for name in (
                    "interaction_action",
                    "energy",
                    "fock",
                    "differential_fock",
                    "validate_live_state",
                    "_expected_component_fingerprints",
                    "_expected_implementation_fingerprint",
                    "_expected_kernel_fingerprint",
                )
            ),
        )
        live_code_inventory = tuple(
            (name, _function_live_fingerprint(function))
            for name, function in live_functions
        )
        semantic_contract = {
            "api_version": VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION,
            "authority": VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY,
            "convention": VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION,
            "flavor_order": INTERNAL_FLAVOR_ORDER,
            "max_nk": VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK,
            "structure_tolerance": VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE,
            "orbital_order": "flavor_major_then_k",
        }
        return sha256(
            json.dumps(
                {
                    "live_code_inventory": live_code_inventory,
                    "semantic_contract": semantic_contract,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _expected_kernel_fingerprint(self) -> str:
        payload = {
            "api_version": self.api_version,
            "authority": self.authority,
            "convention": self.convention,
            "components": self.component_fingerprints,
            "implementation_fingerprint": self.implementation_fingerprint,
            "area_angstrom_squared": self.area_angstrom_squared,
            "provenance": self.provenance,
            "exact_integer_conservation": self.exact_integer_conservation,
            "no_wrap": self.no_wrap,
            "raw_finite_square_total": self.raw_finite_square_total,
            "literal_float_mask_used": self.literal_float_mask_used,
            "source_closure_established": self.source_closure_established,
            "scalar_hessian_authority_established": (
                self.scalar_hessian_authority_established
            ),
            "reciprocity_established": self.reciprocity_established,
            "production_ready": self.production_ready,
            "paper_reproduction_verified": self.paper_reproduction_verified,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def kernel_fingerprint(self) -> str:
        self.validate_live_state()
        return self._kernel_fingerprint

    def validate_live_state(self) -> None:
        arrays = (
            self.integer_mesh_labels,
            self.ordered_mesh,
            self.form_factors_by_flavor,
            self.interaction_kernel_by_mesh_pair,
            self.h0_full,
            self.normal_order_reference,
        )
        if any(not _is_bytes_backed_readonly(value) for value in arrays):
            raise ValueError("exact-integer functional arrays must remain bytes-backed readonly")
        labels, size = _complete_centered_labels(self.integer_mesh_labels)
        if (
            type(self.nk) is not int
            or type(self.dimension) is not int
            or type(self.mesh_size) is not int
            or self.nk != labels.shape[0]
            or self.dimension != len(INTERNAL_FLAVOR_ORDER) * self.nk
            or self.mesh_size != size
        ):
            raise ValueError("exact-integer functional derived dimensions drifted")
        if (
            type(self.area_angstrom_squared) is not float
            or not math.isfinite(self.area_angstrom_squared)
            or self.area_angstrom_squared <= 0.0
            or type(self.provenance) is not str
            or not self.provenance.strip()
        ):
            raise ValueError("exact-integer functional scalar binding drifted")
        if self.component_fingerprints != self._expected_component_fingerprints():
            raise ValueError("exact-integer functional component fingerprint drifted")
        if self.implementation_fingerprint != self._expected_implementation_fingerprint():
            raise ValueError("exact-integer functional implementation fingerprint drifted")
        expected_flags = (
            self.api_version == VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION,
            self.authority == VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY,
            self.convention == VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION,
            self.exact_integer_conservation is True,
            self.no_wrap is True,
            self.raw_finite_square_total is True,
            self.literal_float_mask_used is False,
            self.source_closure_established is False,
            self.scalar_hessian_authority_established is False,
            self.reciprocity_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(expected_flags):
            raise ValueError("exact-integer functional authority contract drifted")
        if self._kernel_fingerprint != self._expected_kernel_fingerprint():
            raise ValueError("exact-integer functional kernel fingerprint drifted")

    def interaction_action(self, density: Array) -> Array:
        self.validate_live_state()
        return vituri2024_exact_integer_projected_interaction_action(
            density,
            integer_mesh_labels=self.integer_mesh_labels,
            form_factors_by_flavor=self.form_factors_by_flavor,
            interaction_kernel_by_mesh_pair=self.interaction_kernel_by_mesh_pair,
            area_angstrom_squared=self.area_angstrom_squared,
        )

    def energy(self, density: Array) -> float:
        self.validate_live_state()
        clean = _readonly_hermitian(density, self.dimension, "functional density")
        q = clean - self.normal_order_reference
        value = np.trace(self.h0_full @ clean) + 0.5 * np.trace(
            q @ self.interaction_action(q)
        )
        scale = max(1.0, abs(value))
        if abs(value.imag) > VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE * scale:
            raise ValueError("exact-integer functional energy is not real")
        return float(value.real)

    def fock(self, density: Array) -> Array:
        self.validate_live_state()
        clean = _readonly_hermitian(density, self.dimension, "functional density")
        result = self.h0_full + self.interaction_action(
            clean - self.normal_order_reference
        )
        return _readonly_hermitian(
            np.asarray(result, dtype=np.complex128), self.dimension, "functional Fock"
        )

    def differential_fock(self, direction: Array) -> Array:
        self.validate_live_state()
        clean = _readonly_hermitian(direction, self.dimension, "functional direction")
        return self.interaction_action(clean)


def _exact_integer_manifest_components(
    inputs: TDHFScalarFunctionalInputsManifest,
) -> tuple[Array, Array, Array, Array, Array, float]:
    if type(inputs) is not TDHFScalarFunctionalInputsManifest:
        raise TypeError("exact-integer callbacks require the exact generic input manifest")
    inputs.validate_live_state()
    names = tuple(item.name for item in inputs.entries)
    if names != VITURI2024_EXACT_INTEGER_FUNCTIONAL_INPUT_NAMES:
        raise ValueError("exact-integer callback input inventory drifted")
    if inputs.value("functional_api_version") != (
        VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION
    ):
        raise ValueError("exact-integer callback API binding drifted")
    fingerprint = inputs.value("kernel_fingerprint")
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("exact-integer callback kernel fingerprint is invalid")
    area_value = inputs.value("area_angstrom_squared")
    if isinstance(area_value, (bool, np.bool_)):
        raise TypeError("exact-integer callback area must be a strict real scalar")
    area = float(area_value)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("exact-integer callback area must be positive")
    return (
        inputs.array("integer_mesh_labels"),
        inputs.array("form_factors_by_flavor"),
        inputs.array("interaction_kernel_by_mesh_pair"),
        inputs.array("h0_full"),
        inputs.array("normal_order_reference"),
        area,
    )


def vituri2024_exact_integer_energy_callback(inputs, P):
    labels, form_factors, interaction, h0, reference, area = (
        _exact_integer_manifest_components(inputs)
    )
    dimension = int(h0.shape[0])
    clean = _readonly_hermitian(P, dimension, "callback density")
    q = clean - reference
    sigma = vituri2024_exact_integer_projected_interaction_action(
        q,
        integer_mesh_labels=labels,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction,
        area_angstrom_squared=area,
    )
    value = np.trace(h0 @ clean) + 0.5 * np.trace(q @ sigma)
    scale = max(1.0, abs(value))
    if abs(value.imag) > VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE * scale:
        raise ValueError("exact-integer callback energy is not real")
    return float(value.real)


def vituri2024_exact_integer_fock_callback(inputs, P):
    labels, form_factors, interaction, h0, reference, area = (
        _exact_integer_manifest_components(inputs)
    )
    dimension = int(h0.shape[0])
    clean = _readonly_hermitian(P, dimension, "callback density")
    sigma = vituri2024_exact_integer_projected_interaction_action(
        clean - reference,
        integer_mesh_labels=labels,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction,
        area_angstrom_squared=area,
    )
    return _readonly_hermitian(
        np.asarray(h0 + sigma, dtype=np.complex128),
        dimension,
        "callback Fock",
    )


def vituri2024_exact_integer_fock_derivative_callback(inputs, P, D):
    labels, form_factors, interaction, h0, _reference, area = (
        _exact_integer_manifest_components(inputs)
    )
    dimension = int(h0.shape[0])
    _readonly_hermitian(P, dimension, "callback anchor density")
    direction = _readonly_hermitian(D, dimension, "callback direction")
    return vituri2024_exact_integer_projected_interaction_action(
        direction,
        integer_mesh_labels=labels,
        form_factors_by_flavor=form_factors,
        interaction_kernel_by_mesh_pair=interaction,
        area_angstrom_squared=area,
    )


def make_vituri2024_exact_integer_scalar_inputs(
    kernel: Vituri2024ExactIntegerFunctionalKernel,
    *,
    source_fingerprint: str,
    provenance: str,
) -> TDHFScalarFunctionalInputsManifest:
    """Copy a validated reduced kernel into the generic scalar ABI."""

    if type(kernel) is not Vituri2024ExactIntegerFunctionalKernel:
        raise TypeError("scalar inputs require the exact Vituri integer kernel")
    kernel.validate_live_state()
    return make_tdhf_scalar_functional_inputs_manifest(
        {
            "area_angstrom_squared": kernel.area_angstrom_squared,
            "form_factors_by_flavor": kernel.form_factors_by_flavor,
            "functional_api_version": kernel.api_version,
            "h0_full": kernel.h0_full,
            "integer_mesh_labels": kernel.integer_mesh_labels,
            "interaction_kernel_by_mesh_pair": kernel.interaction_kernel_by_mesh_pair,
            "kernel_fingerprint": kernel.kernel_fingerprint,
            "normal_order_reference": kernel.normal_order_reference,
            "ordered_mesh": kernel.ordered_mesh,
        },
        source_fingerprint=source_fingerprint,
        provenance=provenance,
    )


def bind_vituri2024_exact_integer_scalar_functional() -> (
    TDHFFullProjectorFunctionalBinding
):
    """Bind three distinct callbacks and their exact-integer implementation closure."""

    dependencies = (
        vituri2024_exact_integer_projected_interaction_action,
        _exact_integer_manifest_components,
    )
    return TDHFFullProjectorFunctionalBinding(
        energy=bind_tdhf_scalar_kernel(
            role="energy",
            callback=vituri2024_exact_integer_energy_callback,
            dependencies=dependencies,
            provenance="Vituri exact-integer reduced scalar energy callback.",
        ),
        fock=bind_tdhf_scalar_kernel(
            role="fock",
            callback=vituri2024_exact_integer_fock_callback,
            dependencies=dependencies,
            provenance="Vituri exact-integer reduced scalar Fock callback.",
        ),
        fock_derivative=bind_tdhf_scalar_kernel(
            role="fock_derivative",
            callback=vituri2024_exact_integer_fock_derivative_callback,
            dependencies=dependencies,
            provenance="Vituri exact-integer reduced scalar dF callback.",
        ),
    )


__all__ = [
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_API_VERSION",
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_AUTHORITY",
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_CONVENTION",
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_MAX_NK",
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_STRUCTURE_TOLERANCE",
    "VITURI2024_EXACT_INTEGER_FUNCTIONAL_INPUT_NAMES",
    "Vituri2024ExactIntegerFunctionalKernel",
    "bind_vituri2024_exact_integer_scalar_functional",
    "make_vituri2024_exact_integer_scalar_inputs",
    "vituri2024_exact_integer_energy_callback",
    "vituri2024_exact_integer_fock_callback",
    "vituri2024_exact_integer_fock_derivative_callback",
    "vituri2024_exact_integer_projected_interaction_action",
]
