"""Exact no-wrap FFT candidate backend for homogeneous Vituri-2024 HF.

The dense :mod:`vituri2024_hf` functional remains the oracle.  This module
changes only the evaluation of its finite-domain exchange sum: every flavor
block is expanded into 36 orbital-component linear convolutions and evaluated
with a zero-padded signed-displacement circulant embedding.  The embedding is
an exact algorithm for the finite square domain; it is not a periodic momentum
model and does not introduce reciprocal wrap.

This is an algebraically equivalent candidate backend.  It establishes no UV
or finite-domain convergence, production authority, SCF branch, or paper
reproduction.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
from types import CodeType
from typing import Final

import numpy as np
import scipy
from scipy import fft as scipy_fft

from . import vituri2024_hf_preflight as _hf_preflight_module
from .vituri2024_hf import (
    VITURI2024_TRANSLATIONAL_HF_CONVENTION,
    VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE,
    VITURI2024_TRANSLATIONAL_MESH_POLICY,
    VITURI2024_TRANSLATIONAL_Q0_POLICY,
    Vituri2024FiniteDomainMeshReceipt,
    Vituri2024TranslationalQ0ReproductionChoice,
    _array_sha256,
    _fingerprint,
    _max_abs,
    _readonly,
    _resolve_interaction,
    vituri2024_native_density_to_conventional_k_diagonal,
    vituri2024_native_operator_to_conventional_k_diagonal,
)
from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from .vituri2024_interaction import (
    InteractionInput,
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
    vituri2024_vtf,
)

Array = np.ndarray

VITURI2024_TRANSLATIONAL_HF_FFT_API_VERSION: Final[str] = (
    "vituri2024_translational_equal_weight_finite_domain_hf_exact_fft.v1"
)
VITURI2024_TRANSLATIONAL_HF_FFT_AUTHORITY: Final[str] = (
    "algebraically_equivalent_candidate_backend_not_uv_domain_scf_production_"
    "or_paper_reproduction_qualified"
)
VITURI2024_TRANSLATIONAL_HF_FFT_POLICY: Final[str] = (
    "exact_linear_convolution_zero_padded_signed_displacement_circulant_"
    "embedding_output_0_to_N_no_fftshift_no_momentum_wrap"
)
VITURI2024_CARTESIAN_FLATTEN_ORDER: Final[str] = (
    "iy_outer_ix_inner_centered_increasing_integer_labels"
)

_PLAN_TOKEN = object()
_FFT2 = scipy_fft.fft2
_IFFT2 = scipy_fft.ifft2
_NEXT_FAST_LEN = scipy_fft.next_fast_len


def _strict_positive_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an explicit positive int")
    if value <= 0:
        raise ValueError(f"{label} must be an explicit positive int")
    return value


def _source_file_sha256(path: str | None, label: str) -> str:
    if not path:
        raise RuntimeError(f"{label} has no inspectable source file")
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"{label} source file is unavailable")
    return sha256(source_path.read_bytes()).hexdigest()


def _callable_implementation_receipt(value: object, label: str) -> dict[str, object]:
    if not callable(value):
        raise RuntimeError(f"{label} binding is no longer callable")
    module = inspect.getmodule(value)
    try:
        source_path = inspect.getsourcefile(value)
    except TypeError:
        source_path = None
    source_path = source_path or getattr(module, "__file__", None)
    code = getattr(value, "__code__", None)
    try:
        callable_source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        callable_source = b""
    return {
        "label": label,
        "module": getattr(value, "__module__", type(value).__module__),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        "callable_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "source_file_sha256": _source_file_sha256(source_path, label),
        "callable_source_sha256": sha256(callable_source).hexdigest(),
        "code_sha256": (
            sha256(code.co_code).hexdigest()
            if isinstance(code, CodeType)
            else None
        ),
    }


def _implementation_fingerprint() -> str:
    return _fingerprint(
        {
            "own_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "callable_bindings": {
                "native_density_converter": _callable_implementation_receipt(
                    vituri2024_native_density_to_conventional_k_diagonal,
                    "native density converter",
                ),
                "native_operator_converter": _callable_implementation_receipt(
                    vituri2024_native_operator_to_conventional_k_diagonal,
                    "native operator converter",
                ),
                "interaction_kernel": _callable_implementation_receipt(
                    vituri2024_vtf, "Vituri interaction kernel"
                ),
                "fft2": _callable_implementation_receipt(_FFT2, "FFT fft2"),
                "ifft2": _callable_implementation_receipt(_IFFT2, "FFT ifft2"),
                "next_fast_len": _callable_implementation_receipt(
                    _NEXT_FAST_LEN, "FFT next_fast_len"
                ),
            },
            "flavor_orders": {
                "source_file_sha256": _source_file_sha256(
                    getattr(_hf_preflight_module, "__file__", None),
                    "Vituri HF flavor-order module",
                ),
                "active_band_states_valley_order_type": type(
                    ACTIVE_BAND_STATES_VALLEY_ORDER
                ).__name__,
                "active_band_states_valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
                "internal_flavor_order_type": type(INTERNAL_FLAVOR_ORDER).__name__,
                "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
            },
        }
    )


_IMPORT_IMPLEMENTATION_FINGERPRINT = _implementation_fingerprint()


def _validate_centered_square_cartesian_mesh(
    integer_mesh_labels: Array,
    ordered_mesh: Array,
    delta_k_inverse_angstrom: float,
) -> tuple[Array, Array, int]:
    if (
        not isinstance(integer_mesh_labels, np.ndarray)
        or integer_mesh_labels.dtype != np.dtype(np.int64)
        or integer_mesh_labels.ndim != 2
        or integer_mesh_labels.shape[1:] != (2,)
    ):
        raise ValueError("FFT labels must be an int64 (Nk,2) array")
    nk = int(integer_mesh_labels.shape[0])
    size = math.isqrt(nk)
    if size * size != nk or size < 3 or size % 2 != 1:
        raise ValueError("FFT labels must form a complete centered odd NxN square")
    half = size // 2
    expected = np.asarray(
        [
            (ix, iy)
            for iy in range(-half, half + 1)
            for ix in range(-half, half + 1)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(integer_mesh_labels, expected):
        raise ValueError(
            "FFT labels must be complete centered square in iy-outer/ix-inner order"
        )
    if (
        not isinstance(ordered_mesh, np.ndarray)
        or ordered_mesh.dtype != np.dtype(np.float64)
        or ordered_mesh.shape != (nk, 2)
        or not np.all(np.isfinite(ordered_mesh))
    ):
        raise ValueError("FFT ordered_mesh must be finite float64 (Nk,2)")
    if isinstance(delta_k_inverse_angstrom, (bool, np.bool_)) or not isinstance(
        delta_k_inverse_angstrom, (float, np.floating)
    ):
        raise TypeError("FFT delta_k must be an explicit real float")
    delta_k = float(delta_k_inverse_angstrom)
    if not math.isfinite(delta_k) or delta_k <= 0.0:
        raise ValueError("FFT delta_k must be positive and finite")
    expected_mesh = np.asarray(expected, dtype=np.float64) * delta_k
    if not np.array_equal(ordered_mesh, expected_mesh):
        raise ValueError("FFT ordered_mesh must equal integer labels times delta_k exactly")
    return (
        _readonly(integer_mesh_labels, dtype=np.dtype(np.int64)),
        _readonly(ordered_mesh, dtype=np.dtype(np.float64)),
        size,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SquareCartesianFFTPlan:
    """Immutable exact linear-convolution plan for one centered odd square.

    ``kernel_by_signed_displacement[dy+N-1, dx+N-1]`` stores the kernel at
    the literal integer displacement ``(dx,dy)``.  It is embedded at
    ``(dy mod P, dx mod P)`` with ``P >= 2N-1``.  Circular convolution on the
    padded array then equals the desired finite-domain linear convolution on
    output indices ``0:N``; no ``fftshift`` or physical wrap is used.
    """

    _factory_token: InitVar[object]
    integer_mesh_labels: Array
    ordered_mesh: Array
    delta_k_inverse_angstrom: float
    kernel_by_signed_displacement: Array
    fft_workers: int
    mesh_size: int = field(init=False)
    nk: int = field(init=False)
    padding_size: int = field(init=False)
    minimum_padding_size: int = field(init=False)
    axial_k_cutoff_inverse_angstrom: float = field(init=False)
    corner_k_cutoff_inverse_angstrom: float = field(init=False)
    domain_label_min: int = field(init=False)
    domain_label_max: int = field(init=False)
    domain_endpoints_included: bool = field(default=True, init=False)
    output_index_start: int = field(default=0, init=False)
    output_index_stop: int = field(init=False)
    flatten_order: str = field(default=VITURI2024_CARTESIAN_FLATTEN_ORDER, init=False)
    no_wrap_policy: str = field(default=VITURI2024_TRANSLATIONAL_HF_FFT_POLICY, init=False)
    labels_sha256: str = field(init=False)
    ordered_mesh_sha256: str = field(init=False)
    signed_displacement_kernel_sha256: str = field(init=False)
    kernel_embedding_sha256: str = field(init=False)
    kernel_fft_sha256: str = field(init=False)
    implementation_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)
    kernel_embedding: Array = field(init=False, repr=False)
    kernel_fft: Array = field(init=False, repr=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PLAN_TOKEN:
            raise TypeError("Vituri FFT plans are factory-only")
        labels, mesh, size = _validate_centered_square_cartesian_mesh(
            self.integer_mesh_labels,
            self.ordered_mesh,
            self.delta_k_inverse_angstrom,
        )
        workers = _strict_positive_int(self.fft_workers, "fft_workers")
        displacement_size = 2 * size - 1
        if (
            not isinstance(self.kernel_by_signed_displacement, np.ndarray)
            or self.kernel_by_signed_displacement.dtype != np.dtype(np.complex128)
            or self.kernel_by_signed_displacement.shape
            != (displacement_size, displacement_size)
            or not np.all(np.isfinite(self.kernel_by_signed_displacement))
        ):
            raise ValueError(
                "signed-displacement kernel must be finite complex128 (2N-1,2N-1)"
            )
        signed_kernel = _readonly(
            self.kernel_by_signed_displacement, dtype=np.dtype(np.complex128)
        )
        implementation_fingerprint = _implementation_fingerprint()
        if implementation_fingerprint != _IMPORT_IMPLEMENTATION_FINGERPRINT:
            raise RuntimeError("Vituri FFT implementation bindings drifted before construction")
        padding = int(_NEXT_FAST_LEN(displacement_size))
        if padding < displacement_size:
            raise RuntimeError("SciPy FFT padding violated P>=2N-1")
        embedding = np.zeros((padding, padding), dtype=np.complex128)
        half_displacement = size - 1
        for dy in range(-half_displacement, half_displacement + 1):
            for dx in range(-half_displacement, half_displacement + 1):
                embedding[dy % padding, dx % padding] = signed_kernel[
                    dy + half_displacement, dx + half_displacement
                ]
        kernel_fft = _FFT2(embedding, workers=workers)
        half = size // 2
        object.__setattr__(self, "integer_mesh_labels", labels)
        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "delta_k_inverse_angstrom", float(self.delta_k_inverse_angstrom))
        object.__setattr__(self, "kernel_by_signed_displacement", signed_kernel)
        object.__setattr__(self, "fft_workers", workers)
        object.__setattr__(self, "mesh_size", size)
        object.__setattr__(self, "nk", size * size)
        object.__setattr__(self, "padding_size", padding)
        object.__setattr__(self, "minimum_padding_size", displacement_size)
        object.__setattr__(self, "axial_k_cutoff_inverse_angstrom", half * float(self.delta_k_inverse_angstrom))
        object.__setattr__(self, "corner_k_cutoff_inverse_angstrom", math.sqrt(2.0) * half * float(self.delta_k_inverse_angstrom))
        object.__setattr__(self, "domain_label_min", -half)
        object.__setattr__(self, "domain_label_max", half)
        object.__setattr__(self, "output_index_stop", size)
        object.__setattr__(self, "labels_sha256", _array_sha256(labels))
        object.__setattr__(self, "ordered_mesh_sha256", _array_sha256(mesh))
        object.__setattr__(self, "signed_displacement_kernel_sha256", _array_sha256(signed_kernel))
        readonly_embedding = _readonly(embedding, dtype=np.dtype(np.complex128))
        readonly_kernel_fft = _readonly(kernel_fft, dtype=np.dtype(np.complex128))
        object.__setattr__(self, "kernel_embedding", readonly_embedding)
        object.__setattr__(self, "kernel_fft", readonly_kernel_fft)
        object.__setattr__(self, "kernel_embedding_sha256", _array_sha256(readonly_embedding))
        object.__setattr__(self, "kernel_fft_sha256", _array_sha256(readonly_kernel_fft))
        object.__setattr__(
            self, "implementation_fingerprint", implementation_fingerprint
        )
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_TRANSLATIONAL_HF_FFT_API_VERSION,
                "mesh_size": self.mesh_size,
                "nk": self.nk,
                "padding_size": self.padding_size,
                "minimum_padding_size": self.minimum_padding_size,
                "delta_k_inverse_angstrom": self.delta_k_inverse_angstrom,
                "axial_k_cutoff_inverse_angstrom": self.axial_k_cutoff_inverse_angstrom,
                "corner_k_cutoff_inverse_angstrom": self.corner_k_cutoff_inverse_angstrom,
                "domain_label_min": self.domain_label_min,
                "domain_label_max": self.domain_label_max,
                "domain_endpoints_included": self.domain_endpoints_included,
                "output_index_start": self.output_index_start,
                "output_index_stop": self.output_index_stop,
                "flatten_order": self.flatten_order,
                "no_wrap_policy": self.no_wrap_policy,
                "fft_workers": self.fft_workers,
                "labels_sha256": self.labels_sha256,
                "ordered_mesh_sha256": self.ordered_mesh_sha256,
                "signed_displacement_kernel_sha256": self.signed_displacement_kernel_sha256,
                "kernel_embedding_sha256": self.kernel_embedding_sha256,
                "kernel_fft_sha256": self.kernel_fft_sha256,
                "implementation_fingerprint": self.implementation_fingerprint,
            }
        )

    def validate_live_state(self) -> None:
        live_implementation = _implementation_fingerprint()
        if (
            live_implementation != _IMPORT_IMPLEMENTATION_FINGERPRINT
            or live_implementation != self.implementation_fingerprint
        ):
            raise ValueError("Vituri FFT implementation binding or source drifted")
        displacement_size = 2 * self.mesh_size - 1
        arrays = (
            (self.integer_mesh_labels, np.dtype(np.int64), (self.nk, 2), self.labels_sha256),
            (self.ordered_mesh, np.dtype(np.float64), (self.nk, 2), self.ordered_mesh_sha256),
            (
                self.kernel_by_signed_displacement,
                np.dtype(np.complex128),
                (displacement_size, displacement_size),
                self.signed_displacement_kernel_sha256,
            ),
            (
                self.kernel_embedding,
                np.dtype(np.complex128),
                (self.padding_size, self.padding_size),
                self.kernel_embedding_sha256,
            ),
            (
                self.kernel_fft,
                np.dtype(np.complex128),
                (self.padding_size, self.padding_size),
                self.kernel_fft_sha256,
            ),
        )
        for value, dtype, shape, digest in arrays:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or not np.all(np.isfinite(value))
                or _array_sha256(value) != digest
            ):
                raise ValueError("Vituri FFT plan array or hash drifted")
        expected_padding = int(_NEXT_FAST_LEN(displacement_size))
        locked = (
            type(self.fft_workers) is int,
            self.fft_workers > 0,
            self.mesh_size >= 3,
            self.mesh_size % 2 == 1,
            self.nk == self.mesh_size * self.mesh_size,
            self.padding_size == expected_padding,
            self.minimum_padding_size == displacement_size,
            self.padding_size >= self.minimum_padding_size,
            self.domain_label_min == -(self.mesh_size // 2),
            self.domain_label_max == self.mesh_size // 2,
            self.domain_endpoints_included is True,
            self.output_index_start == 0,
            self.output_index_stop == self.mesh_size,
            self.flatten_order == VITURI2024_CARTESIAN_FLATTEN_ORDER,
            self.no_wrap_policy == VITURI2024_TRANSLATIONAL_HF_FFT_POLICY,
        )
        if not all(locked) or self._current_fingerprint() != self.fingerprint:
            raise ValueError("Vituri FFT plan metadata or fingerprint drifted")
        _validate_centered_square_cartesian_mesh(
            self.integer_mesh_labels,
            self.ordered_mesh,
            self.delta_k_inverse_angstrom,
        )

    def convolve(self, source: Array) -> Array:
        """Validate once, then return ``sum_r K[m-r] source[r]`` on ``0:N``."""

        self.validate_live_state()
        return self._convolve_validated(source)

    def _convolve_validated(self, source: Array) -> Array:
        """Convolve after the caller has validated this immutable plan once."""

        if (
            not isinstance(source, np.ndarray)
            or source.dtype != np.dtype(np.complex128)
            or source.shape != (self.mesh_size, self.mesh_size)
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("FFT convolution source must be finite complex128 (N,N)")
        padded = np.zeros(
            (self.padding_size, self.padding_size), dtype=np.complex128
        )
        padded[: self.mesh_size, : self.mesh_size] = source
        transformed = _FFT2(padded, workers=self.fft_workers)
        convolved = _IFFT2(
            self.kernel_fft * transformed, workers=self.fft_workers
        )
        return np.asarray(
            convolved[: self.mesh_size, : self.mesh_size], dtype=np.complex128
        )


def make_vituri2024_square_cartesian_fft_plan(
    *,
    integer_mesh_labels: Array,
    ordered_mesh: Array,
    delta_k_inverse_angstrom: float,
    kernel_by_signed_displacement: Array,
    fft_workers: int,
) -> Vituri2024SquareCartesianFFTPlan:
    """Build an immutable exact no-wrap plan from an explicit signed kernel."""

    return Vituri2024SquareCartesianFFTPlan(
        _factory_token=_PLAN_TOKEN,
        integer_mesh_labels=integer_mesh_labels,
        ordered_mesh=ordered_mesh,
        delta_k_inverse_angstrom=delta_k_inverse_angstrom,
        kernel_by_signed_displacement=kernel_by_signed_displacement,
        fft_workers=fft_workers,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024TranslationalHFFFTFunctional:
    """Memory-scalable algebraically equivalent candidate Vituri HF backend.

    It does not establish UV convergence, production readiness, an SCF branch,
    or paper reproduction.  The dense ``Vituri2024TranslationalHFFunctional``
    remains the independent finite-domain algebra oracle.
    """

    ordered_mesh: Array
    integer_mesh_labels: Array
    delta_k_inverse_angstrom: float
    active_band_states: Array
    h0_native: Array
    normal_order_reference_native: Array
    mesh_receipt: Vituri2024FiniteDomainMeshReceipt
    interaction: InteractionInput
    normal_order_reference_fingerprint: str
    q0_choice: Vituri2024TranslationalQ0ReproductionChoice
    provenance: str
    fft_workers: int
    normal_order_reference_conventional: Array = field(init=False, repr=False)
    state_norms_by_flavor: Array = field(init=False, repr=False)
    fft_plan: Vituri2024SquareCartesianFFTPlan = field(init=False, repr=False)
    interaction_receipt: Vituri2024InteractionChoiceReceipt = field(init=False)
    interaction_fingerprint: str = field(init=False)
    nk: int = field(init=False)
    implementation_fingerprint: str = field(init=False)
    construction_fingerprint: str = field(init=False)
    api_version: str = field(default=VITURI2024_TRANSLATIONAL_HF_FFT_API_VERSION, init=False)
    authority: str = field(default=VITURI2024_TRANSLATIONAL_HF_FFT_AUTHORITY, init=False)
    convention: str = field(default=VITURI2024_TRANSLATIONAL_HF_CONVENTION, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    q0_background_authority_established: bool = field(default=False, init=False)
    uv_convergence_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        implementation_fingerprint = _implementation_fingerprint()
        if implementation_fingerprint != _IMPORT_IMPLEMENTATION_FINGERPRINT:
            raise RuntimeError("Vituri FFT implementation bindings drifted before construction")
        labels, mesh, size = _validate_centered_square_cartesian_mesh(
            self.integer_mesh_labels,
            self.ordered_mesh,
            self.delta_k_inverse_angstrom,
        )
        nk = size * size
        workers = _strict_positive_int(self.fft_workers, "fft_workers")
        if (
            not isinstance(self.active_band_states, np.ndarray)
            or self.active_band_states.dtype != np.dtype(np.complex128)
            or self.active_band_states.shape
            != (len(ACTIVE_BAND_STATES_VALLEY_ORDER), 6, nk)
            or not np.all(np.isfinite(self.active_band_states))
        ):
            raise ValueError("FFT active states must be finite complex128 (2,6,Nk)")
        states = _readonly(self.active_band_states, dtype=np.dtype(np.complex128))
        valley_norms = np.sum(np.abs(states) ** 2, axis=1)
        if _max_abs(valley_norms - 1.0) > 5.0e-12:
            raise ValueError("FFT active states are not normalized")
        h0 = vituri2024_native_operator_to_conventional_k_diagonal(self.h0_native)
        if (
            not isinstance(self.normal_order_reference_native, np.ndarray)
            or self.normal_order_reference_native.dtype != np.dtype(np.complex128)
        ):
            raise TypeError("FFT native reference must be exact complex128")
        reference_native = _readonly(
            self.normal_order_reference_native, dtype=np.dtype(np.complex128)
        )
        reference = vituri2024_native_density_to_conventional_k_diagonal(
            reference_native
        )
        if h0.shape != (4, 4, nk) or reference.shape != (4, 4, nk):
            raise ValueError("FFT h0/reference Nk mismatch")
        for momentum in range(nk):
            eigenvalues = np.linalg.eigvalsh(reference[:, :, momentum])
            if eigenvalues[0] < -5.0e-12 or eigenvalues[-1] > 1.0 + 5.0e-12:
                raise ValueError("FFT reference must satisfy 0<=R<=I")
        if type(self.mesh_receipt) is not Vituri2024FiniteDomainMeshReceipt:
            raise TypeError("FFT functional requires an exact mesh receipt")
        if (
            self.mesh_receipt.nk != nk
            or not np.array_equal(self.mesh_receipt.ordered_mesh, mesh)
        ):
            raise ValueError("FFT mesh/receipt binding drifted")
        if type(self.q0_choice) is not Vituri2024TranslationalQ0ReproductionChoice:
            raise TypeError("FFT functional requires an exact q0 reproduction choice")
        interaction, interaction_fingerprint = _resolve_interaction(self.interaction)
        if (
            type(self.normal_order_reference_fingerprint) is not str
            or self.normal_order_reference_fingerprint
            != _array_sha256(reference_native)
        ):
            raise ValueError("FFT normal reference fingerprint mismatch")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("FFT provenance must be explicit")
        displacement_size = 2 * size - 1
        signed_kernel = np.empty(
            (displacement_size, displacement_size), dtype=np.complex128
        )
        offset = size - 1
        delta_k = float(self.delta_k_inverse_angstrom)
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                transfer = delta_k * math.hypot(dx, dy)
                signed_kernel[dy + offset, dx + offset] = vituri2024_vtf(
                    transfer, interaction
                )
        plan = make_vituri2024_square_cartesian_fft_plan(
            integer_mesh_labels=labels,
            ordered_mesh=mesh,
            delta_k_inverse_angstrom=delta_k,
            kernel_by_signed_displacement=signed_kernel,
            fft_workers=workers,
        )
        valley_index = {
            valley: index
            for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        flavor_norms = np.stack(
            [valley_norms[valley_index[valley]] for valley, _spin in INTERNAL_FLAVOR_ORDER],
            axis=0,
        )
        object.__setattr__(self, "ordered_mesh", mesh)
        object.__setattr__(self, "integer_mesh_labels", labels)
        object.__setattr__(self, "delta_k_inverse_angstrom", delta_k)
        object.__setattr__(self, "active_band_states", states)
        object.__setattr__(self, "h0_native", h0)
        object.__setattr__(self, "normal_order_reference_native", reference_native)
        object.__setattr__(self, "normal_order_reference_conventional", reference)
        object.__setattr__(self, "state_norms_by_flavor", _readonly(flavor_norms, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "fft_workers", workers)
        object.__setattr__(self, "fft_plan", plan)
        object.__setattr__(self, "interaction_receipt", interaction)
        object.__setattr__(self, "interaction_fingerprint", interaction_fingerprint)
        object.__setattr__(self, "nk", nk)
        object.__setattr__(
            self, "implementation_fingerprint", implementation_fingerprint
        )
        object.__setattr__(self, "construction_fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": self.api_version,
                "authority": self.authority,
                "convention": self.convention,
                "mesh": _array_sha256(self.ordered_mesh),
                "labels": _array_sha256(self.integer_mesh_labels),
                "delta_k_inverse_angstrom": self.delta_k_inverse_angstrom,
                "states": _array_sha256(self.active_band_states),
                "state_norms": _array_sha256(self.state_norms_by_flavor),
                "h0": _array_sha256(self.h0_native),
                "reference_native": _array_sha256(self.normal_order_reference_native),
                "reference_conventional": _array_sha256(self.normal_order_reference_conventional),
                "mesh_receipt": self.mesh_receipt.fingerprint,
                "interaction": self.interaction_fingerprint,
                "interaction_input_kind": type(self.interaction).__name__,
                "interaction_binding_flags": (
                    (
                        self.interaction.paper_direct_claim_allowed,
                        self.interaction.establishes_hf_q0_background,
                    )
                    if type(self.interaction) is Vituri2024InteractionBinding
                    else (False, False)
                ),
                "normal_reference_fingerprint": self.normal_order_reference_fingerprint,
                "q0_choice": self.q0_choice.fingerprint,
                "fft_workers": self.fft_workers,
                "fft_plan": self.fft_plan.fingerprint,
                "implementation": self.implementation_fingerprint,
                "provenance": self.provenance,
                "authority_flags": (
                    self.source_stationarity_established,
                    self.q0_background_authority_established,
                    self.uv_convergence_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        arrays = (
            (self.ordered_mesh, np.dtype(np.float64), (self.nk, 2)),
            (self.integer_mesh_labels, np.dtype(np.int64), (self.nk, 2)),
            (self.active_band_states, np.dtype(np.complex128), (2, 6, self.nk)),
            (self.h0_native, np.dtype(np.complex128), (4, 4, self.nk)),
            (
                self.normal_order_reference_native,
                np.dtype(np.complex128),
                (4, 4, self.nk),
            ),
            (
                self.normal_order_reference_conventional,
                np.dtype(np.complex128),
                (4, 4, self.nk),
            ),
            (self.state_norms_by_flavor, np.dtype(np.float64), (4, self.nk)),
        )
        for value, dtype, shape in arrays:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("FFT functional live array drifted")
        self.mesh_receipt.validate_live_state()
        self.q0_choice.validate_live_state()
        self.fft_plan.validate_live_state()
        resolved_interaction, resolved_fingerprint = _resolve_interaction(self.interaction)
        locked = (
            type(self.nk) is int,
            self.nk == self.fft_plan.nk,
            self.fft_workers == self.fft_plan.fft_workers,
            self.api_version == VITURI2024_TRANSLATIONAL_HF_FFT_API_VERSION,
            self.authority == VITURI2024_TRANSLATIONAL_HF_FFT_AUTHORITY,
            self.convention == VITURI2024_TRANSLATIONAL_HF_CONVENTION,
            self.mesh_receipt.policy == VITURI2024_TRANSLATIONAL_MESH_POLICY,
            self.q0_choice.policy == VITURI2024_TRANSLATIONAL_Q0_POLICY,
            self.interaction_receipt == resolved_interaction,
            self.interaction_fingerprint == resolved_fingerprint,
            self.normal_order_reference_fingerprint
            == _array_sha256(self.normal_order_reference_native),
            _implementation_fingerprint() == _IMPORT_IMPLEMENTATION_FINGERPRINT,
            self.implementation_fingerprint == _IMPORT_IMPLEMENTATION_FINGERPRINT,
            self.source_stationarity_established is False,
            self.q0_background_authority_established is False,
            self.uv_convergence_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("FFT functional binding or authority drifted")
        if hasattr(self, "construction_fingerprint") and (
            self._current_fingerprint() != self.construction_fingerprint
        ):
            raise ValueError("FFT functional construction drifted")

    @property
    def fingerprint(self) -> str:
        self.validate_live_state()
        return self.construction_fingerprint

    def _hartree_action_conventional_validated(self, clean: Array) -> Array:
        result = np.zeros_like(clean)
        direct_scalar = np.sum(
            self.state_norms_by_flavor * np.diagonal(clean, axis1=0, axis2=1).T
        )
        q0 = self.fft_plan.kernel_by_signed_displacement[
            self.fft_plan.mesh_size - 1, self.fft_plan.mesh_size - 1
        ]
        direct_scalar *= q0
        for flavor in range(4):
            result[flavor, flavor, :] = (
                self.state_norms_by_flavor[flavor] * direct_scalar
            )
        return result

    def hartree_action_conventional(self, density: Array) -> Array:
        """Return the norm-weighted retained-q0 direct term alone."""

        self.validate_live_state()
        clean = vituri2024_native_operator_to_conventional_k_diagonal(density)
        if clean.shape != (4, 4, self.nk):
            raise ValueError("FFT Hartree density Nk mismatch")
        result = self._hartree_action_conventional_validated(clean)
        result *= 1.0 / self.mesh_receipt.area_angstrom_squared
        return _readonly(result, dtype=np.dtype(np.complex128))

    def _exchange_action_conventional_validated(self, clean: Array) -> Array:
        """Expand the dense-oracle overlap product before convolution.

        For orbitals ``c,d`` and flavors ``a,b``,

        ``F_a(m,r) F_b(r,m) P_ab(r)``

        equals

        ``[u_ac(m)* u_bd(m)] [u_ac(r) u_bd(r)* P_ab(r)]``.

        Thus each flavor block is exactly the sum of 6x6 scalar convolutions;
        no phase, valley spinor, or off-diagonal flavor block is discarded.
        """

        size = self.fft_plan.mesh_size
        result = np.zeros_like(clean)
        valley_index = {
            valley: index
            for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        spinors = tuple(
            self.active_band_states[valley_index[valley]].reshape(6, size, size)
            for valley, _spin in INTERNAL_FLAVOR_ORDER
        )
        for left_flavor in range(4):
            left_spinor = spinors[left_flavor]
            for right_flavor in range(4):
                block_flat = clean[left_flavor, right_flavor, :]
                if np.count_nonzero(block_flat) == 0:
                    continue
                block = block_flat.reshape(size, size)
                right_spinor = spinors[right_flavor]
                exchange_block = result[left_flavor, right_flavor, :].reshape(
                    size, size
                )
                for left_orbital in range(6):
                    left_component = left_spinor[left_orbital]
                    for right_orbital in range(6):
                        right_component = right_spinor[right_orbital]
                        source = np.asarray(
                            left_component
                            * right_component.conj()
                            * block,
                            dtype=np.complex128,
                        )
                        convolution = self.fft_plan._convolve_validated(source)
                        exchange_block -= (
                            left_component.conj()
                            * right_component
                            * convolution
                        )
        return result

    def exchange_action_conventional(self, density: Array) -> Array:
        """Return all 16 exchange blocks from 36 convolutions per nonzero block."""

        self.validate_live_state()
        clean = vituri2024_native_operator_to_conventional_k_diagonal(density)
        if clean.shape != (4, 4, self.nk):
            raise ValueError("FFT exchange density Nk mismatch")
        result = self._exchange_action_conventional_validated(clean)
        result *= 1.0 / self.mesh_receipt.area_angstrom_squared
        return _readonly(result, dtype=np.dtype(np.complex128))

    def _interaction_action_conventional_validated(self, density: Array) -> Array:
        clean = vituri2024_native_operator_to_conventional_k_diagonal(density)
        if clean.shape != (4, 4, self.nk):
            raise ValueError("FFT action density Nk mismatch")
        result = self._hartree_action_conventional_validated(clean)
        result += self._exchange_action_conventional_validated(clean)
        result *= 1.0 / self.mesh_receipt.area_angstrom_squared
        residual = _max_abs(result - result.swapaxes(0, 1).conj())
        if residual > VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE * max(
            1.0, _max_abs(result)
        ):
            raise ValueError("FFT interaction action is not Hermitian")
        return _readonly(result, dtype=np.dtype(np.complex128))

    def interaction_action_conventional(self, density: Array) -> Array:
        """Apply the exact finite-domain action without dense Nk-by-Nk arrays."""

        self.validate_live_state()
        return self._interaction_action_conventional_validated(density)

    def interaction_action(self, native_density: Array) -> Array:
        self.validate_live_state()
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        return self._interaction_action_conventional_validated(conventional)

    def make_validated_interaction_action(self):
        """Return an SCF action that rejects post-validation runtime drift.

        The fingerprint check runs once per interaction action, not inside the
        36 orbital convolutions.  This closes the gap between factory-time
        validation and later global lookups made by the FFT hot path.
        """

        self.validate_live_state()
        density_converter = vituri2024_native_density_to_conventional_k_diagonal
        action = self._interaction_action_conventional_validated
        implementation_fingerprint = self.implementation_fingerprint

        def validated_action(native_density: Array) -> Array:
            if _implementation_fingerprint() != implementation_fingerprint:
                raise RuntimeError("FFT validated action runtime binding drifted")
            return action(density_converter(native_density))

        return validated_action

    def energy(self, native_density: Array) -> float:
        self.validate_live_state()
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        if conventional.shape != (4, 4, self.nk):
            raise ValueError("FFT energy density Nk mismatch")
        difference = conventional - self.normal_order_reference_conventional
        interaction = self._interaction_action_conventional_validated(difference)
        one_body = np.einsum(
            "abk,bak->", self.h0_native, conventional, optimize=False
        )
        interaction_energy = 0.5 * np.einsum(
            "abk,bak->", interaction, difference, optimize=False
        )
        total = complex(one_body + interaction_energy)
        if abs(total.imag) > VITURI2024_TRANSLATIONAL_HF_STRUCTURE_TOLERANCE * max(
            1.0, abs(total), abs(one_body), abs(interaction_energy)
        ):
            raise ValueError("FFT scalar energy is materially complex")
        return float(total.real)

    def fock(self, native_density: Array) -> Array:
        self.validate_live_state()
        conventional = vituri2024_native_density_to_conventional_k_diagonal(
            native_density
        )
        if conventional.shape != (4, 4, self.nk):
            raise ValueError("FFT Fock density Nk mismatch")
        interaction = self._interaction_action_conventional_validated(
            conventional - self.normal_order_reference_conventional
        )
        return vituri2024_native_operator_to_conventional_k_diagonal(
            self.h0_native + interaction
        )

    def fock_derivative(
        self, native_density: Array, native_direction: Array
    ) -> Array:
        self.validate_live_state()
        anchor = vituri2024_native_density_to_conventional_k_diagonal(native_density)
        direction = vituri2024_native_density_to_conventional_k_diagonal(
            native_direction
        )
        if anchor.shape != (4, 4, self.nk) or direction.shape != (4, 4, self.nk):
            raise ValueError("FFT dF anchor/direction Nk mismatch")
        return self._interaction_action_conventional_validated(direction)


__all__ = [
    "VITURI2024_CARTESIAN_FLATTEN_ORDER",
    "VITURI2024_TRANSLATIONAL_HF_FFT_API_VERSION",
    "VITURI2024_TRANSLATIONAL_HF_FFT_AUTHORITY",
    "VITURI2024_TRANSLATIONAL_HF_FFT_POLICY",
    "Vituri2024SquareCartesianFFTPlan",
    "Vituri2024TranslationalHFFFTFunctional",
    "make_vituri2024_square_cartesian_fft_plan",
]
