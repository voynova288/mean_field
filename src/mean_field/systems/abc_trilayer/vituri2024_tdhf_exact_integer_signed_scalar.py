"""Scalable signed-channel exact-integer Vituri scalar-action candidate.

This implementation is intentionally independent of the orbital-Hessian
response module.  It consumes only selected-spin source-gauge spinors, a
validated no-wrap FFT plan, an integer displacement/valley charge, and one
signed density block.  It carries no scalar-Hessian, reciprocity, eigensolver,
stability, production, or paper authority.
"""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
from typing import Final

import numpy as np
from scipy.fft import fft2 as _FFT2, ifft2 as _IFFT2

from .vituri2024_hf_fft import Vituri2024SquareCartesianFFTPlan

Array = np.ndarray
_IMPORT_FFT2 = _FFT2
_IMPORT_IFFT2 = _IFFT2

VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_selected_spin_signed_scalar_fft.v1"
)
VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY: Final[str] = (
    "independent_candidate_signed_channel_action_not_scalar_hessian_reciprocity_"
    "eigensolver_stability_production_or_paper_authority"
)


def _callable_dependency_fingerprint(value: object) -> dict[str, str]:
    module = inspect.getmodule(value)
    source_file_value = inspect.getsourcefile(module) if module is not None else None
    if module is None or source_file_value is None:
        raise ValueError("signed scalar dependency has no source file")
    source_file = Path(source_file_value).resolve()
    return {
        "module": getattr(value, "__module__", ""),
        "qualname": getattr(value, "__qualname__", ""),
        "module_file": str(source_file),
        "module_sha256": sha256(source_file.read_bytes()).hexdigest(),
    }


def _validate_import_bindings() -> None:
    if _FFT2 is not _IMPORT_FFT2 or _IFFT2 is not _IMPORT_IFFT2:
        raise RuntimeError("signed scalar FFT runtime binding drifted")


def _readonly_complex(value: object, shape: tuple[int, ...], label: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.complex128)
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{label} must be finite exact complex128 {shape}")
    result = np.frombuffer(value.tobytes(order="C"), dtype=np.complex128).reshape(shape)
    result.setflags(write=False)
    return result


def _strict_displacement(value: object) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise TypeError("signed scalar displacement must be exact tuple[int,int]")
    return value


def _allowed_blocks(valley_charge: int) -> tuple[tuple[int, int], ...]:
    if type(valley_charge) is not int or valley_charge not in (-2, 0, 2):
        raise ValueError("signed scalar valley charge must be one of -2,0,2")
    if valley_charge == 0:
        return ((0, 0), (1, 1))
    if valley_charge == 2:
        return ((1, 0),)
    return ((0, 1),)


def _support(
    plan: Vituri2024SquareCartesianFFTPlan, displacement: tuple[int, int]
) -> tuple[Array, Array]:
    labels = plan.integer_mesh_labels
    half = plan.mesh_size // 2
    dx, dy = displacement
    target_x = labels[:, 0] + dx
    target_y = labels[:, 1] + dy
    keep = (
        (target_x >= -half)
        & (target_x <= half)
        & (target_y >= -half)
        & (target_y <= half)
    )
    bases = np.flatnonzero(keep).astype(np.int64)
    targets = (
        (target_y[keep] + half) * plan.mesh_size + target_x[keep] + half
    ).astype(np.int64)
    if not np.all(labels[targets] - labels[bases] == np.asarray(displacement)):
        raise RuntimeError("signed scalar support indexing drifted")
    return bases, targets


def vituri2024_exact_integer_signed_scalar_fft_action(
    selected_spinors: Array,
    fft_plan: Vituri2024SquareCartesianFFTPlan,
    area_angstrom_squared: float,
    displacement: tuple[int, int],
    valley_charge: int,
    signed_density_block: Array,
) -> Array:
    """Apply an independent no-wrap FFT ``Sigma`` to one signed block."""

    _validate_import_bindings()
    if type(fft_plan) is not Vituri2024SquareCartesianFFTPlan:
        raise TypeError("signed scalar action requires the exact FFT plan type")
    fft_plan.validate_live_state()
    displacement = _strict_displacement(displacement)
    allowed = _allowed_blocks(valley_charge)
    size = fft_plan.mesh_size
    nk = fft_plan.nk
    if any(abs(item) >= size for item in displacement):
        raise ValueError("signed scalar displacement lies outside the finite square")
    if isinstance(area_angstrom_squared, (bool, np.bool_)):
        raise TypeError("signed scalar area must be a strict real scalar")
    area = float(area_angstrom_squared)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("signed scalar area must be positive")
    signed_kernel = fft_plan.kernel_by_signed_displacement
    kernel_scale = max(1.0, float(np.max(np.abs(signed_kernel))))
    kernel_tolerance = 64.0 * np.finfo(np.float64).eps * kernel_scale
    if (
        float(np.max(np.abs(signed_kernel.imag))) > kernel_tolerance
        or float(np.max(np.abs(signed_kernel - signed_kernel[::-1, ::-1])))
        > kernel_tolerance
    ):
        raise ValueError(
            "signed scalar FFT requires the source-bound real-even kernel contract"
        )
    spinors = _readonly_complex(selected_spinors, (2, 6, nk), "selected spinors")
    block = _readonly_complex(
        signed_density_block, (2, 2, nk), "signed density block"
    )
    bases, targets = _support(fft_plan, displacement)
    support = np.zeros(nk, dtype=np.bool_)
    support[bases] = True
    allowed_mask = np.zeros((2, 2), dtype=np.bool_)
    for left, right in allowed:
        allowed_mask[left, right] = True
    if np.count_nonzero(block[:, :, ~support]) != 0:
        raise ValueError("signed density has nonzero values outside no-wrap support")
    if any(
        np.count_nonzero(block[left, right]) != 0 and not allowed_mask[left, right]
        for left in range(2)
        for right in range(2)
    ):
        raise ValueError("signed density violates the selected valley-charge block")

    result = np.zeros_like(block)
    dx, dy = displacement
    if valley_charge == 0:
        displacement_kernel = fft_plan.kernel_by_signed_displacement[
            -dy + size - 1, -dx + size - 1
        ]
        charge = 0.0 + 0.0j
        for flavor in range(2):
            source_factor = np.einsum(
                "cp,cp->p",
                spinors[flavor][:, bases].conj(),
                spinors[flavor][:, targets],
                optimize=True,
            )
            charge += np.sum(source_factor * block[flavor, flavor, bases])
        for flavor in range(2):
            target_factor = np.einsum(
                "cm,cm->m",
                spinors[flavor][:, targets].conj(),
                spinors[flavor][:, bases],
                optimize=True,
            )
            result[flavor, flavor, bases] += (
                target_factor * displacement_kernel * charge
            )

    shifted = np.zeros_like(spinors)
    shifted[:, :, bases] = spinors[:, :, targets]
    spinors_grid = spinors.reshape(2, 6, size, size)
    shifted_grid = shifted.reshape(2, 6, size, size)
    for left, right in allowed:
        values = block[left, right].reshape(size, size)
        if np.count_nonzero(values) == 0:
            continue
        sources = (
            shifted_grid[left][:, None, :, :]
            * spinors_grid[right][None, :, :, :].conj()
            * values[None, None, :, :]
        )
        padded = np.zeros(
            (6, 6, fft_plan.padding_size, fft_plan.padding_size),
            dtype=np.complex128,
        )
        padded[:, :, :size, :size] = sources
        transformed = _FFT2(
            padded, axes=(-2, -1), workers=fft_plan.fft_workers
        )
        convolution = _IFFT2(
            fft_plan.kernel_fft[None, None, :, :] * transformed,
            axes=(-2, -1),
            workers=fft_plan.fft_workers,
        )[:, :, :size, :size]
        result[left, right].reshape(size, size)[:] -= np.einsum(
            "cxy,exy,cexy->xy",
            shifted_grid[left].conj(),
            spinors_grid[right],
            convolution,
            optimize=True,
        )
    result /= area
    result[:, :, ~support] = 0.0
    return _readonly_complex(result, result.shape, "signed scalar response")


def vituri2024_exact_integer_paired_interaction_trace(
    selected_spinors: Array,
    fft_plan: Vituri2024SquareCartesianFFTPlan,
    area_angstrom_squared: float,
    displacement: tuple[int, int],
    valley_charge: int,
    positive_block: Array,
    negative_block: Array,
) -> float:
    """Return ``Tr(W Sigma[W])`` for explicit Hermitian ``{d,-d}`` blocks."""

    dx, dy = _strict_displacement(displacement)
    if dx == 0 and dy == 0 and valley_charge == 0:
        raise ValueError("self-conjugate zero sector is not a two-sign paired orbit")
    positive = _readonly_complex(
        positive_block, (2, 2, fft_plan.nk), "positive signed block"
    )
    negative = _readonly_complex(
        negative_block, (2, 2, fft_plan.nk), "negative signed block"
    )
    positive_bases, positive_targets = _support(fft_plan, (dx, dy))
    negative_bases, negative_targets = _support(fft_plan, (-dx, -dy))
    negative_lookup = {int(base): offset for offset, base in enumerate(negative_bases)}
    allowed = _allowed_blocks(valley_charge)
    for left, right in allowed:
        for base, target in zip(positive_bases, positive_targets, strict=True):
            offset = negative_lookup.get(int(target))
            if offset is None or int(negative_targets[offset]) != int(base):
                raise RuntimeError("paired signed support is not conjugate")
            if negative[right, left, target] != positive[left, right, base].conjugate():
                raise ValueError("signed blocks do not form an exact Hermitian pair")
    positive_action = vituri2024_exact_integer_signed_scalar_fft_action(
        selected_spinors,
        fft_plan,
        area_angstrom_squared,
        (dx, dy),
        valley_charge,
        positive,
    )
    negative_action = vituri2024_exact_integer_signed_scalar_fft_action(
        selected_spinors,
        fft_plan,
        area_angstrom_squared,
        (-dx, -dy),
        -valley_charge,
        negative,
    )
    value = np.vdot(positive, positive_action) + np.vdot(negative, negative_action)
    scale = max(1.0, abs(value))
    if abs(value.imag) > 5.0e-11 * scale:
        raise ValueError("paired exact-integer interaction trace is not real")
    return float(value.real)


def _current_implementation_fingerprint() -> str:
    payload = {
        "api_version": VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION,
        "authority": VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY,
        "sources": tuple(
            (item.__name__, sha256(inspect.getsource(item).encode()).hexdigest())
            for item in (
                vituri2024_exact_integer_signed_scalar_fft_action,
                vituri2024_exact_integer_paired_interaction_trace,
                _support,
                _allowed_blocks,
                _strict_displacement,
                _readonly_complex,
                _validate_import_bindings,
                _callable_dependency_fingerprint,
            )
        ),
        "fft2_binding": _callable_dependency_fingerprint(_FFT2),
        "ifft2_binding": _callable_dependency_fingerprint(_IFFT2),
        "numpy_version": np.__version__,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_IMPORT_IMPLEMENTATION_FINGERPRINT = _current_implementation_fingerprint()


def vituri2024_exact_integer_signed_scalar_implementation_fingerprint() -> str:
    _validate_import_bindings()
    current = _current_implementation_fingerprint()
    if current != _IMPORT_IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("signed scalar implementation source closure drifted")
    return current


__all__ = [
    "VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION",
    "VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY",
    "vituri2024_exact_integer_paired_interaction_trace",
    "vituri2024_exact_integer_signed_scalar_fft_action",
    "vituri2024_exact_integer_signed_scalar_implementation_fingerprint",
]
