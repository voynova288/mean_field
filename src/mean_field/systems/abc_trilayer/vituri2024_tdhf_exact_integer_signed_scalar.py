"""Scalable signed-channel exact-integer Vituri scalar-action candidate.

This implementation is intentionally independent of the orbital-Hessian
response module.  It consumes only selected-spin source-gauge spinors, a
validated no-wrap FFT plan, an integer displacement/valley charge, and one
signed density block.  It carries no scalar-Hessian, reciprocity, eigensolver,
stability, production, or paper authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from hashlib import sha256
import inspect
import json
import math
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np
from scipy.fft import fft2 as _FFT2, ifft2 as _IFFT2
from scipy.linalg import expm as _EXPM

from .vituri2024_hf_fft import Vituri2024SquareCartesianFFTPlan
from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)

Array = np.ndarray
_IMPORT_FFT2 = _FFT2
_IMPORT_IFFT2 = _IFFT2
_IMPORT_EXPM = _EXPM
_IMPORT_FFT_PLAN_TYPE = Vituri2024SquareCartesianFFTPlan

VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_selected_spin_signed_scalar_fft.v1"
)
VITURI2024_EXACT_INTEGER_SOURCE_FOCK_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_k_diagonal_source_fock_fft.v1"
)
VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_selected_spin_unitary_relative_energy.v1"
)
VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_API_VERSION: Final[str] = (
    "vituri2024_exact_integer_selected_spin_orbital_scalar_curvature.v1"
)
VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY: Final[str] = (
    "independent_candidate_signed_channel_action_not_scalar_hessian_reciprocity_"
    "eigensolver_stability_production_or_paper_authority"
)
VITURI2024_EXACT_INTEGER_SOURCE_FOCK_AUTHORITY: Final[str] = (
    "independent_candidate_k_diagonal_exact_integer_fock_evaluation_without_"
    "source_comparison_stationarity_scalar_hessian_or_production_authority"
)
VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_AUTHORITY: Final[str] = (
    "candidate_exact_unitary_relative_energy_without_external_source_fock_closure_"
    "curvature_reciprocity_stability_or_production_authority"
)
VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY: Final[str] = (
    "bound_candidate_analytic_orbital_scalar_curvature_without_source_functional_"
    "fock_closure_full_exact_unitary_reciprocity_stability_or_production_authority"
)


_CURVATURE_TOKEN = object()
_UNITARY_ENERGY_TOKEN = object()


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    return sha256(
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    ).hexdigest()


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
    if (
        _FFT2 is not _IMPORT_FFT2
        or _IFFT2 is not _IMPORT_IFFT2
        or _EXPM is not _IMPORT_EXPM
        or Vituri2024SquareCartesianFFTPlan is not _IMPORT_FFT_PLAN_TYPE
    ):
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


def vituri2024_exact_integer_source_fock_fft(
 active_band_states: Array,
 fft_plan: Vituri2024SquareCartesianFFTPlan,
 area_angstrom_squared: Real,
 h0_conventional: Array,
 density_conventional: Array,
 normal_order_reference_conventional: Array,
) -> Array:
 """Evaluate a four-flavor k-diagonal exact-integer ``F[P]`` independently."""

 _validate_import_bindings()
 implementation = _current_implementation_fingerprint()
 if implementation != _IMPORT_IMPLEMENTATION_FINGERPRINT:
  raise RuntimeError("signed scalar implementation source drifted")
 if type(fft_plan) is not Vituri2024SquareCartesianFFTPlan:
  raise TypeError("source Fock requires the exact FFT plan type")
 fft_plan.validate_live_state()
 if isinstance(area_angstrom_squared, (bool, np.bool_)) or not isinstance(
  area_angstrom_squared, Real
 ):
  raise TypeError("source Fock area must be a strict real scalar")
 area = float(area_angstrom_squared)
 if not math.isfinite(area) or area <= 0.0:
  raise ValueError("source Fock area must be positive")
 nk = fft_plan.nk
 size = fft_plan.mesh_size
 valley_spinors = _readonly_complex(
  active_band_states, (2, 6, nk), "active-band valley spinors"
 )
 valley_to_index = {
  valley: index for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
 }
 spinors = _readonly_complex(
  np.stack(
   tuple(valley_spinors[valley_to_index[valley]] for valley, _spin in INTERNAL_FLAVOR_ORDER)
  ),
  (4, 6, nk),
  "internally ordered flavor spinors",
 )
 h0 = _readonly_complex(h0_conventional, (4, 4, nk), "source h0")
 density = _readonly_complex(
  density_conventional, (4, 4, nk), "source conventional density"
 )
 reference = _readonly_complex(
  normal_order_reference_conventional,
  (4, 4, nk),
  "source normal-order reference",
 )
 for value, label in ((h0, "source h0"), (density, "source density"), (reference, "source reference")):
  scale = max(1.0, float(np.max(np.abs(value))))
  if float(np.max(np.abs(value - value.swapaxes(0, 1).conj()))) > (
   64.0 * np.finfo(np.float64).eps * scale
  ):
   raise ValueError(f"{label} must be k-local Hermitian")
 norms = np.sum(np.abs(spinors) ** 2, axis=1)
 if float(np.max(np.abs(norms - 1.0))) > 5.0e-12:
  raise ValueError("source Fock flavor spinors are not normalized")
 signed_kernel = fft_plan.kernel_by_signed_displacement
 kernel_scale = max(1.0, float(np.max(np.abs(signed_kernel))))
 kernel_tolerance = 64.0 * np.finfo(np.float64).eps * kernel_scale
 if (
  float(np.max(np.abs(signed_kernel.imag))) > kernel_tolerance
  or float(np.max(np.abs(signed_kernel - signed_kernel[::-1, ::-1])))
  > kernel_tolerance
 ):
  raise ValueError("source Fock requires the source-bound real-even kernel contract")
 block = density - reference
 sigma = np.zeros((4, 4, nk), dtype=np.complex128)
 kernel_zero = signed_kernel[size - 1, size - 1]
 charge = 0.0 + 0.0j
 for flavor in range(4):
  charge += np.sum(norms[flavor] * block[flavor, flavor])
 # Apply the common Hartree charge only after its complete flavor sum is known.
 for flavor in range(4):
  sigma[flavor, flavor] += norms[flavor] * kernel_zero * charge
 spinors_grid = spinors.reshape(4, 6, size, size)
 for left in range(4):
  for right in range(4):
   values = block[left, right].reshape(size, size)
   if np.count_nonzero(values) == 0:
    continue
   sources = (
    spinors_grid[left][:, None, :, :]
    * spinors_grid[right][None, :, :, :].conj()
    * values[None, None, :, :]
   )
   padded = np.zeros(
    (6, 6, fft_plan.padding_size, fft_plan.padding_size),
    dtype=np.complex128,
   )
   padded[:, :, :size, :size] = sources
   transformed = _FFT2(padded, axes=(-2, -1), workers=fft_plan.fft_workers)
   convolution = _IFFT2(
    fft_plan.kernel_fft[None, None, :, :] * transformed,
    axes=(-2, -1),
    workers=fft_plan.fft_workers,
   )[:, :, :size, :size]
   sigma[left, right] -= np.einsum(
    "cxy,exy,cexy->xy",
    spinors_grid[left].conj(),
    spinors_grid[right],
    convolution,
    optimize=True,
   ).reshape(-1)
 result = h0 + sigma / area
 scale = max(1.0, float(np.max(np.abs(result))))
 if float(np.max(np.abs(result - result.swapaxes(0, 1).conj()))) > (
  5.0e-11 * scale
 ):
  raise ValueError("independent exact-integer source Fock is not Hermitian")
 return _readonly_complex(result, result.shape, "independent source Fock")


def _readonly_exact_array(
 value: object,
 dtype: np.dtype,
 shape: tuple[int, ...],
 label: str,
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


@dataclass(frozen=True, slots=True)
class Vituri2024ExactIntegerUnitaryRelativeEnergyReceipt:
 """Immutable candidate receipt for one exact-unitary relative energy."""

 _factory_token: InitVar[object]
 parameter: float
 transition_count: int
 active_orbital_count: int
 connected_component_count: int
 signed_block_count: int
 source_fock_sha256: str
 source_occupations_sha256: str
 transition_inventory_sha256: str
 amplitudes_sha256: str
 selected_spinors_sha256: str
 fft_plan_fingerprint: str
 area_angstrom_squared: float
 one_body_relative_energy_ev: float
 interaction_trace_ev: float
 relative_energy_ev: float
 maximum_unitarity_residual: float
 maximum_projector_residual: float
 maximum_hermiticity_residual: float
 maximum_trace_residual: float
 implementation_fingerprint: str
 api_version: str = field(
  default=VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_API_VERSION,
  init=False,
 )
 authority: str = field(
  default=VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_AUTHORITY,
  init=False,
 )
 exact_unitary_projector_path_evaluated: bool = field(default=True, init=False)
 raw_total_no_nk_normalization: bool = field(default=True, init=False)
 external_source_fock_closure_established: bool = field(default=False, init=False)
 scalar_curvature_established: bool = field(default=False, init=False)
 reciprocity_established: bool = field(default=False, init=False)
 production_ready: bool = field(default=False, init=False)
 fingerprint: str = field(init=False)

 def __post_init__(self, _factory_token: object) -> None:
  if _factory_token is not _UNITARY_ENERGY_TOKEN:
   raise TypeError("unitary relative-energy receipts are factory-only")
  for value in (
   self.parameter,
   self.area_angstrom_squared,
   self.one_body_relative_energy_ev,
   self.interaction_trace_ev,
   self.relative_energy_ev,
   self.maximum_unitarity_residual,
   self.maximum_projector_residual,
   self.maximum_hermiticity_residual,
   self.maximum_trace_residual,
  ):
   if type(value) is not float or not math.isfinite(value):
    raise ValueError("unitary relative-energy receipt has a nonfinite scalar")
  if self.area_angstrom_squared <= 0.0:
   raise ValueError("unitary relative-energy receipt area is invalid")
  if type(self.transition_count) is not int or self.transition_count <= 0:
   raise ValueError("unitary relative-energy transition count is invalid")
  for value in (
   self.active_orbital_count,
   self.connected_component_count,
   self.signed_block_count,
  ):
   if type(value) is not int or value < 0:
    raise ValueError("unitary relative-energy structural count is invalid")
  for name in (
   "source_fock_sha256",
   "source_occupations_sha256",
   "transition_inventory_sha256",
   "amplitudes_sha256",
   "selected_spinors_sha256",
   "fft_plan_fingerprint",
   "implementation_fingerprint",
  ):
   value = getattr(self, name)
   if (
    type(value) is not str
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
   ):
    raise ValueError("unitary relative-energy digest is invalid")
  if self.relative_energy_ev != (
   self.one_body_relative_energy_ev + 0.5 * self.interaction_trace_ev
  ):
   raise ValueError("unitary relative-energy decomposition drifted")
  if (
   self.api_version
   != VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_API_VERSION
   or self.authority
   != VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_AUTHORITY
   or self.exact_unitary_projector_path_evaluated is not True
   or self.raw_total_no_nk_normalization is not True
   or self.external_source_fock_closure_established is not False
   or self.scalar_curvature_established is not False
   or self.reciprocity_established is not False
   or self.production_ready is not False
  ):
   raise ValueError("unitary relative-energy authority drifted")
  payload = {
   name: getattr(self, name)
   for name in self.__dataclass_fields__
   if name not in {"_factory_token", "fingerprint"}
  }
  object.__setattr__(
   self,
   "fingerprint",
   sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
   ).hexdigest(),
  )


def vituri2024_exact_integer_unitary_relative_energy(
 source_fock_conventional: Array,
 source_occupations: Array,
 particle_valley_slots: Array,
 particle_k_indices: Array,
 hole_valley_slots: Array,
 hole_k_indices: Array,
 amplitudes: Array,
 parameter: Real,
 selected_spinors: Array,
 fft_plan: Vituri2024SquareCartesianFFTPlan,
 area_angstrom_squared: Real,
) -> Vituri2024ExactIntegerUnitaryRelativeEnergyReceipt:
 """Evaluate ``E[exp(tK)P0exp(-tK)]-E[P0]`` for a paired generator."""

 _validate_import_bindings()
 implementation = _current_implementation_fingerprint()
 if implementation != _IMPORT_IMPLEMENTATION_FINGERPRINT:
  raise RuntimeError("signed scalar implementation source drifted")
 if type(fft_plan) is not Vituri2024SquareCartesianFFTPlan:
  raise TypeError("unitary relative energy requires the exact FFT plan type")
 fft_plan.validate_live_state()
 for value, label in (
  (parameter, "unitary parameter"),
  (area_angstrom_squared, "unitary relative-energy area"),
 ):
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
   raise TypeError(f"{label} must be a strict real scalar")
 t = float(parameter)
 area = float(area_angstrom_squared)
 if not math.isfinite(t) or not math.isfinite(area) or area <= 0.0:
  raise ValueError("unitary parameter/area is invalid")
 nk = fft_plan.nk
 fock = _readonly_complex(
  source_fock_conventional, (2, 2, nk), "selected source Fock"
 )
 fock_residual = float(
  np.max(np.abs(fock - fock.transpose(1, 0, 2).conj()), initial=0.0)
 )
 if fock_residual > 5.0e-12 * max(
  1.0, float(np.max(np.abs(fock), initial=0.0))
 ):
  raise ValueError("selected source Fock must be k-local Hermitian")
 occupations = _readonly_exact_array(
  source_occupations, np.dtype(np.bool_), (2, nk), "source occupations"
 )
 if type(particle_valley_slots) is not np.ndarray or particle_valley_slots.ndim != 1:
  raise ValueError("particle valley slots must be an exact vector")
 count = int(particle_valley_slots.size)
 if count <= 0:
  raise ValueError("unitary relative energy requires transitions")
 particle_slots = _readonly_exact_array(
  particle_valley_slots, np.dtype(np.int64), (count,), "particle valley slots"
 )
 particle_k = _readonly_exact_array(
  particle_k_indices, np.dtype(np.int64), (count,), "particle k indices"
 )
 hole_slots = _readonly_exact_array(
  hole_valley_slots, np.dtype(np.int64), (count,), "hole valley slots"
 )
 hole_k = _readonly_exact_array(
  hole_k_indices, np.dtype(np.int64), (count,), "hole k indices"
 )
 values = _readonly_exact_array(
  amplitudes, np.dtype(np.complex128), (count,), "transition amplitudes"
 )
 spinors = _readonly_complex(selected_spinors, (2, 6, nk), "selected spinors")
 spinor_norm_residual = float(
  np.max(
   np.abs(np.sum(np.abs(spinors) ** 2, axis=1) - 1.0), initial=0.0
  )
 )
 if spinor_norm_residual > 5.0e-12:
  raise ValueError("selected spinors must be normalized")
 if (
  np.any(particle_slots < 0)
  or np.any(particle_slots >= 2)
  or np.any(hole_slots < 0)
  or np.any(hole_slots >= 2)
  or np.any(particle_k < 0)
  or np.any(particle_k >= nk)
  or np.any(hole_k < 0)
  or np.any(hole_k >= nk)
 ):
  raise ValueError("unitary transition lies outside selected source space")
 if np.any(occupations[particle_slots, particle_k]) or not np.all(
  occupations[hole_slots, hole_k]
 ):
  raise ValueError("unitary transition does not map occupied holes to virtual particles")
 inventory = np.stack((particle_slots, particle_k, hole_slots, hole_k), axis=1)
 if np.unique(inventory, axis=0).shape[0] != count:
  raise ValueError("duplicate particle-hole transition")
 dimension = 2 * nk
 adjacency: dict[int, set[int]] = {}
 edges: dict[tuple[int, int], complex] = {}
 for ps, pk, hs, hk, value in zip(
  particle_slots, particle_k, hole_slots, hole_k, values, strict=True
 ):
  particle = int(ps) * nk + int(pk)
  hole = int(hs) * nk + int(hk)
  if value == 0.0:
   continue
  edges[(particle, hole)] = complex(value)
  adjacency.setdefault(particle, set()).add(hole)
  adjacency.setdefault(hole, set()).add(particle)
 active = sorted(adjacency)
 components: list[tuple[int, ...]] = []
 unseen = set(active)
 while unseen:
  root = min(unseen)
  stack = [root]
  component: list[int] = []
  unseen.remove(root)
  while stack:
   current = stack.pop()
   component.append(current)
   for neighbor in sorted(adjacency[current], reverse=True):
    if neighbor in unseen:
     unseen.remove(neighbor)
     stack.append(neighbor)
  components.append(tuple(sorted(component)))
 orbital_to_component = {
  orbital: component_index
  for component_index, component in enumerate(components)
  for orbital in component
 }
 component_edges: list[list[tuple[int, int, complex]]] = [
  [] for _component in components
 ]
 for (particle, hole), value in edges.items():
  component_index = orbital_to_component[particle]
  if orbital_to_component[hole] != component_index:
   raise RuntimeError("unitary generator edge crosses connected components")
  component_edges[component_index].append((particle, hole, value))
 labels = fft_plan.integer_mesh_labels
 blocks: dict[tuple[int, int, int], Array] = {}
 one_body = 0.0 + 0.0j
 max_unitarity = 0.0
 max_projector = 0.0
 max_hermiticity = 0.0
 max_trace = 0.0
 occupation_flat = occupations.reshape(dimension)
 for component, local_edges in zip(components, component_edges, strict=True):
  local = {orbital: index for index, orbital in enumerate(component)}
  generator = np.zeros((len(component), len(component)), dtype=np.complex128)
  for particle, hole, value in local_edges:
   generator[local[particle], local[hole]] = value
   generator[local[hole], local[particle]] = -value.conjugate()
  unitary = _EXPM(t * generator)
  projector0 = np.diag(occupation_flat[np.asarray(component)].astype(np.complex128))
  projector = unitary @ projector0 @ unitary.conj().T
  difference = projector - projector0
  identity = np.eye(len(component), dtype=np.complex128)
  max_unitarity = max(
   max_unitarity, float(np.max(np.abs(unitary.conj().T @ unitary - identity)))
  )
  max_projector = max(
   max_projector, float(np.max(np.abs(projector @ projector - projector)))
  )
  max_hermiticity = max(
   max_hermiticity, float(np.max(np.abs(difference - difference.conj().T)))
  )
  max_trace = max(
   max_trace, float(abs(np.trace(projector) - np.trace(projector0)))
  )
  for local_row, global_row in enumerate(component):
   left, target_k = divmod(global_row, nk)
   for local_column, global_column in enumerate(component):
    value = difference[local_row, local_column]
    if value == 0.0:
     continue
    right, base_k = divmod(global_column, nk)
    dx, dy = (labels[target_k] - labels[base_k]).tolist()
    key = (int(dx), int(dy), 2 * (left - right))
    block = blocks.setdefault(key, np.zeros((2, 2, nk), dtype=np.complex128))
    if block[left, right, base_k] != 0.0:
     raise RuntimeError("unitary density block received duplicate matrix elements")
    block[left, right, base_k] = value
    if target_k == base_k:
     one_body += fock[right, left, base_k] * value
 geometry_tolerance = 5.0e-11
 if max(max_unitarity, max_projector, max_hermiticity, max_trace) > geometry_tolerance:
  raise ValueError("exact-unitary projector geometry failed")
 interaction = 0.0 + 0.0j
 for (dx, dy, charge), block in sorted(blocks.items()):
  action = vituri2024_exact_integer_signed_scalar_fft_action(
   spinors, fft_plan, area, (dx, dy), charge, block
  )
  interaction += np.vdot(block, action)
 scale = max(1.0, abs(one_body), abs(interaction))
 if (
  abs(one_body.imag) > 5.0e-11 * scale
  or abs(interaction.imag) > 5.0e-11 * scale
 ):
  raise ValueError("unitary relative-energy decomposition is not real")
 one_body_real = float(one_body.real)
 interaction_real = float(interaction.real)
 return Vituri2024ExactIntegerUnitaryRelativeEnergyReceipt(
  _factory_token=_UNITARY_ENERGY_TOKEN,
  parameter=t,
  transition_count=count,
  active_orbital_count=len(active),
  connected_component_count=len(components),
  signed_block_count=len(blocks),
  source_fock_sha256=_array_sha256(fock),
  source_occupations_sha256=_array_sha256(occupations),
  transition_inventory_sha256=_array_sha256(inventory),
  amplitudes_sha256=_array_sha256(values),
  selected_spinors_sha256=_array_sha256(spinors),
  fft_plan_fingerprint=fft_plan.fingerprint,
  area_angstrom_squared=area,
  one_body_relative_energy_ev=one_body_real,
  interaction_trace_ev=interaction_real,
  relative_energy_ev=float(one_body_real + 0.5 * interaction_real),
  maximum_unitarity_residual=max_unitarity,
  maximum_projector_residual=max_projector,
  maximum_hermiticity_residual=max_hermiticity,
  maximum_trace_residual=max_trace,
  implementation_fingerprint=implementation,
 )


@dataclass(frozen=True, slots=True)
class Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt:
 """Input- and implementation-bound candidate analytic ``E''(0)``."""

 _factory_token: InitVar[object]
 transition_count: int
 displacement: tuple[int, int]
 valley_charge: int
 area_angstrom_squared: float
 fft_plan_fingerprint: str
 selected_spinors_sha256: str
 positive_block_sha256: str
 negative_block_sha256: str
 source_fock_diagonal_sha256: str
 source_occupations_sha256: str
 transition_inventory_sha256: str
 amplitudes_sha256: str
 one_body_curvature_ev: float
 interaction_curvature_ev: float
 total_scalar_curvature_ev: float
 implementation_fingerprint: str
 api_version: str = field(
  default=VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_API_VERSION,
  init=False,
 )
 authority: str = field(
  default=VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY,
  init=False,
 )
 raw_total_no_nk_normalization: bool = field(default=True, init=False)
 source_functional_fock_closure_established: bool = field(default=False, init=False)
 full_exact_unitary_scalar_curvature_established: bool = field(
  default=False, init=False
 )
 reciprocity_established: bool = field(default=False, init=False)
 production_ready: bool = field(default=False, init=False)
 fingerprint: str = field(init=False)

 def __post_init__(self, _factory_token: object) -> None:
  if _factory_token is not _CURVATURE_TOKEN:
   raise TypeError("orbital scalar-curvature receipts are factory-only")
  if type(self.transition_count) is not int or self.transition_count <= 0:
   raise ValueError("orbital scalar curvature requires transitions")
  _strict_displacement(self.displacement)
  _allowed_blocks(self.valley_charge)
  if (
   type(self.area_angstrom_squared) is not float
   or not math.isfinite(self.area_angstrom_squared)
   or self.area_angstrom_squared <= 0.0
  ):
   raise ValueError("receipt area is invalid")
  digest_names = (
   "fft_plan_fingerprint",
   "selected_spinors_sha256",
   "positive_block_sha256",
   "negative_block_sha256",
   "source_fock_diagonal_sha256",
   "source_occupations_sha256",
   "transition_inventory_sha256",
   "amplitudes_sha256",
   "implementation_fingerprint",
  )
  if any(
   type(getattr(self, name)) is not str
   or len(getattr(self, name)) != 64
   or any(
    character not in "0123456789abcdef"
    for character in getattr(self, name)
   )
   for name in digest_names
  ):
   raise ValueError("orbital scalar-curvature digest is invalid")
  for value in (
   self.one_body_curvature_ev,
   self.interaction_curvature_ev,
   self.total_scalar_curvature_ev,
  ):
   if type(value) is not float or not math.isfinite(value):
    raise ValueError("orbital scalar curvature must be finite float")
  if self.total_scalar_curvature_ev != (
   self.one_body_curvature_ev + self.interaction_curvature_ev
  ):
   raise ValueError("orbital scalar-curvature decomposition drifted")
  if (
   self.api_version
   != VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_API_VERSION
   or self.authority
   != VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY
   or self.raw_total_no_nk_normalization is not True
   or self.source_functional_fock_closure_established is not False
   or self.full_exact_unitary_scalar_curvature_established is not False
   or self.reciprocity_established is not False
   or self.production_ready is not False
  ):
   raise ValueError("orbital scalar-curvature authority drifted")
  payload = {
   name: getattr(self, name)
   for name in self.__dataclass_fields__
   if name not in {"_factory_token", "fingerprint"}
  }
  object.__setattr__(
   self,
   "fingerprint",
   sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
   ).hexdigest(),
  )


def vituri2024_exact_integer_orbital_scalar_curvature(
 source_fock_diagonal_ev: Array,
 source_occupations: Array,
 particle_valley_slots: Array,
 particle_k_indices: Array,
 hole_valley_slots: Array,
 hole_k_indices: Array,
 amplitudes: Array,
 selected_spinors: Array,
 fft_plan: Vituri2024SquareCartesianFFTPlan,
 area_angstrom_squared: Real,
 displacement: tuple[int, int],
 valley_charge: int,
 positive_block: Array,
 negative_block: Array,
) -> Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt:
 """Bind ``Tr(F[K,[K,P]]) + Tr(W Sigma[W])`` for one paired tangent.

 For ``K_ph=X`` and ``P_hh=1, P_pp=0``, the first term is
 ``2 sum_ph (F_p-F_h)|X_ph|^2``. The signed blocks are reconstructed
 independently from the transition coordinates and must match exactly before
 their interaction trace is evaluated.
 """

 _validate_import_bindings()
 implementation = _current_implementation_fingerprint()
 if implementation != _IMPORT_IMPLEMENTATION_FINGERPRINT:
  raise RuntimeError("signed scalar implementation source drifted")
 if type(fft_plan) is not Vituri2024SquareCartesianFFTPlan:
  raise TypeError("orbital scalar curvature requires the exact FFT plan type")
 fft_plan.validate_live_state()
 displacement = _strict_displacement(displacement)
 _allowed_blocks(valley_charge)
 if isinstance(area_angstrom_squared, (bool, np.bool_)) or not isinstance(
  area_angstrom_squared, Real
 ):
  raise TypeError("orbital scalar-curvature area must be a strict real scalar")
 area = float(area_angstrom_squared)
 if not math.isfinite(area) or area <= 0.0:
  raise ValueError("orbital scalar-curvature area must be positive")
 nk = fft_plan.nk
 fock = _readonly_exact_array(
  source_fock_diagonal_ev,
  np.dtype(np.float64),
  (2, nk),
  "source Fock diagonal",
 )
 occupations = _readonly_exact_array(
  source_occupations,
  np.dtype(np.bool_),
  (2, nk),
  "source occupations",
 )
 if type(particle_valley_slots) is not np.ndarray or particle_valley_slots.ndim != 1:
  raise ValueError("particle valley slots must be an exact vector")
 count = int(particle_valley_slots.size)
 if count <= 0:
  raise ValueError("orbital scalar curvature requires transitions")
 particle_slots = _readonly_exact_array(
  particle_valley_slots, np.dtype(np.int64), (count,), "particle valley slots"
 )
 particle_k = _readonly_exact_array(
  particle_k_indices, np.dtype(np.int64), (count,), "particle k indices"
 )
 hole_slots = _readonly_exact_array(
  hole_valley_slots, np.dtype(np.int64), (count,), "hole valley slots"
 )
 hole_k = _readonly_exact_array(
  hole_k_indices, np.dtype(np.int64), (count,), "hole k indices"
 )
 values = _readonly_exact_array(
  amplitudes, np.dtype(np.complex128), (count,), "transition amplitudes"
 )
 spinors = _readonly_complex(selected_spinors, (2, 6, nk), "selected spinors")
 positive = _readonly_complex(positive_block, (2, 2, nk), "positive signed block")
 negative = _readonly_complex(negative_block, (2, 2, nk), "negative signed block")
 if (
  np.any(particle_slots < 0)
  or np.any(particle_slots >= 2)
  or np.any(hole_slots < 0)
  or np.any(hole_slots >= 2)
  or np.any(particle_k < 0)
  or np.any(particle_k >= nk)
  or np.any(hole_k < 0)
  or np.any(hole_k >= nk)
 ):
  raise ValueError("transition coordinate lies outside selected source space")
 if np.any(occupations[particle_slots, particle_k]) or not np.all(
  occupations[hole_slots, hole_k]
 ):
  raise ValueError("transition does not map occupied holes to virtual particles")
 inventory = np.stack((particle_slots, particle_k, hole_slots, hole_k), axis=1)
 if np.unique(inventory, axis=0).shape[0] != count:
  raise ValueError("duplicate particle-hole transition")
 labels = fft_plan.integer_mesh_labels
 expected_positive = np.zeros((2, 2, nk), dtype=np.complex128)
 expected_negative = np.zeros_like(expected_positive)
 opposite = (-displacement[0], -displacement[1])
 for ps, pk, hs, hk, value in zip(
  particle_slots, particle_k, hole_slots, hole_k, values, strict=True
 ):
  signed_displacement = tuple((labels[pk] - labels[hk]).tolist())
  signed_charge = 2 * (int(ps) - int(hs))
  if signed_displacement == displacement and signed_charge == valley_charge:
   expected_positive[ps, hs, hk] += value
   expected_negative[hs, ps, pk] += value.conjugate()
  elif signed_displacement == opposite and signed_charge == -valley_charge:
   expected_negative[ps, hs, hk] += value
   expected_positive[hs, ps, pk] += value.conjugate()
  else:
   raise ValueError("transition lies outside the declared paired sector")
 if not np.array_equal(positive, expected_positive) or not np.array_equal(
  negative, expected_negative
 ):
  raise ValueError("signed blocks do not match the explicit transition tangent")
 gaps = fock[particle_slots, particle_k] - fock[hole_slots, hole_k]
 one_body = float(2.0 * np.sum(gaps * np.abs(values) ** 2))
 interaction = vituri2024_exact_integer_paired_interaction_trace(
  spinors,
  fft_plan,
  area,
  displacement,
  valley_charge,
  positive,
  negative,
 )
 return Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt(
  _factory_token=_CURVATURE_TOKEN,
  transition_count=count,
  displacement=displacement,
  valley_charge=valley_charge,
  area_angstrom_squared=area,
  fft_plan_fingerprint=fft_plan.fingerprint,
  selected_spinors_sha256=_array_sha256(spinors),
  positive_block_sha256=_array_sha256(positive),
  negative_block_sha256=_array_sha256(negative),
  source_fock_diagonal_sha256=_array_sha256(fock),
  source_occupations_sha256=_array_sha256(occupations),
  transition_inventory_sha256=_array_sha256(inventory),
  amplitudes_sha256=_array_sha256(values),
  one_body_curvature_ev=one_body,
  interaction_curvature_ev=interaction,
  total_scalar_curvature_ev=float(one_body + interaction),
  implementation_fingerprint=implementation,
 )

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
        "source_fock_api_version": VITURI2024_EXACT_INTEGER_SOURCE_FOCK_API_VERSION,
        "source_fock_authority": VITURI2024_EXACT_INTEGER_SOURCE_FOCK_AUTHORITY,
        "active_band_states_valley_order": ACTIVE_BAND_STATES_VALLEY_ORDER,
        "internal_flavor_order": INTERNAL_FLAVOR_ORDER,
        "unitary_relative_energy_api_version": (
            VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_API_VERSION
        ),
        "unitary_relative_energy_authority": (
            VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_AUTHORITY
        ),
        "curvature_api_version": (
            VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_API_VERSION
        ),
        "curvature_authority": VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY,
        "sources": tuple(
            (item.__name__, sha256(inspect.getsource(item).encode()).hexdigest())
            for item in (
                vituri2024_exact_integer_signed_scalar_fft_action,
                vituri2024_exact_integer_source_fock_fft,
                vituri2024_exact_integer_paired_interaction_trace,
                vituri2024_exact_integer_unitary_relative_energy,
                Vituri2024ExactIntegerUnitaryRelativeEnergyReceipt,
                vituri2024_exact_integer_orbital_scalar_curvature,
                Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt,
                _array_sha256,
                _support,
                _allowed_blocks,
                _strict_displacement,
                _readonly_complex,
                _readonly_exact_array,
                _validate_import_bindings,
                _callable_dependency_fingerprint,
            )
        ),
        "fft_plan_type_binding": _callable_dependency_fingerprint(
            Vituri2024SquareCartesianFFTPlan
        ),
        "expm_binding": _callable_dependency_fingerprint(_EXPM),
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
    "VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_API_VERSION",
    "VITURI2024_EXACT_INTEGER_ORBITAL_CURVATURE_AUTHORITY",
    "VITURI2024_EXACT_INTEGER_SOURCE_FOCK_API_VERSION",
    "VITURI2024_EXACT_INTEGER_SOURCE_FOCK_AUTHORITY",
    "VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_API_VERSION",
    "VITURI2024_EXACT_INTEGER_UNITARY_RELATIVE_ENERGY_AUTHORITY",
    "VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_API_VERSION",
    "VITURI2024_EXACT_INTEGER_SIGNED_SCALAR_AUTHORITY",
    "Vituri2024ExactIntegerOrbitalScalarCurvatureReceipt",
    "Vituri2024ExactIntegerUnitaryRelativeEnergyReceipt",
    "vituri2024_exact_integer_orbital_scalar_curvature",
    "vituri2024_exact_integer_paired_interaction_trace",
    "vituri2024_exact_integer_source_fock_fft",
    "vituri2024_exact_integer_unitary_relative_energy",
    "vituri2024_exact_integer_signed_scalar_fft_action",
    "vituri2024_exact_integer_signed_scalar_implementation_fingerprint",
]
