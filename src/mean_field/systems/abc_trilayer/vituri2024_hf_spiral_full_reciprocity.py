"""Candidate whole-inventory reciprocity comparison for the Vituri paired Hessian.

The structural comparison in this module carries no reciprocity, scalar-Hessian,
or stability authority.  It binds the exact-integer, finite-square, no-wrap selected-spin
interaction action to the identities

``L_H = c |g*><g*|`` and ``L_F = -sum_s M_s^dagger C_K M_s``.

A real-even source kernel makes both terms self-adjoint.  Exact signed
conjugation and adjoint transition injection then make every paired-sector
real Hessian reciprocal.  No random bilinear probe enters the decision.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np

from . import vituri2024_hf_spiral_full_hessian as _hessian_module
from . import vituri2024_hf_spiral_full_response as _response_module
from . import vituri2024_hf_spiral_full_stability as _stability_module
from ...core.hf import zero_temperature_sector_stability as _core_reciprocity_module
from .vituri2024_hf_fft import Vituri2024SquareCartesianFFTPlan
from .vituri2024_hf_spiral_full_hessian import (
 Vituri2024HFSpiralFullHessianContext,
)
from .vituri2024_hf_spiral_full_response import (
 Vituri2024HFSpiralSignedDisplacementResponse,
)
from .vituri2024_hf_spiral_full_stability import (
 Vituri2024HFSpiralFullSectorInventory,
 Vituri2024HFSpiralFullSectorKey,
)
from .vituri2024_tdhf_exact_integer_signed_scalar import (
 vituri2024_exact_integer_signed_scalar_implementation_fingerprint,
)

Array = np.ndarray

VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION: Final[str] = (
 "vituri2024_selected_spin_exact_integer_whole_inventory_reciprocity.v1"
)
VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM: Final[str] = (
 "real_even_CK_direct_rank_one_exchange_Mdagger_CK_M_signed_conjugation_"
 "adjoint_transition_restriction_implies_real_bilinear_reciprocity"
)
VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE: Final[str] = (
 "selected_spin_fixed_rank_exact_integer_no_wrap_whole_transition_inventory"
)
VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY: Final[str] = (
 "candidate_whole_inventory_reciprocity_structural_comparison_without_"
 "reciprocity_scalar_hessian_eigensolver_stability_or_production_authority"
)

def _module_own_callables(module: object) -> tuple[tuple[str, object], ...]:
 module_name = getattr(module, "__name__", "")
 return tuple(
  (name, value)
  for name, value in sorted(vars(module).items())
  if callable(value) and getattr(value, "__module__", "") == module_name
 )


def _class_descriptors(value: type) -> tuple[tuple[str, tuple[object, ...]], ...]:
 records: list[tuple[str, tuple[object, ...]]] = []
 for name, descriptor in sorted(vars(value).items()):
  if isinstance(descriptor, property):
   records.append((name, (descriptor.fget, descriptor.fset, descriptor.fdel)))
  elif isinstance(descriptor, (staticmethod, classmethod)):
   records.append((name, (descriptor.__func__,)))
  elif callable(descriptor):
   records.append((name, (descriptor,)))
 return tuple(records)


def _module_class_bindings(module: object) -> tuple[tuple[type, tuple[tuple[str, tuple[object, ...]], ...]], ...]:
 module_name = getattr(module, "__name__", "")
 return tuple(
  (value, _class_descriptors(value))
  for _name, value in sorted(vars(module).items())
  if isinstance(value, type) and getattr(value, "__module__", "") == module_name
 )


_RUNTIME_OWNER_MODULES = (
 _response_module,
 _hessian_module,
 _stability_module,
 _core_reciprocity_module,
)
_IMPORT_MODULE_CALLABLE_BINDINGS = tuple(
 (module, _module_own_callables(module)) for module in _RUNTIME_OWNER_MODULES
)
_IMPORT_MODULE_CLASS_BINDINGS = tuple(
 (module, _module_class_bindings(module)) for module in _RUNTIME_OWNER_MODULES
)

_TOKEN = object()
_IMPORT_DIRECT_PRIMITIVE = _response_module._direct_rank_one_numerator
_IMPORT_EXCHANGE_PRIMITIVE = (
 _response_module._accumulate_exchange_mdagger_ck_m_numerator
)
_IMPORT_DIRECT_WRAPPER = (
 _response_module.Vituri2024HFSpiralSignedDisplacementResponse._direct_response
)
_IMPORT_APPLY_FFT = (
 _response_module.Vituri2024HFSpiralSignedDisplacementResponse._apply_fft_validated
)
_IMPORT_ACTION_CALL = (
 _response_module.Vituri2024HFSpiralValidatedSignedDisplacementFFTAction.__call__
)
_IMPORT_VALIDATE_BLOCK = (
 _response_module.Vituri2024HFSpiralSignedDisplacementResponse._validate_signed_block
)
_IMPORT_MAKE_ACTION = (
 _response_module.Vituri2024HFSpiralSignedDisplacementResponse._make_validated_fft_action_unchecked
)
_IMPORT_FACTORY_PREPARE = (
 _response_module.Vituri2024HFSpiralValidatedResponseActionFactory.prepare_fft_action
)
_IMPORT_FFT2 = _response_module._FFT2
_IMPORT_IFFT2 = _response_module._IFFT2
_IMPORT_PAIRED_CALLBACK = _hessian_module._VituriPairedInteractionCallback.__call__
_IMPORT_BUILD_EMBEDDING = (
 _hessian_module.Vituri2024HFSpiralFullHessianContext._build_embedding
)
_IMPORT_BUILD_ORBIT = (
 _hessian_module.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian
)
_IMPORT_UNPACK_REAL = _core_reciprocity_module.PairedOrbitalTransitionFrame.unpack_real
_IMPORT_PACK_COMPLEX = _core_reciprocity_module.PairedOrbitalTransitionFrame.pack_complex
_IMPORT_COMPLEX_ACTION = (
 _core_reciprocity_module.PairedSectorOrbitalHessian.complex_gradient_jacobian_action
)
_IMPORT_MATVEC = _core_reciprocity_module.PairedSectorOrbitalHessian.matvec
_IMPORT_KEY_CONJUGATE = _stability_module.Vituri2024HFSpiralFullSectorKey.conjugate.fget
_IMPORT_SECTOR_DIMENSION = (
 _stability_module.Vituri2024HFSpiralFullSectorInventory.sector_complex_dimension
)
_IMPORT_ITER_SECTORS = (
 _stability_module.Vituri2024HFSpiralFullSectorInventory.iter_sector_keys
)
_IMPORT_ITER_ORBITS = (
 _stability_module.Vituri2024HFSpiralFullSectorInventory.iter_conjugate_orbits
)


def _array_sha256(value: Array) -> str:
 array = np.ascontiguousarray(value)
 return sha256(
  str(array.dtype).encode()
  + b"\0"
  + json.dumps(array.shape).encode()
  + b"\0"
  + array.view(np.uint8).tobytes()
 ).hexdigest()


def _fingerprint(payload: object) -> str:
 return sha256(
  json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
 ).hexdigest()


def _source_sha256(value: object) -> str:
 return sha256(inspect.getsource(value).encode()).hexdigest()


def _module_file_sha256(module: object) -> str:
 source_file = inspect.getsourcefile(module)
 if source_file is None:
  raise ValueError("reciprocity dependency has no source file")
 return sha256(Path(source_file).resolve().read_bytes()).hexdigest()


def _validate_owner_bindings() -> None:
 for module, expected in _IMPORT_MODULE_CALLABLE_BINDINGS:
  current = _module_own_callables(module)
  if len(current) != len(expected) or any(
   current_name != expected_name or current_value is not expected_value
   for (current_name, current_value), (expected_name, expected_value) in zip(
    current, expected, strict=True
   )
  ):
   raise RuntimeError("whole-inventory reciprocity owner callable drifted")
 for module, expected_classes in _IMPORT_MODULE_CLASS_BINDINGS:
  current_classes = _module_class_bindings(module)
  if len(current_classes) != len(expected_classes):
   raise RuntimeError("whole-inventory reciprocity owner class inventory drifted")
  for (current_class, current_descriptors), (
   expected_class,
   expected_descriptors,
  ) in zip(current_classes, expected_classes, strict=True):
   if current_class is not expected_class or len(current_descriptors) != len(
    expected_descriptors
   ):
    raise RuntimeError("whole-inventory reciprocity owner class drifted")
   for (current_name, current_values), (
    expected_name,
    expected_values,
   ) in zip(current_descriptors, expected_descriptors, strict=True):
    if current_name != expected_name or len(current_values) != len(expected_values):
     raise RuntimeError("whole-inventory reciprocity class descriptor drifted")
    if any(
     current_value is not expected_value
     for current_value, expected_value in zip(
      current_values, expected_values, strict=True
     )
    ):
     raise RuntimeError("whole-inventory reciprocity class callable drifted")
 bindings = (
  (_response_module._direct_rank_one_numerator, _IMPORT_DIRECT_PRIMITIVE),
  (
   _response_module._accumulate_exchange_mdagger_ck_m_numerator,
   _IMPORT_EXCHANGE_PRIMITIVE,
  ),
  (
   _response_module.Vituri2024HFSpiralSignedDisplacementResponse._direct_response,
   _IMPORT_DIRECT_WRAPPER,
  ),
  (
   _response_module.Vituri2024HFSpiralSignedDisplacementResponse._apply_fft_validated,
   _IMPORT_APPLY_FFT,
  ),
  (
   _response_module.Vituri2024HFSpiralValidatedSignedDisplacementFFTAction.__call__,
   _IMPORT_ACTION_CALL,
  ),
  (
   _response_module.Vituri2024HFSpiralSignedDisplacementResponse._validate_signed_block,
   _IMPORT_VALIDATE_BLOCK,
  ),
  (
   _response_module.Vituri2024HFSpiralSignedDisplacementResponse._make_validated_fft_action_unchecked,
   _IMPORT_MAKE_ACTION,
  ),
  (
   _response_module.Vituri2024HFSpiralValidatedResponseActionFactory.prepare_fft_action,
   _IMPORT_FACTORY_PREPARE,
  ),
  (_response_module._FFT2, _IMPORT_FFT2),
  (_response_module._IFFT2, _IMPORT_IFFT2),
  (_response_module._FFT2, _response_module._IMPORT_FFT2),
  (_response_module._IFFT2, _response_module._IMPORT_IFFT2),
  (_hessian_module._VituriPairedInteractionCallback.__call__, _IMPORT_PAIRED_CALLBACK),
  (
   _hessian_module.Vituri2024HFSpiralFullHessianContext._build_embedding,
   _IMPORT_BUILD_EMBEDDING,
  ),
  (
   _hessian_module.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian,
   _IMPORT_BUILD_ORBIT,
  ),
  (
   _core_reciprocity_module.PairedOrbitalTransitionFrame.unpack_real,
   _IMPORT_UNPACK_REAL,
  ),
  (
   _core_reciprocity_module.PairedOrbitalTransitionFrame.pack_complex,
   _IMPORT_PACK_COMPLEX,
  ),
  (
   _core_reciprocity_module.PairedSectorOrbitalHessian.complex_gradient_jacobian_action,
   _IMPORT_COMPLEX_ACTION,
  ),
  (_core_reciprocity_module.PairedSectorOrbitalHessian.matvec, _IMPORT_MATVEC),
  (
   _hessian_module.OrbitalTransitionLane,
   _core_reciprocity_module.OrbitalTransitionLane,
  ),
  (
   _hessian_module.PairedOrbitalTransitionFrame,
   _core_reciprocity_module.PairedOrbitalTransitionFrame,
  ),
  (
   _hessian_module.PairedSectorOrbitalHessian,
   _core_reciprocity_module.PairedSectorOrbitalHessian,
  ),
  (
   _stability_module.Vituri2024HFSpiralFullSectorKey.conjugate.fget,
   _IMPORT_KEY_CONJUGATE,
  ),
  (
   _stability_module.Vituri2024HFSpiralFullSectorInventory.sector_complex_dimension,
   _IMPORT_SECTOR_DIMENSION,
  ),
  (
   _stability_module.Vituri2024HFSpiralFullSectorInventory.iter_sector_keys,
   _IMPORT_ITER_SECTORS,
  ),
  (
   _stability_module.Vituri2024HFSpiralFullSectorInventory.iter_conjugate_orbits,
   _IMPORT_ITER_ORBITS,
  ),
 )
 if any(current is not imported for current, imported in bindings):
  raise RuntimeError("whole-inventory reciprocity production binding drifted")


def _current_implementation_fingerprint() -> str:
 _validate_owner_bindings()
 sources = (
  _response_module._direct_rank_one_numerator,
  _response_module._accumulate_exchange_mdagger_ck_m_numerator,
  _response_module._support_indices,
  _response_module._shifted_spinors,
  _response_module._allowed_flavor_blocks,
  _response_module.Vituri2024HFSpiralSignedDisplacementResponse.conjugate_block,
  _response_module.Vituri2024HFSpiralSignedDisplacementResponse._direct_response,
  _response_module.Vituri2024HFSpiralSignedDisplacementResponse._apply_fft_validated,
  _response_module.Vituri2024HFSpiralValidatedSignedDisplacementFFTAction.__call__,
  _response_module.Vituri2024HFSpiralValidatedResponseActionFactory.prepare_fft_action,
  _hessian_module._VituriPairedInteractionCallback.__call__,
  _hessian_module.Vituri2024HFSpiralFullHessianContext._build_embedding,
  _hessian_module.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian,
  _core_reciprocity_module.PairedOrbitalTransitionFrame.unpack_real,
  _core_reciprocity_module.PairedOrbitalTransitionFrame.pack_complex,
  _core_reciprocity_module.PairedSectorOrbitalHessian.complex_gradient_jacobian_action,
  _core_reciprocity_module.PairedSectorOrbitalHessian.matvec,
  _validate_owner_bindings,
  _current_implementation_fingerprint,
  Vituri2024WholeInventoryReciprocityApproval,
  Vituri2024WholeInventoryReciprocityApproval.__post_init__,
  approve_vituri2024_whole_inventory_reciprocity,
  Vituri2024WholeInventoryReciprocityComparisonReceipt,
  Vituri2024WholeInventoryReciprocityComparisonReceipt.__post_init__,
  Vituri2024WholeInventoryReciprocityComparisonReceipt.validate_live_state,
  compare_vituri2024_whole_inventory_reciprocity,
 )
 return _fingerprint(
  {
   "api_version": VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION,
   "theorem": VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM,
   "scope": VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE,
   "authority": VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY,
   "module_files": tuple(
    sorted(
     (
      getattr(module, "__name__", ""),
      _module_file_sha256(module),
     )
     for module in (
      _response_module,
      _hessian_module,
      _core_reciprocity_module,
      _stability_module,
      sys.modules[__name__],
     )
    )
   ),
   "sources": tuple(
    (
     getattr(value, "__module__", ""),
     getattr(value, "__qualname__", ""),
     _source_sha256(value),
    )
    for value in sources
   ),
   "fft2": _source_sha256(_response_module._FFT2),
   "ifft2": _source_sha256(_response_module._IFFT2),
   "exact_integer_scalar_implementation": (
    vituri2024_exact_integer_signed_scalar_implementation_fingerprint()
   ),
   "numpy_version": np.__version__,
  }
 )


def vituri2024_whole_inventory_reciprocity_implementation_fingerprint() -> str:
 current = _current_implementation_fingerprint()
 if current != _IMPORT_IMPLEMENTATION_FINGERPRINT:
  raise RuntimeError("whole-inventory reciprocity implementation drifted")
 return current


def _strict_sha256(value: object, label: str) -> str:
 if (
  type(value) is not str
  or len(value) != 64
  or any(character not in "0123456789abcdef" for character in value)
 ):
  raise ValueError(f"{label} must be a lowercase SHA256 digest")
 return value


@dataclass(frozen=True, slots=True)
class Vituri2024WholeInventoryReciprocityApproval:
 """Detached pre-execution approval for one reviewed implementation."""

 _factory_token: InitVar[object]
 expected_implementation_fingerprint: str
 source_commit: str
 review_record_sha256: str
 reduced_exhaustive_qualification_sha256: str
 approval_record_sha256: str
 rationale: str
 fingerprint: str = field(init=False)
 api_version: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION, init=False
 )
 scope: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE, init=False
 )
 detached_preexecution_approval: bool = field(default=True, init=False)

 def __post_init__(self, _factory_token: object) -> None:
  if _factory_token is not _TOKEN:
   raise TypeError("whole-inventory reciprocity approvals are factory-only")
  _strict_sha256(
   self.expected_implementation_fingerprint,
   "expected reciprocity implementation fingerprint",
  )
  for value, label in (
   (self.review_record_sha256, "review record"),
   (self.reduced_exhaustive_qualification_sha256, "reduced qualification"),
   (self.approval_record_sha256, "approval record"),
  ):
   _strict_sha256(value, label)
  if (
   type(self.source_commit) is not str
   or len(self.source_commit) != 40
   or any(character not in "0123456789abcdef" for character in self.source_commit)
  ):
   raise ValueError("approved source commit must be a full lowercase Git digest")
  if type(self.rationale) is not str or not self.rationale.strip():
   raise ValueError("reciprocity approval rationale must be nonempty")
  if (
   self.api_version != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION
   or self.scope != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE
   or self.detached_preexecution_approval is not True
  ):
   raise ValueError("whole-inventory reciprocity approval metadata drifted")
  payload = {
   name: getattr(self, name)
   for name in self.__dataclass_fields__
   if name not in ("_factory_token", "fingerprint")
  }
  object.__setattr__(self, "fingerprint", _fingerprint(payload))


def approve_vituri2024_whole_inventory_reciprocity(
 *,
 expected_implementation_fingerprint: str,
 source_commit: str,
 review_record_sha256: str,
 reduced_exhaustive_qualification_sha256: str,
 approval_record_sha256: str,
 rationale: str,
) -> Vituri2024WholeInventoryReciprocityApproval:
 """Create detached approval from already pinned review/qualification records."""

 return Vituri2024WholeInventoryReciprocityApproval(
  _factory_token=_TOKEN,
  expected_implementation_fingerprint=expected_implementation_fingerprint,
  source_commit=source_commit,
  review_record_sha256=review_record_sha256,
  reduced_exhaustive_qualification_sha256=reduced_exhaustive_qualification_sha256,
  approval_record_sha256=approval_record_sha256,
  rationale=rationale,
 )


@dataclass(frozen=True, slots=True)
class Vituri2024WholeInventoryReciprocityComparisonReceipt:
 """Immutable positive-or-negative structural comparison."""

 _factory_token: InitVar[object]
 context_fingerprint: str
 response_fingerprint: str
 inventory_fingerprint: str
 fft_plan_fingerprint: str
 implementation_fingerprint: str
 approved_implementation_fingerprint: str
 approval_fingerprint: str
 approved_source_commit: str
 review_record_sha256: str
 reduced_exhaustive_qualification_sha256: str
 approval_record_sha256: str
 integer_mesh_labels_sha256: str
 selected_occupations_sha256: str
 selected_spinors_sha256: str
 selected_fock_diagonal_sha256: str
 mesh_size: int
 nk: int
 signed_sector_count_checked: int
 displacement_count_checked: int
 ordered_mesh_pair_count_covered: int
 nonempty_sector_count: int
 canonical_orbit_count: int
 independently_recomputed_complex_dimension: int
 inventory_complex_dimension: int
 inventory_real_dimension: int
 canonical_orbit_complex_dimension_sum: int
 maximum_kernel_imaginary_residual: float
 maximum_kernel_even_residual: float
 kernel_structure_tolerance: float
 support_mismatch_count: int
 conjugation_structure_mismatch_count: int
 paired_lane_overlap_count: int
 sector_dimension_mismatch_count: int
 canonical_orbit_mismatch_count: int
 support_inventory_fingerprint: str
 sector_dimension_fingerprint: str
 canonical_orbit_fingerprint: str
 direct_rank_one_identity_bound: bool
 exchange_mdagger_ck_m_identity_bound: bool
 signed_conjugation_covariance_derived: bool
 transition_injection_extraction_adjoint_derived: bool
 one_body_real_diagonal: bool
 factor_two_real_packing_bound: bool
 passed: bool
 failed_gates: tuple[str, ...]
 fingerprint: str = field(init=False)
 api_version: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION, init=False
 )
 theorem: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM, init=False
 )
 scope: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE, init=False
 )
 authority: str = field(
  default=VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY, init=False
 )
 candidate_only: bool = field(default=True, init=False)
 whole_inventory_reciprocity_established: bool = field(default=False, init=False)
 full_inventory_exact_unitary_scalar_curvature_established: bool = field(
  default=False, init=False
 )
 scalar_hessian_authority_established: bool = field(default=False, init=False)
 hermitian_eigensolver_authorized: bool = field(default=False, init=False)
 full_local_stability_established: bool = field(default=False, init=False)
 production_ready: bool = field(default=False, init=False)
 paper_reproduction_verified: bool = field(default=False, init=False)

 def __post_init__(self, _factory_token: object) -> None:
  if _factory_token is not _TOKEN:
   raise TypeError("whole-inventory reciprocity receipts are factory-only")
  object.__setattr__(self, "fingerprint", self._expected_fingerprint())
  self.validate_live_state()

 def _payload(self) -> dict[str, object]:
  return {
   name: getattr(self, name)
   for name in self.__dataclass_fields__
   if name not in ("_factory_token", "fingerprint")
  }

 def _expected_fingerprint(self) -> str:
  return _fingerprint(self._payload())

 def validate_live_state(self) -> None:
  for value in (
   self.context_fingerprint,
   self.response_fingerprint,
   self.inventory_fingerprint,
   self.fft_plan_fingerprint,
   self.implementation_fingerprint,
   self.approved_implementation_fingerprint,
   self.approval_fingerprint,
   self.review_record_sha256,
   self.reduced_exhaustive_qualification_sha256,
   self.approval_record_sha256,
   self.integer_mesh_labels_sha256,
   self.selected_occupations_sha256,
   self.selected_spinors_sha256,
   self.selected_fock_diagonal_sha256,
   self.support_inventory_fingerprint,
   self.sector_dimension_fingerprint,
   self.canonical_orbit_fingerprint,
  ):
   if type(value) is not str or len(value) != 64:
    raise ValueError("whole-inventory reciprocity digest is invalid")
  if (
   type(self.approved_source_commit) is not str
   or len(self.approved_source_commit) != 40
   or any(character not in "0123456789abcdef" for character in self.approved_source_commit)
  ):
   raise ValueError("whole-inventory reciprocity approved commit is invalid")
  counts = (
   self.mesh_size,
   self.nk,
   self.signed_sector_count_checked,
   self.displacement_count_checked,
   self.ordered_mesh_pair_count_covered,
   self.nonempty_sector_count,
   self.canonical_orbit_count,
   self.independently_recomputed_complex_dimension,
   self.inventory_complex_dimension,
   self.inventory_real_dimension,
   self.canonical_orbit_complex_dimension_sum,
   self.support_mismatch_count,
   self.conjugation_structure_mismatch_count,
   self.paired_lane_overlap_count,
   self.sector_dimension_mismatch_count,
   self.canonical_orbit_mismatch_count,
  )
  if any(type(value) is not int or value < 0 for value in counts):
   raise ValueError("whole-inventory reciprocity count is invalid")
  for value in (
   self.maximum_kernel_imaginary_residual,
   self.maximum_kernel_even_residual,
   self.kernel_structure_tolerance,
  ):
   if type(value) is not float or not np.isfinite(value) or value < 0.0:
    raise ValueError("whole-inventory reciprocity residual is invalid")
  structural_flags = (
   self.direct_rank_one_identity_bound,
   self.exchange_mdagger_ck_m_identity_bound,
   self.signed_conjugation_covariance_derived,
   self.transition_injection_extraction_adjoint_derived,
   self.one_body_real_diagonal,
   self.factor_two_real_packing_bound,
  )
  if any(type(value) is not bool for value in structural_flags):
   raise TypeError("whole-inventory reciprocity gate must be exact bool")
  if type(self.failed_gates) is not tuple or any(
   type(value) is not str or not value for value in self.failed_gates
  ):
   raise ValueError("whole-inventory reciprocity failed-gate inventory is invalid")
  if self.passed is not (len(self.failed_gates) == 0):
   raise ValueError("whole-inventory reciprocity verdict disagrees with gates")
  if self.passed and not all(structural_flags):
   raise ValueError("positive reciprocity receipt has an unproved structural gate")
  locked = (
   self.api_version == VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION,
   self.theorem == VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM,
   self.scope == VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE,
   self.authority == VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY,
   self.candidate_only is True,
   self.whole_inventory_reciprocity_established is False,
   self.full_inventory_exact_unitary_scalar_curvature_established is False,
   self.scalar_hessian_authority_established is False,
   self.hermitian_eigensolver_authorized is False,
   self.full_local_stability_established is False,
   self.production_ready is False,
   self.paper_reproduction_verified is False,
   self.fingerprint == self._expected_fingerprint(),
  )
  if not all(locked):
   raise ValueError("whole-inventory reciprocity receipt authority drifted")


def compare_vituri2024_whole_inventory_reciprocity(
 context: Vituri2024HFSpiralFullHessianContext,
 approval: Vituri2024WholeInventoryReciprocityApproval,
) -> Vituri2024WholeInventoryReciprocityComparisonReceipt:
 """Exhaustively compare compact identities covering every transition sector."""

 if type(context) is not Vituri2024HFSpiralFullHessianContext:
  raise TypeError("whole-inventory reciprocity requires the exact Vituri context")
 context.validate_live_state()
 implementation = vituri2024_whole_inventory_reciprocity_implementation_fingerprint()
 if type(approval) is not Vituri2024WholeInventoryReciprocityApproval:
  raise TypeError("whole-inventory reciprocity requires an exact detached approval")
 if approval.expected_implementation_fingerprint != implementation:
  raise ValueError("approved reciprocity implementation fingerprint does not match")
 response = context.response
 inventory = context.inventory
 plan = response.fft_plan
 if type(response) is not Vituri2024HFSpiralSignedDisplacementResponse:
  raise TypeError("whole-inventory reciprocity response type drifted")
 if type(inventory) is not Vituri2024HFSpiralFullSectorInventory:
  raise TypeError("whole-inventory reciprocity inventory type drifted")
 if type(plan) is not Vituri2024SquareCartesianFFTPlan:
  raise TypeError("whole-inventory reciprocity FFT plan type drifted")
 start_bindings = (
  context.context_fingerprint,
  response.response_fingerprint,
  inventory.inventory_fingerprint,
  plan.fingerprint,
 )
 size = inventory.mesh_size
 nk = inventory.nk
 labels = inventory.integer_mesh_labels
 occupations = inventory.selected_occupations
 kernel = plan.kernel_by_signed_displacement
 kernel_scale = max(1.0, float(np.max(np.abs(kernel), initial=0.0)))
 kernel_tolerance = float(64.0 * np.finfo(np.float64).eps * kernel_scale)
 imaginary_residual = float(np.max(np.abs(kernel.imag), initial=0.0))
 even_residual = float(
  np.max(np.abs(kernel - kernel[::-1, ::-1]), initial=0.0)
 )
 failed: list[str] = []
 if imaginary_residual > kernel_tolerance:
  failed.append("kernel_not_real_within_source_tolerance")
 if even_residual > kernel_tolerance:
  failed.append("kernel_not_even_within_source_tolerance")
 if np.count_nonzero(kernel.imag) != 0:
  failed.append("kernel_not_exactly_real_for_reciprocity_theorem")
 if not np.array_equal(kernel, kernel[::-1, ::-1]):
  failed.append("kernel_not_exactly_even_for_reciprocity_theorem")
 if plan.minimum_padding_size != 2 * size - 1 or plan.padding_size < 2 * size - 1:
  failed.append("fft_padding_not_linear_convolution_safe")
 support_hash = sha256()
 sector_hash = sha256()
 support_mismatches = 0
 conjugation_mismatches = 0
 paired_lane_overlaps = 0
 dimension_mismatches = 0
 support_total = 0
 independent_total = 0
 displacement_count = 0
 sector_count = 0
 offset = size - 1
 for dy in range(-offset, offset + 1):
  for dx in range(-offset, offset + 1):
   displacement_count += 1
   key0 = Vituri2024HFSpiralFullSectorKey(dx, dy, 0)
   bases, targets = _response_module._support_indices(inventory, key0)
   negative_bases, negative_targets = _response_module._support_indices(
    inventory, key0.conjugate
   )
   expected_support = (size - abs(dx)) * (size - abs(dy))
   inverse = np.full(nk, -1, dtype=np.int64)
   inverse[negative_bases] = negative_targets
   support_ok = (
    bases.size == expected_support
    and np.unique(bases).size == bases.size
    and np.unique(targets).size == targets.size
    and np.all(labels[targets] - labels[bases] == np.asarray((dx, dy)))
    and np.all(inverse[targets] == bases)
   )
   if not support_ok:
    support_mismatches += 1
   support_total += int(bases.size)
   support_hash.update(json.dumps((dx, dy), separators=(",", ":")).encode())
   support_hash.update(_array_sha256(bases).encode())
   support_hash.update(_array_sha256(targets).encode())
   for charge in (-2, 0, 2):
    sector_count += 1
    key = Vituri2024HFSpiralFullSectorKey(dx, dy, charge)
    allowed = _response_module._allowed_flavor_blocks(key)
    partner_allowed = _response_module._allowed_flavor_blocks(key.conjugate)
    if {(right, left) for left, right in allowed} != set(partner_allowed):
     conjugation_mismatches += 1
    independent_dimension = 0
    for particle_slot, hole_slot in allowed:
     forward = (
      occupations[hole_slot, bases]
      & ~occupations[particle_slot, targets]
     )
     reverse = (
      occupations[particle_slot, targets]
      & ~occupations[hole_slot, bases]
     )
     independent_dimension += int(np.count_nonzero(forward))
     paired_lane_overlaps += int(np.count_nonzero(forward & reverse))
    declared = inventory.sector_complex_dimension(key)
    if independent_dimension != declared:
     dimension_mismatches += 1
    independent_total += independent_dimension
    sector_hash.update(
     json.dumps(
      (dx, dy, charge, independent_dimension, declared), separators=(",", ":")
     ).encode()
    )
 if support_total != nk * nk:
  failed.append("ordered_mesh_pair_coverage_failed")
 if support_mismatches:
  failed.append("signed_support_bijection_failed")
 if conjugation_mismatches:
  failed.append("signed_conjugation_structure_failed")
 if paired_lane_overlaps:
  failed.append("paired_transition_lane_disjointness_failed")
 if dimension_mismatches:
  failed.append("sector_dimension_recomputation_failed")
 expected_total = inventory.selected_occupied_count * inventory.selected_virtual_count
 if independent_total != expected_total or independent_total != inventory.complex_dimension:
  failed.append("whole_transition_inventory_dimension_failed")
 orbit_hash = sha256()
 seen: set[tuple[int, int, int]] = set()
 orbit_mismatches = 0
 orbit_dimension_sum = 0
 orbit_count = 0
 for orbit in inventory.iter_conjugate_orbits(include_zero_dimension=False):
  orbit_count += 1
  first = (
   orbit.first.displacement_x,
   orbit.first.displacement_y,
   orbit.first.valley_charge,
  )
  second = (
   orbit.second.displacement_x,
   orbit.second.displacement_y,
   orbit.second.valley_charge,
  )
  declared_first = inventory.sector_complex_dimension(orbit.first)
  declared_second = inventory.sector_complex_dimension(orbit.second)
  for key_tuple, dimension in (
   (first, declared_first),
   (second, declared_second),
  ):
   if dimension > 0:
    if key_tuple in seen:
     orbit_mismatches += 1
    seen.add(key_tuple)
  if (
   orbit.first.conjugate != orbit.second
   or orbit.second < orbit.first
   or orbit.first_complex_dimension != declared_first
   or orbit.second_complex_dimension != declared_second
   or orbit.self_conjugate
  ):
   orbit_mismatches += 1
  orbit_dimension_sum += orbit.complex_dimension
  orbit_hash.update(
   json.dumps(
    (first, second, declared_first, declared_second), separators=(",", ":")
   ).encode()
  )
 if (
  orbit_mismatches
  or len(seen) != inventory.nonempty_sector_count
  or orbit_count != inventory.nonempty_conjugate_orbit_count
  or orbit_dimension_sum != inventory.complex_dimension
 ):
  failed.append("canonical_conjugate_orbit_coverage_failed")
 self_key = Vituri2024HFSpiralFullSectorKey(0, 0, 0)
 if inventory.sector_complex_dimension(self_key) != 0:
  failed.append("nonzero_self_conjugate_sector_requires_separate_proof")
 fock = context.selected_fock_diagonal_ev
 one_body_real = (
  type(fock) is np.ndarray
  and fock.dtype == np.dtype(np.float64)
  and fock.shape == (2, nk)
  and np.all(np.isfinite(fock))
 )
 if not one_body_real:
  failed.append("one_body_not_real_diagonal")
 direct_bound = (
  _response_module._direct_rank_one_numerator is _IMPORT_DIRECT_PRIMITIVE
  and _response_module.Vituri2024HFSpiralSignedDisplacementResponse._direct_response
  is _IMPORT_DIRECT_WRAPPER
 )
 exchange_bound = (
  _response_module._accumulate_exchange_mdagger_ck_m_numerator
  is _IMPORT_EXCHANGE_PRIMITIVE
  and _response_module.Vituri2024HFSpiralSignedDisplacementResponse._apply_fft_validated
  is _IMPORT_APPLY_FFT
 )
 conjugation_derived = (
  direct_bound
  and exchange_bound
  and imaginary_residual == 0.0
  and even_residual == 0.0
  and support_mismatches == 0
  and conjugation_mismatches == 0
 )
 transition_adjoint_derived = (
  paired_lane_overlaps == 0
  and dimension_mismatches == 0
  and _hessian_module.Vituri2024HFSpiralFullHessianContext._build_embedding
  is _IMPORT_BUILD_EMBEDDING
  and _hessian_module._VituriPairedInteractionCallback.__call__
  is _IMPORT_PAIRED_CALLBACK
 )
 packing_bound = (
  _core_reciprocity_module.PairedOrbitalTransitionFrame.unpack_real
  is _IMPORT_UNPACK_REAL
  and _core_reciprocity_module.PairedOrbitalTransitionFrame.pack_complex
  is _IMPORT_PACK_COMPLEX
  and _core_reciprocity_module.PairedSectorOrbitalHessian.matvec is _IMPORT_MATVEC
 )
 for established, gate in (
  (direct_bound, "direct_rank_one_binding_failed"),
  (exchange_bound, "exchange_factorization_binding_failed"),
  (conjugation_derived, "signed_conjugation_derivation_failed"),
  (transition_adjoint_derived, "transition_adjoint_derivation_failed"),
  (packing_bound, "factor_two_real_packing_binding_failed"),
 ):
  if not established:
   failed.append(gate)
 context.validate_live_state()
 end_implementation = vituri2024_whole_inventory_reciprocity_implementation_fingerprint()
 end_bindings = (
  context.context_fingerprint,
  response.response_fingerprint,
  inventory.inventory_fingerprint,
  plan.fingerprint,
 )
 if start_bindings != end_bindings or implementation != end_implementation:
  raise RuntimeError("whole-inventory reciprocity source changed during comparison")
 if approval.expected_implementation_fingerprint != end_implementation:
  raise RuntimeError("approved reciprocity implementation changed during comparison")
 failed_gates = tuple(sorted(set(failed)))
 return Vituri2024WholeInventoryReciprocityComparisonReceipt(
  _factory_token=_TOKEN,
  context_fingerprint=context.context_fingerprint,
  response_fingerprint=response.response_fingerprint,
  inventory_fingerprint=inventory.inventory_fingerprint,
  fft_plan_fingerprint=plan.fingerprint,
  implementation_fingerprint=implementation,
  approved_implementation_fingerprint=approval.expected_implementation_fingerprint,
  approval_fingerprint=approval.fingerprint,
  approved_source_commit=approval.source_commit,
  review_record_sha256=approval.review_record_sha256,
  reduced_exhaustive_qualification_sha256=(
   approval.reduced_exhaustive_qualification_sha256
  ),
  approval_record_sha256=approval.approval_record_sha256,
  integer_mesh_labels_sha256=_array_sha256(labels),
  selected_occupations_sha256=_array_sha256(occupations),
  selected_spinors_sha256=_array_sha256(response.selected_spinors),
  selected_fock_diagonal_sha256=_array_sha256(fock),
  mesh_size=size,
  nk=nk,
  signed_sector_count_checked=sector_count,
  displacement_count_checked=displacement_count,
  ordered_mesh_pair_count_covered=support_total,
  nonempty_sector_count=inventory.nonempty_sector_count,
  canonical_orbit_count=orbit_count,
  independently_recomputed_complex_dimension=independent_total,
  inventory_complex_dimension=inventory.complex_dimension,
  inventory_real_dimension=inventory.real_dimension,
  canonical_orbit_complex_dimension_sum=orbit_dimension_sum,
  maximum_kernel_imaginary_residual=imaginary_residual,
  maximum_kernel_even_residual=even_residual,
  kernel_structure_tolerance=kernel_tolerance,
  support_mismatch_count=support_mismatches,
  conjugation_structure_mismatch_count=conjugation_mismatches,
  paired_lane_overlap_count=paired_lane_overlaps,
  sector_dimension_mismatch_count=dimension_mismatches,
  canonical_orbit_mismatch_count=orbit_mismatches,
  support_inventory_fingerprint=support_hash.hexdigest(),
  sector_dimension_fingerprint=sector_hash.hexdigest(),
  canonical_orbit_fingerprint=orbit_hash.hexdigest(),
  direct_rank_one_identity_bound=direct_bound,
  exchange_mdagger_ck_m_identity_bound=exchange_bound,
  signed_conjugation_covariance_derived=conjugation_derived,
  transition_injection_extraction_adjoint_derived=transition_adjoint_derived,
  one_body_real_diagonal=bool(one_body_real),
  factor_two_real_packing_bound=packing_bound,
  passed=not failed_gates,
  failed_gates=failed_gates,
 )


# Initialized only after every authority-bearing class and function exists.
_IMPORT_IMPLEMENTATION_FINGERPRINT = _current_implementation_fingerprint()


__all__ = [
 "VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION",
 "VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY",
 "VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE",
 "VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM",
 "Vituri2024WholeInventoryReciprocityApproval",
 "Vituri2024WholeInventoryReciprocityComparisonReceipt",
 "approve_vituri2024_whole_inventory_reciprocity",
 "compare_vituri2024_whole_inventory_reciprocity",
 "vituri2024_whole_inventory_reciprocity_implementation_fingerprint",
]
