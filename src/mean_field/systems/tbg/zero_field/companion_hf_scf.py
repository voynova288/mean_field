"""Source-faithful Stage6A companion Hartree--Fock SCF diagnostic.

This module is deliberately isolated from the package front door, production HF,
TDHF, and Fig. 8 workflows.  It consumes the immutable Stage4 prepared action
and, optionally, the Stage5 Eq. (99) diagnostic seed.  The implementation is a
literal diagnostic transcription of ``routines.py`` Aufbau and the
``mainProgram.py`` ODA loop: the stored projector orientation, C-order global
fill, pre-mixing convergence norm, branch order, and zero-based strict
convergence condition are all part of the contract.

System-local source-parity exception
------------------------------------
The reusable ``mean_field.core.hf`` engine remains the only generic HF framework.
It is demonstrably non-equivalent to this pinned companion source in the ordered
ODA branch selection, the pre-mixing convergence norm and zero-based strict
convergence index, and the final mixed-projector action/Aufbau rebuild.  This
module therefore retains those source details only as an isolated diagnostic
compatibility lane.  It is not a reusable framework fork and must not be exposed
through a package front door or used for production HF, TDHF, or Fig. 8.

No artifact writer, production adapter, or Slurm entry point is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Final, Literal

import numpy as np

from .companion_hf_action import (
    TBGZeroFieldCompanionHFAction,
    TBGZeroFieldCompanionHFEvaluation,
    TBGZeroFieldCompanionPreparedHFAction,
    TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD,
    TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE,
    TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE,
    TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256,
    calc_fock_matrix,
)
from .companion_kivc_seed import (
    TBGZeroFieldCompanionKIVCSeedResult,
    TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256,
)

TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_hf_scf"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE: Final[str] = (
    "diagnostic_source_faithful_companion_SCF_only_not_production_HF_TDHF_or_Fig8"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION: Final[str] = (
    "system_local_diagnostic_compatibility_lane_only;generic_core_engine_is_"
    "non_equivalent_in_ODA_branch_convergence_norm_index_and_finalization;"
    "not_reusable_framework_fork;not_frontdoor_production_TDHF_or_Fig8"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_STORED_PROJECTOR_CONVENTION: Final[str] = (
    "P[k1,k2,spin,alpha,beta]=<c_dagger_alpha(k)_spin c_beta(k)_spin>;"
    "Aufbau=sum_n_conj(evec[alpha,n])*evec[beta,n]*fill[n]"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_FILL_ORDER: Final[str] = (
    "global_C_order_flatten_of_[k1,k2,spin,state]_then_numpy_default_argsort"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_DIFFERENCE_CONVENTION: Final[str] = (
    "norm(P_old-P_raw)/Nk_before_ODA_mixing"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_CONVERGENCE_CONVENTION: Final[str] = (
    "zero_based_iteration;strict_difference_lt_tolerance_and_iteration_gt_HF_itermin"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_array_bytes"
)
TBG_ZERO_FIELD_COMPANION_HF_SCF_ENERGY_UNITS: Final[str] = (
    "finite_system_eV_not_per_moire_cell"
)

TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES: Final[str] = "155-198"
TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES: Final[str] = "104-167"
TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES: Final[str] = "120-136"
TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES: Final[str] = "80-87"
TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES: Final[str] = "5-14,35-72"

TBG_ZERO_FIELD_COMPANION_HF_ITERMAX_DEFAULT: Final[int] = 800
TBG_ZERO_FIELD_COMPANION_HF_ITERMIN_DEFAULT: Final[int] = 20
TBG_ZERO_FIELD_COMPANION_HF_TOLERANCE_DEFAULT: Final[float] = 1.0e-8
TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD: Final[float] = 1.0e-12
TBG_ZERO_FIELD_COMPANION_HF_TYPE: Final[str] = "ODA"

ODABranch = Literal[
    "positive_linear",
    "endpoint_quad",
    "linear_near_zero",
    "interior",
]


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    return int(value)


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar (bool is not accepted)")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return resolved


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hexadecimal string")
    resolved = value.strip().lower()
    if len(resolved) != 64 or any(c not in "0123456789abcdef" for c in resolved):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return resolved


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_sha256(values: np.ndarray) -> str:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        dtype = np.dtype("<c16")
    elif source.dtype.kind == "f":
        dtype = np.dtype("<f8")
    elif source.dtype.kind in "iub":
        dtype = np.dtype("<i8")
    else:
        raise TypeError(f"Unsupported array dtype for canonical hashing: {source.dtype}")
    array = np.ascontiguousarray(source, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _validate_live_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must remain a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if values.shape != shape:
        raise ValueError(f"{name} must retain shape {shape}, got {values.shape}")
    if values.dtype != expected_dtype:
        raise ValueError(
            f"{name} must retain dtype {expected_dtype.str}, got {values.dtype.str}"
        )
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must remain C-contiguous")
    if values.flags.writeable:
        raise ValueError(f"{name} must remain read-only")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _max_hermiticity_residual(values: np.ndarray) -> float:
    array = np.asarray(values)
    return _max_abs(array - np.swapaxes(array.conj(), -1, -2))


def _hamiltonian_shape(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
) -> tuple[int, int, int, int, int]:
    params = prepared.params
    dimension = 4 * params.n_active
    return (params.N1, params.N2, 2, dimension, dimension)


def _electron_count(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    filling: object,
) -> tuple[int, int]:
    resolved_filling = _strict_int(filling, name="filling")
    params = prepared.params
    count = params.N1 * params.N2 * (4 * params.n_active + resolved_filling)
    total = params.N1 * params.N2 * 2 * (4 * params.n_active)
    if count < 0 or count > total:
        raise ValueError(
            f"filling={resolved_filling} gives invalid occupied count {count}/{total}"
        )
    return resolved_filling, count


def _tuple_hashes(values: object, *, name: str, length: int) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = tuple(values)  # type: ignore[arg-type]
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    return tuple(
        _validate_sha256(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFSpec:
    """Immutable source-loop controls; only the companion ODA mode is accepted."""

    filling: int = 0
    HF_itermax: int = TBG_ZERO_FIELD_COMPANION_HF_ITERMAX_DEFAULT
    HF_itermin: int = TBG_ZERO_FIELD_COMPANION_HF_ITERMIN_DEFAULT
    tolerance: float = TBG_ZERO_FIELD_COMPANION_HF_TOLERANCE_DEFAULT
    HF_type: str = TBG_ZERO_FIELD_COMPANION_HF_TYPE
    branch_threshold: float = TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD

    def __post_init__(self) -> None:
        filling = _strict_int(self.filling, name="filling")
        itermax = _strict_int(self.HF_itermax, name="HF_itermax")
        itermin = _strict_int(self.HF_itermin, name="HF_itermin")
        tolerance = _finite_real(self.tolerance, name="tolerance", positive=True)
        threshold = _finite_real(
            self.branch_threshold,
            name="branch_threshold",
            positive=True,
        )
        if itermax <= 0:
            raise ValueError("HF_itermax must be positive")
        if itermin < 0:
            raise ValueError("HF_itermin must be nonnegative")
        if self.HF_type != TBG_ZERO_FIELD_COMPANION_HF_TYPE:
            raise ValueError("Stage6A companion SCF supports only HF_type='ODA'")
        if threshold != TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD:
            raise ValueError(
                "branch_threshold is source-pinned to 1e-12 by mainProgram.py"
            )
        object.__setattr__(self, "filling", filling)
        object.__setattr__(self, "HF_itermax", itermax)
        object.__setattr__(self, "HF_itermin", itermin)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "branch_threshold", threshold)

    @property
    def HF_tolerance(self) -> float:
        """Source-name alias retained in metadata and fixture comparisons."""

        return self.tolerance

    def to_companion_input(self) -> dict[str, int | float | str]:
        return {
            "filling": self.filling,
            "HF_itermax": self.HF_itermax,
            "HF_itermin": self.HF_itermin,
            "HF_tolerance": self.tolerance,
            "HF_type": self.HF_type,
            "ODA_branch_threshold": self.branch_threshold,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "input": self.to_companion_input(),
                "schema": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
            }
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFProvenance:
    routines_source: str = TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE
    routines_source_sha256: str = TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256
    aufbau_reference_lines: str = TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES
    main_program_source: str = TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE
    main_program_source_sha256: str = TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256
    main_scf_reference_lines: str = TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES
    main_oda_reference_lines: str = TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES
    projectors_source: str = TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE
    projectors_source_sha256: str = TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256
    average_central_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES
    )
    measure_source: str = TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE
    measure_source_sha256: str = TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256
    measure_reference_lines: str = TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE
    source_parity_exception: str = (
        TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION
    )

    def __post_init__(self) -> None:
        expected = {
            "routines_source": TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE,
            "routines_source_sha256": TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256,
            "aufbau_reference_lines": TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES,
            "main_program_source": TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE,
            "main_program_source_sha256": (
                TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256
            ),
            "main_scf_reference_lines": TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES,
            "main_oda_reference_lines": TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES,
            "projectors_source": TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE,
            "projectors_source_sha256": (
                TBG_ZERO_FIELD_COMPANION_PROJECTORS_SOURCE_SHA256
            ),
            "average_central_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES
            ),
            "measure_source": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE,
            "measure_source_sha256": TBG_ZERO_FIELD_COMPANION_MEASURE_SOURCE_SHA256,
            "measure_reference_lines": TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES,
            "scientific_scope": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
            "source_parity_exception": (
                TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} differs from pinned Stage6A provenance")

    def to_metadata(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_metadata())


def build_companion_average_central_reference(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
) -> np.ndarray:
    """Return literal ``projectors.py`` average-central stored projector.

    For ``n_active == 1`` this is exactly ``0.5 * I_4`` at every momentum and
    spin.  For a general active window it fills lower remote bands, half-fills
    the two central bands, and leaves upper remote bands empty in each valley.
    """

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    _ = prepared.fingerprint
    params = prepared.params
    nactive = params.n_active
    Pex = np.zeros(
        (params.N1, params.N2, 2, 2, 2 * nactive, 2, 2 * nactive),
        dtype=np.complex128,
    )
    for tau in range(2):
        for band in range(nactive - 1):
            Pex[:, :, :, tau, band, tau, band] = 1.0
        for central in range(2):
            band = nactive + central - 1
            Pex[:, :, :, tau, band, tau, band] = 0.5
    result = np.reshape(
        Pex,
        (params.N1, params.N2, 2, 4 * nactive, 4 * nactive),
        order="C",
    )
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionAufbauArrayHashes:
    hamiltonian_ev: str
    projector: str
    eigenvalues_ev: str
    fill_indices: str
    eigenvectors: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )

    @classmethod
    def from_arrays(
        cls,
        *,
        hamiltonian_ev: np.ndarray,
        projector: np.ndarray,
        eigenvalues_ev: np.ndarray,
        fill_indices: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> "TBGZeroFieldCompanionAufbauArrayHashes":
        return cls(
            hamiltonian_ev=_canonical_array_sha256(hamiltonian_ev),
            projector=_canonical_array_sha256(projector),
            eigenvalues_ev=_canonical_array_sha256(eigenvalues_ev),
            fill_indices=_canonical_array_sha256(fill_indices),
            eigenvectors=_canonical_array_sha256(eigenvectors),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionAufbauResult:
    """Typed literal output of companion ``routines.aufbau``."""

    prepared: TBGZeroFieldCompanionPreparedHFAction
    prepared_fingerprint: str
    filling: int
    electron_count: int
    hamiltonian_ev: np.ndarray
    projector: np.ndarray
    eigenvalues_ev: np.ndarray
    fill_indices: np.ndarray
    eigenvectors: np.ndarray
    array_hashes: TBGZeroFieldCompanionAufbauArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
        prepared_fingerprint = self.prepared.fingerprint
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        if self.prepared_fingerprint != prepared_fingerprint:
            raise ValueError("prepared_fingerprint does not match live prepared input")
        filling, expected_count = _electron_count(self.prepared, self.filling)
        count = _strict_int(self.electron_count, name="electron_count")
        if count != expected_count:
            raise ValueError("electron_count does not match source filling formula")
        object.__setattr__(self, "filling", filling)
        object.__setattr__(self, "electron_count", count)
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionAufbauArrayHashes):
            raise TypeError("array_hashes must be typed Aufbau array hashes")

        params = self.prepared.params
        shape = _hamiltonian_shape(self.prepared)
        dimension = shape[-1]
        sectors = params.N1 * params.N2 * 2
        arrays = {
            "hamiltonian_ev": _readonly_array(
                self.hamiltonian_ev,
                name="hamiltonian_ev",
                shape=shape,
                dtype=np.complex128,
            ),
            "projector": _readonly_array(
                self.projector,
                name="projector",
                shape=shape,
                dtype=np.complex128,
            ),
            "eigenvalues_ev": _readonly_array(
                self.eigenvalues_ev,
                name="eigenvalues_ev",
                shape=(sectors * dimension,),
                dtype=np.float64,
            ),
            "fill_indices": _readonly_array(
                self.fill_indices,
                name="fill_indices",
                shape=(count,),
                dtype=np.int64,
            ),
            "eigenvectors": _readonly_array(
                self.eigenvectors,
                name="eigenvectors",
                shape=(sectors, dimension, dimension),
                dtype=np.complex128,
            ),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        self._validate_live_state()

    def _validate_live_state(self) -> None:
        live_prepared_fingerprint = self.prepared.fingerprint
        if self.prepared_fingerprint != live_prepared_fingerprint:
            raise ValueError("Aufbau prepared binding drifted")
        filling, expected_count = _electron_count(self.prepared, self.filling)
        if filling != self.filling or expected_count != self.electron_count:
            raise ValueError("Aufbau filling/count binding drifted")
        params = self.prepared.params
        shape = _hamiltonian_shape(self.prepared)
        dimension = shape[-1]
        sectors = params.N1 * params.N2 * 2
        arrays = {
            "hamiltonian_ev": _validate_live_array(
                self.hamiltonian_ev,
                name="Aufbau.hamiltonian_ev",
                shape=shape,
                dtype=np.complex128,
            ),
            "projector": _validate_live_array(
                self.projector,
                name="Aufbau.projector",
                shape=shape,
                dtype=np.complex128,
            ),
            "eigenvalues_ev": _validate_live_array(
                self.eigenvalues_ev,
                name="Aufbau.eigenvalues_ev",
                shape=(sectors * dimension,),
                dtype=np.float64,
            ),
            "fill_indices": _validate_live_array(
                self.fill_indices,
                name="Aufbau.fill_indices",
                shape=(self.electron_count,),
                dtype=np.int64,
            ),
            "eigenvectors": _validate_live_array(
                self.eigenvectors,
                name="Aufbau.eigenvectors",
                shape=(sectors, dimension, dimension),
                dtype=np.complex128,
            ),
        }
        actual_hashes = TBGZeroFieldCompanionAufbauArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("Aufbau array_hashes no longer match live arrays")
        if _max_hermiticity_residual(arrays["hamiltonian_ev"]) > (
            TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("Aufbau Hamiltonian is materially non-Hermitian")

        expected_fill = np.argsort(
            arrays["eigenvalues_ev"].reshape(-1, order="C")
        )[: self.electron_count].astype(np.int64, copy=False)
        if not np.array_equal(arrays["fill_indices"], expected_fill):
            raise ValueError("fill_indices do not equal source C-order global argsort")
        fill_mask = np.zeros(sectors * dimension, dtype=np.float64)
        fill_mask[arrays["fill_indices"]] = 1.0
        fill_mask = np.reshape(fill_mask, (sectors, dimension), order="C")
        expected_projector = np.einsum(
            "san,sbn,sn->sab",
            arrays["eigenvectors"].conj(),
            arrays["eigenvectors"],
            fill_mask,
            optimize=True,
        )
        expected_projector = np.reshape(expected_projector, shape, order="C")
        if not np.array_equal(arrays["projector"], expected_projector):
            raise ValueError(
                "projector does not equal evecs.conj()*evecs in stored orientation"
            )

        matrices = np.reshape(
            arrays["hamiltonian_ev"],
            (sectors, dimension, dimension),
            order="C",
        )
        eigenvalue_matrix = np.reshape(
            arrays["eigenvalues_ev"],
            (sectors, dimension),
            order="C",
        )
        vectors = arrays["eigenvectors"]
        residual = _max_abs(matrices @ vectors - vectors * eigenvalue_matrix[:, None, :])
        scale = max(1.0, _max_abs(matrices), _max_abs(eigenvalue_matrix))
        if residual > 1.0e-10 * scale:
            raise ValueError("Aufbau eigensystem does not close against Hamiltonian")
        identity = np.eye(dimension, dtype=np.complex128)
        if _max_abs(np.swapaxes(vectors.conj(), -1, -2) @ vectors - identity) > 1.0e-10:
            raise ValueError("Aufbau eigenvectors are not unitary")

    @property
    def P(self) -> np.ndarray:
        return self.projector

    @property
    def HFeigs(self) -> np.ndarray:
        return self.eigenvalues_ev

    @property
    def fill_index(self) -> np.ndarray:
        return self.fill_indices

    @property
    def evecs(self) -> np.ndarray:
        return self.eigenvectors

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": self.array_hashes.fingerprint,
                "electron_count": self.electron_count,
                "filling": self.filling,
                "prepared_fingerprint": self.prepared_fingerprint,
                "scope": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
            }
        )


def companion_aufbau(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    hamiltonian_ev: np.ndarray,
    *,
    filling: int = 0,
) -> TBGZeroFieldCompanionAufbauResult:
    """Literal ``routines.py`` lines 155--198 with C-order global filling."""

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    prepared_fingerprint = prepared.fingerprint
    resolved_filling, electron_count = _electron_count(prepared, filling)
    shape = _hamiltonian_shape(prepared)
    H = np.asarray(hamiltonian_ev, dtype=np.complex128)
    if H.shape != shape:
        raise ValueError(f"hamiltonian_ev must have shape {shape}, got {H.shape}")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian_ev must contain only finite values")
    if _max_hermiticity_residual(H) > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
        raise ValueError("hamiltonian_ev is materially non-Hermitian")

    params = prepared.params
    dimension = shape[-1]
    sectors = params.N1 * params.N2 * 2
    eigenvalues = np.zeros((sectors, dimension), dtype=np.float64)
    eigenvectors = np.zeros((sectors, dimension, dimension), dtype=np.complex128)
    for ik1 in range(params.N1):
        for ik2 in range(params.N2):
            for spin in range(2):
                sector = ik1 * 2 * params.N2 + ik2 * 2 + spin
                eigenvalues[sector], eigenvectors[sector] = np.linalg.eigh(
                    H[ik1, ik2, spin]
                )

    flat_eigenvalues = np.reshape(eigenvalues, (-1), order="C")
    fill_indices = flat_eigenvalues.argsort()[:electron_count]
    fill_mask = np.zeros(sectors * dimension, dtype=np.float64)
    fill_mask[fill_indices] = 1.0
    fill_mask = np.reshape(fill_mask, (sectors, dimension), order="C")
    projector = np.einsum(
        "san,sbn,sn->sab",
        eigenvectors.conj(),
        eigenvectors,
        fill_mask,
        optimize=True,
    )
    projector = np.reshape(projector, shape, order="C")
    hashes = TBGZeroFieldCompanionAufbauArrayHashes.from_arrays(
        hamiltonian_ev=H,
        projector=projector,
        eigenvalues_ev=flat_eigenvalues,
        fill_indices=fill_indices,
        eigenvectors=eigenvectors,
    )
    return TBGZeroFieldCompanionAufbauResult(
        prepared=prepared,
        prepared_fingerprint=prepared_fingerprint,
        filling=resolved_filling,
        electron_count=electron_count,
        hamiltonian_ev=H,
        projector=projector,
        eigenvalues_ev=flat_eigenvalues,
        fill_indices=fill_indices,
        eigenvectors=eigenvectors,
        array_hashes=hashes,
    )


def companion_oda_branch(
    lin: float,
    quad: float,
    *,
    branch_threshold: float = TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD,
) -> tuple[float, ODABranch, bool]:
    """Apply the exact ordered branch block from ``mainProgram.py`` 127--136."""

    resolved_lin = _finite_real(lin, name="lin")
    resolved_quad = _finite_real(quad, name="quad")
    threshold = _finite_real(
        branch_threshold,
        name="branch_threshold",
        positive=True,
    )
    if threshold != TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD:
        raise ValueError("branch_threshold is source-pinned to 1e-12")
    if resolved_lin > 0.0 and abs(resolved_lin) > threshold:
        return 1.0, "positive_linear", True
    if resolved_quad <= -resolved_lin / 2.0:
        return 1.0, "endpoint_quad", False
    if abs(resolved_lin) < threshold:
        return 1.0, "linear_near_zero", False
    return -resolved_lin / 2.0 / resolved_quad, "interior", False


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionODACoefficients:
    prepared: TBGZeroFieldCompanionPreparedHFAction
    c1: float
    c01: float
    c11: float
    lin: float
    quad: float
    mixing_lambda: float
    branch: ODABranch
    positive_linear: bool
    prepared_fingerprint: str
    dP_sha256: str
    dP_action: TBGZeroFieldCompanionHFAction

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
        live_prepared_fingerprint = self.prepared.fingerprint
        for name in ("c1", "c01", "c11", "lin", "quad", "mixing_lambda"):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        object.__setattr__(
            self,
            "dP_sha256",
            _validate_sha256(self.dP_sha256, name="dP_sha256"),
        )
        if self.prepared_fingerprint != live_prepared_fingerprint:
            raise ValueError("prepared_fingerprint does not match live prepared input")
        if not isinstance(self.dP_action, TBGZeroFieldCompanionHFAction):
            raise TypeError("dP_action must be TBGZeroFieldCompanionHFAction")
        if self.dP_action.params_fingerprint != self.prepared.params.fingerprint:
            raise ValueError("dP_action is not bound to prepared params")
        self.dP_action._validate_live_arrays()
        if _canonical_array_sha256(self.dP_action.density_delta) != self.dP_sha256:
            raise ValueError("dP_action is not bound to dP_sha256")
        if self.lin != self.c1 + self.c01:
            raise ValueError("lin must equal c1+c01 exactly")
        if self.quad != 0.5 * self.c11:
            raise ValueError("quad must equal 0.5*c11 exactly")
        expected_lambda, expected_branch, expected_positive = companion_oda_branch(
            self.lin,
            self.quad,
        )
        if self.mixing_lambda != expected_lambda:
            raise ValueError("mixing_lambda does not match exact source ODA branch")
        if self.branch != expected_branch:
            raise ValueError("branch does not match exact source ODA branch order")
        if not isinstance(self.positive_linear, bool):
            raise TypeError("positive_linear must be bool")
        if self.positive_linear != expected_positive:
            raise ValueError("positive_linear does not match exact source branch")

    @property
    def lambda_value(self) -> float:
        return self.mixing_lambda

    @property
    def fingerprint(self) -> str:
        if self.prepared.fingerprint != self.prepared_fingerprint:
            raise ValueError("ODA prepared binding drifted")
        self.dP_action._validate_live_arrays()
        return _json_sha256(
            {
                "branch": self.branch,
                "c01": self.c01,
                "c1": self.c1,
                "c11": self.c11,
                "dP_action": self.dP_action.fingerprint,
                "dP_sha256": self.dP_sha256,
                "lambda": self.mixing_lambda,
                "lin": self.lin,
                "positive_linear": self.positive_linear,
                "prepared_fingerprint": self.prepared_fingerprint,
                "quad": self.quad,
            }
        )


def companion_oda_coefficients(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    projector_old: np.ndarray,
    projector_raw: np.ndarray,
    reference: np.ndarray,
    *,
    branch_threshold: float = TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD,
) -> TBGZeroFieldCompanionODACoefficients:
    """Compute source ODA coefficients using the Stage4 canonical action on dP."""

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    prepared_fingerprint = prepared.fingerprint
    shape = _hamiltonian_shape(prepared)
    old = np.asarray(projector_old, dtype=np.complex128)
    raw = np.asarray(projector_raw, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    for name, array in (("projector_old", old), ("projector_raw", raw), ("reference", ref)):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    dP = np.asarray(raw - old)
    action = calc_fock_matrix(
        prepared.params,
        dP,
        prepared.form,
        prepared.M_ev,
        prepared.tVE_ev,
    )
    H_dP = action.H_interaction_ev
    c1 = float(np.real(np.einsum("kKsab,kKsab->", prepared.H_SP_ev, dP, optimize=True)))
    c01 = float(np.real(np.einsum("kKsab,kKsab->", H_dP, old - ref, optimize=True)))
    c11 = float(np.real(np.einsum("kKsab,kKsab->", H_dP, dP, optimize=True)))
    lin = c1 + c01
    quad = 0.5 * c11
    mixing_lambda, branch, positive = companion_oda_branch(
        lin,
        quad,
        branch_threshold=branch_threshold,
    )
    return TBGZeroFieldCompanionODACoefficients(
        prepared=prepared,
        c1=c1,
        c01=c01,
        c11=c11,
        lin=lin,
        quad=quad,
        mixing_lambda=mixing_lambda,
        branch=branch,
        positive_linear=positive,
        prepared_fingerprint=prepared_fingerprint,
        dP_sha256=_canonical_array_sha256(dP),
        dP_action=action,
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFHistoryHashes:
    records: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            object.__setattr__(self, "records", tuple(self.records))
        names: set[str] = set()
        normalized: list[tuple[str, str]] = []
        for record in self.records:
            if not isinstance(record, tuple) or len(record) != 2:
                raise TypeError("history hash records must be (name, sha256) tuples")
            name, digest = record
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("history hash names must be unique nonempty strings")
            names.add(name)
            normalized.append((name, _validate_sha256(digest, name=name)))
        object.__setattr__(self, "records", tuple(normalized))

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "TBGZeroFieldCompanionHFSCFHistoryHashes":
        return cls(
            records=tuple(
                (name, _canonical_array_sha256(arrays[name]))
                for name in sorted(arrays)
            )
        )

    def to_dict(self) -> dict[str, str]:
        return dict(self.records)

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_dict())


_HISTORY_FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "differences",
    "c1",
    "c01",
    "c11",
    "lin",
    "quad",
    "mixing_lambda",
)


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFHistory:
    iterations: np.ndarray
    differences: np.ndarray
    c1: np.ndarray
    c01: np.ndarray
    c11: np.ndarray
    lin: np.ndarray
    quad: np.ndarray
    mixing_lambda: np.ndarray
    branches: tuple[ODABranch, ...]
    positive_linear: np.ndarray
    energies_ev: np.ndarray
    old_projector_hashes: tuple[str, ...]
    raw_projector_hashes: tuple[str, ...]
    mixed_projector_hashes: tuple[str, ...]
    hamiltonian_hashes: tuple[str, ...]
    old_action_fingerprints: tuple[str, ...]
    dP_action_fingerprints: tuple[str, ...]
    energy_fingerprints: tuple[str, ...]
    array_hashes: TBGZeroFieldCompanionHFSCFHistoryHashes

    def __post_init__(self) -> None:
        iteration_source = np.asarray(self.iterations)
        if iteration_source.ndim != 1:
            raise ValueError("iterations must be one-dimensional")
        count = int(iteration_source.size)
        arrays: dict[str, np.ndarray] = {
            "iterations": _readonly_array(
                self.iterations,
                name="iterations",
                shape=(count,),
                dtype=np.int64,
            )
        }
        for name in _HISTORY_FLOAT_FIELDS:
            arrays[name] = _readonly_array(
                getattr(self, name),
                name=name,
                shape=(count,),
                dtype=np.float64,
            )
        arrays["positive_linear"] = _readonly_array(
            self.positive_linear,
            name="positive_linear",
            shape=(count,),
            dtype=np.bool_,
        )
        arrays["energies_ev"] = _readonly_array(
            self.energies_ev,
            name="energies_ev",
            shape=(count, 4),
            dtype=np.float64,
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        if not isinstance(self.branches, tuple):
            object.__setattr__(self, "branches", tuple(self.branches))
        if len(self.branches) != count:
            raise ValueError("branches must match history length")
        for branch in self.branches:
            if branch not in (
                "positive_linear",
                "endpoint_quad",
                "linear_near_zero",
                "interior",
            ):
                raise ValueError(f"Unknown ODA branch {branch!r}")
        for name in (
            "old_projector_hashes",
            "raw_projector_hashes",
            "mixed_projector_hashes",
            "hamiltonian_hashes",
            "old_action_fingerprints",
            "dP_action_fingerprints",
            "energy_fingerprints",
        ):
            object.__setattr__(
                self,
                name,
                _tuple_hashes(getattr(self, name), name=name, length=count),
            )
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFSCFHistoryHashes):
            raise TypeError("array_hashes must be typed SCF history hashes")
        self._validate_live_state()

    def _live_arrays(self) -> dict[str, np.ndarray]:
        count = len(self.branches)
        arrays = {
            "iterations": _validate_live_array(
                self.iterations,
                name="history.iterations",
                shape=(count,),
                dtype=np.int64,
            ),
            "positive_linear": _validate_live_array(
                self.positive_linear,
                name="history.positive_linear",
                shape=(count,),
                dtype=np.bool_,
            ),
            "energies_ev": _validate_live_array(
                self.energies_ev,
                name="history.energies_ev",
                shape=(count, 4),
                dtype=np.float64,
            ),
        }
        for name in _HISTORY_FLOAT_FIELDS:
            arrays[name] = _validate_live_array(
                getattr(self, name),
                name=f"history.{name}",
                shape=(count,),
                dtype=np.float64,
            )
        return arrays

    def _validate_live_state(self) -> None:
        arrays = self._live_arrays()
        count = arrays["iterations"].size
        if not np.array_equal(arrays["iterations"], np.arange(count, dtype=np.int64)):
            raise ValueError("history iterations must be consecutive and zero-based")
        if np.any(arrays["differences"] < 0.0):
            raise ValueError("history differences must be nonnegative")
        if not np.array_equal(arrays["lin"], arrays["c1"] + arrays["c01"]):
            raise ValueError("history lin must equal c1+c01 exactly")
        if not np.array_equal(arrays["quad"], 0.5 * arrays["c11"]):
            raise ValueError("history quad must equal 0.5*c11 exactly")
        for index in range(count):
            expected_lambda, expected_branch, expected_positive = companion_oda_branch(
                float(arrays["lin"][index]),
                float(arrays["quad"][index]),
            )
            if arrays["mixing_lambda"][index] != expected_lambda:
                raise ValueError(f"history lambda mismatch at iteration {index}")
            if self.branches[index] != expected_branch:
                raise ValueError(f"history branch mismatch at iteration {index}")
            if bool(arrays["positive_linear"][index]) != expected_positive:
                raise ValueError(f"history positive-linear mismatch at iteration {index}")
        energy_closure = np.abs(
            arrays["energies_ev"][:, 0]
            - np.sum(arrays["energies_ev"][:, 1:], axis=1)
        )
        if np.any(energy_closure > 1.0e-12):
            raise ValueError("history energies do not close in finite-system eV")
        actual_hashes = TBGZeroFieldCompanionHFSCFHistoryHashes.from_arrays(arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("history array_hashes no longer match live arrays")
        for name in (
            "old_projector_hashes",
            "raw_projector_hashes",
            "mixed_projector_hashes",
            "hamiltonian_hashes",
            "old_action_fingerprints",
            "dP_action_fingerprints",
            "energy_fingerprints",
        ):
            _tuple_hashes(getattr(self, name), name=name, length=count)

    @property
    def lambdas(self) -> np.ndarray:
        return self.mixing_lambda

    @property
    def positive_linear_flags(self) -> np.ndarray:
        return self.positive_linear

    @property
    def energies(self) -> np.ndarray:
        return self.energies_ev

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": self.array_hashes.fingerprint,
                "branches": list(self.branches),
                "dP_action_fingerprints": list(self.dP_action_fingerprints),
                "energy_fingerprints": list(self.energy_fingerprints),
                "hamiltonian_hashes": list(self.hamiltonian_hashes),
                "mixed_projector_hashes": list(self.mixed_projector_hashes),
                "old_action_fingerprints": list(self.old_action_fingerprints),
                "old_projector_hashes": list(self.old_projector_hashes),
                "raw_projector_hashes": list(self.raw_projector_hashes),
            }
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFRunArrayHashes:
    initial_projector: str
    reference: str
    final_projector_mixed: str
    final_projector_aufbau: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )

    @classmethod
    def from_arrays(
        cls,
        *,
        initial_projector: np.ndarray,
        reference: np.ndarray,
        final_projector_mixed: np.ndarray,
        final_projector_aufbau: np.ndarray,
    ) -> "TBGZeroFieldCompanionHFSCFRunArrayHashes":
        return cls(
            initial_projector=_canonical_array_sha256(initial_projector),
            reference=_canonical_array_sha256(reference),
            final_projector_mixed=_canonical_array_sha256(final_projector_mixed),
            final_projector_aufbau=_canonical_array_sha256(final_projector_aufbau),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class _TBGZeroFieldCompanionHFSCFTrajectoryReplay:
    """Internal exact-kernel replay result; never a separately trusted receipt."""

    history: TBGZeroFieldCompanionHFSCFHistory
    converged: bool
    convergence_iteration: int | None
    final_projector_mixed: np.ndarray
    final_evaluation: TBGZeroFieldCompanionHFEvaluation
    final_aufbau: TBGZeroFieldCompanionAufbauResult
    closure_difference: float
    array_hashes: TBGZeroFieldCompanionHFSCFRunArrayHashes

@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFSCFRun:
    """Immutable diagnostic run with separate mixed and Aufbau-final states."""

    prepared: TBGZeroFieldCompanionPreparedHFAction
    prepared_fingerprint: str
    spec: TBGZeroFieldCompanionHFSCFSpec
    provenance: TBGZeroFieldCompanionHFSCFProvenance
    initial_projector: np.ndarray
    reference: np.ndarray
    initial_source: Literal["array", "stage5_kivc_seed"]
    stage5_seed: TBGZeroFieldCompanionKIVCSeedResult | None
    stage5_seed_fingerprint: str | None
    history: TBGZeroFieldCompanionHFSCFHistory
    converged: bool
    convergence_iteration: int | None
    final_projector_mixed: np.ndarray
    final_evaluation: TBGZeroFieldCompanionHFEvaluation
    final_aufbau: TBGZeroFieldCompanionAufbauResult
    closure_difference: float
    array_hashes: TBGZeroFieldCompanionHFSCFRunArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        if not isinstance(self.spec, TBGZeroFieldCompanionHFSCFSpec):
            raise TypeError("spec must be TBGZeroFieldCompanionHFSCFSpec")
        if not isinstance(self.provenance, TBGZeroFieldCompanionHFSCFProvenance):
            raise TypeError("provenance must be typed Stage6A provenance")
        if not isinstance(self.history, TBGZeroFieldCompanionHFSCFHistory):
            raise TypeError("history must be typed Stage6A history")
        if not isinstance(self.final_evaluation, TBGZeroFieldCompanionHFEvaluation):
            raise TypeError("final_evaluation must be typed Stage4 evaluation")
        if not isinstance(self.final_aufbau, TBGZeroFieldCompanionAufbauResult):
            raise TypeError("final_aufbau must be typed companion Aufbau")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFSCFRunArrayHashes):
            raise TypeError("array_hashes must be typed Stage6A run hashes")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        object.__setattr__(
            self,
            "closure_difference",
            _finite_real(
                self.closure_difference,
                name="closure_difference",
                nonnegative=True,
            ),
        )

        shape = _hamiltonian_shape(self.prepared)
        for name in ("initial_projector", "reference", "final_projector_mixed"):
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    name=name,
                    shape=shape,
                    dtype=np.complex128,
                ),
            )
        self._validate_live_state()

    def _validate_seed_binding(self) -> None:
        if self.initial_source == "array":
            if self.stage5_seed is not None or self.stage5_seed_fingerprint is not None:
                raise ValueError("array initial source must not carry a Stage5 seed")
            return
        if self.initial_source != "stage5_kivc_seed":
            raise ValueError("initial_source must be 'array' or 'stage5_kivc_seed'")
        if not isinstance(self.stage5_seed, TBGZeroFieldCompanionKIVCSeedResult):
            raise TypeError("stage5_kivc_seed source requires a typed Stage5 seed")
        live_seed_fingerprint = self.stage5_seed.fingerprint
        if self.stage5_seed_fingerprint != live_seed_fingerprint:
            raise ValueError("stage5_seed_fingerprint does not match live Stage5 seed")
        if self.stage5_seed.single_particle_source is not self.prepared.single_particle_source:
            raise ValueError("Stage5 seed must bind the direct Stage4 Stage2 source")
        if not np.array_equal(self.initial_projector, self.stage5_seed.P_stored):
            raise ValueError("initial_projector must exactly equal Stage5 P_stored")

    def _validate_live_state(self) -> None:
        live_prepared_fingerprint = self.prepared.fingerprint
        if self.prepared_fingerprint != live_prepared_fingerprint:
            raise ValueError("run prepared_fingerprint does not match live prepared input")
        self._validate_seed_binding()
        self.history._validate_live_state()
        shape = _hamiltonian_shape(self.prepared)
        arrays = {
            "initial_projector": _validate_live_array(
                self.initial_projector,
                name="run.initial_projector",
                shape=shape,
                dtype=np.complex128,
            ),
            "reference": _validate_live_array(
                self.reference,
                name="run.reference",
                shape=shape,
                dtype=np.complex128,
            ),
            "final_projector_mixed": _validate_live_array(
                self.final_projector_mixed,
                name="run.final_projector_mixed",
                shape=shape,
                dtype=np.complex128,
            ),
            "final_projector_aufbau": self.final_aufbau.projector,
        }
        actual_hashes = TBGZeroFieldCompanionHFSCFRunArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("run array_hashes no longer match live arrays")
        if _max_hermiticity_residual(arrays["initial_projector"]) > (
            TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("initial_projector is materially non-Hermitian")
        if _max_hermiticity_residual(arrays["reference"]) > (
            TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("reference is materially non-Hermitian")
        if _max_hermiticity_residual(arrays["final_projector_mixed"]) > (
            TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("final mixed projector is materially non-Hermitian")

        count = len(self.history.branches)
        if count <= 0 or count > self.spec.HF_itermax:
            raise ValueError("history length is outside the SCF iteration contract")
        if self.history.old_projector_hashes[0] != self.array_hashes.initial_projector:
            raise ValueError("first history old projector is not the initial projector")
        for index in range(1, count):
            if (
                self.history.old_projector_hashes[index]
                != self.history.mixed_projector_hashes[index - 1]
            ):
                raise ValueError("history old/mixed projector hash chain is broken")
        if self.history.mixed_projector_hashes[-1] != self.array_hashes.final_projector_mixed:
            raise ValueError("final mixed projector does not close the history chain")

        convergence_mask = np.asarray(
            (self.history.differences < self.spec.tolerance)
            & (self.history.iterations > self.spec.HF_itermin)
        )
        candidate_indices = np.flatnonzero(convergence_mask)
        if self.converged:
            if candidate_indices.size != 1 or int(candidate_indices[0]) != count - 1:
                raise ValueError("converged history must stop at first strict source match")
            if self.convergence_iteration != count - 1:
                raise ValueError("convergence_iteration must be the final zero-based iteration")
        else:
            if self.convergence_iteration is not None:
                raise ValueError("nonconverged run must have convergence_iteration=None")
            if candidate_indices.size:
                raise ValueError("nonconverged run contains a source convergence match")
            if count != self.spec.HF_itermax:
                raise ValueError("nonconverged run must exhaust HF_itermax")

        self.final_evaluation._validate_live_arrays()
        if self.final_evaluation.prepared is not self.prepared:
            raise ValueError("final_evaluation must retain direct prepared identity")
        if not np.array_equal(
            self.final_evaluation.projector,
            arrays["final_projector_mixed"],
        ):
            raise ValueError("final_evaluation was not rebuilt from final mixed projector")
        if not np.array_equal(self.final_evaluation.reference, arrays["reference"]):
            raise ValueError("final_evaluation reference differs from run reference")
        self.final_aufbau._validate_live_state()
        if self.final_aufbau.prepared is not self.prepared:
            raise ValueError("final_aufbau must retain direct prepared identity")
        if self.final_aufbau.filling != self.spec.filling:
            raise ValueError("final_aufbau filling differs from SCF spec")
        if not np.array_equal(
            self.final_aufbau.hamiltonian_ev,
            self.final_evaluation.H_total_ev,
        ):
            raise ValueError("final Aufbau was not rebuilt from final mixed H/action")
        expected_closure = float(
            np.linalg.norm(
                arrays["final_projector_mixed"] - arrays["final_projector_aufbau"]
            )
            / (self.prepared.params.N1 * self.prepared.params.N2)
        )
        if self.closure_difference != expected_closure:
            raise ValueError("closure_difference does not match final mixed/Aufbau states")
        self._validate_deterministic_replay()

    def _validate_deterministic_replay(self) -> None:
        """Re-execute the exact source kernel and bind every retained receipt."""

        replay = _replay_tbg_zero_field_companion_hf_trajectory(
            self.prepared,
            self.initial_projector,
            self.reference,
            self.spec,
        )
        history_array_names = (
            "iterations",
            "differences",
            "c1",
            "c01",
            "c11",
            "lin",
            "quad",
            "mixing_lambda",
            "positive_linear",
            "energies_ev",
        )
        for name in history_array_names:
            if not np.array_equal(
                getattr(self.history, name),
                getattr(replay.history, name),
            ):
                raise ValueError(
                    f"history.{name} does not match deterministic trajectory replay"
                )
        if self.history.branches != replay.history.branches:
            raise ValueError("history.branches do not match deterministic trajectory replay")
        for name in (
            "old_projector_hashes",
            "raw_projector_hashes",
            "mixed_projector_hashes",
            "hamiltonian_hashes",
            "old_action_fingerprints",
            "dP_action_fingerprints",
            "energy_fingerprints",
        ):
            if getattr(self.history, name) != getattr(replay.history, name):
                raise ValueError(
                    f"history.{name} do not match deterministic trajectory replay"
                )
        if self.history.array_hashes != replay.history.array_hashes:
            raise ValueError(
                "history.array_hashes do not match deterministic trajectory replay"
            )
        if self.converged != replay.converged:
            raise ValueError("converged does not match deterministic trajectory replay")
        if self.convergence_iteration != replay.convergence_iteration:
            raise ValueError(
                "convergence_iteration does not match deterministic trajectory replay"
            )

        final_arrays = (
            ("final_projector_mixed", self.final_projector_mixed, replay.final_projector_mixed),
            (
                "final_evaluation.projector",
                self.final_evaluation.projector,
                replay.final_evaluation.projector,
            ),
            (
                "final_evaluation.reference",
                self.final_evaluation.reference,
                replay.final_evaluation.reference,
            ),
            (
                "final_evaluation.density_delta",
                self.final_evaluation.density_delta,
                replay.final_evaluation.density_delta,
            ),
            (
                "final_evaluation.H_SP_ev",
                self.final_evaluation.H_SP_ev,
                replay.final_evaluation.H_SP_ev,
            ),
            (
                "final_evaluation.H_total_ev",
                self.final_evaluation.H_total_ev,
                replay.final_evaluation.H_total_ev,
            ),
            (
                "final_evaluation.action.density_delta",
                self.final_evaluation.action.density_delta,
                replay.final_evaluation.action.density_delta,
            ),
            (
                "final_evaluation.action.H_D_ev",
                self.final_evaluation.action.H_D_ev,
                replay.final_evaluation.action.H_D_ev,
            ),
            (
                "final_evaluation.action.H_E_ev",
                self.final_evaluation.action.H_E_ev,
                replay.final_evaluation.action.H_E_ev,
            ),
            (
                "final_evaluation.action.H_interaction_ev",
                self.final_evaluation.action.H_interaction_ev,
                replay.final_evaluation.action.H_interaction_ev,
            ),
            (
                "final_evaluation.energy.components_ev",
                self.final_evaluation.energy.components_ev,
                replay.final_evaluation.energy.components_ev,
            ),
            (
                "final_aufbau.hamiltonian_ev",
                self.final_aufbau.hamiltonian_ev,
                replay.final_aufbau.hamiltonian_ev,
            ),
            (
                "final_aufbau.projector",
                self.final_aufbau.projector,
                replay.final_aufbau.projector,
            ),
            (
                "final_aufbau.eigenvalues_ev",
                self.final_aufbau.eigenvalues_ev,
                replay.final_aufbau.eigenvalues_ev,
            ),
            (
                "final_aufbau.fill_indices",
                self.final_aufbau.fill_indices,
                replay.final_aufbau.fill_indices,
            ),
            (
                "final_aufbau.eigenvectors",
                self.final_aufbau.eigenvectors,
                replay.final_aufbau.eigenvectors,
            ),
        )
        for name, actual, expected in final_arrays:
            if not np.array_equal(actual, expected):
                raise ValueError(f"{name} does not match deterministic trajectory replay")
        if self.final_evaluation.fingerprint != replay.final_evaluation.fingerprint:
            raise ValueError(
                "final_evaluation fingerprint does not match deterministic trajectory replay"
            )
        if self.final_aufbau.fingerprint != replay.final_aufbau.fingerprint:
            raise ValueError(
                "final_aufbau fingerprint does not match deterministic trajectory replay"
            )
        if self.closure_difference != replay.closure_difference:
            raise ValueError(
                "closure_difference does not match deterministic trajectory replay"
            )
        if self.array_hashes != replay.array_hashes:
            raise ValueError("run array_hashes do not match deterministic trajectory replay")

    @property
    def P_source_final(self) -> np.ndarray:
        return self.final_projector_mixed

    @property
    def P_aufbau_final(self) -> np.ndarray:
        return self.final_aufbau.projector

    @property
    def iteration_count(self) -> int:
        return len(self.history.branches)

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return _json_sha256(
            {
                "array_hashes": self.array_hashes.fingerprint,
                "closure_difference": self.closure_difference,
                "converged": self.converged,
                "convergence_iteration": self.convergence_iteration,
                "final_aufbau": self.final_aufbau.fingerprint,
                "final_evaluation": self.final_evaluation.fingerprint,
                "history": self.history.fingerprint,
                "initial_source": self.initial_source,
                "prepared_fingerprint": self.prepared_fingerprint,
                "provenance": self.provenance.fingerprint,
                "schema": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
                "spec": self.spec.fingerprint,
                "stage5_seed_fingerprint": self.stage5_seed_fingerprint,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "array_hashes": {
                name: getattr(self.array_hashes, name)
                for name in self.array_hashes.__dataclass_fields__
            },
            "closure_difference": self.closure_difference,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "difference_convention": TBG_ZERO_FIELD_COMPANION_HF_SCF_DIFFERENCE_CONVENTION,
            "fingerprint": self.fingerprint,
            "fill_order": TBG_ZERO_FIELD_COMPANION_HF_SCF_FILL_ORDER,
            "history_fingerprint": self.history.fingerprint,
            "initial_source": self.initial_source,
            "iteration_count": self.iteration_count,
            "prepared_fingerprint": self.prepared_fingerprint,
            "provenance": self.provenance.to_metadata(),
            "schema": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE,
            "spec": self.spec.to_companion_input(),
            "stage5_seed_fingerprint": self.stage5_seed_fingerprint,
            "stored_projector_convention": (
                TBG_ZERO_FIELD_COMPANION_HF_SCF_STORED_PROJECTOR_CONVENTION
            ),
        }


def _resolve_initial_projector(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    initial_projector: np.ndarray | TBGZeroFieldCompanionKIVCSeedResult,
) -> tuple[np.ndarray, Literal["array", "stage5_kivc_seed"], TBGZeroFieldCompanionKIVCSeedResult | None, str | None]:
    if isinstance(initial_projector, TBGZeroFieldCompanionKIVCSeedResult):
        seed = initial_projector
        seed_fingerprint = seed.fingerprint
        if seed.single_particle_source is not prepared.single_particle_source:
            raise ValueError("Stage5 seed must bind the direct Stage4 Stage2 source")
        return seed.P_stored, "stage5_kivc_seed", seed, seed_fingerprint
    return np.asarray(initial_projector), "array", None, None


def _replay_tbg_zero_field_companion_hf_trajectory(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    initial_projector: np.ndarray,
    reference: np.ndarray,
    spec: TBGZeroFieldCompanionHFSCFSpec,
) -> _TBGZeroFieldCompanionHFSCFTrajectoryReplay:
    """Pure deterministic replay of the exact pinned iteration kernel.

    This helper never constructs a public run, so public-run validation can use
    it without recursion.  The builder and ``TBGZeroFieldCompanionHFSCFRun``
    constructor intentionally execute the same helper independently.
    """

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    if not isinstance(spec, TBGZeroFieldCompanionHFSCFSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionHFSCFSpec")
    _ = prepared.fingerprint
    _electron_count(prepared, spec.filling)
    shape = _hamiltonian_shape(prepared)
    initial = np.asarray(initial_projector, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    for name, array in (("initial_projector", initial), ("reference", ref)):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        residual = _max_hermiticity_residual(array)
        if residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
            raise ValueError(f"{name} is materially non-Hermitian: {residual:.6e}")

    Nk = prepared.params.N1 * prepared.params.N2
    old = np.array(initial, copy=True)
    scalar_history: dict[str, list[float]] = {
        name: [] for name in _HISTORY_FLOAT_FIELDS
    }
    iterations: list[int] = []
    branches: list[ODABranch] = []
    positive_linear: list[bool] = []
    energies: list[np.ndarray] = []
    old_projector_hashes: list[str] = []
    raw_projector_hashes: list[str] = []
    mixed_projector_hashes: list[str] = []
    hamiltonian_hashes: list[str] = []
    old_action_fingerprints: list[str] = []
    dP_action_fingerprints: list[str] = []
    energy_fingerprints: list[str] = []
    converged = False
    convergence_iteration: int | None = None

    for iteration in range(spec.HF_itermax):
        old_evaluation = prepared.evaluate(old, ref)
        aufbau = companion_aufbau(
            prepared,
            old_evaluation.H_total_ev,
            filling=spec.filling,
        )
        raw = aufbau.projector
        difference = float(np.linalg.norm(old - raw) / Nk)
        oda = companion_oda_coefficients(
            prepared,
            old,
            raw,
            ref,
            branch_threshold=spec.branch_threshold,
        )
        mixing_lambda = oda.mixing_lambda
        mixed = np.asarray((1.0 - mixing_lambda) * old + mixing_lambda * raw)
        mixed_evaluation = prepared.evaluate(mixed, ref)

        iterations.append(iteration)
        scalar_history["differences"].append(difference)
        scalar_history["c1"].append(oda.c1)
        scalar_history["c01"].append(oda.c01)
        scalar_history["c11"].append(oda.c11)
        scalar_history["lin"].append(oda.lin)
        scalar_history["quad"].append(oda.quad)
        scalar_history["mixing_lambda"].append(mixing_lambda)
        branches.append(oda.branch)
        positive_linear.append(oda.positive_linear)
        energies.append(np.array(mixed_evaluation.energy_components_ev, copy=True))
        old_projector_hashes.append(_canonical_array_sha256(old))
        raw_projector_hashes.append(_canonical_array_sha256(raw))
        mixed_projector_hashes.append(_canonical_array_sha256(mixed))
        hamiltonian_hashes.append(_canonical_array_sha256(old_evaluation.H_total_ev))
        old_action_fingerprints.append(old_evaluation.action.fingerprint)
        dP_action_fingerprints.append(oda.dP_action.fingerprint)
        energy_fingerprints.append(mixed_evaluation.energy.fingerprint)

        old = np.array(mixed, copy=True)
        if difference < spec.tolerance and iteration > spec.HF_itermin:
            converged = True
            convergence_iteration = iteration
            break

    history_arrays: dict[str, np.ndarray] = {
        "iterations": np.asarray(iterations, dtype=np.int64),
        "positive_linear": np.asarray(positive_linear, dtype=np.bool_),
        "energies_ev": np.asarray(energies, dtype=np.float64),
    }
    for name in _HISTORY_FLOAT_FIELDS:
        history_arrays[name] = np.asarray(scalar_history[name], dtype=np.float64)
    history = TBGZeroFieldCompanionHFSCFHistory(
        iterations=history_arrays["iterations"],
        differences=history_arrays["differences"],
        c1=history_arrays["c1"],
        c01=history_arrays["c01"],
        c11=history_arrays["c11"],
        lin=history_arrays["lin"],
        quad=history_arrays["quad"],
        mixing_lambda=history_arrays["mixing_lambda"],
        branches=tuple(branches),
        positive_linear=history_arrays["positive_linear"],
        energies_ev=history_arrays["energies_ev"],
        old_projector_hashes=tuple(old_projector_hashes),
        raw_projector_hashes=tuple(raw_projector_hashes),
        mixed_projector_hashes=tuple(mixed_projector_hashes),
        hamiltonian_hashes=tuple(hamiltonian_hashes),
        old_action_fingerprints=tuple(old_action_fingerprints),
        dP_action_fingerprints=tuple(dP_action_fingerprints),
        energy_fingerprints=tuple(energy_fingerprints),
        array_hashes=TBGZeroFieldCompanionHFSCFHistoryHashes.from_arrays(history_arrays),
    )

    final_projector = np.asarray(old)
    # Source finalization: rebuild from the final mixed projector, never from an
    # iteration-local Hamiltonian/action cache.
    final_evaluation = prepared.evaluate(final_projector, ref)
    final_aufbau = companion_aufbau(
        prepared,
        final_evaluation.H_total_ev,
        filling=spec.filling,
    )
    closure_difference = float(
        np.linalg.norm(final_projector - final_aufbau.projector) / Nk
    )
    hashes = TBGZeroFieldCompanionHFSCFRunArrayHashes.from_arrays(
        initial_projector=initial,
        reference=ref,
        final_projector_mixed=final_projector,
        final_projector_aufbau=final_aufbau.projector,
    )
    return _TBGZeroFieldCompanionHFSCFTrajectoryReplay(
        history=history,
        converged=converged,
        convergence_iteration=convergence_iteration,
        final_projector_mixed=final_projector,
        final_evaluation=final_evaluation,
        final_aufbau=final_aufbau,
        closure_difference=closure_difference,
        array_hashes=hashes,
    )

def run_tbg_zero_field_companion_hf_diagnostic(
    prepared: TBGZeroFieldCompanionPreparedHFAction,
    initial_projector: np.ndarray | TBGZeroFieldCompanionKIVCSeedResult,
    reference: np.ndarray,
    spec: TBGZeroFieldCompanionHFSCFSpec | None = None,
) -> TBGZeroFieldCompanionHFSCFRun:
    """Run the isolated source-faithful Stage6A companion SCF diagnostic."""

    if not isinstance(prepared, TBGZeroFieldCompanionPreparedHFAction):
        raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
    resolved_spec = TBGZeroFieldCompanionHFSCFSpec() if spec is None else spec
    if not isinstance(resolved_spec, TBGZeroFieldCompanionHFSCFSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionHFSCFSpec")
    prepared_fingerprint = prepared.fingerprint
    initial_values, initial_source, stage5_seed, seed_fingerprint = (
        _resolve_initial_projector(prepared, initial_projector)
    )
    shape = _hamiltonian_shape(prepared)
    initial = _readonly_array(
        initial_values,
        name="initial_projector",
        shape=shape,
        dtype=np.complex128,
    )
    ref = _readonly_array(
        reference,
        name="reference",
        shape=shape,
        dtype=np.complex128,
    )
    replay = _replay_tbg_zero_field_companion_hf_trajectory(
        prepared,
        initial,
        ref,
        resolved_spec,
    )
    return TBGZeroFieldCompanionHFSCFRun(
        prepared=prepared,
        prepared_fingerprint=prepared_fingerprint,
        spec=resolved_spec,
        provenance=TBGZeroFieldCompanionHFSCFProvenance(),
        initial_projector=initial,
        reference=ref,
        initial_source=initial_source,
        stage5_seed=stage5_seed,
        stage5_seed_fingerprint=seed_fingerprint,
        history=replay.history,
        converged=replay.converged,
        convergence_iteration=replay.convergence_iteration,
        final_projector_mixed=replay.final_projector_mixed,
        final_evaluation=replay.final_evaluation,
        final_aufbau=replay.final_aufbau,
        closure_difference=replay.closure_difference,
        array_hashes=replay.array_hashes,
    )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFQualifierSpec:
    """Frozen typed-unit thresholds; these never promote a production state."""

    projector_hermiticity_threshold: float = 1.0e-9
    hamiltonian_hermiticity_threshold_ev: float = 1.0e-9
    fermi_tie_threshold_ev: float = 1.0e-12
    minimum_positive_gap_ev: float = 0.0
    filling_tolerance: float = 1.0e-10
    minimum_source_ivc: float = 1.0e-12
    tp_break_tolerance: float = 1.0e-8
    spin_block_rms_tolerance: float = 1.0e-8
    polarization_tolerance: float = 1.0e-8
    minimum_stage5_eq99_projection: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _finite_real(
                getattr(self, name),
                name=name,
                nonnegative=True,
            )
            object.__setattr__(self, name, value)
        if self.projector_hermiticity_threshold > 1.0e-9:
            raise ValueError(
                "projector_hermiticity_threshold may not exceed 1e-9"
            )
        if self.hamiltonian_hermiticity_threshold_ev > 1.0e-9:
            raise ValueError(
                "hamiltonian_hermiticity_threshold_ev may not exceed 1e-9 eV"
            )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


    def to_metadata(self) -> dict[str, float | str]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "fingerprint": self.fingerprint,
        }

@dataclass(frozen=True, slots=True)
class _TBGZeroFieldCompanionHFQualificationPayload:
    """Canonical values recomputed only from a live run and qualifier spec."""

    passed: bool
    checks: tuple[tuple[str, bool], ...]
    maximum_projector_hermiticity_residual: float
    maximum_hamiltonian_hermiticity_residual_ev: float
    occupied_count: int
    expected_occupied_count: int
    local_occupations: np.ndarray
    gap_ev: float
    fermi_tie: bool
    nu: float
    nu_residual: float
    source_ivc: float
    tp_break: float
    flavor_occupations: np.ndarray
    valley_polarization: float
    spin_polarization: float
    spin_block_rms_difference: float
    stage5_eq99_projection_magnitude: float | None
    local_occupations_sha256: str
    flavor_occupations_sha256: str

_QUALIFICATION_FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "maximum_projector_hermiticity_residual",
    "maximum_hamiltonian_hermiticity_residual_ev",
    "gap_ev",
    "nu",
    "nu_residual",
    "source_ivc",
    "tp_break",
    "valley_polarization",
    "spin_polarization",
    "spin_block_rms_difference",
)
_QUALIFICATION_FLOAT_MATCH_RTOL: Final[float] = 1.0e-13
_QUALIFICATION_FLOAT_MATCH_ATOL: Final[float] = 1.0e-15

@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFQualificationReport:
    """Fail-closed diagnostic report; never a production/ground-state verdict."""

    run: TBGZeroFieldCompanionHFSCFRun
    run_fingerprint: str
    spec: TBGZeroFieldCompanionHFQualifierSpec
    passed: bool
    checks: tuple[tuple[str, bool], ...]
    maximum_projector_hermiticity_residual: float
    maximum_hamiltonian_hermiticity_residual_ev: float
    occupied_count: int
    expected_occupied_count: int
    local_occupations: np.ndarray
    gap_ev: float
    fermi_tie: bool
    nu: float
    nu_residual: float
    source_ivc: float
    tp_break: float
    flavor_occupations: np.ndarray
    valley_polarization: float
    spin_polarization: float
    spin_block_rms_difference: float
    stage5_eq99_projection_magnitude: float | None
    local_occupations_sha256: str
    flavor_occupations_sha256: str
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.run, TBGZeroFieldCompanionHFSCFRun):
            raise TypeError("run must be TBGZeroFieldCompanionHFSCFRun")
        if not isinstance(self.spec, TBGZeroFieldCompanionHFQualifierSpec):
            raise TypeError("spec must be TBGZeroFieldCompanionHFQualifierSpec")
        object.__setattr__(
            self,
            "run_fingerprint",
            _validate_sha256(self.run_fingerprint, name="run_fingerprint"),
        )
        if self.run_fingerprint != self.run.fingerprint:
            raise ValueError("qualification run_fingerprint does not match live run")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if not isinstance(self.checks, tuple):
            object.__setattr__(self, "checks", tuple(self.checks))
        check_names: set[str] = set()
        for record in self.checks:
            if not isinstance(record, tuple) or len(record) != 2:
                raise TypeError("qualification checks must be (name, bool) tuples")
            name, value = record
            if not isinstance(name, str) or not name or name in check_names:
                raise ValueError("qualification check names must be unique")
            if not isinstance(value, bool):
                raise TypeError("qualification check values must be bool")
            check_names.add(name)
        for name in _QUALIFICATION_FLOAT_FIELDS:
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name),
            )
        if self.stage5_eq99_projection_magnitude is not None:
            object.__setattr__(
                self,
                "stage5_eq99_projection_magnitude",
                _finite_real(
                    self.stage5_eq99_projection_magnitude,
                    name="stage5_eq99_projection_magnitude",
                    nonnegative=True,
                ),
            )
        if not isinstance(self.fermi_tie, bool):
            raise TypeError("fermi_tie must be bool")
        object.__setattr__(
            self,
            "occupied_count",
            _strict_int(self.occupied_count, name="occupied_count"),
        )
        object.__setattr__(
            self,
            "expected_occupied_count",
            _strict_int(
                self.expected_occupied_count,
                name="expected_occupied_count",
            ),
        )
        params = self.run.prepared.params
        object.__setattr__(
            self,
            "local_occupations",
            _readonly_array(
                self.local_occupations,
                name="local_occupations",
                shape=(params.N1, params.N2, 2),
                dtype=np.int64,
            ),
        )
        object.__setattr__(
            self,
            "flavor_occupations",
            _readonly_array(
                self.flavor_occupations,
                name="flavor_occupations",
                shape=(2, 2),
                dtype=np.float64,
            ),
        )
        for name in (
            "local_occupations_sha256",
            "flavor_occupations_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )
        if self.scientific_scope != TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE:
            raise ValueError("qualification scope may not promote production/TDHF/Fig8")

        payload = _compute_qualification_payload(self.run, self.spec)
        _validate_qualification_report_payload(self, payload)
        # Canonicalize all accepted tight-float values to the live recomputation,
        # so one run/spec pair has exactly one report fingerprint.
        for name in _QUALIFICATION_FLOAT_FIELDS:
            object.__setattr__(self, name, getattr(payload, name))
        for name in (
            "passed",
            "checks",
            "occupied_count",
            "expected_occupied_count",
            "fermi_tie",
            "stage5_eq99_projection_magnitude",
            "local_occupations_sha256",
            "flavor_occupations_sha256",
        ):
            object.__setattr__(self, name, getattr(payload, name))
        object.__setattr__(
            self,
            "local_occupations",
            _readonly_array(
                payload.local_occupations,
                name="local_occupations",
                shape=(params.N1, params.N2, 2),
                dtype=np.int64,
            ),
        )
        object.__setattr__(
            self,
            "flavor_occupations",
            _readonly_array(
                payload.flavor_occupations,
                name="flavor_occupations",
                shape=(2, 2),
                dtype=np.float64,
            ),
        )

    @property
    def fingerprint(self) -> str:
        if self.run.fingerprint != self.run_fingerprint:
            raise ValueError("qualification live run binding drifted")
        payload = _compute_qualification_payload(self.run, self.spec)
        _validate_qualification_report_payload(self, payload)
        return _json_sha256(
            {
                "checks": list(self.checks),
                "fermi_tie": self.fermi_tie,
                "flavor_occupations_sha256": self.flavor_occupations_sha256,
                "local_occupations_sha256": self.local_occupations_sha256,
                "metrics": {
                    "expected_occupied_count": self.expected_occupied_count,
                    "gap_ev": self.gap_ev,
                    "maximum_hamiltonian_hermiticity_residual_ev": (
                        self.maximum_hamiltonian_hermiticity_residual_ev
                    ),
                    "maximum_projector_hermiticity_residual": (
                        self.maximum_projector_hermiticity_residual
                    ),
                    "nu": self.nu,
                    "nu_residual": self.nu_residual,
                    "occupied_count": self.occupied_count,
                    "source_ivc": self.source_ivc,
                    "spin_block_rms_difference": self.spin_block_rms_difference,
                    "spin_polarization": self.spin_polarization,
                    "stage5_eq99_projection_magnitude": (
                        self.stage5_eq99_projection_magnitude
                    ),
                    "tp_break": self.tp_break,
                    "valley_polarization": self.valley_polarization,
                },
                "passed": self.passed,
                "run_fingerprint": self.run_fingerprint,
                "scope": self.scientific_scope,
                "spec": self.spec.fingerprint,
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "checks": dict(self.checks),
            "fingerprint": self.fingerprint,
            "flavor_occupations_sha256": self.flavor_occupations_sha256,
            "local_occupations_sha256": self.local_occupations_sha256,
            "metrics": {
                "expected_occupied_count": self.expected_occupied_count,
                "gap_ev": self.gap_ev,
                "fermi_tie": self.fermi_tie,
                "maximum_hamiltonian_hermiticity_residual_ev": (
                    self.maximum_hamiltonian_hermiticity_residual_ev
                ),
                "maximum_projector_hermiticity_residual": (
                    self.maximum_projector_hermiticity_residual
                ),
                "nu": self.nu,
                "nu_residual": self.nu_residual,
                "occupied_count": self.occupied_count,
                "source_ivc": self.source_ivc,
                "spin_block_rms_difference": self.spin_block_rms_difference,
                "spin_polarization": self.spin_polarization,
                "stage5_eq99_projection_magnitude": (
                    self.stage5_eq99_projection_magnitude
                ),
                "tp_break": self.tp_break,
                "valley_polarization": self.valley_polarization,
            },
            "passed": self.passed,
            "run_fingerprint": self.run_fingerprint,
            "scope": self.scientific_scope,
            "spec": self.spec.to_metadata(),
            "spec_fingerprint": self.spec.fingerprint,
        }


def _companion_measure_boost0_tp(projector: np.ndarray, *, nactive: int) -> np.ndarray:
    N1, N2 = projector.shape[:2]
    Pex = np.reshape(
        projector,
        (N1, N2, 2, 2, 2 * nactive, 2, 2 * nactive),
        order="C",
    ).copy()
    P_T = np.flip(Pex, axis=(0, 1, 3, 5)).copy()
    P_T = np.roll(P_T, (1, 1), axis=(0, 1))
    P_T = np.conj(P_T)
    P_Tp = P_T.copy()
    P_Tp[:, :, :, 0, :, 1, :] = -P_T[:, :, :, 0, :, 1, :]
    P_Tp[:, :, :, 1, :, 0, :] = -P_T[:, :, :, 1, :, 0, :]
    return np.reshape(P_Tp, projector.shape, order="C")


def _compute_qualification_payload(
    run: TBGZeroFieldCompanionHFSCFRun,
    spec: TBGZeroFieldCompanionHFQualifierSpec,
) -> _TBGZeroFieldCompanionHFQualificationPayload:
    """Pure canonical qualification computation from a live run and spec."""

    if not isinstance(run, TBGZeroFieldCompanionHFSCFRun):
        raise TypeError("run must be TBGZeroFieldCompanionHFSCFRun")
    if not isinstance(spec, TBGZeroFieldCompanionHFQualifierSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionHFQualifierSpec")
    _ = run.fingerprint
    params = run.prepared.params
    Nk = params.N1 * params.N2
    nactive = params.n_active
    P = run.final_projector_mixed
    aufbau = run.final_aufbau

    fill_mask = np.zeros(aufbau.eigenvalues_ev.size, dtype=bool)
    fill_mask[aufbau.fill_indices] = True
    occupied_count = int(np.count_nonzero(fill_mask))
    local_occupations = np.ascontiguousarray(
        np.sum(
            np.reshape(
                fill_mask,
                (params.N1, params.N2, 2, 4 * nactive),
                order="C",
            ),
            axis=-1,
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    if occupied_count == 0 or occupied_count == fill_mask.size:
        gap = 0.0
        fermi_tie = True
    else:
        occupied_max = float(np.max(aufbau.eigenvalues_ev[fill_mask]))
        empty_min = float(np.min(aufbau.eigenvalues_ev[~fill_mask]))
        gap = empty_min - occupied_max
        fermi_tie = gap <= spec.fermi_tie_threshold_ev

    Pex = np.reshape(
        P,
        (params.N1, params.N2, 2, 2, 2 * nactive, 2, 2 * nactive),
        order="C",
    )
    flavor_occupations = np.ascontiguousarray(
        np.real(np.einsum("kKstata->st", Pex, optimize=True) / Nk),
        dtype=np.float64,
    )
    spin_polarization = float(
        flavor_occupations[0, 0]
        + flavor_occupations[0, 1]
        - flavor_occupations[1, 0]
        - flavor_occupations[1, 1]
    )
    valley_polarization = float(
        flavor_occupations[0, 0]
        + flavor_occupations[1, 0]
        - flavor_occupations[0, 1]
        - flavor_occupations[1, 1]
    )
    source_ivc = float(np.linalg.norm(Pex[:, :, :, 0, :, 1, :]) ** 2 / Nk)
    P_Tp = _companion_measure_boost0_tp(P, nactive=nactive)
    tp_break = float(np.linalg.norm(P - P_Tp) ** 2 / Nk)
    spin_block_rms_difference = float(
        np.linalg.norm(P[:, :, 0] - P[:, :, 1]) / math.sqrt(Nk)
    )
    trace = float(np.real(np.einsum("kKsaa->", P, optimize=True)))
    nu = (trace - 4 * nactive * Nk) / Nk
    nu_residual = abs(nu - run.spec.filling)

    maximum_projector_hermiticity_residual = max(
        _max_hermiticity_residual(P),
        _max_hermiticity_residual(run.reference),
        _max_hermiticity_residual(aufbau.projector),
    )
    maximum_hamiltonian_hermiticity_residual_ev = max(
        _max_hermiticity_residual(run.final_evaluation.H_total_ev),
        _max_hermiticity_residual(run.final_evaluation.action.H_D_ev),
        _max_hermiticity_residual(run.final_evaluation.action.H_E_ev),
    )
    finite = all(
        np.all(np.isfinite(array))
        for array in (
            P,
            run.reference,
            run.final_evaluation.H_total_ev,
            aufbau.eigenvalues_ev,
            run.history.energies_ev,
        )
    )

    stage5_projection: float | None = None
    if run.stage5_seed is not None:
        identity = np.eye(4 * nactive, dtype=np.complex128)
        seed_q = 2.0 * run.stage5_seed.P_stored - identity
        final_q = 2.0 * P - identity
        denominator = float(np.real(np.vdot(seed_q, seed_q)))
        stage5_projection = (
            0.0
            if denominator == 0.0
            else float(abs(np.vdot(seed_q, final_q)) / denominator)
        )

    expected_count = Nk * (4 * nactive + run.spec.filling)
    checks = (
        ("finite", bool(finite)),
        (
            "projector_hermiticity",
            maximum_projector_hermiticity_residual
            <= spec.projector_hermiticity_threshold,
        ),
        (
            "hamiltonian_hermiticity_ev",
            maximum_hamiltonian_hermiticity_residual_ev
            <= spec.hamiltonian_hermiticity_threshold_ev,
        ),
        ("converged", run.converged),
        ("closure_lt_scf_tolerance", run.closure_difference < run.spec.tolerance),
        ("filling_zero", run.spec.filling == 0),
        ("occupied_count", occupied_count == expected_count),
        (
            "local_two_at_filling_zero",
            run.spec.filling == 0 and bool(np.all(local_occupations == 2)),
        ),
        ("positive_gap", gap > spec.minimum_positive_gap_ev),
        ("no_fermi_tie", not fermi_tie),
        ("nu_matches_filling", nu_residual <= spec.filling_tolerance),
        ("source_ivc", source_ivc >= spec.minimum_source_ivc),
        ("boost0_measure_tp", tp_break <= spec.tp_break_tolerance),
        (
            "spin_block_rms_match",
            spin_block_rms_difference <= spec.spin_block_rms_tolerance,
        ),
        (
            "valley_polarization",
            abs(valley_polarization) <= spec.polarization_tolerance,
        ),
        (
            "spin_polarization",
            abs(spin_polarization) <= spec.polarization_tolerance,
        ),
        (
            "stage5_eq99_projection",
            stage5_projection is None
            or stage5_projection >= spec.minimum_stage5_eq99_projection,
        ),
    )
    return _TBGZeroFieldCompanionHFQualificationPayload(
        passed=bool(all(value for _name, value in checks)),
        checks=checks,
        maximum_projector_hermiticity_residual=(
            maximum_projector_hermiticity_residual
        ),
        maximum_hamiltonian_hermiticity_residual_ev=(
            maximum_hamiltonian_hermiticity_residual_ev
        ),
        occupied_count=occupied_count,
        expected_occupied_count=expected_count,
        local_occupations=local_occupations,
        gap_ev=gap,
        fermi_tie=fermi_tie,
        nu=nu,
        nu_residual=nu_residual,
        source_ivc=source_ivc,
        tp_break=tp_break,
        flavor_occupations=flavor_occupations,
        valley_polarization=valley_polarization,
        spin_polarization=spin_polarization,
        spin_block_rms_difference=spin_block_rms_difference,
        stage5_eq99_projection_magnitude=stage5_projection,
        local_occupations_sha256=_canonical_array_sha256(local_occupations),
        flavor_occupations_sha256=_canonical_array_sha256(flavor_occupations),
    )

def _qualification_float_matches(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=_QUALIFICATION_FLOAT_MATCH_RTOL,
        abs_tol=_QUALIFICATION_FLOAT_MATCH_ATOL,
    )

def _validate_qualification_report_payload(
    report: TBGZeroFieldCompanionHFQualificationReport,
    payload: _TBGZeroFieldCompanionHFQualificationPayload,
) -> None:
    if report.passed != payload.passed:
        raise ValueError("qualification passed flag does not match live run")
    if report.checks != payload.checks:
        raise ValueError("qualification checks do not match live run")
    if report.fermi_tie != payload.fermi_tie:
        raise ValueError("qualification fermi_tie does not match live run")
    for name in ("occupied_count", "expected_occupied_count"):
        if getattr(report, name) != getattr(payload, name):
            raise ValueError(f"qualification {name} does not match live run")
    for name in _QUALIFICATION_FLOAT_FIELDS:
        if not _qualification_float_matches(
            getattr(report, name),
            getattr(payload, name),
        ):
            raise ValueError(f"qualification {name} does not match live run")
    actual_stage5 = report.stage5_eq99_projection_magnitude
    expected_stage5 = payload.stage5_eq99_projection_magnitude
    if (actual_stage5 is None) != (expected_stage5 is None):
        raise ValueError(
            "qualification stage5_eq99_projection_magnitude does not match live run"
        )
    if (
        actual_stage5 is not None
        and expected_stage5 is not None
        and not _qualification_float_matches(actual_stage5, expected_stage5)
    ):
        raise ValueError(
            "qualification stage5_eq99_projection_magnitude does not match live run"
        )
    occupation_bindings = (
        ("local_occupations", "local_occupations_sha256"),
        ("flavor_occupations", "flavor_occupations_sha256"),
    )
    for array_name, hash_name in occupation_bindings:
        array = getattr(report, array_name)
        if _canonical_array_sha256(array) != getattr(report, hash_name):
            raise ValueError(f"qualification {array_name} hash drifted")
        if not np.array_equal(array, getattr(payload, array_name)):
            raise ValueError(f"qualification {array_name} do not match live run")
        if getattr(report, hash_name) != getattr(payload, hash_name):
            raise ValueError(f"qualification {hash_name} does not match live run")

def qualify_tbg_zero_field_companion_hf_diagnostic(
    run: TBGZeroFieldCompanionHFSCFRun,
    spec: TBGZeroFieldCompanionHFQualifierSpec | None = None,
) -> TBGZeroFieldCompanionHFQualificationReport:
    """Report Stage6A diagnostic gates without promoting any global state."""

    if not isinstance(run, TBGZeroFieldCompanionHFSCFRun):
        raise TypeError("run must be TBGZeroFieldCompanionHFSCFRun")
    resolved_spec = TBGZeroFieldCompanionHFQualifierSpec() if spec is None else spec
    if not isinstance(resolved_spec, TBGZeroFieldCompanionHFQualifierSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionHFQualifierSpec")
    run_fingerprint = run.fingerprint
    payload = _compute_qualification_payload(run, resolved_spec)
    return TBGZeroFieldCompanionHFQualificationReport(
        run=run,
        run_fingerprint=run_fingerprint,
        spec=resolved_spec,
        passed=payload.passed,
        checks=payload.checks,
        maximum_projector_hermiticity_residual=(
            payload.maximum_projector_hermiticity_residual
        ),
        maximum_hamiltonian_hermiticity_residual_ev=(
            payload.maximum_hamiltonian_hermiticity_residual_ev
        ),
        occupied_count=payload.occupied_count,
        expected_occupied_count=payload.expected_occupied_count,
        local_occupations=payload.local_occupations,
        gap_ev=payload.gap_ev,
        fermi_tie=payload.fermi_tie,
        nu=payload.nu,
        nu_residual=payload.nu_residual,
        source_ivc=payload.source_ivc,
        tp_break=payload.tp_break,
        flavor_occupations=payload.flavor_occupations,
        valley_polarization=payload.valley_polarization,
        spin_polarization=payload.spin_polarization,
        spin_block_rms_difference=payload.spin_block_rms_difference,
        stage5_eq99_projection_magnitude=(
            payload.stage5_eq99_projection_magnitude
        ),
        local_occupations_sha256=payload.local_occupations_sha256,
        flavor_occupations_sha256=payload.flavor_occupations_sha256,
    )


# Discoverable module-local aliases; none is added to the zero-field package
# front door.
TBGZeroFieldCompanionHFSCFQualifierSpec = TBGZeroFieldCompanionHFQualifierSpec
TBGZeroFieldCompanionHFSCFQualificationReport = (
    TBGZeroFieldCompanionHFQualificationReport
)
TBGZeroFieldCompanionHFSCFResult = TBGZeroFieldCompanionHFSCFRun


__all__ = [
    "TBGZeroFieldCompanionAufbauArrayHashes",
    "TBGZeroFieldCompanionAufbauResult",
    "TBGZeroFieldCompanionHFQualificationReport",
    "TBGZeroFieldCompanionHFQualifierSpec",
    "TBGZeroFieldCompanionHFSCFHistory",
    "TBGZeroFieldCompanionHFSCFHistoryHashes",
    "TBGZeroFieldCompanionHFSCFProvenance",
    "TBGZeroFieldCompanionHFSCFQualificationReport",
    "TBGZeroFieldCompanionHFSCFQualifierSpec",
    "TBGZeroFieldCompanionHFSCFResult",
    "TBGZeroFieldCompanionHFSCFRun",
    "TBGZeroFieldCompanionHFSCFRunArrayHashes",
    "TBGZeroFieldCompanionHFSCFSpec",
    "TBGZeroFieldCompanionODACoefficients",
    "TBG_ZERO_FIELD_COMPANION_AUFBAU_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_AVERAGE_CENTRAL_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_ARRAY_HASH_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_CONVERGENCE_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_DIFFERENCE_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_ENERGY_UNITS",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_FILL_ORDER",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_SOURCE_PARITY_EXCEPTION",
    "TBG_ZERO_FIELD_COMPANION_HF_SCF_STORED_PROJECTOR_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_MAIN_ODA_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_MAIN_SCF_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_MEASURE_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_ODA_BRANCH_THRESHOLD",
    "build_companion_average_central_reference",
    "companion_aufbau",
    "companion_oda_branch",
    "companion_oda_coefficients",
    "qualify_tbg_zero_field_companion_hf_diagnostic",
    "run_tbg_zero_field_companion_hf_diagnostic",
]
