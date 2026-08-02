"""Pinned-source companion form-factor and Hartree--Fock action diagnostic.

This module ports only the dense action in ``reference/TBG-HF``.  It consumes
the typed Stage-2 companion single-particle result and Stage-3 raw interaction,
but it is intentionally not exported from :mod:`mean_field.systems.tbg.zero_field`
and is not connected to production HF or TDHF.  There is no SCF or Aufbau path
here.

The executable source, rather than its occasionally stale shape comments, is
the authority.  In particular, this port preserves the temporary
``G in [-NG-1, NG]`` form-factor container, roll followed by parent-edge
zero-fill, carry-dependent slices, final reciprocal-axis flip, intravalley
form factors, ``M = intFT * form.conj()``, source roll/reshape order for
``tVE``, the stored projector orientation ``<c^dagger_alpha c_beta>``, and
finite-system eV energy contractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Final, Literal

import numpy as np

from .companion_geometry import (
    TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
    TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
)
from .companion_interaction import (
    TBGZeroFieldCompanionInteractionArrayHashes,
    TBGZeroFieldCompanionInteractionResult,
)
from .companion_single_particle import (
    TBGZeroFieldCompanionSingleParticleArrayHashes,
    TBGZeroFieldCompanionSingleParticleParams,
    TBGZeroFieldCompanionSingleParticleResult,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
    TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
    TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
    TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256,
)

TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA: Final[str] = (
    "mean_field.tbg.zero_field.companion_hf_action"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION: Final[int] = 1
TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE: Final[str] = (
    "diagnostic_form_factor_HF_action_parity_only_not_production_SCF_HF_or_TDHF"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_CONVENTION: Final[str] = (
    "sha256_little_endian_int64_shape_then_C_order_canonical_float64_or_complex128_bytes"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS: Final[str] = (
    "artifact_integrity_only_not_cross_eigensolver_coefficient_gauge_parity"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_REAL_THRESHOLD: Final[float] = 1.0e-9
TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_SOURCE_ATOL: Final[float] = 2.0e-14
TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD: Final[float] = 1.0e-9
TBG_ZERO_FIELD_COMPANION_HF_ACTION_IMAG_ENERGY_THRESHOLD_EV: Final[float] = 1.0e-9
TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_CLOSURE_ATOL_EV: Final[float] = 1.0e-12
TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS: Final[str] = (
    "finite_system_eV_not_per_moire_cell"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_STORED_PROJECTOR_CONVENTION: Final[str] = (
    "P[k1,k2,spin,alpha,beta]=<c_dagger_alpha(k)_spin c_beta(k)_spin>"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCREENING_CONVENTION: Final[str] = (
    "Stage3_intFT_ev_is_raw_and_is_divided_by_epsr_exactly_once_during_prepare"
)
TBG_ZERO_FIELD_COMPANION_HF_ACTION_BOOST_CONVENTION: Final[str] = (
    "boost1=boost2=0_only"
)

TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE: Final[str] = "routines.py"
TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256: Final[str] = (
    "507e8b9e799f494777d354c9d7d481dd19d6ba42894d393630dd79ef16d02108"
)
TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE: Final[str] = "mainProgram.py"
TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256: Final[str] = (
    "258c97e57164055de3273ba4471cd96be709c1f159e19f73481750c801aed401"
)
TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE: Final[str] = "HF_input.json"
TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256: Final[str] = (
    "d577afffdf80a05a348394c5b813540b8074107fc765454f6e50066d420e25e8"
)

TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES: Final[str] = "389-440"
TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES: Final[str] = "6-22"
TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES: Final[str] = "24-79"
TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES: Final[str] = "81-97"
TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES: Final[str] = "99-153"
TBG_ZERO_FIELD_COMPANION_MAIN_SCREENING_REFERENCE_LINE: Final[str] = "31"
TBG_ZERO_FIELD_COMPANION_MAIN_FORM_REFERENCE_LINES: Final[str] = "33-43"
TBG_ZERO_FIELD_COMPANION_MAIN_BOOST_REFERENCE_LINES: Final[str] = "47-54"


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer (bool is not accepted), got {value!r}")
    return int(value)


def _strict_positive_int(value: object, *, name: str) -> int:
    resolved = _strict_int(value, name=name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return resolved


def _finite_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a finite positive real scalar (bool is not accepted), "
            f"got {value!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return resolved


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real scalar, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return resolved


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hexadecimal string")
    resolved = value.strip().lower()
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
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


def _canonical_array(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        dtype = np.dtype("<c16")
    elif source.dtype.kind in "fiub":
        dtype = np.dtype("<f8")
    else:
        raise TypeError(f"Unsupported array dtype for canonical hashing: {source.dtype}")
    return np.ascontiguousarray(source, dtype=dtype)


def _canonical_array_sha256(values: np.ndarray) -> str:
    array = _canonical_array(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _copy_finite_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    source = np.asarray(values)
    if dtype is None:
        if source.dtype.kind == "c":
            resolved_dtype: np.dtype | type = np.complex128
        elif source.dtype.kind in "fiub":
            resolved_dtype = np.float64
        else:
            raise TypeError(f"{name} must be a real or complex numerical array")
    else:
        resolved_dtype = dtype
    array = np.array(source, dtype=resolved_dtype, order="C", copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _validate_live_array_layout(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    """Validate the exact immutable layout used by a hash-bound live array."""

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

def _max_abs_imag(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0 or array.dtype.kind != "c":
        return 0.0
    return float(np.max(np.abs(np.imag(array))))


def _max_hermiticity_residual(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array - np.swapaxes(array.conj(), -1, -2))))


def _form_shape(
    params: TBGZeroFieldCompanionSingleParticleParams,
    *,
    NG1: int,
    NG2: int,
) -> tuple[int, ...]:
    bands = params.active_band_count
    return (
        params.N1,
        params.N2,
        params.N1,
        params.N2,
        2 * NG1,
        2 * NG2,
        2,
        bands,
        bands,
    )


def _hamiltonian_shape(
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> tuple[int, ...]:
    dimension = 4 * params.n_active
    return (params.N1, params.N2, 2, dimension, dimension)


def _tve_shape(
    params: TBGZeroFieldCompanionSingleParticleParams,
) -> tuple[int, ...]:
    flattened = params.N1 * params.N2 * 4 * params.n_active**2
    return (flattened, flattened, 4)


def _validate_cutoffs(
    params: TBGZeroFieldCompanionSingleParticleParams,
    *,
    NG1: object,
    NG2: object,
) -> tuple[int, int]:
    resolved_NG1 = _strict_positive_int(NG1, name="NG1")
    resolved_NG2 = _strict_positive_int(NG2, name="NG2")
    # This is the executable source guard.  Equality with 2*Ng-1 is allowed.
    if resolved_NG1 > 2 * params.Ng1 - 1 or resolved_NG2 > 2 * params.Ng2 - 1:
        raise ValueError(
            "Companion gen_form_factors requires componentwise "
            "NG1 <= 2*Ng1-1 and NG2 <= 2*Ng2-1"
        )
    return resolved_NG1, resolved_NG2


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFActionSpec:
    """Pinned HF-action controls; only zero intervalley boost is supported."""

    epsr: float = 10.0
    exchange: bool = True
    boost1: int = 0
    boost2: int = 0

    def __post_init__(self) -> None:
        epsr = _finite_positive_real(self.epsr, name="epsr")
        if not isinstance(self.exchange, bool):
            raise TypeError("exchange must be bool")
        boost1 = _strict_int(self.boost1, name="boost1")
        boost2 = _strict_int(self.boost2, name="boost2")
        if boost1 != 0 or boost2 != 0:
            raise ValueError("Stage4 companion HF action supports only boost1=boost2=0")
        object.__setattr__(self, "epsr", epsr)
        object.__setattr__(self, "boost1", boost1)
        object.__setattr__(self, "boost2", boost2)

    def to_companion_input(self) -> dict[str, float | bool | int]:
        return {
            "epsr": self.epsr,
            "exchange": self.exchange,
            "boost1": self.boost1,
            "boost2": self.boost2,
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "boost_convention": TBG_ZERO_FIELD_COMPANION_HF_ACTION_BOOST_CONVENTION,
                "input": self.to_companion_input(),
                "schema": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA,
                "schema_version": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION,
                "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
                "screening_convention": (
                    TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCREENING_CONVENTION
                ),
            }
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "boost_convention": TBG_ZERO_FIELD_COMPANION_HF_ACTION_BOOST_CONVENTION,
            "fingerprint": self.fingerprint,
            "input": self.to_companion_input(),
            "schema": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
            "screening_convention": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCREENING_CONVENTION,
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFActionProvenance:
    """Hash-pinned executable source identity for the isolated Stage4 port."""

    reference_repository: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY
    reference_commit: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT
    single_particle_source: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE
    single_particle_source_sha256: str = TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256
    routines_source: str = TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE
    routines_source_sha256: str = TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256
    main_program_source: str = TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE
    main_program_source_sha256: str = TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256
    constants_source: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE
    constants_source_sha256: str = TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256
    default_input_source: str = TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE
    default_input_source_sha256: str = TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256
    hf_input_source: str = TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE
    hf_input_source_sha256: str = TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256
    form_factor_reference_lines: str = TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES
    gen_H_SP_reference_lines: str = TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES
    gen_M_tVE_reference_lines: str = TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES
    calc_E_reference_lines: str = TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES
    calc_fock_matrix_reference_lines: str = (
        TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES
    )
    main_screening_reference_line: str = TBG_ZERO_FIELD_COMPANION_MAIN_SCREENING_REFERENCE_LINE
    main_form_reference_lines: str = TBG_ZERO_FIELD_COMPANION_MAIN_FORM_REFERENCE_LINES
    main_boost_reference_lines: str = TBG_ZERO_FIELD_COMPANION_MAIN_BOOST_REFERENCE_LINES
    scientific_scope: str = TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE

    def __post_init__(self) -> None:
        expected = {
            "reference_repository": TBG_ZERO_FIELD_COMPANION_REFERENCE_REPOSITORY,
            "reference_commit": TBG_ZERO_FIELD_COMPANION_REFERENCE_COMMIT,
            "single_particle_source": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE,
            "single_particle_source_sha256": TBG_ZERO_FIELD_COMPANION_REFERENCE_SOURCE_SHA256,
            "routines_source": TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE,
            "routines_source_sha256": TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256,
            "main_program_source": TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE,
            "main_program_source_sha256": TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256,
            "constants_source": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE,
            "constants_source_sha256": TBG_ZERO_FIELD_COMPANION_CONSTANTS_SOURCE_SHA256,
            "default_input_source": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE,
            "default_input_source_sha256": TBG_ZERO_FIELD_COMPANION_DEFAULT_INPUT_SOURCE_SHA256,
            "hf_input_source": TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE,
            "hf_input_source_sha256": TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256,
            "form_factor_reference_lines": TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES,
            "gen_H_SP_reference_lines": TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES,
            "gen_M_tVE_reference_lines": TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES,
            "calc_E_reference_lines": TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES,
            "calc_fock_matrix_reference_lines": (
                TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES
            ),
            "main_screening_reference_line": (
                TBG_ZERO_FIELD_COMPANION_MAIN_SCREENING_REFERENCE_LINE
            ),
            "main_form_reference_lines": TBG_ZERO_FIELD_COMPANION_MAIN_FORM_REFERENCE_LINES,
            "main_boost_reference_lines": TBG_ZERO_FIELD_COMPANION_MAIN_BOOST_REFERENCE_LINES,
            "scientific_scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
        }
        for name, pinned in expected.items():
            if getattr(self, name) != pinned:
                raise ValueError(f"{name} differs from pinned companion HF-action provenance")

    def to_metadata(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_metadata())


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFMemoryEstimate:
    """Logical array storage; NumPy/einsum temporaries can raise peak RSS."""

    form_elements: int
    tVE_elements: int
    hamiltonian_elements: int
    legacy_form_scratch_elements: int
    raw_form_bytes: int
    effective_form_bytes: int
    M_bytes: int
    tVE_bytes: int
    H_SP_bytes: int
    screened_intFT_bytes: int
    sp_energy_bytes: int
    prepared_arrays_bytes: int
    worst_case_complex128_prepared_bytes: int
    form_branch: Literal["real", "complex"]
    exchange: bool

    def __post_init__(self) -> None:
        for name in (
            "form_elements",
            "tVE_elements",
            "hamiltonian_elements",
            "legacy_form_scratch_elements",
            "raw_form_bytes",
            "effective_form_bytes",
            "M_bytes",
            "tVE_bytes",
            "H_SP_bytes",
            "screened_intFT_bytes",
            "sp_energy_bytes",
            "prepared_arrays_bytes",
            "worst_case_complex128_prepared_bytes",
        ):
            value = _strict_positive_int(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.form_branch not in ("real", "complex"):
            raise ValueError("form_branch must be 'real' or 'complex'")
        if not isinstance(self.exchange, bool):
            raise TypeError("exchange must be bool")

    @classmethod
    def from_shapes(
        cls,
        params: TBGZeroFieldCompanionSingleParticleParams,
        *,
        NG1: int,
        NG2: int,
        form_branch: Literal["real", "complex"],
        exchange: bool,
    ) -> TBGZeroFieldCompanionHFMemoryEstimate:
        N = params.N1 * params.N2
        nactive = params.n_active
        form_elements = 32 * N**2 * NG1 * NG2 * nactive**2
        tVE_elements = 64 * N**2 * nactive**4
        hamiltonian_elements = 32 * N * nactive**2
        legacy_scratch_elements = 4 * N**2 * (NG1 + 1) * (NG2 + 1)
        effective_itemsize = 8 if form_branch == "real" else 16
        # The source's exchange=False branch deliberately allocates complex zeros.
        tVE_itemsize = 16 if not exchange else effective_itemsize
        screened_elements = params.N1 * params.N2 * 4 * NG1 * NG2
        sp_energy_elements = params.N1 * params.N2 * 4 * nactive
        raw_form_bytes = form_elements * 16
        effective_form_bytes = form_elements * effective_itemsize
        M_bytes = form_elements * effective_itemsize
        tVE_bytes = tVE_elements * tVE_itemsize
        H_SP_bytes = hamiltonian_elements * 8
        screened_bytes = screened_elements * 8
        sp_energy_bytes = sp_energy_elements * 8
        prepared = (
            raw_form_bytes
            + effective_form_bytes
            + M_bytes
            + tVE_bytes
            + H_SP_bytes
            + screened_bytes
            + sp_energy_bytes
        )
        worst_case = (
            3 * form_elements * 16
            + tVE_elements * 16
            + hamiltonian_elements * 16
            + screened_bytes
            + sp_energy_bytes
        )
        return cls(
            form_elements=form_elements,
            tVE_elements=tVE_elements,
            hamiltonian_elements=hamiltonian_elements,
            legacy_form_scratch_elements=legacy_scratch_elements,
            raw_form_bytes=raw_form_bytes,
            effective_form_bytes=effective_form_bytes,
            M_bytes=M_bytes,
            tVE_bytes=tVE_bytes,
            H_SP_bytes=H_SP_bytes,
            screened_intFT_bytes=screened_bytes,
            sp_energy_bytes=sp_energy_bytes,
            prepared_arrays_bytes=prepared,
            worst_case_complex128_prepared_bytes=worst_case,
            form_branch=form_branch,
            exchange=exchange,
        )

    def to_metadata(self) -> dict[str, int | str | bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFPreparationResiduals:
    raw_form_max_abs_imag: float
    effective_form_max_abs_imag: float
    H_SP_hermiticity_max_abs_ev: float
    screening_roundtrip_max_abs_ev: float
    form_branch: Literal["real", "complex"]

    def __post_init__(self) -> None:
        for name in (
            "raw_form_max_abs_imag",
            "effective_form_max_abs_imag",
            "H_SP_hermiticity_max_abs_ev",
            "screening_roundtrip_max_abs_ev",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_real(getattr(self, name), name=name),
            )
        if self.form_branch not in ("real", "complex"):
            raise ValueError("form_branch must be 'real' or 'complex'")

    def to_metadata(self) -> dict[str, float | str]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFPreparedArrayHashes:
    form_raw: str
    form: str
    screened_intFT_ev: str
    M_ev: str
    tVE_ev: str
    H_SP_ev: str
    sp_energy_ev: str
    convention: str = TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_CONVENTION
    semantics: str = TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS

    def __post_init__(self) -> None:
        for name in (
            "form_raw",
            "form",
            "screened_intFT_ev",
            "M_ev",
            "tVE_ev",
            "H_SP_ev",
            "sp_energy_ev",
        ):
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))
        if self.convention != TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_CONVENTION:
            raise ValueError("Unsupported companion HF-action hash convention")
        if self.semantics != TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS:
            raise ValueError("Unsupported companion HF-action hash semantics")

    @classmethod
    def from_arrays(
        cls,
        *,
        form_raw: np.ndarray,
        form: np.ndarray,
        screened_intFT_ev: np.ndarray,
        M_ev: np.ndarray,
        tVE_ev: np.ndarray,
        H_SP_ev: np.ndarray,
        sp_energy_ev: np.ndarray,
    ) -> TBGZeroFieldCompanionHFPreparedArrayHashes:
        return cls(
            form_raw=_canonical_array_sha256(form_raw),
            form=_canonical_array_sha256(form),
            screened_intFT_ev=_canonical_array_sha256(screened_intFT_ev),
            M_ev=_canonical_array_sha256(M_ev),
            tVE_ev=_canonical_array_sha256(tVE_ev),
            H_SP_ev=_canonical_array_sha256(H_SP_ev),
            sp_energy_ev=_canonical_array_sha256(sp_energy_ev),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )

    def to_metadata(self) -> dict[str, str]:
        payload = {name: str(getattr(self, name)) for name in self.__dataclass_fields__}
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFActionResiduals:
    density_hermiticity_max_abs: float
    H_D_hermiticity_max_abs_ev: float
    H_E_hermiticity_max_abs_ev: float
    H_interaction_hermiticity_max_abs_ev: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_real(getattr(self, name), name=name),
            )

    def to_metadata(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFActionArrayHashes:
    density_delta: str
    H_D_ev: str
    H_E_ev: str
    H_interaction_ev: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))

    @classmethod
    def from_arrays(
        cls,
        *,
        density_delta: np.ndarray,
        H_D_ev: np.ndarray,
        H_E_ev: np.ndarray,
        H_interaction_ev: np.ndarray,
    ) -> TBGZeroFieldCompanionHFActionArrayHashes:
        return cls(
            density_delta=_canonical_array_sha256(density_delta),
            H_D_ev=_canonical_array_sha256(H_D_ev),
            H_E_ev=_canonical_array_sha256(H_E_ev),
            H_interaction_ev=_canonical_array_sha256(H_interaction_ev),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFAction:
    """One linear source action evaluated on a stored-orientation density delta."""

    params_fingerprint: str
    density_delta: np.ndarray
    H_D_ev: np.ndarray
    H_E_ev: np.ndarray
    H_interaction_ev: np.ndarray
    residuals: TBGZeroFieldCompanionHFActionResiduals
    array_hashes: TBGZeroFieldCompanionHFActionArrayHashes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "params_fingerprint",
            _validate_sha256(self.params_fingerprint, name="params_fingerprint"),
        )
        if not isinstance(self.residuals, TBGZeroFieldCompanionHFActionResiduals):
            raise TypeError("residuals must be typed companion HF-action residuals")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFActionArrayHashes):
            raise TypeError("array_hashes must be typed companion HF-action hashes")
        shape = np.asarray(self.H_D_ev).shape
        if len(shape) != 5 or shape[-1] != shape[-2]:
            raise ValueError("HF action arrays must have [k1,k2,spin,alpha,beta] shape")
        arrays = {
            name: _copy_finite_array(getattr(self, name), name=name, shape=shape)
            for name in ("density_delta", "H_D_ev", "H_E_ev", "H_interaction_ev")
        }
        if not np.array_equal(
            arrays["H_interaction_ev"],
            arrays["H_D_ev"] + arrays["H_E_ev"],
        ):
            raise ValueError("H_interaction_ev must equal H_D_ev + H_E_ev exactly")
        actual_residuals = TBGZeroFieldCompanionHFActionResiduals(
            density_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["density_delta"]
            ),
            H_D_hermiticity_max_abs_ev=_max_hermiticity_residual(arrays["H_D_ev"]),
            H_E_hermiticity_max_abs_ev=_max_hermiticity_residual(arrays["H_E_ev"]),
            H_interaction_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_interaction_ev"]
            ),
        )
        if actual_residuals != self.residuals:
            raise ValueError("residuals do not match companion HF-action arrays")
        if (
            actual_residuals.density_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("density_delta is materially non-Hermitian")
        if (
            actual_residuals.H_interaction_hermiticity_max_abs_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("companion HF action is materially non-Hermitian")
        actual_hashes = TBGZeroFieldCompanionHFActionArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("array_hashes do not match companion HF-action arrays")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        self._validate_live_arrays()

    def _validate_live_arrays(self) -> None:
        """Fail closed if any action array or its derived closure changed in place."""

        if not isinstance(self.residuals, TBGZeroFieldCompanionHFActionResiduals):
            raise TypeError("residuals must be typed companion HF-action residuals")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFActionArrayHashes):
            raise TypeError("array_hashes must be typed companion HF-action hashes")
        shape = self.density_delta.shape
        if len(shape) != 5 or shape[-1] != shape[-2]:
            raise ValueError(
                "live HF action arrays must retain "
                "[k1,k2,spin,alpha,beta] shape"
            )
        arrays = {
            name: _validate_live_array_layout(
                getattr(self, name),
                name=f"action.{name}",
                shape=shape,
                dtype=np.complex128,
            )
            for name in (
                "density_delta",
                "H_D_ev",
                "H_E_ev",
                "H_interaction_ev",
            )
        }
        actual_hashes = TBGZeroFieldCompanionHFActionArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError(
                "action array_hashes no longer match live companion HF-action arrays"
            )
        if not np.array_equal(
            arrays["H_interaction_ev"],
            arrays["H_D_ev"] + arrays["H_E_ev"],
        ):
            raise ValueError(
                "live H_interaction_ev must equal H_D_ev + H_E_ev exactly"
            )
        actual_residuals = TBGZeroFieldCompanionHFActionResiduals(
            density_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["density_delta"]
            ),
            H_D_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_D_ev"]
            ),
            H_E_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_E_ev"]
            ),
            H_interaction_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_interaction_ev"]
            ),
        )
        if actual_residuals != self.residuals:
            raise ValueError(
                "residuals no longer match live companion HF-action arrays"
            )
        if (
            actual_residuals.density_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("live density_delta is materially non-Hermitian")
        if (
            actual_residuals.H_interaction_hermiticity_max_abs_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("live companion HF action is materially non-Hermitian")

    def _fingerprint_payload(self) -> dict[str, object]:
        return {
            "array_hashes": self.array_hashes.fingerprint,
            "params_fingerprint": self.params_fingerprint,
            "residuals": self.residuals.to_metadata(),
            "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
        }

    def _fingerprint_from_validated_state(self) -> str:
        return _json_sha256(self._fingerprint_payload())

    @property
    def fingerprint(self) -> str:
        self._validate_live_arrays()
        return self._fingerprint_from_validated_state()

    @property
    def memory_bytes(self) -> int:
        return int(
            self.density_delta.nbytes
            + self.H_D_ev.nbytes
            + self.H_E_ev.nbytes
            + self.H_interaction_ev.nbytes
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFEnergy:
    """Source ``calc_E`` output in total finite-system eV."""

    components_ev: np.ndarray
    total_imag_residual_ev: float
    max_component_imag_residual_ev: float
    projector_sha256: str
    reference_sha256: str
    action_fingerprint: str
    units: str = TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS
    _components_ev_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        array = _copy_finite_array(
            self.components_ev,
            name="components_ev",
            shape=(4,),
            dtype=np.float64,
        )
        object.__setattr__(self, "components_ev", array)
        object.__setattr__(
            self,
            "_components_ev_sha256",
            _canonical_array_sha256(array),
        )
        for name in ("total_imag_residual_ev", "max_component_imag_residual_ev"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_real(getattr(self, name), name=name),
            )
        for name in ("projector_sha256", "reference_sha256", "action_fingerprint"):
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))
        self._validate_live_array()

    def _validate_live_array(
        self,
        *,
        projector_sha256: str | None = None,
        reference_sha256: str | None = None,
        action_fingerprint: str | None = None,
    ) -> None:
        """Rehash live components and validate scalar closure and bindings."""

        array = _validate_live_array_layout(
            self.components_ev,
            name="energy.components_ev",
            shape=(4,),
            dtype=np.float64,
        )
        stored_components_hash = _validate_sha256(
            self._components_ev_sha256,
            name="_components_ev_sha256",
        )
        if _canonical_array_sha256(array) != stored_components_hash:
            raise ValueError(
                "energy.components_ev no longer matches its construction hash"
            )
        for name in ("total_imag_residual_ev", "max_component_imag_residual_ev"):
            if _finite_nonnegative_real(getattr(self, name), name=name) != getattr(
                self,
                name,
            ):
                raise ValueError(f"{name} is not canonical")
        for name in ("projector_sha256", "reference_sha256", "action_fingerprint"):
            if _validate_sha256(getattr(self, name), name=name) != getattr(self, name):
                raise ValueError(f"{name} is not canonical")
        if self.units != TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS:
            raise ValueError("Companion energy must use finite-system eV units")
        component_sum = float(array[1] + array[2] + array[3])
        if (
            abs(float(array[0]) - component_sum)
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_CLOSURE_ATOL_EV
        ):
            raise ValueError(
                "companion energy total must equal kinetic + Hartree + exchange "
                "within the fixed closure tolerance"
            )
        if (
            self.total_imag_residual_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_IMAG_ENERGY_THRESHOLD_EV
        ):
            raise ValueError("companion energy has a material imaginary defect")
        expected_bindings = {
            "projector_sha256": projector_sha256,
            "reference_sha256": reference_sha256,
            "action_fingerprint": action_fingerprint,
        }
        for name, expected in expected_bindings.items():
            if expected is None or getattr(self, name) == expected:
                continue
            if name == "projector_sha256":
                raise ValueError(
                    "energy projector hash does not match its live binding"
                )
            if name == "reference_sha256":
                raise ValueError(
                    "energy reference hash does not match its live binding"
                )
            raise ValueError("energy is not bound to its live action")

    def _fingerprint_payload(self) -> dict[str, object]:
        return {
            "action_fingerprint": self.action_fingerprint,
            "components_ev_sha256": self._components_ev_sha256,
            "max_component_imag_residual_ev": self.max_component_imag_residual_ev,
            "projector_sha256": self.projector_sha256,
            "reference_sha256": self.reference_sha256,
            "total_imag_residual_ev": self.total_imag_residual_ev,
            "units": self.units,
        }

    def _fingerprint_from_validated_state(self) -> str:
        return _json_sha256(self._fingerprint_payload())

    @property
    def total_ev(self) -> float:
        return float(self.components_ev[0])

    @property
    def kinetic_ev(self) -> float:
        return float(self.components_ev[1])

    @property
    def hartree_ev(self) -> float:
        return float(self.components_ev[2])

    @property
    def exchange_ev(self) -> float:
        return float(self.components_ev[3])

    @property
    def fingerprint(self) -> str:
        self._validate_live_array()
        return self._fingerprint_from_validated_state()


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFEvaluationResiduals:
    projector_hermiticity_max_abs: float
    reference_hermiticity_max_abs: float
    density_delta_hermiticity_max_abs: float
    density_subtraction_max_abs: float
    H_total_hermiticity_max_abs_ev: float
    H_total_closure_max_abs_ev: float
    total_energy_imag_residual_ev: float
    energy_action_binding_residual: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_real(getattr(self, name), name=name),
            )

    def to_metadata(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFEvaluationArrayHashes:
    projector: str
    reference: str
    density_delta: str
    H_SP_ev: str
    H_total_ev: str
    energy_components_ev: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))

    @classmethod
    def from_arrays(
        cls,
        *,
        projector: np.ndarray,
        reference: np.ndarray,
        density_delta: np.ndarray,
        H_SP_ev: np.ndarray,
        H_total_ev: np.ndarray,
        energy_components_ev: np.ndarray,
    ) -> TBGZeroFieldCompanionHFEvaluationArrayHashes:
        return cls(
            projector=_canonical_array_sha256(projector),
            reference=_canonical_array_sha256(reference),
            density_delta=_canonical_array_sha256(density_delta),
            H_SP_ev=_canonical_array_sha256(H_SP_ev),
            H_total_ev=_canonical_array_sha256(H_total_ev),
            energy_components_ev=_canonical_array_sha256(energy_components_ev),
        )

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionHFEvaluation:
    """Immutable one-shot diagnostic evaluation; never an SCF state."""

    prepared: "TBGZeroFieldCompanionPreparedHFAction"
    prepared_fingerprint: str
    projector: np.ndarray
    reference: np.ndarray
    density_delta: np.ndarray
    action: TBGZeroFieldCompanionHFAction
    energy: TBGZeroFieldCompanionHFEnergy
    H_SP_ev: np.ndarray
    H_total_ev: np.ndarray
    residuals: TBGZeroFieldCompanionHFEvaluationResiduals
    array_hashes: TBGZeroFieldCompanionHFEvaluationArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError(
                "prepared must be TBGZeroFieldCompanionPreparedHFAction"
            )
        self.prepared._validate_live_sources()
        self.prepared._validate_live_arrays()
        live_prepared_fingerprint = (
            self.prepared._fingerprint_from_validated_state()
        )
        object.__setattr__(
            self,
            "prepared_fingerprint",
            _validate_sha256(self.prepared_fingerprint, name="prepared_fingerprint"),
        )
        if self.prepared_fingerprint != live_prepared_fingerprint:
            raise ValueError(
                "prepared_fingerprint does not match the direct prepared reference"
            )
        if not isinstance(self.action, TBGZeroFieldCompanionHFAction):
            raise TypeError("action must be TBGZeroFieldCompanionHFAction")
        self.action._validate_live_arrays()
        live_action_fingerprint = self.action._fingerprint_from_validated_state()
        if not isinstance(self.energy, TBGZeroFieldCompanionHFEnergy):
            raise TypeError("energy must be TBGZeroFieldCompanionHFEnergy")
        if not isinstance(self.residuals, TBGZeroFieldCompanionHFEvaluationResiduals):
            raise TypeError("residuals must be typed companion HF evaluation residuals")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFEvaluationArrayHashes):
            raise TypeError("array_hashes must be typed companion HF evaluation hashes")
        shape = self.action.H_D_ev.shape
        if shape != _hamiltonian_shape(self.prepared.params):
            raise ValueError("action shape does not match prepared params")
        arrays = {
            name: _copy_finite_array(getattr(self, name), name=name, shape=shape)
            for name in (
                "projector",
                "reference",
                "density_delta",
                "H_total_ev",
            )
        }
        if self.H_SP_ev is not self.prepared.H_SP_ev:
            supplied_H_SP = _copy_finite_array(
                self.H_SP_ev,
                name="H_SP_ev",
                shape=shape,
                dtype=np.float64,
            )
            if not np.array_equal(supplied_H_SP, self.prepared.H_SP_ev):
                raise ValueError("H_SP_ev must exactly equal prepared.H_SP_ev")
        arrays["H_SP_ev"] = self.prepared.H_SP_ev

        expected_density_delta = arrays["projector"] - arrays["reference"]
        density_subtraction_residual = float(
            np.max(np.abs(arrays["density_delta"] - expected_density_delta))
        )
        if density_subtraction_residual != 0.0:
            raise ValueError("density_delta must equal projector-reference exactly")
        if _canonical_array_sha256(arrays["density_delta"]) != self.action.array_hashes.density_delta:
            raise ValueError("evaluation density_delta does not match action input")

        canonical_action = calc_fock_matrix(
            self.prepared.params,
            arrays["density_delta"],
            self.prepared.form,
            self.prepared.M_ev,
            self.prepared.tVE_ev,
        )
        for name in (
            "density_delta",
            "H_D_ev",
            "H_E_ev",
            "H_interaction_ev",
        ):
            if not np.array_equal(
                getattr(self.action, name),
                getattr(canonical_action, name),
            ):
                raise ValueError(
                    f"action.{name} does not exactly equal the canonical "
                    "calc_fock_matrix output"
                )
        if self.action.array_hashes != canonical_action.array_hashes:
            raise ValueError(
                "action array_hashes do not match the canonical calc_fock_matrix action"
            )
        if self.action.residuals != canonical_action.residuals:
            raise ValueError(
                "action residuals do not match the canonical calc_fock_matrix action"
            )
        if self.action.fingerprint != canonical_action.fingerprint:
            raise ValueError(
                "action fingerprint does not match the canonical calc_fock_matrix action"
            )

        expected_H_total = self.prepared.H_SP_ev + self.action.H_interaction_ev
        H_total_closure_residual = float(
            np.max(np.abs(arrays["H_total_ev"] - expected_H_total))
        )
        if H_total_closure_residual != 0.0:
            raise ValueError(
                "H_total_ev must equal H_SP_ev + action.H_interaction_ev exactly, "
                "with H_SP_ev bound to prepared.H_SP_ev"
            )

        actual_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
            projector=arrays["projector"],
            reference=arrays["reference"],
            density_delta=arrays["density_delta"],
            H_SP_ev=arrays["H_SP_ev"],
            H_total_ev=arrays["H_total_ev"],
            energy_components_ev=self.energy.components_ev,
        )
        self.energy._validate_live_array(
            projector_sha256=actual_hashes.projector,
            reference_sha256=actual_hashes.reference,
            action_fingerprint=live_action_fingerprint,
        )
        if self.energy.projector_sha256 != actual_hashes.projector:
            raise ValueError("energy projector hash does not match evaluation projector")
        if self.energy.reference_sha256 != actual_hashes.reference:
            raise ValueError("energy reference hash does not match evaluation reference")
        energy_action_binding_residual = float(
            self.energy.action_fingerprint != live_action_fingerprint
        )
        if energy_action_binding_residual != 0.0:
            raise ValueError("energy is not bound to the evaluation action")

        recomputed_energy = calc_E(
            self.prepared.params,
            arrays["projector"],
            arrays["reference"],
            self.prepared.sp_energy_ev,
            self.action,
        )
        if not np.allclose(
            self.energy.components_ev,
            recomputed_energy.components_ev,
            rtol=0.0,
            atol=TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_CLOSURE_ATOL_EV,
        ):
            raise ValueError("energy components do not match recomputed calc_E")
        if (
            _canonical_array_sha256(self.energy.components_ev)
            != _canonical_array_sha256(recomputed_energy.components_ev)
        ):
            raise ValueError("energy component hash does not match recomputed calc_E")
        for name in (
            "projector_sha256",
            "reference_sha256",
            "action_fingerprint",
        ):
            if getattr(self.energy, name) != getattr(recomputed_energy, name):
                raise ValueError(
                    f"energy {name} does not match recomputed calc_E binding"
                )
        for name in (
            "total_imag_residual_ev",
            "max_component_imag_residual_ev",
        ):
            if getattr(self.energy, name) != getattr(recomputed_energy, name):
                raise ValueError(
                    f"energy {name} does not match recomputed calc_E residual"
                )

        actual_residuals = TBGZeroFieldCompanionHFEvaluationResiduals(
            projector_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["projector"]
            ),
            reference_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["reference"]
            ),
            density_delta_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["density_delta"]
            ),
            density_subtraction_max_abs=density_subtraction_residual,
            H_total_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_total_ev"]
            ),
            H_total_closure_max_abs_ev=H_total_closure_residual,
            total_energy_imag_residual_ev=self.energy.total_imag_residual_ev,
            energy_action_binding_residual=energy_action_binding_residual,
        )
        if actual_residuals != self.residuals:
            raise ValueError(
                "residuals do not match companion HF evaluation arrays and bindings"
            )
        if (
            actual_residuals.projector_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("projector is materially non-Hermitian")
        if (
            actual_residuals.reference_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("reference is materially non-Hermitian")
        if (
            actual_residuals.H_total_hermiticity_max_abs_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("H_total_ev is materially non-Hermitian")
        if actual_hashes != self.array_hashes:
            raise ValueError("array_hashes do not match companion HF evaluation arrays")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def _validate_live_arrays(self) -> tuple[str, str, str]:
        """Revalidate all nested fingerprints and canonical evaluation closures."""

        if not isinstance(self.prepared, TBGZeroFieldCompanionPreparedHFAction):
            raise TypeError("prepared must be TBGZeroFieldCompanionPreparedHFAction")
        if not isinstance(self.action, TBGZeroFieldCompanionHFAction):
            raise TypeError("action must be TBGZeroFieldCompanionHFAction")
        if not isinstance(self.energy, TBGZeroFieldCompanionHFEnergy):
            raise TypeError("energy must be TBGZeroFieldCompanionHFEnergy")
        if not isinstance(self.residuals, TBGZeroFieldCompanionHFEvaluationResiduals):
            raise TypeError("residuals must be typed companion HF evaluation residuals")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFEvaluationArrayHashes):
            raise TypeError("array_hashes must be typed companion HF evaluation hashes")

        shape = _hamiltonian_shape(self.prepared.params)
        arrays = {
            "projector": _validate_live_array_layout(
                self.projector,
                name="evaluation.projector",
                shape=shape,
                dtype=np.complex128,
            ),
            "reference": _validate_live_array_layout(
                self.reference,
                name="evaluation.reference",
                shape=shape,
                dtype=np.complex128,
            ),
            "density_delta": _validate_live_array_layout(
                self.density_delta,
                name="evaluation.density_delta",
                shape=shape,
                dtype=np.complex128,
            ),
            "H_SP_ev": _validate_live_array_layout(
                self.H_SP_ev,
                name="evaluation.H_SP_ev",
                shape=shape,
                dtype=np.float64,
            ),
            "H_total_ev": _validate_live_array_layout(
                self.H_total_ev,
                name="evaluation.H_total_ev",
                shape=shape,
                dtype=np.complex128,
            ),
        }
        energy_components = _validate_live_array_layout(
            self.energy.components_ev,
            name="evaluation.energy_components_ev",
            shape=(4,),
            dtype=np.float64,
        )
        actual_hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
            projector=arrays["projector"],
            reference=arrays["reference"],
            density_delta=arrays["density_delta"],
            H_SP_ev=arrays["H_SP_ev"],
            H_total_ev=arrays["H_total_ev"],
            energy_components_ev=energy_components,
        )
        if actual_hashes != self.array_hashes:
            raise ValueError(
                "evaluation array_hashes no longer match live evaluation arrays"
            )

        self.prepared._validate_live_sources()
        self.prepared._validate_live_arrays()
        prepared_fingerprint = self.prepared._fingerprint_from_validated_state()
        if self.prepared_fingerprint != prepared_fingerprint:
            raise ValueError(
                "prepared_fingerprint no longer matches the direct prepared reference"
            )
        if arrays["H_SP_ev"] is not self.prepared.H_SP_ev:
            raise ValueError("H_SP_ev must remain the direct prepared.H_SP_ev array")

        self.action._validate_live_arrays()
        action_fingerprint = self.action._fingerprint_from_validated_state()
        if self.action.params_fingerprint != self.prepared.params.fingerprint:
            raise ValueError("action is not bound to prepared params")
        self.energy._validate_live_array(
            projector_sha256=actual_hashes.projector,
            reference_sha256=actual_hashes.reference,
            action_fingerprint=action_fingerprint,
        )
        energy_fingerprint = self.energy._fingerprint_from_validated_state()

        expected_density_delta = arrays["projector"] - arrays["reference"]
        density_subtraction_residual = float(
            np.max(np.abs(arrays["density_delta"] - expected_density_delta))
        )
        if density_subtraction_residual != 0.0:
            raise ValueError("density_delta must equal projector-reference exactly")
        if not np.array_equal(arrays["density_delta"], self.action.density_delta):
            raise ValueError("evaluation density_delta does not match action input")

        canonical_action = calc_fock_matrix(
            self.prepared.params,
            arrays["density_delta"],
            self.prepared.form,
            self.prepared.M_ev,
            self.prepared.tVE_ev,
        )
        for name in (
            "density_delta",
            "H_D_ev",
            "H_E_ev",
            "H_interaction_ev",
        ):
            if not np.array_equal(
                getattr(self.action, name),
                getattr(canonical_action, name),
            ):
                raise ValueError(
                    f"action.{name} does not exactly equal the canonical "
                    "calc_fock_matrix output"
                )
        canonical_action._validate_live_arrays()
        if action_fingerprint != canonical_action._fingerprint_from_validated_state():
            raise ValueError(
                "action fingerprint does not match the canonical calc_fock_matrix action"
            )

        expected_H_total = arrays["H_SP_ev"] + self.action.H_interaction_ev
        H_total_closure_residual = float(
            np.max(np.abs(arrays["H_total_ev"] - expected_H_total))
        )
        if H_total_closure_residual != 0.0:
            raise ValueError(
                "H_total_ev must equal H_SP_ev + action.H_interaction_ev exactly, "
                "with H_SP_ev bound to prepared.H_SP_ev"
            )

        recomputed_energy = calc_E(
            self.prepared.params,
            arrays["projector"],
            arrays["reference"],
            self.prepared.sp_energy_ev,
            self.action,
        )
        if not np.allclose(
            self.energy.components_ev,
            recomputed_energy.components_ev,
            rtol=0.0,
            atol=TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_CLOSURE_ATOL_EV,
        ):
            raise ValueError("energy components do not match recomputed calc_E")
        if (
            _canonical_array_sha256(self.energy.components_ev)
            != _canonical_array_sha256(recomputed_energy.components_ev)
        ):
            raise ValueError("energy component hash does not match recomputed calc_E")
        for name in (
            "projector_sha256",
            "reference_sha256",
            "action_fingerprint",
            "total_imag_residual_ev",
            "max_component_imag_residual_ev",
        ):
            if getattr(self.energy, name) != getattr(recomputed_energy, name):
                raise ValueError(f"energy {name} does not match recomputed calc_E")

        actual_residuals = TBGZeroFieldCompanionHFEvaluationResiduals(
            projector_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["projector"]
            ),
            reference_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["reference"]
            ),
            density_delta_hermiticity_max_abs=_max_hermiticity_residual(
                arrays["density_delta"]
            ),
            density_subtraction_max_abs=density_subtraction_residual,
            H_total_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_total_ev"]
            ),
            H_total_closure_max_abs_ev=H_total_closure_residual,
            total_energy_imag_residual_ev=self.energy.total_imag_residual_ev,
            energy_action_binding_residual=float(
                self.energy.action_fingerprint != action_fingerprint
            ),
        )
        if actual_residuals != self.residuals:
            raise ValueError(
                "residuals no longer match live companion HF evaluation arrays"
            )
        if (
            actual_residuals.projector_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("live projector is materially non-Hermitian")
        if (
            actual_residuals.reference_hermiticity_max_abs
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("live reference is materially non-Hermitian")
        if (
            actual_residuals.H_total_hermiticity_max_abs_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("live H_total_ev is materially non-Hermitian")
        return prepared_fingerprint, action_fingerprint, energy_fingerprint

    @property
    def H_D_ev(self) -> np.ndarray:
        return self.action.H_D_ev

    @property
    def H_E_ev(self) -> np.ndarray:
        return self.action.H_E_ev

    @property
    def energy_components_ev(self) -> np.ndarray:
        return self.energy.components_ev

    @property
    def memory_bytes(self) -> int:
        return int(
            self.projector.nbytes
            + self.reference.nbytes
            + self.density_delta.nbytes
            + self.H_SP_ev.nbytes
            + self.H_total_ev.nbytes
            + self.action.memory_bytes
            + self.energy.components_ev.nbytes
        )

    def _fingerprint_payload(
        self,
        *,
        action_fingerprint: str,
        energy_fingerprint: str,
    ) -> dict[str, object]:
        return {
            "action": action_fingerprint,
            "array_hashes": self.array_hashes.fingerprint,
            "energy": energy_fingerprint,
            "prepared_fingerprint": self.prepared_fingerprint,
            "residuals": self.residuals.to_metadata(),
            "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
        }

    @property
    def fingerprint(self) -> str:
        _prepared, action_fingerprint, energy_fingerprint = (
            self._validate_live_arrays()
        )
        return _json_sha256(
            self._fingerprint_payload(
                action_fingerprint=action_fingerprint,
                energy_fingerprint=energy_fingerprint,
            )
        )


@dataclass(frozen=True, slots=True)
class TBGZeroFieldCompanionPreparedHFAction:
    """Source-bound form-factor/action provider for one typed Stage2/Stage3 pair."""

    params: TBGZeroFieldCompanionSingleParticleParams
    interaction_NG1: int
    interaction_NG2: int
    spec: TBGZeroFieldCompanionHFActionSpec
    single_particle_source: TBGZeroFieldCompanionSingleParticleResult
    interaction_source: TBGZeroFieldCompanionInteractionResult
    single_particle_fingerprint: str
    interaction_fingerprint: str
    raw_interaction_array_sha256: str
    form_raw: np.ndarray
    form: np.ndarray
    screened_intFT_ev: np.ndarray
    M_ev: np.ndarray
    tVE_ev: np.ndarray
    H_SP_ev: np.ndarray
    sp_energy_ev: np.ndarray
    form_branch: Literal["real", "complex"]
    residuals: TBGZeroFieldCompanionHFPreparationResiduals
    memory_estimate: TBGZeroFieldCompanionHFMemoryEstimate
    provenance: TBGZeroFieldCompanionHFActionProvenance
    array_hashes: TBGZeroFieldCompanionHFPreparedArrayHashes

    def __post_init__(self) -> None:
        if not isinstance(self.params, TBGZeroFieldCompanionSingleParticleParams):
            raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
        if not isinstance(
            self.single_particle_source,
            TBGZeroFieldCompanionSingleParticleResult,
        ):
            raise TypeError(
                "single_particle_source must be "
                "TBGZeroFieldCompanionSingleParticleResult"
            )
        if not isinstance(
            self.interaction_source,
            TBGZeroFieldCompanionInteractionResult,
        ):
            raise TypeError(
                "interaction_source must be TBGZeroFieldCompanionInteractionResult"
            )
        if not isinstance(self.spec, TBGZeroFieldCompanionHFActionSpec):
            raise TypeError("spec must be TBGZeroFieldCompanionHFActionSpec")
        for name in (
            "single_particle_fingerprint",
            "interaction_fingerprint",
            "raw_interaction_array_sha256",
        ):
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))

        self._validate_live_sources()

        NG1, NG2 = _validate_cutoffs(
            self.params,
            NG1=self.interaction_NG1,
            NG2=self.interaction_NG2,
        )
        if NG1 != self.interaction_source.spec.NG1:
            raise ValueError("interaction_NG1 differs from interaction_source.spec.NG1")
        if NG2 != self.interaction_source.spec.NG2:
            raise ValueError("interaction_NG2 differs from interaction_source.spec.NG2")
        object.__setattr__(self, "interaction_NG1", NG1)
        object.__setattr__(self, "interaction_NG2", NG2)

        if self.form_branch not in ("real", "complex"):
            raise ValueError("form_branch must be 'real' or 'complex'")
        if not isinstance(self.residuals, TBGZeroFieldCompanionHFPreparationResiduals):
            raise TypeError("residuals must be typed preparation residuals")
        if self.residuals.form_branch != self.form_branch:
            raise ValueError("residual form branch does not match prepared form branch")
        if not isinstance(self.memory_estimate, TBGZeroFieldCompanionHFMemoryEstimate):
            raise TypeError("memory_estimate must be typed companion memory estimate")
        if self.memory_estimate.form_branch != self.form_branch:
            raise ValueError("memory-estimate form branch does not match prepared form branch")
        if self.memory_estimate.exchange != self.spec.exchange:
            raise ValueError("memory-estimate exchange flag does not match spec")
        if not isinstance(self.provenance, TBGZeroFieldCompanionHFActionProvenance):
            raise TypeError("provenance must be typed companion HF-action provenance")
        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFPreparedArrayHashes):
            raise TypeError("array_hashes must be typed prepared-array hashes")

        form_shape = _form_shape(self.params, NG1=NG1, NG2=NG2)
        int_shape = (self.params.N1, self.params.N2, 2 * NG1, 2 * NG2)
        ham_shape = _hamiltonian_shape(self.params)
        sp_shape = (
            self.params.N1,
            self.params.N2,
            2,
            self.params.active_band_count,
        )
        form_dtype = np.float64 if self.form_branch == "real" else np.complex128
        arrays = {
            "form_raw": _copy_finite_array(
                self.form_raw,
                name="form_raw",
                shape=form_shape,
                dtype=np.complex128,
            ),
            "form": _copy_finite_array(
                self.form,
                name="form",
                shape=form_shape,
                dtype=form_dtype,
            ),
            "screened_intFT_ev": _copy_finite_array(
                self.screened_intFT_ev,
                name="screened_intFT_ev",
                shape=int_shape,
                dtype=np.float64,
            ),
            "M_ev": _copy_finite_array(
                self.M_ev,
                name="M_ev",
                shape=form_shape,
                dtype=form_dtype,
            ),
            "tVE_ev": _copy_finite_array(
                self.tVE_ev,
                name="tVE_ev",
                shape=_tve_shape(self.params),
                dtype=(
                    np.complex128
                    if not self.spec.exchange or self.form_branch == "complex"
                    else np.float64
                ),
            ),
            "H_SP_ev": _copy_finite_array(
                self.H_SP_ev,
                name="H_SP_ev",
                shape=ham_shape,
                dtype=np.float64,
            ),
            "sp_energy_ev": _copy_finite_array(
                self.sp_energy_ev,
                name="sp_energy_ev",
                shape=sp_shape,
                dtype=np.float64,
            ),
        }
        if np.any(arrays["screened_intFT_ev"] < 0.0):
            raise ValueError("screened_intFT_ev must be nonnegative")
        if not np.array_equal(
            arrays["sp_energy_ev"],
            self.single_particle_source.sp_energy_ev,
        ):
            raise ValueError(
                "sp_energy_ev does not exactly equal "
                "single_particle_source.sp_energy_ev"
            )
        expected_form_raw = gen_full_form_factors(
            self.params,
            self.single_particle_source.coeff,
            NG1=NG1,
            NG2=NG2,
        )
        if not np.allclose(
            arrays["form_raw"],
            expected_form_raw,
            rtol=0.0,
            atol=TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_SOURCE_ATOL,
        ):
            raise ValueError(
                "form_raw does not match recomputation from "
                "single_particle_source.coeff"
            )
        expected_form, expected_form_branch, _ = main_program_realify_form(
            arrays["form_raw"]
        )
        if self.form_branch != expected_form_branch:
            raise ValueError(
                "form_branch does not match the branch recomputed from "
                "single_particle_source.coeff"
            )
        if not np.array_equal(arrays["form"], expected_form):
            raise ValueError(
                "form does not exactly match source-recomputed form_raw and branch"
            )
        expected_screened_intFT = np.asarray(
            self.interaction_source.intFT_ev / self.spec.epsr
        )
        if not np.array_equal(arrays["screened_intFT_ev"], expected_screened_intFT):
            raise ValueError(
                "screened_intFT_ev does not exactly equal "
                "interaction_source.intFT_ev / epsr"
            )
        expected_H_SP = gen_H_SP(self.params, arrays["sp_energy_ev"])
        if not np.array_equal(arrays["H_SP_ev"], expected_H_SP):
            raise ValueError("H_SP_ev does not exactly equal gen_H_SP(sp_energy_ev)")
        expected_M, expected_tVE = gen_M_tVE(
            self.params,
            arrays["form"],
            arrays["screened_intFT_ev"],
            exchange=self.spec.exchange,
        )
        if not np.array_equal(arrays["M_ev"], expected_M):
            raise ValueError(
                "M_ev does not exactly equal gen_M_tVE(form, screened_intFT_ev, "
                "exchange)[0]"
            )
        if not np.array_equal(arrays["tVE_ev"], expected_tVE):
            raise ValueError(
                "tVE_ev does not exactly equal gen_M_tVE(form, screened_intFT_ev, "
                "exchange)[1]"
            )
        actual_residuals = TBGZeroFieldCompanionHFPreparationResiduals(
            raw_form_max_abs_imag=_max_abs_imag(arrays["form_raw"]),
            effective_form_max_abs_imag=_max_abs_imag(arrays["form"]),
            H_SP_hermiticity_max_abs_ev=_max_hermiticity_residual(
                arrays["H_SP_ev"]
            ),
            screening_roundtrip_max_abs_ev=float(
                np.max(
                    np.abs(
                        arrays["screened_intFT_ev"] * self.spec.epsr
                        - self.interaction_source.intFT_ev
                    )
                )
            ),
            form_branch=self.form_branch,
        )
        if actual_residuals != self.residuals:
            raise ValueError("residuals do not match prepared companion arrays")
        if (
            actual_residuals.H_SP_hermiticity_max_abs_ev
            > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
        ):
            raise ValueError("H_SP_ev is materially non-Hermitian")
        if not self.spec.exchange and np.any(arrays["tVE_ev"] != 0.0):
            raise ValueError("exchange=False requires the exact source zero tVE")
        actual_hashes = TBGZeroFieldCompanionHFPreparedArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError("array_hashes do not match prepared companion arrays")
        if self.memory_estimate.prepared_arrays_bytes != sum(
            int(array.nbytes) for array in arrays.values()
        ):
            raise ValueError("memory_estimate does not match prepared array storage")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        self._validate_live_arrays()

    def _validate_live_sources(self) -> None:
        """Fail closed if the bound typed Stage-2/Stage-3 sources changed in place."""

        if not isinstance(self.params, TBGZeroFieldCompanionSingleParticleParams):
            raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
        if not isinstance(
            self.single_particle_source,
            TBGZeroFieldCompanionSingleParticleResult,
        ):
            raise TypeError(
                "single_particle_source must be "
                "TBGZeroFieldCompanionSingleParticleResult"
            )
        if not isinstance(
            self.interaction_source,
            TBGZeroFieldCompanionInteractionResult,
        ):
            raise TypeError(
                "interaction_source must be TBGZeroFieldCompanionInteractionResult"
            )
        if not isinstance(
            self.single_particle_source.array_hashes,
            TBGZeroFieldCompanionSingleParticleArrayHashes,
        ):
            raise TypeError(
                "single_particle_source.array_hashes must be typed companion hashes"
            )
        if not isinstance(
            self.interaction_source.array_hashes,
            TBGZeroFieldCompanionInteractionArrayHashes,
        ):
            raise TypeError(
                "interaction_source.array_hashes must be typed companion interaction hashes"
            )
        if self.params != self.single_particle_source.params:
            raise ValueError("params differ from single_particle_source.params")
        if self.params != self.interaction_source.params:
            raise ValueError("params differ from interaction_source.params")
        if self.params.fingerprint != self.single_particle_source.params.fingerprint:
            raise ValueError(
                "params fingerprint differs from single_particle_source.params"
            )
        if self.params.fingerprint != self.interaction_source.params.fingerprint:
            raise ValueError("params fingerprint differs from interaction_source.params")

        actual_single_particle_hashes = (
            TBGZeroFieldCompanionSingleParticleArrayHashes.from_arrays(
                coeff=self.single_particle_source.coeff,
                sp_energy_ev=self.single_particle_source.sp_energy_ev,
                U_C2T=self.single_particle_source.U_C2T,
            )
        )
        for name in ("coeff", "sp_energy_ev", "U_C2T"):
            if getattr(actual_single_particle_hashes, name) != getattr(
                self.single_particle_source.array_hashes,
                name,
            ):
                raise ValueError(
                    f"single_particle_source.{name} no longer matches its source hash"
                )
        actual_interaction_hashes = (
            TBGZeroFieldCompanionInteractionArrayHashes.from_array(
                self.interaction_source.intFT_ev
            )
        )
        if actual_interaction_hashes.intFT_ev != (
            self.interaction_source.array_hashes.intFT_ev
        ):
            raise ValueError(
                "interaction_source.intFT_ev no longer matches its source hash"
            )

        if self.single_particle_fingerprint != self.single_particle_source.fingerprint:
            raise ValueError(
                "single_particle_fingerprint does not match single_particle_source"
            )
        if self.interaction_fingerprint != self.interaction_source.fingerprint:
            raise ValueError(
                "interaction_fingerprint does not match interaction_source"
            )
        if (
            self.raw_interaction_array_sha256
            != self.interaction_source.array_hashes.intFT_ev
        ):
            raise ValueError(
                "raw_interaction_array_sha256 does not match "
                "interaction_source.array_hashes.intFT_ev"
            )

        single_particle_rlv_fingerprint = (
            self.single_particle_source.rlv_geometry.fingerprint
        )
        interaction_rlv_fingerprint = self.interaction_source.rlv_geometry.fingerprint
        if (
            self.single_particle_source.geometry_fingerprints.rlv_geometry
            != single_particle_rlv_fingerprint
        ):
            raise ValueError(
                "single_particle_source reciprocal-geometry fingerprint is stale"
            )
        if single_particle_rlv_fingerprint != interaction_rlv_fingerprint:
            raise ValueError(
                "single_particle_source and interaction_source reciprocal geometries differ"
            )
        if (
            self.single_particle_source.rlv_geometry.params_fingerprint
            != self.params.fingerprint
        ):
            raise ValueError(
                "single_particle_source reciprocal geometry is not bound to params"
            )
        if (
            self.interaction_source.rlv_geometry.params_fingerprint
            != self.params.fingerprint
        ):
            raise ValueError("interaction_source reciprocal geometry is not bound to params")

    def _validate_live_arrays(self) -> None:
        """Rehash every prepared array and recheck its exact storage contract."""

        if not isinstance(self.array_hashes, TBGZeroFieldCompanionHFPreparedArrayHashes):
            raise TypeError("array_hashes must be typed prepared-array hashes")
        if not isinstance(self.memory_estimate, TBGZeroFieldCompanionHFMemoryEstimate):
            raise TypeError("memory_estimate must be typed companion memory estimate")
        form_shape = _form_shape(
            self.params,
            NG1=self.interaction_NG1,
            NG2=self.interaction_NG2,
        )
        int_shape = (
            self.params.N1,
            self.params.N2,
            2 * self.interaction_NG1,
            2 * self.interaction_NG2,
        )
        ham_shape = _hamiltonian_shape(self.params)
        sp_shape = (
            self.params.N1,
            self.params.N2,
            2,
            self.params.active_band_count,
        )
        form_dtype = np.float64 if self.form_branch == "real" else np.complex128
        tVE_dtype = (
            np.complex128
            if not self.spec.exchange or self.form_branch == "complex"
            else np.float64
        )
        layouts = {
            "form_raw": (form_shape, np.complex128, "raw_form_bytes"),
            "form": (form_shape, form_dtype, "effective_form_bytes"),
            "screened_intFT_ev": (
                int_shape,
                np.float64,
                "screened_intFT_bytes",
            ),
            "M_ev": (form_shape, form_dtype, "M_bytes"),
            "tVE_ev": (_tve_shape(self.params), tVE_dtype, "tVE_bytes"),
            "H_SP_ev": (ham_shape, np.float64, "H_SP_bytes"),
            "sp_energy_ev": (sp_shape, np.float64, "sp_energy_bytes"),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, (shape, dtype, memory_name) in layouts.items():
            array = _validate_live_array_layout(
                getattr(self, name),
                name=f"prepared.{name}",
                shape=shape,
                dtype=dtype,
            )
            if array.nbytes != getattr(self.memory_estimate, memory_name):
                raise ValueError(
                    f"prepared.{name} storage no longer matches "
                    f"memory_estimate.{memory_name}"
                )
            arrays[name] = array
        if self.memory_estimate.form_elements != arrays["form"].size:
            raise ValueError("prepared form size no longer matches memory estimate")
        if self.memory_estimate.tVE_elements != arrays["tVE_ev"].size:
            raise ValueError("prepared tVE size no longer matches memory estimate")
        if self.memory_estimate.hamiltonian_elements != arrays["H_SP_ev"].size:
            raise ValueError("prepared Hamiltonian size no longer matches memory estimate")
        if self.memory_estimate.prepared_arrays_bytes != sum(
            int(array.nbytes) for array in arrays.values()
        ):
            raise ValueError("prepared array storage no longer matches memory estimate")
        actual_hashes = TBGZeroFieldCompanionHFPreparedArrayHashes.from_arrays(**arrays)
        if actual_hashes != self.array_hashes:
            raise ValueError(
                "prepared array_hashes no longer match live prepared arrays"
            )

    def _fingerprint_payload(self) -> dict[str, object]:
        return {
            "array_hashes": self.array_hashes.fingerprint,
            "form_branch": self.form_branch,
            "interaction_fingerprint": self.interaction_fingerprint,
            "interaction_source_fingerprint": self.interaction_source.fingerprint,
            "memory_estimate": self.memory_estimate.to_metadata(),
            "params_fingerprint": self.params.fingerprint,
            "provenance": self.provenance.fingerprint,
            "raw_interaction_array_sha256": self.raw_interaction_array_sha256,
            "residuals": self.residuals.to_metadata(),
            "schema": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
            "single_particle_fingerprint": self.single_particle_fingerprint,
            "single_particle_source_fingerprint": (
                self.single_particle_source.fingerprint
            ),
            "spec_fingerprint": self.spec.fingerprint,
        }

    def _fingerprint_from_validated_state(self) -> str:
        return _json_sha256(self._fingerprint_payload())

    @property
    def fingerprint(self) -> str:
        self._validate_live_sources()
        self._validate_live_arrays()
        return self._fingerprint_from_validated_state()

    def to_metadata(self) -> dict[str, object]:
        self._validate_live_sources()
        self._validate_live_arrays()
        live_fingerprint = self._fingerprint_from_validated_state()
        arrays = {
            "form_raw": self.form_raw,
            "form": self.form,
            "screened_intFT_ev": self.screened_intFT_ev,
            "M_ev": self.M_ev,
            "tVE_ev": self.tVE_ev,
            "H_SP_ev": self.H_SP_ev,
            "sp_energy_ev": self.sp_energy_ev,
        }
        return {
            "array_dtypes": {name: array.dtype.str for name, array in arrays.items()},
            "array_hashes": self.array_hashes.to_metadata(),
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "fingerprint": live_fingerprint,
            "form_branch": self.form_branch,
            "interaction_NG1": self.interaction_NG1,
            "interaction_NG2": self.interaction_NG2,
            "interaction_fingerprint": self.interaction_fingerprint,
            "interaction_source_fingerprint": self.interaction_source.fingerprint,
            "memory_estimate": self.memory_estimate.to_metadata(),
            "params": self.params.to_metadata(),
            "provenance": self.provenance.to_metadata(),
            "provenance_fingerprint": self.provenance.fingerprint,
            "raw_interaction_array_sha256": self.raw_interaction_array_sha256,
            "residuals": self.residuals.to_metadata(),
            "schema": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION,
            "scope": TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE,
            "single_particle_fingerprint": self.single_particle_fingerprint,
            "single_particle_source_fingerprint": (
                self.single_particle_source.fingerprint
            ),
            "spec": self.spec.to_metadata(),
            "stored_projector_convention": (
                TBG_ZERO_FIELD_COMPANION_HF_ACTION_STORED_PROJECTOR_CONVENTION
            ),
        }

    def evaluate(
        self,
        projector: np.ndarray,
        reference: np.ndarray,
    ) -> TBGZeroFieldCompanionHFEvaluation:
        """Evaluate one projector/reference pair; subtraction occurs exactly once."""

        self._validate_live_sources()
        self._validate_live_arrays()
        live_fingerprint = self._fingerprint_from_validated_state()
        shape = _hamiltonian_shape(self.params)
        P = _copy_finite_array(
            projector,
            name="projector",
            shape=shape,
            dtype=np.complex128,
        )
        P_ref = _copy_finite_array(
            reference,
            name="reference",
            shape=shape,
            dtype=np.complex128,
        )
        P_residual = _max_hermiticity_residual(P)
        P_ref_residual = _max_hermiticity_residual(P_ref)
        if P_residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
            raise ValueError(
                f"projector is materially non-Hermitian: residual={P_residual:.6e}"
            )
        if P_ref_residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
            raise ValueError(
                f"reference is materially non-Hermitian: residual={P_ref_residual:.6e}"
            )
        density_delta = np.asarray(P - P_ref)
        action = calc_fock_matrix(
            self.params,
            density_delta,
            self.form,
            self.M_ev,
            self.tVE_ev,
        )
        energy = calc_E(
            self.params,
            P,
            P_ref,
            self.sp_energy_ev,
            action,
        )
        H_total = np.asarray(self.H_SP_ev + action.H_interaction_ev)
        H_total_residual = _max_hermiticity_residual(H_total)
        if H_total_residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
            raise ValueError(
                "total companion Hamiltonian is materially non-Hermitian: "
                f"residual={H_total_residual:.6e}"
            )
        residuals = TBGZeroFieldCompanionHFEvaluationResiduals(
            projector_hermiticity_max_abs=P_residual,
            reference_hermiticity_max_abs=P_ref_residual,
            density_delta_hermiticity_max_abs=(
                action.residuals.density_hermiticity_max_abs
            ),
            density_subtraction_max_abs=float(
                np.max(np.abs(density_delta - (P - P_ref)))
            ),
            H_total_hermiticity_max_abs_ev=H_total_residual,
            H_total_closure_max_abs_ev=float(
                np.max(
                    np.abs(
                        H_total - (self.H_SP_ev + action.H_interaction_ev)
                    )
                )
            ),
            total_energy_imag_residual_ev=energy.total_imag_residual_ev,
            energy_action_binding_residual=float(
                energy.action_fingerprint != action.fingerprint
            ),
        )
        hashes = TBGZeroFieldCompanionHFEvaluationArrayHashes.from_arrays(
            projector=P,
            reference=P_ref,
            density_delta=density_delta,
            H_SP_ev=self.H_SP_ev,
            H_total_ev=H_total,
            energy_components_ev=energy.components_ev,
        )
        return TBGZeroFieldCompanionHFEvaluation(
            prepared=self,
            prepared_fingerprint=live_fingerprint,
            projector=P,
            reference=P_ref,
            density_delta=density_delta,
            action=action,
            energy=energy,
            H_SP_ev=self.H_SP_ev,
            H_total_ev=H_total,
            residuals=residuals,
            array_hashes=hashes,
        )


def gen_form_factors(
    params: TBGZeroFieldCompanionSingleParticleParams,
    c: np.ndarray,
    cp: np.ndarray,
    *,
    NG1: int,
    NG2: int,
) -> np.ndarray:
    """Literal ``singleParticle.gen_form_factors`` lines 389--440."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    resolved_NG1, resolved_NG2 = _validate_cutoffs(params, NG1=NG1, NG2=NG2)
    coefficient_shape = (
        params.N1,
        params.N2,
        2 * params.Ng1,
        2 * params.Ng2,
        4,
    )
    bra = np.asarray(c, dtype=np.complex128)
    ket = np.asarray(cp, dtype=np.complex128)
    if bra.shape != coefficient_shape:
        raise ValueError(f"c must have shape {coefficient_shape}, got {bra.shape}")
    if ket.shape != coefficient_shape:
        raise ValueError(f"cp must have shape {coefficient_shape}, got {ket.shape}")
    if not np.all(np.isfinite(bra)) or not np.all(np.isfinite(ket)):
        raise ValueError("c and cp must contain only finite values")

    N1 = params.N1
    N2 = params.N2
    Ng1 = params.Ng1
    Ng2 = params.Ng2
    form = np.zeros(
        (N1, N2, N1, N2, 2 * (resolved_NG1 + 1), 2 * (resolved_NG2 + 1)),
        dtype=complex,
    )
    for G1 in np.arange(-resolved_NG1 - 1, resolved_NG1 + 1):
        for G2 in np.arange(-resolved_NG2 - 1, resolved_NG2 + 1):
            cp_copy = np.roll(ket, (G1, G2), axis=(2, 3))
            if G1 > 0:
                cp_copy[:, :, 0:G1, :, :] = 0
            if G1 < 0:
                cp_copy[:, :, 2 * Ng1 + G1 :, :, :] = 0
            if G2 > 0:
                cp_copy[:, :, :, 0:G2, :] = 0
            if G2 < 0:
                cp_copy[:, :, :, 2 * Ng2 + G2 :, :] = 0
            form[:, :, :, :, G1 + resolved_NG1 + 1, G2 + resolved_NG2 + 1] = (
                np.einsum("abefz,cdefz->abcd", np.conj(bra), cp_copy, optimize=True)
            )

    form_new = np.zeros(
        (N1, N2, N1, N2, 2 * resolved_NG1, 2 * resolved_NG2),
        dtype=complex,
    )
    for ik1 in range(N1):
        for ik2 in range(N2):
            for iq1 in range(N1):
                for iq2 in range(N2):
                    form_new[ik1, ik2, iq1, iq2, :, :] = form[
                        ik1,
                        ik2,
                        (ik1 + iq1) % N1,
                        (ik2 + iq2) % N2,
                        2 - (ik1 + iq1) // N1 : 2 * resolved_NG1
                        + 2
                        - (ik1 + iq1) // N1,
                        2 - (ik2 + iq2) // N2 : 2 * resolved_NG2
                        + 2
                        - (ik2 + iq2) // N2,
                    ]
    return np.flip(form_new, axis=(4, 5))


def gen_full_form_factors(
    params: TBGZeroFieldCompanionSingleParticleParams,
    coeff: np.ndarray,
    *,
    NG1: int,
    NG2: int,
) -> np.ndarray:
    """Build all intravalley/band form factors exactly as ``mainProgram.py``."""

    resolved_NG1, resolved_NG2 = _validate_cutoffs(params, NG1=NG1, NG2=NG2)
    bands = params.active_band_count
    expected_shape = (
        params.N1,
        params.N2,
        2 * params.Ng1,
        2 * params.Ng2,
        2,
        bands,
        4,
    )
    coefficients = np.asarray(coeff, dtype=np.complex128)
    if coefficients.shape != expected_shape:
        raise ValueError(f"coeff must have shape {expected_shape}, got {coefficients.shape}")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coeff must contain only finite values")
    form = np.zeros(
        _form_shape(params, NG1=resolved_NG1, NG2=resolved_NG2),
        dtype=complex,
    )
    for band1 in range(bands):
        for band2 in range(bands):
            for tau in range(2):
                form[:, :, :, :, :, :, tau, band1, band2] = gen_form_factors(
                    params,
                    coefficients[..., tau, band1, :],
                    coefficients[..., tau, band2, :],
                    NG1=resolved_NG1,
                    NG2=resolved_NG2,
                )
    return form


def main_program_realify_form(
    form: np.ndarray,
) -> tuple[np.ndarray, Literal["real", "complex"], float]:
    """Apply the exact ``mainProgram.py`` ``1e-9`` real/complex branch."""

    array = np.asarray(form)
    if array.dtype.kind not in "fc":
        raise TypeError("form must be a real or complex floating array")
    if not np.all(np.isfinite(array)):
        raise ValueError("form must contain only finite values")
    max_abs_imag = _max_abs_imag(array)
    if max_abs_imag > TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_REAL_THRESHOLD:
        return np.array(array, dtype=np.complex128, order="C", copy=True), "complex", max_abs_imag
    return np.array(np.real(array), dtype=np.float64, order="C", copy=True), "real", max_abs_imag


def gen_H_SP(
    params: TBGZeroFieldCompanionSingleParticleParams,
    sp_energy_ev: np.ndarray,
) -> np.ndarray:
    """Literal ``routines.gen_H_SP`` lines 6--22."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    expected_shape = (
        params.N1,
        params.N2,
        2,
        params.active_band_count,
    )
    energy = np.asarray(sp_energy_ev, dtype=np.float64)
    if energy.shape != expected_shape:
        raise ValueError(f"sp_energy_ev must have shape {expected_shape}, got {energy.shape}")
    if not np.all(np.isfinite(energy)):
        raise ValueError("sp_energy_ev must contain only finite values")
    kron = np.eye(2)
    kronband = np.eye(2 * params.n_active)
    idspin = np.ones(2)
    H_SP = np.einsum("ab,kKta->kKtab", kronband, energy)
    H_SP = np.einsum("s,tT,kKtab->kKstaTb", idspin, kron, H_SP, optimize=True)
    return np.reshape(
        H_SP,
        (params.N1, params.N2, 2, 4 * params.n_active, 4 * params.n_active),
    )


def gen_M_tVE(
    params: TBGZeroFieldCompanionSingleParticleParams,
    form: np.ndarray,
    intFT_screened_ev: np.ndarray,
    *,
    exchange: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal ``routines.gen_M_tVE`` including the source real branch."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    if not isinstance(exchange, bool):
        raise TypeError("exchange must be bool")
    interaction = np.asarray(intFT_screened_ev, dtype=np.float64)
    if interaction.ndim != 4 or interaction.shape[:2] != (params.N1, params.N2):
        raise ValueError(
            "intFT_screened_ev must have shape [N1,N2,2*NG1,2*NG2]"
        )
    if interaction.shape[2] % 2 or interaction.shape[3] % 2:
        raise ValueError("intFT_screened_ev reciprocal dimensions must be even")
    NG1 = interaction.shape[2] // 2
    NG2 = interaction.shape[3] // 2
    _validate_cutoffs(params, NG1=NG1, NG2=NG2)
    if not np.all(np.isfinite(interaction)) or np.any(interaction < 0.0):
        raise ValueError("intFT_screened_ev must contain finite nonnegative values")
    effective_form, form_branch, _max_abs_imag = main_program_realify_form(form)
    expected_form_shape = _form_shape(params, NG1=NG1, NG2=NG2)
    if effective_form.shape != expected_form_shape:
        raise ValueError(
            f"form must have shape {expected_form_shape}, got {effective_form.shape}"
        )

    M = np.einsum(
        "qQgG,kKqQgGtab->kKqQgGtab",
        interaction,
        effective_form.conj(),
        optimize=True,
    )
    if not exchange:
        VE = np.zeros(_tve_shape(params), dtype=complex)
        return M, VE

    N1 = params.N1
    N2 = params.N2
    nactive = params.n_active
    dtype: type[np.float64] | type[np.complex128]
    dtype = np.complex128 if form_branch == "complex" else np.float64
    VE = np.empty(
        (
            N1,
            N2,
            2 * nactive,
            2 * nactive,
            N1,
            N2,
            2 * nactive,
            2 * nactive,
            2,
            2,
        ),
        dtype=dtype,
    )
    for ik1 in range(N1):
        VE[ik1, ...] = np.einsum(
            "KqQgGtab,KqQgGTdc->KadqQbctT",
            effective_form[ik1, ...],
            M[ik1, ...],
            optimize=True,
        ).copy()
    for ik1 in range(N1):
        VE[ik1, :, ...] = np.roll(VE[ik1, :, ...], ik1, axis=3)
    for ik2 in range(N2):
        VE[:, ik2, ...] = np.roll(VE[:, ik2, ...], ik2, axis=4)
    VE.shape = _tve_shape(params)
    return M, VE


def calc_fock_matrix(
    params: TBGZeroFieldCompanionSingleParticleParams,
    density_delta: np.ndarray,
    form: np.ndarray,
    M_ev: np.ndarray,
    tVE_ev: np.ndarray,
) -> TBGZeroFieldCompanionHFAction:
    """Apply exact source Hartree/Fock contractions and fail closed."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    density_shape = _hamiltonian_shape(params)
    density = np.asarray(density_delta)
    if density.shape != density_shape:
        raise ValueError(
            f"density_delta must have shape {density_shape}, got {density.shape}"
        )
    if not np.all(np.isfinite(density)):
        raise ValueError("density_delta must contain only finite values")
    density = np.asarray(density, dtype=np.complex128)
    density_residual = _max_hermiticity_residual(density)
    if density_residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
        raise ValueError(
            "density_delta is materially non-Hermitian: "
            f"residual={density_residual:.6e}"
        )

    form_array, _form_branch, _form_imag = main_program_realify_form(form)
    M = np.asarray(M_ev)
    if form_array.ndim != 9 or form_array.shape != M.shape:
        raise ValueError("form and M_ev must have identical nine-dimensional shapes")
    if form_array.shape[:4] != (params.N1, params.N2, params.N1, params.N2):
        raise ValueError("form mesh dimensions do not match params")
    if form_array.shape[6:] != (
        2,
        params.active_band_count,
        params.active_band_count,
    ):
        raise ValueError("form valley/band dimensions do not match params")
    if not np.all(np.isfinite(form_array)) or not np.all(np.isfinite(M)):
        raise ValueError("form and M_ev must contain only finite values")
    tVE = np.asarray(tVE_ev)
    if tVE.shape != _tve_shape(params):
        raise ValueError(f"tVE_ev must have shape {_tve_shape(params)}, got {tVE.shape}")
    if not np.all(np.isfinite(tVE)):
        raise ValueError("tVE_ev must contain only finite values")

    N1 = params.N1
    N2 = params.N2
    nactive = params.n_active
    Pr = np.reshape(
        density,
        (N1, N2, 2, 2, 2 * nactive, 2, 2 * nactive),
    ).copy()

    PrH = np.sum(Pr, axis=2)
    PrH = np.diagonal(PrH, axis1=2, axis2=4)
    MPrH = np.einsum("kKgGtab,kKbat->gG", M[:, :, 0, 0, ...], PrH, optimize=True)
    H_D = np.einsum(
        "kKgGtab,gG->kKabt",
        form_array[:, :, 0, 0, ...],
        MPrH,
        optimize=True,
    )
    H_D = np.einsum("tT,s,kKabt->kKstaTb", np.eye(2), np.ones(2), H_D, optimize=True)
    H_D = np.reshape(H_D, density_shape)

    Pr = np.transpose(Pr, (2, 0, 1, 6, 4, 5, 3))
    Pr = np.reshape(Pr, (2, N1 * N2 * 4 * nactive**2, 4))
    H_E = -np.einsum("abc,sbc->asc", tVE, Pr, optimize=True)
    H_E = np.reshape(H_E, (N1, N2, 2 * nactive, 2 * nactive, 2, 2, 2))
    H_E = np.transpose(H_E, axes=(0, 1, 4, 5, 2, 6, 3))
    H_E = np.reshape(H_E, density_shape)
    H_interaction = np.asarray(H_D + H_E)
    if not np.all(np.isfinite(H_D)) or not np.all(np.isfinite(H_E)):
        raise ValueError("companion HF action produced nonfinite values")

    residuals = TBGZeroFieldCompanionHFActionResiduals(
        density_hermiticity_max_abs=density_residual,
        H_D_hermiticity_max_abs_ev=_max_hermiticity_residual(H_D),
        H_E_hermiticity_max_abs_ev=_max_hermiticity_residual(H_E),
        H_interaction_hermiticity_max_abs_ev=_max_hermiticity_residual(H_interaction),
    )
    if (
        residuals.H_interaction_hermiticity_max_abs_ev
        > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD
    ):
        raise ValueError(
            "companion HF action is materially non-Hermitian: residual="
            f"{residuals.H_interaction_hermiticity_max_abs_ev:.6e}"
        )
    hashes = TBGZeroFieldCompanionHFActionArrayHashes.from_arrays(
        density_delta=density,
        H_D_ev=H_D,
        H_E_ev=H_E,
        H_interaction_ev=H_interaction,
    )
    return TBGZeroFieldCompanionHFAction(
        params_fingerprint=params.fingerprint,
        density_delta=density,
        H_D_ev=H_D,
        H_E_ev=H_E,
        H_interaction_ev=H_interaction,
        residuals=residuals,
        array_hashes=hashes,
    )


def calc_E(
    params: TBGZeroFieldCompanionSingleParticleParams,
    projector: np.ndarray,
    reference: np.ndarray,
    sp_energy_ev: np.ndarray,
    action: TBGZeroFieldCompanionHFAction,
) -> TBGZeroFieldCompanionHFEnergy:
    """Literal stored-orientation ``calc_E`` with fail-closed imaginary energy."""

    if not isinstance(params, TBGZeroFieldCompanionSingleParticleParams):
        raise TypeError("params must be TBGZeroFieldCompanionSingleParticleParams")
    if not isinstance(action, TBGZeroFieldCompanionHFAction):
        raise TypeError("action must be TBGZeroFieldCompanionHFAction")
    if action.params_fingerprint != params.fingerprint:
        raise ValueError("action is not bound to params")
    shape = _hamiltonian_shape(params)
    P = np.asarray(projector, dtype=np.complex128)
    P_ref = np.asarray(reference, dtype=np.complex128)
    if P.shape != shape or P_ref.shape != shape:
        raise ValueError(f"projector and reference must both have shape {shape}")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(P_ref)):
        raise ValueError("projector and reference must contain only finite values")
    density_delta = np.asarray(P - P_ref)
    if _canonical_array_sha256(density_delta) != action.array_hashes.density_delta:
        raise ValueError("action was not evaluated on projector-reference")
    energy_shape = (
        params.N1,
        params.N2,
        2,
        params.active_band_count,
    )
    sp_energy = np.asarray(sp_energy_ev, dtype=np.float64)
    if sp_energy.shape != energy_shape:
        raise ValueError(f"sp_energy_ev must have shape {energy_shape}, got {sp_energy.shape}")
    if not np.all(np.isfinite(sp_energy)):
        raise ValueError("sp_energy_ev must contain only finite values")

    Psplit = np.reshape(
        P,
        (
            params.N1,
            params.N2,
            2,
            2,
            2 * params.n_active,
            2,
            2 * params.n_active,
        ),
    ).copy()
    E_kin = np.einsum("kKta,kKstata->", sp_energy, Psplit, optimize=True)
    E_D = 0.5 * np.einsum(
        "kpsAB,kpsAB->",
        action.H_D_ev,
        density_delta,
        optimize=True,
    )
    E_E = 0.5 * np.einsum(
        "kpsAB,kpsAB->",
        action.H_E_ev,
        density_delta,
        optimize=True,
    )
    complex_components = np.asarray([E_kin + E_D + E_E, E_kin, E_D, E_E])
    if not np.all(np.isfinite(complex_components)):
        raise ValueError("companion energy contraction produced nonfinite values")
    total_imag = float(abs(np.imag(complex_components[0])))
    max_component_imag = float(np.max(np.abs(np.imag(complex_components))))
    if total_imag > TBG_ZERO_FIELD_COMPANION_HF_ACTION_IMAG_ENERGY_THRESHOLD_EV:
        raise ValueError(
            "companion finite-system energy has a material imaginary defect: "
            f"{total_imag:.6e} eV"
        )
    return TBGZeroFieldCompanionHFEnergy(
        components_ev=np.real(complex_components),
        total_imag_residual_ev=total_imag,
        max_component_imag_residual_ev=max_component_imag,
        projector_sha256=_canonical_array_sha256(P),
        reference_sha256=_canonical_array_sha256(P_ref),
        action_fingerprint=action.fingerprint,
    )


def prepare_tbg_zero_field_companion_hf_action(
    single_particle: TBGZeroFieldCompanionSingleParticleResult,
    interaction: TBGZeroFieldCompanionInteractionResult,
    *,
    spec: TBGZeroFieldCompanionHFActionSpec | None = None,
) -> TBGZeroFieldCompanionPreparedHFAction:
    """Prepare the isolated Stage4 provider from typed Stage2 and Stage3 only."""

    if not isinstance(single_particle, TBGZeroFieldCompanionSingleParticleResult):
        raise TypeError(
            "single_particle must be TBGZeroFieldCompanionSingleParticleResult"
        )
    if not isinstance(interaction, TBGZeroFieldCompanionInteractionResult):
        raise TypeError("interaction must be TBGZeroFieldCompanionInteractionResult")
    resolved_spec = TBGZeroFieldCompanionHFActionSpec() if spec is None else spec
    if not isinstance(resolved_spec, TBGZeroFieldCompanionHFActionSpec):
        raise TypeError("spec must be TBGZeroFieldCompanionHFActionSpec")
    if single_particle.params != interaction.params:
        raise ValueError("Stage2 and Stage3 params differ")
    if single_particle.params.fingerprint != interaction.params.fingerprint:
        raise ValueError("Stage2 and Stage3 parameter fingerprints differ")
    if single_particle.rlv_geometry.fingerprint != interaction.rlv_geometry.fingerprint:
        raise ValueError("Stage2 and Stage3 reciprocal geometries differ")
    params = single_particle.params
    NG1, NG2 = _validate_cutoffs(
        params,
        NG1=interaction.spec.NG1,
        NG2=interaction.spec.NG2,
    )

    raw_intFT = np.asarray(interaction.intFT_ev, dtype=np.float64)
    screened_intFT = np.asarray(raw_intFT / resolved_spec.epsr)
    if not np.all(np.isfinite(screened_intFT)):
        raise ValueError("screening produced nonfinite intFT values")
    screening_roundtrip = float(
        np.max(np.abs(screened_intFT * resolved_spec.epsr - raw_intFT))
    )
    form_raw = gen_full_form_factors(
        params,
        single_particle.coeff,
        NG1=NG1,
        NG2=NG2,
    )
    form, form_branch, raw_form_max_abs_imag = main_program_realify_form(form_raw)
    M, tVE = gen_M_tVE(
        params,
        form,
        screened_intFT,
        exchange=resolved_spec.exchange,
    )
    H_SP = gen_H_SP(params, single_particle.sp_energy_ev)
    H_SP_residual = _max_hermiticity_residual(H_SP)
    if H_SP_residual > TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD:
        raise ValueError(
            f"H_SP is materially non-Hermitian: residual={H_SP_residual:.6e} eV"
        )
    residuals = TBGZeroFieldCompanionHFPreparationResiduals(
        raw_form_max_abs_imag=raw_form_max_abs_imag,
        effective_form_max_abs_imag=_max_abs_imag(form),
        H_SP_hermiticity_max_abs_ev=H_SP_residual,
        screening_roundtrip_max_abs_ev=screening_roundtrip,
        form_branch=form_branch,
    )
    memory_estimate = TBGZeroFieldCompanionHFMemoryEstimate.from_shapes(
        params,
        NG1=NG1,
        NG2=NG2,
        form_branch=form_branch,
        exchange=resolved_spec.exchange,
    )
    hashes = TBGZeroFieldCompanionHFPreparedArrayHashes.from_arrays(
        form_raw=form_raw,
        form=form,
        screened_intFT_ev=screened_intFT,
        M_ev=M,
        tVE_ev=tVE,
        H_SP_ev=H_SP,
        sp_energy_ev=single_particle.sp_energy_ev,
    )
    return TBGZeroFieldCompanionPreparedHFAction(
        params=params,
        interaction_NG1=NG1,
        interaction_NG2=NG2,
        spec=resolved_spec,
        single_particle_source=single_particle,
        interaction_source=interaction,
        single_particle_fingerprint=single_particle.fingerprint,
        interaction_fingerprint=interaction.fingerprint,
        raw_interaction_array_sha256=interaction.array_hashes.intFT_ev,
        form_raw=form_raw,
        form=form,
        screened_intFT_ev=screened_intFT,
        M_ev=M,
        tVE_ev=tVE,
        H_SP_ev=H_SP,
        sp_energy_ev=single_particle.sp_energy_ev,
        form_branch=form_branch,
        residuals=residuals,
        memory_estimate=memory_estimate,
        provenance=TBGZeroFieldCompanionHFActionProvenance(),
        array_hashes=hashes,
    )


__all__ = [
    "TBGZeroFieldCompanionHFAction",
    "TBGZeroFieldCompanionHFActionArrayHashes",
    "TBGZeroFieldCompanionHFActionProvenance",
    "TBGZeroFieldCompanionHFActionResiduals",
    "TBGZeroFieldCompanionHFActionSpec",
    "TBGZeroFieldCompanionHFEnergy",
    "TBGZeroFieldCompanionHFEvaluation",
    "TBGZeroFieldCompanionHFEvaluationArrayHashes",
    "TBGZeroFieldCompanionHFEvaluationResiduals",
    "TBGZeroFieldCompanionHFMemoryEstimate",
    "TBGZeroFieldCompanionHFPreparationResiduals",
    "TBGZeroFieldCompanionHFPreparedArrayHashes",
    "TBGZeroFieldCompanionPreparedHFAction",
    "TBG_ZERO_FIELD_COMPANION_CALC_E_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_CALC_FOCK_MATRIX_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_FORM_FACTOR_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_GEN_H_SP_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_GEN_M_TVE_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_ARRAY_HASH_SEMANTICS",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_BOOST_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_CLOSURE_ATOL_EV",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_ENERGY_UNITS",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_REAL_THRESHOLD",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_FORM_SOURCE_ATOL",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_HERMITICITY_THRESHOLD",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_IMAG_ENERGY_THRESHOLD_EV",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCHEMA_VERSION",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCOPE",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_SCREENING_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_ACTION_STORED_PROJECTOR_CONVENTION",
    "TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_HF_INPUT_SOURCE_SHA256",
    "TBG_ZERO_FIELD_COMPANION_MAIN_BOOST_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_MAIN_FORM_REFERENCE_LINES",
    "TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_MAIN_PROGRAM_SOURCE_SHA256",
    "TBG_ZERO_FIELD_COMPANION_MAIN_SCREENING_REFERENCE_LINE",
    "TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE",
    "TBG_ZERO_FIELD_COMPANION_ROUTINES_SOURCE_SHA256",
    "calc_E",
    "calc_fock_matrix",
    "gen_H_SP",
    "gen_M_tVE",
    "gen_form_factors",
    "gen_full_form_factors",
    "main_program_realify_form",
    "prepare_tbg_zero_field_companion_hf_action",
]
