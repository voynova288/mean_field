from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from mean_field.core.io import read_json_artifact, write_json_artifact
from mean_field.crpa.grid import build_q_shift_table
from mean_field.crpa.workflow import required_hf_periodic_lg

__all__ = [
    "coverage_policy",
    "load_and_validate_chunks",
    "write_merged_crpa_base_artifacts",
]

_REQUIRED_FILES = (
    "crpa_params.json",
    "chi0_q.npz",
    "dielectric_matrix.npz",
    "effective_epsilon.npz",
    "screened_coulomb.npz",
)
_REQUIRED_PARAM_KEYS = frozenset(
    {
        "theta_deg",
        "lk",
        "lg",
        "q_lg",
        "bands_per_valley",
        "eta_mev",
        "coulomb_params",
        "q_point_count",
        "q_shift_count",
        "metadata",
    }
)
_COULOMB_PARAM_KEYS = frozenset(
    {
        "epsilon_bn",
        "ds_angstrom",
        "graphene_lattice_angstrom",
        "finite_zero_limit",
        "zero_cutoff",
    }
)
_DYNAMIC_METADATA_KEYS = frozenset(
    {
        "k_periodic_max_wrap_shell",
        "periodic_roll_required_lg",
        "periodic_roll_no_alias",
    }
)
_MERGE_LOCAL_METADATA_KEYS = frozenset({"merge_coverage"})
_NPZ_KEYS = {
    "chi0_q.npz": frozenset({"chi0", "q_indices", "q_tilde_real", "q_tilde_imag", "q_shifts"}),
    "dielectric_matrix.npz": frozenset({"epsilon", "epsilon_inv", "q_indices", "q_shifts"}),
    "effective_epsilon.npz": frozenset(
        {
            "effective_epsilon",
            "epsilon_times_bn",
            "q_abs",
            "q_abs_nm_inv",
            "q_real",
            "q_imag",
            "q_indices",
            "q_shifts",
        }
    ),
    "screened_coulomb.npz": frozenset(
        {
            "screened_v",
            "effective_epsilon",
            "q_indices",
            "q_shifts",
            "q_abs_nm_inv",
            "q_vectors_real",
            "q_vectors_imag",
        }
    ),
}


@dataclass(frozen=True)
class _ValidatedChunk:
    path: Path
    params: dict[str, object]
    arrays: dict[str, np.ndarray]


def _load_json(path: Path) -> dict[str, object]:
    payload = read_json_artifact(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _metadata_for_compare(metadata: object) -> dict[str, object]:
    normalized = dict(metadata) if isinstance(metadata, dict) else {}
    normalized.setdefault("legacy_zero_fill_test", False)
    excluded = _DYNAMIC_METADATA_KEYS | _MERGE_LOCAL_METADATA_KEYS
    return {key: value for key, value in normalized.items() if key not in excluded}


def _recursive_values_equivalent(left: object, right: object) -> bool:
    """Compare JSON-native metadata without numeric duck-typing."""

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if left.keys() != right.keys():
            return False
        return all(_recursive_values_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right):
            return False
        return len(left) == len(right) and all(
            _recursive_values_equivalent(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, int) or isinstance(right, int):
        return (
            isinstance(left, int)
            and not isinstance(left, bool)
            and isinstance(right, int)
            and not isinstance(right, bool)
            and left == right
        )
    if isinstance(left, float) or isinstance(right, float):
        return (
            isinstance(left, float)
            and isinstance(right, float)
            and math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)
        )
    if left is None or right is None:
        return left is None and right is None
    return type(left) is type(right) and left == right


def _values_match_for_key(key: str, left: object, right: object) -> bool:
    if key == "metadata":
        return _recursive_values_equivalent(
            _metadata_for_compare(left),
            _metadata_for_compare(right),
        )
    return left == right


def _require_plain_int(params: Mapping[str, object], key: str, chunk: Path) -> int:
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Chunk {chunk} parameter {key} must be an integer, got {value!r}")
    return int(value)


def _require_finite_number(params: Mapping[str, object], key: str, chunk: Path) -> float:
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Chunk {chunk} parameter {key} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Chunk {chunk} parameter {key} must be finite, got {value!r}")
    return result


def _validate_chunk_params(params: dict[str, object], chunk: Path) -> tuple[int, int, int, int]:
    actual_keys = frozenset(params)
    if actual_keys != _REQUIRED_PARAM_KEYS:
        missing = sorted(_REQUIRED_PARAM_KEYS.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(_REQUIRED_PARAM_KEYS))
        raise ValueError(
            f"Chunk {chunk} crpa_params.json schema mismatch: missing={missing}, unexpected={unexpected}"
        )

    _require_finite_number(params, "theta_deg", chunk)
    eta_mev = _require_finite_number(params, "eta_mev", chunk)
    if eta_mev < 0.0:
        raise ValueError(f"Chunk {chunk} eta_mev must be nonnegative, got {eta_mev}")

    lk = _require_plain_int(params, "lk", chunk)
    lg = _require_plain_int(params, "lg", chunk)
    q_lg = _require_plain_int(params, "q_lg", chunk)
    q_shift_count = _require_plain_int(params, "q_shift_count", chunk)
    q_point_count = _require_plain_int(params, "q_point_count", chunk)
    if lk <= 0:
        raise ValueError(f"Chunk {chunk} lk must be positive, got {lk}")
    for key, value in (("lg", lg), ("q_lg", q_lg)):
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"Chunk {chunk} {key} must be a positive odd integer, got {value}")
    if q_shift_count != q_lg * q_lg:
        raise ValueError(
            f"Chunk {chunk} q_shift_count={q_shift_count} does not match q_lg^2={q_lg * q_lg}"
        )
    if q_point_count <= 0 or q_point_count > lk * lk:
        raise ValueError(
            f"Chunk {chunk} q_point_count must be in [1, {lk * lk}], got {q_point_count}"
        )

    bands_per_valley = params["bands_per_valley"]
    if bands_per_valley is not None:
        if isinstance(bands_per_valley, bool) or not isinstance(bands_per_valley, int):
            raise ValueError(
                f"Chunk {chunk} bands_per_valley must be null or an integer, got {bands_per_valley!r}"
            )
        if bands_per_valley <= 0:
            raise ValueError(
                f"Chunk {chunk} bands_per_valley must be positive when present, got {bands_per_valley}"
            )

    if not isinstance(params["metadata"], dict):
        raise ValueError(f"Chunk {chunk} metadata must be a JSON object")
    metadata = params["metadata"]
    dynamic_present = {key: key in metadata for key in _DYNAMIC_METADATA_KEYS}
    if any(dynamic_present.values()) and not all(dynamic_present.values()):
        raise ValueError(
            f"Chunk {chunk} dynamic metadata triple must be all missing or all present; "
            f"presence={dynamic_present}"
        )
    if all(dynamic_present.values()):
        max_wrap = metadata["k_periodic_max_wrap_shell"]
        required_lg = metadata["periodic_roll_required_lg"]
        no_alias = metadata["periodic_roll_no_alias"]
        if type(max_wrap) is not int:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata k_periodic_max_wrap_shell must be an integer"
            )
        if type(required_lg) is not int:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata periodic_roll_required_lg must be an integer"
            )
        if type(no_alias) is not bool:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata periodic_roll_no_alias must be a boolean"
            )
        if max_wrap < 0:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata k_periodic_max_wrap_shell must be nonnegative"
            )
        owner_required_lg = required_hf_periodic_lg(q_lg, max_wrap_shell=max_wrap)
        if required_lg != owner_required_lg:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata periodic_roll_required_lg={required_lg} "
                f"must equal owner formula q_lg + 2*max_wrap={owner_required_lg}"
            )
        expected_no_alias = lg >= required_lg
        if no_alias is not expected_no_alias:
            raise ValueError(
                f"Chunk {chunk} dynamic metadata periodic_roll_no_alias={no_alias} "
                f"must equal (lg >= periodic_roll_required_lg)={expected_no_alias}"
            )
    coulomb = params["coulomb_params"]
    if not isinstance(coulomb, dict):
        raise ValueError(f"Chunk {chunk} coulomb_params must be a JSON object")
    actual_coulomb_keys = frozenset(coulomb)
    if actual_coulomb_keys != _COULOMB_PARAM_KEYS:
        missing = sorted(_COULOMB_PARAM_KEYS.difference(actual_coulomb_keys))
        unexpected = sorted(actual_coulomb_keys.difference(_COULOMB_PARAM_KEYS))
        raise ValueError(
            f"Chunk {chunk} coulomb_params schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    epsilon_bn = _require_finite_number(coulomb, "epsilon_bn", chunk)
    ds_angstrom = _require_finite_number(coulomb, "ds_angstrom", chunk)
    graphene_lattice = _require_finite_number(coulomb, "graphene_lattice_angstrom", chunk)
    zero_cutoff = _require_finite_number(coulomb, "zero_cutoff", chunk)
    finite_zero_limit = coulomb["finite_zero_limit"]
    if epsilon_bn <= 0.0:
        raise ValueError(f"Chunk {chunk} epsilon_bn must be finite and positive, got {epsilon_bn}")
    if ds_angstrom < 0.0:
        raise ValueError(f"Chunk {chunk} ds_angstrom must be finite and nonnegative, got {ds_angstrom}")
    if graphene_lattice <= 0.0:
        raise ValueError(
            f"Chunk {chunk} graphene_lattice_angstrom must be finite and positive, got {graphene_lattice}"
        )
    if zero_cutoff < 0.0:
        raise ValueError(f"Chunk {chunk} zero_cutoff must be finite and nonnegative, got {zero_cutoff}")
    if not isinstance(finite_zero_limit, bool):
        raise ValueError(
            f"Chunk {chunk} finite_zero_limit must be a boolean, got {finite_zero_limit!r}"
        )
    return lk, q_lg, q_shift_count, q_point_count


def _load_npz_strict(path: Path, expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        actual_keys = frozenset(payload.files)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys.difference(actual_keys))
            unexpected = sorted(actual_keys.difference(expected_keys))
            raise ValueError(
                f"NPZ labels mismatch for {path}: missing={missing}, unexpected={unexpected}"
            )
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _require_shape(array: np.ndarray, expected: tuple[int, ...], label: str, chunk: Path) -> None:
    if array.shape != expected:
        raise ValueError(f"Chunk {chunk} array {label} has shape {array.shape}, expected {expected}")


def _validate_array_dtype_and_finite(
    array: np.ndarray,
    kind: str,
    label: str,
    chunk: Path,
) -> None:
    if kind == "integer":
        valid_dtype = np.issubdtype(array.dtype, np.integer)
    elif kind == "floating":
        valid_dtype = np.issubdtype(array.dtype, np.floating)
    elif kind == "complex":
        valid_dtype = np.issubdtype(array.dtype, np.complexfloating)
    else:
        raise AssertionError(f"Unknown array kind {kind!r}")
    if not valid_dtype:
        raise ValueError(f"Chunk {chunk} array {label} must have {kind} dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Chunk {chunk} array {label} must contain only finite values")


def _require_array_equal(left: np.ndarray, right: np.ndarray, label: str, chunk: Path) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"Chunk {chunk} has inconsistent redundant array {label}")


def _require_tight_allclose(left: np.ndarray, right: np.ndarray, label: str, chunk: Path) -> None:
    if not np.allclose(left, right, rtol=1.0e-13, atol=0.0):
        max_abs = float(np.max(np.abs(left - right)))
        raise ValueError(
            f"Chunk {chunk} has inconsistent derived array {label}; max_abs_difference={max_abs:.17g}"
        )


def _load_and_validate_chunk(chunk: Path) -> _ValidatedChunk:
    chunk = Path(chunk)
    missing_files = [name for name in _REQUIRED_FILES if not (chunk / name).is_file()]
    if missing_files:
        raise ValueError(f"Chunk {chunk} is missing required files: {missing_files}")

    params = _load_json(chunk / "crpa_params.json")
    _lk, _q_lg, q_shift_count, q_point_count = _validate_chunk_params(params, chunk)
    npz_arrays = {
        filename: _load_npz_strict(chunk / filename, expected_keys)
        for filename, expected_keys in _NPZ_KEYS.items()
    }

    for filename, arrays in npz_arrays.items():
        for key in ("q_indices", "q_shifts"):
            _validate_array_dtype_and_finite(arrays[key], "integer", f"{filename}:{key}", chunk)
        _require_shape(arrays["q_indices"], (q_point_count, 2), f"{filename}:q_indices", chunk)
        _require_shape(arrays["q_shifts"], (q_shift_count, 2), f"{filename}:q_shifts", chunk)

    q_indices = npz_arrays["chi0_q.npz"]["q_indices"]
    q_shifts = npz_arrays["chi0_q.npz"]["q_shifts"]
    canonical_q_shifts = build_q_shift_table(_q_lg)[1]
    for filename, arrays in npz_arrays.items():
        if not np.array_equal(q_indices, arrays["q_indices"]):
            raise ValueError(f"Chunk {chunk} has inconsistent q_indices in {filename}")
        if not np.array_equal(q_shifts, arrays["q_shifts"]):
            raise ValueError(f"Chunk {chunk} has inconsistent q_shifts in {filename}")
        if not np.array_equal(canonical_q_shifts, arrays["q_shifts"]):
            raise ValueError(
                f"Chunk {chunk} {filename}:q_shifts does not match canonical build_q_shift_table({_q_lg})"
            )

    chi0_arrays = npz_arrays["chi0_q.npz"]
    for key in ("q_tilde_real", "q_tilde_imag"):
        _validate_array_dtype_and_finite(chi0_arrays[key], "floating", f"chi0_q.npz:{key}", chunk)
        _require_shape(chi0_arrays[key], (q_point_count,), f"chi0_q.npz:{key}", chunk)
    _validate_array_dtype_and_finite(chi0_arrays["chi0"], "complex", "chi0_q.npz:chi0", chunk)
    _require_shape(
        chi0_arrays["chi0"],
        (q_point_count, q_shift_count, q_shift_count),
        "chi0_q.npz:chi0",
        chunk,
    )

    dielectric_arrays = npz_arrays["dielectric_matrix.npz"]
    for key in ("epsilon", "epsilon_inv"):
        _validate_array_dtype_and_finite(
            dielectric_arrays[key], "complex", f"dielectric_matrix.npz:{key}", chunk
        )
        _require_shape(
            dielectric_arrays[key],
            (q_point_count, q_shift_count, q_shift_count),
            f"dielectric_matrix.npz:{key}",
            chunk,
        )

    effective_arrays = npz_arrays["effective_epsilon.npz"]
    for key in ("effective_epsilon", "epsilon_times_bn", "q_abs", "q_abs_nm_inv", "q_real", "q_imag"):
        _validate_array_dtype_and_finite(
            effective_arrays[key], "floating", f"effective_epsilon.npz:{key}", chunk
        )
        _require_shape(
            effective_arrays[key],
            (q_point_count, q_shift_count),
            f"effective_epsilon.npz:{key}",
            chunk,
        )

    screened_arrays = npz_arrays["screened_coulomb.npz"]
    _validate_array_dtype_and_finite(
        screened_arrays["screened_v"], "complex", "screened_coulomb.npz:screened_v", chunk
    )
    _require_shape(
        screened_arrays["screened_v"],
        (q_point_count, q_shift_count, q_shift_count),
        "screened_coulomb.npz:screened_v",
        chunk,
    )
    for key in ("effective_epsilon", "q_abs_nm_inv", "q_vectors_real", "q_vectors_imag"):
        _validate_array_dtype_and_finite(
            screened_arrays[key], "floating", f"screened_coulomb.npz:{key}", chunk
        )
        _require_shape(
            screened_arrays[key],
            (q_point_count, q_shift_count),
            f"screened_coulomb.npz:{key}",
            chunk,
        )

    if np.any(effective_arrays["q_abs"] < 0.0):
        raise ValueError(f"Chunk {chunk} effective_epsilon.npz:q_abs must be nonnegative")
    if np.any(effective_arrays["q_abs_nm_inv"] < 0.0):
        raise ValueError(f"Chunk {chunk} effective_epsilon.npz:q_abs_nm_inv must be nonnegative")
    if np.any(screened_arrays["q_abs_nm_inv"] < 0.0):
        raise ValueError(f"Chunk {chunk} screened_coulomb.npz:q_abs_nm_inv must be nonnegative")

    _require_array_equal(
        screened_arrays["effective_epsilon"],
        effective_arrays["effective_epsilon"],
        "screened effective_epsilon == effective NPZ effective_epsilon",
        chunk,
    )
    _require_array_equal(
        screened_arrays["q_abs_nm_inv"],
        effective_arrays["q_abs_nm_inv"],
        "screened q_abs_nm_inv == effective NPZ q_abs_nm_inv",
        chunk,
    )
    _require_array_equal(
        screened_arrays["q_vectors_real"],
        effective_arrays["q_real"],
        "screened q_vectors_real == effective NPZ q_real",
        chunk,
    )
    _require_array_equal(
        screened_arrays["q_vectors_imag"],
        effective_arrays["q_imag"],
        "screened q_vectors_imag == effective NPZ q_imag",
        chunk,
    )
    coulomb = params["coulomb_params"]
    if not isinstance(coulomb, dict):
        raise AssertionError("validated coulomb_params unexpectedly ceased to be a mapping")
    epsilon_bn = float(coulomb["epsilon_bn"])
    graphene_lattice_angstrom = float(coulomb["graphene_lattice_angstrom"])
    _require_tight_allclose(
        effective_arrays["epsilon_times_bn"],
        effective_arrays["effective_epsilon"] * epsilon_bn,
        "epsilon_times_bn == effective_epsilon * epsilon_bn",
        chunk,
    )
    _require_tight_allclose(
        effective_arrays["q_abs"],
        np.hypot(effective_arrays["q_real"], effective_arrays["q_imag"]),
        "q_abs == hypot(q_real, q_imag)",
        chunk,
    )
    _require_tight_allclose(
        effective_arrays["q_abs_nm_inv"],
        effective_arrays["q_abs"] / (graphene_lattice_angstrom / 10.0),
        "q_abs_nm_inv == q_abs / (graphene_lattice_angstrom / 10)",
        chunk,
    )
    _require_tight_allclose(
        effective_arrays["effective_epsilon"],
        np.real(np.diagonal(dielectric_arrays["epsilon"], axis1=1, axis2=2)),
        "effective_epsilon == real(diag(epsilon))",
        chunk,
    )
    zero_shift_columns = np.flatnonzero(np.all(q_shifts == 0, axis=1))
    if zero_shift_columns.size != 1:
        raise ValueError(
            f"Chunk {chunk} canonical q_shifts must contain exactly one (0, 0) column; "
            f"found {zero_shift_columns.size}"
        )
    zero_shift_column = int(zero_shift_columns[0])
    _require_tight_allclose(
        chi0_arrays["q_tilde_real"],
        effective_arrays["q_real"][:, zero_shift_column],
        "q_tilde_real == q_real at canonical zero shift",
        chunk,
    )
    _require_tight_allclose(
        chi0_arrays["q_tilde_imag"],
        effective_arrays["q_imag"][:, zero_shift_column],
        "q_tilde_imag == q_imag at canonical zero shift",
        chunk,
    )

    arrays = {
        "q_indices": np.array(q_indices, copy=True),
        "q_shifts": np.array(q_shifts, copy=True),
        "q_tilde_real": np.array(chi0_arrays["q_tilde_real"], copy=True),
        "q_tilde_imag": np.array(chi0_arrays["q_tilde_imag"], copy=True),
        "chi0": np.array(chi0_arrays["chi0"], copy=True),
        "epsilon": np.array(dielectric_arrays["epsilon"], copy=True),
        "epsilon_inv": np.array(dielectric_arrays["epsilon_inv"], copy=True),
        "screened_v": np.array(screened_arrays["screened_v"], copy=True),
        "effective_epsilon": np.array(effective_arrays["effective_epsilon"], copy=True),
        "q_real": np.array(effective_arrays["q_real"], copy=True),
        "q_imag": np.array(effective_arrays["q_imag"], copy=True),
        "q_abs": np.array(effective_arrays["q_abs"], copy=True),
        "q_abs_nm_inv": np.array(effective_arrays["q_abs_nm_inv"], copy=True),
    }
    return _ValidatedChunk(path=chunk, params=params, arrays=arrays)


def _aggregate_metadata(validated: tuple[_ValidatedChunk, ...]) -> dict[str, object]:
    metadata_items = [dict(item.params["metadata"]) for item in validated]
    merged = dict(metadata_items[0])
    merged.setdefault("legacy_zero_fill_test", False)
    for key in _MERGE_LOCAL_METADATA_KEYS:
        merged.pop(key, None)
    triple_present = [all(key in metadata for key in _DYNAMIC_METADATA_KEYS) for metadata in metadata_items]
    if any(triple_present) and not all(triple_present):
        raise ValueError("Chunks must either all provide or all omit the dynamic metadata triple")
    if not any(triple_present):
        for key in _DYNAMIC_METADATA_KEYS:
            merged.pop(key, None)
        return merged
    merged["k_periodic_max_wrap_shell"] = max(
        int(metadata["k_periodic_max_wrap_shell"]) for metadata in metadata_items
    )
    merged["periodic_roll_required_lg"] = max(
        int(metadata["periodic_roll_required_lg"]) for metadata in metadata_items
    )
    merged["periodic_roll_no_alias"] = all(
        bool(metadata["periodic_roll_no_alias"]) for metadata in metadata_items
    )
    return merged


def coverage_policy(allow_partial: bool) -> str:
    return "nonempty_valid_subset" if allow_partial else "exact_full_grid"


def load_and_validate_chunks(
    chunks: tuple[Path, ...],
    *,
    allow_partial: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    if not chunks:
        raise ValueError("No chunks supplied")
    validated = tuple(_load_and_validate_chunk(chunk) for chunk in chunks)
    params = dict(validated[0].params)
    params["metadata"] = _aggregate_metadata(validated)
    comparable_keys = (
        "theta_deg",
        "lk",
        "lg",
        "q_lg",
        "bands_per_valley",
        "eta_mev",
        "coulomb_params",
        "q_shift_count",
        "metadata",
    )
    for item in validated[1:]:
        for key in comparable_keys:
            if not _values_match_for_key(key, item.params.get(key), params.get(key)):
                raise ValueError(
                    f"Chunk {item.path} has incompatible {key}: "
                    f"{item.params.get(key)!r} != {params.get(key)!r}"
                )

    q_shifts_ref = validated[0].arrays["q_shifts"]
    for item in validated[1:]:
        if not np.array_equal(q_shifts_ref, item.arrays["q_shifts"]):
            raise ValueError(f"Chunk {item.path} has incompatible q_shifts")

    concatenated = {
        key: np.concatenate([item.arrays[key] for item in validated], axis=0)
        for key in (
            "q_indices",
            "q_tilde_real",
            "q_tilde_imag",
            "chi0",
            "epsilon",
            "epsilon_inv",
            "screened_v",
            "effective_epsilon",
            "q_real",
            "q_imag",
            "q_abs",
            "q_abs_nm_inv",
        )
    }
    q_indices = concatenated["q_indices"]
    lk = int(params["lk"])
    if np.any(q_indices < 0) or np.any(q_indices >= lk):
        bad = q_indices[np.any((q_indices < 0) | (q_indices >= lk), axis=1)]
        raise ValueError(f"q_indices must lie in [0, {lk}) on both axes; invalid rows={bad.tolist()}")
    q_indices = q_indices.astype(np.int64, copy=False)
    concatenated["q_indices"] = q_indices
    flat_indices = q_indices[:, 0] + lk * q_indices[:, 1]
    unique_flat, counts = np.unique(flat_indices, return_counts=True)
    duplicate_flat = unique_flat[counts > 1]
    if duplicate_flat.size:
        raise ValueError(f"Duplicate q coordinates with canonical flat indices {duplicate_flat.tolist()}")
    if flat_indices.size == 0:
        raise ValueError("Merged q coverage must be nonempty")

    expected_flat = np.arange(lk * lk, dtype=int)
    missing_flat = np.setdiff1d(expected_flat, unique_flat, assume_unique=True)
    complete = bool(unique_flat.size == expected_flat.size and missing_flat.size == 0)
    if not allow_partial and not complete:
        raise ValueError(
            f"Default merge requires exact lk^2 coverage; got {unique_flat.size}/{lk * lk} "
            f"q points with missing canonical flat indices {missing_flat.tolist()}"
        )

    order = np.argsort(flat_indices, kind="stable")
    merged = {key: values[order] for key, values in concatenated.items()}
    merged["q_shifts"] = np.asarray(q_shifts_ref, dtype=np.int64)
    coverage = {
        "allow_partial": bool(allow_partial),
        "policy": coverage_policy(bool(allow_partial)),
        "complete": complete,
        "q_point_count": int(flat_indices.size),
        "expected_q_point_count": int(lk * lk),
        "missing_flat_indices": [int(value) for value in missing_flat],
    }
    return params, merged, coverage

def write_merged_crpa_base_artifacts(
    params: dict[str, object],
    arrays: Mapping[str, np.ndarray],
    coverage: Mapping[str, object],
    output_dir: Path,
) -> None:
    q_indices = arrays["q_indices"]
    q_shifts_ref = arrays["q_shifts"]
    q_tilde_real = arrays["q_tilde_real"]
    q_tilde_imag = arrays["q_tilde_imag"]
    chi0 = arrays["chi0"]
    epsilon = arrays["epsilon"]
    epsilon_inv = arrays["epsilon_inv"]
    screened_v = arrays["screened_v"]
    effective = arrays["effective_epsilon"]
    q_real = arrays["q_real"]
    q_imag = arrays["q_imag"]
    q_abs = arrays["q_abs"]
    q_abs_nm_inv = arrays["q_abs_nm_inv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    params["q_point_count"] = int(q_indices.shape[0])
    params["q_shift_count"] = int(q_shifts_ref.shape[0])
    metadata = dict(params["metadata"])
    metadata["merge_coverage"] = dict(coverage)
    params["metadata"] = metadata
    write_json_artifact(params, output_dir / "crpa_params.json")

    _save_npz(
        output_dir / "chi0_q.npz",
        chi0=chi0,
        q_indices=q_indices,
        q_tilde_real=q_tilde_real,
        q_tilde_imag=q_tilde_imag,
        q_shifts=q_shifts_ref,
    )
    _save_npz(
        output_dir / "dielectric_matrix.npz",
        epsilon=epsilon,
        epsilon_inv=epsilon_inv,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
    )
    coulomb_params = params["coulomb_params"]
    if not isinstance(coulomb_params, dict):
        raise ValueError("Validated coulomb_params unexpectedly ceased to be a mapping")
    epsilon_bn = float(coulomb_params["epsilon_bn"])
    _save_npz(
        output_dir / "effective_epsilon.npz",
        effective_epsilon=effective,
        epsilon_times_bn=effective * epsilon_bn,
        q_abs=q_abs,
        q_abs_nm_inv=q_abs_nm_inv,
        q_real=q_real,
        q_imag=q_imag,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
    )
    _save_npz(
        output_dir / "screened_coulomb.npz",
        screened_v=screened_v,
        effective_epsilon=effective,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
        q_abs_nm_inv=q_abs_nm_inv,
        q_vectors_real=q_real,
        q_vectors_imag=q_imag,
    )
