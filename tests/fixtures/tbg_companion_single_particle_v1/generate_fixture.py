#!/usr/bin/env python3
"""Generate the v1 oracle from the exact pinned ``reference/TBG-HF`` source.

The reference module is not imported when this file is imported.  It is loaded
only through ``main()`` after all pinned input/source identities have passed.
The generated NPZ is deterministic: array keys are sorted, NPY format v1.0 is
used, and ZIP timestamps and permissions are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import zipfile

import numpy as np


REFERENCE_REPOSITORY = "reference/TBG-HF"
REFERENCE_COMMIT = "0d2a3d742aa901fa45ce46690c1385887165f58c"
DEFAULT_INPUT_SOURCE = "reference/TBG-HF/int_input.json"
DEFAULT_INPUT_SOURCE_SHA256 = (
    "c143c294ad95cf94d91cfbabd0437556e5c2a342850d54484c9b47caaf84b4de"
)
SINGLE_PARTICLE_SOURCE = "reference/TBG-HF/singleParticle.py"
SINGLE_PARTICLE_SOURCE_SHA256 = (
    "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
)
CONSTANTS_SOURCE = "reference/TBG-HF/constants.py"
CONSTANTS_SOURCE_SHA256 = (
    "8d25bcccd54e41207788ff4a9e1b934a50347fabdad46ba44408fc535573ec62"
)
ARRAY_HASH_CONVENTION = (
    "sha256_little_endian_int64_shape_then_C_order_little_endian_array_bytes"
)
ARRAY_HASH_SEMANTICS = "artifact_integrity_only_not_cross_eigensolver_parity"
POINTWISE_GAUGE_WARNING = (
    "pinned_singleParticle.py_lines_177-188_apply_a_pointwise_nondegenerate_"
    "C2T_phase_choice_only;insufficient_inside_degenerate_subspaces"
)
RESIDUAL_GAUGE_AMBIGUITY = (
    "residual_real_sign_per_nondegenerate_state_and_U(N)_rotation_within_"
    "degenerate_subspaces"
)
CUSTOM_INPUT: dict[str, int | float] = {
    "N1": 2,
    "N2": 3,
    "Ng1": 2,
    "Ng2": 2,
    "n_active": 1,
    "theta": 1.08,
    "wAA": 0.07,
    "wAB": 0.11,
    "strain": 0.003,
    "varphi": 17.0,
}
SOURCE_LINE_RANGES = {
    "gen_RLVs": "20-49",
    "gen_moire_hamiltonian": "51-111",
    "gen_coeff": "113-202",
    "pointwise_C2T_gauge": "177-188",
    "Kprime_time_reversal": "190-200",
    "C2T_symmetry": "265-279",
}
FIXTURE_SCHEMA = "mean_field.tbg.companion_single_particle.source_fixture"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_FILENAME = "fixture.npz"
MANIFEST_FILENAME = "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _little_endian_array(values: np.ndarray, *, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned source file is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Pinned source identity drift for {path}: expected {expected_sha256}, got {actual}"
        )


def _verify_reference_checkout(reference_root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = completed.stdout.strip()
    if actual_commit != REFERENCE_COMMIT:
        raise RuntimeError(
            "Pinned reference commit drift: "
            f"expected {REFERENCE_COMMIT}, got {actual_commit}"
        )


def _module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pinned_single_particle(reference_root: Path) -> ModuleType:
    """Load the exact source with its exact ``constants`` module."""

    constants_path = reference_root / "constants.py"
    source_path = reference_root / "singleParticle.py"
    constants_module = _module_from_file("_pinned_tbg_hf_constants", constants_path)

    sentinel = object()
    previous_constants = sys.modules.get("constants", sentinel)
    sys.modules["constants"] = constants_module
    try:
        return _module_from_file("_pinned_tbg_hf_single_particle", source_path)
    finally:
        if previous_constants is sentinel:
            del sys.modules["constants"]
        else:
            sys.modules["constants"] = previous_constants  # type: ignore[assignment]


def _source_sub_index(
    resolved_input: dict[str, object],
    *,
    b1: np.ndarray,
    b2: np.ndarray,
    ik1: int,
    ik2: int,
) -> np.ndarray:
    """Evaluate pinned ``singleParticle.py`` lines 132-167 literally."""

    N1 = int(resolved_input["N1"])
    N2 = int(resolved_input["N2"])
    Ng1 = int(resolved_input["Ng1"])
    Ng2 = int(resolved_input["Ng2"])
    X = 2 / 3 * b1 + 1 / 3 * b2
    Y = 1 / 3 * b1 - 1 / 3 * b2
    RX1 = np.linalg.norm((Ng1 * b1 - X) - b2 * np.dot(Ng1 * b1 - X, b2) / np.dot(b2, b2))
    RX2 = np.linalg.norm((Ng1 * b2 - X) - b1 * np.dot(Ng1 * b2 - X, b1) / np.dot(b1, b1))
    RY1 = np.linalg.norm((Ng1 * b1 - Y) - b2 * np.dot(Ng1 * b1 - Y, b2) / np.dot(b2, b2))
    RY2 = np.linalg.norm((-Ng1 * b2 - Y) - b1 * np.dot(-Ng1 * b2 - Y, b1) / np.dot(b1, b1))
    radius = np.min((RX1, RX2, RY1, RY2))

    stau = 1
    sub_index = np.array([], dtype=int)
    for g1 in np.arange(-Ng1, Ng1):
        for g2 in np.arange(-Ng2, Ng2):
            indexg = 4 * (g2 + Ng2) + 8 * Ng2 * (g1 + Ng1)
            Q = (
                ik1 * b1 / N1
                + ik2 * b2 / N2
                + g1 * b1
                + g2 * b2
                + stau * 1 / 3 * b1
                - stau * 1 / 3 * b2
            )
            modQ = np.sqrt(Q[0] ** 2 + Q[1] ** 2)
            if modQ < radius - 0.00001:
                sub_index = np.append(sub_index, np.arange(indexg, indexg + 2))
            Q = (
                ik1 * b1 / N1
                + ik2 * b2 / N2
                + g1 * b1
                + g2 * b2
                + stau * 2 / 3 * b1
                + stau * 1 / 3 * b2
            )
            modQ = np.sqrt(Q[0] ** 2 + Q[1] ** 2)
            if modQ < radius - 0.00001:
                sub_index = np.append(sub_index, np.arange(indexg + 2, indexg + 4))
    return _little_endian_array(sub_index, dtype="<i8")


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    _npy_payload(arrays[key]),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_arrays(
    reference: ModuleType,
    resolved_input: dict[str, object],
) -> dict[str, np.ndarray]:
    M1, M2, b1, b2, Etens1, Etens2 = reference.gen_RLVs(resolved_input)
    coeff, sp_energy = reference.gen_coeff(resolved_input)
    U_C2T = reference.C2T_symmetry(resolved_input, coeff)

    arrays: dict[str, np.ndarray] = {
        "parent_h_0_0": _little_endian_array(
            reference.gen_moire_hamiltonian(resolved_input, (0, 0)), dtype="<c16"
        ),
        "parent_h_1_2": _little_endian_array(
            reference.gen_moire_hamiltonian(resolved_input, (1, 2)), dtype="<c16"
        ),
        "rlv_Etens1": _little_endian_array(Etens1, dtype="<f8"),
        "rlv_Etens2": _little_endian_array(Etens2, dtype="<f8"),
        "rlv_M1": _little_endian_array(M1, dtype="<f8"),
        "rlv_M2": _little_endian_array(M2, dtype="<f8"),
        "rlv_b1": _little_endian_array(b1, dtype="<f8"),
        "rlv_b2": _little_endian_array(b2, dtype="<f8"),
        "source_U_C2T": _little_endian_array(U_C2T, dtype="<c16"),
        "source_coeff": _little_endian_array(coeff, dtype="<c16"),
        "source_sp_energy_ev": _little_endian_array(sp_energy, dtype="<f8"),
    }
    for ik1 in range(int(resolved_input["N1"])):
        for ik2 in range(int(resolved_input["N2"])):
            arrays[f"sub_index_{ik1}_{ik2}"] = _source_sub_index(
                resolved_input,
                b1=b1,
                b2=b2,
                ik1=ik1,
                ik2=ik2,
            )
    return arrays


def _manifest(
    *,
    script_path: Path,
    resolved_input: dict[str, object],
    arrays: dict[str, np.ndarray],
    fixture_path: Path,
) -> dict[str, object]:
    array_records = {
        key: {
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
            "shape": list(array.shape),
        }
        for key, array in sorted(arrays.items())
    }
    return {
        "array_hash_convention": ARRAY_HASH_CONVENTION,
        "array_hash_semantics": ARRAY_HASH_SEMANTICS,
        "arrays": array_records,
        "fixture_npz": FIXTURE_FILENAME,
        "fixture_npz_sha256": _sha256_file(fixture_path),
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "generator_script": script_path.name,
        "generator_script_sha256": _sha256_file(script_path),
        "input": dict(CUSTOM_INPUT),
        "pinned_source": {
            "constants": {
                "path": CONSTANTS_SOURCE,
                "sha256": CONSTANTS_SOURCE_SHA256,
            },
            "default_input": {
                "path": DEFAULT_INPUT_SOURCE,
                "sha256": DEFAULT_INPUT_SOURCE_SHA256,
            },
            "reference_commit": REFERENCE_COMMIT,
            "reference_repository": REFERENCE_REPOSITORY,
            "single_particle": {
                "path": SINGLE_PARTICLE_SOURCE,
                "sha256": SINGLE_PARTICLE_SOURCE_SHA256,
            },
            "source_line_ranges": dict(SOURCE_LINE_RANGES),
        },
        "pointwise_gauge_warning": POINTWISE_GAUGE_WARNING,
        "residual_gauge_ambiguity": RESIDUAL_GAUGE_AMBIGUITY,
        "resolved_input": resolved_input,
        "resolved_input_sha256": _json_sha256(resolved_input),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Location of the pinned TBG-HF checkout (identity checks remain mandatory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing fixture and manifest.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    reference_root = (
        args.reference_root.resolve()
        if args.reference_root is not None
        else repository_root / REFERENCE_REPOSITORY
    )
    fixture_path = script_path.parent / FIXTURE_FILENAME
    manifest_path = script_path.parent / MANIFEST_FILENAME
    if not args.force and (fixture_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "Fixture or manifest already exists; use --force only after reviewing the pinned identities"
        )

    input_path = reference_root / "int_input.json"
    source_path = reference_root / "singleParticle.py"
    constants_path = reference_root / "constants.py"
    _verify_reference_checkout(reference_root)
    _verify_file(input_path, DEFAULT_INPUT_SOURCE_SHA256)
    _verify_file(source_path, SINGLE_PARTICLE_SOURCE_SHA256)
    _verify_file(constants_path, CONSTANTS_SOURCE_SHA256)

    resolved_input = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(resolved_input, dict):
        raise TypeError("Pinned int_input.json must decode to a JSON object")
    resolved_input.update(CUSTOM_INPUT)
    # ``intdir`` is an execution-host output path, not a scientific input.
    resolved_input.pop("intdir", None)

    # This is the only pinned-reference import, and it occurs only on explicit CLI execution.
    reference = _load_pinned_single_particle(reference_root)
    arrays = _build_arrays(reference, resolved_input)
    _write_deterministic_npz(fixture_path, arrays)
    manifest = _manifest(
        script_path=script_path,
        resolved_input=resolved_input,
        arrays=arrays,
        fixture_path=fixture_path,
    )
    _write_json(manifest_path, manifest)
    print(f"Wrote {fixture_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
