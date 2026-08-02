#!/usr/bin/env python3
"""Generate the Stage6A four-step SCF oracle from pinned companion source only.

This ignored generator deliberately does not import the Mean_Field Stage4 or
Stage6A implementations.  It verifies the exact Stage4 source fixture,
including its canonical resolved-input digest, and the pinned companion
``routines.py``, ``mainProgram.py``, ``projectors.py``, and
``measure.py`` identities, then uses source ``calc_fock_matrix``/``aufbau`` and
literal transcriptions of ``mainProgram.py`` lines 120--136 and the applicable
``measure.py`` lines 5-14 and 35-72 qualification observables for four ODA
iterations at filling zero.  ``fixture.npz`` and ``manifest.json`` are absent until this script is run
deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
import zipfile
import zlib

import numpy as np

REFERENCE_REPOSITORY = "reference/TBG-HF"
REFERENCE_COMMIT = "0d2a3d742aa901fa45ce46690c1385887165f58c"
ROUTINES_SOURCE = "reference/TBG-HF/routines.py"
ROUTINES_SOURCE_SHA256 = "507e8b9e799f494777d354c9d7d481dd19d6ba42894d393630dd79ef16d02108"
MAIN_PROGRAM_SOURCE = "reference/TBG-HF/mainProgram.py"
MAIN_PROGRAM_SOURCE_SHA256 = "258c97e57164055de3273ba4471cd96be709c1f159e19f73481750c801aed401"
PROJECTORS_SOURCE = "reference/TBG-HF/projectors.py"
PROJECTORS_SOURCE_SHA256 = "d7c7138ddf2107a71c24194ac70790bd27cdc05297ee9cdc997c1dc3882e5ede"
MEASURE_SOURCE = "reference/TBG-HF/measure.py"
MEASURE_SOURCE_SHA256 = "d1a47420400c3381247f4bc8c2e7700935077536b7782a14e52e1d25a1fd516e"

STAGE4_FIXTURE_DIRECTORY = "tests/fixtures/tbg_companion_hf_action_v1"
STAGE4_FIXTURE_NPZ_SHA256 = "fc2e916cf3cfee9a69eebec79d6494106f3dcdbe80d823d19d03fea9031a6eac"
STAGE4_GENERATOR_SHA256 = "3e084ed78a782d9b414836f16baee2e9be37b931f0cf4acc7be6fe69d4c53ce8"
STAGE4_RESOLVED_INPUT_SHA256 = (
    "c7922c6e78d8bf23eb633877b8655d9aa71634b3ba24fe8595ddf6ff17496881"
)
STAGE4_ARRAY_SHA256 = {
    "P": "f458298de7f722d6026a4c3ac68e15c9ad3081a70bc1944deefeb7bf2eef5972",
    "P_ref": "254777bb0dd7d352a6ad6594c4a8d2dfb4788e0ef9fbc673dadda49e8c67da04",
    "source_H_D_ev": "c9bb5c034767cff8265143c85ad3f9d562134f6cc0a2dc3b59cd02950f4b6cd2",
    "source_H_E_ev": "0eb43c7315373a311b7dd58f1086503efa077fc8a7f163412099647c83d8a285",
    "source_H_SP_ev": "3574afb01bbeff87cd16e733e3f8bd86ace1de3d09996817a81f7efdd34cbb6d",
    "source_M_ev": "13e891267f78bea66ebc6559dd0f3990f65db0de52cb6916e038e48e6d5ab5f2",
    "source_coeff": "666d106e52bf130ac341e0c706ae08472e6cdd506a93662bc03bb6fb920c9add",
    "source_energy_ev": "f6f9c96c6750d4f8cea14cd886b02a250ea1a0874782bff8f2e81dc9d4169447",
    "source_form": "ed65815ef55f8f7b4e4e6d99e932fd6b60c8762782c6d3978670be97db323bf0",
    "source_form_branch": "df9d591c4b439d156805c7798ba20eb7cdb044e86caafd1de9609cce8d71eb4d",
    "source_form_raw": "554b56e7d4a0e85da5a9ccf0bf53f8eb614cdcd512450666f3f45551b11635e8",
    "source_raw_intFT_ev": "e70889243a36bc4b294b6264b990f8204b29d5d40db23b9fe6fcb8c7e486db26",
    "source_screened_intFT_ev": "b51a70c2296e262cef65bef525991585bc08e29a9c7a119c62f26dd87aa981f4",
    "source_sp_energy_ev": "7e2628b33b1b1ed5f336238ebbfaa76206ae3c7c3d4f5dab12fad26f9a4df549",
    "source_tVE_ev": "1110dd453f43d6be62797990f39aa3356461e3acbe0f1becdd846df032ea266e",
}

FIXTURE_SCHEMA = "mean_field.tbg.companion_hf_scf.source_fixture"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_FILENAME = "fixture.npz"
MANIFEST_FILENAME = "manifest.json"
ARRAY_HASH_CONVENTION = (
    "sha256_little_endian_int64_shape_then_C_order_little_endian_array_bytes"
)
ARRAY_HASH_SEMANTICS = (
    "source_fixture_integrity_and_same_environment_parity_not_production_result"
)
SOURCE_LINE_RANGES = {
    "routines.calc_E": "81-97",
    "routines.calc_fock_matrix": "99-153",
    "routines.aufbau": "155-198",
    "mainProgram.SCF_and_final_rebuild": "104-167",
    "mainProgram.ODA": "120-136",
    "projectors.average_central": "80-87",
    "measure.boost0_Tp_and_observables": "5-14,35-72",
}
SCF_INPUT = {
    "filling": 0,
    "HF_itermax": 4,
    "HF_itermin": 20,
    "HF_tolerance": 1.0e-8,
    "HF_type": "ODA",
    "ODA_branch_threshold": 1.0e-12,
}


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


def _little_endian_array(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        dtype = np.dtype("<c16")
    elif source.dtype.kind == "f":
        dtype = np.dtype("<f8")
    elif source.dtype.kind in "iu":
        dtype = np.dtype("<i8")
    elif source.dtype.kind == "b":
        dtype = np.dtype("?")
    elif source.dtype.kind == "U":
        dtype = source.dtype
    else:
        raise TypeError(f"Unsupported fixture dtype {source.dtype}")
    converted = np.asarray(source, dtype=dtype)
    return converted if converted.ndim == 0 else np.ascontiguousarray(converted)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned file is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Pinned identity drift for {path}: expected {expected_sha256}, got {actual}"
        )


def _verify_reference_checkout(reference_root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != REFERENCE_COMMIT:
        raise RuntimeError("Pinned companion reference commit drifted")


def _module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned source module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage4_fixture(stage4_root: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest_path = stage4_root / "manifest.json"
    generator_path = stage4_root / "generate_fixture.py"
    fixture_path = stage4_root / "fixture.npz"
    _verify_file(generator_path, STAGE4_GENERATOR_SHA256)
    _verify_file(fixture_path, STAGE4_FIXTURE_NPZ_SHA256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fixture_schema") != "mean_field.tbg.companion_hf_action.source_fixture":
        raise RuntimeError("Stage4 fixture schema drifted")
    if manifest.get("fixture_npz_sha256") != STAGE4_FIXTURE_NPZ_SHA256:
        raise RuntimeError("Stage4 manifest NPZ pin drifted")
    if manifest.get("generator_script_sha256") != STAGE4_GENERATOR_SHA256:
        raise RuntimeError("Stage4 manifest generator pin drifted")
    resolved_input = manifest.get("resolved_input")
    if not isinstance(resolved_input, dict):
        raise RuntimeError("Stage4 manifest resolved_input is not an object")
    if manifest.get("resolved_input_sha256") != STAGE4_RESOLVED_INPUT_SHA256:
        raise RuntimeError("Stage4 manifest resolved_input digest pin drifted")
    if _json_sha256(resolved_input) != STAGE4_RESOLVED_INPUT_SHA256:
        raise RuntimeError("Stage4 manifest resolved_input canonical digest drifted")
    records = manifest.get("arrays")
    if not isinstance(records, dict) or set(records) != set(STAGE4_ARRAY_SHA256):
        raise RuntimeError("Stage4 manifest array inventory drifted")
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if set(arrays) != set(STAGE4_ARRAY_SHA256):
        raise RuntimeError("Stage4 NPZ array inventory drifted")
    for name, expected in STAGE4_ARRAY_SHA256.items():
        record = records[name]
        if not isinstance(record, dict) or record.get("sha256") != expected:
            raise RuntimeError(f"Stage4 manifest hash drift for {name}")
        if _array_sha256(arrays[name]) != expected:
            raise RuntimeError(f"Stage4 NPZ hash drift for {name}")
    return manifest, arrays


def _literal_oda(
    routines: ModuleType,
    pd: dict[str, object],
    H_SP: np.ndarray,
    P_old: np.ndarray,
    P_raw: np.ndarray,
    P_ref: np.ndarray,
    form: np.ndarray,
    M: np.ndarray,
    tVE: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]:
    """Literal ``mainProgram.py`` lines 120--136, including branch order."""

    dP = P_raw - P_old
    c1 = np.real(np.einsum("kKsab,kKsab->", H_SP, dP, optimize=True))
    H_D_dP, H_E_dP = routines.calc_fock_matrix(pd, dP, form, M, tVE)
    H_dP = H_D_dP + H_E_dP
    c01 = np.real(
        np.einsum("kKsab,kKsab->", H_dP, P_old - P_ref, optimize=True)
    )
    c11 = np.real(np.einsum("kKsab,kKsab->", H_dP, dP, optimize=True))
    lin = c1 + c01
    quad = 0.5 * c11
    if lin > 0 and np.abs(lin) > 1.0e-12:
        mixing_lambda = 1.0
        branch = "positive_linear"
        positive_linear = True
    elif quad <= -lin / 2:
        mixing_lambda = 1.0
        branch = "endpoint_quad"
        positive_linear = False
    elif np.abs(lin) < 1.0e-12:
        mixing_lambda = 1.0
        branch = "linear_near_zero"
        positive_linear = False
    else:
        mixing_lambda = -lin / 2 / quad
        branch = "interior"
        positive_linear = False
    coefficients = np.asarray(
        [c1, c01, c11, lin, quad, mixing_lambda],
        dtype=np.float64,
    )
    mixed = (1.0 - mixing_lambda) * P_old + mixing_lambda * P_raw
    return mixed, coefficients, np.stack((H_D_dP, H_E_dP)), branch, positive_linear


def _literal_source_qualification_observables(
    pd: dict[str, object],
    P: np.ndarray,
    energy: np.ndarray,
    eigenvalues: np.ndarray,
    fill_indices: np.ndarray,
    difference: float,
) -> dict[str, np.ndarray]:
    """Literal applicable ``measure.py`` lines 5--14 and 35--72 observables."""

    N1 = int(pd["N1"])
    N2 = int(pd["N2"])
    nactive = int(pd["n_active"])
    boost1 = int(pd["boost1"])
    boost2 = int(pd["boost2"])
    Nk = N1 * N2
    Pex = np.reshape(P, (N1, N2, 2, 2, 2 * nactive, 2, 2 * nactive)).copy()

    mask = np.zeros_like(eigenvalues, dtype=bool)
    mask[fill_indices] = True
    gap = float(np.min(eigenvalues[~mask]) - np.max(eigenvalues[mask]))
    local_occupations = np.einsum(
        "kKsa->kKs",
        np.reshape(mask, (N1, N2, 2, 4 * nactive)).astype(int),
    )
    flavor_occupations = np.real(
        np.einsum("kKstata->st", Pex, optimize=True) / Nk
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

    P_T = np.flip(Pex, axis=(0, 1, 3, 5)).copy()
    P_T = np.roll(P_T, (1, 1), axis=(0, 1))
    P_T = np.roll(P_T, (-boost1, -boost2), axis=(0, 1))
    P_T = np.conj(P_T)
    P_Tp = P_T.copy()
    P_Tp[:, :, :, 0, :, 1, :] = -P_T[:, :, :, 0, :, 1, :]
    P_Tp[:, :, :, 1, :, 0, :] = -P_T[:, :, :, 1, :, 0, :]
    tp_break = float(np.square(np.linalg.norm(Pex - P_Tp)) / Nk)

    return {
        "final_source_energy_ev": _little_endian_array(np.asarray(energy[0])),
        "final_source_gap_ev": _little_endian_array(np.asarray(gap)),
        "final_source_difference": _little_endian_array(np.asarray(difference)),
        "final_source_local_occupations": _little_endian_array(local_occupations),
        "final_source_flavor_occupations": _little_endian_array(flavor_occupations),
        "final_source_valley_polarization": _little_endian_array(
            np.asarray(valley_polarization)
        ),
        "final_source_ivc": _little_endian_array(np.asarray(source_ivc)),
        "final_source_spin_polarization": _little_endian_array(
            np.asarray(spin_polarization)
        ),
        "final_source_tp_break": _little_endian_array(np.asarray(tp_break)),
    }

def _build_arrays(
    routines: ModuleType,
    stage4_manifest: dict[str, object],
    stage4: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    resolved = stage4_manifest["resolved_input"]
    if not isinstance(resolved, dict):
        raise TypeError("Stage4 resolved_input must be an object")
    pd = dict(resolved)
    pd.update(SCF_INPUT)
    P_old = np.array(stage4["P"], dtype=np.complex128, copy=True)
    P_ref = np.array(stage4["P_ref"], dtype=np.complex128, copy=True)
    H_SP = np.asarray(stage4["source_H_SP_ev"], dtype=np.float64)
    sp_energy = np.asarray(stage4["source_sp_energy_ev"], dtype=np.float64)
    form = np.asarray(stage4["source_form"], dtype=np.float64)
    M = np.asarray(stage4["source_M_ev"], dtype=np.float64)
    tVE = np.asarray(stage4["source_tVE_ev"], dtype=np.float64)
    Nk = int(pd["N1"]) * int(pd["N2"])

    arrays: dict[str, np.ndarray] = {
        "initial_P": _little_endian_array(P_old),
        "P_ref": _little_endian_array(P_ref),
    }
    differences: list[float] = []
    coefficients_history: list[np.ndarray] = []
    energies_history: list[np.ndarray] = []
    branches: list[str] = []
    positive_history: list[bool] = []

    for iteration in range(4):
        prefix = f"iter_{iteration:03d}"
        H_D_old, H_E_old = routines.calc_fock_matrix(
            pd, P_old - P_ref, form, M, tVE
        )
        H_old = H_SP + H_D_old + H_E_old
        P_raw, eigenvalues, fill_indices, _ = routines.aufbau(pd, H_old)
        difference = float(np.linalg.norm(P_old - P_raw) / Nk)
        P_mixed, coefficients, dP_action, branch, positive = _literal_oda(
            routines,
            pd,
            H_SP,
            P_old,
            P_raw,
            P_ref,
            form,
            M,
            tVE,
        )
        H_D_mixed, H_E_mixed = routines.calc_fock_matrix(
            pd, P_mixed - P_ref, form, M, tVE
        )
        energy = routines.calc_E(
            pd,
            P_mixed,
            P_ref,
            sp_energy,
            H_D_mixed,
            H_E_mixed,
        )
        arrays.update(
            {
                f"{prefix}_P_old": _little_endian_array(P_old),
                f"{prefix}_H_D_old_ev": _little_endian_array(H_D_old),
                f"{prefix}_H_E_old_ev": _little_endian_array(H_E_old),
                f"{prefix}_H_old_ev": _little_endian_array(H_old),
                f"{prefix}_P_raw": _little_endian_array(P_raw),
                f"{prefix}_eigenvalues_ev": _little_endian_array(eigenvalues),
                f"{prefix}_fill_indices": _little_endian_array(fill_indices),
                f"{prefix}_H_D_dP_ev": _little_endian_array(dP_action[0]),
                f"{prefix}_H_E_dP_ev": _little_endian_array(dP_action[1]),
                f"{prefix}_P_mixed": _little_endian_array(P_mixed),
                f"{prefix}_H_D_mixed_ev": _little_endian_array(H_D_mixed),
                f"{prefix}_H_E_mixed_ev": _little_endian_array(H_E_mixed),
                f"{prefix}_difference": _little_endian_array(np.asarray(difference)),
                f"{prefix}_coefficients": _little_endian_array(coefficients),
                f"{prefix}_branch": _little_endian_array(np.asarray(branch)),
                f"{prefix}_positive_linear": _little_endian_array(np.asarray(positive)),
                f"{prefix}_energy_ev": _little_endian_array(energy),
            }
        )
        differences.append(difference)
        coefficients_history.append(coefficients)
        energies_history.append(np.asarray(energy))
        branches.append(branch)
        positive_history.append(positive)
        P_old = np.array(P_mixed, copy=True)

    final_H_D, final_H_E = routines.calc_fock_matrix(
        pd, P_old - P_ref, form, M, tVE
    )
    final_H = H_SP + final_H_D + final_H_E
    final_raw, final_eigenvalues, final_fill, _ = routines.aufbau(pd, final_H)
    final_energy = routines.calc_E(
        pd, P_old, P_ref, sp_energy, final_H_D, final_H_E
    )
    final_closure_difference = float(np.linalg.norm(P_old - final_raw) / Nk)
    final_source_observables = _literal_source_qualification_observables(
        pd,
        P_old,
        final_energy,
        final_eigenvalues,
        final_fill,
        final_closure_difference,
    )
    arrays.update(
        {
            "history_differences": _little_endian_array(np.asarray(differences)),
            "history_coefficients": _little_endian_array(
                np.asarray(coefficients_history)
            ),
            "history_energies_ev": _little_endian_array(np.asarray(energies_history)),
            "history_branches": _little_endian_array(np.asarray(branches)),
            "history_positive_linear": _little_endian_array(
                np.asarray(positive_history)
            ),
            "final_P_mixed": _little_endian_array(P_old),
            "final_H_D_ev": _little_endian_array(final_H_D),
            "final_H_E_ev": _little_endian_array(final_H_E),
            "final_H_ev": _little_endian_array(final_H),
            "final_P_raw": _little_endian_array(final_raw),
            "final_eigenvalues_ev": _little_endian_array(final_eigenvalues),
            "final_fill_indices": _little_endian_array(final_fill),
            "final_energy_ev": _little_endian_array(final_energy),
            "final_closure_difference": _little_endian_array(
                np.asarray(final_closure_difference)
            ),
            **final_source_observables,
        }
    )
    return arrays


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    payload = array if array.ndim == 0 else np.ascontiguousarray(array)
    np.lib.format.write_array(buffer, payload, version=(1, 0), allow_pickle=False)
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
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment_metadata() -> dict[str, str]:
    return {
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing outputs only after explicit source review.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    reference_root = (
        args.reference_root.resolve()
        if args.reference_root is not None
        else repository_root / REFERENCE_REPOSITORY
    )
    stage4_root = repository_root / STAGE4_FIXTURE_DIRECTORY
    fixture_path = script_path.parent / FIXTURE_FILENAME
    manifest_path = script_path.parent / MANIFEST_FILENAME
    if not args.force and (fixture_path.exists() or manifest_path.exists()):
        raise FileExistsError("Fixture or manifest exists; use --force after review")

    _verify_reference_checkout(reference_root)
    for relative, digest in (
        ("routines.py", ROUTINES_SOURCE_SHA256),
        ("mainProgram.py", MAIN_PROGRAM_SOURCE_SHA256),
        ("projectors.py", PROJECTORS_SOURCE_SHA256),
        ("measure.py", MEASURE_SOURCE_SHA256),
    ):
        _verify_file(reference_root / relative, digest)
    stage4_manifest, stage4_arrays = _load_stage4_fixture(stage4_root)
    routines = _module_from_file(
        "_pinned_tbg_hf_routines_stage6a",
        reference_root / "routines.py",
    )
    arrays = _build_arrays(routines, stage4_manifest, stage4_arrays)
    _write_deterministic_npz(fixture_path, arrays)

    manifest = {
        "array_hash_convention": ARRAY_HASH_CONVENTION,
        "array_hash_semantics": ARRAY_HASH_SEMANTICS,
        "arrays": {
            name: {
                "dtype": array.dtype.str,
                "sha256": _array_sha256(array),
                "shape": list(array.shape),
            }
            for name, array in sorted(arrays.items())
        },
        "environment": _environment_metadata(),
        "fixture_npz": FIXTURE_FILENAME,
        "fixture_npz_sha256": _sha256_file(fixture_path),
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "generator_script": script_path.name,
        "generator_script_sha256": _sha256_file(script_path),
        "pinned_source": {
            "reference_repository": REFERENCE_REPOSITORY,
            "reference_commit": REFERENCE_COMMIT,
            "routines": {"path": ROUTINES_SOURCE, "sha256": ROUTINES_SOURCE_SHA256},
            "main_program": {
                "path": MAIN_PROGRAM_SOURCE,
                "sha256": MAIN_PROGRAM_SOURCE_SHA256,
            },
            "projectors": {
                "path": PROJECTORS_SOURCE,
                "sha256": PROJECTORS_SOURCE_SHA256,
            },
            "measure": {"path": MEASURE_SOURCE, "sha256": MEASURE_SOURCE_SHA256},
            "source_line_ranges": SOURCE_LINE_RANGES,
        },
        "scf_input": SCF_INPUT,
        "scf_input_sha256": _json_sha256(SCF_INPUT),
        "scope": "four_iteration_source_oracle_not_production_HF_TDHF_or_Fig8",
        "stage4_fixture": {
            "directory": STAGE4_FIXTURE_DIRECTORY,
            "fixture_npz_sha256": STAGE4_FIXTURE_NPZ_SHA256,
            "generator_sha256": STAGE4_GENERATOR_SHA256,
            "resolved_input_sha256": STAGE4_RESOLVED_INPUT_SHA256,
            "array_sha256": dict(sorted(STAGE4_ARRAY_SHA256.items())),
        },
        "stored_projector_orientation": "P[alpha,beta]=<c_dagger_alpha c_beta>",
        "units": {
            "Hamiltonian_action_eigenvalues_energy": "eV",
            "energy": "finite_system_eV_not_per_moire_cell",
        },
    }
    _write_json(manifest_path, manifest)
    print(f"Wrote {fixture_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
