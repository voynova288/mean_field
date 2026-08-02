#!/usr/bin/env python3
"""Generate the Stage4 companion HF-action oracle from pinned source only.

The exact hash-checked ``singleParticle.py`` and ``routines.py`` are imported
only inside ``main()``.  This generator never imports the Stage4 port.  Its
projector is a generic analytic IVC algebra input used to exercise contractions;
it is explicitly not a K-IVC seed.  ``fixture.npz`` and ``manifest.json`` are
absent until this script is run deliberately.
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
DEFAULT_INPUT_SOURCE = "reference/TBG-HF/int_input.json"
DEFAULT_INPUT_SOURCE_SHA256 = (
    "c143c294ad95cf94d91cfbabd0437556e5c2a342850d54484c9b47caaf84b4de"
)
HF_INPUT_SOURCE = "reference/TBG-HF/HF_input.json"
HF_INPUT_SOURCE_SHA256 = (
    "d577afffdf80a05a348394c5b813540b8074107fc765454f6e50066d420e25e8"
)
SINGLE_PARTICLE_SOURCE = "reference/TBG-HF/singleParticle.py"
SINGLE_PARTICLE_SOURCE_SHA256 = (
    "a050fa545c4d399b227a178bcc4705a110bd7962edcb9e1f69e300b5e1a3e43b"
)
ROUTINES_SOURCE = "reference/TBG-HF/routines.py"
ROUTINES_SOURCE_SHA256 = (
    "507e8b9e799f494777d354c9d7d481dd19d6ba42894d393630dd79ef16d02108"
)
MAIN_PROGRAM_SOURCE = "reference/TBG-HF/mainProgram.py"
MAIN_PROGRAM_SOURCE_SHA256 = (
    "258c97e57164055de3273ba4471cd96be709c1f159e19f73481750c801aed401"
)
CONSTANTS_SOURCE = "reference/TBG-HF/constants.py"
CONSTANTS_SOURCE_SHA256 = (
    "8d25bcccd54e41207788ff4a9e1b934a50347fabdad46ba44408fc535573ec62"
)
ARRAY_HASH_CONVENTION = (
    "sha256_little_endian_int64_shape_then_C_order_little_endian_array_bytes"
)
ARRAY_HASH_SEMANTICS = (
    "source_fixture_integrity_and_same_environment_parity_not_cross_eigensolver_raw_gauge"
)
FIXTURE_SCHEMA = "mean_field.tbg.companion_hf_action.source_fixture"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_FILENAME = "fixture.npz"
MANIFEST_FILENAME = "manifest.json"
FORM_REAL_THRESHOLD = 1.0e-9
PROJECTOR_SCOPE = "generic_IVC_algebra_input_not_K_IVC_seed"

INPUT_OVERRIDES: dict[str, int | float] = {
    "N1": 2,
    "N2": 3,
    "Ng1": 3,
    "Ng2": 3,
    "n_active": 1,
    "theta": 1.08,
    "wAA": 0.07,
    "wAB": 0.11,
    "strain": 0.0,
    "varphi": 0.0,
}
INHERITED_INTERACTION_FIELDS = ("NG1", "NG2", "dsc", "gates", "include_q=0")
INHERITED_HF_FIELDS = ("epsr", "exchange", "boost1", "boost2")
SOURCE_LINE_RANGES = {
    "singleParticle.gen_form_factors": "389-440",
    "routines.gen_H_SP": "6-22",
    "routines.gen_M_tVE": "24-79",
    "routines.calc_E": "81-97",
    "routines.calc_fock_matrix": "99-153",
    "mainProgram.screen_raw_intFT": "31",
    "mainProgram.build_and_realify_form": "33-43",
    "mainProgram.zero_or_nonzero_boost": "47-54",
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
    elif source.dtype.kind == "U":
        dtype = source.dtype
    else:
        raise TypeError(f"Unsupported fixture dtype {source.dtype}")
    converted = np.asarray(source, dtype=dtype)
    if converted.ndim == 0:
        return converted
    return np.ascontiguousarray(converted)


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
            f"Pinned reference commit drift: expected {REFERENCE_COMMIT}, got {actual_commit}"
        )


def _module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pinned_sources(reference_root: Path) -> tuple[ModuleType, ModuleType]:
    constants_module = _module_from_file(
        "_pinned_tbg_hf_constants_stage4",
        reference_root / "constants.py",
    )
    sentinel = object()
    previous_constants = sys.modules.get("constants", sentinel)
    sys.modules["constants"] = constants_module
    try:
        single_particle = _module_from_file(
            "_pinned_tbg_hf_single_particle_stage4",
            reference_root / "singleParticle.py",
        )
    finally:
        if previous_constants is sentinel:
            del sys.modules["constants"]
        else:
            sys.modules["constants"] = previous_constants  # type: ignore[assignment]
    routines = _module_from_file(
        "_pinned_tbg_hf_routines_stage4",
        reference_root / "routines.py",
    )
    return single_particle, routines


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    payload = array if array.ndim == 0 else np.ascontiguousarray(array)
    np.lib.format.write_array(
        buffer,
        payload,
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


def _generic_ivc_projectors(pd: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return source-oriented P=sum outer(conj(v),v), never a K-IVC claim."""

    N1 = int(pd["N1"])
    N2 = int(pd["N2"])
    nactive = int(pd["n_active"])
    if nactive != 1:
        raise ValueError("The v1 analytic fixture is defined only for n_active=1")
    P = np.zeros((N1, N2, 2, 4, 4), dtype=complex)
    P_ref = np.zeros_like(P)
    for ik1 in range(N1):
        for ik2 in range(N2):
            phase = np.exp(2j * np.pi * (ik1 / N1 + ik2 / N2))
            v1 = np.asarray(
                [np.sqrt(0.7), 0.0, phase * np.sqrt(0.3), 0.0],
                dtype=complex,
            )
            v2 = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=complex)
            stored_projector = np.outer(np.conj(v1), v1) + np.outer(np.conj(v2), v2)
            for spin in range(2):
                P[ik1, ik2, spin] = stored_projector
                P_ref[ik1, ik2, spin] = 0.5 * np.eye(4)
    return P, P_ref


def _build_arrays(
    single_particle: ModuleType,
    routines: ModuleType,
    pd: dict[str, object],
) -> tuple[dict[str, np.ndarray], str, float]:
    coeff, sp_energy = single_particle.gen_coeff(pd)
    raw_intFT = single_particle.gen_interaction(pd)
    screened_intFT = raw_intFT / pd["epsr"]

    N1 = int(pd["N1"])
    N2 = int(pd["N2"])
    NG1 = int(pd["NG1"])
    NG2 = int(pd["NG2"])
    nactive = int(pd["n_active"])
    form_raw = np.zeros(
        (N1, N2, N1, N2, 2 * NG1, 2 * NG2, 2, 2 * nactive, 2 * nactive),
        dtype=complex,
    )
    for band1 in range(2 * nactive):
        for band2 in range(2 * nactive):
            form_raw[:, :, :, :, :, :, 0, band1, band2] = (
                single_particle.gen_form_factors(
                    pd,
                    coeff[..., 0, band1, :],
                    coeff[..., 0, band2, :],
                )
            )
            form_raw[:, :, :, :, :, :, 1, band1, band2] = (
                single_particle.gen_form_factors(
                    pd,
                    coeff[..., 1, band1, :],
                    coeff[..., 1, band2, :],
                )
            )
    if not np.all(np.isfinite(form_raw)):
        raise RuntimeError("The v1 HF-action fixture requires finite form_raw values")
    max_abs_imag = float(np.max(np.abs(np.imag(form_raw))))
    form_branch = (
        "complex" if max_abs_imag > FORM_REAL_THRESHOLD else "real"
    )
    if max_abs_imag > FORM_REAL_THRESHOLD or form_branch != "real":
        raise RuntimeError(
            "The v1 HF-action fixture requires max_abs_imag(form_raw) <= "
            f"{FORM_REAL_THRESHOLD:.1e} and form_branch == 'real'; got "
            f"{max_abs_imag:.17e} and {form_branch!r}"
        )
    form = np.asarray(np.real(form_raw), dtype=np.float64)
    if not np.array_equal(form, np.real(form_raw)):
        raise RuntimeError("source_form must equal real(source_form_raw) exactly")

    H_SP = routines.gen_H_SP(pd, sp_energy)
    source_form, M, tVE = routines.gen_M_tVE(pd, form, screened_intFT)
    if source_form is not form:
        raise RuntimeError("Pinned gen_M_tVE unexpectedly replaced the form object")
    for name, values in (
        ("source_form", source_form),
        ("source_M_ev", M),
        ("source_tVE_ev", tVE),
    ):
        if np.asarray(values).dtype != np.dtype(np.float64):
            raise RuntimeError(f"The v1 real branch requires {name} dtype float64")
    P, P_ref = _generic_ivc_projectors(pd)
    H_D, H_E = routines.calc_fock_matrix(pd, P - P_ref, form, M, tVE)
    energy = routines.calc_E(pd, P, P_ref, sp_energy, H_D, H_E)

    arrays = {
        "P": _little_endian_array(P),
        "P_ref": _little_endian_array(P_ref),
        "source_H_D_ev": _little_endian_array(H_D),
        "source_H_E_ev": _little_endian_array(H_E),
        "source_H_SP_ev": _little_endian_array(H_SP),
        "source_M_ev": _little_endian_array(M),
        "source_coeff": _little_endian_array(coeff),
        "source_energy_ev": _little_endian_array(energy),
        "source_form": _little_endian_array(source_form),
        "source_form_branch": _little_endian_array(np.asarray(form_branch)),
        "source_form_raw": _little_endian_array(form_raw),
        "source_raw_intFT_ev": _little_endian_array(raw_intFT),
        "source_screened_intFT_ev": _little_endian_array(screened_intFT),
        "source_sp_energy_ev": _little_endian_array(sp_energy),
        "source_tVE_ev": _little_endian_array(tVE),
    }
    if arrays["source_form_raw"].dtype != np.dtype("<c16"):
        raise RuntimeError("source_form_raw must be stored as complex128")
    for name in ("source_form", "source_M_ev", "source_tVE_ev"):
        if arrays[name].dtype != np.dtype("<f8"):
            raise RuntimeError(f"{name} must be stored as float64")
    if not np.array_equal(arrays["source_form"], np.real(arrays["source_form_raw"])):
        raise RuntimeError("stored source_form must equal real(source_form_raw) exactly")
    return arrays, form_branch, max_abs_imag


def _manifest(
    *,
    script_path: Path,
    resolved_input: dict[str, object],
    inherited_interaction: dict[str, object],
    inherited_hf: dict[str, object],
    arrays: dict[str, np.ndarray],
    form_branch: str,
    form_raw_max_abs_imag: float,
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
        "environment": _environment_metadata(),
        "fixture_npz": FIXTURE_FILENAME,
        "fixture_npz_sha256": _sha256_file(fixture_path),
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "form_branch": form_branch,
        "form_raw_max_abs_imag": form_raw_max_abs_imag,
        "form_real_threshold": FORM_REAL_THRESHOLD,
        "generator_script": script_path.name,
        "generator_script_sha256": _sha256_file(script_path),
        "inherited_hf_input": inherited_hf,
        "inherited_interaction_input": inherited_interaction,
        "input_overrides": dict(INPUT_OVERRIDES),
        "output_units": {
            "H_D_H_E_H_SP_M_tVE": "eV",
            "energy": "finite_system_eV_not_per_moire_cell",
            "intFT": "eV",
        },
        "pinned_source": {
            "constants": {"path": CONSTANTS_SOURCE, "sha256": CONSTANTS_SOURCE_SHA256},
            "default_input": {
                "path": DEFAULT_INPUT_SOURCE,
                "sha256": DEFAULT_INPUT_SOURCE_SHA256,
            },
            "hf_input": {"path": HF_INPUT_SOURCE, "sha256": HF_INPUT_SOURCE_SHA256},
            "main_program": {
                "path": MAIN_PROGRAM_SOURCE,
                "sha256": MAIN_PROGRAM_SOURCE_SHA256,
            },
            "reference_commit": REFERENCE_COMMIT,
            "reference_repository": REFERENCE_REPOSITORY,
            "routines": {"path": ROUTINES_SOURCE, "sha256": ROUTINES_SOURCE_SHA256},
            "single_particle": {
                "path": SINGLE_PARTICLE_SOURCE,
                "sha256": SINGLE_PARTICLE_SOURCE_SHA256,
            },
            "source_line_ranges": dict(SOURCE_LINE_RANGES),
        },
        "projector": {
            "definition": (
                "per_k_and_spin:v1=(sqrt(0.7),0,phase*sqrt(0.3),0),"
                "v2=(0,1,0,0);phase=exp(2pi*i*(k1/N1+k2/N2));"
                "P=sum_outer(conj(v),v);P_ref=0.5*I4"
            ),
            "scope": PROJECTOR_SCOPE,
            "spin_duplication": "identical_two_spin_blocks",
            "stored_orientation": "P[alpha,beta]=<c_dagger_alpha c_beta>",
        },
        "resolved_input": resolved_input,
        "resolved_input_sha256": _json_sha256(resolved_input),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Pinned TBG-HF checkout (commit and file hashes remain mandatory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing fixture and manifest after explicit review.",
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
            "Fixture or manifest exists; use --force only after reviewing pinned identities"
        )

    paths_and_hashes = (
        (reference_root / "int_input.json", DEFAULT_INPUT_SOURCE_SHA256),
        (reference_root / "HF_input.json", HF_INPUT_SOURCE_SHA256),
        (reference_root / "singleParticle.py", SINGLE_PARTICLE_SOURCE_SHA256),
        (reference_root / "routines.py", ROUTINES_SOURCE_SHA256),
        (reference_root / "mainProgram.py", MAIN_PROGRAM_SOURCE_SHA256),
        (reference_root / "constants.py", CONSTANTS_SOURCE_SHA256),
    )
    _verify_reference_checkout(reference_root)
    for path, digest in paths_and_hashes:
        _verify_file(path, digest)

    pinned_interaction_input = json.loads(
        (reference_root / "int_input.json").read_text(encoding="utf-8")
    )
    pinned_hf_input = json.loads(
        (reference_root / "HF_input.json").read_text(encoding="utf-8")
    )
    if not isinstance(pinned_interaction_input, dict) or not isinstance(pinned_hf_input, dict):
        raise TypeError("Pinned input JSON files must decode to objects")
    inherited_interaction = {
        key: pinned_interaction_input[key] for key in INHERITED_INTERACTION_FIELDS
    }
    inherited_hf = {key: pinned_hf_input[key] for key in INHERITED_HF_FIELDS}
    if inherited_interaction != {
        "NG1": 5,
        "NG2": 5,
        "dsc": 2.5000000000000002e-8,
        "gates": "dual",
        "include_q=0": True,
    }:
        raise RuntimeError("Pinned interaction fields differ from the v1 fixture contract")
    if inherited_hf != {
        "epsr": 10,
        "exchange": True,
        "boost1": 0,
        "boost2": 0,
    }:
        raise RuntimeError("Pinned HF fields differ from the v1 fixture contract")
    if set(INPUT_OVERRIDES).intersection(INHERITED_INTERACTION_FIELDS + INHERITED_HF_FIELDS):
        raise RuntimeError("Inherited interaction/HF fields must not be overridden")

    resolved_input = dict(pinned_interaction_input)
    resolved_input.update(inherited_hf)
    resolved_input.update(INPUT_OVERRIDES)
    resolved_input.pop("intdir", None)
    resolved_input.pop("outdir", None)
    if int(resolved_input["NG1"]) != 2 * int(resolved_input["Ng1"]) - 1:
        raise RuntimeError("Fixture must saturate the executable NG1 <= 2*Ng1-1 guard")
    if int(resolved_input["NG2"]) != 2 * int(resolved_input["Ng2"]) - 1:
        raise RuntimeError("Fixture must saturate the executable NG2 <= 2*Ng2-1 guard")

    single_particle, routines = _load_pinned_sources(reference_root)
    arrays, form_branch, form_raw_max_abs_imag = _build_arrays(
        single_particle,
        routines,
        resolved_input,
    )
    _write_deterministic_npz(fixture_path, arrays)
    manifest = _manifest(
        script_path=script_path,
        resolved_input=resolved_input,
        inherited_interaction=inherited_interaction,
        inherited_hf=inherited_hf,
        arrays=arrays,
        form_branch=form_branch,
        form_raw_max_abs_imag=form_raw_max_abs_imag,
        fixture_path=fixture_path,
    )
    _write_json(manifest_path, manifest)
    print(f"Wrote {fixture_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
