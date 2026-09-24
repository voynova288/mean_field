#!/usr/bin/env python3
"""Detached, read-only postflight checker for job 517203.

This script imports no capsule/production module.  It recomputes identities and the
seven-mode field directly from the published NPZ/CSV, and independently verifies
hash/mode closure of the immutable checkpoint/output namespaces.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np

CAPSULE = Path(
    "/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/"
    "ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/"
    "physical_operator_hole_crosscheck_v3"
)
OUT = CAPSULE / "runtime/output/job_517203"
CHECKPOINT = CAPSULE / "runtime/checkpoint/oracle_job_517203"
ORACLE_PUBLICATION = CAPSULE / "runtime/.oracle_publication.517203"
CROSSCHECK_PUBLICATION = CAPSULE / "runtime/.crosscheck_publication.517203"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def complex_obj(z: complex) -> dict[str, float]:
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def maxloc(a: np.ndarray) -> dict[str, object]:
    flat = int(np.argmax(np.abs(a)))
    idx = tuple(int(i) for i in np.unravel_index(flat, a.shape))
    return {"max_abs": float(np.abs(a[idx])), "index": list(idx), "value": complex_obj(a[idx])}


def verify_namespace(root: Path, manifest_name: str, sentinel_name: str) -> dict[str, object]:
    manifest_path = root / manifest_name
    sentinel_path = root / sentinel_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    expected_payloads = set(manifest["files"])
    actual_files = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    }
    expected_files = expected_payloads | {manifest_name, sentinel_name}
    payload_checks = {}
    for rel, expected_hash in sorted(manifest["files"].items()):
        p = root / rel
        payload_checks[rel] = {
            "sha256": sha256(p),
            "expected_sha256": expected_hash,
            "hash_match": sha256(p) == expected_hash,
            "mode": mode(p),
            "regular_nonsymlink": p.is_file() and not p.is_symlink(),
            "size_bytes": p.stat().st_size,
        }
    manifest_hash = sha256(manifest_path)
    sentinel_hash = sha256(sentinel_path)
    return {
        "root": str(root),
        "root_mode": mode(root),
        "manifest_mode": mode(manifest_path),
        "sentinel_mode": mode(sentinel_path),
        "manifest_sha256": manifest_hash,
        "sentinel_sha256": sentinel_hash,
        "manifest_binding_match": sentinel.get("output_manifest_sha256") == manifest_hash,
        "exact_file_namespace": actual_files == expected_files,
        "actual_file_count": len(actual_files),
        "expected_file_count": len(expected_files),
        "unexpected_files": sorted(actual_files - expected_files),
        "missing_files": sorted(expected_files - actual_files),
        "all_payload_hashes_match": all(x["hash_match"] for x in payload_checks.values()),
        "all_payloads_regular_nonsymlink": all(x["regular_nonsymlink"] for x in payload_checks.values()),
        "payload_modes": sorted(set(x["mode"] for x in payload_checks.values())),
        "payload_checks": payload_checks,
        "sentinel": sentinel,
    }


def mirror_check(final: Path, mirror: Path, relative_files: set[str]) -> dict[str, object]:
    rows = {}
    for rel in sorted(relative_files):
        a, b = final / rel, mirror / rel
        rows[rel] = {
            "exists": a.is_file() and b.is_file(),
            "sha256_equal": a.is_file() and b.is_file() and sha256(a) == sha256(b),
            "final_mode": mode(a) if a.exists() else None,
            "mirror_mode": mode(b) if b.exists() else None,
        }
    return {
        "mirror": str(mirror),
        "all_exist": all(x["exists"] for x in rows.values()),
        "all_sha256_equal": all(x["sha256_equal"] for x in rows.values()),
        "all_modes_equal": all(x["final_mode"] == x["mirror_mode"] for x in rows.values()),
        "files": rows,
    }


def main() -> None:
    host = os.uname().nodename
    if host.startswith("login"):
        raise RuntimeError(f"refusing numerical postflight on login node {host}")

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    complete = json.loads((OUT / "COMPLETE").read_text(encoding="utf-8"))
    oracle_complete = json.loads((CHECKPOINT / "ORACLE_COMPLETE").read_text(encoding="utf-8"))

    with np.load(OUT / "crosscheck_arrays.npz", allow_pickle=False) as z:
        axes = [str(x) for x in z["axes"]]
        projectors = [str(x) for x in z["projector_names"]]
        labels = [tuple(int(y) for y in x) for x in z["physical_stored_labels"]]
        coefficients_npz = np.array(z["coefficients"], copy=True)
        residual_er_npz = np.array(z["electron_plus_direct_hole_minus_reference"], copy=True)
        residual_dc_npz = np.array(z["direct_hole_minus_complement_hole"], copy=True)
        n_grid_npz = np.array(z["hole_density_grid_per_nm2"], copy=True)
        s_grid_npz = np.array(z["hole_spin_grid_over_hbar_per_nm2"], copy=True)
        margin_npz = np.array(z["finite_seven_mode_margin_per_nm2"], copy=True)
        area = float(z["area_nm2"])
        npz_schema = str(z["schema"])

    ai = {name: i for i, name in enumerate(axes)}
    pi = {name: i for i, name in enumerate(projectors)}
    qi = {label: i for i, label in enumerate(labels)}

    coeff_csv = np.full(coefficients_npz.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    er_csv = np.full(residual_er_npz.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    dc_csv = np.full(residual_dc_npz.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    with (OUT / "coefficients.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        q = (int(row["physical_m"]), int(row["physical_n"]))
        a = ai[row["axis"]]
        value = complex(float(row["real"]), float(row["imag"]))
        if row["record"] == "coefficient":
            coeff_csv[pi[row["projector"]], a, qi[q]] = value
        elif row["record"] == "electron_plus_direct_hole_minus_reference":
            er_csv[a, qi[q]] = value
        elif row["record"] == "direct_hole_minus_complement_hole":
            dc_csv[a, qi[q]] = value
        else:
            raise ValueError(f"unexpected CSV record {row['record']!r}")
    if np.isnan(coeff_csv.real).any() or np.isnan(er_csv.real).any() or np.isnan(dc_csv.real).any():
        raise RuntimeError("CSV did not fill every expected coefficient/residual slot")

    # Independent identities formed from published coefficients, not saved residual arrays.
    er_from_npz = coefficients_npz[pi["electron_stored_abi"]] + coefficients_npz[pi["hole_direct_eigh"]] - coefficients_npz[pi["full_reference_I24"]]
    dc_from_npz = coefficients_npz[pi["hole_direct_eigh"]] - coefficients_npz[pi["hole_complement"]]
    er_from_csv = coeff_csv[pi["electron_stored_abi"]] + coeff_csv[pi["hole_direct_eigh"]] - coeff_csv[pi["full_reference_I24"]]
    dc_from_csv = coeff_csv[pi["hole_direct_eigh"]] - coeff_csv[pi["hole_complement"]]

    # Independent seven-mode reconstruction from CSV coefficients and declared Fourier sign.
    nmesh = int(margin_npz.shape[0])
    if margin_npz.shape != (nmesh, nmesh):
        raise ValueError("expected square finite-seven-mode mesh")
    x = np.arange(nmesh, dtype=np.float64) / nmesh
    xx, yy = np.meshgrid(x, x, indexing="ij")
    csv_fields = []
    for axis in axes:
        field = np.zeros((nmesh, nmesh), dtype=np.complex128)
        for m, n in labels:
            field += coeff_csv[pi["hole_direct_eigh"], ai[axis], qi[(m, n)]] * np.exp(-2j * np.pi * (m * xx + n * yy))
        csv_fields.append(field / area)
    n_grid_csv = csv_fields[0]
    s_grid_csv = np.stack(csv_fields[1:4], axis=0)
    margin_csv = n_grid_csv.real / 2.0 - np.sqrt(np.sum(s_grid_csv.real**2, axis=0))

    # Independent spin-major sign test. chi=(1,i)/sqrt(2), S=sigma/2.
    chi = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2.0
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2.0
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2.0
    spin_expect = np.array([np.vdot(chi, op @ chi) for op in (sx, sy, sz)])

    q0 = qi[(0, 0)]
    tsb = [qi[q] for q in labels if q != (0, 0)]
    ref_tsb = coeff_csv[pi["full_reference_I24"], :, tsb]
    q0_counts = {
        "N_e": complex_obj(coeff_csv[pi["electron_stored_abi"], ai["charge"], q0]),
        "N_h_direct": complex_obj(coeff_csv[pi["hole_direct_eigh"], ai["charge"], q0]),
        "N_h_complement": complex_obj(coeff_csv[pi["hole_complement"], ai["charge"], q0]),
        "N_ref": complex_obj(coeff_csv[pi["full_reference_I24"], ai["charge"], q0]),
    }
    q0_ref_spin = {
        axis: complex_obj(coeff_csv[pi["full_reference_I24"], ai[axis], q0])
        for axis in axes[1:4]
    }

    output_closure = verify_namespace(OUT, "OUTPUT_SHA256.json", "COMPLETE")
    oracle_closure = verify_namespace(CHECKPOINT, "OUTPUT_SHA256.json", "ORACLE_COMPLETE")
    output_manifest_files = set(json.loads((OUT / "OUTPUT_SHA256.json").read_text())["files"])
    oracle_manifest_files = set(json.loads((CHECKPOINT / "OUTPUT_SHA256.json").read_text())["files"])
    output_mirror = mirror_check(OUT, CROSSCHECK_PUBLICATION, output_manifest_files | {"OUTPUT_SHA256.json", "COMPLETE"})
    # The hidden oracle publication path is a receipt pair, not a payload mirror;
    # compare only the two files it actually publishes.  The complete 70-file
    # immutable payload is independently closed under CHECKPOINT above.
    oracle_publication_receipt = mirror_check(
        CHECKPOINT, ORACLE_PUBLICATION, {"oracle_manifest.json", "ORACLE_COMPLETE"}
    )

    closure_cross = {
        "complete_summary_binding_match": complete["summary_sha256"] == sha256(OUT / "summary.json"),
        "complete_oracle_sentinel_binding_match": complete["oracle_checkpoint_sentinel_sha256"] == sha256(CHECKPOINT / "ORACLE_COMPLETE"),
        "oracle_complete_oracle_manifest_binding_match": oracle_complete["oracle_manifest_sha256"] == sha256(CHECKPOINT / "oracle_manifest.json"),
        "producer_job_ids": [complete["producer_job_id"], oracle_complete["producer_job_id"], summary["runtime"]["job_id"]],
        "source_frozen_sha256s": [complete["source_frozen_sha256"], oracle_complete["source_frozen_sha256"], summary["inputs"]["source_frozen_sha256"]],
    }

    result = {
        "schema": "ptse2_physical_operator_hole_crosscheck_v3_detached_recompute/v1",
        "producer_job_id": "517203",
        "execution": {
            "hostname": host,
            "python": sys.executable,
            "numpy": np.__version__,
            "production_modules_imported": False,
            "capsule_mutated": False,
            "sources": [str(OUT / "crosscheck_arrays.npz"), str(OUT / "coefficients.csv")],
        },
        "npz_schema": npz_schema,
        "inventory": {"axes": axes, "projectors": projectors, "physical_labels": [list(q) for q in labels], "csv_row_count": len(rows)},
        "csv_npz_agreement": {
            "coefficient_max_abs_difference": float(np.max(np.abs(coeff_csv - coefficients_npz))),
            "saved_electron_plus_hole_residual_max_abs_difference": float(np.max(np.abs(er_csv - residual_er_npz))),
            "saved_direct_minus_complement_residual_max_abs_difference": float(np.max(np.abs(dc_csv - residual_dc_npz))),
        },
        "independent_residuals": {
            "electron_plus_direct_hole_minus_reference_from_npz_coefficients": maxloc(er_from_npz),
            "electron_plus_direct_hole_minus_reference_from_csv_coefficients": maxloc(er_from_csv),
            "direct_hole_minus_complement_hole_from_npz_coefficients": maxloc(dc_from_npz),
            "direct_hole_minus_complement_hole_from_csv_coefficients": maxloc(dc_from_csv),
            "npz_formed_vs_npz_saved_er_max_abs_difference": float(np.max(np.abs(er_from_npz - residual_er_npz))),
            "npz_formed_vs_npz_saved_dc_max_abs_difference": float(np.max(np.abs(dc_from_npz - residual_dc_npz))),
            "csv_formed_vs_csv_saved_er_max_abs_difference": float(np.max(np.abs(er_from_csv - er_csv))),
            "csv_formed_vs_csv_saved_dc_max_abs_difference": float(np.max(np.abs(dc_from_csv - dc_csv))),
            "production_summary_er_difference": float(np.max(np.abs(er_from_csv)) - summary["raw_residuals"]["electron_plus_direct_hole_minus_reference_max_abs"]),
            "production_summary_dc_difference": float(np.max(np.abs(dc_from_csv)) - summary["raw_residuals"]["direct_hole_minus_complement_hole_max_abs"]),
        },
        "Q0": {
            "counts": q0_counts,
            "expected_counts": {"N_e": 22, "N_h_direct": 2, "N_h_complement": 2, "N_ref": 24},
            "reference_spin_S_over_hbar": q0_ref_spin,
        },
        "full_reference_TSB_exact_zero": {
            "entry_count": int(ref_tsb.size),
            "all_exact_complex_zero": bool(np.all(ref_tsb == 0.0 + 0.0j)),
            "max_abs": float(np.max(np.abs(ref_tsb))),
            "labels": [list(q) for q in labels if q != (0, 0)],
        },
        "spin_major_synthetic_plus_y": {
            "chi": "(1,i)/sqrt(2)",
            "S_over_hbar": [complex_obj(x) for x in spin_expect],
            "is_plus_y_half_with_zero_xz": bool(spin_expect[0] == 0 and spin_expect[2] == 0 and spin_expect[1].real > 0 and np.isclose(spin_expect[1].real, 0.5)),
        },
        "finite_seven_mode": {
            "grid": [nmesh, nmesh],
            "area_nm2": area,
            "csv_reconstructed_minimum_n_h_over_2_minus_abs_S_h_over_hbar_per_nm2": float(np.min(margin_csv)),
            "npz_saved_minimum_per_nm2": float(np.min(margin_npz)),
            "csv_vs_npz_density_max_abs_difference": float(np.max(np.abs(n_grid_csv - n_grid_npz))),
            "csv_vs_npz_spin_max_abs_difference": float(np.max(np.abs(s_grid_csv - s_grid_npz))),
            "csv_vs_npz_margin_max_abs_difference": float(np.max(np.abs(margin_csv - margin_npz))),
            "max_reconstruction_imaginary_abs_per_nm2": float(max(np.max(np.abs(n_grid_csv.imag)), np.max(np.abs(s_grid_csv.imag)))),
            "authority": "finite seven retained Fourier modes only; no positivity proof",
        },
        "closure": {
            "output": output_closure,
            "oracle_checkpoint": oracle_closure,
            "output_publication_mirror": output_mirror,
            "oracle_publication_receipt_mirror": oracle_publication_receipt,
            "cross_bindings": closure_cross,
        },
        "scope_review": {
            "declared_scope": summary["scope"],
            "classification": summary["classification"],
            "coefficient_thresholds": summary["coefficient_thresholds"],
            "uncertainty": summary["uncertainty"],
            "no_overclaim_assessment": "consistent: raw seven-channel cross-validation only; no coefficient classification, positivity proof, global-ground-state claim, or authority beyond Q0 plus six shell-1 TSB channels",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
