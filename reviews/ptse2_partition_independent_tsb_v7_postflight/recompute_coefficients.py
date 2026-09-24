#!/usr/bin/env python3
"""Independent coefficient-space postflight for partition-independent TSB v7.

Does not import/execute v7 analyze_tsb.py or formula_core.py, perform real-space
reconstruction, or read analysis_arrays_v7.npz. Production analysis_v7.json is
read only after recomputation to report differences.
"""
from __future__ import annotations

import hashlib
import json
import stat
import struct
import sys
from pathlib import Path

sys.path.insert(0, "/data/home/ziyuzhu/miniconda3/envs/moirekp/lib/python3.11/site-packages")
import numpy as np

ROOT = Path("/data/home/ziyuzhu/Mean_Field")
BASE = ROOT / "results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1"
CAP = BASE / "partition_independent_tsb_v7"
OUT = CAP / "runtime/output/job_516988_v7"
SOURCE = BASE / "physical_three_subcell_shell6_v26_mode_recovery/runtime/output/job_513422/matched_R6.npz"
REGIONAL = BASE / "physical_three_subcell_shell6_v26_presentation_v3/runtime/output/regional_integrals_exact.json"
PRODUCTION = OUT / "analysis_v7.json"
APPROVAL = ROOT / "reviews/ptse2_partition_independent_tsb_v7/REVIEW_APPROVAL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("angle is undefined for a zero vector")
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))))


def rms_metrics(charge, hole, spin, mask, area) -> dict[str, float]:
    electron = float(np.sqrt(np.sum(np.abs(charge[mask]) ** 2)) / area)
    holes = float(np.sqrt(np.sum(np.abs(hole[mask]) ** 2)) / area)
    physical_spin = float(np.sqrt(np.sum(np.abs(spin[:, mask]) ** 2)) / area)
    return {
        "electron_n_tsb_rms_e_nm2": electron,
        "electron_n_tsb_rms_over_22_over_A": electron / (22.0 / area),
        "hole_n_tsb_rms_e_nm2": holes,
        "hole_n_tsb_rms_over_2_over_A": holes / (2.0 / area),
        "physical_S_tsb_vector_rms_hbar_nm2": physical_spin,
    }


def covariance_metrics(spin, mask, area) -> dict[str, object]:
    selected = spin[:, mask]
    covariance = np.real(selected @ selected.conj().T) / area**2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    axis = eigenvectors[:, 0].copy()
    if axis[0] < 0.0:
        axis *= -1.0
    trace = float(np.trace(covariance))
    return {
        "matrix": covariance,
        "eigenvalues_descending": eigenvalues,
        "eigenfractions_descending": eigenvalues / trace,
        "trace": trace,
        "principal_axis_x_positive": axis,
        "lambda1_over_trace": float(eigenvalues[0] / trace),
        "lambda3_over_trace": float(eigenvalues[-1] / trace),
    }


def canonical(label) -> bool:
    first, second = map(int, label)
    return first > 0 or (first == 0 and second > 0)


def all_pair_z2(charge, spin, labels, mask, axis) -> dict[str, object]:
    by_label = {tuple(map(int, label)): index for index, label in enumerate(labels)}
    pair_indices = [index for index in np.flatnonzero(mask) if canonical(labels[index])]
    numerator, denominator = 0.0j, 0.0
    positive_terms, zero_terms = 0, 0
    max_charge_conjugacy, max_spin_conjugacy = 0.0, 0.0
    for plus in pair_indices:
        label = tuple(map(int, labels[plus]))
        reverse_label = (-label[0], -label[1])
        if reverse_label not in by_label:
            raise ValueError(f"missing reverse label {reverse_label}")
        minus = by_label[reverse_label]
        if not mask[minus]:
            raise ValueError(f"reverse label outside mask: {reverse_label}")
        s_plus, s_minus = complex(axis @ spin[:, plus]), complex(axis @ spin[:, minus])
        max_charge_conjugacy = max(max_charge_conjugacy, abs(charge[minus] - charge[plus].conjugate()))
        max_spin_conjugacy = max(max_spin_conjugacy, abs(s_minus - s_plus.conjugate()))
        for index, projected in ((plus, s_plus), (minus, s_minus)):
            weight = float(abs(charge[index]) * abs(projected))
            if weight == 0.0:
                zero_terms += 1
                continue
            relative = charge[index] * projected.conjugate()
            numerator += weight * (relative / abs(relative)) ** 2
            denominator += weight
            positive_terms += 1
    if denominator == 0.0:
        raise ValueError("all-pair z2 has zero total weight")
    z2 = numerator / denominator
    return {
        "canonical_pair_count": len(pair_indices),
        "positive_weight_channel_count": positive_terms,
        "exact_zero_weight_channel_count": zero_terms,
        "total_weight": denominator,
        "z2": z2,
        "z2_abs": float(abs(z2)),
        "z2_phase_deg": float(np.degrees(np.angle(z2))),
        "maximum_charge_reverse_conjugacy_abs": float(max_charge_conjugacy),
        "maximum_longitudinal_spin_reverse_conjugacy_abs": float(max_spin_conjugacy),
    }


def png_metadata(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    width, height, depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload[16:29])
    software, offset = None, 8
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        chunk = payload[offset + 8:offset + 8 + length]
        if kind == b"tEXt" and chunk.startswith(b"Software\x00"):
            software = chunk.split(b"\x00", 1)[1].decode("latin-1")
        offset += 12 + length
        if kind == b"IEND":
            break
    return {
        "width": width, "height": height, "bit_depth": depth, "color_type": color_type,
        "compression": compression, "filter": filtering, "interlace": interlace,
        "software": software, "sha256": sha256(path), "size_bytes": path.stat().st_size,
        "mode": mode(path),
    }


def as_complex(value):
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        return complex(float(value["real"]), float(value["imag"]))
    return complex(value)


def delta(independent, production) -> float:
    return float(abs(as_complex(production) - as_complex(independent)))


with np.load(SOURCE, allow_pickle=False) as archive:
    data = {name: np.asarray(archive[name]) for name in archive.files}

numerator3 = np.asarray(data["numerator3"], dtype=np.int64)
labels = np.asarray(data["supercell_labels"], dtype=np.int64)
shell = np.asarray(data["shell"], dtype=np.int64)
charge = np.asarray(data["charge_fourier"], dtype=np.complex128)
hole = np.asarray(data["hole_charge_fourier"], dtype=np.complex128)
spin = np.asarray(data["spin_pauli_fourier"], dtype=np.complex128) / 2.0
area = float(data["area_nm2"])
if np.any((numerator3[:, 0] - numerator3[:, 1]) % 3 != 0):
    raise ValueError("u-v is not divisible by three")
residues = numerator3[:, 0] % 3
full_tsb = residues != 0
q0 = np.all(labels == 0, axis=1)
seven_selected = q0 | ((shell == 1) & full_tsb)
seven_tsb = seven_selected & full_tsb

full_rms = rms_metrics(charge, hole, spin, full_tsb, area)
seven_rms = rms_metrics(charge, hole, spin, seven_tsb, area)
full_cov = covariance_metrics(spin, full_tsb, area)
seven_cov = covariance_metrics(spin, seven_tsb, area)
full_axis = np.asarray(full_cov["principal_axis_x_positive"])
seven_axis = np.asarray(seven_cov["principal_axis_x_positive"])
full_z2 = all_pair_z2(charge, spin, labels, full_tsb, full_axis)
seven_z2 = all_pair_z2(charge, spin, labels, seven_tsb, seven_axis)

regional = json.loads(REGIONAL.read_text(encoding="utf-8"))
table = np.asarray(regional["values"]["wigner_seitz"]["R6"], dtype=np.float64)
spin_by_owner = table[2:5].T
A, B, C = spin_by_owner[0], spin_by_owner[2], spin_by_owner[1]
U, L, T = A + B + C, (A - B) / 2.0, (A + B - 2.0 * C) / 6.0
reconstructed = np.stack((U / 3.0 + L + T, U / 3.0 - L + T, U / 3.0 - 2.0 * T))
raw_axis_angle = angle_degrees(L, full_axis)

independent = {
    "input_sha256s": {"matched_R6.npz": sha256(SOURCE), "regional_integrals_exact.json": sha256(REGIONAL)},
    "area_nm2": area,
    "residue_counts_0_1_2": [int(np.count_nonzero(residues == value)) for value in range(3)],
    "primitive_count": int(np.count_nonzero(~full_tsb)),
    "tsb_count": int(np.count_nonzero(full_tsb)),
    "full_R6": {"rms": full_rms, "covariance": full_cov, "all_pair_z2": full_z2},
    "seven_mode": {
        "definition": "Q0 plus six shell-1 TSB channels; TSB metrics use those six channels",
        "selected_count": int(np.count_nonzero(seven_selected)), "tsb_count": int(np.count_nonzero(seven_tsb)),
        "rms": seven_rms, "covariance": seven_cov, "all_pair_z2": seven_z2,
    },
    "seven_over_full": {
        "electron_tsb_rms_ratio": seven_rms["electron_n_tsb_rms_e_nm2"] / full_rms["electron_n_tsb_rms_e_nm2"],
        "hole_tsb_rms_ratio": seven_rms["hole_n_tsb_rms_e_nm2"] / full_rms["hole_n_tsb_rms_e_nm2"],
        "spin_tsb_rms_ratio": seven_rms["physical_S_tsb_vector_rms_hbar_nm2"] / full_rms["physical_S_tsb_vector_rms_hbar_nm2"],
        "principal_axis_absolute_dot": float(abs(full_axis @ seven_axis)),
        "lambda1_fraction_difference": seven_cov["lambda1_over_trace"] - full_cov["lambda1_over_trace"],
        "lambda3_fraction_difference": seven_cov["lambda3_over_trace"] - full_cov["lambda3_over_trace"],
        "z2_difference": complex(seven_z2["z2"]) - complex(full_z2["z2"]),
    },
    "corrected_WS_ULT": {
        "source_contract": {"owner_labels": regional["owner_labels"], "canonical_ABC": regional["owner_labels_are_canonical_ABC"]},
        "task_assignment": {"A": "owner0", "B": "owner2", "C": "owner1"},
        "A_hbar": A, "B_hbar": B, "C_hbar": C, "U_hbar": U, "L_hbar": L, "T_hbar": T,
        "U_norm_hbar": float(np.linalg.norm(U)), "L_norm_hbar": float(np.linalg.norm(L)),
        "T_norm_hbar": float(np.linalg.norm(T)), "T_norm_over_L_norm": float(np.linalg.norm(T) / np.linalg.norm(L)),
        "angle_A_B_deg": angle_degrees(A, B), "angle_L_to_principal_axis_raw_deg": raw_axis_angle,
        "angle_L_to_principal_axis_acute_deg": min(raw_axis_angle, 180.0 - raw_axis_angle),
        "inverse_reconstruction_max_abs_hbar": float(np.max(np.abs(reconstructed - np.stack((A, B, C))))),
    },
}

production = json.loads(PRODUCTION.read_text(encoding="utf-8"))
pif = production["partition_independent_fourier"]
pfull, pseven = pif["full_R6"], pif["seven_mode_subset"]
pult = production["partition_specific_noncanonical_ws_ult"]
comparisons = {
    "residue_counts_exact": independent["residue_counts_0_1_2"] == pif["sector_split"]["counts"],
    "full": {
        "electron_rms_normalized_abs_delta": delta(full_rms["electron_n_tsb_rms_over_22_over_A"], pfull["rms"]["electron_n_tsb_rms_over_22_over_A"]),
        "hole_rms_normalized_abs_delta": delta(full_rms["hole_n_tsb_rms_over_2_over_A"], pfull["rms"]["hole_n_tsb_rms_over_2_over_A"]),
        "spin_rms_abs_delta": delta(full_rms["physical_S_tsb_vector_rms_hbar_nm2"], pfull["rms"]["physical_S_tsb_vector_rms_hbar_nm2"]),
        "covariance_max_abs_delta": float(np.max(np.abs(np.asarray(full_cov["matrix"]) - np.asarray(pfull["covariance"]["matrix"])))),
        "eigenvalues_max_abs_delta": float(np.max(np.abs(np.asarray(full_cov["eigenvalues_descending"]) - np.asarray(pfull["covariance"]["eigenvalues_descending"])))),
        "principal_axis_max_abs_delta": float(np.max(np.abs(full_axis - np.asarray(pfull["covariance"]["principal_axis_sign_fixed"])))),
        "lambda1_fraction_abs_delta": delta(full_cov["lambda1_over_trace"], pfull["covariance"]["lambda1_over_trace"]),
        "lambda3_fraction_abs_delta": delta(full_cov["lambda3_over_trace"], pfull["covariance"]["lambda3_over_trace"]),
        "z2_abs_delta": delta(full_z2["z2"], pfull["phase"]["z2"]),
        "z2_pair_count_exact": full_z2["canonical_pair_count"] == pfull["phase"]["pair_count"],
        "z2_positive_terms_exact": full_z2["positive_weight_channel_count"] == pfull["phase"]["channel_terms_with_positive_weight"],
    },
    "seven": {
        "selected_count_exact": independent["seven_mode"]["selected_count"] == pseven["selected_channel_count"],
        "tsb_count_exact": independent["seven_mode"]["tsb_count"] == pseven["tsb_channel_count"],
        "electron_rms_abs_delta": delta(seven_rms["electron_n_tsb_rms_e_nm2"], pseven["rms"]["electron_n_tsb_rms_e_nm2"]),
        "hole_rms_abs_delta": delta(seven_rms["hole_n_tsb_rms_e_nm2"], pseven["rms"]["hole_n_tsb_rms_e_nm2"]),
        "spin_rms_abs_delta": delta(seven_rms["physical_S_tsb_vector_rms_hbar_nm2"], pseven["rms"]["physical_S_tsb_vector_rms_hbar_nm2"]),
        "covariance_max_abs_delta": float(np.max(np.abs(np.asarray(seven_cov["matrix"]) - np.asarray(pseven["covariance"]["matrix"])))),
        "principal_axis_max_abs_delta": float(np.max(np.abs(seven_axis - np.asarray(pseven["covariance"]["principal_axis_sign_fixed"])))),
        "z2_abs_delta": delta(seven_z2["z2"], pseven["phase"]["z2"]),
    },
    "seven_over_full": {
        "electron_rms_ratio_abs_delta": delta(independent["seven_over_full"]["electron_tsb_rms_ratio"], pif["seven_mode_comparison"]["electron_tsb_rms_ratio_seven_over_full"]),
        "hole_rms_ratio_abs_delta": delta(independent["seven_over_full"]["hole_tsb_rms_ratio"], pif["seven_mode_comparison"]["hole_tsb_rms_ratio_seven_over_full"]),
        "spin_rms_ratio_abs_delta": delta(independent["seven_over_full"]["spin_tsb_rms_ratio"], pif["seven_mode_comparison"]["spin_tsb_rms_ratio_seven_over_full"]),
        "principal_axis_dot_abs_delta": delta(independent["seven_over_full"]["principal_axis_absolute_dot"], pif["seven_mode_comparison"]["principal_axis_absolute_dot"]),
        "lambda1_difference_abs_delta": delta(independent["seven_over_full"]["lambda1_fraction_difference"], pif["seven_mode_comparison"]["lambda1_over_trace_difference_seven_minus_full"]),
        "lambda3_difference_abs_delta": delta(independent["seven_over_full"]["lambda3_fraction_difference"], pif["seven_mode_comparison"]["lambda3_over_trace_difference_seven_minus_full"]),
        "z2_difference_abs_delta": delta(independent["seven_over_full"]["z2_difference"], pif["seven_mode_comparison"]["z2_difference"]),
    },
    "corrected_WS_ULT": {
        "U_max_abs_delta": float(np.max(np.abs(U - np.asarray(pult["U_hbar"])))),
        "L_max_abs_delta": float(np.max(np.abs(L - np.asarray(pult["L_hbar"])))),
        "T_max_abs_delta": float(np.max(np.abs(T - np.asarray(pult["T_hbar"])))),
        "U_norm_abs_delta": delta(float(np.linalg.norm(U)), pult["U_norm_hbar"]),
        "T_over_L_abs_delta": delta(float(np.linalg.norm(T) / np.linalg.norm(L)), pult["T_norm_over_L_norm"]),
        "angle_A_B_abs_delta_deg": delta(angle_degrees(A, B), pult["angle_A_B_deg"]),
        "raw_axis_angle_abs_delta_deg": delta(raw_axis_angle, pult["angle_L_to_full_covariance_principal_axis_signed_raw_deg"]),
        "acute_axis_angle_abs_delta_deg": delta(min(raw_axis_angle, 180.0 - raw_axis_angle), pult["angle_L_to_full_covariance_axis_acute_sign_invariant_deg"]),
    },
}

config = json.loads((CAP / "CONFIG.json").read_text(encoding="utf-8"))
manifest = json.loads((OUT / "OUTPUT_MANIFEST_V7.json").read_text(encoding="utf-8"))
completion = json.loads((OUT / "COMPLETION_RECORD_V7.json").read_text(encoding="utf-8"))
sentinel_path = CAP / "runtime/ANALYSIS_COMPLETE_V7"
sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
expected_names = set(manifest["files"]) | {"OUTPUT_MANIFEST_V7.json", "COMPLETION_RECORD_V7.json"}
actual_names = {path.name for path in OUT.iterdir()}
output_checks = {}
for name, record in manifest["files"].items():
    path, info = OUT / name, (OUT / name).lstat()
    output_checks[name] = {
        "regular_no_symlink": stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "sha256_matches": sha256(path) == record["sha256"], "size_matches": info.st_size == record["size_bytes"],
        "mode": mode(path),
    }

binding_paths = {
    "approval_sha256": APPROVAL,
    "attested_chain_sha256": CAP / "runtime/control/ATTESTED_CHAIN.json",
    "release_verification_sha256": CAP / "runtime/control/analysis.RELEASE_VERIFICATION.json",
    "resource_gate_sha256": CAP / "runtime/job_516988/resource_gate.json",
    "running_receipt_sha256": CAP / "runtime/control/analysis.RUNNING_RECEIPT.json",
    "runtime_attestation_sha256": CAP / "runtime/job_516988/RUNTIME_ATTESTATION_V7.json",
    "source_frozen_sha256": CAP / "SOURCE_FROZEN.json",
    "submission_receipt_sha256": CAP / "runtime/control/analysis.SUBMISSION_RECEIPT.json",
}
binding_checks = {}
for key, path in binding_paths.items():
    observed, expected = sha256(path), completion["execution_bindings"][key]
    binding_checks[key] = {"path": str(path), "expected": expected, "observed": observed, "matches": observed == expected}

figures = {name: png_metadata(OUT / name) for name in (
    "tsb_charge_3x3.png", "tsb_total_physical_spin_3x3.png", "covariance_phase_diagnostics.png"
)}
config_input_sha256s = {name: record["sha256"] for name, record in config["inputs"].items()}
analysis_input_sha256s = {name: record["sha256"] for name, record in production["source_inputs"].items()}
artifact = {
    "output_directory_mode": mode(OUT), "output_directory_exact_inventory": actual_names == expected_names,
    "output_manifest_mode": mode(OUT / "OUTPUT_MANIFEST_V7.json"),
    "completion_record_mode": mode(OUT / "COMPLETION_RECORD_V7.json"),
    "actual_output_names": sorted(actual_names), "manifest_file_checks": output_checks,
    "all_manifest_hash_size_mode_checks_pass": all(
        item["regular_no_symlink"] and item["sha256_matches"] and item["size_matches"] and item["mode"] == "0444"
        for item in output_checks.values()
    ),
    "completion_record_sha256": sha256(OUT / "COMPLETION_RECORD_V7.json"),
    "completion_record_matches_sentinel": sha256(OUT / "COMPLETION_RECORD_V7.json") == sentinel["completion_record_sha256"],
    "output_manifest_sha256": sha256(OUT / "OUTPUT_MANIFEST_V7.json"),
    "manifest_matches_completion": sha256(OUT / "OUTPUT_MANIFEST_V7.json") == completion["output_manifest_sha256"],
    "manifest_matches_sentinel": sha256(OUT / "OUTPUT_MANIFEST_V7.json") == sentinel["output_manifest_sha256"],
    "sentinel_mode": mode(sentinel_path), "sentinel_output_directory_exact": sentinel["output_directory"] == str(OUT),
    "sentinel_claims_last_namespace_operation": sentinel["sentinel_last_namespace_operation"] is True,
    "execution_bindings_equal_across_manifest_completion_sentinel_analysis": (
        manifest["execution_bindings"] == completion["execution_bindings"] == sentinel["execution_bindings"] == production["execution_bindings"]
    ),
    "execution_binding_hash_checks": binding_checks,
    "all_execution_binding_hash_checks_pass": all(item["matches"] for item in binding_checks.values()),
    "scientific_input_hashes_equal_across_config_completion_analysis": config_input_sha256s == completion["scientific_input_sha256s"] == analysis_input_sha256s,
    "scientific_input_hashes": config_input_sha256s,
    "source_inputs_used_by_recompute_match_config": (
        independent["input_sha256s"]["matched_R6.npz"] == config_input_sha256s["matched_R6.npz"]
        and independent["input_sha256s"]["regional_integrals_exact.json"] == config_input_sha256s["regional_integrals_exact.json"]
    ),
    "stderr_zero_bytes": (CAP / "runtime/slurm/ptse2_tsb_v7_c9e3acc1_516988.err").stat().st_size == 0,
    "figures": figures,
}

result = {
    "schema": "ptse2_partition_independent_tsb_v7_independent_coefficient_postflight/v1",
    "method": {
        "coefficient_space_only": True, "real_space_reconstruction": False,
        "production_analyze_tsb_imported_or_executed": False, "production_formula_core_imported": False,
        "production_analysis_arrays_read": False,
        "rms_formula": "sqrt(sum_TSB |f_Q|^2)/A; spin sums Cartesian components after S=sigma/2",
        "covariance_formula": "C_ab=Re sum_TSB S_aQ S_bQ*/A^2",
        "z2_formula": "both members of every canonical +/-Q pair; w=|n_Q||ell.S_Q|; term=w*(n_Q s_Q*/|n_Q s_Q*|)^2",
        "ult_formula": "U=A+B+C; L=(A-B)/2; T=(A+B-2C)/6",
    },
    "independent_recompute": independent,
    "production_absolute_differences": comparisons,
    "artifact_closure": artifact,
}
print(json.dumps(native(result), indent=2, sort_keys=True, allow_nan=False))
