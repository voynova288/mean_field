#!/data/home/ziyuzhu/miniconda3/bin/python -S
"""Publish execution closure only after the wrapper observed successful srun."""
from __future__ import annotations
from hashlib import sha256
import json, math, os, re, stat
from pathlib import Path
from control_common import (
    ROOT,
    file_sha256,
    load_canonical_json,
    unique_submission_receipt,
    verify_capsule_manifest,
)

_HEX = re.compile(r"[0-9a-f]{64}")


def canonical(payload):
    return json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"


def write_exclusive(path,payload):
    data=canonical(payload)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0),0o400)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0: raise OSError("short exclusive publication write")
            view = view[written:]
        os.fsync(fd)
    finally: os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)


def load_canonical(path):
    info = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise RuntimeError("unsafe dynamic finalization input")
    data=path.read_bytes(); payload=json.loads(data)
    if data != canonical(payload): raise RuntimeError("noncanonical finalization input")
    return payload


def array_sha256(value):
    import numpy as np

    array = np.ascontiguousarray(value)
    from hashlib import sha256
    return sha256(
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    ).hexdigest()


def require_digest(value, label):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256")


def require_nonnegative_float(value, label):
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} must be an exact finite nonnegative float")


def require_exact_key(value, expected, label):
    if (
        type(value) is not list
        or len(value) != 3
        or any(type(item) is not int for item in value)
        or value != expected
    ):
        raise RuntimeError(f"{label} is not an exact sector key")


def main():
    verify_capsule_manifest(require_dynamic_empty=False)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise RuntimeError("missing Slurm job identity")
    ready_path = ROOT / "output" / f"READY_COMPLETE_{job_id}.json"
    ready = load_canonical(ready_path)
    expected_result_relative = f"output/job_{job_id}/result.json"
    if ready.get("result_path") != expected_result_relative:
        raise RuntimeError("READY result path is not the exact job namespace")
    result_path = ROOT / expected_result_relative
    result = load_canonical(result_path)
    _receipt_path, _receipt, receipt_digest = unique_submission_receipt()
    expected_result_keys = {
        "schema", "job_id", "source_commit", "submission_receipt_sha256",
        "focused_tests", "groups", "both_groups_evaluated",
        "all_candidate_structural_comparisons_passed", "candidate_only",
        "whole_inventory_reciprocity_established",
        "scalar_hessian_authority_established", "hermitian_eigensolver_authorized",
        "full_local_stability_established", "scientific_authority_promoted",
        "production_ready", "paper_reproduction_verified",
        "requires_separate_postrun_review",
    }
    if (
        set(result) != expected_result_keys
        or result["schema"] != "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_result.v24"
        or result["job_id"] != job_id
        or result["source_commit"] != (ROOT / "provenance/SOURCE_COMMIT.txt").read_text().strip()
        or result["submission_receipt_sha256"] != receipt_digest
        or result["both_groups_evaluated"] is not True
        or result["all_candidate_structural_comparisons_passed"] is not True
        or result["candidate_only"] is not True
        or result["requires_separate_postrun_review"] is not True
        or any(result[name] is not False for name in (
            "whole_inventory_reciprocity_established",
            "scalar_hessian_authority_established",
            "hermitian_eigensolver_authorized",
            "full_local_stability_established",
            "scientific_authority_promoted",
            "production_ready",
            "paper_reproduction_verified",
        ))
    ):
        raise RuntimeError("candidate reciprocity result failed exact top-level validation")
    focused = result["focused_tests"]
    contract = load_canonical_json(ROOT / "config/contract.json")
    if (
        type(contract) is not dict
        or set(contract) != {
            "schema", "source_commit", "scope", "physical", "lineage",
            "representatives", "detached_approval", "tests", "execution",
            "authority", "prior_failure",
        }
        or contract["schema"] != "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_contract.v24"
        or contract["source_commit"] != result["source_commit"]
        or type(contract["tests"]) is not list
        or any(type(item) is not str for item in contract["tests"])
    ):
        raise RuntimeError("contract exact top-level schema drifted")
    prior_failure = contract["prior_failure"]
    if (
        type(prior_failure) is not dict
        or set(prior_failure) != {
            "job_id", "state", "exit_code", "source_commit",
            "diagnostic_sha256", "stderr_sha256", "stdout_sha256",
            "failure_classification",
        }
        or prior_failure["job_id"] != "492866"
        or prior_failure["state"] != "FAILED"
        or prior_failure["exit_code"] != "1:0"
        or prior_failure["source_commit"] != "6dcc7248ae581dfd3f1d5e98a4c25cfa2d543e86"
        or prior_failure["diagnostic_sha256"]
        != file_sha256(ROOT / "prior_failure/DIAGNOSTIC_INCOMPLETE_492866.json")
        or prior_failure["stderr_sha256"]
        != file_sha256(ROOT / "prior_failure/slurm_492866.err")
        or prior_failure["stdout_sha256"]
        != file_sha256(ROOT / "prior_failure/slurm_492866.out")
        or prior_failure["failure_classification"]
        != "runner_validate_live_state_signature_mismatch"
    ):
        raise RuntimeError("prior failed capsule binding drifted")
    expected_pytest_argv = [
        "/data/home/ziyuzhu/miniconda3/bin/python",
        "-m",
        "pytest",
        "-q",
        *contract["tests"],
    ]
    if (
        type(focused) is not dict
        or set(focused) != {"argv", "returncode", "stdout_sha256", "stderr_sha256"}
        or type(focused["returncode"]) is not int
        or focused["returncode"] != 0
        or type(focused["argv"]) is not list
        or focused["argv"] != expected_pytest_argv
        or any(type(item) is not str for item in focused["argv"])
    ):
        raise RuntimeError("focused test evidence is malformed")
    require_digest(focused["stdout_sha256"], "pytest stdout")
    require_digest(focused["stderr_sha256"], "pytest stderr")
    approval = contract["detached_approval"]
    if (
        type(approval) is not dict
        or set(approval) != {
            "expected_implementation_fingerprint", "source_commit",
            "review_record_sha256", "reduced_exhaustive_qualification_sha256",
            "approval_record_sha256", "approval_fingerprint", "rationale",
        }
        or approval["source_commit"] != contract["source_commit"]
        or type(approval["rationale"]) is not str
        or not approval["rationale"]
    ):
        raise RuntimeError("detached approval exact schema drifted")
    for name in (
        "expected_implementation_fingerprint", "review_record_sha256",
        "reduced_exhaustive_qualification_sha256", "approval_record_sha256",
        "approval_fingerprint",
    ):
        require_digest(approval[name], f"approval {name}")
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_reciprocity import (
        VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION,
        VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY,
        VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE,
        VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM,
        vituri2024_whole_inventory_reciprocity_implementation_fingerprint,
    )
    live_implementation = (
        vituri2024_whole_inventory_reciprocity_implementation_fingerprint()
    )
    if live_implementation != approval["expected_implementation_fingerprint"]:
        raise RuntimeError("finalizer live implementation differs from approval")
    expected_comparison_keys = {
        "context_fingerprint", "response_fingerprint", "inventory_fingerprint",
        "fft_plan_fingerprint", "implementation_fingerprint",
        "approved_implementation_fingerprint", "approval_fingerprint",
        "approved_source_commit", "review_record_sha256",
        "reduced_exhaustive_qualification_sha256", "approval_record_sha256",
        "integer_mesh_labels_sha256", "selected_occupations_sha256",
        "selected_spinors_sha256", "selected_fock_diagonal_sha256", "mesh_size",
        "nk", "signed_sector_count_checked", "displacement_count_checked",
        "ordered_mesh_pair_count_covered", "nonempty_sector_count",
        "canonical_orbit_count", "independently_recomputed_complex_dimension",
        "inventory_complex_dimension", "inventory_real_dimension",
        "canonical_orbit_complex_dimension_sum", "maximum_kernel_imaginary_residual",
        "maximum_kernel_even_residual", "kernel_structure_tolerance",
        "support_mismatch_count", "conjugation_structure_mismatch_count",
        "paired_lane_overlap_count", "sector_dimension_mismatch_count",
        "canonical_orbit_mismatch_count", "support_inventory_fingerprint",
        "sector_dimension_fingerprint", "canonical_orbit_fingerprint",
        "direct_rank_one_identity_bound", "exchange_mdagger_ck_m_identity_bound",
        "signed_conjugation_covariance_derived",
        "transition_injection_extraction_adjoint_derived", "one_body_real_diagonal",
        "factor_two_real_packing_bound", "passed", "failed_gates", "fingerprint",
        "api_version", "theorem", "scope", "authority", "candidate_only",
        "whole_inventory_reciprocity_established",
        "full_inventory_exact_unitary_scalar_curvature_established",
        "scalar_hessian_authority_established", "hermitian_eigensolver_authorized",
        "full_local_stability_established", "production_ready",
        "paper_reproduction_verified",
    }
    groups = result["groups"]
    if (
        type(groups) is not list or len(groups) != 2
        or {group.get("group_label") for group in groups} != {"40fd", "3307"}
    ):
        raise RuntimeError("finalizer requires both exact q003 groups")
    representative_by_label = {row["group_label"]: row for row in contract["representatives"]}
    for group in groups:
        if set(group) != {
            "group_label", "path_id", "source_artifact", "source_artifact_sha256",
            "density_sha256_stability_v2", "fresh_hamiltonian_sha256_stability_v2",
            "comparison",
        }:
            raise RuntimeError("group result key set drifted")
        representative = representative_by_label[group["group_label"]]
        if (
            group["path_id"] != representative["path_id"]
            or group["source_artifact"] != representative["artifact"]
            or group["source_artifact_sha256"] != file_sha256(ROOT / representative["artifact"])
            or group["density_sha256_stability_v2"] != representative["density_sha256_stability_v2"]
            or group["fresh_hamiltonian_sha256_stability_v2"] != representative["fresh_hamiltonian_sha256_stability_v2"]
        ):
            raise RuntimeError("group source lineage drifted")
        comparison = group["comparison"]
        if type(comparison) is not dict or set(comparison) != expected_comparison_keys:
            raise RuntimeError("comparison nested exact-key schema drifted")
        for name in (
            "context_fingerprint", "response_fingerprint", "inventory_fingerprint",
            "fft_plan_fingerprint", "implementation_fingerprint",
            "approved_implementation_fingerprint", "approval_fingerprint",
            "review_record_sha256", "reduced_exhaustive_qualification_sha256",
            "approval_record_sha256", "integer_mesh_labels_sha256",
            "selected_occupations_sha256", "selected_spinors_sha256",
            "selected_fock_diagonal_sha256", "support_inventory_fingerprint",
            "sector_dimension_fingerprint", "canonical_orbit_fingerprint", "fingerprint",
        ):
            require_digest(comparison[name], f"comparison {name}")
        if (
            comparison["approved_source_commit"] != contract["source_commit"]
            or comparison["implementation_fingerprint"] != approval["expected_implementation_fingerprint"]
            or comparison["approved_implementation_fingerprint"] != approval["expected_implementation_fingerprint"]
            or comparison["approval_fingerprint"] != approval["approval_fingerprint"]
            or comparison["review_record_sha256"] != approval["review_record_sha256"]
            or comparison["reduced_exhaustive_qualification_sha256"] != approval["reduced_exhaustive_qualification_sha256"]
            or comparison["approval_record_sha256"] != approval["approval_record_sha256"]
            or comparison["api_version"] != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_API_VERSION
            or comparison["theorem"] != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_THEOREM
            or comparison["scope"] != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_SCOPE
            or comparison["authority"] != VITURI2024_WHOLE_INVENTORY_RECIPROCITY_AUTHORITY
            or type(comparison["mesh_size"]) is not int
            or comparison["mesh_size"] != 81
            or type(comparison["nk"]) is not int
            or comparison["nk"] != 6561
            or type(comparison["displacement_count_checked"]) is not int
            or comparison["displacement_count_checked"] != 25921
            or type(comparison["signed_sector_count_checked"]) is not int
            or comparison["signed_sector_count_checked"] != 77763
            or type(comparison["ordered_mesh_pair_count_covered"]) is not int
            or comparison["ordered_mesh_pair_count_covered"] != 43046721
            or any(type(comparison[name]) is not int for name in (
                "nonempty_sector_count", "canonical_orbit_count",
                "independently_recomputed_complex_dimension",
                "inventory_complex_dimension", "inventory_real_dimension",
                "canonical_orbit_complex_dimension_sum",
            ))
            or comparison["nonempty_sector_count"] <= 0
            or comparison["canonical_orbit_count"] <= 0
            or comparison["nonempty_sector_count"]
            != 2 * comparison["canonical_orbit_count"]
            or comparison["inventory_complex_dimension"] != 15052040
            or comparison["inventory_real_dimension"] != 30104080
            or comparison["independently_recomputed_complex_dimension"] != 15052040
            or comparison["canonical_orbit_complex_dimension_sum"] != 15052040
            or type(comparison["maximum_kernel_imaginary_residual"]) is not float
            or comparison["maximum_kernel_imaginary_residual"] != 0.0
            or type(comparison["maximum_kernel_even_residual"]) is not float
            or comparison["maximum_kernel_even_residual"] != 0.0
            or type(comparison["kernel_structure_tolerance"]) is not float
            or not math.isfinite(comparison["kernel_structure_tolerance"])
            or comparison["kernel_structure_tolerance"] <= 0.0
            or comparison["kernel_structure_tolerance"] > 1.0e-10
            or any(type(comparison[name]) is not int for name in (
                "support_mismatch_count", "conjugation_structure_mismatch_count",
                "paired_lane_overlap_count", "sector_dimension_mismatch_count",
                "canonical_orbit_mismatch_count",
            ))
            or comparison["support_mismatch_count"] != 0
            or comparison["conjugation_structure_mismatch_count"] != 0
            or comparison["paired_lane_overlap_count"] != 0
            or comparison["sector_dimension_mismatch_count"] != 0
            or comparison["canonical_orbit_mismatch_count"] != 0
            or comparison["failed_gates"] != []
            or any(comparison[name] is not True for name in (
                "direct_rank_one_identity_bound", "exchange_mdagger_ck_m_identity_bound",
                "signed_conjugation_covariance_derived",
                "transition_injection_extraction_adjoint_derived",
                "one_body_real_diagonal", "factor_two_real_packing_bound", "passed",
                "candidate_only",
            ))
            or any(comparison[name] is not False for name in (
                "whole_inventory_reciprocity_established",
                "full_inventory_exact_unitary_scalar_curvature_established",
                "scalar_hessian_authority_established", "hermitian_eigensolver_authorized",
                "full_local_stability_established", "production_ready",
                "paper_reproduction_verified",
            ))
        ):
            raise RuntimeError("q003 comparison invariant failed")
        receipt_payload = {name: value for name, value in comparison.items() if name != "fingerprint"}
        recomputed = sha256(json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if comparison["fingerprint"] != recomputed:
            raise RuntimeError("comparison receipt fingerprint mismatch")
    expected_ready = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_ready.v24",
        "job_id": job_id,
        "result_path": expected_result_relative,
        "result_sha256": file_sha256(result_path),
        "execution_ready_for_wrapper_finalization": True,
        "scientific_authority_promoted": False,
        "requires_separate_postrun_review": True,
    }
    if ready != expected_ready:
        raise RuntimeError("READY record failed exact finalization validation")
    srun = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_srun_success.v24",
        "job_id": job_id,
        "wrapper_observed_srun_exit_zero": True,
        "ready_sha256": file_sha256(ready_path),
        "result_sha256": file_sha256(result_path),
    }
    srun_path = ROOT / "output" / f"SRUN_SUCCESS_EVIDENCE_{job_id}.json"
    write_exclusive(srun_path, srun)
    complete = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_complete.v24",
        "status": "execution_complete",
        "job_id": job_id,
        "ready_sha256": file_sha256(ready_path),
        "srun_success_evidence_sha256": file_sha256(srun_path),
        "result_sha256": file_sha256(result_path),
        "scientific_authority_promoted": False,
        "requires_separate_postrun_review": True,
    }
    write_exclusive(ROOT / "output" / f"COMPLETE_{job_id}.json", complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
