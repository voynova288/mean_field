#!/data/home/ziyuzhu/miniconda3/bin/python -S
"""Hash-bound q003 paired-sector candidate qualifier; no eigensolver authority."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from control_common import (
    ROOT,
    SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    file_sha256,
    load_canonical_json,
    scontrol_job,
    unique_submission_receipt,
    validate_submission_receipt,
    verify_runtime_launch_chain,
    verify_capsule_manifest,
)


def canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def write_exclusive(path: Path, payload: object) -> None:
    data = canonical(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o400)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short exclusive publication write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def array_sha256(value: object) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return sha256(
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    ).hexdigest()


def write_npy_exclusive(path: Path, value: object) -> None:
    import numpy as np

    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o400,
    )
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
            stream.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def jsonable(value: object) -> object:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        return jsonable(value.item())
    return value


def verify_static_and_launch() -> tuple[dict[str, object], str, str]:
    verify_capsule_manifest(require_dynamic_empty=False)
    preparation = file_sha256(ROOT / "provenance/PREPARATION_RECORD.json")
    if os.environ.get("EXPECTED_PREPARATION_SHA256") != preparation:
        raise RuntimeError("detached preparation hash mismatch")
    contract = load_canonical_json(ROOT / "config/contract.json")
    if contract.get("source_commit") != (ROOT / "provenance/SOURCE_COMMIT.txt").read_text().strip():
        raise RuntimeError("source commit binding mismatch")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise RuntimeError("direct/non-Slurm execution is forbidden")
    _receipt_path, receipt, receipt_digest = unique_submission_receipt()
    if receipt.get("job_id") != job_id:
        raise RuntimeError("submission receipt job mismatch")
    if validate_submission_receipt(receipt, receipt_digest) != job_id:
        raise RuntimeError("submission receipt authorization chain mismatch")
    _argv, _raw, current_fields = scontrol_job(
        job_id, allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS
    )
    runtime_chain = verify_runtime_launch_chain(job_id, current_fields)
    if runtime_chain["submission_receipt_sha256"] != receipt_digest:
        raise RuntimeError("runtime launch-chain receipt mismatch")
    manifest = load_canonical_json(ROOT / "input/INPUT_MANIFEST.json")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise RuntimeError("input manifest mismatch")
    return contract, job_id, receipt_digest


def build_prepared(contract: dict[str, object]):
    import numpy as np
    from mean_field.systems.abc_trilayer.vituri2024 import VITURI2024_PARAMETERS
    from mean_field.systems.abc_trilayer.vituri2024_hf_scf import (
        make_vituri2024_cartesian_hf_spec_from_spacing,
        prepare_vituri2024_homogeneous_hf_fft,
    )
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral import (
        Vituri2024FiniteQSpiralChoice,
        prepare_vituri2024_hf_spiral,
    )

    physical = contract["physical"]
    spec = make_vituri2024_cartesian_hf_spec_from_spacing(
        int(physical["mesh_size"]),
        int(physical["holes_per_valley"]),
        float(physical["delta_k_a0"]),
        delta1_ev=float(physical["delta1_ev"]),
        gate_distance_angstrom=float(physical["gate_distance_angstrom"]),
        precision=float(physical["precision"]),
    )
    base = prepare_vituri2024_homogeneous_hf_fft(spec, fft_workers=1)
    choice = Vituri2024FiniteQSpiralChoice(
        q_inverse_angstrom=np.asarray(
            [float(physical["q_a0"]) / VITURI2024_PARAMETERS.a0, 0.0], dtype=np.float64
        ),
        selected_spin=int(physical["selected_spin"]),
        gauge_mode=str(physical["gauge_mode"]),
        b3_anchor_floor=float(physical["b3_anchor_floor"]),
        occupation_gap_floor_ev=float(physical["occupation_gap_floor_ev"]),
        spin_block_tolerance_ev=float(physical["spin_block_tolerance_ev"]),
    )
    prepared = prepare_vituri2024_hf_spiral(base, choice)
    if prepared.fingerprint != contract["lineage"]["worker1_prepared_fingerprint"]:
        raise RuntimeError("worker1 prepared fingerprint mismatch")
    prepared.validate_live_state()
    return prepared


def run_tests(job_dir: Path, contract: dict[str, object]) -> dict[str, object]:
    tests = contract["tests"]
    argv = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(
        argv,
        cwd=ROOT / "source",
        text=False,
        capture_output=True,
        timeout=600,
        check=False,
    )
    (job_dir / "pytest.stdout").write_bytes(completed.stdout)
    (job_dir / "pytest.stderr").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("sealed focused pytest suite failed")
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout_sha256": sha256(completed.stdout).hexdigest(),
        "stderr_sha256": sha256(completed.stderr).hexdigest(),
    }


def run_group(prepared, row: dict[str, object], contract: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_stability import (
        prepare_vituri2024_hf_spiral_restricted_stability,
    )
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_stability import (
        build_vituri2024_hf_spiral_full_sector_inventory,
    )
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_response import (
        build_vituri2024_hf_spiral_signed_displacement_response,
    )
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_hessian import (
        build_vituri2024_hf_spiral_full_hessian_context,
    )
    from mean_field.systems.abc_trilayer.vituri2024_hf_spiral_full_reciprocity import (
        approve_vituri2024_whole_inventory_reciprocity,
        compare_vituri2024_whole_inventory_reciprocity,
        vituri2024_whole_inventory_reciprocity_implementation_fingerprint,
    )

    with np.load(ROOT / row["artifact"], allow_pickle=False) as archive:
        density = np.asarray(archive["final_density"])
        raw = np.asarray(archive["fresh_raw_density"])
        hamiltonian = np.asarray(archive["fresh_hamiltonian"])
    if not np.array_equal(density, raw):
        raise RuntimeError("fresh raw density differs from endpoint")
    restricted = prepare_vituri2024_hf_spiral_restricted_stability(
        prepared,
        density,
        hamiltonian,
        expected_density_native_sha256=row["density_sha256_stability_v2"],
        expected_fresh_hamiltonian_conventional_sha256=row["fresh_hamiltonian_sha256_stability_v2"],
    )
    inventory = build_vituri2024_hf_spiral_full_sector_inventory(restricted)
    response = build_vituri2024_hf_spiral_signed_displacement_response(inventory)
    context = build_vituri2024_hf_spiral_full_hessian_context(response)
    if inventory.complex_dimension != 15_052_040 or inventory.real_dimension != 30_104_080:
        raise RuntimeError("q003 full selected-spin dimension drifted")
    if inventory.selected_occupied_count != 11_852 or inventory.selected_virtual_count != 1_270:
        raise RuntimeError("q003 selected rank inventory drifted")
    approval_row = contract["detached_approval"]
    live_implementation = vituri2024_whole_inventory_reciprocity_implementation_fingerprint()
    if live_implementation != approval_row["expected_implementation_fingerprint"]:
        raise RuntimeError("live reciprocity implementation differs from approval")
    approval = approve_vituri2024_whole_inventory_reciprocity(
        expected_implementation_fingerprint=approval_row["expected_implementation_fingerprint"],
        source_commit=approval_row["source_commit"],
        review_record_sha256=approval_row["review_record_sha256"],
        reduced_exhaustive_qualification_sha256=approval_row["reduced_exhaustive_qualification_sha256"],
        approval_record_sha256=approval_row["approval_record_sha256"],
        rationale=approval_row["rationale"],
    )
    if approval.fingerprint != approval_row["approval_fingerprint"]:
        raise RuntimeError("detached approval fingerprint drifted")
    comparison = compare_vituri2024_whole_inventory_reciprocity(context, approval)
    comparison.validate_live_state()
    if (
        not comparison.passed
        or not comparison.candidate_only
        or comparison.whole_inventory_reciprocity_established
        or comparison.scalar_hessian_authority_established
        or comparison.hermitian_eigensolver_authorized
        or comparison.full_local_stability_established
    ):
        raise RuntimeError("candidate whole-inventory comparison gates failed")
    return {
        "group_label": row["group_label"],
        "path_id": row["path_id"],
        "source_artifact": row["artifact"],
        "source_artifact_sha256": file_sha256(ROOT / row["artifact"]),
        "density_sha256_stability_v2": row["density_sha256_stability_v2"],
        "fresh_hamiltonian_sha256_stability_v2": row["fresh_hamiltonian_sha256_stability_v2"],
        "comparison": jsonable(comparison),
    }


def main() -> int:
    contract, job_id, receipt_digest = verify_static_and_launch()
    if len(sys.argv) == 2 and sys.argv[1] == "--launch-preflight":
        print(json.dumps({"launch_preflight": "passed", "job_id": job_id}, sort_keys=True))
        return 0
    if len(sys.argv) != 1:
        raise RuntimeError("unexpected runner arguments")
    job_dir = ROOT / "output" / f"job_{job_id}"
    job_dir.mkdir(mode=0o700)
    tests = run_tests(job_dir, contract)
    prepared = build_prepared(contract)
    groups = [run_group(prepared, row, contract) for row in contract["representatives"]]
    if len(groups) != 2 or {row["group_label"] for row in groups} != {"40fd", "3307"}:
        raise RuntimeError("both declared q003 groups were not evaluated")
    if any(not group["comparison"]["passed"] for group in groups):
        raise RuntimeError("one or more structural comparisons failed")
    result = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_result.v24",
        "job_id": job_id,
        "source_commit": contract["source_commit"],
        "submission_receipt_sha256": receipt_digest,
        "focused_tests": tests,
        "groups": groups,
        "both_groups_evaluated": True,
        "all_candidate_structural_comparisons_passed": True,
        "candidate_only": True,
        "whole_inventory_reciprocity_established": False,
        "scalar_hessian_authority_established": False,
        "hermitian_eigensolver_authorized": False,
        "full_local_stability_established": False,
        "scientific_authority_promoted": False,
        "production_ready": False,
        "paper_reproduction_verified": False,
        "requires_separate_postrun_review": True,
    }
    result_path = job_dir / "result.json"
    write_exclusive(result_path, result)
    ready = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_ready.v24",
        "job_id": job_id,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": file_sha256(result_path),
        "execution_ready_for_wrapper_finalization": True,
        "scientific_authority_promoted": False,
        "requires_separate_postrun_review": True,
    }
    write_exclusive(ROOT / "output" / f"READY_COMPLETE_{job_id}.json", ready)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"QUALIFIER_FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    raise SystemExit(code)
