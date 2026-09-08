#!/data/home/ziyuzhu/miniconda3/bin/python -S
"""No-argument, one-use trusted held-submission shim for the sealed v24 capsule."""
from __future__ import annotations

import base64
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from control_common import (
    CONTROL,
    FIXED_AUTH_COMMENT,
    FIXED_JOB_NAME,
    FIXED_SCHEDULER_USER,
    ROOT,
    SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    SCHEDULER_CALL_TIMEOUT_SECONDS,
    SCHEDULER_POLL_CALL_CAP_SECONDS,
    SUBMISSION_RECONCILIATION_POLL_SECONDS,
    SUBMISSION_RECONCILIATION_TIMEOUT_SECONDS,
    SchedulerCommandTimeout,
    WRAPPER_PATH,
    approved_sbatch_argv,
    bounded_poll_call_timeout,
    canonical_json_bytes,
    exact_held_submission_candidate,
    expected_receipt_comment,
    load_authorization,
    parse_squeue_job_ids,
    run_scheduler_command,
    scontrol_job,
    submission_candidate_query_argv,
    validate_control_directory_empty_for_submission,
    validate_submission_attempt,
    validate_submission_receipt,
    verify_capsule_manifest,
    write_exclusive_canonical,
)


def run_submission_command(
    argv: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    return run_scheduler_command(
        argv, timeout_seconds=timeout_seconds, check=False
    )


def run_submission_candidate_query(
    *, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    return run_scheduler_command(
        submission_candidate_query_argv(),
        timeout_seconds=timeout_seconds,
        check=False,
    )


def wait_for_not_before_submit_time(not_before_epoch: int) -> None:
    while time.time() < float(not_before_epoch):
        time.sleep(min(0.01, float(not_before_epoch) - time.time()))


def _local_scheduler_second(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))


def _candidate_snapshot(
    sequence: int,
    not_before_submit_time: str,
    *,
    deadline: float,
    per_call_cap_seconds: float,
) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise SchedulerCommandTimeout("submission reconciliation budget expired fail-closed")
    completed = run_submission_candidate_query(
        timeout_seconds=bounded_poll_call_timeout(remaining, per_call_cap_seconds)
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("bounded squeue candidate query failed")
    ids = parse_squeue_job_ids(completed.stdout)
    candidate_records: list[dict[str, Any]] = []
    exact_ids: list[str] = []
    for job_id in ids:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise SchedulerCommandTimeout(
                "submission candidate inspection exceeded the bounded budget fail-closed"
            )
        argv, raw, fields = scontrol_job(
            job_id,
            timeout_seconds=bounded_poll_call_timeout(
                remaining, per_call_cap_seconds
            ),
            allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
        )
        matched, mismatches = exact_held_submission_candidate(
            fields, job_id, not_before_submit_time
        )
        candidate_records.append(
            {
                "job_id": job_id,
                "scontrol_argv": argv,
                "scontrol_stdout_b64": base64.b64encode(raw).decode("ascii"),
                "scontrol_stdout_sha256": sha256(raw).hexdigest(),
                "exact_match": matched,
                "mismatch_fields": mismatches,
            }
        )
        if matched:
            exact_ids.append(job_id)
    return {
        "sequence": sequence,
        "query_status": "COMPLETED",
        "squeue_argv": submission_candidate_query_argv(),
        "squeue_returncode": completed.returncode,
        "squeue_stdout_b64": base64.b64encode(completed.stdout).decode("ascii"),
        "squeue_stdout_sha256": sha256(completed.stdout).hexdigest(),
        "squeue_stderr_b64": base64.b64encode(completed.stderr).decode("ascii"),
        "squeue_stderr_sha256": sha256(completed.stderr).hexdigest(),
        "candidate_job_ids": ids,
        "candidate_records": candidate_records,
        "exact_matching_job_ids": exact_ids,
    }


def reconcile_submission_timeout(
    authorization_digest: str,
    intent_digest: str,
    timeout_digest: str,
    not_before_submit_time: str,
    *,
    timeout_seconds: float = SUBMISSION_RECONCILIATION_TIMEOUT_SECONDS,
    poll_seconds: float = SUBMISSION_RECONCILIATION_POLL_SECONDS,
    per_call_cap_seconds: float = SCHEDULER_POLL_CALL_CAP_SECONDS,
) -> tuple[dict[str, Any], bytes]:
    if timeout_seconds <= 0.0 or poll_seconds <= 0.0 or per_call_cap_seconds <= 0.0:
        raise RuntimeError("submission reconciliation bounds must be positive")
    deadline = time.monotonic() + timeout_seconds
    snapshots: list[dict[str, Any]] = []
    stable_unique: str | None = None
    saw_candidate = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 and snapshots:
            outcome = "UNSTABLE_CANDIDATES" if saw_candidate else "ZERO_CANDIDATES"
            break
        snapshot = _candidate_snapshot(
            len(snapshots) + 1,
            not_before_submit_time,
            deadline=deadline,
            per_call_cap_seconds=per_call_cap_seconds,
        )
        snapshots.append(snapshot)
        matches = snapshot["exact_matching_job_ids"]
        if len(matches) == 1:
            if stable_unique == matches[0]:
                outcome = "UNIQUE_HELD_RECONCILED"
                break
            if saw_candidate:
                outcome = "UNSTABLE_CANDIDATES"
                break
            stable_unique = matches[0]
            saw_candidate = True
        elif len(matches) > 1:
            outcome = "MULTIPLE_CANDIDATES"
            break
        else:
            if saw_candidate:
                outcome = "UNSTABLE_CANDIDATES"
                break
            stable_unique = None
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            outcome = "UNSTABLE_CANDIDATES" if saw_candidate else "ZERO_CANDIDATES"
            break
        time.sleep(min(poll_seconds, remaining))
    final_matches = snapshots[-1]["exact_matching_job_ids"]
    payload = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_reconciliation.v24",
        "authorization_sha256": authorization_digest,
        "submission_intent_sha256": intent_digest,
        "submission_timeout_sha256": timeout_digest,
        "candidate_query_argv": submission_candidate_query_argv(),
        "query_snapshots": snapshots,
        "outcome": outcome,
        "matched_job_ids": final_matches,
        "receipt_permitted": outcome == "UNIQUE_HELD_RECONCILED",
        "resubmission_permitted": False,
    }
    data = write_exclusive_canonical(
        CONTROL / "SUBMISSION_RECONCILIATION.json", payload
    )
    return payload, data


def _publish_reconciliation_error(
    intent_digest: str, timeout_digest: str, error: BaseException
) -> None:
    stdout = error.stdout if isinstance(error, SchedulerCommandTimeout) else b""
    stderr = error.stderr if isinstance(error, SchedulerCommandTimeout) else b""
    payload = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_reconciliation_error.v24",
        "record_class": "FAIL_CLOSED_RAW_QUERY_EVIDENCE",
        "submission_intent_sha256": intent_digest,
        "submission_timeout_sha256": timeout_digest,
        "candidate_query_argv": submission_candidate_query_argv(),
        "error_class": type(error).__name__,
        "error_message": str(error),
        "partial_stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "partial_stdout_sha256": sha256(stdout).hexdigest(),
        "partial_stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "partial_stderr_sha256": sha256(stderr).hexdigest(),
        "receipt_permitted": False,
        "resubmission_permitted": False,
    }
    write_exclusive_canonical(
        CONTROL / "SUBMISSION_RECONCILIATION_ERROR.json", payload
    )


def _initial_held_record(job_id: str) -> tuple[list[str], bytes, dict[str, str]]:
    argv, raw, fields = scontrol_job(
        job_id,
        timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS,
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )
    # The receipt validator repeats this exact check from durable bytes.
    return argv, raw, fields


def main() -> int:
    if len(sys.argv) != 1 or Path(__file__).resolve(strict=True) != ROOT / "bin/submit_held.py":
        raise RuntimeError("submit_held.py is a sealed no-argument entry point")
    validate_control_directory_empty_for_submission()
    verify_capsule_manifest(require_dynamic_empty=True)
    authorization, authorization_digest = load_authorization()
    preparation_digest = authorization["preparation_sha256"]
    argv = approved_sbatch_argv(preparation_digest)
    if argv != authorization["approved_sbatch_argv"] or argv[0] != "/usr/bin/sbatch":
        raise RuntimeError("approved absolute sbatch argv drifted")

    created_epoch = int(time.time())
    not_before_epoch = created_epoch + 1
    intent = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_intent.v24",
        "authorization_sha256": authorization_digest,
        "preparation_template_sha256": authorization["preparation_template_sha256"],
        "preparation_sha256": preparation_digest,
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "wrapper_sha256": authorization["wrapper_sha256"],
        "sbatch_argv": argv,
        "fixed_initial_scheduler_comment": FIXED_AUTH_COMMENT,
        "fixed_scheduler_user": FIXED_SCHEDULER_USER,
        "fixed_job_name": FIXED_JOB_NAME,
        "intent_created_at": _local_scheduler_second(created_epoch),
        "not_before_submit_time": _local_scheduler_second(not_before_epoch),
        "held_submission_only": True,
        "one_use_fail_closed": True,
    }
    intent_bytes = write_exclusive_canonical(CONTROL / "SUBMISSION_INTENT.json", intent)
    intent_digest = sha256(intent_bytes).hexdigest()
    # Slurm SubmitTime is second-resolution. This barrier makes every authorized
    # scheduler SubmitTime strictly later than the already fsynced intent time.
    wait_for_not_before_submit_time(not_before_epoch)

    try:
        completed = run_submission_command(
            argv, timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS
        )
    except SchedulerCommandTimeout as error:
        timeout_payload = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_timeout.v24",
            "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
            "authorization_sha256": authorization_digest,
            "submission_intent_sha256": intent_digest,
            "sbatch_argv": argv,
            "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
            "sbatch_partial_stdout_b64": base64.b64encode(error.stdout).decode("ascii"),
            "sbatch_partial_stdout_sha256": sha256(error.stdout).hexdigest(),
            "sbatch_partial_stderr_b64": base64.b64encode(error.stderr).decode("ascii"),
            "sbatch_partial_stderr_sha256": sha256(error.stderr).hexdigest(),
            "scheduler_side_effect": "UNKNOWN",
            "resubmission_permitted": False,
        }
        timeout_bytes = write_exclusive_canonical(
            CONTROL / "SUBMISSION_TIMEOUT.json", timeout_payload
        )
        timeout_digest = sha256(timeout_bytes).hexdigest()
        try:
            reconciliation, reconciliation_bytes = reconcile_submission_timeout(
                authorization_digest,
                intent_digest,
                timeout_digest,
                intent["not_before_submit_time"],
                timeout_seconds=SUBMISSION_RECONCILIATION_TIMEOUT_SECONDS,
                poll_seconds=SUBMISSION_RECONCILIATION_POLL_SECONDS,
                per_call_cap_seconds=SCHEDULER_POLL_CALL_CAP_SECONDS,
            )
        except Exception as query_error:
            _publish_reconciliation_error(intent_digest, timeout_digest, query_error)
            raise RuntimeError(
                "submission timeout reconciliation query was ambiguous; fail-closed with durable evidence"
            ) from query_error
        if reconciliation["outcome"] != "UNIQUE_HELD_RECONCILED":
            raise RuntimeError(
                f"submission timeout reconciliation was {reconciliation['outcome']}; no receipt/release is permitted"
            )
        job_id = reconciliation["matched_job_ids"][0]
        attempt = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_attempt.v24",
            "authorization_sha256": authorization_digest,
            "preparation_template_sha256": authorization["preparation_template_sha256"],
            "preparation_sha256": preparation_digest,
            "source_manifest_sha256": authorization["source_manifest_sha256"],
            "wrapper_sha256": authorization["wrapper_sha256"],
            "submission_intent_sha256": intent_digest,
            "sbatch_argv": argv,
            "outcome": "TIMEOUT_RECONCILED_UNIQUE_HELD",
            "sbatch_returncode": None,
            "sbatch_stdout_b64": timeout_payload["sbatch_partial_stdout_b64"],
            "sbatch_stdout_sha256": timeout_payload["sbatch_partial_stdout_sha256"],
            "sbatch_stderr_b64": timeout_payload["sbatch_partial_stderr_b64"],
            "sbatch_stderr_sha256": timeout_payload["sbatch_partial_stderr_sha256"],
            "submission_timeout_sha256": timeout_digest,
            "submission_reconciliation_sha256": sha256(reconciliation_bytes).hexdigest(),
            "reconciled_job_id": job_id,
            "scheduler_command_succeeded": False,
            "scheduler_acceptance_verified": True,
        }
        attempt_bytes = write_exclusive_canonical(
            CONTROL / "SUBMISSION_ATTEMPT.json", attempt
        )
        attempt_digest = sha256(attempt_bytes).hexdigest()
    else:
        attempt = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_attempt.v24",
            "authorization_sha256": authorization_digest,
            "preparation_template_sha256": authorization["preparation_template_sha256"],
            "preparation_sha256": preparation_digest,
            "source_manifest_sha256": authorization["source_manifest_sha256"],
            "wrapper_sha256": authorization["wrapper_sha256"],
            "submission_intent_sha256": intent_digest,
            "sbatch_argv": argv,
            "outcome": "COMMAND_COMPLETED",
            "sbatch_returncode": completed.returncode,
            "sbatch_stdout_b64": base64.b64encode(completed.stdout).decode("ascii"),
            "sbatch_stdout_sha256": sha256(completed.stdout).hexdigest(),
            "sbatch_stderr_b64": base64.b64encode(completed.stderr).decode("ascii"),
            "sbatch_stderr_sha256": sha256(completed.stderr).hexdigest(),
            "submission_timeout_sha256": None,
            "submission_reconciliation_sha256": None,
            "reconciled_job_id": None,
            "scheduler_command_succeeded": completed.returncode == 0 and not completed.stderr,
            "scheduler_acceptance_verified": completed.returncode == 0 and not completed.stderr,
        }
        # Persist uninterpreted return code and raw streams before every result
        # check, stdout framing check, job-ID parse, or scheduler query.
        attempt_bytes = write_exclusive_canonical(
            CONTROL / "SUBMISSION_ATTEMPT.json", attempt
        )
        attempt_digest = sha256(attempt_bytes).hexdigest()
        validate_submission_attempt(
            authorization,
            authorization_digest,
            intent_digest,
            require_scheduler_success=False,
        )
        if completed.returncode != 0 or completed.stderr:
            raise RuntimeError("held sbatch failed; no retry is authorized")
        raw_job = completed.stdout
        if raw_job.count(b"\n") != 1 or not raw_job.endswith(b"\n"):
            raise RuntimeError("sbatch --parsable did not emit exactly one LF-terminated job ID")
        try:
            job_id = raw_job[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise RuntimeError("sbatch job ID is not ASCII") from error
        if not job_id.isdigit() or job_id.startswith("0"):
            raise RuntimeError("sbatch --parsable output is not one numeric job ID")
    validate_submission_attempt(
        authorization,
        authorization_digest,
        intent_digest,
        require_scheduler_success=True,
    )

    scontrol_argv, initial_raw, fields = _initial_held_record(job_id)
    matched, mismatches = exact_held_submission_candidate(
        fields, job_id, intent["not_before_submit_time"]
    )
    if not matched:
        raise RuntimeError(
            "new job is not the exact held authorized job; it remains unreleased: "
            + ",".join(mismatches)
        )
    receipt = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_receipt.v24",
        "job_id": job_id,
        "submission_outcome": attempt["outcome"],
        "submission_attempt_sha256": attempt_digest,
        "submission_intent_sha256": intent_digest,
        "authorization_sha256": authorization_digest,
        "preparation_template_sha256": authorization["preparation_template_sha256"],
        "preparation_sha256": preparation_digest,
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "wrapper_sha256": authorization["wrapper_sha256"],
        "initial_scontrol_argv": scontrol_argv,
        "initial_scontrol_stdout_b64": base64.b64encode(initial_raw).decode("ascii"),
        "initial_scontrol_stdout_sha256": sha256(initial_raw).hexdigest(),
        "initial_scheduler_comment": FIXED_AUTH_COMMENT,
        "initial_state": "PENDING",
        "initial_hold_reason": "JobHeldUser",
        "initial_user_hold_verified": True,
    }
    receipt_path = CONTROL / f"SUBMISSION_RECEIPT_{job_id}.json"
    receipt_bytes = write_exclusive_canonical(receipt_path, receipt)
    receipt_digest = sha256(receipt_bytes).hexdigest()
    validate_submission_receipt(receipt, receipt_digest)

    comment = expected_receipt_comment(receipt_digest)
    update_argv = ["/usr/bin/scontrol", "update", f"JobId={job_id}", f"Comment={comment}"]
    update_outcome: str
    update_exit_status: int | None
    update_timeout_digest: str | None
    try:
        updated = run_scheduler_command(
            update_argv,
            timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS,
            check=False,
        )
    except SchedulerCommandTimeout as error:
        timeout_payload = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_update_timeout.v24",
            "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
            "job_id": job_id,
            "submission_receipt_sha256": receipt_digest,
            "comment_update_argv": update_argv,
            "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
            "partial_stdout_b64": base64.b64encode(error.stdout).decode("ascii"),
            "partial_stdout_sha256": sha256(error.stdout).hexdigest(),
            "partial_stderr_b64": base64.b64encode(error.stderr).decode("ascii"),
            "partial_stderr_sha256": sha256(error.stderr).hexdigest(),
            "scheduler_side_effect": "UNKNOWN",
            "repeat_update_permitted": False,
        }
        timeout_bytes = write_exclusive_canonical(
            CONTROL / f"COMMENT_UPDATE_TIMEOUT_{job_id}.json", timeout_payload
        )
        update_timeout_digest = sha256(timeout_bytes).hexdigest()
        update_stdout, update_stderr = error.stdout, error.stderr
        update_exit_status = None
        try:
            reread_argv, reread_raw, reread_fields = scontrol_job(
                job_id,
                timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS,
                allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
            )
        except Exception as query_error:
            error_payload = {
                "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_update_reconciliation_error.v24",
                "record_class": "FAIL_CLOSED_RAW_QUERY_EVIDENCE",
                "job_id": job_id,
                "submission_receipt_sha256": receipt_digest,
                "comment_update_timeout_sha256": update_timeout_digest,
                "error_class": type(query_error).__name__,
                "error_message": str(query_error),
                "comment_attestation_permitted": False,
                "release_permitted": False,
            }
            write_exclusive_canonical(
                CONTROL / f"COMMENT_UPDATE_RECONCILIATION_ERROR_{job_id}.json",
                error_payload,
            )
            raise RuntimeError("comment-update timeout query was ambiguous; held job remains unreleased") from query_error
        desired_applied = (
            reread_fields.get("Comment") == comment
            and reread_fields.get("JobId") == job_id
            and reread_fields.get("JobName") == FIXED_JOB_NAME
            and reread_fields.get("JobState") == "PENDING"
            and reread_fields.get("Priority") == "0"
            and reread_fields.get("Reason") == "JobHeldUser"
            and reread_fields.get("Command") == str(WRAPPER_PATH)
            and reread_fields.get("UserId", "").split("(", 1)[0] == FIXED_SCHEDULER_USER
        )
        if not desired_applied:
            old_held = (
                reread_fields.get("Comment") == FIXED_AUTH_COMMENT
                and reread_fields.get("JobId") == job_id
                and reread_fields.get("JobName") == FIXED_JOB_NAME
                and reread_fields.get("JobState") == "PENDING"
                and reread_fields.get("Priority") == "0"
                and reread_fields.get("Reason") == "JobHeldUser"
                and reread_fields.get("Command") == str(WRAPPER_PATH)
                and reread_fields.get("UserId", "").split("(", 1)[0] == FIXED_SCHEDULER_USER
            )
            reconciliation = {
                "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_update_reconciliation.v24",
                "job_id": job_id,
                "submission_receipt_sha256": receipt_digest,
                "comment_update_timeout_sha256": update_timeout_digest,
                "reread_scontrol_argv": reread_argv,
                "reread_scontrol_stdout_b64": base64.b64encode(reread_raw).decode("ascii"),
                "reread_scontrol_stdout_sha256": sha256(reread_raw).hexdigest(),
                "outcome": "NOT_APPLIED" if old_held else "AMBIGUOUS",
                "comment_attestation_permitted": False,
                "repeat_update_permitted": False,
                "release_permitted": False,
            }
            write_exclusive_canonical(
                CONTROL / f"COMMENT_UPDATE_RECONCILIATION_{job_id}.json",
                reconciliation,
            )
            raise RuntimeError(
                "comment update timeout was not positively reconciled as applied; held job remains unreleased"
            )
        update_outcome = "TIMEOUT_RECONCILED_APPLIED"
    else:
        if updated.returncode != 0 or updated.stdout or updated.stderr:
            raise RuntimeError("scheduler receipt-comment update failed; held job remains unreleased")
        update_outcome = "COMMAND_COMPLETED"
        update_exit_status = updated.returncode
        update_stdout, update_stderr = updated.stdout, updated.stderr
        update_timeout_digest = None
        reread_argv, reread_raw, reread_fields = scontrol_job(
            job_id,
            timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS,
            allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
        )

    if not (
        reread_fields.get("JobId") == job_id
        and reread_fields.get("Comment") == comment
        and reread_fields.get("JobName") == FIXED_JOB_NAME
        and reread_fields.get("JobState") == "PENDING"
        and reread_fields.get("Priority") == "0"
        and reread_fields.get("Reason") == "JobHeldUser"
        and reread_fields.get("Command") == str(WRAPPER_PATH)
        and reread_fields.get("UserId", "").split("(", 1)[0] == FIXED_SCHEDULER_USER
    ):
        raise RuntimeError("scheduler receipt-comment reread failed; held job remains unreleased")
    comment_attestation = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_attestation.v24",
        "job_id": job_id,
        "submission_receipt_sha256": receipt_digest,
        "comment_update_argv": update_argv,
        "comment_update_outcome": update_outcome,
        "comment_update_exit_status": update_exit_status,
        "comment_update_stdout_b64": base64.b64encode(update_stdout).decode("ascii"),
        "comment_update_stderr_b64": base64.b64encode(update_stderr).decode("ascii"),
        "comment_update_timeout_sha256": update_timeout_digest,
        "reread_scontrol_argv": reread_argv,
        "reread_scontrol_stdout_b64": base64.b64encode(reread_raw).decode("ascii"),
        "reread_scontrol_stdout_sha256": sha256(reread_raw).hexdigest(),
        "scheduler_comment": comment,
        "pending_user_hold_reverified": True,
        "hold_reason": "JobHeldUser",
    }
    write_exclusive_canonical(
        CONTROL / f"COMMENT_ATTESTATION_{job_id}.json", comment_attestation
    )
    print(job_id, flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"HELD_SUBMISSION_INCOMPLETE: {error}", file=sys.stderr, flush=True)
        os._exit(2)
    else:
        os._exit(code)
