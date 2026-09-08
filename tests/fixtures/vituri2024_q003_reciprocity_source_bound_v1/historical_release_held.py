#!/data/home/ziyuzhu/miniconda3/bin/python -S
"""No-argument timeout-safe release/recovery helper for one receipt-bound v13 job."""
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
    MAX_RELEASE_ATTEMPTS,
    RELEASE_QUERY_POLL_SECONDS,
    RELEASE_QUERY_TIMEOUT_SECONDS,
    ROOT,
    SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    SCHEDULER_CALL_TIMEOUT_SECONDS,
    SCHEDULER_POLL_CALL_CAP_SECONDS,
    SchedulerCommandTimeout,
    _validate_release_action,
    bounded_poll_call_timeout,
    exact_receipt_bound_job,
    exact_user_held_job,
    expected_receipt_comment,
    file_sha256,
    load_canonical_json,
    normalize_scontrol_one_line,
    parse_scontrol_one_line,
    positive_hold_absent,
    release_action_path,
    release_intent_path,
    release_reconciliation_path,
    release_timeout_path,
    scontrol_job,
    unique_submission_receipt,
    validate_comment_attestation,
    validate_release_intent,
    validate_submission_receipt,
    verify_capsule_manifest,
    write_exclusive_canonical,
    run_scheduler_command,
)


def run_release_command(
    argv: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    return run_scheduler_command(
        argv, timeout_seconds=timeout_seconds, check=False
    )


def query_release_job(
    job_id: str, *, timeout_seconds: float
) -> tuple[list[str], bytes, dict[str, str]]:
    return scontrol_job(
        job_id,
        timeout_seconds=timeout_seconds,
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )


def poll_until_positive_hold_absence(
    job_id: str,
    comment: str,
    *,
    timeout_seconds: float = RELEASE_QUERY_TIMEOUT_SECONDS,
    poll_seconds: float = RELEASE_QUERY_POLL_SECONDS,
    per_call_cap_seconds: float = SCHEDULER_POLL_CALL_CAP_SECONDS,
    query: Callable[..., tuple[list[str], bytes, dict[str, str]]] | None = None,
) -> tuple[list[str], bytes, dict[str, str], int]:
    if timeout_seconds <= 0.0 or poll_seconds <= 0.0 or per_call_cap_seconds <= 0.0:
        raise RuntimeError("release query poll bounds must be positive")
    query = query_release_job if query is None else query
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise SchedulerCommandTimeout(
                "bounded release scheduler poll timed out fail-closed"
            )
        attempts += 1
        argv, raw, fields = query(
            job_id,
            timeout_seconds=bounded_poll_call_timeout(
                remaining, per_call_cap_seconds
            ),
        )
        if not exact_receipt_bound_job(fields, job_id, comment):
            raise RuntimeError("post-release scheduler identity/comment/command mismatch")
        if positive_hold_absent(fields):
            return argv, raw, fields, attempts
        transitional_pending = (
            fields.get("JobState") == "PENDING"
            and fields.get("Reason") != "JobHeldUser"
        )
        if not exact_user_held_job(fields, job_id, comment) and not transitional_pending:
            raise RuntimeError("post-release scheduler state is ambiguous")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise SchedulerCommandTimeout(
                "bounded release scheduler poll timed out fail-closed while job remained held"
            )
        time.sleep(min(poll_seconds, remaining))


def _release_query_snapshot(
    sequence: int,
    job_id: str,
    comment: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    argv = ["/usr/bin/scontrol", "show", "job", job_id, "-o"]
    try:
        actual_argv, raw, fields = query_release_job(
            job_id, timeout_seconds=timeout_seconds
        )
    except SchedulerCommandTimeout as error:
        return {
            "sequence": sequence,
            "query_status": "TIMEOUT",
            "scontrol_argv": argv,
            "scontrol_returncode": None,
            "scontrol_stdout_b64": base64.b64encode(error.stdout).decode("ascii"),
            "scontrol_stdout_sha256": sha256(error.stdout).hexdigest(),
            "scontrol_stderr_b64": base64.b64encode(error.stderr).decode("ascii"),
            "scontrol_stderr_sha256": sha256(error.stderr).hexdigest(),
            "error_class": "SchedulerCommandTimeout",
            "classification": "AMBIGUOUS",
        }
    except Exception as error:
        return {
            "sequence": sequence,
            "query_status": "ERROR",
            "scontrol_argv": argv,
            "scontrol_returncode": None,
            "scontrol_stdout_b64": "",
            "scontrol_stdout_sha256": sha256(b"").hexdigest(),
            "scontrol_stderr_b64": "",
            "scontrol_stderr_sha256": sha256(b"").hexdigest(),
            "error_class": type(error).__name__,
            "classification": "AMBIGUOUS",
        }
    if actual_argv != argv:
        raise RuntimeError("release reconciliation query argv drifted")
    if exact_user_held_job(fields, job_id, comment):
        classification = "HELD"
    elif exact_receipt_bound_job(fields, job_id, comment) and positive_hold_absent(fields):
        classification = "HOLD_ABSENT"
    else:
        classification = "AMBIGUOUS"
    return {
        "sequence": sequence,
        "query_status": "COMPLETED",
        "scontrol_argv": argv,
        "scontrol_returncode": 0,
        "scontrol_stdout_b64": base64.b64encode(raw).decode("ascii"),
        "scontrol_stdout_sha256": sha256(raw).hexdigest(),
        "scontrol_stderr_b64": "",
        "scontrol_stderr_sha256": sha256(b"").hexdigest(),
        "error_class": None,
        "classification": classification,
    }


def reconcile_release_timeout(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    intent_digest: str,
    timeout_digest: str,
    *,
    timeout_seconds: float = RELEASE_QUERY_TIMEOUT_SECONDS,
    poll_seconds: float = RELEASE_QUERY_POLL_SECONDS,
    per_call_cap_seconds: float = SCHEDULER_POLL_CALL_CAP_SECONDS,
) -> tuple[dict[str, Any], bytes]:
    if timeout_seconds <= 0.0 or poll_seconds <= 0.0 or per_call_cap_seconds <= 0.0:
        raise RuntimeError("release reconciliation bounds must be positive")
    comment = expected_receipt_comment(submission_digest)
    deadline = time.monotonic() + timeout_seconds
    snapshots: list[dict[str, Any]] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            outcome = (
                "NOT_APPLIED"
                if snapshots and all(item["classification"] == "HELD" for item in snapshots)
                else "AMBIGUOUS"
            )
            break
        call_timeout = bounded_poll_call_timeout(
            remaining, per_call_cap_seconds
        )
        snapshot = _release_query_snapshot(
            len(snapshots) + 1,
            job_id,
            comment,
            timeout_seconds=call_timeout,
        )
        snapshots.append(snapshot)
        classification = snapshot["classification"]
        if classification == "HOLD_ABSENT":
            outcome = "APPLIED"
            break
        if classification == "AMBIGUOUS":
            outcome = "AMBIGUOUS"
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            outcome = "NOT_APPLIED"
            break
        time.sleep(min(poll_seconds, remaining))
    payload = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_reconciliation.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "release_intent_sha256": intent_digest,
        "release_timeout_sha256": timeout_digest,
        "query_snapshots": snapshots,
        "outcome": outcome,
        "retry_permitted": outcome == "NOT_APPLIED" and attempt_number < MAX_RELEASE_ATTEMPTS,
        "release_verification_permitted": outcome == "APPLIED",
    }
    data = write_exclusive_canonical(
        release_reconciliation_path(job_id, attempt_number), payload
    )
    return payload, data


def _publish_verification_from_raw(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    action_digest: str,
    post_argv: list[str],
    post_raw: bytes,
    post: dict[str, str],
    poll_attempts: int,
    *,
    recovered: bool,
) -> None:
    action = load_canonical_json(release_action_path(job_id, attempt_number))
    if not exact_receipt_bound_job(
        post, job_id, expected_receipt_comment(submission_digest)
    ) or not positive_hold_absent(post):
        raise RuntimeError("positive release verification state is absent")
    verification = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_verification.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "release_action_path": release_action_path(job_id, attempt_number).name,
        "release_action_sha256": action_digest,
        "action_attempt_number": attempt_number,
        "action_outcome": action["outcome"],
        "post_release_scontrol_argv": post_argv,
        "post_release_scontrol_stdout_b64": base64.b64encode(post_raw).decode("ascii"),
        "post_release_scontrol_sha256": sha256(post_raw).hexdigest(),
        "post_release_state": post.get("JobState"),
        "post_release_reason": post.get("Reason"),
        "post_release_priority": post.get("Priority"),
        "post_release_poll_attempts": poll_attempts,
        "scheduler_comment": post.get("Comment"),
        "positive_hold_absence_verified": True,
        "timeout_reconciled": action["outcome"] == "TIMEOUT_RECONCILED_APPLIED",
        "deliberate_retry_used": attempt_number > 1,
        "recovered_after_prior_post_query_incomplete": recovered,
        "release_verified": True,
    }
    write_exclusive_canonical(
        CONTROL / f"RELEASE_VERIFICATION_{job_id}.json", verification
    )


def _publish_verification_by_poll(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    action_digest: str,
    *,
    recovered: bool,
) -> None:
    post_argv, post_raw, post, attempts = poll_until_positive_hold_absence(
        job_id, expected_receipt_comment(submission_digest)
    )
    _publish_verification_from_raw(
        job_id,
        submission_digest,
        attempt_number,
        action_digest,
        post_argv,
        post_raw,
        post,
        attempts,
        recovered=recovered,
    )


def _existing_actions(job_id: str) -> list[tuple[int, Path, dict[str, Any]]]:
    result: list[tuple[int, Path, dict[str, Any]]] = []
    for number in range(1, MAX_RELEASE_ATTEMPTS + 1):
        path = release_action_path(job_id, number)
        if path.exists():
            result.append((number, path, load_canonical_json(path)))
    if [number for number, _, _ in result] not in (
        [],
        [1],
        list(range(1, len(result) + 1)),
    ):
        raise RuntimeError("release action numbering is noncontiguous")
    return result


def _action_payload(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    intent_digest: str,
    *,
    outcome: str,
    returncode: int | None,
    stdout: bytes,
    stderr: bytes,
    timeout_digest: str | None,
    reconciliation_digest: str | None,
) -> dict[str, Any]:
    applied = outcome in {
        "COMMAND_COMPLETED_APPLIED",
        "TIMEOUT_RECONCILED_APPLIED",
    }
    return {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_action.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "release_intent_sha256": intent_digest,
        "release_argv": ["/usr/bin/scontrol", "release", job_id],
        "outcome": outcome,
        "release_returncode": returncode,
        "release_stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "release_stdout_sha256": sha256(stdout).hexdigest(),
        "release_stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "release_stderr_sha256": sha256(stderr).hexdigest(),
        "release_timeout_sha256": timeout_digest,
        "release_reconciliation_sha256": reconciliation_digest,
        "scheduler_effect_applied": applied,
        "retry_permitted": (
            outcome == "TIMEOUT_RECONCILED_NOT_APPLIED"
            and attempt_number < MAX_RELEASE_ATTEMPTS
        ),
    }


def main() -> int:
    if len(sys.argv) != 1 or Path(__file__).resolve(strict=True) != ROOT / "bin/release_held.py":
        raise RuntimeError("release_held.py is a sealed no-argument entry point")
    verify_capsule_manifest(require_dynamic_empty=False)
    receipt_path, submission, submission_digest = unique_submission_receipt()
    job_id = validate_submission_receipt(submission, submission_digest)
    validate_comment_attestation(job_id, submission_digest)
    comment = expected_receipt_comment(submission_digest)
    release_argv = ["/usr/bin/scontrol", "release", job_id]
    verification_path = CONTROL / f"RELEASE_VERIFICATION_{job_id}.json"
    if verification_path.exists():
        raise RuntimeError("release already has a durable positive verification")

    actions = _existing_actions(job_id)
    if actions:
        number, path, action = actions[-1]
        _validate_release_action(
            action,
            job_id,
            submission_digest,
            number,
            require_applied=False,
        )
        if action["scheduler_effect_applied"] is True:
            _publish_verification_by_poll(
                job_id,
                submission_digest,
                number,
                file_sha256(path),
                recovered=True,
            )
            print(job_id, flush=True)
            return 0
        if (
            action["outcome"] != "TIMEOUT_RECONCILED_NOT_APPLIED"
            or action["retry_permitted"] is not True
            or number >= MAX_RELEASE_ATTEMPTS
        ):
            raise RuntimeError("release chain is not eligible for deliberate retry")
        attempt_number = number + 1
    else:
        # An intent/reconciliation without an action is ambiguous and cannot be retried.
        if any(CONTROL.glob(f"RELEASE_INTENT_{job_id}_*.json")):
            raise RuntimeError("release has ambiguous durable evidence and no retryable action")
        attempt_number = 1

    pre_argv, pre_raw, pre = query_release_job(
        job_id, timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS
    )
    if not exact_user_held_job(pre, job_id, comment):
        raise RuntimeError("job is not uniquely receipt-bound, pending, and user-held")
    retry_precheck_argv: list[str] | None = None
    retry_precheck_raw: bytes | None = None
    if attempt_number > 1:
        retry_precheck_argv = pre_argv
        retry_precheck_raw = pre_raw
        # A deliberate retry requires a second fresh, separated exact-held
        # observation after the prior bounded NOT_APPLIED reconciliation. The
        # timed-out client process has already been killed/waited by subprocess.
        time.sleep(RELEASE_QUERY_POLL_SECONDS)
        pre_argv, pre_raw, pre = query_release_job(
            job_id, timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS
        )
        if not exact_user_held_job(pre, job_id, comment):
            raise RuntimeError("release retry lost its second exact still-held proof")
    prior_digest = (
        file_sha256(release_action_path(job_id, attempt_number - 1))
        if attempt_number > 1
        else None
    )
    intent = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_intent.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "prior_not_applied_action_sha256": prior_digest,
        "retry_precheck_scontrol_argv": retry_precheck_argv,
        "retry_precheck_scontrol_stdout_b64": (
            base64.b64encode(retry_precheck_raw).decode("ascii")
            if retry_precheck_raw is not None else None
        ),
        "retry_precheck_scontrol_sha256": (
            sha256(retry_precheck_raw).hexdigest()
            if retry_precheck_raw is not None else None
        ),
        "release_argv": release_argv,
        "pre_release_scontrol_argv": pre_argv,
        "pre_release_scontrol_stdout_b64": base64.b64encode(pre_raw).decode("ascii"),
        "pre_release_scontrol_sha256": sha256(pre_raw).hexdigest(),
        "pending_user_hold_verified": True,
        "hold_reason": "JobHeldUser",
        "deliberate_retry": attempt_number > 1,
    }
    intent_bytes = write_exclusive_canonical(
        release_intent_path(job_id, attempt_number), intent
    )
    intent_digest = sha256(intent_bytes).hexdigest()
    validate_release_intent(job_id, submission_digest, attempt_number)

    try:
        released = run_release_command(
            release_argv, timeout_seconds=SCHEDULER_CALL_TIMEOUT_SECONDS
        )
    except SchedulerCommandTimeout as error:
        timeout_payload = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_timeout.v24",
            "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
            "job_id": job_id,
            "submission_receipt_sha256": submission_digest,
            "attempt_number": attempt_number,
            "release_intent_sha256": intent_digest,
            "release_argv": release_argv,
            "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
            "release_partial_stdout_b64": base64.b64encode(error.stdout).decode("ascii"),
            "release_partial_stdout_sha256": sha256(error.stdout).hexdigest(),
            "release_partial_stderr_b64": base64.b64encode(error.stderr).decode("ascii"),
            "release_partial_stderr_sha256": sha256(error.stderr).hexdigest(),
            "scheduler_side_effect": "UNKNOWN",
            "retry_permitted_before_reconciliation": False,
        }
        timeout_bytes = write_exclusive_canonical(
            release_timeout_path(job_id, attempt_number), timeout_payload
        )
        timeout_digest = sha256(timeout_bytes).hexdigest()
        reconciliation, reconciliation_bytes = reconcile_release_timeout(
            job_id,
            submission_digest,
            attempt_number,
            intent_digest,
            timeout_digest,
            timeout_seconds=RELEASE_QUERY_TIMEOUT_SECONDS,
            poll_seconds=RELEASE_QUERY_POLL_SECONDS,
            per_call_cap_seconds=SCHEDULER_POLL_CALL_CAP_SECONDS,
        )
        reconciliation_digest = sha256(reconciliation_bytes).hexdigest()
        if reconciliation["outcome"] == "AMBIGUOUS":
            raise RuntimeError("release timeout query is ambiguous; fail-closed")
        outcome = (
            "TIMEOUT_RECONCILED_APPLIED"
            if reconciliation["outcome"] == "APPLIED"
            else "TIMEOUT_RECONCILED_NOT_APPLIED"
        )
        action = _action_payload(
            job_id,
            submission_digest,
            attempt_number,
            intent_digest,
            outcome=outcome,
            returncode=None,
            stdout=error.stdout,
            stderr=error.stderr,
            timeout_digest=timeout_digest,
            reconciliation_digest=reconciliation_digest,
        )
        action_bytes = write_exclusive_canonical(
            release_action_path(job_id, attempt_number), action
        )
        _validate_release_action(
            action,
            job_id,
            submission_digest,
            attempt_number,
            require_applied=False,
        )
        if outcome == "TIMEOUT_RECONCILED_NOT_APPLIED":
            raise RuntimeError(
                "release timeout remained held through the bounded window; rerun the no-argument helper deliberately to retry"
            )
        final_snapshot = reconciliation["query_snapshots"][-1]
        post_raw = base64.b64decode(final_snapshot["scontrol_stdout_b64"], validate=True)
        query = (
            final_snapshot["scontrol_argv"],
            post_raw,
            parse_scontrol_one_line(
                normalize_scontrol_one_line(post_raw),
                allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
            ),
        )
        _publish_verification_from_raw(
            job_id,
            submission_digest,
            attempt_number,
            sha256(action_bytes).hexdigest(),
            query[0],
            query[1],
            query[2],
            len(reconciliation["query_snapshots"]),
            recovered=False,
        )
    else:
        outcome = (
            "COMMAND_COMPLETED_APPLIED"
            if released.returncode == 0 and not released.stdout and not released.stderr
            else "COMMAND_COMPLETED_FAILED"
        )
        action = _action_payload(
            job_id,
            submission_digest,
            attempt_number,
            intent_digest,
            outcome=outcome,
            returncode=released.returncode,
            stdout=released.stdout,
            stderr=released.stderr,
            timeout_digest=None,
            reconciliation_digest=None,
        )
        action_bytes = write_exclusive_canonical(
            release_action_path(job_id, attempt_number), action
        )
        _validate_release_action(
            action,
            job_id,
            submission_digest,
            attempt_number,
            require_applied=False,
        )
        if outcome != "COMMAND_COMPLETED_APPLIED":
            raise RuntimeError("release command failed; fail-closed without retry")
        _publish_verification_by_poll(
            job_id,
            submission_digest,
            attempt_number,
            sha256(action_bytes).hexdigest(),
            recovered=False,
        )

    print(job_id, flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"RELEASE_INCOMPLETE: {error}", file=sys.stderr, flush=True)
        os._exit(2)
    else:
        os._exit(code)
