#!/usr/bin/env python3
"""Standard-library-only control-plane validation for the sealed v24 capsule."""
from __future__ import annotations

import base64
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable

ROOT = Path("/data/home/ziyuzhu/.runs/Mean_Field_6dcc724_vituri_fig2_q003_whole_inventory_reciprocity_comparison_v24_20260907")
AUTH_PATH = ROOT / "config/LAUNCH_AUTHORIZATION.json"
PREPARATION_PATH = ROOT / "provenance/PREPARATION_RECORD.json"
TEMPLATE_PATH = ROOT / "provenance/PREPARATION_TEMPLATE.json"
SOURCE_MANIFEST_PATH = ROOT / "provenance/SOURCE_SHA256SUMS.txt"
WRAPPER_PATH = ROOT / "submit.sbatch"
CONTROL = ROOT / "control"
CAPSULE_MANIFEST_PATH = ROOT / "provenance/CAPSULE_MANIFEST.json"
FIXED_AUTH_COMMENT = "MF-Q003-RECIP-V24-6DCC724-HELD"
FIXED_JOB_NAME = "vituri_q003_reciprocity_v24"
FIXED_SCHEDULER_USER = "ziyuzhu"
_JOB_ID = re.compile(r"[1-9][0-9]*")
_POSITIVE_PRIORITY = re.compile(r"[1-9][0-9]*")
_PENDING_REASON = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SUBMIT_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
_HEX = re.compile(r"[0-9a-f]{64}")
SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS = frozenset({"AllocTRES", "NodeList"})
_SCONTROL_SCHEMA_EMPTY_FIELDS = SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
RELEASE_RECORD_WAIT_SECONDS = 120.0
RELEASE_RECORD_POLL_SECONDS = 0.05
RELEASE_QUERY_TIMEOUT_SECONDS = 30.0
RELEASE_QUERY_POLL_SECONDS = 0.05
SUBMISSION_RECONCILIATION_TIMEOUT_SECONDS = 30.0
SUBMISSION_RECONCILIATION_POLL_SECONDS = 0.05
SCHEDULER_CALL_TIMEOUT_SECONDS = 10.0
SCHEDULER_POLL_CALL_CAP_SECONDS = 5.0
MAX_RELEASE_ATTEMPTS = 2


class SchedulerCommandTimeout(RuntimeError):
    """A scheduler subprocess timed out with exact partial-stream evidence."""

    def __init__(
        self,
        message: str,
        *,
        argv: list[str] | None = None,
        timeout_seconds: float | None = None,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        super().__init__(message)
        self.argv = list(argv) if argv is not None else []
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr


def _finite_positive_seconds(value: object, label: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeError(f"{label} must be a finite positive number")
    return float(value)


def bounded_poll_call_timeout(remaining_seconds: float, per_call_cap_seconds: float) -> float:
    remaining = _finite_positive_seconds(remaining_seconds, "remaining monotonic budget")
    cap = _finite_positive_seconds(per_call_cap_seconds, "scheduler per-call timeout cap")
    return min(cap, remaining)


def run_scheduler_command(
    argv: list[str],
    *,
    timeout_seconds: float,
    check: bool,
) -> subprocess.CompletedProcess[bytes]:
    timeout = _finite_positive_seconds(timeout_seconds, "scheduler command timeout")
    if type(argv) is not list or not argv or any(type(token) is not str for token in argv):
        raise RuntimeError("scheduler argv must be one nonempty exact string list")
    try:
        return subprocess.run(
            argv,
            check=check,
            text=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raw_stdout = error.stdout if error.stdout is not None else getattr(error, "output", None)
        raw_stderr = error.stderr
        if raw_stdout is None:
            raw_stdout = b""
        if raw_stderr is None:
            raw_stderr = b""
        if isinstance(raw_stdout, str):
            raw_stdout = raw_stdout.encode("utf-8")
        if isinstance(raw_stderr, str):
            raw_stderr = raw_stderr.encode("utf-8")
        if not isinstance(raw_stdout, bytes) or not isinstance(raw_stderr, bytes):
            raw_stdout = b""
            raw_stderr = b""
        raise SchedulerCommandTimeout(
            f"scheduler command timed out fail-closed after {timeout:.6g}s: {argv[0]}",
            argv=argv,
            timeout_seconds=timeout,
            stdout=raw_stdout,
            stderr=raw_stderr,
        ) from error


def require_exact_json_record(
    actual: object, expected: object, label: str, path: str = "$"
) -> None:
    """Reject equality aliases, extra keys, and all recursive JSON type drift."""
    if type(actual) is not type(expected):
        raise RuntimeError(
            f"{label} exact type mismatch at {path}: "
            f"{type(actual).__name__} != {type(expected).__name__}"
        )
    if type(expected) is dict:
        actual_dict = actual
        expected_dict = expected
        if set(actual_dict) != set(expected_dict):
            raise RuntimeError(f"{label} exact key-set mismatch at {path}")
        for key in expected_dict:
            require_exact_json_record(
                actual_dict[key], expected_dict[key], label, f"{path}.{key}"
            )
        return
    if type(expected) is list:
        actual_list = actual
        expected_list = expected
        if len(actual_list) != len(expected_list):
            raise RuntimeError(f"{label} exact list length mismatch at {path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual_list, expected_list, strict=True)
        ):
            require_exact_json_record(
                actual_item, expected_item, label, f"{path}[{index}]"
            )
        return
    if actual != expected:
        raise RuntimeError(f"{label} exact value mismatch at {path}")


def require_job_id(value: object, label: str = "job ID") -> str:
    if type(value) is not str or _JOB_ID.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not one canonical positive Slurm job ID")
    return value


def require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not one strict lowercase SHA-256 string")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_owned(path: Path, allowed_modes: set[int]) -> os.stat_result:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"not one owner-controlled regular file: {path}")
    if stat.S_IMODE(info.st_mode) not in allowed_modes:
        raise RuntimeError(f"sealed file mode mismatch: {path}")
    return info


def load_canonical_json(path: Path, *, expected_sha256: str | None = None, modes: set[int] | None = None) -> dict[str, Any]:
    _regular_owned(path, modes or {0o400, 0o440, 0o444})
    data = path.read_bytes()
    if expected_sha256 is not None and sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError(f"hash mismatch: {path}")
    payload = json.loads(data)
    if not isinstance(payload, dict) or data != canonical_json_bytes(payload):
        raise RuntimeError(f"noncanonical JSON: {path}")
    return payload


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_canonical(path: Path, payload: dict[str, Any], mode: int = 0o400) -> bytes:
    if path.parent.resolve(strict=True) != CONTROL.resolve(strict=True):
        raise RuntimeError("control receipt must be a direct child of the canonical control directory")
    data = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW, 0o200)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short control-record write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(CONTROL)
    if path.read_bytes() != data:
        raise RuntimeError("durable control-record reread mismatch")
    return data


def normalize_scontrol_one_line(raw: bytes) -> str:
    if not isinstance(raw, bytes) or raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise RuntimeError("scontrol response is not exactly one LF-terminated byte record")
    line = raw[:-1]
    if b"\r" in line or any(byte < 32 or byte == 127 for byte in line):
        raise RuntimeError("scontrol response contains CR/control bytes")
    trailing = len(line) - len(line.rstrip(b" "))
    if trailing > 1:
        raise RuntimeError("scontrol response has multiple trailing spaces")
    if trailing == 1:
        line = line[:-1]
    if not line or line.endswith(b" "):
        raise RuntimeError("scontrol response is empty/noncanonical")
    try:
        return line.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("scontrol response is not ASCII") from error


def parse_scontrol_one_line(
    text: str, *, allow_empty_fields: frozenset[str] = frozenset()
) -> dict[str, str]:
    if type(allow_empty_fields) is not frozenset or not allow_empty_fields.issubset(
        _SCONTROL_SCHEMA_EMPTY_FIELDS
    ):
        raise RuntimeError("scontrol empty-field allowance is not one exact schema subset")
    if not text or text != text.strip() or "\n" in text or "\r" in text:
        raise RuntimeError("parsed scontrol input is not one canonical line")
    matches = list(re.finditer(r"(?:^| )([A-Za-z][A-Za-z0-9_/:.]*)=", text))
    if not matches or matches[0].start() != 0:
        raise RuntimeError("scontrol record lacks a leading identity field")
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise RuntimeError(f"duplicate scontrol field: {key}")
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:stop].strip()
        if not value and key not in allow_empty_fields:
            raise RuntimeError(f"empty scontrol field: {key}")
        fields[key] = value
    return fields


def scontrol_job(
    job_id: str,
    *,
    timeout_seconds: float = SCHEDULER_CALL_TIMEOUT_SECONDS,
    allow_empty_fields: frozenset[str] = frozenset(),
) -> tuple[list[str], bytes, dict[str, str]]:
    if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
        raise RuntimeError("invalid Slurm job ID")
    argv = ["/usr/bin/scontrol", "show", "job", job_id, "-o"]
    completed = run_scheduler_command(
        argv, timeout_seconds=timeout_seconds, check=True
    )
    if completed.stderr:
        raise RuntimeError("scontrol emitted stderr")
    raw = completed.stdout
    return argv, raw, parse_scontrol_one_line(
        normalize_scontrol_one_line(raw), allow_empty_fields=allow_empty_fields
    )


def approved_export(preparation_sha256: str) -> str:
    require_sha256(preparation_sha256, "preparation hash")
    return f"NONE,EXPECTED_PREPARATION_SHA256={preparation_sha256}"


def approved_sbatch_argv(preparation_sha256: str) -> list[str]:
    return [
        "/usr/bin/sbatch",
        "--hold",
        "--parsable",
        f"--comment={FIXED_AUTH_COMMENT}",
        f"--export={approved_export(preparation_sha256)}",
        "--account=hmt03",
        "--partition=regular256",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=64",
        "--mem=0",
        "--exclusive",
        f"--job-name={FIXED_JOB_NAME}",
        f"--output={ROOT}/logs/slurm_%j.out",
        f"--error={ROOT}/logs/slurm_%j.err",
        str(WRAPPER_PATH),
    ]

def submission_candidate_query_argv() -> list[str]:
    return [
        "/usr/bin/squeue",
        "--noheader",
        f"--user={FIXED_SCHEDULER_USER}",
        f"--name={FIXED_JOB_NAME}",
        "--states=PENDING",
        "--format=%A",
    ]

def parse_squeue_job_ids(raw: bytes) -> list[str]:
    if not isinstance(raw, bytes) or b"\r" in raw:
        raise RuntimeError("squeue candidate output is not exact LF-delimited bytes")
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("squeue candidate output lacks final LF")
    ids: list[str] = []
    for line in raw.splitlines():
        try:
            value = line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise RuntimeError("squeue candidate output is not ASCII") from error
        require_job_id(value, "squeue candidate job ID")
        if value in ids:
            raise RuntimeError("squeue returned a duplicate candidate job ID")
        ids.append(value)
    return ids

def scheduler_username(fields: dict[str, str]) -> str | None:
    user_id = fields.get("UserId")
    if not isinstance(user_id, str):
        return None
    matched = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.-]*)\(([0-9]+)\)", user_id)
    return matched.group(1) if matched is not None else None

def _parse_submit_time(value: object, label: str) -> datetime:
    if type(value) is not str or _SUBMIT_TIME.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not one exact scheduler timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError as error:
        raise RuntimeError(f"{label} is not a real calendar timestamp") from error

def exact_resource_binding(fields: dict[str, str]) -> bool:
    return (
        fields.get("Account") == "hmt03"
        and fields.get("Partition") == "regular256"
        and fields.get("NumNodes")
        == ("1-1" if fields.get("JobState") == "PENDING" else "1")
        and fields.get("NumCPUs") == "64"
        and fields.get("NumTasks") == "1"
        and fields.get("CPUs/Task") == "64"
        and fields.get("MinCPUsNode") == "64"
        and fields.get("MinMemoryNode") == "0"
        and fields.get("OverSubscribe") == "NO"
    )


def exact_held_submission_candidate(
    fields: dict[str, str], job_id: str, not_before_submit_time: str
) -> tuple[bool, list[str]]:
    require_job_id(job_id)
    not_before = _parse_submit_time(
        not_before_submit_time, "submission intent not-before time"
    )
    expected = {
        "JobId": job_id,
        "JobName": FIXED_JOB_NAME,
        "Comment": FIXED_AUTH_COMMENT,
        "Command": str(WRAPPER_PATH),
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Priority": "0",
    }
    mismatches = [key for key, value in expected.items() if fields.get(key) != value]
    if scheduler_username(fields) != FIXED_SCHEDULER_USER:
        mismatches.append("UserId")
    if not exact_resource_binding(fields):
        mismatches.append("ResourceBinding")
    submit_time = fields.get("SubmitTime")
    try:
        submitted = _parse_submit_time(submit_time, "candidate SubmitTime")
    except RuntimeError:
        mismatches.append("SubmitTime")
    else:
        if submitted < not_before:
            mismatches.append("SubmitTime")
    return not mismatches, sorted(set(mismatches))

def exact_receipt_bound_job(
    fields: dict[str, str], job_id: str, comment: str
) -> bool:
    return (
        fields.get("JobId") == job_id
        and fields.get("Comment") == comment
        and fields.get("Command") == str(WRAPPER_PATH)
        and fields.get("JobName") == FIXED_JOB_NAME
        and scheduler_username(fields) == FIXED_SCHEDULER_USER
        and exact_resource_binding(fields)
    )

def exact_user_held_job(fields: dict[str, str], job_id: str, comment: str) -> bool:
    return (
        exact_receipt_bound_job(fields, job_id, comment)
        and fields.get("JobState") == "PENDING"
        and fields.get("Priority") == "0"
        and fields.get("Reason") == "JobHeldUser"
    )


def verify_capsule_manifest(*, require_dynamic_empty: bool = False) -> str:
    payload = load_canonical_json(CAPSULE_MANIFEST_PATH)
    if payload.get("schema") != "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_capsule_manifest.v24":
        raise RuntimeError("capsule manifest schema mismatch")
    rows = payload.get("files")
    directories = payload.get("directories")
    if not isinstance(rows, list) or not isinstance(directories, list):
        raise RuntimeError("capsule manifest inventories are malformed")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes", "mode"}:
            raise RuntimeError("capsule file-manifest row shape mismatch")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or relative in indexed
            or relative.startswith(("output/", "logs/", "control/"))
            or relative in {
                "provenance/CAPSULE_MANIFEST.json",
                "config/LAUNCH_AUTHORIZATION.json",
            }
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _HEX.fullmatch(str(row.get("sha256")))
            or type(row.get("size_bytes")) is not int
            or not re.fullmatch(r"0[0-7]{3}", str(row.get("mode")))
        ):
            raise RuntimeError("unsafe capsule file-manifest row")
        path = ROOT / relative
        info = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != row["size_bytes"]
            or stat.S_IMODE(info.st_mode) != int(row["mode"], 8)
            or file_sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"sealed capsule file mismatch: {relative}")
        indexed[relative] = row
    actual_files: set[str] = set()
    actual_directories: dict[str, int] = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("output/", "logs/", "control/")):
            continue
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"capsule symlink is forbidden: {relative}")
        if stat.S_ISDIR(info.st_mode):
            actual_directories[relative] = stat.S_IMODE(info.st_mode)
        elif stat.S_ISREG(info.st_mode):
            if relative in {
                "provenance/CAPSULE_MANIFEST.json",
                "config/LAUNCH_AUTHORIZATION.json",
            } or relative.startswith(("output/", "logs/", "control/")):
                continue
            actual_files.add(relative)
        else:
            raise RuntimeError(f"non-file capsule member is forbidden: {relative}")
    if set(indexed) != actual_files:
        raise RuntimeError("sealed capsule file inventory is incomplete")
    expected_directories: dict[str, int] = {}
    for row in directories:
        if not isinstance(row, dict) or set(row) != {"path", "mode"}:
            raise RuntimeError("capsule directory-manifest row shape mismatch")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or relative in expected_directories
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not re.fullmatch(r"0[0-7]{3}", str(row.get("mode")))
        ):
            raise RuntimeError("unsafe capsule directory-manifest row")
        expected_directories[relative] = int(row["mode"], 8)
    if expected_directories != actual_directories:
        raise RuntimeError("sealed capsule directory inventory/modes mismatch")
    if (
        payload.get("dynamic_empty_directories") != ["control", "logs", "output"]
        or payload.get("root_mode") != "0555"
        or stat.S_IMODE(os.stat(ROOT, follow_symlinks=False).st_mode) != 0o555
    ):
        raise RuntimeError("capsule root/dynamic directory policy drifted")
    if require_dynamic_empty and any(
        any((ROOT / name).iterdir()) for name in ("control", "logs", "output")
    ):
        raise RuntimeError("sealed capsule dynamic namespaces are not initially empty")
    return file_sha256(CAPSULE_MANIFEST_PATH)


def load_authorization() -> tuple[dict[str, Any], str]:
    preparation_digest = file_sha256(PREPARATION_PATH)
    template_digest = file_sha256(TEMPLATE_PATH)
    authorization = load_canonical_json(AUTH_PATH)
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_launch_authorization.v24",
        "decision": "AUTHORIZED_EXACTLY_ONE_HELD_SUBMISSION_NOT_RELEASE",
        "capsule_path": str(ROOT),
        "preparation_template_sha256": template_digest,
        "preparation_sha256": preparation_digest,
        "capsule_manifest_sha256": file_sha256(CAPSULE_MANIFEST_PATH),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST_PATH),
        "wrapper_sha256": file_sha256(WRAPPER_PATH),
        "fixed_initial_scheduler_comment": FIXED_AUTH_COMMENT,
        "approved_export": approved_export(preparation_digest),
        "approved_sbatch_argv": approved_sbatch_argv(preparation_digest),
        "sbatch_export_option_count": 1,
        "exported_variable_names": ["EXPECTED_PREPARATION_SHA256"],
        "launch_authorization_path": str(AUTH_PATH),
        "launch_authorization_path_source": "fixed_sealed_internal_path_not_sbatch_export",
        "submission_must_be_held": True,
        "release_authorized_by_this_file": False,
        "scheduler_comment_after_receipt_publication": "bare_submission_receipt_sha256_64_lowercase_hex",
        "scheduler_comment_exact_bare_equality_required": True,
        "receipt_protocol": "intent_O_EXCL_fsync_then_sbatch_timeout_record_then_bounded_unique_reconciliation_then_typed_attempt_receipt_v3",
        "release_control_protocol": "numbered_intent_then_command_or_timeout_reconciliation_then_typed_action_then_positive_verification_v3",
        "release_post_query_exception_recovery_without_second_action": True,
        "release_positive_hold_absence_policy": "RUNNING_OR_CONFIGURING_OR_PENDING_WITH_PRESENT_VALID_NONHOLD_REASON_AND_PRIORITY_CANONICAL_POSITIVE_INTEGER",
        "release_query_timeout_seconds": RELEASE_QUERY_TIMEOUT_SECONDS,
        "release_query_poll_seconds": RELEASE_QUERY_POLL_SECONDS,
        "runner_release_record_wait_seconds": RELEASE_RECORD_WAIT_SECONDS,
        "runner_release_record_poll_seconds": RELEASE_RECORD_POLL_SECONDS,
        "scheduler_call_timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
        "scheduler_poll_call_cap_seconds": SCHEDULER_POLL_CALL_CAP_SECONDS,
        "scheduler_poll_call_timeout_policy": "min(per_call_cap,remaining_monotonic_budget)",
        "scheduler_timeout_exception": "SchedulerCommandTimeout",
        "all_scheduler_subprocesses_finite_timeout": True,
        "held_chain_exact_key_sets_and_types": True,
        "held_chain_bool_int_aliases_rejected": True,
        "held_chain_extra_keys_rejected": True,
        "comment_update_exit_status_exact_int": True,
        "submitted_during_preparation": False,
    }
    require_exact_json_record(
        authorization, expected, "LAUNCH_AUTHORIZATION.json"
    )
    return authorization, sha256(canonical_json_bytes(authorization)).hexdigest()


def validate_control_directory_empty_for_submission() -> None:
    info = os.stat(CONTROL, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("control directory must be owner-only mode 0700")
    if any(CONTROL.iterdir()):
        raise RuntimeError("control directory is not unused; authorization is one-submission only")


def decode_exact_raw(receipt: dict[str, Any], key: str) -> bytes:
    encoded = receipt.get(key)
    if not isinstance(encoded, str):
        raise RuntimeError(f"receipt lacks {key}")
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise RuntimeError(f"receipt {key} is not canonical base64") from error
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise RuntimeError(f"receipt {key} is not canonical base64")
    return decoded


def validate_submission_intent(authorization: dict[str, Any], authorization_digest: str) -> str:
    require_sha256(authorization_digest, "authorization digest")
    path = CONTROL / "SUBMISSION_INTENT.json"
    intent = load_canonical_json(path)
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_intent.v24",
        "authorization_sha256": authorization_digest,
        "preparation_template_sha256": authorization["preparation_template_sha256"],
        "preparation_sha256": authorization["preparation_sha256"],
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "wrapper_sha256": authorization["wrapper_sha256"],
        "sbatch_argv": authorization["approved_sbatch_argv"],
        "fixed_initial_scheduler_comment": FIXED_AUTH_COMMENT,
        "fixed_scheduler_user": FIXED_SCHEDULER_USER,
        "fixed_job_name": FIXED_JOB_NAME,
        "intent_created_at": intent.get("intent_created_at"),
        "not_before_submit_time": intent.get("not_before_submit_time"),
        "held_submission_only": True,
        "one_use_fail_closed": True,
    }
    require_exact_json_record(intent, expected, "submission intent")
    created = intent.get("intent_created_at")
    not_before = intent.get("not_before_submit_time")
    if (
        type(created) is not str
        or type(not_before) is not str
    ):
        raise RuntimeError("submission intent timestamps are invalid")
    if _parse_submit_time(not_before, "intent not-before time") <= _parse_submit_time(
        created, "intent creation time"
    ):
        raise RuntimeError("submission intent not-before time is not later than creation")
    return file_sha256(path)


def validate_submission_timeout(
    authorization: dict[str, Any],
    authorization_digest: str,
    intent_digest: str,
) -> tuple[dict[str, Any], str]:
    path = CONTROL / "SUBMISSION_TIMEOUT.json"
    payload = load_canonical_json(path)
    stdout = decode_exact_raw(payload, "sbatch_partial_stdout_b64")
    stderr = decode_exact_raw(payload, "sbatch_partial_stderr_b64")
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_timeout.v24",
        "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
        "authorization_sha256": authorization_digest,
        "submission_intent_sha256": intent_digest,
        "sbatch_argv": authorization["approved_sbatch_argv"],
        "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
        "sbatch_partial_stdout_b64": payload.get("sbatch_partial_stdout_b64"),
        "sbatch_partial_stdout_sha256": sha256(stdout).hexdigest(),
        "sbatch_partial_stderr_b64": payload.get("sbatch_partial_stderr_b64"),
        "sbatch_partial_stderr_sha256": sha256(stderr).hexdigest(),
        "scheduler_side_effect": "UNKNOWN",
        "resubmission_permitted": False,
    }
    require_exact_json_record(payload, expected, "submission timeout")
    return payload, file_sha256(path)


def _validate_submission_query_snapshot(
    snapshot: dict[str, Any], sequence: int, not_before_submit_time: str
) -> list[str]:
    raw = decode_exact_raw(snapshot, "squeue_stdout_b64")
    err = decode_exact_raw(snapshot, "squeue_stderr_b64")
    ids = parse_squeue_job_ids(raw)
    candidates = snapshot.get("candidate_records")
    if type(candidates) is not list or len(candidates) != len(ids):
        raise RuntimeError("submission reconciliation candidate inventory mismatch")
    exact_ids: list[str] = []
    for candidate, candidate_id in zip(candidates, ids, strict=True):
        if type(candidate) is not dict:
            raise RuntimeError("submission candidate record is not an object")
        raw_job = decode_exact_raw(candidate, "scontrol_stdout_b64")
        fields = parse_scontrol_one_line(
            normalize_scontrol_one_line(raw_job),
            allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
        )
        matched, mismatches = exact_held_submission_candidate(
            fields, candidate_id, not_before_submit_time
        )
        expected_candidate = {
            "job_id": candidate_id,
            "scontrol_argv": ["/usr/bin/scontrol", "show", "job", candidate_id, "-o"],
            "scontrol_stdout_b64": candidate.get("scontrol_stdout_b64"),
            "scontrol_stdout_sha256": sha256(raw_job).hexdigest(),
            "exact_match": matched,
            "mismatch_fields": mismatches,
        }
        require_exact_json_record(candidate, expected_candidate, "submission candidate record")
        if matched:
            exact_ids.append(candidate_id)
    expected_snapshot = {
        "sequence": sequence,
        "query_status": "COMPLETED",
        "squeue_argv": submission_candidate_query_argv(),
        "squeue_returncode": 0,
        "squeue_stdout_b64": snapshot.get("squeue_stdout_b64"),
        "squeue_stdout_sha256": sha256(raw).hexdigest(),
        "squeue_stderr_b64": snapshot.get("squeue_stderr_b64"),
        "squeue_stderr_sha256": sha256(err).hexdigest(),
        "candidate_job_ids": ids,
        "candidate_records": candidates,
        "exact_matching_job_ids": exact_ids,
    }
    require_exact_json_record(snapshot, expected_snapshot, "submission reconciliation snapshot")
    if err:
        raise RuntimeError("submission candidate query emitted stderr")
    return exact_ids


def validate_submission_reconciliation(
    authorization: dict[str, Any],
    authorization_digest: str,
    intent_digest: str,
    timeout_digest: str,
) -> tuple[dict[str, Any], str]:
    path = CONTROL / "SUBMISSION_RECONCILIATION.json"
    payload = load_canonical_json(path)
    intent = load_canonical_json(CONTROL / "SUBMISSION_INTENT.json")
    snapshots = payload.get("query_snapshots")
    if type(snapshots) is not list or not snapshots:
        raise RuntimeError("submission reconciliation lacks bounded raw query snapshots")
    histories: list[list[str]] = []
    for sequence, snapshot in enumerate(snapshots, 1):
        if type(snapshot) is not dict:
            raise RuntimeError("submission reconciliation snapshot is not an object")
        histories.append(_validate_submission_query_snapshot(
            snapshot, sequence, intent["not_before_submit_time"]
        ))
    matched_ids = histories[-1]
    outcome = payload.get("outcome")
    permitted = payload.get("receipt_permitted")
    if outcome == "UNIQUE_HELD_RECONCILED":
        valid = (
            len(histories) >= 2
            and len(matched_ids) == 1
            and histories[-2] == matched_ids
            and all(not item for item in histories[:-2])
            and permitted is True
        )
    elif outcome == "ZERO_CANDIDATES":
        valid = all(not item for item in histories) and permitted is False
    elif outcome == "MULTIPLE_CANDIDATES":
        valid = (
            len(matched_ids) > 1
            and all(len(item) <= 1 for item in histories[:-1])
            and permitted is False
        )
    elif outcome == "UNSTABLE_CANDIDATES":
        valid = (
            any(item for item in histories)
            and all(len(item) <= 1 for item in histories)
            and not (
                len(histories) >= 2
                and len(matched_ids) == 1
                and histories[-2] == matched_ids
            )
            and permitted is False
        )
    else:
        valid = False
    if not valid:
        raise RuntimeError("submission reconciliation outcome promotes ambiguous authority")
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_reconciliation.v24",
        "authorization_sha256": authorization_digest,
        "submission_intent_sha256": intent_digest,
        "submission_timeout_sha256": timeout_digest,
        "candidate_query_argv": submission_candidate_query_argv(),
        "query_snapshots": snapshots,
        "outcome": outcome,
        "matched_job_ids": matched_ids,
        "receipt_permitted": permitted,
        "resubmission_permitted": False,
    }
    require_exact_json_record(payload, expected, "submission reconciliation")
    return payload, file_sha256(path)


def validate_submission_attempt(
    authorization: dict[str, Any],
    authorization_digest: str,
    intent_digest: str,
    *,
    require_scheduler_success: bool,
) -> tuple[dict[str, Any], str, bytes, bytes]:
    require_sha256(authorization_digest, "authorization digest")
    require_sha256(intent_digest, "submission intent digest")
    path = CONTROL / "SUBMISSION_ATTEMPT.json"
    attempt = load_canonical_json(path)
    stdout = decode_exact_raw(attempt, "sbatch_stdout_b64")
    stderr = decode_exact_raw(attempt, "sbatch_stderr_b64")
    outcome = attempt.get("outcome")
    returncode = attempt.get("sbatch_returncode")
    timeout_digest = attempt.get("submission_timeout_sha256")
    reconciliation_digest = attempt.get("submission_reconciliation_sha256")
    reconciled_job_id = attempt.get("reconciled_job_id")
    if outcome == "COMMAND_COMPLETED":
        scheduler_command_succeeded = (
            type(returncode) is int and returncode == 0 and not stderr
        )
        scheduler_acceptance_verified = scheduler_command_succeeded
        valid_variant = (
            type(returncode) is int
            and timeout_digest is None
            and reconciliation_digest is None
            and reconciled_job_id is None
        )
    elif outcome == "TIMEOUT_RECONCILED_UNIQUE_HELD":
        timeout_payload, actual_timeout_digest = validate_submission_timeout(
            authorization, authorization_digest, intent_digest
        )
        reconciliation, actual_reconciliation_digest = validate_submission_reconciliation(
            authorization, authorization_digest, intent_digest, actual_timeout_digest
        )
        scheduler_command_succeeded = False
        scheduler_acceptance_verified = True
        valid_variant = (
            returncode is None
            and timeout_digest == actual_timeout_digest
            and reconciliation_digest == actual_reconciliation_digest
            and reconciliation.get("outcome") == "UNIQUE_HELD_RECONCILED"
            and reconciliation.get("matched_job_ids") == [reconciled_job_id]
            and stdout == decode_exact_raw(timeout_payload, "sbatch_partial_stdout_b64")
            and stderr == decode_exact_raw(timeout_payload, "sbatch_partial_stderr_b64")
        )
    else:
        scheduler_command_succeeded = False
        scheduler_acceptance_verified = False
        valid_variant = False
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_attempt.v24",
        "authorization_sha256": authorization_digest,
        "preparation_template_sha256": authorization["preparation_template_sha256"],
        "preparation_sha256": authorization["preparation_sha256"],
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "wrapper_sha256": authorization["wrapper_sha256"],
        "submission_intent_sha256": intent_digest,
        "sbatch_argv": authorization["approved_sbatch_argv"],
        "outcome": outcome,
        "sbatch_returncode": returncode,
        "sbatch_stdout_b64": attempt.get("sbatch_stdout_b64"),
        "sbatch_stdout_sha256": sha256(stdout).hexdigest(),
        "sbatch_stderr_b64": attempt.get("sbatch_stderr_b64"),
        "sbatch_stderr_sha256": sha256(stderr).hexdigest(),
        "submission_timeout_sha256": timeout_digest,
        "submission_reconciliation_sha256": reconciliation_digest,
        "reconciled_job_id": reconciled_job_id,
        "scheduler_command_succeeded": scheduler_command_succeeded,
        "scheduler_acceptance_verified": scheduler_acceptance_verified,
    }
    require_exact_json_record(attempt, expected, "submission attempt")
    if not valid_variant:
        raise RuntimeError("submission attempt outcome variant is invalid")
    if require_scheduler_success and not scheduler_acceptance_verified:
        raise RuntimeError("submission attempt lacks verified scheduler acceptance")
    return attempt, file_sha256(path), stdout, stderr


def receipt_path_for_job(job_id: str) -> Path:
    if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
        raise RuntimeError("invalid receipt job ID")
    return CONTROL / f"SUBMISSION_RECEIPT_{job_id}.json"


def unique_submission_receipt() -> tuple[Path, dict[str, Any], str]:
    candidates = sorted(CONTROL.glob("SUBMISSION_RECEIPT_*.json"))
    if len(candidates) != 1:
        raise RuntimeError("exactly one submission receipt is required")
    path = candidates[0]
    info = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise RuntimeError("submission receipt must be an owner-only regular file")
    receipt = load_canonical_json(path)
    digest = file_sha256(path)
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str) or path != receipt_path_for_job(job_id):
        raise RuntimeError("submission receipt filename/job mismatch")
    return path, receipt, digest


def validate_submission_receipt(receipt: dict[str, Any], receipt_digest: str) -> str:
    require_sha256(receipt_digest, "submission receipt digest")
    authorization, authorization_digest = load_authorization()
    intent_digest = validate_submission_intent(authorization, authorization_digest)
    attempt, attempt_digest, sbatch_stdout, _ = validate_submission_attempt(
        authorization,
        authorization_digest,
        intent_digest,
        require_scheduler_success=True,
    )
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise RuntimeError("receipt job ID is invalid")
    raw = decode_exact_raw(receipt, "initial_scontrol_stdout_b64")
    fields = parse_scontrol_one_line(
        normalize_scontrol_one_line(raw),
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )
    fixed = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_submission_receipt.v24",
        "job_id": job_id,
        "submission_outcome": attempt["outcome"],
        "submission_attempt_sha256": attempt_digest,
        "submission_intent_sha256": intent_digest,
        "authorization_sha256": authorization_digest,
        "preparation_template_sha256": authorization["preparation_template_sha256"],
        "preparation_sha256": authorization["preparation_sha256"],
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "wrapper_sha256": authorization["wrapper_sha256"],
        "initial_scontrol_argv": ["/usr/bin/scontrol", "show", "job", job_id, "-o"],
        "initial_scontrol_stdout_b64": receipt.get("initial_scontrol_stdout_b64"),
        "initial_scontrol_stdout_sha256": sha256(raw).hexdigest(),
        "initial_scheduler_comment": FIXED_AUTH_COMMENT,
        "initial_state": "PENDING",
        "initial_hold_reason": "JobHeldUser",
        "initial_user_hold_verified": True,
    }
    require_exact_json_record(receipt, fixed, "submission receipt")
    if attempt["outcome"] == "COMMAND_COMPLETED":
        if sbatch_stdout != (job_id + "\n").encode("ascii"):
            raise RuntimeError("completed submission stdout/job ID drifted")
    elif attempt["outcome"] == "TIMEOUT_RECONCILED_UNIQUE_HELD":
        if attempt["reconciled_job_id"] != job_id:
            raise RuntimeError("reconciled submission job ID drifted")
    else:
        raise RuntimeError("receipt binds an unauthorized submission outcome")
    intent = load_canonical_json(CONTROL / "SUBMISSION_INTENT.json")
    matched, mismatches = exact_held_submission_candidate(
        fields, job_id, intent["not_before_submit_time"]
    )
    if not matched:
        raise RuntimeError(
            "initial scheduler record is not the exact held authorized job: "
            + ",".join(mismatches)
        )
    if type(receipt_digest) is not str or _HEX.fullmatch(receipt_digest) is None:
        raise RuntimeError("receipt digest is invalid")
    return job_id


def expected_receipt_comment(receipt_digest: str) -> str:
    return require_sha256(receipt_digest, "receipt digest")


def validate_comment_attestation(job_id: str, submission_digest: str) -> dict[str, Any]:
    require_job_id(job_id)
    require_sha256(submission_digest, "submission receipt digest")
    path = CONTROL / f"COMMENT_ATTESTATION_{job_id}.json"
    payload = load_canonical_json(path)
    raw = decode_exact_raw(payload, "reread_scontrol_stdout_b64")
    fields = parse_scontrol_one_line(
        normalize_scontrol_one_line(raw),
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )
    update_stdout = decode_exact_raw(payload, "comment_update_stdout_b64")
    update_stderr = decode_exact_raw(payload, "comment_update_stderr_b64")
    comment = expected_receipt_comment(submission_digest)
    update_argv = [
        "/usr/bin/scontrol", "update", f"JobId={job_id}", f"Comment={comment}"
    ]
    outcome = payload.get("comment_update_outcome")
    timeout_digest = payload.get("comment_update_timeout_sha256")
    if outcome == "COMMAND_COMPLETED":
        exit_status: int | None = 0
        valid_variant = (
            timeout_digest is None and not update_stdout and not update_stderr
        )
    elif outcome == "TIMEOUT_RECONCILED_APPLIED":
        exit_status = None
        timeout_path = CONTROL / f"COMMENT_UPDATE_TIMEOUT_{job_id}.json"
        timeout_payload = load_canonical_json(timeout_path)
        timeout_stdout = decode_exact_raw(timeout_payload, "partial_stdout_b64")
        timeout_stderr = decode_exact_raw(timeout_payload, "partial_stderr_b64")
        expected_timeout = {
            "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_update_timeout.v24",
            "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
            "job_id": job_id,
            "submission_receipt_sha256": submission_digest,
            "comment_update_argv": update_argv,
            "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
            "partial_stdout_b64": timeout_payload.get("partial_stdout_b64"),
            "partial_stdout_sha256": sha256(timeout_stdout).hexdigest(),
            "partial_stderr_b64": timeout_payload.get("partial_stderr_b64"),
            "partial_stderr_sha256": sha256(timeout_stderr).hexdigest(),
            "scheduler_side_effect": "UNKNOWN",
            "repeat_update_permitted": False,
        }
        require_exact_json_record(timeout_payload, expected_timeout, "comment update timeout")
        valid_variant = (
            timeout_digest == file_sha256(timeout_path)
            and update_stdout == timeout_stdout
            and update_stderr == timeout_stderr
        )
    else:
        exit_status = payload.get("comment_update_exit_status")
        valid_variant = False
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_comment_attestation.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "comment_update_argv": update_argv,
        "comment_update_outcome": outcome,
        "comment_update_exit_status": exit_status,
        "comment_update_stdout_b64": payload.get("comment_update_stdout_b64"),
        "comment_update_stderr_b64": payload.get("comment_update_stderr_b64"),
        "comment_update_timeout_sha256": timeout_digest,
        "reread_scontrol_argv": [
            "/usr/bin/scontrol", "show", "job", job_id, "-o"
        ],
        "reread_scontrol_stdout_b64": payload.get("reread_scontrol_stdout_b64"),
        "reread_scontrol_stdout_sha256": sha256(raw).hexdigest(),
        "scheduler_comment": comment,
        "pending_user_hold_reverified": True,
        "hold_reason": "JobHeldUser",
    }
    require_exact_json_record(payload, expected, "comment attestation")
    if not valid_variant or not exact_user_held_job(fields, job_id, comment):
        raise RuntimeError("scheduler comment attestation is invalid")
    return payload


def release_intent_path(job_id: str, attempt_number: int) -> Path:
    require_job_id(job_id)
    if type(attempt_number) is not int or not (1 <= attempt_number <= MAX_RELEASE_ATTEMPTS):
        raise RuntimeError("release attempt number is invalid")
    return CONTROL / f"RELEASE_INTENT_{job_id}_{attempt_number}.json"


def release_timeout_path(job_id: str, attempt_number: int) -> Path:
    release_intent_path(job_id, attempt_number)
    return CONTROL / f"RELEASE_TIMEOUT_{job_id}_{attempt_number}.json"


def release_reconciliation_path(job_id: str, attempt_number: int) -> Path:
    release_intent_path(job_id, attempt_number)
    return CONTROL / f"RELEASE_RECONCILIATION_{job_id}_{attempt_number}.json"


def release_action_path(job_id: str, attempt_number: int) -> Path:
    release_intent_path(job_id, attempt_number)
    return CONTROL / f"RELEASE_ACTION_{job_id}_{attempt_number}.json"


def validate_release_intent(
    job_id: str, submission_digest: str, attempt_number: int = 1
) -> dict[str, Any]:
    require_job_id(job_id)
    require_sha256(submission_digest, "submission receipt digest")
    path = release_intent_path(job_id, attempt_number)
    intent = load_canonical_json(path)
    pre_raw = decode_exact_raw(intent, "pre_release_scontrol_stdout_b64")
    pre = parse_scontrol_one_line(
        normalize_scontrol_one_line(pre_raw),
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )
    expected_comment = expected_receipt_comment(submission_digest)
    prior_digest = intent.get("prior_not_applied_action_sha256")
    expected_prior: str | None = None
    retry_precheck_argv: list[str] | None = None
    retry_precheck_b64: str | None = None
    retry_precheck_sha256: str | None = None
    if attempt_number > 1:
        prior_path = release_action_path(job_id, attempt_number - 1)
        prior = load_canonical_json(prior_path)
        _validate_release_action(
            prior, job_id, submission_digest, attempt_number - 1,
            require_applied=False,
        )
        if prior.get("outcome") != "TIMEOUT_RECONCILED_NOT_APPLIED":
            raise RuntimeError("release retry lacks a proven not-applied predecessor")
        expected_prior = file_sha256(prior_path)
        retry_precheck_argv = [
            "/usr/bin/scontrol", "show", "job", job_id, "-o"
        ]
        retry_precheck_b64 = intent.get("retry_precheck_scontrol_stdout_b64")
        if type(retry_precheck_b64) is not str:
            raise RuntimeError("release retry lacks its first durable held observation")
        retry_precheck_raw = decode_exact_raw(
            intent, "retry_precheck_scontrol_stdout_b64"
        )
        retry_precheck_sha256 = sha256(retry_precheck_raw).hexdigest()
        retry_precheck = parse_scontrol_one_line(
            normalize_scontrol_one_line(retry_precheck_raw),
            allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
        )
        if not exact_user_held_job(
            retry_precheck, job_id, expected_comment
        ):
            raise RuntimeError("release retry first held observation is invalid")
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_intent.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "prior_not_applied_action_sha256": expected_prior,
        "retry_precheck_scontrol_argv": retry_precheck_argv,
        "retry_precheck_scontrol_stdout_b64": retry_precheck_b64,
        "retry_precheck_scontrol_sha256": retry_precheck_sha256,
        "release_argv": ["/usr/bin/scontrol", "release", job_id],
        "pre_release_scontrol_argv": ["/usr/bin/scontrol", "show", "job", job_id, "-o"],
        "pre_release_scontrol_stdout_b64": intent.get("pre_release_scontrol_stdout_b64"),
        "pre_release_scontrol_sha256": sha256(pre_raw).hexdigest(),
        "pending_user_hold_verified": True,
        "hold_reason": "JobHeldUser",
        "deliberate_retry": attempt_number > 1,
    }
    require_exact_json_record(intent, expected, "release intent")
    if prior_digest != expected_prior or not exact_user_held_job(
        pre, job_id, expected_comment
    ):
        raise RuntimeError("release intent is not bound to an exact still-held job")
    return intent


def validate_release_timeout(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    intent_digest: str,
) -> tuple[dict[str, Any], str]:
    require_sha256(intent_digest, "release intent digest")
    path = release_timeout_path(job_id, attempt_number)
    payload = load_canonical_json(path)
    stdout = decode_exact_raw(payload, "release_partial_stdout_b64")
    stderr = decode_exact_raw(payload, "release_partial_stderr_b64")
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_timeout.v24",
        "record_class": "AMBIGUOUS_SCHEDULER_SIDE_EFFECT",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "release_intent_sha256": intent_digest,
        "release_argv": ["/usr/bin/scontrol", "release", job_id],
        "timeout_seconds": SCHEDULER_CALL_TIMEOUT_SECONDS,
        "release_partial_stdout_b64": payload.get("release_partial_stdout_b64"),
        "release_partial_stdout_sha256": sha256(stdout).hexdigest(),
        "release_partial_stderr_b64": payload.get("release_partial_stderr_b64"),
        "release_partial_stderr_sha256": sha256(stderr).hexdigest(),
        "scheduler_side_effect": "UNKNOWN",
        "retry_permitted_before_reconciliation": False,
    }
    require_exact_json_record(payload, expected, "release timeout")
    return payload, file_sha256(path)


def _validate_release_reconciliation_snapshot(
    snapshot: dict[str, Any], sequence: int, job_id: str, comment: str
) -> str:
    status = snapshot.get("query_status")
    raw = decode_exact_raw(snapshot, "scontrol_stdout_b64")
    stderr = decode_exact_raw(snapshot, "scontrol_stderr_b64")
    if status == "COMPLETED":
        fields = parse_scontrol_one_line(
            normalize_scontrol_one_line(raw),
            allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
        )
        if exact_user_held_job(fields, job_id, comment):
            classification = "HELD"
        elif exact_receipt_bound_job(fields, job_id, comment) and positive_hold_absent(fields):
            classification = "HOLD_ABSENT"
        else:
            classification = "AMBIGUOUS"
        returncode: int | None = 0
        error_class: str | None = None
    elif status == "TIMEOUT":
        classification = "AMBIGUOUS"
        returncode = None
        error_class = "SchedulerCommandTimeout"
    elif status == "ERROR":
        classification = "AMBIGUOUS"
        returncode = None
        error_class = snapshot.get("error_class")
        if type(error_class) is not str or not error_class:
            raise RuntimeError("release reconciliation error class is malformed")
    else:
        raise RuntimeError("release reconciliation query status is invalid")
    expected = {
        "sequence": sequence,
        "query_status": status,
        "scontrol_argv": ["/usr/bin/scontrol", "show", "job", job_id, "-o"],
        "scontrol_returncode": returncode,
        "scontrol_stdout_b64": snapshot.get("scontrol_stdout_b64"),
        "scontrol_stdout_sha256": sha256(raw).hexdigest(),
        "scontrol_stderr_b64": snapshot.get("scontrol_stderr_b64"),
        "scontrol_stderr_sha256": sha256(stderr).hexdigest(),
        "error_class": error_class,
        "classification": classification,
    }
    require_exact_json_record(snapshot, expected, "release reconciliation snapshot")
    if status == "COMPLETED" and stderr:
        raise RuntimeError("release reconciliation query emitted stderr")
    return classification


def validate_release_reconciliation(
    job_id: str,
    submission_digest: str,
    attempt_number: int,
    intent_digest: str,
    timeout_digest: str,
) -> tuple[dict[str, Any], str]:
    path = release_reconciliation_path(job_id, attempt_number)
    payload = load_canonical_json(path)
    snapshots = payload.get("query_snapshots")
    if type(snapshots) is not list or not snapshots:
        raise RuntimeError("release reconciliation lacks durable raw query evidence")
    classifications = [
        _validate_release_reconciliation_snapshot(snapshot, sequence, job_id, expected_receipt_comment(submission_digest))
        for sequence, snapshot in enumerate(snapshots, 1)
    ]
    outcome = payload.get("outcome")
    final = classifications[-1]
    if outcome == "APPLIED":
        valid = final == "HOLD_ABSENT"
    elif outcome == "NOT_APPLIED":
        valid = final == "HELD" and all(item == "HELD" for item in classifications)
    elif outcome == "AMBIGUOUS":
        valid = final == "AMBIGUOUS"
    else:
        valid = False
    if not valid:
        raise RuntimeError("release reconciliation outcome is inconsistent")
    expected = {
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
    require_exact_json_record(payload, expected, "release reconciliation")
    return payload, file_sha256(path)


def _wait_for_sealed_control_record(
    path: Path,
    *,
    stage: str,
    deadline: float,
    poll_seconds: float,
    wait_observer: Callable[[str], None] | None,
) -> dict[str, Any]:
    while True:
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RuntimeError(f"{stage} publication is not an owner-controlled regular file")
            mode = stat.S_IMODE(info.st_mode)
            if mode == 0o400:
                return load_canonical_json(path, modes={0o400})
            if mode != 0o200:
                raise RuntimeError(f"{stage} publication has invalid mode {mode:04o}")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError(f"bounded wait expired before sealed {stage} publication")
        if wait_observer is not None:
            wait_observer(stage)
        time.sleep(min(poll_seconds, remaining))


def _validate_release_action(
    action: dict[str, Any],
    job_id: str,
    submission_digest: str,
    attempt_number: int = 1,
    *,
    require_applied: bool = True,
) -> None:
    require_job_id(job_id)
    require_sha256(submission_digest, "submission receipt digest")
    release_stdout = decode_exact_raw(action, "release_stdout_b64")
    release_stderr = decode_exact_raw(action, "release_stderr_b64")
    outcome = action.get("outcome")
    returncode = action.get("release_returncode")
    timeout_digest = action.get("release_timeout_sha256")
    reconciliation_digest = action.get("release_reconciliation_sha256")
    intent_path = release_intent_path(job_id, attempt_number)
    intent_digest = file_sha256(intent_path)
    if outcome in {"COMMAND_COMPLETED_APPLIED", "COMMAND_COMPLETED_FAILED"}:
        applied = (
            outcome == "COMMAND_COMPLETED_APPLIED"
            and type(returncode) is int
            and returncode == 0
            and not release_stdout
            and not release_stderr
        )
        retry_permitted = False
        valid = (
            type(returncode) is int
            and timeout_digest is None
            and reconciliation_digest is None
            and (
                applied
                or (
                    outcome == "COMMAND_COMPLETED_FAILED"
                    and (returncode != 0 or bool(release_stdout) or bool(release_stderr))
                )
            )
        )
    elif outcome in {"TIMEOUT_RECONCILED_APPLIED", "TIMEOUT_RECONCILED_NOT_APPLIED"}:
        timeout_payload, actual_timeout_digest = validate_release_timeout(
            job_id, submission_digest, attempt_number, intent_digest
        )
        reconciliation, actual_reconciliation_digest = validate_release_reconciliation(
            job_id, submission_digest, attempt_number, intent_digest, actual_timeout_digest
        )
        applied = outcome == "TIMEOUT_RECONCILED_APPLIED"
        retry_permitted = (
            outcome == "TIMEOUT_RECONCILED_NOT_APPLIED"
            and attempt_number < MAX_RELEASE_ATTEMPTS
        )
        expected_reconciliation_outcome = "APPLIED" if applied else "NOT_APPLIED"
        valid = (
            returncode is None
            and timeout_digest == actual_timeout_digest
            and reconciliation_digest == actual_reconciliation_digest
            and reconciliation.get("outcome") == expected_reconciliation_outcome
            and release_stdout == decode_exact_raw(timeout_payload, "release_partial_stdout_b64")
            and release_stderr == decode_exact_raw(timeout_payload, "release_partial_stderr_b64")
        )
    else:
        applied = False
        retry_permitted = False
        valid = False
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_action.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "attempt_number": attempt_number,
        "release_intent_sha256": intent_digest,
        "release_argv": ["/usr/bin/scontrol", "release", job_id],
        "outcome": outcome,
        "release_returncode": returncode,
        "release_stdout_b64": action.get("release_stdout_b64"),
        "release_stdout_sha256": sha256(release_stdout).hexdigest(),
        "release_stderr_b64": action.get("release_stderr_b64"),
        "release_stderr_sha256": sha256(release_stderr).hexdigest(),
        "release_timeout_sha256": timeout_digest,
        "release_reconciliation_sha256": reconciliation_digest,
        "scheduler_effect_applied": applied,
        "retry_permitted": retry_permitted,
    }
    require_exact_json_record(action, expected, "release action")
    if not valid:
        raise RuntimeError("release action variant is invalid")
    if require_applied and not applied:
        raise RuntimeError("release action does not prove the scheduler effect applied")


def positive_hold_absent(fields: dict[str, str]) -> bool:
    state = fields.get("JobState")
    if state in {"RUNNING", "CONFIGURING"}:
        return True
    reason = fields.get("Reason")
    priority = fields.get("Priority")
    return (
        state == "PENDING"
        and isinstance(reason, str)
        and _PENDING_REASON.fullmatch(reason) is not None
        and reason not in {"JobHeldUser", "None"}
        and isinstance(priority, str)
        and _POSITIVE_PRIORITY.fullmatch(priority) is not None
    )


def _validate_release_verification(
    verification: dict[str, Any],
    action_digest: str,
    job_id: str,
    submission_digest: str,
) -> None:
    require_job_id(job_id)
    require_sha256(action_digest, "release action digest")
    require_sha256(submission_digest, "submission receipt digest")
    attempt_number = verification.get("action_attempt_number")
    if type(attempt_number) is not int:
        raise RuntimeError("release verification action attempt is malformed")
    action_path = release_action_path(job_id, attempt_number)
    action = load_canonical_json(action_path)
    _validate_release_action(
        action, job_id, submission_digest, attempt_number, require_applied=True
    )
    if file_sha256(action_path) != action_digest:
        raise RuntimeError("release verification action digest drifted")
    post_raw = decode_exact_raw(verification, "post_release_scontrol_stdout_b64")
    post = parse_scontrol_one_line(
        normalize_scontrol_one_line(post_raw),
        allow_empty_fields=SCONTROL_JOB_OPTIONAL_EMPTY_FIELDS,
    )
    expected_comment = expected_receipt_comment(submission_digest)
    poll_attempts = verification.get("post_release_poll_attempts")
    recovered = verification.get("recovered_after_prior_post_query_incomplete")
    verified = (
        exact_receipt_bound_job(post, job_id, expected_comment)
        and positive_hold_absent(post)
        and type(poll_attempts) is int
        and poll_attempts >= 1
        and type(recovered) is bool
    )
    expected = {
        "schema": "mean_field.vituri2024.fig2_q003_whole_inventory_reciprocity_comparison_release_verification.v24",
        "job_id": job_id,
        "submission_receipt_sha256": submission_digest,
        "release_action_path": action_path.name,
        "release_action_sha256": action_digest,
        "action_attempt_number": attempt_number,
        "action_outcome": action["outcome"],
        "post_release_scontrol_argv": ["/usr/bin/scontrol", "show", "job", job_id, "-o"],
        "post_release_scontrol_stdout_b64": verification.get("post_release_scontrol_stdout_b64"),
        "post_release_scontrol_sha256": sha256(post_raw).hexdigest(),
        "post_release_state": post.get("JobState"),
        "post_release_reason": post.get("Reason"),
        "post_release_priority": post.get("Priority"),
        "post_release_poll_attempts": poll_attempts,
        "scheduler_comment": post.get("Comment"),
        "positive_hold_absence_verified": verified,
        "timeout_reconciled": action["outcome"] == "TIMEOUT_RECONCILED_APPLIED",
        "deliberate_retry_used": attempt_number > 1,
        "recovered_after_prior_post_query_incomplete": recovered,
        "release_verified": verified,
    }
    require_exact_json_record(verification, expected, "release verification")
    if not verified:
        raise RuntimeError("post-release scheduler verification is invalid, held, or failed")


def validate_release_chain(
    job_id: str,
    submission_digest: str,
    *,
    wait_seconds: float = RELEASE_RECORD_WAIT_SECONDS,
    poll_seconds: float = RELEASE_RECORD_POLL_SECONDS,
    wait_observer: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require one applied typed action and a separate positive verification."""
    if wait_seconds <= 0.0 or poll_seconds <= 0.0:
        raise RuntimeError("release record wait bounds must be positive")
    deadline = time.monotonic() + wait_seconds
    verification_path = CONTROL / f"RELEASE_VERIFICATION_{job_id}.json"
    verification = _wait_for_sealed_control_record(
        verification_path,
        stage="RELEASE_VERIFICATION",
        deadline=deadline,
        poll_seconds=poll_seconds,
        wait_observer=wait_observer,
    )
    attempt_number = verification.get("action_attempt_number")
    if type(attempt_number) is not int or not (1 <= attempt_number <= MAX_RELEASE_ATTEMPTS):
        raise RuntimeError("release verification references an invalid attempt")
    for number in range(1, attempt_number + 1):
        validate_release_intent(job_id, submission_digest, number)
        action = load_canonical_json(release_action_path(job_id, number))
        _validate_release_action(
            action,
            job_id,
            submission_digest,
            number,
            require_applied=number == attempt_number,
        )
        if number < attempt_number and action.get("outcome") != "TIMEOUT_RECONCILED_NOT_APPLIED":
            raise RuntimeError("release retry lineage contains a non-retryable action")
    final_path = release_action_path(job_id, attempt_number)
    final_action = load_canonical_json(final_path)
    action_digest = file_sha256(final_path)
    _validate_release_verification(
        verification, action_digest, job_id, submission_digest
    )
    return final_action, verification


def verify_runtime_launch_chain(
    job_id: str,
    current_fields: dict[str, str],
    *,
    release_wait_seconds: float = RELEASE_RECORD_WAIT_SECONDS,
    release_poll_seconds: float = RELEASE_RECORD_POLL_SECONDS,
    release_wait_observer: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    path, receipt, digest = unique_submission_receipt()
    receipt_job = validate_submission_receipt(receipt, digest)
    if receipt_job != job_id:
        raise RuntimeError("runtime job was not launched through the held-submission shim")
    validate_comment_attestation(job_id, digest)
    action, verification = validate_release_chain(
        job_id,
        digest,
        wait_seconds=release_wait_seconds,
        poll_seconds=release_poll_seconds,
        wait_observer=release_wait_observer,
    )
    if not exact_receipt_bound_job(current_fields, job_id, expected_receipt_comment(digest)):
        raise RuntimeError("live scheduler identity/comment/command does not bind the durable receipt")
    action_path = release_action_path(job_id, verification["action_attempt_number"])
    return {
        "submission_receipt_path": path.relative_to(ROOT).as_posix(),
        "submission_receipt_sha256": digest,
        "submission_outcome": receipt["submission_outcome"],
        "scheduler_comment": expected_receipt_comment(digest),
        "release_action_path": action_path.relative_to(ROOT).as_posix(),
        "release_action_sha256": file_sha256(action_path),
        "release_verification_sha256": file_sha256(CONTROL / f"RELEASE_VERIFICATION_{job_id}.json"),
        "release_argv": action["release_argv"],
        "release_action_outcome": action["outcome"],
        "release_attempt_number": verification["action_attempt_number"],
        "post_release_state": verification["post_release_state"],
        "release_record_wait_seconds": release_wait_seconds,
        "direct_sbatch_rejected": True,
    }

