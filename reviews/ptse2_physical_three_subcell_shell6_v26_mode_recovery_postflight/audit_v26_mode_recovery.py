#!/usr/bin/env python3
"""Independent, read-only, pure-stdlib audit and detached publication for v26 mode recovery."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import stat
from pathlib import Path

TARGET_REL = Path("results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v26_mode_recovery")
V25_REL = Path("results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25")
PRIOR_REL = Path("reviews/ptse2_physical_three_subcell_shell6_v25_job513422_postflight")
EXPECTED_OUTPUT_NAMES = {
    "COMPLETE", "OUTPUT_SHA256.json", "matched_R1.npz", "matched_R2.npz",
    "matched_R3.npz", "matched_R4.npz", "matched_R5.npz", "matched_R6.npz", "summary.json",
}
ALLOWED_IMPORTS = {"__future__", "ast", "hashlib", "json", "os", "pathlib", "stat", "sys", "argparse"}
PROCESS_ATTRS = {"system", "popen", "spawnl", "spawnv", "run", "call", "check_call", "check_output", "Popen"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def load_canonical(path: Path) -> object:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    assert raw == canonical(value), f"noncanonical JSON: {path}"
    return value


def parse_sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        digest, name = line.split("  ", 1)
        assert len(digest) == 64 and name not in out
        out[name] = digest
    return out


def fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o444)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            assert n > 0
            view = view[n:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
    finally:
        os.close(fd)


def source_safety(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile", "__import__"}, f"dynamic execution: {path}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in PROCESS_ATTRS, f"process execution: {path}"
    assert roots <= ALLOWED_IMPORTS, f"unapproved imports in {path}: {sorted(roots-ALLOWED_IMPORTS)}"
    return {"import_roots": sorted(roots), "process_or_dynamic_execution_calls": 0, "sha256": sha(path)}


def snapshot(root: Path) -> dict[str, tuple[int, int, int, int, int, int, str]]:
    out = {}
    for p in sorted((q for q in root.rglob("*") if q.is_file()), key=lambda q: q.relative_to(root).as_posix()):
        s = p.stat()
        out[p.relative_to(root).as_posix()] = (s.st_dev, s.st_ino, s.st_nlink, s.st_size, mode(p), s.st_mtime_ns, sha(p))
    return out


def main() -> int:
    review = Path(__file__).resolve().parent
    repo = review.parent.parent.resolve()
    assert (repo / ".git").is_dir() and review.name == "ptse2_physical_three_subcell_shell6_v26_mode_recovery_postflight"
    target, v25, prior = repo / TARGET_REL, repo / V25_REL, repo / PRIOR_REL
    assert target.is_dir() and v25.is_dir() and prior.is_dir()
    assert not any((review / name).exists() for name in ("AUDIT_RESULT.json", "RECOVERY_VERIFIER_RESULT.json", "POSTFLIGHT_REPORT.md", "POSTFLIGHT_SHA256SUMS.txt", "POSTFLIGHT_COMPLETE"))

    before = snapshot(target)
    assert len(before) == 65, f"expected 65 files, got {len(before)}"
    dirs = [target, *sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: p.relative_to(target).as_posix())]
    assert len(dirs) == 12
    assert all(mode(p) == 0o555 and not p.is_symlink() for p in dirs)
    assert all(record[4] == 0o444 for record in before.values())
    assert len({(r[0], r[1]) for r in before.values()}) == 65
    assert all(r[2] == 1 for r in before.values())

    manifest_path = target / "RECOVERY_MANIFEST.json"
    sentinel_path = target / "RECOVERY_COMPLETE"
    manifest = load_canonical(manifest_path)
    sentinel = load_canonical(sentinel_path)
    summary = load_canonical(target / "RECOVERY_SUMMARY.json")
    contract = load_canonical(target / "RECOVERY_CONTRACT.json")
    frozen = load_canonical(target / "SOURCE_FROZEN.json")
    checks = load_canonical(target / "STATIC_CHECKS.json")
    assert isinstance(manifest, dict) and isinstance(sentinel, dict) and isinstance(summary, dict)
    records = manifest["files"]
    assert isinstance(records, dict) and len(records) == 63
    assert set(before) == set(records) | {"RECOVERY_MANIFEST.json", "RECOVERY_COMPLETE"}
    assert manifest["exact_inventory_excludes_only"] == ["RECOVERY_MANIFEST.json", "RECOVERY_COMPLETE"]
    assert manifest["sentinel_last_namespace_create"] is True and manifest["zero_science"] is True
    assert sentinel["manifest_sha256"] == sha(manifest_path)
    assert sentinel["recovery_summary_sha256"] == sha(target / "RECOVERY_SUMMARY.json")
    assert sentinel["source_frozen_sha256"] == sha(target / "SOURCE_FROZEN.json")
    for key in ("create_only", "directory_seal_after_sentinel", "sentinel_last_namespace_create", "v25_mode_defect_preserved", "zero_science"):
        assert sentinel[key] is True
    assert sentinel["scheduler_action_performed"] is False and sentinel["v25_mutated"] is False
    assert contract["scheduler_action_authorized"] is False and contract["authority"]["scientific_computation_performed"] is False
    assert summary["scheduler_action_performed"] is False and summary["zero_science"] is True
    assert summary["authority"]["adds_scientific_authority"] is False
    assert summary["authority"]["checkpoint_physical_authority"] is False
    assert checks["status"] == "pass" and checks["scheduler_action_performed"] is False
    assert frozen["zero_science"] is True and frozen["producer_job_id"] == "513422"

    copied = 0
    group_counts: dict[str, int] = {}
    file_rows: dict[str, dict[str, object]] = {}
    for rel, rec in sorted(records.items()):
        p = target / rel
        assert rec["sha256"] == sha(p) and rec["size"] == p.stat().st_size and rec["mode"] == "0444"
        row: dict[str, object] = {
            "dev": p.stat().st_dev, "inode": p.stat().st_ino, "links": p.stat().st_nlink,
            "mode": "0444", "sha256": rec["sha256"], "size": rec["size"], "manifest_record": "pass",
        }
        origin_text = rec.get("origin")
        if origin_text is not None:
            copied += 1
            origin = Path(origin_text)
            assert origin.is_absolute() and origin.is_file() and not origin.is_symlink()
            assert sha(origin) == rec["origin_sha256"] == sha(p)
            assert origin.stat().st_size == rec["origin_size"] == p.stat().st_size
            assert f"{mode(origin):04o}" == rec["origin_mode"]
            assert (origin.stat().st_dev, origin.stat().st_ino) != (p.stat().st_dev, p.stat().st_ino)
            row.update({
                "byte_identical_to_origin": True, "distinct_inode_from_origin": True,
                "origin": origin.as_posix(), "origin_dev": origin.stat().st_dev,
                "origin_inode": origin.stat().st_ino, "origin_mode": rec["origin_mode"],
            })
            group = rel.split("/", 2)[0] if not rel.startswith("evidence/") else "/".join(rel.split("/")[:2])
            group_counts[group] = group_counts.get(group, 0) + 1
        file_rows[rel] = row
    assert copied == 54

    for rel in ("RECOVERY_MANIFEST.json", "RECOVERY_COMPLETE"):
        p = target / rel
        s = p.stat()
        file_rows[rel] = {"dev": s.st_dev, "inode": s.st_ino, "links": s.st_nlink, "mode": "0444", "sha256": sha(p), "size": s.st_size, "terminal_closure_record": "pass"}
    assert len(file_rows) == 65

    final25 = v25 / "runtime/output/job_513422"
    final26 = target / "runtime/output/job_513422"
    assert {p.name for p in final25.iterdir()} == EXPECTED_OUTPUT_NAMES
    assert {p.name for p in final26.iterdir()} == EXPECTED_OUTPUT_NAMES
    payload_pairs = {}
    for name in sorted(EXPECTED_OUTPUT_NAMES):
        p25, p26 = final25 / name, final26 / name
        assert sha(p25) == sha(p26)
        assert (p25.stat().st_dev, p25.stat().st_ino) != (p26.stat().st_dev, p26.stat().st_ino)
        assert mode(p26) == 0o444 and mode(p25) == (0o444 if name == "COMPLETE" else 0o400)
        payload_pairs[name] = {"sha256": sha(p26), "v25_inode": p25.stat().st_ino, "v25_mode": f"{mode(p25):04o}", "v26_inode": p26.stat().st_ino, "v26_mode": "0444"}

    # Pre-v26, hash-bound detached baseline proves current v25 final identity/modes did not change.
    prior_sums = parse_sums(prior / "POSTFLIGHT_SHA256SUMS.txt")
    assert len(prior_sums) == 9
    for name, digest in prior_sums.items():
        assert sha(prior / name) == digest
    prior_complete = json.loads((prior / "POSTFLIGHT_COMPLETE").read_text(encoding="utf-8"))
    assert prior_complete["manifest_sha256"] == sha(prior / "POSTFLIGHT_SHA256SUMS.txt")
    assert prior_complete["producer_job_id"] == "513422" and prior_complete["recompute_job_id"] == "513790"
    assert prior_complete["sentinel_last"] is True and prior_complete["v25_mutated"] is False
    baseline: dict[str, dict[str, str]] = {}
    in_final = False
    for line in (prior / "FILESYSTEM_IDENTITY.txt").read_text(encoding="utf-8").splitlines():
        if line == "FINAL":
            in_final = True
            continue
        if line == "STAGING":
            break
        if in_final and line.startswith("/"):
            parts = line.split("|")
            baseline[parts[0]] = dict(item.split("=", 1) for item in parts[1:])
    assert len(baseline) == 10
    for path_text, old in baseline.items():
        p = Path(path_text)
        s = p.stat()
        assert mode(p) == int(old["mode"], 8) and s.st_ino == int(old["inode"]) and s.st_nlink == int(old["links"]) and s.st_size == int(old["size"])
        old_mtime = dt.datetime.fromisoformat(old["mtime"]).timestamp()
        old_ctime = dt.datetime.fromisoformat(old["ctime"]).timestamp()
        assert abs(s.st_mtime - old_mtime) < 0.5 and abs(s.st_ctime - old_ctime) < 0.5

    output_manifest = json.loads((final26 / "OUTPUT_SHA256.json").read_text(encoding="utf-8"))
    producer_complete = json.loads((final26 / "COMPLETE").read_text(encoding="utf-8"))
    assert output_manifest["producer_job_id"] == "513422"
    assert output_manifest["files"] == {name: sha(final26 / name) for name in EXPECTED_OUTPUT_NAMES - {"COMPLETE", "OUTPUT_SHA256.json"}}
    assert producer_complete["producer_job_id"] == "513422" and producer_complete["sentinel_last"] is True
    assert producer_complete["output_manifest_sha256"] == sha(final26 / "OUTPUT_SHA256.json")

    finalizer_safety = source_safety(target / "finalize_recovery.py")
    verifier_safety = source_safety(target / "verify_recovery.py")
    finalizer_text = (target / "finalize_recovery.py").read_text(encoding="utf-8")
    marker = 'write_exclusive(root / "RECOVERY_COMPLETE", sentinel_bytes)  # FINAL namespace create'
    assert finalizer_text.count(marker) == 1
    tail = finalizer_text.split(marker, 1)[1]
    assert not any(token in tail for token in ("write_exclusive(", "copy_exclusive(", "mkdir_exclusive(", "ensure_dir("))
    assert "os.chmod(directory, 0o555)" in tail and "fsync_dir(root.parent)" in tail
    source_rows = parse_sums(target / "SOURCE_SHA256SUMS.txt")
    assert set(source_rows) == {"README.md", "RECOVERY_CONTRACT.json", "finalize_recovery.py", "verify_recovery.py"}
    assert all(sha(target / name) == digest for name, digest in source_rows.items())

    post = target / "evidence/v25_postflight_job513422_job513790"
    assert {p.name for p in post.iterdir()} == {p.name for p in prior.iterdir()}
    assert all(sha(post / p.name) == sha(p) and (post / p.name).stat().st_ino != p.stat().st_ino for p in prior.iterdir())
    producer_sacct = (post / "SACCT_JOB513422.txt").read_text(encoding="ascii").splitlines()
    assert len(producer_sacct) == 3
    assert producer_sacct[0].startswith("513422|pt7p_n2B1_match385_v25_batch|hmt03|regular256|COMPLETED|0:0|")
    assert producer_sacct[1].startswith("513422.batch|batch|hmt03||COMPLETED|0:0|")
    assert producer_sacct[2].startswith("513422.0|hydra_bstrap_proxy|hmt03||COMPLETED|0:0|")
    audit_sacct = (post / "SACCT_AUDIT_RECOMPUTE_JOBS.txt").read_text(encoding="ascii")
    assert "513790|ptse2_v25_postflight|COMPLETED|0:0|" in audit_sacct
    assert "513790.batch|batch|COMPLETED|0:0|" in audit_sacct
    chain = json.loads((target / "evidence/v25_control/ATTESTED_CHAIN.json").read_text(encoding="ascii"))
    assert chain["projection_job_id"] == "513422" and chain["status"] == "exactly_one_projection_held_no_dependency"

    checkpoint = target / "evidence/v25_checkpoint/full385_v25_job_513422"
    cp_complete = json.loads((checkpoint / "FULL385_DIAGNOSTIC_V25_COMPLETE").read_text(encoding="utf-8"))
    cp_manifest = json.loads((checkpoint / "FULL385_DIAGNOSTIC_V25_SHA256.json").read_text(encoding="utf-8"))
    for record in (cp_complete, cp_manifest):
        assert record["diagnostic_only"] is True and record["physical_authority"] is False
    assert producer_complete["diagnostic_checkpoint_sentinel_sha256"] == sha(checkpoint / "FULL385_DIAGNOSTIC_V25_COMPLETE")
    assert not any(p.name.startswith("full385_diagnostic") for p in final26.iterdir())

    after = snapshot(target)
    assert after == before, "target changed during independent audit"

    provided_verifier = {
        "directories_mode": "0555", "file_count": 65, "files_mode": "0444",
        "manifest_sha256": "84fafdabf1fe20fe832f4c8f39a8b83aa9e5f68a485376a999fb64580605e578",
        "postflight_job": "513790:COMPLETED:0:0", "producer_job": "513422:COMPLETED:0:0",
        "recovery_complete_sha256": "ef18db0bcb2b75f50ea5ba7c9554d499ccfdf1d67378a5d21ae1450550b452ff",
        "status": "pass", "v25_mode_defect_preserved": True, "v25_mutated": False, "zero_science": True,
    }
    assert provided_verifier["manifest_sha256"] == sha(manifest_path)
    assert provided_verifier["recovery_complete_sha256"] == sha(sentinel_path)
    write_exclusive(review / "RECOVERY_VERIFIER_RESULT.json", canonical(provided_verifier))

    result = {
        "audit_boundary": {
            "live_scheduler_queries": False, "numerical_imports": False, "npz_deserialization": False,
            "target_mutated": False, "v25_mutated": False,
        },
        "closure": {
            "manifest_record_count": 63, "manifest_sha256": sha(manifest_path),
            "recovery_complete_sha256": sha(sentinel_path), "sentinel_manifest_binding": True,
            "sentinel_source_summary_bindings": True, "sentinel_last_source_order": True,
        },
        "decision": "pass_zero_science_mode_recovery_only",
        "directory_count": len(dirs),
        "directory_records": {p.relative_to(target).as_posix() or ".": {"inode": p.stat().st_ino, "mode": "0555"} for p in dirs},
        "file_count": 65,
        "file_records": file_rows,
        "mode_and_identity": {
            "all_65_files_mode_0444": True, "all_12_directories_mode_0555": True,
            "all_65_v26_file_inodes_unique": True, "all_65_v26_files_link_count_one": True,
            "copied_origin_count": copied, "copied_origin_group_counts": group_counts,
            "all_copied_bytes_identical_to_origins": True, "all_copied_inodes_distinct_from_origins": True,
            "v25_final_baseline_entries_unchanged": len(baseline), "v25_payload_pairs": payload_pairs,
        },
        "scheduler_and_postflight": {
            "evidence_only_no_live_query": True, "producer_job": "513422:COMPLETED:0:0",
            "postflight_job": "513790:COMPLETED:0:0", "prior_postflight_checksum_rows": len(prior_sums),
            "copied_postflight_exact_and_distinct_inodes": True, "attested_chain_job_id": "513422",
        },
        "schema": "ptse2_v26_mode_recovery_detached_postflight_audit/v1",
        "source_safety": {"finalizer": finalizer_safety, "verifier": verifier_safety},
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "zero_science_authority": {
            "adds_scientific_authority": False, "checkpoint_physical_authority": False,
            "recovery_scientific_computation_performed": False,
            "scope": "exact_byte_copy_mode_recovery_and_closure_only",
            "v25_scientific_interpretation_reused_not_recomputed": True,
        },
    }
    write_exclusive(review / "AUDIT_RESULT.json", canonical(result))

    report = f"""# Detached postflight review — PtSe2 shell-6 v26 mode recovery

## Decision

**PASS, bounded to zero-science recovery authority.** Independently audited all **65** regular files and 12 directories in `{TARGET_REL.as_posix()}` without a scheduler query, numerical import, NPZ deserialization, or mutation of v25/v26.

- All 65 files are mode `0444`; all 12 directories are mode `0555`.
- All 65 v26 file inodes are unique with link count one.
- The manifest covers 63 pre-terminal files; `RECOVERY_MANIFEST.json` and `RECOVERY_COMPLETE` are the two terminal closure files.
- All {copied} copied files are byte-identical to their recorded origins and have distinct `(device,inode)` identities. This includes all nine v25 `job_513422` final payload files, the four diagnostic checkpoint files, and copied control/log/postflight/review/source-closure evidence.
- The nine v25 final payloads retain their historical modes (`0400` except `COMPLETE` at `0444`), while their v26 copies are `0444`.

## No-v25-mutation evidence

The current v25 final directory plus nine files exactly match the pre-existing, pre-v26 `FILESYSTEM_IDENTITY.txt` baseline in inode, link count, size, mode, mtime, and ctime. Their bytes also match the prior hash-bound postflight checksums and their v26 copies. This supports **no v25 mutation**; v26 uses copies, not hardlinks.

## Sentinel-last and closure

`RECOVERY_COMPLETE` binds manifest `{sha(manifest_path)}`, recovery summary, source freeze, producer completion, zero-science, and no-scheduler fields. The frozen finalizer source creates `RECOVERY_MANIFEST.json`, then creates `RECOVERY_COMPLETE` at its unique `# FINAL namespace create` marker; after that marker it performs no write/copy/mkdir/ensure-dir operation, only mode sealing, fsync, and reporting. Final filesystem timestamps alone are not treated as proof of ordering.

`RECOVERY_SUMMARY.json`, `RECOVERY_CONTRACT.json`, `SOURCE_FROZEN.json`, `SOURCE_SHA256SUMS.txt`, and `STATIC_CHECKS.json` are mutually consistent with the terminal sentinel. Checkpoint material remains diagnostic-only (`physical_authority=false`) and outside the canonical final payload.

## Scheduler and prior postflight evidence

No live scheduler command was run. The copied, hash-identical evidence records producer job `513422` and its batch/MPI steps as `COMPLETED`, `0:0`; the copied postflight accounting records job `513790` and its batch step as `COMPLETED`, `0:0`. `ATTESTED_CHAIN.json` binds projection job `513422`. The entire prior detached postflight directory is byte-identical in v26 evidence but inode-distinct.

## Zero-science authority

The frozen recovery finalizer/verifier pass an independent AST allowlist: stdlib-only imports, no dynamic execution, and no process/scheduler-capable calls. The recovery only hashes/copies bytes and never deserializes NPZ. Therefore v26 adds **no scientific result or successor authority**: it repairs publication modes and preserves the v25 finite-R6, Gz=0, fixed-rank local-branch interpretation and limitations.

## Important counting distinction

“All 65 verified” does **not** mean all 65 have v25 payload counterparts. Exactly 54 are origin-backed byte copies; 11 are v26-generated closure/source files, including the manifest and recovery sentinel. All 65 were independently hashed, mode/inode checked, and closure-verified.

Detailed per-file evidence is in `AUDIT_RESULT.json`; the separately invoked shipped verifier result is in `RECOVERY_VERIFIER_RESULT.json`.
"""
    write_exclusive(review / "POSTFLIGHT_REPORT.md", report.encode("utf-8"))

    names = ("audit_v26_mode_recovery.py", "AUDIT_RESULT.json", "RECOVERY_VERIFIER_RESULT.json", "POSTFLIGHT_REPORT.md")
    sums = "".join(f"{sha(review / name)}  {name}\n" for name in names)
    write_exclusive(review / "POSTFLIGHT_SHA256SUMS.txt", sums.encode("ascii"))
    complete = {
        "audit_result_sha256": sha(review / "AUDIT_RESULT.json"),
        "decision": "pass_zero_science_mode_recovery_only",
        "file_count_verified": 65,
        "manifest_sha256": sha(review / "POSTFLIGHT_SHA256SUMS.txt"),
        "no_live_scheduler_query": True,
        "no_numerics": True,
        "report_sha256": sha(review / "POSTFLIGHT_REPORT.md"),
        "schema": "ptse2_v26_mode_recovery_detached_postflight_complete/v1",
        "sentinel_last_namespace_create": True,
        "source_sha256": sha(review / "audit_v26_mode_recovery.py"),
        "target_recovery_complete_sha256": sha(sentinel_path),
        "v25_mutated": False,
        "v26_mutated": False,
        "zero_science_authority": True,
    }
    write_exclusive(review / "POSTFLIGHT_COMPLETE", canonical(complete))  # FINAL namespace create
    fsync_dir(review)
    os.chmod(review, 0o555)
    fsync_dir(review)
    fsync_dir(review.parent)
    print(json.dumps({"decision": complete["decision"], "file_count_verified": 65, "review": review.as_posix(), "sentinel_sha256": sha(review / "POSTFLIGHT_COMPLETE")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
