# Detached postflight review — PtSe2 shell-6 v26 mode recovery

## Decision

**PASS, bounded to zero-science recovery authority.** Independently audited all **65** regular files and 12 directories in `results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v26_mode_recovery` without a scheduler query, numerical import, NPZ deserialization, or mutation of v25/v26.

- All 65 files are mode `0444`; all 12 directories are mode `0555`.
- All 65 v26 file inodes are unique with link count one.
- The manifest covers 63 pre-terminal files; `RECOVERY_MANIFEST.json` and `RECOVERY_COMPLETE` are the two terminal closure files.
- All 54 copied files are byte-identical to their recorded origins and have distinct `(device,inode)` identities. This includes all nine v25 `job_513422` final payload files, the four diagnostic checkpoint files, and copied control/log/postflight/review/source-closure evidence.
- The nine v25 final payloads retain their historical modes (`0400` except `COMPLETE` at `0444`), while their v26 copies are `0444`.

## No-v25-mutation evidence

The current v25 final directory plus nine files exactly match the pre-existing, pre-v26 `FILESYSTEM_IDENTITY.txt` baseline in inode, link count, size, mode, mtime, and ctime. Their bytes also match the prior hash-bound postflight checksums and their v26 copies. This supports **no v25 mutation**; v26 uses copies, not hardlinks.

## Sentinel-last and closure

`RECOVERY_COMPLETE` binds manifest `84fafdabf1fe20fe832f4c8f39a8b83aa9e5f68a485376a999fb64580605e578`, recovery summary, source freeze, producer completion, zero-science, and no-scheduler fields. The frozen finalizer source creates `RECOVERY_MANIFEST.json`, then creates `RECOVERY_COMPLETE` at its unique `# FINAL namespace create` marker; after that marker it performs no write/copy/mkdir/ensure-dir operation, only mode sealing, fsync, and reporting. Final filesystem timestamps alone are not treated as proof of ordering.

`RECOVERY_SUMMARY.json`, `RECOVERY_CONTRACT.json`, `SOURCE_FROZEN.json`, `SOURCE_SHA256SUMS.txt`, and `STATIC_CHECKS.json` are mutually consistent with the terminal sentinel. Checkpoint material remains diagnostic-only (`physical_authority=false`) and outside the canonical final payload.

## Scheduler and prior postflight evidence

No live scheduler command was run. The copied, hash-identical evidence records producer job `513422` and its batch/MPI steps as `COMPLETED`, `0:0`; the copied postflight accounting records job `513790` and its batch step as `COMPLETED`, `0:0`. `ATTESTED_CHAIN.json` binds projection job `513422`. The entire prior detached postflight directory is byte-identical in v26 evidence but inode-distinct.

## Zero-science authority

The frozen recovery finalizer/verifier pass an independent AST allowlist: stdlib-only imports, no dynamic execution, and no process/scheduler-capable calls. The recovery only hashes/copies bytes and never deserializes NPZ. Therefore v26 adds **no scientific result or successor authority**: it repairs publication modes and preserves the v25 finite-R6, Gz=0, fixed-rank local-branch interpretation and limitations.

## Important counting distinction

“All 65 verified” does **not** mean all 65 have v25 payload counterparts. Exactly 54 are origin-backed byte copies; 11 are v26-generated closure/source files, including the manifest and recovery sentinel. All 65 were independently hashed, mode/inode checked, and closure-verified.

Detailed per-file evidence is in `AUDIT_RESULT.json`; the separately invoked shipped verifier result is in `RECOVERY_VERIFIER_RESULT.json`.
