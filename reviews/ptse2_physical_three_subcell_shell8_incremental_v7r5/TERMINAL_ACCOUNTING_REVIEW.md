# Terminal-accounting execution review — PtSe2 shell-8 recovery V7R5

## Decision

**APPROVED for exactly one held, projection-only V7R5 recovery submission and its separate receipt-bound release.**

This approval does not validate any future numerical result. It authorizes the frozen recovery workflow identified below; all runtime, projection, merge, and sentinel gates remain mandatory. No scheduler command, numerical import, or scientific calculation was performed in this review.

## Terminal producer closure

The detached accounting artifact `SACCT_519573.txt` has SHA-256 `7a3fadbf20f73b74477a2dd9e22840a29394a9ac7c48cc420ed873b0a200a9cd` and records:

- allocation `519573`: `FAILED`, exit `1:0`, 00:11:33, node038;
- batch step: `FAILED`, exit `1:0`;
- steps `.0`, `.1`, and `.2`: `COMPLETED`, exit `0:0`;
- terminal end time `2026-09-24T04:06:15`.

Together with the frozen V7R3 wrapper order, the immutable `NEW306_ORACLE_COMPLETE_519573` sentinel, exact failure stderr, and absence of a projection sentinel, this closes the producer as: oracle stage succeeded and was committed; the batch later failed in projection on missing `fold_shifts`. It does not relabel the overall job as successful.

## Audit findings

1. **External oracle lineage:** the V7R3 sentinel/manifests/inventory bind producer job 519573, 306 channels, exactly **64** shard rows (`rank00000`–`rank00063`, 24,473,720,992 bytes total), producer source/runtime/control lineage, and 64 successful rank-monitor receipts. Published shard files are mode 0400 under a mode-0500 checkpoint. Static review verified exact names/sizes and manifest equality; the production consumer rehashes the complete published bundle before projection. No 24.47 GB login-node rehash was performed.
2. **Failure boundary:** frozen stderr is exactly the root-state `KeyError: 'fold_shifts is not a file in the archive'`; the oracle completion sentinel exists and no job-519573 projection completion sentinel exists. Detached terminal accounting independently confirms terminal FAILED/1:0.
3. **V25 ABI parity:** V7R5 computes `fold_map()` before opening the root state, uses the saved `fold_shifts` if present and the computed shifts otherwise, and then requires `np.array_equal(saved_shifts, shifts)`. The actual pinned NPZ inventory lacks `fold_shifts.npy`; therefore the fallback is necessary, not dead code. This is the same fallback/equality contract used by shell6 V25.
4. **Producer/consumer separation:** projection rejects a current `SLURM_JOB_ID` equal to 519573. Projection reports/manifests/sentinels carry both the current producer identity and external oracle producer identity; merge/final consume current-job checkpoints only.
5. **Projection-only scope:** V7R5 contains no oracle prepare/finalize/monitor entrypoints and the wrapper executes only projection, merge, and final analysis after runtime gates. The exact runtime profile set is `project_mpi`, `analyze_merge`, `analyze_final`; no OpenMX process profile or shard-copy stage is present.
6. **Science/publication preservation:** `formula_core.py` and `secure_publication.py` are byte-identical to V7R3. The projection numerical path differs only for external-oracle provenance/current-job separation, the V25 fallback, and versioned output bindings; analysis changes are schema/provenance consumers. Projection and merge are independently sentinel-last immutable checkpoints, so later merge/metrics/plot failure preserves prior science. Final publication remains sentinel-last.

## Pinned identities

- `SOURCE_FROZEN.json`: `41ca17269dc896928e545bb9620e4e51f066f0418ec040de17fd214825ec0c3b`
- `SOURCE_SHA256SUMS.txt`: `ba8c9a3e95c3047bda37b3e483be9723ba11af96e558fe7ddc759f39b253cc88`
- `CAPSULE_SHA256SUMS.txt`: `b61f248197c09aa8db30b92bd491ef78fe1f1783d139c0027d06db79f17984e1`
- `CONFIG.json`: `a10f25d32c42d7523320b9a7bda0b14cf2e0fbf22bee27d9ef4f4392197a265a`
- `EXTERNAL_RECOVERY_INPUT.json`: `8c8a33406fbd6a922e70db8b27e8d5eb8919ad8454a78e1a5ef19645a67dd357`
- detached `SACCT_519573.txt`: `7a3fadbf20f73b74477a2dd9e22840a29394a9ac7c48cc420ed873b0a200a9cd`
- review ID: `1e940823-4e5f-5afc-ac5c-d20d8c602787`

## Authority boundary and commands

Authority is limited to one no-argument held submission through `scheduler_workflow.py`, followed by a separately invoked receipt-bound `release_shell8.py`. Ambiguous submission must not be retried. Exact commands are in `AUTHORIZED_COMMANDS.txt`; they were not executed during review.

The older `REVIEW_REPORT.md` and `STATIC_REVIEW.json` remain preserved as the correct pre-accounting, nonauthorizing static decision. This terminal review supersedes only their missing-accounting blocker; it does not rewrite those historical records.
