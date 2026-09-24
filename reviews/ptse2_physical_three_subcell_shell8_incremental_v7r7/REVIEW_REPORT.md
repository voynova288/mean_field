# Static review — PtSe2 shell-8 incremental recovery V7R7

## Decision

**APPROVED for exactly one held V7R7 merge/final recovery submission and its separate receipt-bound release.**

This review is static only. No scheduler command/query, numerical import, projection, merge, final analysis, publication, or scientific calculation was performed. It authorizes the frozen workflow but does not validate a future numerical result.

## Rejected V7R6 and immutable history

`INVALID_NOTICE.md` binds the pre-existing V7R6 approval at `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell8_incremental_v7r6/REVIEW_APPROVAL.json` (SHA-256 `7ae6e0aaed2fd0bd11754ca9e53c4f2db3bbbe5bda6a140584e689316319e516`). That approval and all V7R6 files remain untouched. Its execution authority is invalid because the analyze-only runtime producer intentionally emitted no `openmx_expected_records`/`openmx_expected_sha256`, while the consumer unconditionally indexed those stale keys. The failure is deterministic before project/numerical imports; V7R6 runtime namespaces are empty.

## Exact repair and regression evidence

1. After V7R6/V7R7 identity normalization, the only changed `runtime_preamble.py` function is `_validate_runtime_receipt`; the stale two-field OpenMX closure access is removed.
2. `runtime_inventory.py` adds pure `finalized_runtime_receipt_value`, and `finalize_runtime_receipt` now uses it. No other normalized inventory function changes.
3. The pure-stdlib end-to-end test builds the exact finalized producer schema, passes both `analyze_merge` and `analyze_final` through the real `_validate_runtime_receipt`, and rejects a missing required field, stale top-level OpenMX fields, and stale per-profile fields.
4. The strict top-level/profile key sets remain fail closed; no OpenMX field is made optional or silently ignored.

## Preserved scientific and publication contract

- The completed external V7R5 job-519622 projection checkpoint, sentinel, manifest, coefficient NPZ, report, map receipts, failure logs, and detached accounting remain unchanged and are consumed in place.
- The actual parent R6 literal remains `ptse2_n2_B1_matched_charge_spin_shell6/v7`, dtype `<U41`, with exact key/order/dtype/shape/finiteness tests.
- `formula_core.py` and `secure_publication.py` are byte-identical to V7R6 (and V7R5); all named formula-bearing analysis functions are AST-identical.
- Merge and final remain separate current-job sentinel-last publication boundaries. Projection/reprojection and checkpoint copying remain forbidden.

## Resource classification

The request remains one `regular256` node, 64 tasks, one CPU/task, `--exclusive`, and `--mem=0`, with the same account and exclusions. This is retained as a **conservative operational allocation**, not claimed minimal. Changing it would require separate resource requalification.

## Pinned identities

- `SOURCE_FROZEN.json`: `844f1862855587eef246cf91d219ff1259033e1a0e5333973e09e161f54403a1`
- `SOURCE_SHA256SUMS.txt`: `2c1884b175b0ac671252c4c9ae941f998529083894599c44b09b688d401695c9`
- `CAPSULE_SHA256SUMS.txt`: `d2e2e419258d9ac4b0d794c0b0199db920cc99899bda2686541bc18444cbbd60`
- `CONFIG.json`: `5b50fd2612291cd43d48b7ed07c062d372b5a99abe84d0bb57b9b8797a25a22d`
- `EXTERNAL_PROJECTION_INPUT.json`: `416fece478f10c49b7c6d757f7c269428fe8221d9de57712c30c9ed3fb069172`
- `STATIC_VALIDATION_RECEIPT.json`: `90f5beb6838f7a25c6d1124a6da5801d5389ee25cc287d701d52f05bc182a297`
- review ID: `5d05a348-1ec7-5512-a304-263e69c9fe3c`

## Authority boundary

Static evidence establishes source/schema/provenance/control consistency only. Runtime mappings, current-job full-node evidence, actual merge/final behavior, and final payload remain unexecuted and must pass the frozen gates. No R8 convergence, infinite-shell, global-ground-state, or scientific-result claim is made.
