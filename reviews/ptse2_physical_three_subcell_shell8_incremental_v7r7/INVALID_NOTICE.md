# INVALID_NOTICE — V7R6 execution approval

Status: **INVALID FOR EXECUTION**

The pre-existing V7R6 approval remains immutable at `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell8_incremental_v7r6/REVIEW_APPROVAL.json` (SHA-256 `7ae6e0aaed2fd0bd11754ca9e53c4f2db3bbbe5bda6a140584e689316319e516`, review ID `2dcc3450-3c33-5a7a-8ee8-38c20d630774`). It is not mutated or deleted.

V7R6 is rejected because its strict analyze-only finalized runtime receipt intentionally omits `openmx_expected_records` and `openmx_expected_sha256`, while `runtime_preamble._validate_runtime_receipt` retained unconditional access to both stale fields. Both production analysis profiles therefore fail deterministically during receipt validation before project or numerical imports. No V7R6 scheduler, numerical, merge, final-analysis, or publication action occurred; its runtime namespaces remain empty.

V7R7 supersedes only that execution authorization. It removes the stale access, adds producer-to-real-consumer pure-stdlib tests for both exact analysis profiles, and preserves the external projection checkpoint, literal R6 schema, formulas, publication protocol, and full-node allocation.
