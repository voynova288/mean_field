# Static review — PtSe2 shell-8 incremental recovery V7R6

## Decision

**APPROVED for exactly one held V7R6 merge/final recovery submission and its separate receipt-bound release.**

This review is static only. It authorizes the frozen workflow but does not validate any future numerical result. No scheduler command/query, numerical import, projection, merge, final analysis, or scientific calculation was performed.

## Failure and recovery boundary

Detached `SACCT_519622.txt` (SHA-256 `7525456060a4af9d289272566b5c6f82a1a35b2c82342acc007ff5bd8366d12c`) records allocation and batch `FAILED|1:0`; completed steps `.0` and `.1` remain explicit. Frozen stderr identifies the later failure as `R6 NPZ exact schema/dtype/shape/finiteness mismatch`. The immutable V7R5 projection checkpoint and sentinel were already published; no merge checkpoint exists.

V7R6 binds V7R5 job `519622`, its separate checkpoint root, sentinel, manifest, coefficient NPZ, report, runtime map receipts, source/runtime records, failure logs, and terminal accounting in `EXTERNAL_PROJECTION_INPUT.json`. The future current merge job must differ from `519622`; its runtime root is V7R6. No projection checkpoint is copied into V7R6.

## Repair findings

1. The actual hash-pinned parent `matched_R6.npz` has exact ordered keys `schema`, `numerator3`, `supercell_labels`, `shell`, `axes`, `charge_fourier`, `hole_charge_fourier`, `spin_pauli_fourier`, `supercell_matrix`, `supercell_lattice_nm`, `area_nm2`.
2. Its scalar schema is literal `ptse2_n2_B1_matched_charge_spin_shell6/v7`, dtype `<U41`, shape `()`—not a V7R5 successor tag.
3. The pure-stdlib NPZ/NPY test loads the actual file and verifies every key, dtype, shape, Fortran-order flag, schema/axes value, payload length, and numeric finiteness. Production now requires the same `<U41` literal.
4. The wrapper has exactly two production actions in order: `analyze_merge` (`--publish-merge-checkpoint`) then `analyze_final` (`--analyze-from-merge-checkpoint`). Runtime profiles are exactly those two. Projection code/profile/launch and qualified projection source are absent.
5. `formula_core.py` and `secure_publication.py` are byte-identical to exact V7R5. Eighteen named formula-bearing analysis functions are AST-identical. The changed analysis functions are limited to runtime identity, R6 validator, external-checkpoint provenance, successor schemas, and publication consumers.
6. Merge and final publication retain separate sentinel-last immutable boundaries. A downstream final failure preserves the merge checkpoint; the reused projection checkpoint remains external and immutable.

## Pinned identities

- `SOURCE_FROZEN.json`: `d2c81144b2015a82c7fce288681d2aac4eef7360e8523a0fe7e2c380bf6c567c`
- `SOURCE_SHA256SUMS.txt`: `6dab162fac6a18b3788a6318e7c9fbfdfdec66f42650555f85b5ea9db6adfbb3`
- `CAPSULE_SHA256SUMS.txt`: `1d115a186125484a2ab2dbb0efb04ac5e9f033f77601d3d3239362f460225bd3`
- `CONFIG.json`: `751bfa175d4f5e290d975d53d2b4c44fa8c1a41e7afaf375facc197661f120fb`
- `EXTERNAL_PROJECTION_INPUT.json`: `505aa1d7563cb2a71e03ba857578c300ba851c30cea39e60e7ac834ca1ff00ff`
- `STATIC_VALIDATION_RECEIPT.json`: `41b6d8dc86043e6e77f0505c5c6294dc7a1bf52cb9492582cee9aa1bd15aeb0a`
- review ID: `2dcc3450-3c33-5a7a-8ee8-38c20d630774`

## Uncertainty and authority boundary

Static evidence establishes schema/provenance/control correctness, not a completed merge, plotted response, or physical result. Runtime library mappings, current-job resource evidence, actual merge/final numerical behavior, and final published payload remain unexecuted and must pass the frozen runtime and publication gates. No claim is made that R8 is converged or sufficient.
