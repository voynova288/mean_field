# Static review — PtSe2 shell-8 incremental recovery V7R5

## Decision

**PASS_STATIC_NON_AUTHORIZING_TERMINAL_ACCOUNTING_GAP.**

The capsule is frozen at `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell8_incremental_v7r5`. It reuses the immutable job-519573 V7R3 306-channel oracle in place and starts at projection. Static validation proved the exact 64-shard manifest binding, 64 OpenMX monitor receipts, producer source/runtime/control lineage, current/producer job separation, V25-style `fold_map`/optional-`fold_shifts` ABI, actual NPZ fallback necessity, and projection → merge → final-only wrapper/profile scope. No OpenMX execution, shard copy, scheduler contact, numerical import, or scientific calculation occurred.

## Blocking authority gap

No pre-existing terminal `sacct` record for job `519573` exists in the workspace, and the task prohibited scheduler contact. The runtime `scontrol` receipt is a RUNNING-state allocation record and is not terminal accounting. Therefore no `REVIEW_APPROVAL.json` was created; submission, release, execution, and scientific-result authority are all false.

## Identities

- `SOURCE_FROZEN.json`: `41ca17269dc896928e545bb9620e4e51f066f0418ec040de17fd214825ec0c3b`
- `SOURCE_SHA256SUMS.txt`: `ba8c9a3e95c3047bda37b3e483be9723ba11af96e558fe7ddc759f39b253cc88`
- `CAPSULE_SHA256SUMS.txt`: `b61f248197c09aa8db30b92bd491ef78fe1f1783d139c0027d06db79f17984e1`
- `CONFIG.json`: `a10f25d32c42d7523320b9a7bda0b14cf2e0fbf22bee27d9ef4f4392197a265a`
- `EXTERNAL_RECOVERY_INPUT.json`: `8c8a33406fbd6a922e70db8b27e8d5eb8919ad8454a78e1a5ef19645a67dd357`
- `STATIC_VALIDATION_RECEIPT.json`: `a707e15053c8a60ad8242dbfbc1251e02a313a123f31a512f037dc72ba825cc5`
