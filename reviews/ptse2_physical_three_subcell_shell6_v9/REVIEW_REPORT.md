# External review report — PtSe2 physical three-subcell shell-6 v9 projection-only recovery

- **Review ID:** `934d6317-780d-46f1-9393-78dcc2ec8efb`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T03:59:52Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v9`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v9`

## Decision

`approved_for_one_v9_projection_only_v8_oracle_recovery_chain`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v9 projection release helper. It authorizes reuse of the immutable v8 oracle for the unchanged projection calculation; it does **not** authorize an OpenMX rerun, an oracle-generation job, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypassing the release-verification wait, or any scientific conclusion before the production gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `1349b81af874a3725f620700ac6159d81cce0522571ab997fba9f369162ef190` |
| source frozen object | `032606a6f66225e85aa0cddf0affafa384367f09a94a07cf692236d4fd71c6dd` |
| source checksum manifest | `c4b254aeb8aa4d4e79b33e1cae565a9839a42b80054e5e6a2cd3e2a33512abdd` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e1347bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `fecfb83c37f1e6d51376be4666cd1adf52b72797390a68ac4f188d1c7cf7bb50` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v9` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v9/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v8 oracle inventory is exact at review time.** I independently walked and SHA256-read all 72 files (20,634,837,311 bytes) under the v8 oracle root. The actual 73-entry tree (72 files plus `vertex/`) exactly equals `V8_ORACLE_INVENTORY.json` in path, kind, size, digest, and mode. It contains 64 mode-`0444` vertex shards, four mode-`0400` files (`OUTPUT_SHA256SUMS.txt` plus the three observed metadata files), and 68 mode-`0444` files total; the root and `vertex/` are mode `0555`. The 70 output-manifest rows exactly equal the non-control payload inventory, and every row digest equals the independently observed digest.
2. **Sentinel and v8 job lineage are coherent.** `MISSING_ORACLE_COMPLETE` is mode `0444`, SHA256 `79023a5c6cfaf54e044cd1328dc2faa162f43186ccd2e11b81431cc8852e0dcc`, is hardlinked to the retained publication staging inode, and binds producer oracle job `511739`, failed projection job `511740`, 258 channels, 64 ranks, output-manifest SHA256 `628c5e146e54afa04911cf3c10b3084b02f780d71aec454b76eb919d72f6f63f`, and v8 chain SHA256 `d27855f892a184f2e383b21ae19f5be10fe986770eb6b294a017da25b874a218`. The current mode-`0400` v8 `ATTESTED_CHAIN.json` hashes to that chain digest and names jobs 511739/511740; `projection.ORACLE_COMPLETION_RECEIPT.json` hashes to `de61fa12b443f74bf467cb3aee873d518f980c706d69ba2e794379d3b70c4409`, embeds the exact sentinel, and is hash-bound by the copied v8 projection running receipt. The pinned terminal `scontrol` record reports oracle job 511739 `COMPLETED`, `ExitCode=0:0`, 64 tasks/CPUs, one node, and `OverSubscribe=NO`. There is no independently archived raw `sacct` row; this remains an explicit authority limitation.
3. **No OpenMX rerun path is present.** v9 has no oracle wrapper/preparer/finalizer/release helper. The only batch wrapper invokes the digest-bound Python projection through MPI. The OpenMX executable appears only as a hash-pinned scientific input; no reviewed v9 production source invokes it. Thus this is projection-only recovery from v8 oracle bytes, not OpenMX regeneration.
4. **Projection science is preserved.** `SCIENTIFIC_INPUT_SHA256SUMS.txt`, `NUMERICAL_RUNTIME.json`, `contracts.py`, `inventory.py`, `secure_io.py`, and the complete qualified PtSe2 source subtree are byte-identical to v8. A complete v8→v9 config comparison found only schema, external-review/runtime paths, scheduler-control metadata, and added v8 provenance. Manual full source diff of `project_matched_mpi.py` found only bootstrap/schema/source-closure changes, replacement of the v8-local oracle path by the pinned v8 oracle path, exact observed-mode/provenance gates, and output lineage labels. The projection formulas, basis and Q inventories, Pauli convention, normalization, reverse symmetrization, partition/quadrature logic, tolerances, and scientific gates are unchanged. This is static byte/AST/source-diff evidence, not a fresh numerical-equivalence run.
5. **Held submission and release-wait semantics are fail-closed within the reviewed static contract.** The exact no-argument workflow submits only one held projection job with no dependency. The separate no-argument helper verifies the receipt-bound held state, applies one release, requires two identical positive scheduler snapshots, and writes `RELEASE_VERIFICATION.json`. A released job that starts before that file exists waits up to 75 seconds before numerical imports; release verification has a 60-second budget. Submission/release ambiguity, partial state, terminal failure, job mismatch, or missing verification prevents scientific execution. Scheduler-free mocks covered held submit→release→stable RUNNING, ambiguous submission timeout, and runner-before-verification interleaving.
6. **Resource and output isolation are exact.** The request is account `hmt03`, partitions `regular256,regular6430`, excludes `node037`, uses one node with 64 MPI tasks, `--exclusive --mem=0`, and has no explicit time request. v9 `runtime/control/`, `runtime/output/`, and `logs/` contained no files before approval. All mutable control, staging, logs, and final projection output paths are under the v9 root; the v8 oracle is read as an external immutable input. Publication uses an exact allowlist under v9 `runtime/output/job_<v9-job-id>` with a `COMPLETE` sentinel.
7. Fresh scheduler-free execution of `static_check.py`, `pure_stdlib_tests.py`, and `bootstrap.py --verify-only` passed. `bash -n` passed. All 22 frozen-source rows, 14 scientific-input rows, and 11 v8-evidence rows passed strict SHA256 verification. No scheduler command/query, NumPy/SciPy/mpi4py import, MPI projection, OpenMX execution, or scientific numerical workload was performed.

No P0/P1 blocker was found within this immutable, projection-only recovery review scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v9/scheduler_workflow.py
```

This command only submits the v9 projection in held state. After its immutable submission receipt exists, the only reviewed release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v9/release_projection.py
```

Do not replace either helper with direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, scientific gate, or output publication was executed. v8 job 511739 completion is supported by the pinned terminal `scontrol`/sentinel/log chain, not a separately archived raw `sacct` row. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection. Final scientific authority remains contingent on the exact v9 runtime gates and completed v9 output sentinel.
