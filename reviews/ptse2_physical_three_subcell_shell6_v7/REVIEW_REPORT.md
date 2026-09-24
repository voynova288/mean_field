# External review report — PtSe2 physical three-subcell shell-6 v7

- **Review ID:** `909e8836-cb13-472d-b119-3e426d2559b7`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T02:47:45Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v7`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v7`

## Decision

`approved_for_one_v7_exact_dependency_lifecycle_oracle_completion_gated_chain`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below and to the digest-bound v7 release lifecycle encoded by the reviewed capsule. It is not approval of changed bytes, argument-bearing invocation, direct production entry points, manual `sbatch`/`scontrol` commands, bypassing the oracle-completion gate, scientific output, convergence, a global-ground-state claim, or the statement “shell6 sufficient under declared thresholds.” That scientific statement remains contingent on the unexecuted production gates.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `9dc34a11e4f4b38fa8666460a92d0f1517752dfad4cc0ab4bae3fb1780fc5ab7` |
| source frozen object | `7ce7539671a234ea0811855ffc05807ddfd858a3009ae2c6717252ef47f3e0f8` |
| source checksum manifest | `03ac8479c91e0d9a36e628baebab8e71139eb3c9dd9bc65cd8fac94e1be31b89` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e1347bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| historical-input checksum manifest | `4de0ebb985cf3aa33114e54466ca76dbdd68d8970a988ba4160a376430200510` |
| static checks artifact | `f68dc97e1ff18f30a59dc02498576f2c6f80fcd1b7c50f8e11ee34d4880dd771` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `90ae77354891860adf432dffc6e6575a6bc42680d66ee75e518b8d20f1f8b32a` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v7` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v7/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON separately hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. All 27 frozen-source rows, 14 scientific-input rows, and 28 historical-input rows passed strict SHA256 verification. The capsule root and frozen directories are mode `0555`; frozen regular files are non-writable; mutable `runtime/`, `runtime/control/`, `runtime/output/`, and `logs/` are mode `0700` and empty. No symlink or bytecode residue was found.
2. The v6 P0 prepare-chain defect is repaired on the production path. `prepare_missing_oracle.py` now calls `validate_oracle_chain`, requires the exact nine-key v7 chain, and checks `requested_projection_dependency == afterok:<oracle_job_id>`; the nonexistent v6 `dependency` lookup is absent. The pure-stdlib test extracts and executes that actual production validator, accepts the exact chain, rejects legacy-key, extra-key, schema/status/job/hash, wrong-dependency, and suffix mutations, and statically confirms `main()` invokes it before workspace creation.
3. The held-submission lifecycle is exact: oracle `Dependency=(null)` and projection `Dependency=afterok:<oracle>(unfulfilled)`. Projection release is impossible until the oracle has a persisted `COMPLETED`, `ExitCode=0:0` query and a validated immutable `MISSING_ORACLE_COMPLETE` sentinel; projection release then requires exact consumed `Dependency=(null)`. The projection wrapper seals a pre-MPI `RUNNING`, `ExitCode=0:0`, `Dependency=(null)` receipt, and every rank validates the submission/completion/release/running lineage. Arbitrary suffixes and never-satisfied dependencies fail closed.
4. Submission and release mocks passed normal, timeout-reconciled, zero/multiple-candidate, query-failure, bounded-retry, orphaned-intent, release-before-completion, failed-oracle, never-satisfied, consumed-null, and running-null cases. All scheduler mutations remain behind digest-loaded bootstrap validation. These are mocks, not fresh scheduler observations.
5. The separately pinned `sacct` evidence has an exact three-row inventory for jobs `511413`, `511566`, and `511567`; every row reports `CANCELLED by 1091`, `Elapsed=00:00:00`, `Start=None`, and `NodeList=None assigned`. Its provenance records the exact `login002` query and no mutation. This supports terminal cancellation/zero-allocation classification only; it is historical scheduler evidence, not scientific evidence or v7 execution authority.
6. The v6→v7 no-scientific-drift gate passed. Ten scientific/runtime/input files are byte-identical; every enumerated pre-existing non-operational formula/data-path function is AST-identical; the scientific config payload is identical after removing versioned control-plane keys. The only allowed added function is the prepare-chain validator. This is static byte/AST lineage evidence, not numerical equivalence or fresh physical validation.
7. Both wrappers pass `bash -n`, retain account `hmt03`, the two-partition request, `node037` exclusion, one node, 64 tasks, `--exclusive`, `--mem=0`, and no explicit time directive. The initial exact workflow is no-argument and submits both jobs held; it does not release either job.
8. `static_check.py`, `pure_stdlib_tests.py`, and bootstrap `--verify-only` passed scheduler-free. Their JSON exactly matched the frozen `STATIC_CHECKS.json`; bootstrap reported 27 source files, 14 scientific inputs, and `writes=false`.

No P0/P1 blocker was found within this static/control-plane review scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v7/scheduler_workflow.py
```

The script enforces `len(sys.argv) == 1`, enters the digest-bound bootstrap submission action, and submits the oracle and projection held with immutable requested `afterok:<oracle_job_id>` lineage. It performs no release. Any later release must use the reviewed no-argument v7 helpers and remains fail-closed on their exact completion/dependency evidence gates. This approval does not authorize repeated initial submissions.

## Review limitations

No scheduler command or query was made during this review. No NumPy, SciPy, mpi4py, OpenMX, MPI projection, HF solve, production publication, or other scientific numerical workload was run. The consumed/running `Dependency=(null)` forms are reviewed contract/mock expectations, not a fresh live observation; any different live spelling must fail closed and be handled in a new immutable successor, not by weakening v7.
