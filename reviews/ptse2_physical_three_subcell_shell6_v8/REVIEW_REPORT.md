# External review report — PtSe2 physical three-subcell shell-6 v8

- **Review ID:** `cb992c27-3649-4150-9a74-80e6c788dc54`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T03:17:21Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v8`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v8`

## Decision

`approved_for_one_v8_receipt_bound_release_verification_wait_chain`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below and to the digest-bound v8 release lifecycle encoded by the reviewed immutable capsule. It does not approve changed bytes, argument-bearing invocation, direct production entry points, manual `sbatch`/`scontrol`, bypassing release verification or the oracle-completion gate, scientific output, convergence, a global-ground-state claim, or the statement “shell6 sufficient under declared thresholds.” That scientific statement remains contingent on unexecuted production gates.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `c9285181bafbcbac71a357ede220b1d3895cfb5b3b0aa8dd84a769b6fa7a9cb8` |
| source frozen object | `db53e61dd5a97450fa7b4d10027e07f43911277b0199bf24bb507ba905f32f95` |
| source checksum manifest | `4c5c688af9104e496633d65242df87627373917c544eb6df1d001fd1a9542354` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e134bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| historical-input checksum manifest | `cbc5df08564020b2c02546aba1d71ee01de5eece466ca2b3edbc20d0bdb2567b` |
| static checks artifact | `f09e327c28db5f2f76b6c424de16201332fa21dde178e3c9ef124a0971f52a7c` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `90ae77354891860adf432dffc6e6575a6bc42680d66ee75e518b8d20f1f8b32a` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v8` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v8/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON separately hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **Runner/helper interleaving is repaired within the reviewed contract.** The runner enters `validate_runner_chain()` before numerical-runtime attestation and bounded-waits on a monotonic 75-second budget for both the receipt-bound final applied `RELEASE_ACTION.<n>.json` and `RELEASE_VERIFICATION.json`. The helper uses a separate 60-second monotonic verification budget and requires two stable receipt-bound positive snapshots. The static interleaving mock starts the runner before verification publication, observes it remain waiting, then publishes verification through the helper and requires runner success. `RUNNING` and `CONFIGURING` are accepted only after exact scheduler-field and receipt/job validation.
2. **Timeout, terminal failure, ambiguity, and mismatch remain fail-closed.** Missing verification exhausts the runner budget with `TimeoutError`; a receipt-bound failed/nonzero terminal preflight cannot mint positive verification; immutable ambiguous release actions are rejected; and a mismatched verification job binding is rejected. The final runner validation revalidates the full action/intent chain, action hash, role, job ID, positive flag, scheduler fields, and applied state before numerical imports. These are scheduler-free mocks, not live Slurm observations.
3. **The pinned v7 incident is exact negative evidence.** The v8 historical manifest pins 35 v7 files: five capsule identities, the detached v7 report/approval, incident report, 25 runtime-control records, and two logs. The pinned error names the missing `oracle.RELEASE_VERIFICATION.json`; no final verification is present. The incident summary classifies oracle 511729 as failed preflight with zero elapsed on node017 and projection 511730 as cancelled while held with zero elapsed/no node. There is no independent raw v7 `sacct` capture, so approval relies only on the pinned incident-report accounting summary and does not promote it to scientific evidence.
4. **No science drift was found from v7.** Ten scientific/runtime/input files are byte-identical; all enumerated pre-existing non-operational formula/data-path functions are AST-identical; and the scientific configuration payload is identical after removing versioned control-plane keys. Manual v7→v8 inspection found the excluded production-function changes limited to version/schema/bootstrap/source-closure/publication lineage updates; `preflight.py` and `secure_io.py` differ only in v8 labels. This is static byte/AST and source-diff evidence, not numerical equivalence.
5. All 27 frozen-source rows, 14 scientific-input rows, and 64 historical-input rows passed strict SHA256 verification. The capsule root/frozen directories are `0555`, frozen files are non-writable, mutable `runtime/`, `runtime/control/`, `runtime/output/`, and `logs/` are `0700`, and the mutable trees contained no files. Both wrappers passed `bash -n`.
6. Fresh scheduler-free execution of `static_check.py`, `pure_stdlib_tests.py`, and `bootstrap.py --verify-only` passed. The first two outputs exactly matched frozen `STATIC_CHECKS.json`; bootstrap reported 27 source files, 14 scientific inputs, and `writes=false`. Frozen status remained `passed_static_mock_not_reviewed_not_submitted` because that artifact records pre-review static state.
7. The initial workflow is exact and no-argument. It enters the digest-bound bootstrap submission action and can only submit the oracle and projection held, with immutable requested `afterok:<oracle_job_id>` lineage. It performs no release. Subsequent releases remain separate no-argument, digest-loaded helpers and projection release remains oracle-completion/dependency gated.

No P0/P1 blocker was found within this static control-plane and lineage review scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v8/scheduler_workflow.py
```

The script itself enforces `len(sys.argv) == 1`. Any later release must use the reviewed no-argument v8 release helpers and remains subject to their exact immutable receipt, completion, dependency, action, and positive-verification gates. This approval does not authorize repeated initial submissions.

## Review limitations

No scheduler command or query was made. No NumPy, SciPy, mpi4py, OpenMX, MPI projection, HF solve, production publication, or other scientific numerical workload was run. The v8 race repair is established by source inspection and scheduler-free mocks only. Live startup timing, runtime packages, the consumed/running `Dependency=(null)` forms, physical gates, and all scientific conclusions remain unverified by this review.
