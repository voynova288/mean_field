# External review report — PtSe2 physical three-subcell shell-6 v25

- **Review ID:** `c983d0b3-3d0d-4958-a80a-fe637477dcfe`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T12:44:23Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v25`

## Decision

`approved_for_one_v25_v24_qualified_axis_major_full385_checkpoint_recovery`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separate no-argument receipt-bound release helper. It authorizes one immutable v25 full-385 projection recovery attempt using the exact v24 method-B batching helper, the v23-bounded primitive-127 replay ceiling, and diagnostic-only checkpoint publication before the late physical gates. It does not authorize OpenMX/oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, checkpoint reuse, bypass of any gate, or a physical claim before successful final sentinel-last publication.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `72938e1c7da2180d22e052db5f90d5a66c0e3ff4c2bd432b6d02bb1755c56622` |
| source frozen object | `e74702e8b51a64f7e2ec06dc9e15d88e94bb2e7a1328c164f2bc990e18fddbd5` |
| source checksum manifest | `854ed7f4a30aa0d6701fae75c4bc5480305b64f3e720052ef3a6b3fa93138f1d` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| observables-v1 evidence manifest | `6dcb40a266a51a44a6b709be6b86a008fb7b2b3b0d4cc1d8da3f2937f3bddd1c` |
| v23 failure-evidence manifest | `98c16127fd90b22b8f05ddf8f64fcd82c4e317e27d0126c383b6892de4e5e3c6` |
| v24 completed-evidence manifest | `1d886b212c9f769d439c624222ced720bbb70ba655397e6151c5bf2d45d30be0` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `7d32488f135700285c3296fa5c4591948be176dc5a43372a48117c6eb2d57176` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v25` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow and all mandatory identities from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v25 batching implementation is the exact v24 method-B function.** AST comparison is exact for `project_batched_axis_operands`. It concatenates the unchanged request objects axis-major as `[charge,sx,sy,sz]`, calls `project_oracle_blocks_active_batch` once for each unchanged operand `[raw,anchor,grid0]`, reshapes only the explicit packing, and preserves the post-reduction control variate `(raw + anchor) - grid0`. The qualified `grid_oracle.py`, source loader, transfer code, inventory, and secure publication implementation are byte-identical to v23. The only v23 production helper removed is the 12-call separate-axis helper.
2. **The v24 evidence supports exact parity only in its declared seven-label scope.** The pinned job `513340` diagnostic output/sentinel chain verifies method A at 12 calls per Q against method B at 3 calls per Q. All recorded local/global operand, pre/post reverse-vertex, physical `-Q` relabelled contraction, and hole maxima are exactly zero for Q0 plus three reverse pairs. The output is explicitly diagnostic-only, unthresholded, `classification=null`, and grants no physical authority. This evidence does not numerically qualify the other 378 labels; v25 remains the first full-385 use of method B.
3. **The sole numerical ceiling change is bounded by preserved v23 failure evidence.** Recursive config comparison finds exactly one changed tolerance: `baseline_replay: 2e-10 -> 8e-8`. The immutable v23 job `512928` traceback records `6.177174549648612e-8`, below the new ceiling by a factor of about 1.295. Seven-Q ceilings remain `8e-8` for charge/hole, `2e-8` for spin, and `5e-10` for Q0 spin; every other tolerance is unchanged. The new baseline ceiling is a narrow deterministic replay admission bound, not a physical-error estimate.
4. **The diagnostic checkpoint is correctly placed and nonauthoritative.** It is invoked only after full coefficient/hole construction and the finite, reverse, Q0, and seven-Q gates, but before applying the primitive-127 baseline ceiling and before real-space, positivity/Pauli, quadrature, R5-to-R6 convergence, and cyclic robustness gates. It records all 385 charge/spin/hole slots, the primitive-baseline array and mask, measured early-gate metrics, and the ungated baseline discrepancy. NPZ, JSON, and hash manifest are committed by the existing no-follow hardlink transaction; the completion sentinel is linked last. Payloads/sentinel verify as `0444`, the committed directory as `0555`, and same-job committed reuse is rejected.
5. **Checkpoint publication grants no downstream authority.** JSON, manifest, and sentinel state `diagnostic_only=true` and `physical_authority=false`; the sentinel additionally requires reuse only by a separately reviewed successor. V25 contains no checkpoint-consumption route. The final output summary and `COMPLETE` sentinel can be produced only after all unchanged late physical gates pass. The final sentinel binds the diagnostic-checkpoint sentinel hash without converting that checkpoint into evidence of physical validity.
6. **Late failures preserve the diagnostic checkpoint while remaining failures.** Checkpoint publication fsyncs the sentinel-last transaction before the baseline gate executes. Any later raised gate reaches the existing MPI abort path and prevents final output publication, while the committed job-scoped checkpoint remains. Final publication remains the same exact seven-payload allowlist and sentinel-last protocol as v23, with only versioned v7/v10 schemas and the added checkpoint-sentinel binding. The recovery path is restricted to an already serialized final staging bundle from the same attested job and requires the committed checkpoint; it cannot promote a checkpoint-only late-gate failure.
7. **Downstream scientific gates and formulas are unchanged.** Direct source diff limits common function changes to versioned config/source closure, publication schema, the intended `run` edits, and checkpoint support. The real-space reconstruction sign, physical `-Q` labels, `1/48` normalization, Q0 authority split, reverse symmetrization, partition/integration, positivity/Pauli, quadrature, convergence, and fold-robustness code remain v23-exact. Final authority still requires all late gates and final sentinel-last publication.
8. **Failure and completion evidence are preserved without overstating scheduler accounting.** The v23 30-file manifest preserves source/review/control/log evidence and the baseline traceback; no v23 completion sentinel exists. The v24 32-file manifest preserves source/control/log and immutable diagnostic output closure. Both include 64-CPU running-allocation receipts. No terminal `sacct` or terminal-state record is present for either job, so v25 correctly makes no terminal scheduler-accounting claim.
9. **Evidence/control separation is fail-closed.** The capsule is mode `0555`; only empty mode-`0700` runtime/control/output and logs namespaces are mutable. Bootstrap under `-I -S` verifies the exact detached approval, source/input/evidence manifests, runtime identity, and receipt-bound held/released/running control chain before numerical imports or production. Submission is one-shot, held, dependency-free, full-node 64-rank, account `hmt03`, excludes `node037`, and release is a separate no-argument action.
10. **Fresh scheduler-free verification passed.** `pure_stdlib_tests.py`, `static_check.py`, `bootstrap.py --verify-only`, `bash -n`, independent predecessor byte-identity checks, an independent v24 output hash-chain check, AST comparison, and namespace/mode checks passed. No scheduler command/query, NumPy/SciPy/MPI production import, projection, contraction, or scientific numerical workload ran in this Audit.

No blocker was found within the immutable v25 one-attempt full-385 recovery scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25/scheduler_workflow.py
```

This command only submits v25 in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No v25 numerical runtime, MPI projection, contraction, checkpoint publication, late-gate evaluation, or final publication was executed. V24 exact parity covers only seven labels, not all 385. The v23 observation bounds the chosen replay ceiling but does not establish a physical accuracy scale. A diagnostic checkpoint after a late failure has no physical authority and may be reused only by a separately reviewed successor. Even successful final publication would validate only the declared finite shell/grid and fixed-rank local HF branch, not an infinite-shell result or the global ground state.
