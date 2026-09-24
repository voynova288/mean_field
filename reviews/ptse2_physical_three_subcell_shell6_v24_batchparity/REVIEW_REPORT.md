# External review report — PtSe2 physical three-subcell shell-6 v24 batch parity

- **Review ID:** `086b80cf-70e5-4186-8c5c-aad018be95a0`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T12:07:37Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v24_batchparity`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v24_batchparity`

## Decision

`approved_for_one_v24_unthresholded_sevenq_batchparity_diagnostic`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separate no-argument receipt-bound release helper. It authorizes one immutable v24 **diagnostic-only** comparison on Q0 plus three reverse pairs. It does **not** authorize OpenMX/oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, a numerical acceptance threshold, a pass/fail classification, repair or relaxation of the v23 primitive-127 baseline gate, projection of the other 378 channels, real-space reconstruction, or any physical claim.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `492f0c15919026c2919f27685c017d2211aa432653a744cbb51531e5aed5ec88` |
| source frozen object | `ee756b88438453f83708b5e5487e525985c480235aa675aeb80f03d0cf5f326b` |
| source checksum manifest | `47c233ea8446c8dca8b9cbbe09d3144eeb0867f4a006c1f58576c7cf68871676` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| v22 completed-evidence manifest | `9ec9b22adaf2bccc501063d3d66d483d322f2720fa58acaec08dde41320266fc` |
| v23 failure-evidence manifest | `98c16127fd90b22b8f05ddf8f64fcd82c4e317e27d0126c383b6892de4e5e3c6` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `abb5c6712f6c21b875634edb9d3b86655a01cfb53c74a7ba2f5775270abf8750` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_batchparity_external_review/v24` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow and mandatory identities from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The A/B comparison isolates batching.** Method A is AST-identical to the pinned v23 `apply_pauli_axis`, `project_separate_axis_operands`, `requests_for`, `assemble`, and `contract_stored` functions. It makes three projection calls inside each of four independently sourced axis families, hence 12 calls per selected Q. Method B concatenates the same request objects axis-major as `[charge,sx,sy,sz]`, makes one call for each of `[raw,anchor,grid0]`, and reshapes without arithmetic before the common MPI reduction, control-variate combination, assembly, reverse symmetrization, and contraction. The qualified `grid_oracle.py` is byte-identical to the current reviewed source copy.
2. **The compared inventories and tensors are complete for the declared seven-label scope.** The accepted artifact supplies exactly Q0 plus three reverse pairs. Both methods process the same seven oracle items and record rank-local operand maxima, separately MPI-summed operands, pre-reverse vertices, post-reverse vertices, physical-label contractions, holes, and method-internal reverse residuals. A and B are reduced separately, so their distinct local contraction order is preserved rather than hidden by pre-reduction subtraction.
3. **The `-Q` convention is correct and applied once.** The oracle supplies `V_Q=C_target^dagger M(+Q) C_source` with `target=source+Q`. `contract_stored` evaluates `sum Pstored_abk [V_Q,bak]* / 48`; under the package convention this belongs to physical output label `-Q`. `finish_method` therefore assigns the item-`Q` contraction to the partner slot and the dagger-partner contraction to `+Q`, with charge, hole, sx, sy, and sz moving together. The pinned observables-v1 producer independently uses the same reverse-label assignment. No second label reversal occurs in the v24 report or arrays.
4. **Q0 is a method-parity datum, not a production-authority substitution.** Both methods use the same fresh `raw + anchor - grid0` charge and Pauli projections at Q0. The cache-authoritative v23 Q0 charge path is intentionally absent. This makes the A/B test interpretable while preventing the diagnostic from being mistaken for v23 production replay.
5. **The diagnostic-only stop is enforced in code and publication metadata.** The runtime projects seven of 385 parsed rows, contains no accepted-coefficient threshold or primitive-baseline decision, performs no full-385 contraction or real-space integration, and emits `classification=null`, `thresholds_applied=false`, `physical_claim_authorized=false`, and diagnostic-only scope in the report and sentinel. Publication is allowlisted to JSON, NPZ, manifest, and sentinel-last completion. Any numerical A/B discrepancy remains a measurement requiring separate interpretation.
6. **V23 failure evidence is narrow and correctly represented.** The 30-file manifest binds the v23 source/review/control/log bundle. Job `512928` had an exact held/released/running runtime-control chain, then stderr reached the primitive-127 replay gate and reported `6.177174549648612e-08`; no v23 output sentinel exists. No terminal `sacct` or terminal `scontrol` record is present, so the evidence supports a scientific traceback/abort but not a scheduler-terminal-state claim. V24 does not rerun, relax, or classify that failed gate.
7. **V22 evidence is completed but remains diagnostic-only.** The seven-file bundle binds the v22 report, arrays, output manifest, sentinel, and allocation accounting. Job `512854` is recorded as `COMPLETED`, `ExitCode=0:0`, with 64 CPUs. Its sentinel states seven labels, no full-385 projection, no real-space integration, no thresholds, null classification, and no physical authority. It supports label/convention lineage and the choice of the same seven labels, not full-shell validation.
8. **Runtime and control are fail-closed and separate from evidence.** The capsule root is mode `0555`; only empty `0700` `runtime/control`, `runtime/output`, and `logs` namespaces are mutable. The source, 19 scientific inputs, predecessor evidence, and transport/oracle manifests verify. Bootstrap starts under `-I -S`, verifies the detached approval before any control action, digest-loads approved sources, reattests the pinned numerical interpreter/packages/shared libraries before production, and requires the exact source-bound runtime/control chain. Submission is one-shot, held, dependency-free, 64-rank/full-node, account `hmt03`, excludes `node037`, and release is a separate receipt-bound action followed by running attestation.
9. **Fresh scheduler-free verification passed.** `pure_stdlib_tests.py`, `static_check.py`, `bootstrap.py --verify-only`, `bash -n`, and strict verification of all source/input/evidence manifests passed. The namespace closure reports 74 referenced globals and zero unresolved names. A separate current-v24 state-machine mock passed held submission, two stable positive release snapshots, running attestation, timeout sealing, and runner/release interleaving without scheduler contact. No NumPy/SciPy/MPI production import, projection, contraction, scheduler query/action, or scientific numerical workload was run in this Audit.

No blocker was found within the immutable v24 unthresholded seven-label A/B batching diagnostic scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v24_batchparity/scheduler_workflow.py
```

This command only submits the v24 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v24_batchparity/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No v24 numerical runtime, MPI projection, contraction, or output publication was executed. Static identity and control mocks do not establish future A/B values. Seven-label parity cannot validate the other 378 channels, explain or clear the v23 primitive-127 discrepancy, establish shell convergence, validate a physical observable, or prove that the selected fixed-rank HF branch is the global ground state.
