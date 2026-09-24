# External review report — PtSe2 physical three-subcell shell-6 v14 Q0 diagnostic

- **Review ID:** `e2674b04-efe0-4f58-9a36-1a1ab57812e4`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T02:06:24Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v14_q0diag`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v14_q0diag`

## Decision

`approved_for_one_v14_projection_only_q0_diagnostic`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v14 release helper. The run is **diagnostic-only**: it may expose the unthresholded Q0 assembly values hidden by the pinned v13 failure, but it may not project later Q rows, perform the full-385 projection, run OpenMX, or support a physical claim. Changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, or bypass of receipt-bound release verification are not approved.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `57d306f5f803c00a4d8aa9e291f7763dabe641cae4a7149938719d34e72bdbdb` |
| source frozen object | `a84fdb6bf7a22e01da5d95d48bc3b80187fcac685ccfdd316d8567ff7cc2346b` |
| source checksum manifest | `665edf71cf9dd9e79cb4c7149c3ec5ccb441b7bd9f54c23acc5d1e3a712081a7` |
| scientific-input checksum manifest | `d184b8412fc271adff209d368f1fccdf07a2df05cae25a1066391d7ac9deaccc` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| v13 failure-evidence manifest | `3aa6368ab1d2f94fed7be58e48446e26eb28fedfa496f6053b957efe1cc4b8d0` |
| static checks artifact | `ad2e140392c4a8545525f1d4d4921013b6882bed11e03e5881cc75cd15049805` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `286e896040e352451e7a0db3b6f5e211933925641151b17203956e7624b5463c` |
| approval schema | `ptse2_physical_three_subcell_shell6_q0diag_external_review/v14` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v14_q0diag/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v13 Q0 scientific path is retained exactly through assembly.** After excluding docstrings, AST bodies are identical between v13 and v14 for `load_active`, `fold_map`, `pauli_sources`, `validate_pauli`, `project_control_variate_active`, `requests_for`, `assemble`, and `contract`; the active-lineage digest helper is also identical. The v14 prefix reader parses the same complete primitive Q-row table and first Q0 block tuple. It proves that row 0 is exact Q0, uses that same tuple as both `raw(Q0)` and `grid0`, projects raw/grid0/physical-overlap anchor independently with the same charge/sx/sy/sz request list, combines `(P(raw)-P(grid0))+P(anchor)`, applies the same 64-rank MPI sum, and uses the unchanged `(24,24,48)` folded assembly. This is static path equivalence, not a numerical parity result.
2. **The requested diagnostic surface is complete and unthresholded.** For every reduced K, the JSON records charge-versus-I24 max/Frobenius residuals, charge Hermiticity max/Frobenius residuals, and for sx/sy/sz the Hermiticity max/Frobenius residuals, Hermitian-part minimum/maximum eigenvalues, spectral radius, and unit-bound excess. It also records every axis and all nine fold-block max/Frobenius norms; all 144 active primitive Gram max/Frobenius and Frobenius/sqrt(8) residuals; all 144 direct-projected versus assembled diagonal fold-block residuals for four axes; the full primitive-coordinate/reduced-K/target-fold/source-fold inventory and per-K counts; assembled charge/Pauli contractions against the pinned seven-Q direct Q0 replay and the available prior contracted Pauli values; and all 64 prefix-read receipts with `full_shard_payload_rehash_performed=false` and `later_q_payload_bytes_read=0`. No diagnostic threshold or pass/fail classification is applied after payload construction; prerequisite source/projector/Q-coordinate integrity gates remain prerequisites and are not reclassified as diagnostic thresholds.
3. **Execution stops before the full385 path.** The run calls one Q0-prefix reader and one assembly. It has no call to v13's full-shard streamer, reverse-pair `finish_pair`, later-Q projection, missing-258 payload projection, Fourier contraction over 385 channels, shell reconstruction, or physical gate. The v8 missing-258 oracle is checked only for immutable inventory/mode/control lineage; its payload is not read or rehashed. The exact outputs are `q0_diagnostic.json`, `q0_vertices.npz`, `OUTPUT_SHA256.json`, and sentinel-last `Q0_DIAGNOSTIC_COMPLETE`. Both fresh and recovery publication bind `diagnostic_only=true`, `full385_projection_performed=false`, and `physical_claim_authorized=false`.
4. **Diagnostic-only wording and authority limits are consistent end to end.** README, config, review template, source-frozen authorization, runner status/authority fields, wrapper comment, output schema, recovery gate, and sentinel all deny full-385 or physical authority. `COMPLETE` is not emitted. The approved decision itself is narrowly named `approved_for_one_v14_projection_only_q0_diagnostic`.
5. **Release verification is mandatory and waited on.** The no-argument submission helper submits exactly one held, no-dependency projection and does not release it. The separate no-argument release helper is receipt-bound, writes an intent before `scontrol release`, rejects ambiguous timeout/retry, and requires stable positive verification. The allocated wrapper first calls `attest-running-projection`; both that path and the production runner wait for the matching release action and `projection.RELEASE_VERIFICATION.json`, and the runner additionally requires the bound running receipt before numerical imports/execution.
6. **The v13 failure is pinned exactly and remains negative-only.** `V13_FAILURE_EVIDENCE_SHA256SUMS.txt` has 23 read-only hash-verified rows: detached v13 review approval/report, incident record, empty stdout, stderr, and 18 submission/release/running control records. The stderr records `RuntimeError: Q0 projected I24/S-tensor-Pauli gate failed` and MPI abort for job 511760. No terminal accounting evidence, v13 output, or v13 completion sentinel is claimed or adopted.
7. **The Slurm surface remains projection-only and held.** The exact wrapper requests account `hmt03`, partitions `regular256,regular6430`, excludes `node037`, requests one full node with 64 one-CPU MPI tasks, `--exclusive --mem=0`, and has no explicit time or dependency directive. It invokes no OpenMX executable. Before this approval, `runtime/control`, `runtime/output`, and `logs` were empty.
8. **Fresh static verification passed without Slurm or numerics.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and all seven strict checksum-manifest checks passed: 22 source rows, 17 scientific-input rows, 11 v8-evidence rows, and 23 rows each for v9, v10, v12, and v13 failure evidence. Scheduler-free mocks covered held submission, stable release/running verification, ambiguous submission timeout sealing, and runner waiting for receipt-bound release verification. No Slurm command/query, NumPy/SciPy/MPI import, projection, OpenMX execution, or scientific numerical workload was run.

No blocker was found within this immutable v14 Q0 diagnostic-only scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v14_q0diag/scheduler_workflow.py
```

This command only submits the Q0 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v14_q0diag/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, Q0 diagnostic, or output publication was executed. The Q0 values and the cause of the v13 failure remain unknown. Approval is based on static source/provenance review and scheduler-free mocks only; the resulting diagnostic will remain non-authoritative for full385 sufficiency or any physical claim.
