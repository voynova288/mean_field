# External review report — PtSe2 physical three-subcell shell-6 v21 seven-Q diagnostic

- **Review ID:** `1dc40e07-5ecc-4733-b9f5-51fe69a330ee`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T05:57:54Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v21_sevenqdiag`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v21_sevenqdiag`

## Decision

`approved_for_one_v21_unthresholded_sevenq_replay_diagnostic`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. It authorizes a **diagnostic-only** replay of the seven labels already present in the pinned accepted artifact: Q0 plus three reverse pairs. It does **not** authorize OpenMX/oracle generation, projection or contraction of the other 378 shell-6 labels, real-space reconstruction/integration, threshold-based acceptance, a physical pass/fail classification, a v20 recovery conclusion, direct `sbatch`/`scontrol`, changed bytes, argument-bearing invocation, or repeated submission.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `1fea6d2d3f5dbf5a3487b61d8b0a958cf63d8362c536f48de781f280fde3f383` |
| source frozen object | `46d8cf238fa340131e6856d7d2f74808b33c3135600ff5bf455daf0f61d46779` |
| source checksum manifest | `7a52f2b9904ef86c8c73d3b22a7a928012d1f14fc23df7394cd102f2b470dd25` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| v19 completed-evidence manifest | `84a4e5baa889240dd0185ca1c8a10b04b2ec0f4ae4387cf83f65460f3ca1f27b` |
| v20 failure-evidence manifest | `7c0bc696974698089f071a8b98ec090b6f6c8a0caba0098d6c2b69f5db237abc` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `8b23a61e460999a8fd5db602289cec983857cddcc2b9a538811ef77c5cf38eae` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_sevenqdiag_external_review/v21` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The seven-label construction retains the relevant v20 formulas.** Thirteen named construction helpers have exact AST identity with the hash-pinned v20 job-512808 source, including active-frame loading, spin-major Pauli transforms, separate raw/anchor/grid0 projection, determinant-three mapping and fold assembly, cache-authoritative Q0 charge assembly, and the `1/48` contraction. Manual review of the selected branch confirms the same `raw + anchor - grid0` combination, cache substitution only for Q0 charge, fresh sx/sy/sz, reverse-pair projection `[V(Q)+V(-Q)^dagger]/2`, and stored-projector contractions. V21 removes the v20 acceptance gates and downstream full-shell/real-space work; it does not silently replace these formulas.
2. **The comparisons are unthresholded.** For all seven labels the output records computed and pinned accepted complex charge, hole, sx, sy, and sz values and exact computed-minus-accepted differences. It also records per-channel maxima, an overall coefficient-difference maximum, Q0 cache-versus-fresh diagnostics, and reverse-pair residuals. No accepted-value discrepancy is compared with a tolerance, clipped, rounded into a verdict, or used to suppress publication. The only remaining numeric bounds are structural input/coordinate/projector checks and a Pauli convention self-test; they are preconditions, not replay acceptance thresholds.
3. **Stream selection occurs before scientific projection.** Each rank sequentially parses and inline-hashes its complete primitive-127 and missing-258 shards, preserving the two frozen stream identities. The callback stores primitive Q0 as the grid0 control operand, then returns immediately for every key outside the broadcast seven-key reverse-closed set. Only the selected seven keys build requests, invoke separate-axis raw/anchor/grid0 projection, reduce, assemble, and contract. Runtime closure requires exactly seven selected keys, Q0, three finite reverse pairs, and no pending pair.
4. **The diagnostic-only stop is explicit.** Full-shell coefficient arrays, primitive-baseline replay, shell convergence, real-space reconstruction, regional integration, positivity/Pauli-density conclusions, and physical classification are absent. Both report and completion sentinel carry `classification=null`, `thresholds_applied=false`, `diagnostic_only=true`, `full385_projection_performed=false`, `realspace_integration_performed=false`, and `physical_claim_authorized=false`. Publication is limited to the diagnostic JSON, arrays NPZ, output hash manifest, and sentinel-last completion record.
5. **Evidence and control are separated.** Evidence is the immutable completed v19 Q0 diagnostic/accounting, the pinned v20 job-512808 in-job collective Q0 exception and absent copied output/sentinel, the accepted seven-Q artifact, and the v8 oracle inventory. Control is the read-only v21 source closure, exact external-input hashes, pinned runtime, detached approval requirement, empty isolated mutable namespaces, one-shot held submission, separate receipt-bound release, and runtime source/oracle rechecks. Neither this approval nor prior evidence classifies the future diagnostic values.
6. **The v20 failure evidence is scoped honestly.** The 31-file manifest binds the v20 source/config/review/control chain and stderr exception. It contains no terminal `sacct` record, so it does not independently certify the scheduler's terminal state. The v21 report preserves that uncertainty rather than upgrading the failure evidence.
7. **The accepted comparator remains diagnostic evidence, not physical authority.** Its pinned summary labels the source as a diagnostic physical finite-Q projection and states `hf_physics_authorized=false`. V21 uses it only as the comparison target and publishes raw differences; this approval does not promote that artifact or the local fixed-rank HF branch to a global-ground-state or physical-result claim.
8. **Publication and recovery remain fail-closed.** Job-scoped staging and final paths must be absent; payload names and hashes are exact; the immutable sentinel is linked last. Recovery accepts only the same live Slurm job ID and the same exact staged three-payload contract. No foreign payload or pre-existing committed sentinel is accepted.
9. **Evidence/control workflow review passed.** The scheduler wrapper is 64-rank, one-node, exclusive, held, dependency-free, and contains no OpenMX path. Submission authority is consumed one-shot; release is a separate no-argument action bound to the held receipt and stable positive release verification. Direct scheduler commands are outside this approval.
10. **Fresh scheduler-free verification passed.** `pure_stdlib_tests.py`, `static_check.py`, `bootstrap.py --verify-only`, `bash -n`, and strict validation of 21 source rows, 19 scientific inputs, 13 v19 evidence rows, and 31 v20 evidence rows all passed. The tests reported `scheduler_contact=false` and `scientific_numerics=false`; runtime/control, runtime/output, and logs were empty. No scheduler command/query, NumPy/SciPy/MPI import, oracle projection, contraction, or scientific numerical workload was performed in this Audit.

No blocker was found within the immutable v21 seven-label, unthresholded, diagnostic-only scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v21_sevenqdiag/scheduler_workflow.py
```

This command only submits the v21 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v21_sevenqdiag/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, seven-Q contraction, or output publication was executed. Static AST identity and mocks do not establish the future discrepancy values or real-MPI behavior. V20 job 512808 lacks pinned terminal accounting. The accepted seven-Q artifact and the selected HF root retain their stated diagnostic/local-branch limitations.
