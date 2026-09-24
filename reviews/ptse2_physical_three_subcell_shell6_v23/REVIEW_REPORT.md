# External review report — PtSe2 physical three-subcell shell-6 v23 convention recovery

- **Review ID:** `66c4d9c3-0768-45da-9465-e2bc834237c2`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T09:07:31Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v23`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v23`

## Decision

`approved_for_one_v23_physical_label_minus_iqr_full385_projection_recovery`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. It authorizes the immutable v23 **projection-only full-385 recovery attempt**. It does not authorize OpenMX/oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of any Q0/replay/full-shell physical gate, or a physical claim before successful sentinel-last publication. V22 remains diagnostic-only evidence and is not itself a full-385 or physical validation.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `79792e522e2239dd859995d87558e722c940a6209f4108665d6707b3e0f53c35` |
| source frozen object | `88388b919d1798ddbf73c1fed37db8d53570178ed81a1ac8ae9ca263674fc0f3` |
| source checksum manifest | `3133f0a84a4846b2d139a49122fd79992f983dae818ea652856eeb0f79ad591d` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| observables-v1 evidence manifest | `6dcb40a266a51a44a6b709be6b86a008fb7b2b3b0d4cc1d8da3f2937f3bddd1c` |
| v19 completed-evidence manifest | `84a4e5baa889240dd0185ca1c8a10b04b2ec0f4ae4387cf83f65460f3ca1f27b` |
| v20 failure-evidence manifest | `7c0bc696974698089f071a8b98ec090b6f6c8a0caba0098d6c2b69f5db237abc` |
| v22 completed-evidence manifest | `732a6b261a3779d601dc17e6e984642a2a9ada136800b42530c16cfd16da3c14` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `b6cf22670552bca4f5bb77217700e98353a3989b929b46a0ed741365c5ffe3d0` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v23` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object and all mandatory identities from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The convention correction is derived, not cosmetic.** The pinned oracle convention is `V_Q=C_target^dagger M(+Q) C_source`, with `M(+Q)=exp(+iQ.r)` and `target=source+Q`. `contract_stored` evaluates the stored-projector contraction with `conj(V_Q)`, so its output is the physical coefficient at `-Q` under the package convention `f(r)=A^-1 sum_Q f_Q exp(-iQ.r)`. The hash-pinned observables-v1 producer independently assigns `contract_stored(vertex[label])` to `reverse=-label` and reconstructs with `exp(-2 pi i label.r)`.
2. **Reverse-pair assignment is complete.** In v23, `finish_pair` assigns the contraction of the projected item-Q vertex to partner slot `-Q`, and the contraction of its dagger/reverse partner to `+Q`. Charge, hole, sx, sy, and sz move together. The later seven-Q gate compares already relabeled production coefficients directly to accepted physical labels; it does not apply a second runtime reversal. The linear determinant-three label map makes the exact `(u,v)->(-u,-v)` partner also `label->-label` for all 385 channels.
3. **The real-space field is unchanged by the coordinated relabel/sign change.** The scheduler-free reverse-closed complex mock verifies `sum_Q g_Q exp(+iQ.r)=sum_Q f_Q exp(-iQ.r)` for `f_Q=g_-Q` at four points. Maximum field difference is `2.220446049250313e-16`; maximum imaginary residue is `1.1102230246251565e-16`. This proves field equivalence for the relabeling identity; it is not a substitute for a v23 full-shell execution.
4. **Completed v22 evidence supports the replay ceilings within its narrow scope.** The pinned v22 bundle binds diagnostic JSON/NPZ, output manifest, sentinel, allocation accounting, and raw accounting row for job `512854`: `COMPLETED`, `ExitCode=0:0`, 64 CPUs. Reversing computed labels before comparison gives charge/hole `6.729381390865313e-8`, sx `1.1979082467629634e-8`, sy `7.499314893022193e-9`, sz `1.413565901166216e-8`, and Q0 fresh spin `4.038995596311601e-10`. The ceilings `8e-8`, `2e-8`, and `5e-10` provide factors `1.1888`, `1.4149`, and `1.2379` over the respective largest observations. They are narrow deterministic replay ceilings, not uncertainty estimates or accuracy claims.
5. **V22 evidence/control lineage verifies.** The v22 sentinel is diagnostic-only, seven-label, unthresholded, `classification=null`, with no full-385 contraction, real-space integration, or physical authority. Its original immutable capsule remains mode `0555`; strict source checks pass. The v22 source-frozen hash is `203fe4f0e61b6bb88c5c1a173eadd131e547aa6f2c46f53fd048222c188656c3`, source-manifest hash is `8494df1ed19a571b3482617b10da2ec1a1560ffcd3f7ec7a1f3c136804d298c8`, projection-source hash is `84bfb999b68079338dc52a82fdf0f4015707f3f605d5cc5fd41838b082086f6b`, and detached v22 approval/report hashes are `9d9f247d9a620f0ecdb8070fd8c1bb1067a1998fb435535ed83505e87af9743d` / `6109bcf611bc34c819b6b2714ad9f03fb594b797e3247cc4891a3afb4d334f18`.
6. **Q0 authority remains split correctly.** Authoritative Q0 charge/hole still comes from the observables-v1 cache mapped through the physical-Q chart and retains the strict `2e-11` replay gate. Fresh charge remains diagnostic-only behind the source-bound direct-Gram `5e-8` gate. Fresh sx/sy/sz retain Hermiticity `2e-12` and Pauli spectral-excess `1e-9` gates; only their accepted-artifact replay ceiling changes to the v22-backed `5e-10`. The 64-participant collective Q0 verdict still precedes any later-Q consumption.
7. **Unrelated full-385 physical gates are unchanged from v20.** Twenty-one common scientific/helper function ASTs are exact matches. Of 28 common tolerance keys, 27 are unchanged; the one intentional change is Q0 fresh-spin replay. The old combined seven-Q overlap key is replaced by separate v22-calibrated charge/hole and spin ceilings. Reverse raw/projected checks, coefficient reality, `1/48` normalization, Q0 counts, primitive-127 baseline replay, 385-channel inventory, real-space partition/integral/positivity/Pauli checks, 192-to-384 quadrature checks, R5-to-R6 pointwise/vector/regional convergence, and cyclic-fold robustness remain in the production path with their v20 thresholds.
8. **Immutable evidence and control are fail-closed.** The capsule root is mode `0555`; only empty `0700` runtime/control/output/log namespaces are mutable. Source, 19 scientific inputs, observables evidence, oracle evidence, and all predecessor evidence manifests pass strict checks. Submission is one-shot, held, dependency-free, full-node 64-rank, account `hmt03`, excludes `node037`, and requires separate receipt-bound release plus running attestation. Bootstrap verification requires this detached approval before any control or production action.
9. **Fresh scheduler-free verification passed.** `pure_stdlib_tests.py`, `static_check.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checks of all 13 source/input/evidence manifests passed. Reports state `scheduler_contact=false`, `scientific_numerics=false`, and `writes=false`; no bytecode, runtime/control/output, or log files were created. No scheduler command/query, NumPy/SciPy/MPI import, projection, contraction, or scientific numerical workload was run in this Audit.

No blocker was found within the immutable v23 convention-correction and one-attempt full-385 projection-recovery scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v23/scheduler_workflow.py
```

This command only submits the v23 projection in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v23/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No v23 numerical runtime, MPI projection, full-385 contraction, real-space integration, or output publication was executed. Static source identity, algebraic field equivalence, and completed v22 seven-label replay evidence do not establish later-Q passage, shell convergence, a future published observable, or global-ground-state selection. V20 has no pinned terminal accounting, and v22 cannot validate the 378 channels it did not project.
