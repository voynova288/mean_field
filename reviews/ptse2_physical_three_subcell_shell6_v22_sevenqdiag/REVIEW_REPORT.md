# External review report — PtSe2 physical three-subcell shell-6 v22 seven-Q diagnostic

- **Review ID:** `8127b219-11c5-4942-b53d-fe60f5a2f454`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T06:36:44Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v22_sevenqdiag`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v22_sevenqdiag`

## Decision

`approved_for_one_v22_runtime_closed_unthresholded_sevenq_replay_diagnostic`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. It authorizes a **diagnostic-only** replay of the seven labels already present in the pinned accepted artifact: Q0 plus three reverse pairs. It does **not** authorize OpenMX/oracle generation, projection or contraction of the other 378 shell-6 labels, real-space reconstruction/integration, threshold-based acceptance, a physical pass/fail classification, direct `sbatch`/`scontrol`, changed bytes, argument-bearing invocation, or repeated submission.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `a4b043c36947003c84a8d4693a87e242c66b0fb66bafe7895b32d3b15b1c5b09` |
| source frozen object | `203fe4f0e61b6bb88c5c1a173eadd131e547aa6f2c46f53fd048222c188656c3` |
| source checksum manifest | `8494df1ed19a571b3482617b10da2ec1a1560ffcd3f7ec7a1f3c136804d298c8` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| v19 completed-evidence manifest | `84a4e5baa889240dd0185ca1c8a10b04b2ec0f4ae4387cf83f65460f3ca1f27b` |
| v20 failure-evidence manifest | `7c0bc696974698089f071a8b98ec090b6f6c8a0caba0098d6c2b69f5db237abc` |
| v21 failure-evidence manifest | `56868631fa342100b0395e17a2aaff4d671a4588790ad85bcede9747e77da47d` |
| qualified transport-source manifest | `51b065dce701f2354f343e27fefdaaf60ae48260319fa4a16590ea3ab4849693` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| static checks artifact | `15cfbf4c0d67bb44232f2c44345e711be060ba586c415d4af9e81cf6398176a7` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_sevenqdiag_external_review/v22` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v21 failure is correctly isolated as a runtime import-closure defect.** Pinned job-512818 stderr reaches `requests_for` and raises `NameError: name 'plus_iqr_transport' is not defined` before the first selected-Q projection call. The v22 runner adds the exact `from contracts import plus_iqr_transport` binding and does not alter `requests_for` itself.
2. **All projection globals are resolved before scheduler action and again before `main`.** The pure-stdlib transitive analysis rooted at `run`/`main` follows 25 projection functions and inventories 76 referenced globals, comprising 52 non-builtin module names and zero unresolved names. The bootstrap runs the actual resolver before external-review verification or any scheduler action. An independent exhaustive static rehearsal removed each of the 52 module names in turn and every omission raised `NameError`; substitution of `plus_iqr_transport` by another callable was also rejected. The production runner repeats the resolver against live `globals()` after imports and before `main`.
3. **The qualified transport contract is exact.** Runtime `contracts.py` has SHA256 `b879513baccd3da0d35c397b8ed91cc9ec9a9c46678fa900a23baf7a54595d08`, byte-identical to the pinned v20 qualified contract module. `plus_iqr_transport` has exact AST identity across the pinned v16, v20, and v22 copies. The runtime resolver additionally requires object identity with `contracts.plus_iqr_transport`, so a same-name substitute does not satisfy the gate.
4. **The v21 seven-Q diagnostic is preserved.** After normalization of v22-only schema/path/payload names, 24 of 26 v21 diagnostic functions are AST-identical; the only function differences are `config` and `verify_source_closure`, which update provenance bindings. The 12 construction functions pinned to v20—including `requests_for`, separate-axis projection, determinant-three assembly, cache-authoritative Q0 charge construction, and `1/48` contraction—remain exact AST matches. Direct source diff shows no scientific-formula change beyond the added qualified import/runtime preflight and versioned provenance/publication names.
5. **The diagnostic scope remains seven-Q and unthresholded.** All 385 oracle rows are parsed and hash-checked, but only Q0 plus three reverse pairs reach request construction, projection, assembly, and contraction. Output still records computed, accepted, and computed-minus-accepted charge/hole/sx/sy/sz data, Q0 cache/fresh diagnostics, and reverse-pair residuals with `classification=null` and `thresholds_applied=false`. No full-shell contraction or real-space path is present.
6. **The source and evidence contracts are exact and detached.** The bootstrap starts under `-I -S`, verifies the 23-row source manifest and 19-row scientific-input manifest, installs digest-bound source loaders, and requires a detached approval that binds the template, source freeze, static checks, runtime, evidence manifests, and exact workflow. V21 failure evidence is copied as 33 immutable records with no output/sentinel and no claimed terminal accounting; v19 completed and v20 failure evidence retain their prior scope.
7. **Publication and control remain fail-closed.** V22 uses new staging/final paths, payload names, schemas, manifest, and sentinel, preventing reuse of v21 mutable state. The held workflow is one-shot, dependency-free, 64-rank, one-node, exclusive, account `hmt03`, with separate receipt-bound release. Direct scheduler commands remain outside this approval.
8. **Fresh static verification passed.** `pure_stdlib_tests.py`, `static_check.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checks of all source/input/evidence manifests passed. Reports were exactly `scheduler_contact=false`, `scientific_numerics=false`, zero unresolved projection globals, and empty v22 runtime/control, runtime/output, and log namespaces. No scheduler command/query, NumPy/SciPy/MPI import, oracle projection, contraction, or scientific numerical workload was performed in this Audit.

No blocker was found within the immutable v22 runtime-closure-only, seven-label, unthresholded, diagnostic-only scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v22_sevenqdiag/scheduler_workflow.py
```

This command only submits the v22 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v22_sevenqdiag/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, seven-Q contraction, or output publication was executed. Static namespace closure and AST identity do not establish real-MPI execution or future numerical discrepancy values. Job 512818 lacks pinned terminal accounting. The accepted seven-Q comparator remains diagnostic evidence, and the selected HF root remains a local fixed-rank branch rather than proof of global-ground-state selection.
