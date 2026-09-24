# External review report — PtSe2 physical three-subcell shell-6 v5

- **Review ID:** `2b14d46b-b6d3-4367-8cfe-d7b406efd30c`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-18T18:41:35Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v5`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v5`

## Decision

`approved_for_one_digest_loaded_timeout_reconciled_held_oracle_afterok_projection_chain`

This approval is limited to the exact no-argument, held Slurm workflow bound below. It is not approval of modified bytes, argument-bearing invocation, direct production entry points, manual scheduler commands, release commands, scientific output, convergence, a global-ground-state claim, or the statement “shell6 sufficient under declared thresholds.” That scientific statement remains contingent on the unexecuted production gates.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `b26608f0e302ccf04d3ec04f33e4d512764fae6361f161dce3165a040f8fbb0c` |
| source frozen object | `d7f9b7306dc0b2c7efb75decddbe51eec4d566c5a9e66c016ff4dbc1c98014c2` |
| source checksum manifest | `7b852f34d46076a0a3a32ea01b74581cd8c920db584862fd22d1e78d3f9e84f0` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e1347bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| historical-input checksum manifest | `af6206e9cdf97fc9e00348df808cda9088d48f0c447ae23bd6d77040ea5d6775` |
| static checks artifact | `3283335b9692c9ffc1cbe089dd2aa4fb755382236bd404d01bad4f579641ee54` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `90ae77354891860adf432dffc6e6575a6bc42680d66ee75e518b8d20f1f8b32a` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v5` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v5/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON separately hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. All 27 frozen source rows, 14 scientific-input rows, and 5 historical-input rows passed strict SHA256 verification. The capsule is mode `0555`; frozen files/directories are read-only; mutable `runtime/`, `runtime/control/`, `runtime/output/`, and `logs/` are mode `0700` and empty. No symlink or bytecode residue was found.
2. The historical v4 intent, attempt, detached approval, and incident report are byte-identical to their source artifacts. Job `511413` is bound as historical negative control-plane evidence only. The incident fixture is a reconstruction of the captured one-line record; it does not independently prove terminal cancellation and does not carry v5 execution or scientific authority.
3. The exact job-511413 fixture parses to the observed held fields. Generic syntactic key framing terminates target values at colon-bearing non-target fields such as `AllocNode:Sid`, `ReqB:S:C:T`, and `NtasksPerN:B:S:C`; missing, duplicate, malformed-boundary, and resource mutations fail closed.
4. Held attestation requires `PENDING/JobHeldUser/Priority=0`, `Partition=regular256,regular6430`, `NumNodes=1-1`, `NumCPUs=64`, `NumTasks=64`, `CPUs/Task=1`, `MinMemoryNode=0`, `OverSubscribe=NO`, exact `ReqTRES=cpu=64,mem=250G,node=1,billing=64`, and role-correct dependency. Running/configuring/completed attestation requires one selected partition and `NumNodes=1`. The held values are fixture-backed; allocated-state values are pure-stdlib contract fixtures, not a fresh live Slurm observation. Full-node evidence here is job-record scoped; no node/partition record was queried in this no-scheduler review.
5. Submission mocks passed normal return, timeout with one stable exact candidate, zero/multiple candidates, and query failure. Release mocks passed normal release, timeout-after-side-effect, timeout-before-side-effect with one bounded retry, query-failure ambiguity without retry, and orphaned-intent fail-closed recovery. The no-argument shims load scheduler control only after source, historical, template, and detached-review verification.
6. The v4→v5 science lineage gate passed. Ten science/runtime/input files are byte-identical; all enumerated formula/data-path functions are AST-identical; the scientific config payload is identical after removing versioned control-plane keys. Direct diff confirms remaining production changes are v5 provenance/source-closure/schema/path changes. This is static no-drift evidence, not fresh numerical validation.
7. Both Slurm wrappers passed `bash -n`, retain account `hmt03`, the two-partition request, `node037` exclusion, 64 tasks, `--exclusive`, `--mem=0`, and no explicit time directive. The exact workflow is no-argument and submits both jobs held; it does not release either job.
8. `static_check.py`, `pure_stdlib_tests.py`, and bootstrap `--verify-only` passed without scheduler contact, project/numerical imports, production writes, or scientific execution.

No P0/P1 blocker was found within this static/control-plane scope.

## Authorized workflow

Exactly one invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v5/scheduler_workflow.py
```

The script enforces `len(sys.argv) == 1` and enters the digest-bound bootstrap submission action. It submits the oracle held and the projection held with exact `afterok:<oracle_job_id>` lineage; it does not release either job. This review authorizes one operator invocation of that exact command, not repeated submissions.

## Review limitations

No scheduler command or query was made. No NumPy, SciPy, mpi4py, OpenMX, MPI projection, HF solve, production publication, or other scientific numerical workload was run. No release helper is approved by this detached decision. The review establishes static identity and one held-chain control-plane authorization only.
