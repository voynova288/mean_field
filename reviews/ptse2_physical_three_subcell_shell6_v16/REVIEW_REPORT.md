# External review report — PtSe2 physical three-subcell shell-6 v16 collective Q0 verdict recovery

- **Review ID:** `51c8e347-3876-4461-96dc-5aa4db9df7f7`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T03:03:46Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v16`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v16`

## Decision

`approved_for_one_v16_projection_only_collective_q0_ordering_recovery`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v16 release helper. It authorizes projection-only reuse of the immutable v8 oracle with the v15 separate-axis science and the v16 collective Q0 verdict ordering. It does **not** authorize an OpenMX rerun, oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of receipt-bound release verification, or any physical conclusion before all production Q0, seven-Q, reverse, normalization, convergence, and publication gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `5ce60c41c7d879210b04979670e8df8f850413ea91c83e797b337c3dfd58ed6b` |
| source frozen object | `7af77458121026bd4cf52f3c256346368fd3ad44e92659811fe762a33167de35` |
| source checksum manifest | `235b208452390e3a908f9869b9fe831e9f00fcd01e9f0431e980b53456d3f548` |
| scientific-input checksum manifest | `d184b8412fc271adff209d368f1fccdf07a2df05cae25a1066391d7ac9deaccc` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| v13 failure-evidence manifest | `3aa6368ab1d2f94fed7be58e48446e26eb28fedfa496f6053b957efe1cc4b8d0` |
| v14 Q0-diagnostic evidence manifest | `8ff1ef896521feae6505dc4b49ee1d71c77bfaa4164cbbfddc56b174665debac` |
| static checks artifact | `9da52ae66c26a87daced3b40d2361964c8d765c5dd0b137b3bf177ad46fcd377` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v16` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v16/scheduler_workflow.py` |
| workflow arguments | `[]` |

The detached approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **Q0 verdict participation is collective in the production path.** Every rank executes the unchanged `comm.Reduce(local_operands, global_operands, op=MPI.SUM, root=0)` before entering the Q0 branch. For Q0, every rank then calls `collective_q0_verdict`; the helper requires exactly 64 ranks and performs one `comm.bcast(..., root=0)`. The Q0 callback cannot return before that call completes on the calling rank. The non-Q0 assembly/pairing path is textually below the Q0 collective-and-return branch.
2. **Rank-0 scientific failure is broadcast before any rank raises from the verdict.** Rank 0 alone invokes the unchanged Q0 evaluator inside `try/except BaseException`, converts either outcome into one exact `ptse2_q0_collective_verdict/v1` record, and enters the broadcast. The original rank-0 exception is not re-raised locally before broadcast. After broadcast, every rank validates the exact schema, phase, rank count, gate inventory, success type, and error fields; a failed record then raises the same collective `RuntimeError` on all ranks. Fatal process/MPI failures outside Python exception delivery remain outside this static guarantee.
3. **No later Q is reachable before a successful Q0 verdict.** The Q0 branch unconditionally calls the collective helper and immediately returns only if it receives a valid success record. A failed or malformed record raises before callback return. Only after a successful Q0 callback return can `stream_rank_owned_shard` consume the next row. No later-Q assembly is performed by nonroot ranks, preserving the predecessor execution shape.
4. **The v16 scientific operations preserve v15.** The complete v15→v16 production diff changes bootstrap/schema identities, imports the verdict helper, moves the existing Q0 body into the rank-0 evaluator, and versions output schemas. The Q0 raw/anchor/grid0 combination, `assemble`, identity/Hermiticity tests, `finish_pair`, direct charge/hole/spin replay, and strict v14-improvement comparisons are unchanged. Later-Q combination, reverse pairing, contraction, reconstruction, and all downstream gates remain unchanged. Independent semantic comparison of `config.json` found no tolerance or scientific-input change.
5. **The v15 separate-axis science is retained rather than silently replaced.** Charge, sx, sy, and sz remain four independent request families. For each axis, raw, physical anchor, and grid0 are projected separately and globally combined as `raw + anchor - grid0` before `(4,24,24,48)` assembly. The Pauli source maps remain `sx=(down,up)`, `sy=(-i down,+i up)`, and `sz=(up,-down)`. The primitive 127-Q and missing 258-Q inventories, `exp(+iQ.r)` transport, determinant-three folding, reverse symmetrization, `1/48` normalization, charge Q0=22, hole Q0=2, physical spin=sigma/2, quadrature, and all numerical thresholds are preserved. `SCIENTIFIC_INPUT_SHA256SUMS.txt`, `NUMERICAL_RUNTIME.json`, the v8 inventory/evidence, v9/v10/v12/v13/v14 evidence manifests, `inventory.py`, `secure_io.py`, `scheduler_workflow.py`, and the full qualified source subtree are byte-identical between v15 and v16.
6. **Scheduler-free mocks cover both collective outcomes.** The 64-thread communicator mock requires all 64 participants at one exact broadcast. In the success case it observes 64 valid records, zero raises, and 64 later-Q advances, each ordered after that rank's success. In the injected rank-0 `ValueError` case it observes one failure broadcast, 64 identical collective raises, zero records returned, and zero later-Q advances. This validates Python control ordering only; it is not a real-MPI runtime test.
7. **The control plane remains held, projection-only, and receipt-bound.** `scheduler_workflow.py` is byte-identical to v15. The v15→v16 `scheduler_control.py` diff is limited to versioned schemas, capsule/job identity, and authorization comment; held submission, one-submission sealing, timeout no-retry, separate release intent/action, two stable positive verification snapshots, runner wait, and running receipt checks remain. The wrapper requests one 64-task full node, account `hmt03`, partitions `regular256,regular6430`, excludes `node037`, uses `--exclusive --mem=0`, has no dependency or explicit time directive, and invokes no OpenMX executable.
8. **The immutable closure and prior evidence are intact.** The capsule root is mode 0555; frozen files are read-only; mutable `runtime/control`, `runtime/output`, and `logs` are isolated mode-0700 directories and empty. v15 is also unsubmitted with empty mutable directories. The v14 artifact remains diagnostic-only (`full385_projection_performed=false`, `physical_claim_authorized=false`) and does not independently authorize the v16 projection or any physical claim.
9. **Fresh static verification passed without scheduler contact or numerics.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and all eight strict checksum-manifest checks passed. Counts were 22 source files, 17 scientific inputs, 11 v8-evidence files, 23 files each for v9/v10/v12/v13 evidence, and 10 v14-diagnostic evidence files. No Slurm command/query, NumPy/SciPy/MPI import, projection, OpenMX execution, or scientific numerical workload was run.

No blocker was found within this immutable v16 projection-only collective-Q0 recovery scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v16/scheduler_workflow.py
```

This command only submits the v16 projection in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v16/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, Q0 gate, full-385 contraction, or output publication was executed. Approval is based on static source/provenance review and scheduler-free mocks; the first real-MPI result remains contingent on the exact runtime Q0 broadcast/gates and every downstream scientific/publication gate. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection.
