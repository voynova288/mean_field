# External review report — PtSe2 physical three-subcell shell-6 v13 active-space control-variate recovery

- **Review ID:** `de669090-50f1-49ad-8295-e7355925b26d`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T05:45:47Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v13`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v13`

## Decision

`approved_for_one_v13_projection_only_v8_oracle_active_space_control_variate_recovery`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v13 projection release helper. It authorizes projection-only reuse of the immutable v8 oracle with the corrected active-space control-variate construction. It does **not** authorize an OpenMX rerun, oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of release verification, or any scientific conclusion before all production seven-Q, Q0, reverse, normalization, convergence, and publication gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `a9cad7a5fde08de6f8d0a16920830a9c1328e7b54d5e4986a77e163b97f5b778` |
| source frozen object | `ed89b155df8cb37c9cfd99e0ca14d499334e6e38070a936822470cc2fa03bb98` |
| source checksum manifest | `12da557836bdca4976dbeea991a8d0d7997ccaa4c30ad35efc565b729ad60b56` |
| scientific-input checksum manifest | `d184b8412fc271adff209d368f1fccdf07a2df05cae25a1066391d7ac9deaccc` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| static checks artifact | `19d3d359cc36d3de2ac44866c3f5920b6eb6c3917ac73a951f0d438dfac46c56` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v13` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v13/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v13 correction matches the validated seven-Q control-variate construction.** For a fixed batch of active requests, `project_oracle_blocks_active_batch` is linear in the sparse PAO block sequence. Therefore independently computing `P(raw(Q))`, `P(grid0)`, and `P(anchor)` and returning `(P(raw)-P(grid0))+P(anchor)` is the zero-extended-union realization of `P(raw)+P(anchor)-P(grid0)` even when the three sparse key sets differ. The same request list object is passed to all three projections. This matches the pinned seven-Q implementation, which separately projected raw, grid0, and anchor before combining in active space. At Q0, raw and grid0 are the same shard tuple, and subtraction-first gives exact anchor replay before the pre-symmetrization Q0 identity/Hermiticity gates.
2. **The batched charge/Pauli source convention is preserved.** `requests_for()` constructs one batch whose source columns are charge, `sx=(down,up)`, `sy=(-i down,+i up)`, and `sz=(up,-down)`. `assemble()` slices the resulting `(144,8,32)` batch into four `(24,24,48)` vertices. The scheduler-free mock uses three deliberately different sparse key sets, proves parity with an explicit implicit-zero union, rejects the v12 equal-key premise, requires identical request-object identity, verifies exact Q0 anchor replay, and retains all four axes. Fresh execution reported six production-helper projection calls across its two evaluations (plus one independent expected-anchor call used by the assertion).
3. **Reverse symmetrization and normalization did not drift.** The full v12→v13 production diff changes the failed PAO-space alignment helper to projection-then-combine, adds v12 failure provenance, and versions path/schema/control identities. Reverse pairing remains `V(Q)<-[V(Q)+V(-Q)^dagger]/2` with exact reverse assignment before contraction. `contract()` remains normalized by `1/REDUCED_NK=1/48`; real-space fields remain divided by the physical supercell area; physical spin remains Pauli expectation divided by two. Q0 charge/hole targets remain 22 and 2. The exact seven-Q charge/hole/spin replay and direct-spin Q0 gates remain in the production path with unchanged tolerances.
4. **No unrelated scientific drift was found.** Excluding mutable `runtime/` and `logs/`, v12 has 87 files and v13 has 111: all 87 paths are common, 73 are byte-identical, and the 24 additions are exactly `V12_FAILURE_EVIDENCE_SHA256SUMS.txt` plus its 23 pinned files. The 14 changed common files are the versioned README/template/frozen/check artifacts and the intended bootstrap/config/runner/control/wrapper changes. `SCIENTIFIC_INPUT_SHA256SUMS.txt`, `NUMERICAL_RUNTIME.json`, `contracts.py`, `inventory.py`, `secure_io.py`, `scheduler_workflow.py`, and the entire qualified PtSe2 source subtree are byte-identical. A complete source diff and the fresh AST gate found the other 19 common production/helper functions identical; manual inspection found no hidden scientific change in the admitted monolithic `run()` exclusions.
5. **The v8 oracle continuity is intact.** The current tree has exactly 73 entries: 72 files plus `vertex/`; names, sizes, and modes equal `V8_ORACLE_INVENTORY.json`, including 64 five-digit mode-0444 shards and the exact three stricter mode-0400 metadata files. Both control-file hashes and the sentinel lineage pass. The detached v9 review records an independent full SHA256 read of all 20,634,837,311 oracle bytes. I did not repeat that 20.6 GB payload rehash; v13 rehashes each rank-owned payload inline before contraction, and current inventory/mode/size/control continuity is exact.
6. **The v12 failure evidence is exact and negative-only.** All 23 manifest rows hash-check and are read-only. Job 511758's pinned stderr records `ValueError: anchor/grid0 shard keys differ` on rank 7 followed by `MPI_Abort`; its chain and running receipt bind the approved v12 no-dependency projection. The v12 output directory is empty, and no v12 output or completion sentinel is adopted. No terminal accounting row was added, so the classification remains limited to the pinned stderr and control records.
7. **The workflow remains projection-only and fail-closed.** The v13 capsule contains no oracle preparer/finalizer/wrapper/release surface. The no-argument workflow submits one held projection with no dependency; the separate no-argument helper releases only the receipt-bound job, and the runner waits for release verification. The wrapper invokes no OpenMX executable. The exact request remains account `hmt03`, partitions `regular256,regular6430`, exclusion `node037`, one full node, 64 MPI tasks, `--exclusive --mem=0`, and no explicit time directive. Before approval, v13 `runtime/control/`, `runtime/output/`, and `logs/` were empty.
8. **Fresh static/mock verification passed without scheduler contact or numerics.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checksum verification passed for 22 source rows, 17 scientific-input rows, 11 v8-evidence rows, 23 v9-failure rows, 23 v10-failure rows, and 23 v12-failure rows. Scheduler mocks passed held submission/release/stable RUNNING, ambiguous timeout without retry, and runner-before-verification interleaving. No Slurm command/query, numerical-package import, MPI projection, OpenMX execution, or scientific numerical workload was performed.

No P0/P1 blocker was found within this immutable v13 projection-only active-space control-variate recovery scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v13/scheduler_workflow.py
```

This command only submits the v13 projection in held state. After its immutable submission receipt exists, the only reviewed release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v13/release_projection.py
```

Do not replace either helper with direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live v13 scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, physical gate, or output publication was executed. The v13 fix is approved from derivation, source/AST preservation, exact pinned-artifact continuity, and scheduler-free mocks; final physical authority remains contingent on the exact runtime seven-Q/Q0/reverse/normalization/convergence gates and a completed v13 output sentinel. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection.
