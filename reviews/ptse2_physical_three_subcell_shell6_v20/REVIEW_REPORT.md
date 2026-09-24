# External review report — PtSe2 physical three-subcell shell-6 v20 Q0 authority split

- **Review ID:** `441ab9ff-fd0a-4a99-873c-05dde324fd5a`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T05:17:34Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v20`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v20`

## Decision

`approved_for_one_v20_full385_cache_q0_charge_fresh_spin_projection_recovery`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. It authorizes projection-only reuse of the immutable v8 oracle with cache-authoritative Q0 charge, fresh separate-axis Q0 spin, fresh nonzero-Q charge/spin, and the collective Q0 verdict. It does **not** authorize OpenMX execution, oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of receipt-bound release verification, or a physical conclusion before all Q0, nonzero-Q, full-385, normalization, convergence, and publication gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `7264c7d8f5b796213b3fa186d8b01ea486b3d0362e55b5b47ac7d62d20c7d9a6` |
| source frozen object | `327b566a7d2d53dba45342a0ebfa4ce7acf3792ef8cb50d7d5cf399f0f3f8305` |
| source checksum manifest | `932c9fda558fd9764c1b3b8e087b260897c2bf9edacf76ca4f8a33ad138a44a3` |
| scientific-input checksum manifest | `a24819c30384d07f0cd4501b690e41f001caab9de8013ac7de4a6c5e9df5034c` |
| observables-v1 evidence manifest | `6dcb40a266a51a44a6b709be6b86a008fb7b2b3b0d4cc1d8da3f2937f3bddd1c` |
| v16 failure-evidence manifest | `ad5c7447364f02e35cb83e557900e678594dd86982afe87b056c573db9701ed9` |
| v19 completed-evidence manifest | `84a4e5baa889240dd0185ca1c8a10b04b2ec0f4ae4387cf83f65460f3ca1f27b` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| v13 failure-evidence manifest | `3aa6368ab1d2f94fed7be58e48446e26eb28fedfa496f6053b957efe1cc4b8d0` |
| v14 Q0-diagnostic evidence manifest | `8ff1ef896521feae6505dc4b49ee1d71c77bfaa4164cbbfddc56b174665debac` |
| static checks artifact | `1ee093149e34ce48cf38f867442775c48cb0942bc9a3f867ecd3daeb95c2c4d9` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v20` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The cache-derived Q0 mapping reproduces the pinned observables-v1 construction.** V20 hash-checks `final_candidate_cache.npz`, its qualifying summary, and `physical_q_chart.npz`; identifies the unique Cartesian Q0; requires exactly 144 admitted `(local,target,source)` rows; requires primitive identity and reduced-k diagonality; and assigns `projected_overlaps[local,:,target,:,source]` into the same determinant-three `(24,24,48)` folded blocks. The cache/chart admitted mask, channel-Q map, physical-Q numerators, rank, run ID, and reverse/C3/Q0 qualification flags are checked before use. The copied observables-v1 source shows the same cache slice and folded target/source placement.
2. **The authority split is explicit rather than hidden.** V20 first constructs the fresh `raw + anchor - grid0` four-axis vertex. It computes the fresh-charge/cache maximum difference, gates it, then copies that fresh vertex and replaces only `vertex[0]` with the cache charge. Axes 1–3 remain fresh spin. The cache charge alone drives authoritative Q0 charge identity, occupied charge, and hole charge; the fresh charge remains diagnostic-only and is separately reported.
3. **The direct-Gram gate is exactly the requested source-bound bound.** The runtime rejects only when `max_abs > 5e-8`, so acceptance is `<=5e-8`; no looser fallback exists. Pinned completed v19 job 512792 observed `4.079806092960325e-08`, below the bound, while the literal cache vertex had Q0 identity maximum `6.149009702369654e-13`. V19 remains diagnostic-only (`full385_projection_performed=false`, `physical_claim_authorized=false`); it justifies this narrow diagnostic bound but does not pre-authorize v20 science.
4. **Fresh Q0 spin remains strict.** sx/sy/sz are independently projected from separate Pauli-transformed source families, with raw, physical-anchor, and grid0 operands reduced before `raw + anchor - grid0`. V20 gates fresh-spin Hermiticity at `2e-12`, spectral excess beyond unit Pauli radius at `1e-9`, and seven-Q replay at `2e-11`. No cache spin array is loaded or substituted. V19 reports exact A/B equality for the spin raw/anchor/grid0 operands and control variates.
5. **The collective Q0 verdict is preserved.** All 64 ranks enter the Q0 reduction and then one exact verdict broadcast. Rank 0 captures cache-charge, fresh-spin, direct-Gram, and seven-Q replay failures; every rank raises on failure, and the Q0 branch returns before any later-Q row can be consumed. Scheduler-free 64-participant success/failure mocks passed. Fatal process/MPI failures remain outside this Python-level guarantee.
6. **Every nonzero-Q vertex remains fresh.** The nonzero-Q branch uses the reduced fresh operands, combines `(raw + anchor) - grid0`, assembles the four charge/sx/sy/sz axes, and performs the inherited reverse pairing. The v16 comparison gates retain fresh nonzero-Q projection, the exact 127+258=385 inventory, seven-Q replay, primitive baseline replay, and downstream real-space/convergence gates.
7. **No normalization drift was found.** The contraction remains `einsum(...)/REDUCED_NK` with `REDUCED_NK=48`; Q0 charge and hole remain 22 and 2; stored spin coefficients remain Pauli expectation values and are divided by 2 only when converted to physical spin density. Reverse symmetrization and determinant-three folding are unchanged. Static AST preservation reports the non-Q0 helpers and all retained scientific gate markers unchanged from v16.
8. **Evidence and control are separated.** Evidence consists of the hash-pinned observables-v1 source/output records, v16 collective Q0 negative evidence, and completed v19 Q0 diagnostic/accounting. Control consists of the read-only v20 source closure, empty isolated mutable directories, mandatory detached approval, one-shot held submission, separate receipt-bound release, and runtime hash rechecks. Neither evidence nor this control approval is a substitute for a successful v20 result.
9. **No hidden execution surface was found.** The capsule exposes no OpenMX/oracle-generation path. The wrapper is projection-only, 64-rank, full-node, dependency-free, and held by the control workflow. Mutable `runtime/control`, `runtime/output`, and `logs` were empty at review time; the capsule root and frozen files had the required read-only modes.
10. **Fresh scheduler-free verification passed.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checksum validation for source, 19 scientific inputs, observables-v1 evidence, v16/v19 evidence, and all inherited evidence manifests passed. No scheduler command/query, NumPy/SciPy/MPI import, projection, OpenMX execution, or scientific numerical workload was performed in this Audit.

No blocker was found within the immutable v20 projection-only recovery scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v20/scheduler_workflow.py
```

This command only submits the v20 projection in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v20/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, Q0 gate, nonzero-Q stream, full-385 contraction, or output publication was executed. Approval is based on static source/provenance review, pinned prior evidence, and scheduler-free mocks. The first v20 real-MPI result remains contingent on every runtime and publication gate. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection.
