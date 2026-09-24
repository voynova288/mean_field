# External review report — PtSe2 physical three-subcell shell-6 v19 Q0 A/B/C spin-operand parity diagnostic

- **Review ID:** `2ee42651-9b0c-4dfc-90d3-de58642b1622`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T04:39:13Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v19_q0parity`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v19_q0parity`

## Decision

`approved_for_one_v19_q0_abc_spin_operand_parity_diagnostic_only`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. The run is **diagnostic-only** and Q0-only. It may compare the A/B/C constructions, but it may not project later Q, consume the missing-258 payload, perform the full-385/full-Q contraction workflow, run OpenMX, or support a physical claim. Changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, or bypass of receipt-bound release verification are not approved.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `b33bf9a4accbac782921c4758bede885e00d3c4bc26996002e6ce0b3af66276a` |
| source frozen object | `570bbb9968b502ca786e05652ca36ead478208971cf3c29a89a69929a8ff0435` |
| source checksum manifest | `a1ae81b51d3acd4fd18ac6a0c4aeb4b0ca72fd9b1c874899d14b73bffb37f7ca` |
| scientific-input checksum manifest | `acf05d285629b8dca344221d7a53150acf8e234bf755379d6426417387022c70` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| v13 failure-evidence manifest | `3aa6368ab1d2f94fed7be58e48446e26eb28fedfa496f6053b957efe1cc4b8d0` |
| v14 Q0-diagnostic evidence manifest | `8ff1ef896521feae6505dc4b49ee1d71c77bfaa4164cbbfddc56b174665debac` |
| v16 failure-evidence manifest | `ad5c7447364f02e35cb83e557900e678594dd86982afe87b056c573db9701ed9` |
| v18 failure-evidence manifest | `5d5dfe810f180702a61b4767cf62d803ac329adbfde804c28906357644fe81e0` |
| observables-v1 evidence manifest | `6dcb40a266a51a44a6b709be6b86a008fb7b2b3b0d4cc1d8da3f2937f3bddd1c` |
| full-Q oracle evidence manifest | `08dabf096dd23848410a927e16b51536bf9dcfc4911eccccf6a23b7dfd719325` |
| full-Q Q0 shard lineage | `a3612f2a4badd9a504640b49e47dd850f30bc7a9496ddf66ba17bb558fa0f622` |
| static checks artifact | `00f2090f082f1fa7c742287d3ed5c68538ef98bbf35c06b6628eddb9e62a156b` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_q0_abc_spin_operand_parity_external_review/v19` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v18 row/operand failure is correctly diagnosed and narrowly repaired.** `observables_q0_literal` returns spin stacks with exact shape `(axis=3, operand=3, row=144, active=8, active=8)`, with B/C operand order `(raw, grid0, anchor)`. V18 selected only `axis`, passed `(3,144,8,8)` to a row aligner requiring 144 entries on axis 0, and failed with `ValueError: row/value cardinality mismatch`. V19 adds `align_spin_stacks_to_primitive`, requires `(3,3,144,8,8)`, selects each `(axis, operand)` slice as `(144,8,8)`, aligns it, and restacks without transposing the axis or operand inventories.
2. **The A/B comparison convention remains correct.** A stores operands as `(raw, anchor, grid0)`, while B/C store `(raw, grid0, anchor)`. The local and global raw-operand reports retain the explicit B/C permutation `[[0,2,1]]` when comparing to A. The global control-variate diagnostics retain `raw + anchor - grid0`, implemented for B/C as indices `0 + 2 - 1`. Thus the repair fixes row placement without silently changing operand semantics.
3. **The production-shape mock is discriminating within its static scope.** `pure_stdlib_tests.py` AST-extracts the real helper, supplies a production-shape mock `(3,3,144,8,8)`, requires exactly nine aligner calls in `(axis,operand)` traversal order, rejects any aligner input other than `(144,8,8)`, and requires the final shape `(3,3,144,8,8)`. It also checks all four local/global B/C assignments use the helper. This is a static shape/indexing test, not a NumPy/MPI numerical replay.
4. **Literal B and hybrid C computations are preserved.** The full v18-to-v19 runner diff changes only version/provenance/publication identities, the four local/global spin alignment call sites, and the added helper. AST gates bind `observables_q0_literal` exactly to v18 and preserve the observables-v1 assignment fragments and two `Allreduce` sites. Caller checks retain B as `active_b_loaded + full_q0` from the pinned full-Q reader and C as `active_a + primitive_q0`. C remains explicitly labeled `common_input_hybrid`; this review does not promote it to a literal observables-v1 construction.
5. **Global diagnostic preservation is adequate.** The local report, global differences, mappings, folded-vertex differences, contraction differences, and Q0 metrics assignment formulas are AST-equal to v18. The repaired helper is used for both local and root-global spin stacks. All A/B/C pairs remain present for charge, each spin operand stack, control variates, mappings, folded vertices, and contractions, with no mismatch threshold or scientific pass/fail classification.
6. **One inherited summary caveat remains.** `exact_first_mismatch` does not insert the separately published rank-local Q0 block inventory/value comparisons into its ordering. It therefore is not the first mismatch over the complete provenance DAG. The complete block records remain independently published and must be inspected directly; this is not a blocker for the diagnostic-only run.
7. **V18 negative evidence is pinned and not adopted as science.** The 37-file v18 evidence manifest passed. The copied v18 runner, stderr/stdout, detached review, and approval match their source bytes. The stderr contains the report-adapter cardinality exception; stdout is empty; no v18 diagnostic payload, manifest, or completion sentinel is adopted. The evidence proves the observed pre-publication exception and authorized running lineage only. Because no terminal accounting query was preserved, it does not claim a terminal Slurm state or any A/B/C numerical verdict.
8. **Diagnostic-only control remains fail-closed.** The runner has one primitive Q0 prefix read, one bound full-Q Q0 block read per rank, two observables-adapter calls, and no later-Q projector/streamer, missing-258 consumer, full-385 contraction, or OpenMX call. Publication is restricted to the versioned v19 diagnostic JSON, arrays NPZ, output manifest, and sentinel-last completion record; the sentinel denies full-385 execution and physical authority.
9. **Submission control remains detached, one-shot, and held.** V19 has an empty mutable control/output/log surface at review time. The no-argument workflow submits one dependency-free held 64-rank full-node CPU diagnostic. Release is a separate no-argument, receipt-bound helper. Scheduler mocks cover held submission, ambiguous-timeout sealing, stable positive release verification, and runner waiting. No live scheduler action was performed in this Audit.
10. **Fresh scheduler-free verification passed.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checksum validation for source, scientific inputs, v18 evidence, and full-Q oracle evidence all passed. These checks imported no NumPy/SciPy/MPI and performed no scientific numerics.

No blocker was found within the immutable v19 Q0 A/B/C spin-operand diagnostic-only scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v19_q0parity/scheduler_workflow.py
```

This command only submits the Q0 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v19_q0parity/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No NumPy/SciPy/MPI numerical runtime, Q0 parity calculation, or output publication was executed. The static production-shape mock validates operand/row indexing but does not establish real-MPI behavior or numerical A/B/C parity. The large full-Q shard payloads were not rehashed during this Audit; the bound runtime reader performs those integrity checks before parsing Q0 blocks. Approval is non-authoritative for later-Q/full-385 correctness, physical observables, or global-ground-state selection.
