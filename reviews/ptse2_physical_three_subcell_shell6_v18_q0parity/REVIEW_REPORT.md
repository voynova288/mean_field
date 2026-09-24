# External review report — PtSe2 physical three-subcell shell-6 v18 Q0 A/B/C parity diagnostic

- **Review ID:** `80a00108-f3b6-4e67-9379-43d37d9c7dbc`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T04:11:02Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v18_q0parity`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v18_q0parity`

## Decision

`approved_for_one_v18_q0_abc_parity_diagnostic_only`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument release helper. The run is **diagnostic-only** and Q0-only. It may compare the A/B/C constructions, but it may not project later Q, consume the missing-258 payload, perform the full-385/full-Q contraction workflow, run OpenMX, or support a physical claim. Changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, or bypass of receipt-bound release verification are not approved.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `1184057b9fc2c607c000d8abce166f2cabf969fb124a8989fcceee701c1740f7` |
| source frozen object | `c8852f52f7a9fa3c37e8afef9d673dbf1f00d60ac1f4ad1b93c153f990bdbb8d` |
| source checksum manifest | `52500a0af8c1488d08d7e7b585e4ecef971d2a260a27a16698088f4eaa10834f` |
| scientific-input checksum manifest | `acf05d285629b8dca344221d7a53150acf8e234bf755379d6426417387022c70` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| v12 failure-evidence manifest | `9910a9c73aadc96116447e24c089f60413e33c3a19fdddc0748941ccbdd3f7f3` |
| v13 failure-evidence manifest | `3aa6368ab1d2f94fed7be58e48446e26eb28fedfa496f6053b957efe1cc4b8d0` |
| v14 Q0-diagnostic evidence manifest | `8ff1ef896521feae6505dc4b49ee1d71c77bfaa4164cbbfddc56b174665debac` |
| v16 failure-evidence manifest | `ad5c7447364f02e35cb83e557900e678594dd86982afe87b056c573db9701ed9` |
| observables-v1 evidence manifest | `6dcb40a266a51a44a6b709be6b86a008fb7b2b3b0d4cc1d8da3f2937f3bddd1c` |
| full-Q oracle evidence manifest | `08dabf096dd23848410a927e16b51536bf9dcfc4911eccccf6a23b7dfd719325` |
| full-Q Q0 shard lineage | `a3612f2a4badd9a504640b49e47dd850f30bc7a9496ddf66ba17bb558fa0f622` |
| static checks artifact | `9fda3554a4bbd752277568865c7612be15cdac52c68877244dec6b32e5d3f519` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_q0_abc_parity_external_review/v18` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **Literal B now has the requested input provenance.** The runner assigns `active_b_loaded` from the hash-bound copy of observables-v1 `load_active_coefficients`. It obtains metadata from the pinned full-Q oracle through `read_bound_oracle_metadata(...)`, proves a unique chart Q0 at index 0, and passes `read_oracle_shard_blocks(full_q_metadata, q0_index, rank)` as both B raw and B grid-zero blocks. The B call is exactly `active_b_loaded + full_q0`; it no longer receives the v16 active array or primitive-oracle Q0 blocks. The copied loader/build-map/Pauli bodies, observables assignment fragments, and both `Allreduce` source sites are statically tied to the pinned observables-v1 source. This is literal only within the capsule's explicitly bounded Q0 loader/reader/request/reduction/fold/cache-assignment claim, not a claim that the entire historical observables-v1 main program is replayed.
2. **C is separated from B and honestly labeled.** C alone preserves the former v17 hybrid: `active_a + primitive_q0` are passed through the observables adapter. A independently retains the exact v16 functions and uses `active_a + primitive_q0` through its separate-axis path and one root `Reduce`. Output keys, array names, construction descriptions, and uncertainty text consistently call C `common_input_hybrid` and reserve `literal` for B.
3. **A/B/C mismatch coverage is adequate and unthresholded.** All three pairs (`A_vs_B`, `A_vs_C`, `B_vs_C`) are reported for active arrays; every rank-owned Q0 block key inventory, offsets, values, and compatible common-key values; local charge/Pauli projections; global anchor/operand/control-variate projections; mappings; folded vertices; and the five contraction components. Same-shape comparisons include byte equality, max absolute difference, Frobenius norm, and deterministic first differing scalar bytes. Per-construction Q0 charge/Hermiticity/Pauli/contraction diagnostics are also retained. No numerical mismatch threshold or pass/fail scientific classification is applied.
4. **The single `exact_first_mismatch` convenience field has a narrow ordering.** It starts with active arrays and mappings, then local/global projections, folded vertices, and contractions; it does not insert the separately reported rank-local Q0 block comparisons into that one ordered summary. Therefore it must not be interpreted as the first mismatch over the entire provenance DAG. This is not a blocker for the approved diagnostic because the complete Q0 input-block mismatch records are independently published, but downstream interpretation must use those records directly.
5. **Full-Q Q0 shard lineage is closed at the declared boundary.** `FULL_Q_Q0_SHARD_LINEAGE.json` binds the full-Q oracle manifest SHA, unique exact Q0 vector/index, source/input/structure/anchor/operator conventions, and all 64 shard rank/path/size/full-file-SHA records. Bootstrap reconstructs those records from the pinned manifest and rejects any mismatch. At runtime, the exact bound reader rehashes every full shard before the rank-owned Q0 blocks are parsed. Static preparation and this Audit did not rehash the large shards or inspect Q0 numerical payloads; numerical parity remains untested.
6. **Diagnostic-only control is fail-closed.** The runner has one primitive Q0 prefix read, one full-Q Q0 block read per rank, two observables-adapter calls (B and C), and no later-Q projector/streamer, missing-258 consumer, full-385 contraction, or OpenMX call. Publication is allowlisted to `q0_abc_parity_diagnostic.json`, `q0_abc_parity_arrays.npz`, `OUTPUT_SHA256.json`, and sentinel-last `Q0_ABC_PARITY_DIAGNOSTIC_COMPLETE`; sentinel fields deny full-385 execution and physical authority. Full-shard hashing by the bound reader is integrity I/O, not later-Q projection.
7. **Approval and scheduler control remain detached and held.** The capsule itself forbids local approval and requires the external read-only approval path. The no-argument workflow submits one held, dependency-free, 64-rank full-node CPU diagnostic; release is a separate receipt-bound action with stable positive verification. The wrapper invokes no OpenMX executable. Before this approval, runtime control/output and logs contained no files.
8. **Fresh scheduler-free validation passed.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and the strict source checksum manifest passed. Static checks covered literal caller arguments, A/B/C separation, all three mismatch pair labels, Q0-only call graph, 64-shard lineage, held-scheduler mocks, and diagnostic-only flags. No scheduler query/submission/release, NumPy/SciPy/MPI numerical execution, or scientific numerical workload was run.

No blocker was found within the immutable v18 Q0 A/B/C diagnostic-only scope.

## Authorized submit/release sequence

Exactly one initial invocation is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v18_q0parity/scheduler_workflow.py
```

This command only submits the Q0 diagnostic in held state. After its immutable submission receipt exists, the only approved release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v18_q0parity/release_projection.py
```

Do not use direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, Q0 parity diagnostic, or output publication was executed. Approval is based on static source/provenance review and scheduler-free mocks only. The eventual diagnostic remains non-authoritative for full-Q/full-385 correctness, numerical parity before execution, branch optimality, or any physical claim.
