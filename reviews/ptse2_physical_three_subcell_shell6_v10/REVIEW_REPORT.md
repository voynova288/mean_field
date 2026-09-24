# External review report — PtSe2 physical three-subcell shell-6 v10 inventory-width recovery

- **Review ID:** `497a06f1-4cac-4f6b-befe-85ee38a6cc4b`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T04:22:41Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v10`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v10`

## Decision

`approved_for_one_v10_projection_only_v8_oracle_recovery_chain`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v10 projection release helper. It authorizes reuse of the immutable v8 oracle for the unchanged projection calculation. It does **not** authorize an OpenMX rerun, oracle generation, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of release verification, or any scientific conclusion before the production gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `91989bd83cbc292634e82e7de4085e45c41aa8d6a242f2a30b9dfef6234555e7` |
| source frozen object | `c34ce4192a3d2f6e8ad0abfa7dcd99cc71178c26bfca427e5e505db52c597c26` |
| source checksum manifest | `de833053da1f62330a2a4f6633d85b1a939e05b78424ec00d599f27c994a2c22` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e1347bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| static checks artifact | `75ee2bdd98470fbb027e4662488d67282d52eecbd814b0351858381d3b9435ca` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v10` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v10/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The v10 rank-width repair matches the immutable v8 inventory.** The production validator now requires exactly 64 shard names `vertex/missing258.rank00000.bin` through `...rank00063.bin`, six fixed non-shard payloads, and two control files. It requires equality among the 70-row output-manifest name set, the 72-file actual filesystem set, and the 72-file pinned inventory. The prior v9 `%04d` expectation cannot match the actual `%05d` names. The capsule pins the unchanged v8 inventory digest and exact modes; the scheduler-free check walked the actual tree and matched all 72 files by path, kind, size, and mode. A fresh 20.6 GB payload rehash was not repeated in this v10 review; v9's detached review records that full independent rehash, and v10 production rehashes every manifest payload before any projection contraction.
2. **The mock exercises the production validator rather than a duplicate implementation.** `pure_stdlib_tests.py` AST-extracts `project_matched_mpi.py::validate_missing_oracle_exact_sets`, executes it on the pinned inventory, checks the exact 70/72/72 counts and 64 five-digit shards, and rejects four-digit, missing-shard, omitted-control, extra-file, and wrong-count mutations. The production `run()` has exactly one direct call to that validator after manifest parsing and filesystem walking and before the MPI barrier that precedes eigensystem loading and contractions.
3. **The v9 failure is pinned without overstating terminal authority.** `V9_FAILURE_EVIDENCE_SHA256SUMS.txt` binds 23 read-only files: the exact detached v9 report/approval, held submission and release/running control records for job 511749, and exact stdout/stderr. The copied review bytes equal the detached v9 originals. Stderr identifies `missing oracle exact publication inventory failed` at the pre-contraction inventory gate and MPI abort. The v9 output directory is empty. No independently archived terminal accounting row is present, so this review makes no terminal-state/accounting claim for job 511749.
4. **No projection-science drift was found.** The scientific-input and numerical-runtime manifests, v8 inventory, `contracts.py`, `inventory.py`, `secure_io.py`, and the complete qualified PtSe2 source subtree are byte-identical to v9. A recursive config comparison found only v10 review/runtime/schema/status/control-plane lineage changes and added v9-failure provenance. All common scientific/helper function ASTs in `project_matched_mpi.py` are identical. Direct statement-level diff of the monolithic `run()` found one changed rank-0 preflight block: the four-digit inline expected-name set was replaced by the pinned five-digit exact-set validator and v8 inventory binding; all subsequent eigensystem, projection, contraction, normalization, partition, gate, and publication statements are unchanged. This is static byte/AST evidence, not a numerical-equivalence run.
5. **The control plane remains projection-only.** The v10 capsule has no oracle preparer/finalizer/wrapper/release surface. The only held submission names one projection wrapper with no dependency; the separate helper releases only that receipt-bound job. The wrapper attests the running projection and invokes the Python MPI projection/recovery path. It never invokes the OpenMX executable. OpenMX-related files occur only as hash-pinned inputs and reader/model names, not as a generation command.
6. **Submission, release, resources, and output remain fail-closed and isolated.** The exact no-argument workflow submits one held job under account `hmt03`, partitions `regular256,regular6430`, excluding `node037`, with one node, 64 MPI tasks, `--exclusive --mem=0`, and no explicit time request. Release requires the separate no-argument helper and stable positive scheduler snapshots; the runner waits for receipt-bound release verification. v10 has distinct schema, comment, job name, logs, control records, staging, and output paths. Its `runtime/control/`, `runtime/output/`, and `logs/` were empty at approval time.
7. **Fresh static-only checks passed.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and strict SHA256 verification passed for 22 frozen-source rows, 14 scientific-input rows, 11 v8-evidence rows, and 23 v9-failure-evidence rows. The scheduler mocks covered held submit/release/two stable RUNNING observations, ambiguous submission timeout with no retry, and runner-before-verification interleaving. No live scheduler command/query, MPI projection, OpenMX execution, or scientific numerical workload was performed.

No P0/P1 blocker was found within this immutable, projection-only v10 recovery scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v10/scheduler_workflow.py
```

This command only submits the v10 projection in held state. After its immutable submission receipt exists, the only reviewed release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v10/release_projection.py
```

Do not replace either helper with direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, scientific gate, or output publication was executed. No fresh full-payload rehash of the 20.6 GB v8 oracle was performed in this v10 review; immutable inventory continuity and fail-closed production rehashing are relied upon as stated above. v8 job 511739 completion remains supported by pinned terminal `scontrol` plus sentinel/log evidence, not a separately archived raw `sacct` row. v9 job 511749 has no pinned terminal accounting row. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection. Final scientific authority remains contingent on the exact v10 runtime gates and completed v10 output sentinel.
