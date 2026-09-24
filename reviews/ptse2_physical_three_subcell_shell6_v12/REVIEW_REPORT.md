# External review report — PtSe2 physical three-subcell shell-6 v12 full-Q cache-chain recovery

- **Review ID:** `b27c922b-25b2-455c-a8d5-57333e1954f0`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-19T05:17:15Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v12`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v12`

## Decision

`approved_for_one_v12_projection_only_v8_oracle_full_q_cache_chain_recovery_chain`

This approval is limited to one invocation of the exact no-argument held-submission workflow bound below, followed only by the separately reviewed no-argument v12 projection release helper. It authorizes reuse of the immutable v8 oracle for the preserved v11 projection calculation and accepts the raw→control-variate→final cache chain only as active-lineage provenance. It does **not** authorize an OpenMX rerun, oracle generation, use of the full-Q cache as independently validated HF/vertex-accuracy evidence, changed bytes, argument-bearing invocation, direct `sbatch`/`scontrol`, repeated submission, bypass of release verification, or any scientific conclusion before the production gates pass.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `9ac49fc0f35de0fbfe120441590c46c2a2018a8860ba53e9ff3b52597314b20a` |
| source frozen object | `5ad18cd1a0715190b75bcc2bd4e6fc17e746d841fa432f5ab8a914430fc4a572` |
| source checksum manifest | `4c97b7b17ffdfdce8fd8166a387a6ae71bdcf4325782b572a46a4fcaa2106370` |
| scientific-input checksum manifest | `d184b8412fc271adff209d368f1fccdf07a2df05cae25a1066391d7ac9deaccc` |
| v8 evidence checksum manifest | `033d064cddcc77cf9517315d04907b45089143d2747bde96a8cff34569a2ff82` |
| v8 oracle inventory | `7db564ab9d5462756c1c27d94ead7ec0757ae2917dc1a630f6d0e1503afb8eae` |
| v9 failure-evidence manifest | `bc2be5d8df20c57254f911b38841bec81f3e57e9c16963bdc425464c182a7786` |
| v10 failure-evidence manifest | `8157cac334c0cda8367926ab6dc38b723e8e0c9b6028ba0fe419e2eb6675a579` |
| static checks artifact | `f6c0843ca857de4849a327c7f34f8838f94a33ba84c47e5245710cbb1261816e` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `aa8ef719f200f7da41381162235035c1f5c6e9ad5579d35ba0528db20919b6de` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v12` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v12/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Audit findings

1. **The full raw→CV→final cache lineage is exact.** I independently SHA256-read the three actual cache files, not only their summaries. `raw_cache.npz` is 175,194,418 bytes with SHA256 `c09ed20c32c49f7c1f736e6ca15f49a33ca8f96645a682228dd558b373e1cae1`; `control_variate_cache.npz` is 196,428,114 bytes with SHA256 `bb1d9ecd3dd305c88660528f0e7f3b30c8bb353543daf05b1f6722313ad20883`; and `final_candidate_cache.npz` is 175,194,612 bytes with SHA256 `9737ac87ca863dfc233eea281172898b9eb88c6a308e1daa1e3b0fb3514210e1`. The raw output equals the control-variate input, the control-variate output equals the final-candidate input, and all three summaries carry run ID `d9a08d376a4c88a01cb964df11fa037e63416c57ffc97cc632ca31aa3951f1a5`. The exact summary bytes are pinned by the 17-row scientific-input manifest. Production rank 0 rehashes all three summaries and calls the production validator once before active-frame loading. The cache chain is provenance only: its own summaries say `hf_authorized=false`, and the final projection is construction-enforced rather than independent vertex-accuracy evidence.
2. **The active-lineage derivation is independently complete.** I SHA256-read all 144 point JSON receipts and all 144 point NPZ files (2,537,023,968 NPZ bytes). Every receipt matches the exact eigensystem manifest digest and run ID, is qualified, and carries the freshly observed NPZ digest. Canonical serialization of the ordered records `{index, point_json_sha256, point_npz_sha256}` is 26,819 bytes and gives `c426d2f451f4659c21210d55e734e34dcf88fd25995cf1e6a8b1708d473dfa2d`. Production `load_active()` rehashes each NPZ before calling the same canonical digest helper. Scheduler-free tests reject order, index, receipt-hash, NPZ-hash, summary, schema, run-ID, canonical-JSON, and both cache-edge mutations.
3. **v11 projection science is preserved.** A complete v11→v12 source comparison found the same relative file inventory, 72 byte-identical files among 87 common non-runtime files, and no qualified-source drift. `NUMERICAL_RUNTIME.json`, `contracts.py`, `inventory.py`, `secure_io.py`, the v8 inventory/evidence manifests, the v9/v10 failure manifests, and the complete qualified PtSe2 subtree are byte-identical. The full `project_matched_mpi.py` diff contains only v12 bootstrap/schema bindings plus the added control-variate summary load and raw→CV→final validator edges. `load_active()` is statement-identical; all common scientific/helper function ASTs and 112 stable top-level `run()` statements are identical. Config changes are versioned path/schema/control-plane identities, v11 predecessor bindings, and the added CV/run-ID provenance fields. This is static preservation evidence, not a numerical-equivalence run.
4. **The reused v8 oracle remains inventory-exact.** The current tree has exactly 73 entries: 72 files and `vertex/`. Paths, kinds, sizes, and modes equal `V8_ORACLE_INVENTORY.json`; the 70-row output manifest name set and digest table equal the pinned non-control payload records; the two control hashes and sentinel lineage pass; and the 64 shard names use exact five-digit rank width. The detached v9 review records an independent full read of all 20,634,837,311 bytes with every payload digest matching. I did not repeat that 20.6 GB payload rehash for v12; current immutable mode/size/name/control continuity and production's pre-contraction payload rehash are relied upon. v8 completion authority remains pinned terminal `scontrol` plus sentinel/log evidence, with no separately archived raw `sacct` row.
5. **The workflow is projection-only and fail-closed.** The capsule contains no oracle preparer/finalizer/wrapper/release surface. The sole no-argument workflow submits one held projection job with no dependency; the separate no-argument release helper verifies and releases only the receipt-bound job, and the runner waits for release verification. The wrapper never invokes OpenMX. The exact request is account `hmt03`, partitions `regular256,regular6430`, exclusion `node037`, one node, 64 MPI tasks, `--exclusive --mem=0`, and no explicit time directive. v12 schema, job name, comment, logs, control records, staging, and output paths are isolated. Before approval, `runtime/control/`, `runtime/output/`, and `logs/` were empty.
6. **Fresh static-only verification passed.** `static_check.py`, `pure_stdlib_tests.py`, `bootstrap.py --verify-only`, `bash -n`, and strict checksum verification passed for 22 source rows, 17 scientific-input rows, 11 v8-evidence rows, 23 v9-failure rows, and 23 v10-failure rows. Scheduler mocks covered held submission/release/stable RUNNING, ambiguous submission timeout without retry, and runner-before-verification interleaving. No live scheduler query/action, numerical-package import, MPI projection, OpenMX execution, or scientific numerical workload was performed.

No P0/P1 blocker was found within this immutable v12, projection-only recovery scope.

## Authorized initial workflow

Exactly one initial invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v12/scheduler_workflow.py
```

This command only submits the v12 projection in held state. After its immutable submission receipt exists, the only reviewed release invocation is:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v12/release_projection.py
```

Do not replace either helper with direct scheduler commands and do not repeat the initial submission.

## Review limitations

No live v12 scheduler state was queried and no job was submitted or released. No numerical runtime, MPI projection, physical gate, or output publication was executed. The fresh full cache and active-point reads establish current byte identities but do not convert the cache summaries into HF authorization. The accepted mean-field state remains a local fixed-rank branch, not proof of global-ground-state selection. Final scientific authority remains contingent on the exact v12 runtime gates and a completed v12 output sentinel.
