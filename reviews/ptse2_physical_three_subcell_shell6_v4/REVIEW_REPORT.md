# External review report — PtSe2 physical three-subcell shell-6 v4

- **Review ID:** `a05e7526-3165-4ae5-a9a1-448249b42003`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-18T16:31:59Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v4`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v4`

## Decision

`approved_for_one_digest_loaded_timeout_reconciled_held_oracle_afterok_projection_chain`

This approval is limited to the exact no-argument held scheduler workflow bound below. It is not approval of modified bytes, argument-bearing invocation, direct production entry points, manual scheduler commands, scientific output, convergence, a global-ground-state claim, or the statement “shell6 sufficient under declared thresholds.” The latter remains contingent on the unexecuted production gates.

## Required identity bindings

| Field | SHA256 / exact value |
|---|---|
| external review template | `faf793c6d27f21e863f9d272cb3a359825869a617828c51b6b4c2c000855e7f1` |
| source frozen object | `d226205997ff61218e63f817f1f9650864b0a9630382c4c2e4095f2240610fcd` |
| source checksum manifest | `118908797b650ec358cc69fa986d3ca37fec5931493c33942617bf0d8b1c4185` |
| scientific-input checksum manifest | `6d80b839fbfa2a7e1347bf1728c52c30fad30761f5501f71e20ef985162f3e88` |
| static checks artifact | `3f7978afaf344e577a9fceb2a50a513a190803e3c9a4eb2ae0bea01e892c4b7a` |
| numerical runtime artifact | `e2e503892b7d991378cd04f101c974cb749346ffbb62b250a274e2db168b7188` |
| scheduler workflow | `90ae77354891860adf432dffc6e6575a6bc42680d66ee75e518b8d20f1f8b32a` |
| config (supporting path check) | `335e6f12f46c072eda4d46c4f0c3b9db48944a4d4293b253ce49f043e4b0090d` |
| approval schema | `ptse2_physical_three_subcell_shell6_external_review/v4` |
| workflow interpreter | `/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11` |
| Python flags | `-I -S` |
| workflow script | `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v4/scheduler_workflow.py` |
| workflow arguments | `[]` |

The approval JSON separately hash-binds this report and copies the exact workflow object from `EXTERNAL_REVIEW_TEMPLATE.json`.

## Re-audit findings

1. `SOURCE_SHA256SUMS.txt` passed strict verification for all 27 frozen source entries.
2. `SCIENTIFIC_INPUT_SHA256SUMS.txt` passed strict verification for all 14 pinned inputs. The aggregate v4 identities above remain exactly those embedded in `SOURCE_FROZEN.json` and `EXTERNAL_REVIEW_TEMPLATE.json`.
3. Both Slurm wrappers passed `bash -n`. No wrapper has an explicit time directive; the pinned account, partition list, exclusion, 64 tasks, exclusive-node request, and `mem=0` contract were retained.
4. The pure-stdlib static/mock audit returned the exact v4 schema and `passed_static_mock_not_reviewed_not_submitted`, with `execution=false`, `submission=false`, `scheduler_queries=false`, and `scientific_numerics=false`.
5. The six previously reported blocker categories remain closed: exact source-closure binding; standard oracle null-dependency versus exact projection `afterok` handling; bootstrap-only scheduler mutation loading; exact chain/receipt/evidence validation before release; durable timeout reconciliation with bounded retry; and mock coverage of submission/release failure windows plus independent regional Fourier normalization checks.
6. The static lineage gate retained eight byte-identical science files and the named v3-to-v4 scientific AST identities. This is lineage evidence, not fresh numerical validation.
7. Capsule mode is `0555`; frozen files are manifest-bound; mutable `runtime/`, `runtime/control/`, `runtime/output/`, and `logs/` directories are `0700` and empty. No bytecode residue was found.
8. The configured detached approval path is exactly `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v4/REVIEW_APPROVAL.json`; it is outside the capsule and was absent at audit start.

No additional static/control-plane blocker was found within this review scope.

## Authorized workflow

Exactly one invocation shape is approved:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v4/scheduler_workflow.py
```

The script itself enforces `len(sys.argv) == 1` and enters the digest-bound bootstrap submission action. It submits the oracle held and the projection held with exact `afterok:<oracle_job_id>` lineage; it does not release either job.

## Review limitations

No scheduler command or query was made. No NumPy, SciPy, mpi4py, OpenMX, MPI projection, HF solve, production publication, or other scientific numerical workload was run. The review establishes static identity and control-plane authorization only.
