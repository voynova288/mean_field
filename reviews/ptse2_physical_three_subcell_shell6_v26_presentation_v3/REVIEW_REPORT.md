# External review report — PtSe2 physical three-subcell shell-6 v26 presentation v3

- **Review ID:** `468beabf-b8ed-4f83-bc99-92aea28d0c51`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-20T18:03:55Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v26_presentation_v3`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v26_presentation_v3`

## Decision

`approved_for_one_v26_presentation_v3_render`

This approval is limited to one exact no-argument held submission of the frozen
presentation-v3 capsule, followed only by its separate no-argument,
receipt-bound release workflow (including only the workflow's internally
bounded recovery/retry semantics). It authorizes presentation rendering and the
post-render seal only. It does not authorize direct scheduler commands, changed
bytes, argument-bearing invocation, repeated initial submission, HF/projection
recomputation, repair of the preserved v25 mode defect, or new physical claims.

## Audit findings

1. **Output sealing is fail-closed and source-bound.** The sealer validates the
   exact eleven-file preserved-renderer inventory, every payload digest and
   size in `OUTPUT_SHA256.json`, the renderer completion schema, its output
   manifest binding, and its source-manifest binding before creating the v3
   manifest. The v3 manifest records every renderer file's hash, size, and
   intended `0444` mode, plus the manifest and directory modes.
2. **Modes and terminal sentinel ordering are explicit.** The sealer fsyncs and
   fchmods all twelve final output files (the eleven renderer files plus the v3
   manifest) to `0444`, fsyncs and fchmods `runtime/output/` to `0555`,
   then exclusively writes and fsyncs `runtime/PRESENTATION_SEALED` at
   `0444` in the still-private `0700` runtime parent. The wrapper removes
   its temporary cache, disables its EXIT trap, freezes resource evidence, and
   `exec`s the bootstrap/sealer path, so it performs no post-sentinel write.
   A crash producing malformed partial control data remains invalid rather than
   authorizing output.
3. **Preparation and authorization states are correctly separated.** Prep
   validation requires the untouched empty runtime but allows a detached
   approval to exist. Authorized validation requires the same untouched runtime
   and verifies the exact immutable approval, template, freeze, both manifests,
   all source hashes, and the detached report hash. Submission, release,
   rendering, and sealing all pass through the authorized bootstrap path.
4. **Mocked control coverage passed without scheduler access.** The frozen
   pure-stdlib suite passed ten scenarios: zero/one/multiple submission-timeout
   candidates; post-submit query recovery without resubmission; release-timeout
   applied and stable-not-applied/retry paths; post-release verification
   recovery without a second release; runner/release race rejection; nonzero
   release handling; and no-argument helper subprocess exit codes 0/2. All
   scheduler boundaries in state-machine tests are injected.
5. **Renderer and control semantics are preserved.** The renderer is
   byte-identical to v2 at
   `5359299b9194a771a58c0dae631f5269237c94878313e476a8449b67c97c8d67`.
   `resource_gate.py`, `scheduler_workflow.py`, and `release_render.py`
   are byte-identical to v2. The renderer config is identical after removing
   only the v3 namespace/review/scheduler-identity fields. The scheduler control
   is identical after normalizing only its v3 schema and capsule-config
   indirection. Thus formulas, five input identities, grid, tolerances, plots,
   resource gates, and held/release state-machine logic are preserved.
6. **Fresh static validation passed.** Both checksum manifests verified; the
   source root is `0555`; `runtime/`, `runtime/control/`, and
   `runtime/slurm/` are `0700`; output and final sentinel are absent; and
   control/slurm are empty. `static_validate.py --prep-state` and
   `bootstrap.py --verify-prep` passed. The sealer fixture verified final
   file/directory modes and sentinel publication. No scheduler command/query,
   numerical import, rendering, HF, or projection ran during this Audit.

No blocker was found within the narrow one-render presentation-only scope.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v26_presentation_v3/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v26_presentation_v3/release_render.py
```

Do not use direct `sbatch`/`scontrol` commands and do not repeat the initial
submission. If the bounded release workflow records stable `not_applied`, only
its same no-argument release helper may be invoked again under the frozen retry
cap; ambiguous outcomes authorize no retry.

## Limitations

No live Slurm normalization, timeout timing, release behavior, numerical
rendering, or runtime-dependent figure hash was validated. Mock evidence does
not replace live behavior. The source is presentation-only recovered-v26 data,
performs no HF/projection, and preserves the documented v25 mode defect. A
successful sealed render does not repair that defect or add scientific authority
beyond the already recovered finite source.
