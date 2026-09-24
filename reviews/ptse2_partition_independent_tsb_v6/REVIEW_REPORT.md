# External review report — partition-independent TSB v6

- **Review ID:** `40d54fd7-c3fc-49ca-943d-0aece75b5eaf`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-22T15:29:12Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v6`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_partition_independent_tsb_v6`

## Decision

`approved_for_one_partition_independent_tsb_v6_analysis`

This approval authorizes exactly one initial invocation of the frozen V6
no-argument held-submission helper, followed only by its separate no-argument,
receipt-bound release workflow and that workflow's internally bounded release
recovery/retry semantics. It authorizes only source-bound postprocessing of the
same pinned recovered-R6 arrays previously approved for V5. It does not
authorize direct `sbatch`/`scontrol`, direct `bootstrap.py --run`, changed
capsule bytes, argument-bearing helper invocations, repeated initial submission,
fresh SCF or PAO-oracle work, coefficient modification, or claims beyond the
recorded finite-R6 fixed-rank-local-branch scope.

## Audit findings

1. **The V6 inventory repair is exact and minimal.** V5's configured
   `required_inventory` omitted only `metric = "u^2+uv+v^2 <= 9R^2; u-v=0 mod
   3"` and `missing_count = 258`; the hash/size-pinned `summary.json` contains
   both. V6 adds exactly those two key/value pairs and retains strict full
   Python-dictionary equality. The summary path, size, SHA-256, inherited
   schema, status, and all other scientific input records are unchanged.
2. **The static full-dict gate is on the production authority path.** The new
   pure-stdlib `verify_inherited_summary_inventory()` verifies the immutable
   summary byte identity and digest binding, requires exact input/inherited
   record key sets, and compares the complete inventory dictionary. Bootstrap
   invokes it before detached-review validation, scheduler-module loading,
   submission, or release. The static gate accepted the pinned full dictionary
   and rejected missing-key, added-key, changed-`metric`, and
   changed-`missing_count` variants.
3. **V5 job 516735 is correctly classified as failed before derived TSB
   analysis, not as zero-science or pre-numerics.** The deterministic 46-member
   evidence archive and its binding preserve all 19 control records, seven
   job-evidence files, both terminal logs, exact V5 capsule/review identities,
   the pinned summary, and two identical allocation plus two identical
   step-inclusive accounting snapshots. They establish `FAILED`, exit `1:0`,
   25 seconds on `node034`, a passed resource gate, completed runtime
   attestation, executed unthresholded formula diagnostics, numerical-package
   import and input-validation entry, and the exact terminal
   `ValueError: inherited validated-source inventory mismatch`. The traceback
   reaches `load_inputs` before either reconstruction call; the captured/live
   V5 runtime tree has no output, job output, staging, manifest, completion
   record, or sentinel.
4. **Scientific semantic parity with V5 is exact within the repair scope.** V6
   `formula_core.py` and `FORMULA_NOTE.md` are byte-identical to V5.
   `formula_tests.py`, `analyze_tsb.py`, `PINNED_RUNTIME.json`,
   `resource_gate.py`, and `run_analysis.sbatch` normalize exactly to V5 after
   reversing only their V6 schema/provenance/job/output/sentinel names. V5/V6
   CONFIG analysis settings, all three scientific-input records, authority
   chain, inherited non-inventory fields, and resource-evidence settings are
   exactly equal. This is static lineage evidence, not a fresh numerical
   equivalence result or new physical validation.
5. **Scheduler/control semantics are preserved while authority is fresh.** The
   no-argument submission and release helpers are byte-identical to V5.
   `scheduler_control.py` normalizes exactly to V5 after only V6 scheduler and
   CONFIG schema substitutions. The requested account, partition, exclusion,
   full-node resource shape, absent explicit time limit, held one-shot
   submission, receipt-bound bounded release, timeout reconciliation, running
   receipt, and direct-run rejection are unchanged. V6 has a distinct job
   name, authorization comment, schemas, capsule/source hashes, runtime/output
   namespace, sentinel, and detached-review path; V5 approval cannot authorize
   V6.
6. **Fresh scheduler-free validation passed.** Both source and capsule manifests
   passed strict SHA-256 verification (18 and 21 entries), `bash -n` passed,
   all 22 pure-stdlib mocked control tests passed, `static_validate.py
   --prep-state` passed, and `bootstrap.py --verify-prep` passed. Independent
   normalization checks and archive metadata checks also passed. V6 remained
   with empty `runtime/control/` and `runtime/slurm/`, absent output/staging and
   sentinel, and no bytecode residue. This Audit performed no scheduler command,
   formula execution, numerical import, or scientific computation.

No blocker was found within the frozen one-run V6 inventory-repair scope.

## Residual limitations

- V6 has not executed its formulas or numerical analysis; this approval is a
  static source/evidence/control authorization, not a V6 scientific result.
- The early inventory gate preserves V5's full Python-dictionary equality
  semantics. It is not a general schema validator; later `load_inputs` retains
  the separate inherited schema/status and convention checks. The summary's
  exact pinned SHA-256 prevents a different source dictionary from entering
  this approved run.
- The V5 pre-analysis boundary is supported by the pinned traceback, source
  order, accounting/logs, and complete artifact absence. The terminal
  accounting was not queried again during this no-scheduler Audit; the two
  preparation-time read-only snapshots are the evidence authorized here.
- `PINNED_RUNTIME.json` pins the interpreter and NumPy/Matplotlib origins and
  RECORD files, not every transitive shared-library byte. Mock checks do not
  prove future live Slurm behavior or numerical output.
- This approval does not promote finite R6 to an infinite-shell result, prove a
  global HF ground state, canonicalize A/B/C partition labels, or promote the
  seven-mode comparison to an order claim.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v6/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v6/release_analysis.py
```

Do not invoke direct `sbatch`, `scontrol`, or `bootstrap.py --run`, and do not
repeat the initial submission. If the bounded release workflow records stable
`not_applied`, only the same no-argument release helper may be invoked again
under the frozen retry cap. An ambiguous release outcome authorizes no retry.
