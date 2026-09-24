# External review report — partition-independent TSB v5

- **Review ID:** `791d954f-5810-49d3-a011-caadc068ad99`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-22T13:49:45Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v5`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_partition_independent_tsb_v5`

## Decision

`approved_for_one_partition_independent_tsb_v5_analysis`

This approval authorizes exactly one initial invocation of the frozen V5
no-argument held-submission helper, followed only by its separate no-argument,
receipt-bound release workflow and that workflow's internally bounded release
recovery/retry semantics. It authorizes only source-bound postprocessing of the
pinned recovered-R6 arrays previously approved for V4. It does not authorize
direct `sbatch`/`scontrol`, direct `bootstrap.py --run`, changed capsule bytes,
argument-bearing helper invocations, repeated initial submission, fresh SCF or
PAO-oracle work, coefficient modification, or claims beyond the recorded
finite-R6 fixed-rank-local-branch scope.

## Audit findings

1. **The locale repair is explicit and command-local.** The frozen wrapper uses
   `LC_ALL=C` with absolute paths for `mpstat`, `vmstat`, `free`, `squeue`,
   `hostname`, and `date`. The resource captures therefore no longer inherit
   the Chinese locale that caused V4's aggregate-average misclassification.
2. **The mpstat parser enforces the declared 3+1 contract.** It accepts only a
   consistent C-locale 12-hour or 24-hour timestamp style for aggregate
   interval rows, requires exactly three timestamped `all` rows and exactly one
   exact `Average: all` row, and requires exactly ten numeric, finite percentage
   fields in `[0,100]` on every aggregate row. The average is recorded
   separately as `cpu_busy_percent_average`; the interval list and observed
   average-row count are separately emitted and revalidated by bootstrap.
3. **The scheduler-free mocks exercise both positive layouts and fail-closed
   malformed layouts.** Fresh execution passed all 22 mock cases. The mpstat
   cases accept 12-hour and 24-hour C-locale layouts and reject localized or
   unexpected averages, missing/extra intervals, duplicate averages,
   short/long rows, nonnumeric/nonfinite/out-of-range metrics, invalid AM/PM
   timestamps, and mixed timestamp styles.
4. **V4's failure is bound as pre-numerics evidence.** The V5 predecessor record
   and deterministic 35-member evidence archive bind V4 source/capsule
   identities, detached approval/report, all extant control records, raw
   mpstat/vmstat/free/squeue evidence, failed resource gate, terminal stdout and
   stderr, and the raw C-locale terminal accounting row. The gate records three
   true intervals plus one localized average counted as a fourth sample and
   `passed=false`; stderr is exactly `structural resource evidence gate failed`
   and contains no Python traceback. The frozen V4 wrapper places that gate
   before runtime attestation and `bootstrap.py --run`. Independent read-only
   inspection again found no V4 runtime attestation, output, staging, or
   completion sentinel.
5. **Scientific semantic parity is exact within the recovery scope.** V5
   `formula_core.py` and `FORMULA_NOTE.md` are byte-identical to V4. Reversing
   only V5 schema/provenance/output/sentinel names makes `formula_tests.py` and
   `analyze_tsb.py` exactly equal to V4. V4/V5 config differences are limited
   to the new authority/provenance namespace, predecessor binding, and the new
   one-average resource-evidence count; the inputs and analysis settings are
   unchanged. `scheduler_workflow.py` and `release_analysis.py` are
   byte-identical, while `scheduler_control.py` normalizes exactly to V4 after
   the V5 schema-name substitutions. This establishes parity, not a new
   physical validation.
6. **Authority and control remain fail-closed.** V5 has a distinct job name,
   authorization comment, schemas, source/capsule hashes, output namespace,
   sentinel, and detached review path. Submission remains held and one-shot;
   release remains receipt-bound and bounded. The runner requires the approved
   detached review, full release/running evidence, the structural resource gate,
   and runtime attestation before formula tests or numerical imports.
7. **Fresh non-scheduler validation passed.** Both
   `static_validate.py --prep-state` and `bootstrap.py --verify-prep` passed;
   both source and capsule manifests verified (18 and 21 entries), and
   `bash -n run_analysis.sbatch` passed. V5 remained with empty control/Slurm
   namespaces and absent output/sentinel. This Audit performed no scheduler
   command or numerical/scientific computation.

No blocker was found within the frozen one-run V5 recovery scope.

## Residual limitations

- The resource observations are deliberately unthresholded. The structural
  gate proves framing/cardinality and allocation identity, not CPU or memory
  performance or scientific sufficiency.
- The mpstat parser enforces row shapes, counts, timestamp style, and metric
  domains; it does not assert timestamp monotonicity or require the average row
  to occur after the intervals. Those properties are not part of the declared
  structural-count contract, and the wrapper invokes standard C-locale
  `/usr/bin/mpstat 1 3` directly.
- The V4 zero-science conclusion relies on hash-bound terminal evidence,
  wrapper order, and archived/current artifact absence, not on elapsed time
  alone. Current V4 absence checks concern a mutable runtime namespace, while
  all extant predecessor records used by V5 are separately copied into the
  immutable archive.
- `PINNED_RUNTIME.json` pins the interpreter and NumPy/Matplotlib origins and
  RECORD files, not every transitive shared-library byte. Mock checks do not
  prove future live Slurm behavior or numerical output.
- This approval does not promote finite R6 to an infinite-shell result, prove a
  global HF ground state, canonicalize A/B/C partition labels, or promote the
  seven-mode comparison to an order claim.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v5/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v5/release_analysis.py
```

Do not invoke direct `sbatch`, `scontrol`, or `bootstrap.py --run`, and do not
repeat the initial submission. If the bounded release workflow records stable
`not_applied`, only the same no-argument release helper may be invoked again
under the frozen retry cap. An ambiguous release outcome authorizes no retry.
