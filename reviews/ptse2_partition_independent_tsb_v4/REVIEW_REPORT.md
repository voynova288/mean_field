# External review report — partition-independent TSB v4

- **Review ID:** `878c064a-8841-491f-b87f-310f2e5a4a2f`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-22T12:23:30Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v4`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_partition_independent_tsb_v4`

## Decision

`approved_for_one_partition_independent_tsb_v4_analysis`

This approval authorizes exactly one initial invocation of the frozen V4
no-argument held-submission helper, followed only by its separate no-argument,
receipt-bound release workflow and that workflow's internally bounded release
recovery/retry semantics. It authorizes only the source-bound postprocessing of
the pinned recovered-R6 arrays already approved in V3. It does not authorize
direct `sbatch`/`scontrol`, direct `bootstrap.py --run`, changed capsule bytes,
argument-bearing helper invocations, repeated initial submission, fresh SCF or
PAO-oracle work, coefficient modification, or claims beyond the recorded
finite-R6 fixed-rank-local-branch scope.

## Audit findings

1. **The Slurm `%a` repair is exact for the observed cluster interface.** The
   frozen wrapper retains `%i|%A|%a|%u|%T|%N|%C|%m|%j`; job 516227's pinned raw
   row has `%a=hmt03`. V4 requires field 2 to equal `hmt03`, emits the key
   `account`, and rejects the V3 `N/A`/`array_task_id` interpretation. Bootstrap
   requires the same V4 row schema. The mock suite accepts the exact observed
   account layout and rejects `N/A`, another account, missing node, wrong CPU
   count, wrong job name, and malformed framing.
2. **The predecessor failure boundary is sufficient for this narrow recovery.**
   Hash-bound V3 stdout shows both source manifests passed. Hash-bound stderr is
   the terminal traceback from `resource_gate.py -> parse_squeue`. The frozen V3
   wrapper orders the resource gate before runtime attestation and before
   `bootstrap.py --run`; the resource gate writes `resource_gate.json` only
   after the failing parse. Current read-only inspection found no V3 resource
   gate, runtime attestation, output directory, staging directory, completion
   sentinel, formula-test result, or scientific-analysis artifact. Therefore
   job 516227 failed before numerical imports/formula tests/R6 analysis.
3. **Terminal accounting is corroborative, not capsule-bound.** The capsule
   explicitly records that it does not contain a raw terminal accounting query.
   Independently supplied terminal accounting is `516227 FAILED 1:0 12s
   node038`. That is sufficient to corroborate terminal failure, while the
   pre-numerics conclusion comes from the hash-bound traceback, wrapper order,
   and artifact absence rather than duration or state alone. This approval does
   not claim that the terminal accounting row itself is archived or reproducible
   from capsule bytes.
4. **Scientific semantics are unchanged from the approved V3 capsule.**
   `formula_core.py` and `FORMULA_NOTE.md` are byte-identical. After reversing
   only V4 schema/provenance/output/sentinel names, `formula_tests.py` and
   `analyze_tsb.py` are exactly equal to V3. Thus the Fourier sign and `A^-1`
   reconstruction, determinant-three sector map, physical-spin factor `1/2`,
   Parseval `A^-2` normalization, covariance, pair-symmetrized positive-weight
   `z2`, U/L/T transform, pinned inputs, and claim boundaries have semantic
   parity. This is postprocessing parity, not a new physical validation.
5. **Submission/release/runtime control remains fail-closed.** V4 has a distinct
   job name, authorization comment, schemas, source hashes, output namespace,
   and sentinel. Submission is held and one-shot; release is receipt-bound and
   bounded; the runner requires positive release verification, a live RUNNING
   receipt, resource evidence, and pinned-runtime attestation before loading
   formula tests or numerical packages. Direct `--run` lacks authority unless
   the complete chain is present.
6. **Fresh scheduler-free checks passed.** `static_validate.py --prep-state`,
   `bootstrap.py --verify-prep`, and `bash -n run_analysis.sbatch` passed. The
   21-case pure-stdlib mock suite passed, including account-schema, submission,
   timeout reconciliation, bounded release, live-allocation, and direct-run
   negative cases. Capsule manifests/modes and empty V4 runtime namespaces also
   passed. No scheduler command/query and no NumPy/Matplotlib or scientific
   numerics were run during this review.

No blocker was found within the frozen one-run V4 recovery scope.

## Residual limitations

- The V4 predecessor verifier validates the archived Boolean absence claims and
  pinned failure files, but does not itself re-enumerate the mutable V3 runtime
  namespace. This review independently checked the relevant absences immediately
  before approval; the scientific V4 inputs are separately immutable/hash-bound.
- The squeue parser preserves `%m` as `memory` but does not independently require
  it to equal `0`. This does not weaken the repaired `%a` contract because the
  receipt-bound `scontrol` chain separately requires `MinMemoryNode=0` for the
  same job, but the squeue row alone is not a complete resource attestation.
- `PINNED_RUNTIME.json` pins the interpreter and NumPy/Matplotlib origins and
  RECORD files, not every transitive shared-library byte. Mock checks do not
  prove future live Slurm behavior or numerical output.
- This approval does not promote finite R6 to an infinite-shell result, prove a
  global HF ground state, canonicalize A/B/C partition labels, or promote the
  seven-mode comparison to an order claim.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v4/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v4/release_analysis.py
```

Do not invoke direct `sbatch`, `scontrol`, or `bootstrap.py --run`, and do not
repeat the initial submission. If the bounded release workflow records stable
`not_applied`, only the same no-argument release helper may be invoked again
under the frozen retry cap. An ambiguous release outcome authorizes no retry.
