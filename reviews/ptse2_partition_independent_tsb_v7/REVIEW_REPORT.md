# External review report — partition-independent TSB v7

- **Review ID:** `57a75846-4ab5-4ed6-9a03-8dc38bb3d5ee`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-22T17:02:44Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v7`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_partition_independent_tsb_v7`

## Decision

`approved_for_one_partition_independent_tsb_v7_analysis`

This approval authorizes exactly one initial invocation of the frozen V7
no-argument held-submission helper, followed only by its separate no-argument,
receipt-bound release workflow and that workflow's internally bounded release
recovery/retry semantics. It authorizes source-bound postprocessing of the same
pinned recovered-R6 arrays used by completed V6, with only the requested
partition-specific noncanonical Wigner-Seitz U/L/T correction. It does not
authorize direct `sbatch`/`scontrol`, direct `bootstrap.py --run`, changed
capsule bytes, argument-bearing helper invocations, repeated initial submission,
fresh SCF or PAO-oracle work, coefficient modification, or claims beyond the
recorded finite-R6 fixed-rank-local-branch scope.

## Audit findings

1. **The U/L/T correction is algebraically correct and internally complete.**
   V7 implements `U=A+B+C`, `L=(A-B)/2`, and `T=(A+B-2C)/6`. Substitution gives
   the stated inverse `A=U/3+L+T`, `B=U/3-L+T`, `C=U/3-2T` component by
   component. The correction is present consistently in `formula_core.py`, the
   structural test contract, analysis JSON fields, report text, and the
   partition-specific CSV path. The obsolete V6 `net_3U_hbar` alias is removed.
2. **The scope and supersession boundary are correct.** The task-defined mapping
   remains `A=owner0`, `B=owner2`, `C=owner1`; the source still does not establish
   canonical A/B/C owner labels. Therefore U/L/T remains partition-specific and
   noncanonical. V6's alternative U/L/T basis and quantities derived from it are
   superseded, while its A/B/C source vectors and `angle(A,B)` remain valid and
   unchanged. V7 does not relabel this correction as partition-independent.
3. **The expected corrected diagnostics are consistent with the pinned V6
   vectors and axis.** The frozen capsule records `|U|≈7.2821e-4 hbar`,
   `|T|/|L|≈8.39966e-2`, `angle(A,B)≈170.480181322 deg`, signed-axis raw angle
   `≈179.993166399 deg`, and sign-invariant acute axis angle `≈0.00683360 deg`.
   Their directions and scales follow from the corrected definitions: U is the
   near-canceling vector sum, L is half the A-B contrast, and T is the residual
   symmetric contrast against C. Under the explicit no-numerics instruction,
   these are approved as source-bound expected V7 diagnostics, not claimed as a
   fresh V7 computation or production output.
4. **The angle conventions are explicit and non-conflicting.** `angle(A,B)` is
   the ordinary vector angle in `[0,180]`. The raw L/axis quantity is likewise
   an unsigned vector angle after fixing the covariance eigenvector sign by the
   documented positive-x convention; “signed-axis” refers to that sign-fixed
   representative, not to an oriented signed angle. The physical covariance
   axis is unoriented, so `min(theta,180-theta)` correctly gives the
   sign-invariant acute line angle in `[0,90]`. Both values are reported rather
   than silently replacing one with the other.
5. **Partition-independent Fourier formulas and prior results are preserved.**
   Static AST comparison confines `formula_core.py` changes to `ult_decompose`
   and `ult_inverse`. After V7-to-V6 namespace normalization,
   `analyze_tsb.py` changes only `ult_analysis`, `output_report`, and `main`;
   the main changes are versioned output/sentinel metadata plus the explicit
   supersession record. CONFIG analysis settings, scientific inputs, authority
   chain, inherited source contract, and resource evidence are exactly equal to
   V6. No coefficient, reconstruction, determinant-three split, physical-spin
   normalization, Parseval, covariance, z2, phase-pair, field-array, plot-data,
   SCF, or PAO-oracle formula changes. The V6 output tree remains byte-bound by
   its original output manifest; V7 does not mutate it.
6. **Completed V6 job 516880 is strongly bound, with the stated accounting
   limit.** `PREDECESSOR_COMPLETION_516880.json` and its deterministic 23-member
   evidence archive bind the V6 capsule/review identities, submission and live
   allocation records, resource gate, runtime attestation, stdout, zero-byte
   stderr, completion sentinel, completion record, output manifest, and every
   output path/hash/size. Scheduler-free bootstrap verification rechecked those
   live identities and the archive payloads. This establishes sealed completion
   and stored execution accounting on account `hmt03`, partition `regular256`,
   node `node034`; it is intentionally not a fresh terminal `sacct` claim.
7. **Fresh authority and control are fail-closed.** V7 has distinct schemas,
   hashes, review path, job name `ptse2_tsb_v7_c9e3acc1`, authorization comment,
   output/staging names, and `ANALYSIS_COMPLETE_V7`; V6 approval cannot authorize
   V7. Source and capsule manifests passed strict SHA-256 verification,
   `bash -n` passed, scheduler-free bootstrap prep verification passed, and all
   22 pure-stdlib control mocks passed, including held submission, timeout
   reconciliation, bounded release, running receipt, approval/release/live
   evidence rejection, and direct-run rejection. Runtime control/slurm remained
   empty and output/staging/sentinel remained absent. No scheduler command,
   scientific numerical import, production formula execution, or V7 numerical
   analysis was performed in this review.

No blocker was found within the frozen one-run V7 correction scope.

## Residual limitations

- V7 has not produced a numerical output. The listed values are expected
  diagnostics derived by the frozen source path and remain unthresholded until
  an authorized run publishes them.
- Exact preservation here means the V6 partition-independent formulas and the
  existing V6 output bytes remain unchanged and hash-bound. Versioned V7 JSON,
  report, archive schema strings, figure metadata, and filenames are not
  expected to be byte-identical to V6.
- The covariance principal eigenvector sign is conventional. Only the acute
  axis angle is invariant under eigenvector sign reversal; neither angle is
  protected if the leading covariance eigenspace is physically degenerate.
  The inherited V6 covariance diagnostics, not a new V7 tolerance, delimit that
  interpretation.
- No fresh terminal scheduler/accounting query was made. Mock checks do not
  prove future live Slurm behavior, and the pinned runtime does not cover every
  transitive shared-library byte.
- This approval does not promote finite R6 to an infinite-shell result, prove a
  global HF ground state, canonicalize A/B/C partition labels, or promote the
  seven-mode comparison to an order claim.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v7/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v7/release_analysis.py
```

Do not invoke direct `sbatch`, `scontrol`, or `bootstrap.py --run`, and do not
repeat the initial submission. If the bounded release workflow records stable
`not_applied`, only the same no-argument release helper may be invoked again
under the frozen retry cap. An ambiguous release outcome authorizes no retry.
