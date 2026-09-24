# External review report — partition-independent TSB v3

- **Review ID:** `86228c93-6c89-48df-b95d-98ab84c9b83e`
- **Reviewer:** OpenAI Codex critical reviewer
- **Reviewed UTC:** `2026-09-22T08:39:59Z`
- **Capsule:** `/data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v3`
- **Detached review directory:** `/data/home/ziyuzhu/Mean_Field/reviews/ptse2_partition_independent_tsb_v3`

## Decision

`approved_for_one_partition_independent_tsb_v3_analysis`

This approval authorizes exactly one initial invocation of the frozen V3
no-argument held-submission helper, followed only by its separate no-argument,
receipt-bound release workflow and that workflow's internally bounded release
recovery/retry semantics. It authorizes source-bound postprocessing of the
pinned recovered-R6 arrays, including the explicitly approved complete
ranked-harmonic diagnostic plot described below. It does not authorize direct
`sbatch`/`scontrol`, direct `bootstrap.py --run`, changed capsule bytes,
argument-bearing helper invocations, repeated initial submission, fresh SCF or
PAO-oracle work, coefficient modification, or claims beyond the recorded
finite-R6 fixed-rank-local-branch scope.

## Audit findings

1. **All previously recorded V3 blockers are closed.** V2 remains byte-identical
   to its frozen manifest. V3 binds the V2 capsule-manifest and source-freeze
   digests, requires detached V3 approval, and carries V3-specific source,
   scheduler, runtime, output, completion, and terminal-sentinel schemas.
2. **The new direct-run mocks genuinely reach the intended fail-closed
   boundaries.** Each fixture reseals a temporary capsule, installs a valid
   predecessor binding and (except in approval-negative cases) a complete
   detached synthetic approval, then invokes the real `bootstrap.py --run`
   subprocess with both required job-scoped resource/runtime arguments.
   Missing/partial/mismatched approval cases fail in approval verification;
   release cases first install a real mocked held submission and fail at the
   absent/partial/mismatched positive-release chain; live cases additionally
   install a complete positive release and fail at the absent/partial/mismatched
   RUNNING receipt. Every case asserts nonzero exit and absence of output and
   the completion sentinel. The boundary-specific stderr assertions prevent a
   shallow required-argument-only test from passing.
3. **Submission and release races are covered with the production control
   implementation.** Same-second scheduler `SubmitTime` is rejected, a
   guaranteed next-second time succeeds, an injected pre-`sbatch` crash leaves
   durable intent and a retry performs zero `sbatch` calls, and a timeout with
   zero candidates can be re-observed without resubmission. Existing tests also
   cover unique/multiple timeout candidates, post-submit query recovery,
   release timeout applied/not-applied paths, bounded retry, post-release
   verification recovery, already-running release races, nonzero release, and
   live-allocation mismatch.
4. **Complete ranked-harmonic plot authorization is explicit.** This review
   explicitly authorizes the retained V2 complete ranked-harmonic diagnostic
   figure for this one V3 run. Its ranked inventory contains every canonical
   `+/-Q` pair, including exact-zero-weight rows, with no top-N, amplitude, or
   display cutoff. Undefined phases for exact-zero relative amplitudes are not
   invented; those rows remain present in the complete rank/weight inventory.
   This is display authorization only, not a scientific acceptance gate.
5. **The phase aggregate is separate from display policy.** The implemented
   `z2 = sum w exp(2 i phi) / sum w` uses both members of every canonical pair
   whenever the mathematical weight `w=|n_Q||ell.S_Q|` is strictly positive.
   Exact-zero-weight channels are counted but excluded because their phase is
   undefined. No numerical amplitude threshold is applied.
6. **Formula and convention audit found no blocking mismatch.** For row-vector
   lattice convention and `M=[[1,-1],[1,2]]`, the stored label map
   `((u-v)/3,(u+2v)/3)` and primitive translation `(2/3,1/3)` are consistent;
   residue zero is exactly the primitive sector. Reconstruction uses the pinned
   `exp(-i Q.r)/A` convention. Parseval budgets carry `A^-2`; physical spin is
   `sigma/2`; covariance is `A^-2 Re sum_Q S_aQ S_bQ*`; the pair phase is
   `arg[n_Q (ell.S_Q)*]`; and the stated U/L/T transform and inverse agree.
   Newly evaluated floating identities remain unthresholded diagnostics rather
   than silently becoming acceptance criteria.
7. **Scope and publication control are fail-closed.** Fourier TSB
   RMS/covariance/phase is labeled partition-independent, while Wigner-Seitz
   U/L/T is separately labeled partition-specific and noncanonical under the
   task owner mapping. The wrapper orders positive release, live RUNNING
   attestation, structural resource evidence, runtime attestation, and then a
   full-chain-revalidating scientific action. Output/completion records bind
   all execution hashes, and the terminal sentinel is published last.
8. **Fresh scheduler-free validation passed.** Both V2 and V3 capsule manifests
   verified; `static_validate.py --prep-state`, `bootstrap.py --verify-prep`,
   shell syntax, the 20-case pure-stdlib mock suite, immutable modes, empty
   control/slurm directories, and absent output/sentinel all passed. No
   scheduler command/query, NumPy/Matplotlib import, formula numerical test,
   R6 analysis, plotting, or scientific output was run during this review.

No blocker was found within the frozen one-run, source-bound V3 postprocessing
scope.

## Exact authorized commands

Exactly one initial submission invocation is authorized:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v3/scheduler_workflow.py
```

After that helper creates the immutable held-job receipt, use only:

```bash
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 -I -S /data/home/ziyuzhu/Mean_Field/results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v3/release_analysis.py
```

Do not invoke direct `sbatch`, `scontrol`, or `bootstrap.py --run`, and do not
repeat the initial submission. If the bounded release workflow records stable
`not_applied`, only the same no-argument release helper may be invoked again
under the frozen retry cap. An ambiguous release outcome authorizes no retry.

## Limitations

No live Slurm behavior, runtime numerical stack, R6 numerical output, or figure
was validated in this review. Mock evidence does not prove live scheduler
behavior. The approved job is postprocessing only: it does not establish a new
SCF solution, PAO-oracle result, infinite-shell theorem, global HF ground state,
canonical A/B/C partition observable, or promoted seven-harmonic order claim.
