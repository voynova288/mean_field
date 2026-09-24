# v7 oracle release-verification race

- UTC observation date: 2026-09-19
- Oracle 511729 was released onto node017 and failed before OpenMX/numerical work.
- Failure: runner immediately required `runtime/control/oracle.RELEASE_VERIFICATION.json`, while the release helper was still polling to create that positive verification. The job began before the helper observed two stable positive snapshots, so the runner raised FileNotFoundError and exited in 0 seconds.
- Release evidence present: intent, precheck, normal return, and `oracle.RELEASE_ACTION.01.json`; four verification polls were unable to seal the final verification before terminal failure.
- Projection 511730 remained held and was cancelled explicitly.
- Accounting:
  - `511729|pt7p_n2B1_missing258_v7|FAILED|1:0|...|Elapsed=00:00:00|regular6430|node017`
  - `511730|pt7p_n2B1_match385_v7|CANCELLED by 1091|0:0|...|Elapsed=00:00:00|regular256,regular6430|None assigned`
- No OpenMX/MPI scientific calculation ran.
- Required v8 fix: runner must bounded-wait for the receipt-bound release verification/action chain before validating and continuing, as specified by the sealed-held workflow; release helper verification window must allow startup transition and terminal preflight outcomes without racing the runner.
