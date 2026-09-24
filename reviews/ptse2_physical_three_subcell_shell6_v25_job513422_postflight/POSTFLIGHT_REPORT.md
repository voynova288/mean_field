# Detached postflight audit — PtSe2 shell-6 v25 job 513422

- **Reviewed artifact:** `results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell6_v25`
- **Producer job:** `513422`
- **Postflight recomputation job:** `513790` (`test`, 56 CPUs reserved, one small NumPy process; `COMPLETED`, `0:0`)
- **Reviewed UTC:** `2026-09-20T15:42:19Z`
- **Scope:** immutable outputs and the fixed-rank `n2_B1`, epsilon-10, determinant-3 local HF branch only

## Decision

**Scientific payload: internally consistent and all declared numerical gates pass. Publication closure: nonconforming in one mode check.**

The final data support only **“shell 6 sufficient under the declared finite thresholds.”** They do not prove infinite-shell convergence, an atomic-resolution density, a unique partition-independent subcell observable, or global-ground-state selection.

The strict vector hypothesis `(N_A≈N_B, M_A=-M_B, M_C=0)` is **not supported as a partition-independent statement**. A Wigner–Seitz (WS) partition supports its **charge and longitudinal-spin projection** after a cyclic relabeling, but the third region has a resolved transverse moment and is not a zero vector. The parallelogram partition does not exhibit the same three-region pattern.

## Critical publication-contract finding

All hashes, exact inventories, sentinel bindings, hardlink identities, read-only write bits, and committed-directory modes verify. However, every non-sentinel final and checkpoint payload is mode **`0400`**, not the source contract's required **`0444`**. `secure_io.verify_published_tree(...)` and `pure_stdlib_tests.py` explicitly require `0444`; the frozen `publish_allowlisted_tree(...)` only changes a file to `0444` if a write bit was initially present, while `write_npz`/`write_json` create files as `0400`. Thus the actual publication would fail its own verifier.

This is a real protocol mismatch, not a numerical discrepancy. Mode `0400` still has no write bits and all bytes hash correctly, so it does not invalidate the recomputed science. Per instruction, v25 was not modified. Any repair must be a new versioned closure, not an in-place chmod or edit of v25.

## Scheduler and accounting

Live `sacct` on `login002` records:

- job `513422`, account `hmt03`, partition `regular256`, node `node039`;
- `COMPLETED`, `ExitCode=0:0`;
- submitted `2026-09-20T20:47:37+08:00`, started `20:47:41`, ended `23:07:15`;
- elapsed `02:19:34`, 64 CPUs, one node, requested memory `250G`;
- batch MaxRSS `158016K`; MPI step MaxRSS `100314288K`.

The immutable held/released/running receipt chain binds job 513422, and the summary independently reports 64 ranks on node039. `scontrol` no longer retains the job, so terminal authority comes from `sacct`, not a current `scontrol` record. Exact rows are in `SACCT_JOB513422.txt`.

## Sentinel-last, checksums, and checkpoint/final separation

- Frozen v25 source manifest: every row passes; `SOURCE_FROZEN.json` SHA256 is `e74702e8b51a64f7e2ec06dc9e15d88e94bb2e7a1328c164f2bc990e18fddbd5`.
- Final exact inventory is the six `matched_R*.npz` files, `summary.json`, `OUTPUT_SHA256.json`, and `COMPLETE`; no foreign entry exists.
- All final payload hashes match `OUTPUT_SHA256.json`; the sentinel correctly binds the manifest, summary, `matched_R6`, checkpoint sentinel, source, state, and scheduler-chain receipt.
- Checkpoint NPZ/JSON/manifest hashes and checkpoint sentinel bindings all match.
- Final and staging payloads are the same hardlink inodes. Source order links all payloads first and the sentinel last, then fsyncs the committed directory and parent. The exact namespace and sentinel claims verify. Filesystem timestamps have only one-second granularity, so they corroborate checkpoint-before-final (23:07:10 vs 23:07:12) but are not, by themselves, an execution-order proof.
- The checkpoint is separately committed, declares `diagnostic_only=true` and `physical_authority=false`, and precedes the late gates. Its seven scientific arrays are element-for-element identical to the final R6 arrays, but final authority comes only from the later final sentinel.

## NPZ/summary and 385-channel consistency

Independent code in `recompute_matched_r6.py` imports no v25 production module. From the six NPZ files it reproduces:

- exact nested channel counts `13, 43, 97, 169, 271, 385`;
- 385 unique determinant-3 labels, sectors `127/129/129`, exact reverse closure and C3 closure;
- exact determinant-3 map from `(u,v)` to saved physical labels;
- every R1–R6 NPZ as the exact nested subset of R6;
- all scalar shell metrics and every recorded parallelogram/WS regional integral with maximum discrepancy **0.0**;
- checkpoint-to-final array identity for labels, shells, charge, hole, and all three Pauli-spin coefficient arrays.

## Fourier convention and early gates

The convention chain is coherent:

1. Oracle item `Q` is `V_Q=C_target† exp(+iQ·r) C_source`, with `target=source+Q` modulo reciprocal lattice.
2. `contract_stored` conjugates `V_Q`; that contraction is therefore stored at physical coefficient label `-Q`.
3. Charge, hole, and all spin components are moved together to the reverse slot exactly once.
4. Real space is reconstructed as `f(r)=A^-1 Σ_Q f_Q exp(-iQ·r)`.

Independent direct comparison at the already physical saved labels—without a second reversal—reproduces the seven-Q maxima: charge/hole `6.729381390865313e-8`, sx `1.1979082467629634e-8`, sy `7.499314893022193e-9`, sz `1.413565901166216e-8`. Reverse residuals are charge `5.7902148899581444e-15`, hole `4.335559509131367e-16`, spin `8.326672684688674e-16`.

Key declared gates all pass:

| gate | observed | threshold |
|---|---:|---:|
| raw reverse | `2.6018888816e-7` | `2e-6` |
| projected reverse | `0` | `1e-13` |
| Q0 charge error from 22 | `2.1316282073e-14` | `2e-11` |
| Q0 hole error from 2 | `2.2204460493e-15` | `2e-11` |
| fresh direct-Gram vs cache | `4.0798060930e-8` | `5e-8` |
| fresh Q0 spin replay | `4.0389955963e-10` | `5e-10` |
| seven-Q charge/hole | `6.7293813909e-8` | `8e-8` |
| seven-Q spin | `1.4135659012e-8` | `2e-8` |
| primitive-127 baseline | `6.1771745496e-8` | `8e-8` |
| fold cyclic permutation | `1.2808774336e-15` | `2e-10` |

The Q0, seven-Q, and baseline ceilings are replay tolerances, not physical uncertainty estimates. The baseline and fresh-Q0-spin margins are narrow (factors about 1.30 and 1.24).

## Real-space, positivity, Pauli, sums, and convergence

Independent R6 reconstruction gives:

- total active electrons `22.000000000000025`, total holes `2.000000000000002`;
- maximum imaginary residue `6.5396244094e-16`;
- minimum electron density `0.2327872527 e/nm²`;
- minimum hole density `0.00204683154 e/nm²`;
- minimum local Pauli margin `n_e/2-|S| = 0.1155988616 ħ/nm²`.

Hence final-R6 positivity and local Pauli gates pass with large positive margins. For both partition choices and all three translated representatives, the recomputed R6 regional sums equal Q0 to `3.55e-15`; the summary's all-shell maximum is `7.11e-15`. Every physical translation is the exact required cyclic region permutation (maximum residual `0.0`).

R5→R6 convergence indicators are:

- relative charge L2 `0.00758437264` (`<0.05`);
- pointwise charge max `0.0327017874 e/nm²` (`<0.15`);
- spin-component max `0.0144088663 ħ/nm²` (`<0.15`);
- spin-vector max `0.0163320189 ħ/nm²` (`<0.15`);
- regional charge max `0.00074789357 e` (`<0.15`);
- regional spin-component max `0.000687275054 ħ` (`<0.10`).

The 192→384 regional differences are charge `0.000836558166 e` and spin `0.000830308339 ħ`, both below `0.02`.

These are deterministic last-step differences, not statistical error bars. Shell behavior is not asymptotically monotone: R5→R6 pointwise and regional changes are larger than R4→R5. Therefore they justify only the declared threshold-based sufficiency, not an infinite-shell extrapolation.

## Exact meaning of `N_A`, `N_B`, `N_C` and `M`

The source defines owner indices `0,1,2` by quotient representatives `(0,0),(1,0),(2,0)`. It does **not** bind these owners to externally named A/B/C sites. Translating the partition cyclically permutes the owners. Thus any A/B/C naming is provisional and must state its owner map.

For a region `i`:

- `N_i^e = ∫_i n_e(r)d²r` is the active-valence electron count; `ΣN_i^e=22`.
- `N_i^h = ∫_i n_h(r)d²r` is the hole count relative to the 24-electron active reference; `ΣN_i^h=2` and here `N_i^e+N_i^h=8`.
- `M_i = ∫_i S(r)d²r = (1/2)∫_i σ(r)d²r`, in units of `ħ`, is the physical spin vector, not the raw Pauli vector.

At R6/grid384, using provisional `A/B/C = owner0/owner1/owner2`:

| partition | owner | `N^h` | `N^e` | `M=(Mx,My,Mz) [ħ]` |
|---|---|---:|---:|---|
| parallelogram | A/0 | `0.6667112572` | `7.3332887428` | `(0.0013788390,-0.0008404132,-0.0080038586)` |
| parallelogram | B/1 | `0.6658191522` | `7.3341808478` | `(-0.0003754788,0.0014716306,0.0164544718)` |
| parallelogram | C/2 | `0.6674695906` | `7.3325304094` | `(-0.0006089569,-0.0012425385,-0.0084186190)` |
| WS | A/0 | `0.8584671285` | `7.1415328715` | `(-0.2946200260,-0.1570998648,0.0000378442)` |
| WS | B/1 | `0.2830264373` | `7.7169735627` | `(0.0304427584,-0.0471681170,0.0000524350)` |
| WS | C/2 | `0.8585064342` | `7.1414935658` | `(0.2645716709,0.2036566607,-0.0000582849)` |

Partition-specific maxima of the two convergence indicators are:

| partition | R5→R6 `max|δN^h|` | R5→R6 `max|δM_component|` | 192→384 `max|δN^h|` | 192→384 `max|δM_component|` |
|---|---:|---:|---:|---:|
| parallelogram | `5.84e-7` | `1.79e-4 ħ` | `8.37e-4` | `8.30e-4 ħ` |
| WS | `7.48e-4` | `6.87e-4 ħ` | `1.97e-5` | `8.56e-6 ħ` |

Exact componentwise R5→R6 and grid384-minus-grid192 arrays are preserved in `RECOMPUTE.json`.

## Hypothesis verdict

The hypothesis-matched WS relabeling is `(A,B,C)=(owner0,owner2,owner1)`:

- holes: `(N_A^h,N_B^h,N_C^h)=(0.8584671285,0.8585064342,0.2830264373)`;
- `|N_A^h-N_B^h|=3.9306e-5`, which is within the deterministic shell/quadrature resolution;
- define the longitudinal Néel axis from `M_A-M_B`: `l=(-0.8403050671,-0.5421138011,0.0001444545)`;
- longitudinal projections are `(M_A·l,M_B·l,M_C·l)=(+0.3327367110,-0.3327260106,-1.07094e-5) ħ`;
- full moment magnitudes are `(0.3338881979,0.3338775343,0.0561390733) ħ`;
- transverse magnitudes are `(0.0277057719,0.0277057719,0.0561390723) ħ`.

Therefore:

1. **`N_A≈N_B`: supported by the WS partition.**
2. **Opposite longitudinal A/B spin: supported to the numerical resolution.**
3. **`C≈0`: supported only for the longitudinal projection.** The full `M_C` is about 16.8% of the large-region moment and exceeds both shell and quadrature changes by orders of magnitude, so `M_C≈0` as a vector is rejected.
4. **Exact `(M,-M,0)`: rejected.** A/B share transverse components and C carries the compensating transverse moment.
5. **Partition independence: rejected.** The parallelogram gives nearly equal `2/3` hole counts in all regions and an approximately `(-1,+2,-1)` z-dominated moment pattern rather than the WS pattern.

The defensible scientific wording is: **the WS integration reveals two nearly equally hole-rich regions with nearly opposite longitudinal moments and a hole-poor third region whose longitudinal moment is near zero but whose transverse moment is finite.** This remains a finite-R6, Gz=0, fixed-rank local-branch statement. It does not authorize “honeycomb,” “Néel,” “altermagnet,” or global-ground-state labels.

## Core hypothesis

- Possible convention, normalization, inventory, or publication faults could have produced a plausible but incorrect three-region pattern. Numerically these are excluded; the surviving issue is the `0400` versus required `0444` publication-mode defect and the physical nonuniqueness of the partition interpretation.

## Causal chain

- Frozen local HF state → `+Q` oracle vertices → conjugating stored-projector contraction → physical `-Q` coefficient labels → `exp(-iQ·r)/A` R1–R6 fields → physical region masks → charge/spin integrals → hypothesis test.

## Discriminating check

- A standalone test-node recomputation from `matched_R*.npz` reproduced all shell summaries, gate metrics, region sums, translations, and R5/R6 and 192/384 differences; it directly compared saved physical labels to the accepted seven-Q artifact. This excludes a summary-only, second-reversal, partition-sum, and NPZ/report mismatch. It does not test an R7 shell, Gz dependence, another partition family, another HF basin, or the global ground state.

## Code changes

- No v25 source, control, checkpoint, output, or metadata was modified. Only this detached review directory was created. No framework, physics, plotting, or result code changed.

## Audit-run provenance

Audit jobs `513769`, `513779`, and `513781` failed in the detached checker before JSON serialization while correcting, respectively, the numerator/physical-label baseline index and JSON complex serialization. Job `513785` completed the R5/R6 recomputation. Job `513786` failed before execution because the existing Slurm output files had been made read-only. After reopening only the detached audit outputs, final extended all-shell job `513790` completed. None of these jobs wrote into v25; exact accounting is retained in `SACCT_AUDIT_RECOMPUTE_JOBS.txt`.

## Validation

- Producer scheduler/accounting: complete, `0:0`.
- Final/checkpoint hashes and bindings: pass.
- Independent NPZ/summary and regional recomputation: exact agreement.
- All declared scientific gates: pass.
- Publication mode contract: **fail (`0400`, required `0444`)**.
- Remaining uncertainty: finite shell/grid, Gz=0 projection, partition dependence, local fixed-rank branch, no global-ground-state claim.
