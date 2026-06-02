# TBG cRPA Dielectric Failure, 2026-05-30

## Status

The `lk18/lg13/q11` HF-compatible run finished cleanly, but its cRPA dielectric
function is physically wrong and the downstream HF result should not be used as
final evidence.

Finished chain:

- cRPA artifact:
  `results/TBG_HF_cRPA/crpa_lk18_lg13_q11_hfcompatible_20260527/crpa_lk18_lg13_q11_hf_compatible_merged`
- HF output:
  `results/TBG_HF_cRPA/hf_crpa_lk18_lg9_q11_tests_20260527/theta_105_nu_-3000_lk18_lg9_q11_iter300_vp_w1_97p4`

The HF SCF result converged, but it consumed a bad dielectric artifact.

## Numerical Symptom

Representative Fig. 1(e)-window diagnostics:

| Artifact | BM grid | q shell | form factor | peak eps*eps_bn | q=0.4 | q=0.8 | q=1.2 |
|---|---:|---:|---|---:|---:|---:|---:|
| paper-reference `lk24/lg9/q11` | 9 | 11 | `zhang_zero_fill`, non-periodic-G BM | 20.446 | 17.031 | 13.109 | 11.414 |
| old HF-compatible `lk24/lg9/q11` | 9 | 11 | old `hf_periodic` periodic-G roll | 27.378 | 23.884 | 19.714 | 18.605 |
| legal no-alias small `lk6/lg11/q9` | 11 | 9 | old `hf_periodic` periodic-G roll | 28.799 | 24.053 | 20.228 | 19.086 |
| new `lk18/lg13/q11` | 13 | 11 | old `hf_periodic` periodic-G roll | 27.723 | 23.425 | 19.795 | 18.636 |

This shows the failure is not caused by Slurm, merge, `lk=18`, or the old
`lg9/q11` alias alone.  The high dielectric curve is systematic to the old
periodic-G-roll cRPA vertex.

## Formula-Level Diagnosis

For a density vertex with momentum `q + Q`, if `k + q` is folded back to the
stored k mesh as

```text
k + q = k' + W,
```

then the plane-wave coefficient contraction is

```text
lambda_Q(k,q) = sum_G C_left(k', G + Q + W)^* C_right(k, G).
```

The k mesh is periodic, so the reciprocal wrap `W` must be included.  The
finite plane-wave G cutoff is not a torus.  If `G + Q + W` leaves the retained
plane-wave shell, the coefficient is zero.

Therefore the correct production convention is:

```text
periodic k wrapping: yes
use Q + W in the vertex: yes
periodic roll of finite G cutoff: no
finite-G zero fill: yes
```

The old `hf_periodic` implementation did:

```python
np.roll(values, shift=(0, dm, dn, 0), axis=(0, 1, 2, 3))
```

inside the cRPA form factor.  That adds unphysical boundary terms from the
opposite side of the truncated plane-wave grid and over-screens the dielectric
function.

This also explains why the old no-alias gate was insufficient.  It made the
periodic-G torus labels distinct, but the torus itself was the wrong object for
the finite plane-wave response vertex.

## Source Reference Check

The cloned `TBG-HF` reference code uses periodic k wrapping but zero-fills G
coefficients that roll outside the finite coefficient grid:

```python
cp_copy = np.roll(cp, (G1,G2), axis = (2,3))
if G1 > 0:
    cp_copy[:,:,0:G1,:,:] = 0
...
```

This matches the formula above: implement the shift conveniently, then remove
the out-of-cutoff components instead of treating the G shell as periodic.

## Source Changes

- `src/mean_field/crpa/form_factor.py`
  - Production mode is now `k_periodic_zero_fill`.
  - Legacy CLI alias `hf_periodic` is accepted but normalized to
    `k_periodic_zero_fill`.
  - The old periodic-G roll is retained only as
    `hf_periodic_roll` diagnostic mode.

- `src/mean_field/crpa/workflow.py`
  - Metadata now records `form_factor_mode = k_periodic_zero_fill`.
  - Metadata records `form_factor_g_boundary = zero_fill`.
  - The old no-alias fields were replaced by diagnostic
    `periodic_roll_*` metadata because production no longer rolls G.

- `src/mean_field/crpa/hf_validation.py`
  - HF-compatible validation now rejects old artifacts with
    `form_factor_mode = hf_periodic`.
  - Old periodic-G-roll cRPA caches must be regenerated.

- cRPA devtools and submission wrappers now default to
  `k_periodic_zero_fill`.

## Invalidated Artifacts

Do not use these as final HF+cRPA physics evidence:

- `results/TBG_HF_cRPA/crpa_lk18_lg13_q11_hfcompatible_20260527/...`
- old `hf_periodic` artifacts under
  `results/TBG_HF_cRPA/crpa_alias_gate_20260524/...`
- old `crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged`

They are still useful as diagnostics of the periodic-G-roll failure.

## Next Validation Gate

Do not scale `chi0` and do not infer from reconstructed path bands.

Run a small cRPA diagnostic with:

```text
periodic_g_grid = true
form_factor_mode = k_periodic_zero_fill
finite-G boundary = zero_fill
```

Suggested first gate:

```text
lk = 6 or 8
lg = 5 or 7
q_lg = 5 or 7
full q table
```

Acceptance:

1. The representative epsilon(q) curve should return near the Zhang
   Fig. 1(e) scale, not the old high band around 24-29.
2. The q-epsilon plot may still contain local-field scatter if every diagonal
   `(q_tilde + Q)` element is plotted, but the representative Fig. 1(e)-window
   curve should no longer be over-screened.
3. Only after the dielectric curve passes this gate should an HF SCF-grid line
   plot be used as the band diagnostic.

## Small Slurm Gate Results

Job `134061` on `test001` completed successfully.

Output:

```text
results/TBG_HF_cRPA/crpa_kperiodic_zerofill_smoke_20260530/lk6_lg5_q5
```

Run parameters:

```text
lk = 6
lg = 5
q_lg = 5
periodic_g_grid = true
form_factor_mode = k_periodic_zero_fill
form_factor_g_boundary = zero_fill
occupation_mode = cnp_index
```

Representative Fig. 1(e)-window diagnostics:

```text
q_peak_nm_inv = 0.156023556476
eps_total_peak = 20.8149796285
eps_total_q0 = 4
eps_total_q04 = 16.4704064213
eps_total_q08 = 11.3071037178
eps_total_q12 = 8.67985682746
radial_std_max_0_1p2 = 0.467239252288
```

This is a strong source-level check: the peak returned from the old
periodic-G-roll value near `28` to the Zhang-scale value near `20`.  The
remaining `q=0.8` and `q=1.2` undershoot are expected finite-shell/finite-k
errors for this intentionally small gate and should be checked next with a
larger but still controlled cRPA run.

Job `134064` then checked a larger shell on the same code path.

Output:

```text
results/TBG_HF_cRPA/crpa_kperiodic_zerofill_smoke_20260530/lk6_lg7_q7
```

Run parameters:

```text
lk = 6
lg = 7
q_lg = 7
periodic_g_grid = true
form_factor_mode = k_periodic_zero_fill
form_factor_g_boundary = zero_fill
occupation_mode = cnp_index
```

Representative diagnostics:

```text
q_peak_nm_inv = 0.180160484663
eps_total_peak = 21.0950846621
eps_total_q0 = 4
eps_total_q04 = 17.0432942084
eps_total_q08 = 12.7379834246
eps_total_q12 = 10.6167170216
radial_std_max_0_1p2 = 0.427552970387
```

Increasing the shell from `lg5/q5` to `lg7/q7` moves the tail toward the
paper-reference values (`q=0.8`: 11.31 -> 12.74, paper reference about 13.11;
`q=1.2`: 8.68 -> 10.62, paper reference about 11.41).  This supports the
finite-shell interpretation of the remaining tail error.

Focused regression tests were run on Slurm job `134066`:

```text
/data/home/ziyuzhu/miniconda3/bin/python -m pytest tests/test_crpa_core.py
44 passed in 3.33s
```

## Full `lk18/lg9/q11` cRPA Resubmission

A corrected cRPA-only full-table chain was submitted after the small gates.
HF was deliberately not submitted yet.

Manifest:

```text
results/TBG_HF_cRPA/submission_jobs_crpa_lk18_lg9_q11_kperiodic_zerofill_20260530.tsv
```

Jobs:

```text
BM cache: 134067
chunk array: 134068
merge: 134069
```

Expected merged artifact:

```text
results/TBG_HF_cRPA/crpa_lk18_lg9_q11_hf_kperiodic_zerofill_20260530_merged
```

The BM stage started on `node052` and completed.  The chunk array started with
packed `2 x 32`-thread chunks per `regular128` node.  Inspect the merged
artifact's `crpa_epsilon_diagnostics_summary.md` before launching any HF run.
