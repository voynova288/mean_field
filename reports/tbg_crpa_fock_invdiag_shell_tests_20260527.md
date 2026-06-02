# TBG HF+cRPA Fock `epsilon_inv` Shell Tests, 2026-05-27

> Superseded on 2026-05-30 for final physics.  The old `hf_periodic` cRPA
> artifacts used a periodic-G roll in the response form factor and over-screen
> the dielectric curve.  Keep this file only as historical diagnostic context.
> Current source-level diagnosis is recorded in
> `reports/tbg_crpa_dielectric_failure_20260530.md`.

## Scope

This note records the focused tests after changing the HF+cRPA Fock scalar
lookup from the raw dielectric diagonal to the scalar divisor implied by the
stored inverse dielectric matrix:

```text
W_QQ = V_Q Re[(epsilon_inv)_QQ]
epsilon_Fock(Q) = 1 / Re[(epsilon_inv)_QQ]
```

Only saved SCF-grid data and SCF-grid line plots are treated as physics
diagnostics here.  Reconstructed off-grid band plots were deliberately skipped.

## Source And Regression Checks

- `src/mean_field/crpa/screened_coulomb.py`: `CRPAScreenedCoulomb.fock_epsilon_array()` now uses `1 / Re diag(epsilon_inv)` for the matrix-diagonal Fock scalar.
- `src/mean_field/crpa/hf_interface.py`: production Fock docstring updated to match the `diag(V) @ epsilon_inv` convention.
- `tests/test_crpa_core.py`: added a non-diagonal dielectric regression test proving that `Re diag(epsilon)` and `1 / Re diag(epsilon_inv)` are not interchangeable.
- `scripts/make_crpa_fock_wdiag_artifact.py`: marked as plotting/legacy diagnostic only.

Passed focused tests:

```text
/data/home/ziyuzhu/miniconda3/bin/python -m pytest tests/test_crpa_core.py
44 passed

/data/home/ziyuzhu/miniconda3/bin/python -m pytest \
  tests/test_crpa_core.py \
  tests/test_core_hf_engine.py \
  tests/test_core_hf_coulomb.py \
  tests/test_b0_overlap_helpers.py
58 passed
```

The full `pytest tests` run was not clean: two B0 helper/path-export tests fail
outside the cRPA Fock lookup change, and the full run was terminated after
hanging.  Those failures were not used as HF+cRPA evidence.

## Artifact Lookup Checks

All legal no-alias artifacts below validated with zero q-lookup failures.

| Artifact | Validation | Fock epsilon min / median / max | Old/raw-diagonal to new ratio min / median / max |
|---|---:|---:|---:|
| `hf_periodic_lk6_lg7_q5` | pass | `1 / 4.55409 / 7.33737` | `1 / 1.00331 / 1.03169` |
| `hf_periodic_lk8_lg9_q7_regular1h` | pass | `1 / 4.3474 / 7.26944` | `1 / 1.00044 / 1.02701` |
| `hf_periodic_lk6_lg11_q9_regular2h` | pass | `1 / 4.25259 / 7.35465` | `1 / 1.00008 / 1.03109` |
| old `crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged` | fail no-alias gate | `1 / 2.01067 / 7.21755` | `1 / 1.95965 / 3.07113` |

Interpretation:

- The `epsilon_inv` source fix is required by the screened-interaction formula.
- On legal small artifacts the correction is numerically small, so it does not
  explain the remaining physics issue by itself.
- The old merged `lg9/q11` artifact still fails the hard no-alias condition and
  must not be used for final physics.

## Slurm Gate Runs

All HF gates were submitted through `scripts/submit_mean_field.sbatch` on the
`test` partition with `--account=hmt03`.  Each run used:

```text
theta = 1.05 deg
nu = -3
lk = 6
init = vp:1 except the explicit BM smoke
fock_interpolation = matrix_diagonal
split_mode = active_cnp_fock_reference_projector
periodic_g_grid = true
g_boundary_mode = periodic
write_scf_path = true
write_reconstructed_path = false
w1 = 97.4
```

One metadata gate failed as intended:

- Job `131534`: rejected because the cRPA artifact had `w1=97.4` but the HF
  default was `w1=97.5`.  This confirms that the metadata compatibility gate is active.

The BM-init smoke was not physically useful:

| Job | Run tag | Iterations | Direct gap (meV) | Indirect gap (meV) | Top valence width (meV) | First conduction width (meV) |
|---:|---|---:|---:|---:|---:|---:|
| `131536` | `small_lk6_lg3_q5_iter20_w1_97p4` | 20 | `0.0000036` | `-30.535` | `30.535` | `30.535` |

The useful shell series is:

| Job | cRPA artifact | HF `lg` / `overlap_lg` | `q_lg` | Iterations | Residual | Direct gap (meV) | Indirect gap (meV) | Top valence width (meV) | First conduction width (meV) | Mean conduction width (meV) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `131541` | `hf_periodic_lk6_lg7_q5` | `3 / 3` | 5 | 120 | `6.20e-05` | `14.161` | `7.769` | `17.677` | `17.017` | `12.997` |
| `131542` | `hf_periodic_lk6_lg9_q7` | `5 / 5` | 7 | 120 | `2.19e-05` | `3.313` | `-39.073` | `42.744` | `51.458` | `22.514` |
| `131544` | `hf_periodic_lk6_lg11_q9_regular2h` | `7 / 7` | 9 | 120 | `4.01e-05` | `4.310` | `-47.799` | `52.139` | `54.579` | `26.397` |
| `131547` | `hf_periodic_lk6_lg11_q9_regular2h` | `7 / 7` | 9 | 300 | `1.10e-05` | `4.311` | `-47.799` | `52.139` | `54.579` | `26.397` |

Output roots:

- `results/TBG_HF_cRPA/crpa_fock_invdiag_tests_20260527/theta_105_nu_-3000_small_lk6_lg3_q5_iter120_vp_w1_97p4`
- `results/TBG_HF_cRPA/crpa_fock_invdiag_tests_20260527/theta_105_nu_-3000_small_lk6_lg5_q7_iter120_vp_w1_97p4`
- `results/TBG_HF_cRPA/crpa_fock_invdiag_tests_20260527/theta_105_nu_-3000_small_lk6_lg7_q9_iter120_vp_w1_97p4`
- `results/TBG_HF_cRPA/crpa_fock_invdiag_tests_20260527/theta_105_nu_-3000_small_lk6_lg7_q9_iter300_vp_w1_97p4`

Each output directory contains:

- `density_matrix_final_nu_-3.npz`
- `hf_summary_nu_-3_crpa.json`
- `scf_bandwidth_report.json`
- `path_bands/*_hf_scf_path.tsv`
- `path_bands/*_scf_grid_band_plot.png`
- `path_bands/*_scf_grid_band_plot.pdf`

## Shell Interpretation

The shell trend is physically informative:

1. `q_lg=5, lg=3` has a clean positive gap but too-small flat-band widths
   (`~17 meV` for the top valence and first conduction bands).
2. Increasing to `q_lg=7, lg=5` restores a much larger first-conduction width
   (`~51 meV`), so the earlier small bandwidth was largely a finite-shell
   artifact.
3. Increasing again to `q_lg=9, lg=7` gives a minimum direct gap of `4.31 meV`,
   close to Zhang's quoted `nu=-3` HF+cRPA gap scale (`4.4 meV`), and widths
   around `52-55 meV`.
4. Extending the same `q_lg=9, lg=7` run from 120 to 300 iterations changes the
   direct gap by only `0.00025 meV` and the first-conduction width by only
   `0.00010 meV`, so the SCF-grid gap/width numbers are stable even though the
   residual is just above the strict `1e-5` stop.
5. The same `q_lg=9` SCF-grid state still has a large negative indirect gap
   (`-47.80 meV`), so it is not yet a clean final reproduction.  It is only a
   useful small-system signal.

## Epsilon-vs-q Scatter Check

The legal no-alias cRPA artifacts still show comparable radial scatter in the
flattened `epsilon_vs_q` diagnostic:

| Artifact | Bins in `0 <= q <= 1.2 nm^-1` | Mean range / median | Max range / median | Mean std / mean | Max std / mean |
|---|---:|---:|---:|---:|---:|
| `hf_periodic_lk6_lg7_q5` | 38 | `0.208911` | `0.281760` | `0.084111` | `0.121413` |
| `hf_periodic_lk6_lg9_q7` | 38 | `0.208908` | `0.284091` | `0.083944` | `0.122307` |
| `hf_periodic_lk6_lg11_q9_regular2h` | 38 | `0.206096` | `0.280280` | `0.082738` | `0.120807` |

This argues against aliasing as the sole explanation of the multi-valued
`epsilon_vs_q` scatter.  The flattened plot is still a matrix-valued
`(q_tilde, Q)` / local-field diagnostic, not a scalar `epsilon(|q|)` curve.
For paper comparison, the representative window/median curve remains the
better diagnostic.

## Current Conclusion

The source-level Fock scalar convention is now corrected and regression-tested.
The small legal shell series shows that increasing the HF/cRPA shell can recover
the expected direct-gap scale and much larger bandwidth.  However, the negative
indirect gap and the persistent matrix-diagonal epsilon scatter mean the
HF+cRPA reproduction is not yet settled.

The next physics-first audit target remains the definition of the constrained
polarizability target subspace: hard flat-flat transition removal versus a
projector-weighted cRPA exclusion.  Spin scaling or ad hoc chi0 rescaling should
not be used as a fix unless the formula chain demands it.
