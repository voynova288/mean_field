# Du2017 senior-agent calibrated Fig. 2 forensic audit

**Date:** 2026-07-28

**Status:** completed as a forensic reconstruction; reproduction claim not verified; no production release

## Core hypothesis

The supplied prose report might contain enough information to reconstruct its
claimed fitted self-consistent curve even though the referenced
`teib_kane_bcs_stage_20260722` source bundle is absent.

Three hypotheses were tested:

1. the reported fitted dispersion, vertex, chemical potential, and Coulomb
   scales generate a paired fixed point close to Fig. 2 on the archived TEIB
   Kane parent;
2. the result is robust to seed and branch schedule rather than selected by a
   particular continuation path;
3. the fitted vertex resembles a simple independently archived Kane
   orbital/layer-character factor.

All three fail in the available reconstruction.

## Causal chain

```text
immutable TEIB N=7 fixed-density Kane-Poisson E1/H1 bands
-> ordinary-electron E1/H1 conversion
-> reported C_pair(k)
-> reported z(k) z(k') and channel scales
-> eta-resolved gap/exchange map
-> direct fixed-mu and density-to-mu branches
-> postsolve-only Fig. 2 scoring
```

The digitized Fig. 2 vectors were not passed into the solver.  They were used
only after convergence to calculate RMS errors.

## Missing original evidence

The report references the following files, but they are absent from the current
filesystem and all tracked branches searched:

```text
teib_kane_bcs_stage_20260722/src/excitonic_bcs.py
teib_kane_bcs_stage_20260722/tests/test_workflow.py
teib_kane_bcs_stage_20260722/validation_kane_bcs.json
```

Therefore this audit checks the prose-defined equations, not the unarchived
implementation or its claimed line-level provenance.

The missing details include:

- the exact legacy diagonal singular-cell construction;
- the exact branch-selection and continuation schedule;
- the energy-gauge bridge defining `mu_pair`;
- the exact Kane-Poisson parent used by that separate code agent.

## Input and convention

The immutable band input was

```text
reference/teib_20260720_unpacked/teib/data/kp_final_E1H1.npz
SHA-256 83b3322908d0c14a95e5d79bd192113595ef416d584a58178b47a6faebfc44a4
```

This is the archived N=7 TEIB fixed-density parent.  It has no Kane-stage
same-gauge common chemical potential and is neither z-basis nor in-plane-UV
certified.

The audit retained

\[
E=\sqrt{\xi^2+\Delta^2}
\]

and used the eta-resolved thermal factor

\[
F=1-f\!\left(\frac{E+\eta}{2}\right)
    -f\!\left(\frac{E-\eta}{2}\right).
\]

The one effective pair chemical potential shifts xi.  At the low-density stage
it can constrain only the average of electron and hole densities; it cannot in
general enforce both independently.

## Report-parameter checkpoints

The fitted formulas themselves were reproduced exactly:

```text
C_pair(0.000) =  0.000000 meV
C_pair(0.005) = -0.135478 meV
C_pair(0.025) = +0.441702 meV
C_pair(0.035) = +6.068543 meV

z(0.000) = 0.380000
z(0.025) = 0.890227
z(0.035) = 0.979171
```

Thus later disagreement is not caused by a transcription error in these
reported functions.

## Reduced diagnostic

The reduced Test001 audit used 61 radial points, 48 angular points, and 16
self-cell subcells.  It found:

```text
direct fixed-mu normal gap-map eigenvalue 0.44036
number of paired direct nonzero seeds      0
E(0) direct normal                         7.7496 meV
E_min direct normal                        7.3424 meV at 0.014 nm^-1
energy RMS versus Fig. 2                   4.15 meV
gap RMS versus Fig. 2                      1.10 meV
```

The low-density stage did produce a paired state with approximately
`Delta_max=1.36 meV`, but moving to the fitted `mu_pair=-37.61102 meV` on the
coarse five-step schedule collapsed it to the normal branch.

## Full Slurm audit

Slurm job `195814` ran the full 241-point grid, 192 angular points, 64 self-cell
subcells, and a 40-step chemical-potential homotopy.

Engineering result:

```text
state      COMPLETED
exit code  0
elapsed    24 s
TotalCPU   75.4 s
MaxRSS     286 MB
```

### Direct fitted-mu branches

For the principal cell-quadrature model:

```text
normal gap-map eigenvalue               0.4486949
paired direct roots from 0.2/1.5/5 meV  0 of 3
E(0)                                    7.74961 meV
E_min                                   7.34243 meV at 0.014 nm^-1
energy RMS versus Fig. 2                4.14399 meV
gap RMS versus Fig. 2                   1.10127 meV
```

The hybrid application of `diagonal_scale=0.782` changed the normal eigenvalue
only to `0.4460000`; it did not create a paired branch.

The direct fixed-mu branch is therefore linearly normal and does not reproduce
Fig. 2 under the modern cell quadrature.  The fitted interlayer scale would
need to increase by more than a factor of approximately `1/0.4487=2.23` just
to reach the linear instability, before any target-shape question.

### Density-selected continuation branch

At average density `9.4e-5 nm^-2`, nonzero seeds converged to a paired branch.
The 40-step mu homotopy preserved a paired endpoint, but all nonzero seeds gave
approximately

```text
E(0)       24.1933 meV
E_min       0.7932 meV
k_min       0.0395 nm^-1
Delta_max   0.6023 meV
energy RMS 16.0657 meV
Delta RMS   0.8159 meV
n_e-n_h    -2.6291e-6 nm^-2
```

This is a reproducible path-dependent paired fixed point of the reconstructed
map, but it is far from Fig. 2 and is not neutral.  The imbalance is much larger
than the declared `2e-8 nm^-2` gate.

The direct and density-continuation calculations also reach different normal or
paired endpoints at the same fitted chemical potential.  This is numerical
branch dependence, not a thermodynamic phase comparison, because the report's
finite-temperature exchange map has not been derived from a common free-energy
functional.

## Ablation observations

All direct fixed-mu ablations remained normal.  Selected normal gap-map
eigenvalues were

```text
full reported terms      0.4487
no pair curvature        0.1304
no pair Gaussian         0.2514
no pair ring             0.4454
no pair vertex           0.6627
unscaled interlayer      0.2147
```

This confirms that the Gaussian/curvature combination controls the normal
spectrum strongly and that the fitted vertex actually suppresses pairing.  It
does not recover the reported Fig. 2 branch.

## Does the fitted vertex resemble archived Kane character?

A separate read-only Test001 diagnostic compared

\[
z(k)=1-0.62e^{-(k/0.019)^2}
\]

to the archived N=7 Gamma6/valence and layer weights.

Representative values are

```text
                         k=0       k≈0.025    k≈0.035
z(k)                     0.380     0.898      0.977
electron Gamma6 weight   0.576     0.586      0.591
layer product             0.671     0.683      0.689
orbital-layer product     0.387     0.400      0.407
```

The orbital-layer product happens to match `z(0)` but has almost none of the
required momentum growth.  Several quantities have high narrow-window
correlation only because both are monotonic; reproducing the fitted rise needs
large post-hoc affine scales and offsets.  Therefore these simple character
weights do not provide a microscopic derivation of the fitted vertex.

Moreover, classification and layer weights are not Coulomb matrix elements.
A real projected density vertex must be calculated from same-parent E/H
wavefunctions, momentum transfer, z profiles, and dielectric Green functions,
and must satisfy the appropriate normalization and U(2) covariance checks.

## Discriminating conclusion

The prose report is transparent that its successful old curve was calibrated,
but the reported parameter list is not sufficient to reproduce that success on
the available archived Kane parent under controlled cell quadrature.

What was verified:

- the fitted C_pair and z formulas and numerical checkpoints;
- the reconstructed map has real self-consistent direct and continuation roots;
- branch schedule materially changes the endpoint;
- the final Fig. 2 similarity gate fails.

What was not verified:

- the absent senior-agent source implementation;
- its exact old diagonal/node quadrature;
- its parent/gauge/chemical-potential bridge;
- a neutral one-common-mu branch;
- UV, z-basis, or Kane-Poisson closure;
- a physical or parameter-free Fig. 2 prediction.

## Useful inspiration

The report suggests two hypotheses worth testing independently, without
importing its fitted numbers:

1. **Momentum-dependent normal self-energy/downfolding.**  The strong
   curvature/Gaussian cancellation indicates that a simple Kane E1/H1 pair
   dispersion may omit remote-band or screening-induced momentum dependence.
   It should be derived from a source-backed full-band/Wannier model, not fitted
   to Fig. 2.
2. **Nonseparable projected interaction vertex.**  Compute the actual
   E/H-projected Coulomb kernel from the same parent, then perform a weighted
   SVD.  Only if an independently calculated leading singular mode is dominant
   should a separable `z(k)z(k')` approximation be considered.  It must not be
   normalized by fitting Fig. 2.

The fitted `s_aa`, `s_ab`, and `diagonal_scale` themselves are not useful
microscopic estimates.  The channel enhancements are incompatible with one
ordinary dielectric constant, and the diagonal factor points to the legacy
quadrature rather than material physics.

## Decision

Do not import the fitted parameters into production code.  Keep Poisson,
HF/BCS, and JDOS release disabled.

The decisive next request is the actual `teib_kane_bcs_stage_20260722` bundle,
including source hash, command, raw NPZ, validation JSON, and generated figure
hash.  Without it, the claim can only be treated as an unarchived calibrated
model report.

## Evidence

```text
full forensic JSON
4a338d66c515912d5ef9033c0a4a6fa78ce62e76ebf0bf4cf9c96ff4e19d4c84

full forensic NPZ
af650628311172288124fcd71f7bbed1149774f5858e7d6d784a7dd40585e26e

vertex-character audit JSON
77f71ed98912e99f870b182c6014a09812dee20eefc5e1ed8f710f065043e7d5
```
