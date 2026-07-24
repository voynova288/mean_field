# Du2017/InAs–GaSb reference-subtracted UV trial

**Date:** 2026-07-24
**Status:** attempted and failed closed at the W4 fixed-U scout stage; Poisson/HF/BCS not released

## 1. Question tested

We tested the proposed modern continuum prescription

\[
\delta n_e(z)=\int_k\operatorname{Tr}[R_e(z)(F_U-F_0)],
\qquad
\delta n_h(z)=\int_k\operatorname{Tr}[R_h(z)(F_0-F_U)],
\]

where `F_0` is a frozen zero-potential occupied density matrix built with the
same Kane parent, material profile, k/z mesh, selected window, and source
code as the physical source.

This is explicitly a new renormalized-model trial.  It is not claimed to be
the unpublished Du2017 background prescription.

## 2. One-common-mu implementation

The counterterm stores the density matrix obtained from one zero-potential
neutrality root.  Its origin `mu` is immutable provenance, not a second SCF
reservoir.

For every physical potential `U`, exactly one dynamic chemical potential is
solved from

\[
\delta n_e(\mu,U)=\delta n_h(\mu,U).
\]

No `mu_e`, `mu_h`, imposed experimental density, or independently optimized
reference chemical potential is present.

An important fail-closed result was found during implementation: the old raw
Note-6 root `n_e=n_h` is not generally the same as the induced-neutrality root
when the selected W4 available-valence projector changes with `U`.  Slurm job
`193264` stopped on this mismatch.  The final trial therefore uses the
induced-neutrality root above.

## 3. Implementation and validation

Local prototype paths:

```text
src/mean_field/systems/inas_gasb/kane_uv_reference.py
src/mean_field/systems/inas_gasb/kane_poisson.py
 tests/test_inas_gasb_kane_uv_reference.py
 tmp/audit_du2017_reference_subtracted_uv_actual8.py
```

The implementation:

- exposes per-k, per-z occupied-electron, occupied-valence, and
  available-valence integrands;
- subtracts physical/reference occupied density pointwise before radial
  integration;
- binds reference and physical sources to the exact same parent specification;
- continues the probe W4 quartet from the zero-potential quartet with one
  stateful builder;
- independently records reference-to-probe principal singular values;
- checks random internal U(4) covariance;
- stores strict JSON/numeric NPZ with marker-last process completion.

Focused validation on `test001`:

```text
27 passed in 1.82s
```

## 4. Slurm evidence chain

| Job | Purpose | Result |
|---:|---|---|
| 193254 | first source snapshot | packaging failure before Kane compute; missing top-level `analysis` package |
| 193256 | raw Note-6 profile subtraction scout | completed; rejected because changing W4 available-valence term is part of the subtraction |
| 193264 | occupied-density response with old raw root | failed closed: raw neutrality does not imply induced neutrality |
| 193277 | corrected single-mu induced-neutrality cutoff ladder | completed; `uv_scout_pass=false` |
| 193279 | fixed `kmax=0.30`, radial `nr=200→400` | completed; radial convergence failed |
| 193282 | fixed `kmax=0.30`, radial `nr=400→800` | completed; radial convergence still failed |

All numerical jobs used full 64-core `regular256` nodes with account `hmt03`.

## 5. Positive checks

At the largest tested mesh:

- reference-to-probe W4 minimum principal singular value:
  approximately `0.99868905`;
- internal U(4) gauge error: `1.43e-13`;
- minimum candidate-edge gap: approximately `9.3274 meV`;
- maximum selected eigen-residual: approximately `1.25e-11 meV`;
- induced global neutrality residual: approximately `4e-18 nm^-2`.

Thus the failure is not explained by gauge drift, loss of the quartet,
chemical-potential duplication, or eigensolver residuals.

The electron-response outer tail becomes small and steep at `kmax=0.30`:

```text
outer 10% L1 fraction ~0.0060
tail power           ~10.11
```

The hole response does not reach the required 2D integrability exponent:

```text
outer 10% L1 fraction ~0.0406
tail power           ~1.323 < 2
```

## 6. Cutoff failure

For the manufactured bounded probe potential, the induced density in `nm^-2`
changed as follows:

```text
kmax 0.12: +6.5521e-5
kmax 0.15: +9.9832e-5
kmax 0.18: +1.2356e-4
kmax 0.24: -3.9000e-5
kmax 0.30: -1.5485e-5
```

The response changes sign between `0.18` and `0.24 nm^-1`.  The integrated
profile change from `0.24` to `0.30 nm^-1` is about `77.4%`.  Therefore the
subtraction does not establish a cutoff-independent W4 response in the tested
range.

## 7. Radial failure

At fixed `kmax=0.30 nm^-1`:

```text
nr=200 -> 400:
  induced-density relative change ~45.1%
  profile relative change         ~31.8%

nr=400 -> 800:
  induced-density relative change ~18.2%
  profile relative change         ~14.1%
```

The response is converging more slowly with radial resolution, but it remains
far outside the predeclared 1% refinement gate.  We stopped rather than
escalating resolution until a desired answer appeared.

## 8. Interpretation

The two-band same-symbol subtraction argument is mathematically valid, but the
current full-`8Nz`-parent/W4 implementation has not inherited its clean
`k^-4` behavior in every channel.  The likely unresolved ingredients are:

1. the W4 selected occupied-density response has no proven remote-subband
   completion;
2. the hole response still has a nonintegrable fitted tail over the tested
   interval;
3. the low-temperature radial occupation/root is a small difference of larger
   quantities and requires a better source-backed quadrature than the present
   one-point equal-area cells;
4. the finite reference condition remains a declared modern-model choice.

These findings do not disprove reference subtraction in a complete microscopic
or sufficiently enlarged retained-subband model.  They show that applying it
to the present W4 quartet is insufficient.

## 9. Decision

The attempted production sequence

```text
finite z Kane -> W4 occupied-density reference subtraction
-> explicit-boundary Poisson
```

is stopped before Poisson because both cutoff and radial gates fail.

The next defensible investigation would require either:

- a source-method finite-plane-wave parent with a controlled enlarged
  anticrossing-subband sum/remote complement; or
- an explicitly lattice-completed finite-BZ model.

No self-consistent reference-subtracted Poisson, HF, BCS, JDOS, or Du2017
reproduction claim is authorized by this trial.

## 10. Evidence hashes

```text
raw-profile scout JSON:
1957137f8cfb20cedbf2c0a588f49e1bbb064a982c5416bb766f1d7c54993d59

induced-neutrality cutoff scout JSON:
5bcf9258985d1956cab9e8e68732a01756f6eda4686755ec7b7c0a7038c2653e

nr=200->400 JSON:
89965c491ed13b50419aef1c3e89093ef6575a3dd2ad1f51d3a84b0e4b9297c8

nr=400->800 JSON:
dbe723bba18b3a22c2ebc4138fc9d06c03f50042458b1c2ffa967e96812fd518
```
