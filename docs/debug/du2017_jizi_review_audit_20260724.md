# `jizi.zip` Du2017 Kane–Poisson review audit

**Date:** 2026-07-24
**Archive:** `reference/jizi.zip`
**SHA-256:** `ae9fe75b4f3fcbffade911dbb6698ec6b8e7d592036fc06bedffaf6ef2502421`
**Verdict:** useful toy diagnostics; proposed production closure and BCS release are not accepted

## 1. Non-negotiable chemical-potential contract

The Du2017 canonical Kane–Poisson workflow has exactly one chemical potential.
For every electrostatic-potential iterate `U`, the code must rebuild the Kane
Hamiltonian and solve

\[
n_e(\mu,U)=n_h(\mu,U)
\]

for one common `mu`.  Separate `mu_e`/`mu_h`, separate roots later combined,
an imposed experimental pair density, or an independently optimized reference
chemical potential are different ensembles and are not allowed in this
workflow.

The current implementation already follows this contract:

```text
solve_canonical_split_zero_kane_poisson
  -> builder(U)
  -> _evaluate_source(...)
  -> build_charge_neutral_kane_reference(...)
  -> one Brent root for mu
  -> electron/hole profiles from the same density matrix
  -> Poisson(U)
```

`KanePoissonIterationRecord` stores one field, `mu_mev`.  The solver API accepts
neither an external `mu` nor a target pair density.

A focused regression was added locally:

```text
test_canonical_kane_poisson_solves_one_common_mu_at_every_u_iterate
```

It checks every SCF history entry for one finite `mu_mev` and the same-iterate
neutrality residual.  On `test001`:

```text
21 passed in 5.55s
```

for `tests/test_inas_gasb_kane_poisson.py` and
`tests/test_inas_gasb_normal_reference.py`.

## 2. Archive provenance and validation scope

The archive contains:

```text
du2017_kane_poisson_uv_resolution_20260724.md
du2017_kane_poisson_uv_resolution_20260724.patch
mean_field-debug-du2017-kane-poisson-uv-resolved-20260724.zip
```

The nested snapshot SHA-256 is
`b0a3768b87cf8eb424bd5ba41637a1702c260eaae18115b16851de9febb533c1`.
It was extracted with absolute paths, parent traversal, and symlinks rejected.

The supplied snapshot's own test suite was independently rerun:

```text
119 passed in 23.01s
```

This confirms that its nine new toy/unit tests execute as claimed.  It does
not validate integration with the actual source-pinned 8-band builder,
physical CB1/VB1 projectors, common-`mu` root, Poisson fixed point, or replay
archive.  The supplied report explicitly states that those implementations
and production archives were not available to its author.

## 3. Findings accepted as diagnostics

### 3.1 Two-band UV asymptotic

For

\[
H_m(k)=Ak\sigma_x+(Bk^2+m)\sigma_z,
\]

the unwanted orbital weight has a `1/k^2` tail, giving a logarithmic radial
integral in two dimensions.  Subtracting a second toy model with the same
linear and quadratic high-`k` structure cancels that leading tail and leaves a
`1/k^4` difference.  The supplied analytic and numerical toy tests support
this statement.

This strengthens the diagnosis that raw Note-6 component-transfer charge is
UV sensitive.  It does not specify the finite physical subtraction required
for Du2017.

### 3.2 Quartet charge gauge invariance

For a complete isolated quartet projector `Q` and occupation matrix `F`,

\[
n_e=\operatorname{Tr}(R_eF),\qquad
n_h=\operatorname{Tr}[R_h(Q-F)]
\]

is invariant under internal `U(4)` basis rotations.  This is a valid
algebraic diagnostic and explains why quartet-only charge can remain stable
while an adiabatic rank-two label becomes near-degenerate.

It does not prove that the selected quartet exhausts the source-required
`sum_s` over the full `k`, `U`, and z-basis domain, and it does not establish
the physical rank-two E/H sectors required downstream.

### 3.3 Other useful confirmations

- Finite plane-wave z regularization is source-compatible; unlimited FDM
  refinement is a different UV model and remains rejected for production.
- The table-internal Luttinger relation strongly supports GaSb `gamma2=0.08`.
  This strengthens, but does not replace, the existing source-hash-attested
  and not-author-confirmed material policy.
- An explicit Dirichlet finite-volume Poisson solver is a reasonable device
  component once actual boundary voltages, fixed charges, units, and the
  `U=-eV` bridge are supplied.

## 4. Proposals rejected for the current workflow

### 4.1 `experimental_density_calibrated`

The supplied `KanePoissonClosurePolicy` allows an imposed experimental pair
density and performs no chemical-potential root.  This is explicitly outside
the requested Kane–Poisson ensemble.  It may remain a separately labelled
forensic model, but it cannot replace the one-common-`mu` normal parent or
release HF/BCS.

### 4.2 Separate `mu_E` and `mu_H`

The proposed BCS matrix in the supplied report contains separate `mu_E` and
`mu_H`.  This directly violates equilibrium with one common chemical
potential and is rejected.  No BCS/HF release follows from this archive.

### 4.3 Production reference subtraction

`UVReferenceDefinition` proves compatibility only by comparing a free-form
`principal_symbol_id` string.  It does not compare actual 8-band Kane
couplings, quadratic tensors, angular structure, material profiles, z basis,
observables, grids, or source hashes.  `reference_subtracted_population`
simply subtracts caller-provided arrays and has no common-`mu` root or SCF.

Moreover, the finite reference condition and its occupation are not specified
by Du2017.  Introducing `mu_ref` without defining its relation to the sole
physical `mu` is not allowed.  Therefore same-symbol subtraction remains a
new-model proposal, not a recovered Du2017 closure.

### 4.4 Removing the separate E/H production gate

Quartet-only charge is useful diagnostically, but the current project contract
still requires separate physical E/H rank-two certification.  The supplied
character split uses only a local projected `Gamma6-valence` sign operator and
is not bound to layer character, ancestry, parent/source hashes, or continuity
across `k`, `U`, and basis cutoff.  It cannot silently replace the existing
pair identity contract.

### 4.5 Claiming a predictive fixed-gate model

The supplied policy and Dirichlet solver do not provide Wafer-B boundary
voltages, work functions, interface charge, voltage zero, absolute units, or
the electron-energy sign conversion.  Passing a generic Poisson unit test is
not a fixed-gate Kane–Poisson calculation.

## 5. Additional implementation issues in the supplied package

- `principal_symbol_id` is an unverified string attestation.
- Physical/reference `ComponentPopulation` arrays can broadcast rather than
  requiring identical grids/shapes.
- Local z-resolved populations are labelled `nm^-2` even though a normalized
  local observable contributes an additional inverse-length factor.
- The generic SI Poisson solver is not connected to the current
  `meV`/`nm`/relative-permittivity convention or `U=-eV` sign.
- `validate_du2017_gasb_gamma2(np.nan)` can pass because NaN is not rejected.
- The quartet toy tests contain no remote bands and do not test actual
  source-pinned CB1/VB1 data.
- The advertised `119 passed` includes the snapshot's pre-existing tests; only
  nine tests concern the proposed closure, and none is end-to-end 8-band SCF.

## 6. Decision matrix

| Proposal | Decision |
|---|---|
| one common `mu(U)` neutrality root | required and regression-tested |
| two-band UV toy derivation | accepted diagnostic |
| same-symbol subtraction as mathematical new model | plausible prototype only |
| same-symbol subtraction as Du2017 reproduction | rejected without source/reference |
| quartet-only charge | accepted diagnostic only |
| remove separate E/H production gate | rejected |
| experimental-density replacement for Kane–Poisson | rejected |
| separate `mu_E`, `mu_H` | rejected |
| unfreeze HF/BCS/JDOS | rejected |
| explicit device Dirichlet solver | future work after device inputs |
| GaSb `gamma2=0.08` internal-consistency evidence | accepted as stronger inference, still not author-confirmed |

## 7. Next allowed step

Continue only when one of the following is available:

1. author/source evidence for the in-plane momentum range, retained subbands,
   reference/background, and electrostatic boundary; or
2. explicit authorization to build a new, separately labelled renormalized
   continuum model.

Any future renormalized model must still solve exactly one common `mu(U)` at
every Kane–Poisson iterate, bind the reference to actual 8-band source hashes
and matrices, prove the high-`k` cancellation in the full model, preserve
units/signs, and pass z/k/temperature/replay gates.  It must not be presented
as the hidden Du2017 calculation unless provenance is obtained.
