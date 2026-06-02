# TBG cRPA External Code Comparison, 2026-05-27

## Scope

This note compares the current `Mean_Field` cRPA implementation against external cRPA/downfolding code references.  It deliberately treats the no-cRPA / bare-HF framework as already validated enough for the present debugging line; the comparison below is focused on the cRPA artifact, dielectric normalization, screened-interaction insertion, and target-subspace exclusion.

## Local External References

### RESPACK

Local paths:

- Archive: `reference/crpa_external/RESPACK-20240804.tar.gz`
- Extracted source tree used for inspection: `reference/crpa_external/RESPACK-20240804-dist`

Reference value:

- This is the most useful downloaded cRPA code reference for formula and convention comparison.
- Relevant source directories:
  - `src/chiqw`: polarizability and dielectric-matrix construction.
  - `src/calc_int`: projection of the screened interaction into Wannier matrix elements.
  - `man/en/main.tex`: documented output convention.

Integrity note:

- The official tarball at the downloaded URL reports `gzip: unexpected end of file` under `gzip -t` on this machine, although the key source directories were extracted and are readable.  Use the extracted source for formula inspection only; do not treat the local tarball as a build-quality archive until it is re-fetched cleanly or verified against an upstream checksum.

### Davydov/Choo/Fischer/Neupert Materials Cloud TBG Archive

Local attempted paths:

- `reference/crpa_external/materialscloud_tbg_codes.tar.gz`
- `reference/crpa_external/materialscloud_tbg_codes.tar.gz.part`

Reference value:

- This is TBG-specific and tied to a Wannier-Hubbard workflow, so it is potentially useful for units and projected interaction conventions.
- It is not a drop-in reference for the present continuum BM HF+cRPA solver.

Status:

- The archive link was found and download attempts reached large partial files, but the completed/resumed file failed `gzip -t` with invalid compressed data.  Because the archive is not cleanly extracted, I did not use it as code evidence in the comparison below.

## Formula-Level Comparison

### 1. Dielectric Data Are Matrix-Valued

RESPACK explicitly treats the dielectric object as a matrix in a `q+G` basis.  Its manual says the binary output stores the inverse dielectric matrix `epsilon^{-1}_{GG'}(q, omega)`, and the text-file diagnostics store diagonal terms versus `|q+G|`.

This supports our earlier interpretation of `epsilon_vs_q`: a scatter with many values at apparently similar radial `q` is not automatically wrong.  It is the flattened diagonal of a matrix-valued object indexed by `(q_tilde, Q)`, not a single scalar function of `|q|`.

Implication for our code:

- The current diagnostic distinction is sound:
  - flattened matrix diagnostics: `epsilon_vs_q.*`, `crpa_epsilon_diagnostics.csv`;
  - representative single-curve plot: `epsilon_fig1e_window.*`.
- A "band" in the `epsilon(q)` plot can still signal aliasing or wrong basis selection, but multiplicity by itself is expected.

### 2. cRPA Target-Subspace Exclusion Is the Most Important Conceptual Difference

RESPACK's default band-cRPA path does not simply delete all transitions between selected band indices.  It reads a Wannier target-subspace weight `prob(band,k)` from `dir-wan/dat.umat` and weights each occupied-to-virtual transition by

```text
1 - P_OCC(j_band,k) * P_VIR(i_band,k+q)
```

in the polarizability accumulation.

Current `Mean_Field` code instead uses a hard flat-band mask:

```python
flat_flat = left_flat[:, None] & right_flat[None, :]
if pair_mode == "constrained":
    lindhard[flat_flat] = 0.0
```

Interpretation:

- If the target subspace is exactly the two BM flat eigenbands at each `k`, the hard mask is the clean continuum analogue of a projector with weights 0 or 1.
- If Zhang's cRPA definition instead excludes a smooth projected active subspace with remote-flat hybridization weights, then our hard `flat_flat` exclusion is not the same operation.
- This is a real physics-first audit target.  It is more credible than retesting bare HF, because the memory/project evidence says no-cRPA HF already reproduces many paper-level checks.

Recommended next audit:

- Re-read Zhang supplement around the definition of the constrained polarizability and decide whether the excluded subspace is "flat bands by eigenvalue index" or a projector-defined target subspace.
- If it is projector-defined, implement or prototype a weight-factor exclusion analogous to RESPACK's `1 - P_occ P_vir`, using the active flat-subspace projector appropriate for the continuum model.

### 3. Spin, k-Sum, and Volume/Area Normalization

RESPACK shows the following structure:

- A `1/NTK` factor appears during the k-sum accumulation.
- The accumulated `chiqw` is divided by `VOLUME`.
- A factor of 2 for spin is applied before building the dielectric matrix.

Current `Mean_Field` structure:

- `_compute_chi0` uses `prefactor = spin_degeneracy / grid.nk`.
- The two valleys are explicitly summed through `solution.n_eta`.
- The Coulomb table supplies the moire-cell area normalization through the existing TBG `coulomb_unit`.

Interpretation:

- RESPACK does not support changing our spin factor from 2 to 1.  It is another independent code where a spin factor 2 is present.
- A residual factor-of-two error is still possible if valley/spin/basis degeneracies are double-counted relative to Zhang's continuum convention, but RESPACK does not make that the leading conclusion.
- The more precise audit is dimensional: verify that local `chi0` has units `1/meV`, local `V(q)` has units `meV`, and their product is dimensionless, with the moire-cell area appearing exactly once.

### 4. Sign and Matrix Orientation

RESPACK constructs a symmetrized dielectric matrix with the visible convention

```text
epsilon = 1 - chi0 * Coulomb_kernel
```

and then inverts it.  The current local code has

```python
epsilon = I + chi0 @ diag(V)
screened_v = diag(V) @ epsilon_inv
```

This is not by itself a sign bug.  In the local code the static Lindhard factor is positive for an occupied-to-empty transition because it computes

```text
(f_left - f_right) * (E_right - E_left)
```

with both factors negative for an occupied right state and empty left state.  In other words, local `chi0` is the positive screening susceptibility, while many RPA formulas write the physical polarizability with the opposite sign.

The matrix side is also convention-dependent:

```text
W = V (I + chi V)^(-1)
```

is algebraically equivalent to

```text
W = (I + V chi)^(-1) V
```

when the inverses exist.  The current `diag(V) @ epsilon_inv` convention is therefore not contradicted by RESPACK.

### 5. Full Matrix W Versus Diagonal Fock Scalar

RESPACK's `calc_intW` projects the full inverse dielectric matrix into screened Wannier interactions schematically as

```text
rho_i(G) * epsilon_inv(G,G') * rho_j(G')
```

Current `Mean_Field` has a split convention documented in `plan/crpa工作文档.md`:

- Hartree uses the full non-diagonal screened matrix at `q_tilde = 0`.
- Fock uses a diagonal scalar lookup

```text
epsilon_cRPA(q) = Re[epsilon(q_tilde)]_{Q,Q}
V_F(q) = V_bare(q) / epsilon_cRPA(q)
```

This difference is not automatically a bug, because it follows the local Zhang-workflow contract.  But it is a useful warning: if legal no-alias artifacts and target-subspace normalization are settled and the HF+cRPA bands are still qualitatively wrong, the next paper-level question is whether Zhang's Fock insertion really uses only the diagonal scalar epsilon or a fuller local-field screened interaction.

### 6. Periodic Boundary / q-Wrapping

RESPACK explicitly tracks `k+q` wrapping and a reciprocal shift when searching the shifted k point.  This is consistent with our `hf_periodic` convention requiring `Q + wrap` form factors and therefore supports the no-alias gate:

```text
crpa_lg >= q_lg + 2
```

for the current endpoint-including HF-compatible production path.

## Current Verdict

No external-reference evidence points back to the already validated bare HF code as the leading suspect.

The strongest cRPA-specific audit target from this comparison is the definition of the excluded target subspace in `compute_constrained_chi0`:

1. If Zhang excludes exactly flat-band eigenstate pairs, the current hard `flat_flat` mask is physically consistent.
2. If Zhang excludes a projector/Wannier-weighted target subspace, the current hard mask is not equivalent; a RESPACK-like `1 - P_left P_right` weighting is the right diagnostic/prototype direction.

The next strongest audit target is units/normalization: spin factor 2 is not suspicious by itself, but the product of explicit valley sum, k-average, moire-cell area in `V(q)`, and BN/double-gate convention must be checked as one dimensionless chain.

## Practical Next Steps

1. Use the extracted RESPACK files only as formula references, not as a build artifact.
2. Do not use the incomplete Materials Cloud archive as evidence until it passes `gzip -t`.
3. Re-check Zhang's constrained-polarizability definition before adding any new numerical branch.
4. If the paper definition is projector-weighted, add a small-system diagnostic that compares hard flat-flat exclusion against a projector-weighted exclusion on the same legal no-alias artifact.
5. Keep acceptance based on SCF-grid point line plots and direct-gap/bandwidth summaries, not reconstructed path bands.

## 2026-05-27 Source-Audit Update: Fock Scalar Uses `epsilon_inv`

The earlier warning in section 5 turned into a concrete source fix.  The local
cRPA artifact defines the screened interaction as

```text
screened_v = diag(V) @ epsilon_inv
```

Therefore a diagonal Fock scalar approximation to the same screened
interaction must use

```text
W_QQ = V_Q * Re[(epsilon_inv)_QQ]
epsilon_Fock(Q) = 1 / Re[(epsilon_inv)_QQ]
```

The old production lookup used

```text
epsilon_Fock(Q) = Re[epsilon_QQ]
```

which is equivalent only when the dielectric matrix is diagonal.  It is not the
diagonal of the stored screened interaction once local-field off-diagonal terms
are present.

Source change:

- `src/mean_field/crpa/screened_coulomb.py`: `CRPAScreenedCoulomb.fock_epsilon_array()` now returns the scalar divisor implied by `epsilon_inv`.
- `src/mean_field/crpa/hf_interface.py`: docstring updated so the production Fock path is described as the `diag(V) @ epsilon_inv` scalar divisor, not the raw dielectric diagonal.
- `scripts/make_crpa_fock_wdiag_artifact.py`: clarified that this script is now only a plotting/legacy diagnostic; it no longer changes production Fock behavior.
- `tests/test_crpa_core.py`: added a non-diagonal dielectric test where `Re[epsilon_QQ] != 1 / Re[(epsilon_inv)_QQ]`.

Existing artifacts do not need regeneration for this fix because `epsilon_inv`
is already stored.

Observed size of the correction on existing artifacts:

```text
hf_periodic_lk6_lg7_q5:
  median Re diag(epsilon) / [1/Re diag(epsilon_inv)] = 1.003
  max ratio = 1.032
  median relative offdiag norm = 0.118

hf_periodic_lk8_lg9_q7_regular1h:
  median ratio = 1.000
  max ratio = 1.027
  median relative offdiag norm = 0.085

crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged:
  median ratio = 1.960
  max ratio = 3.071
  median relative offdiag norm = 0.515
```

Interpretation:

- On legal small no-alias artifacts this Fock-scalar fix is physically required
  but numerically small, so it probably does not explain the remaining small
  bandwidth by itself.
- On the old `lg9/q11` merged artifact it changes the effective Fock screening
  by order one, which is another reason that artifact cannot be used for final
  physics.
- The next source-level suspect remains the constrained-polarizability target
  subspace definition: hard flat-flat exclusion versus projector-weighted cRPA
  exclusion.

Verification:

```text
/data/home/ziyuzhu/miniconda3/bin/python -m pytest tests/test_crpa_core.py
44 passed
```
