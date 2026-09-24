# Du2017 Fig. 2 pair-curvature and Hamiltonian-channel audit

**Date:** 2026-08-07
**Scope:** Du main paper, Supplementary Notes 2/6, Du thesis, Li–Yang–Chang Kane lineage, and preserved TEIB/senior-code lineage only
**Status:** source/logic audit complete; no new parameter scan authorized by this report

## Question

Can an additional physically motivated `C*k^2` term reproduce Fig. 2 while keeping `epsilon_r=15` and every Coulomb-channel scale equal to one?

## Short answer

Not from the currently available Du lineage.

1. The source two-band model already contains the mass-derived pair kinetic term

   \[
   c_{\rm pair}k^2,
   \qquad
   c_{\rm pair}=\frac{\hbar^2}{2m_0}
   \left(\frac1{m_e}+\frac1{m_h}\right)
   =1.4707651566\ {\rm eV\,nm^2}
   \]

   for `m_e=0.032 m0`, `m_h=0.136 m0`.

2. A BHZ **identity-channel** term such as `C-D*k^2` cancels from the electron–hole pair detuning and cannot directly repair `E_pair=sqrt(xi_pair^2+Delta^2)` in the balanced zero-temperature scalar model.

3. A quadratic term that changes the E1–H1 separation is a **detuning/mass channel** (`tau_z*k^2`, conventionally related to BHZ `B`), not the BHZ common-energy `C` channel. Its leading coefficient is already supplied either by the reported effective masses or by using the full Kane E1/H1 dispersions directly.

4. The restored senior bundle's additional `16.60946 eV nm^2` term is explicitly a calibrated pair-dispersion correction. It is absent from the paper, supplement, thesis, cited Kane method, and preserved no-scale TEIB source.

5. Existing unit-Coulomb calculations have already tested both source-derived alternatives—mass-derived parabolic dispersion and direct Kane dispersion—and neither reproduces Fig. 2. Adding another fitted quadratic term would double-modify the normal dispersion rather than add missing published physics.

## 1. Channel algebra

Write a normal two-band ordinary-electron Hamiltonian as

\[
H_0(k)=\epsilon_0(k)\,\mathbb 1+d_z(k)\,\tau_z+H_{\rm mix}(k),
\]

and temporarily set `H_mix=0` to identify diagonal channels. Then

\[
E_c=\epsilon_0+d_z,\qquad E_v=\epsilon_0-d_z.
\]

For the fixed ordinary-electron chemical-potential convention,

\[
\xi_{\rm pair}=E_c-E_v=2d_z,
\]

\[
\eta_{\rm charge}=E_c+E_v-2\mu_{\rm el}=2(\epsilon_0-\mu_{\rm el}).
\]

Therefore:

| normal-state term | change in `xi_pair` | change in `eta_charge` | direct effect on balanced scalar `E_pair` |
|---|---:|---:|---|
| `delta C(k) * I` | `0` | `2 delta C(k)` | none unless blocking/charge asymmetry is active |
| `delta B(k) * tau_z` | `2 delta B(k)` | `0` | yes; changes pair detuning |
| common energy-zero shift | `0` | shift absorbed with same-gauge `mu_el` | none |

For a BHZ block,

\[
h(k)=[C-Dk^2]\mathbb 1+[M-Bk^2]\tau_z+A(k_x\tau_x-k_y\tau_y),
\]

so the `C-D*k^2` identity term is not a new pair-curvature knob. The interband splitting is

\[
E_+-E_-=2\sqrt{(M-Bk^2)^2+A^2k^2},
\]

which is independent of `C-D*k^2`.

The senior code instead applies

```python
electron_bare += 0.5 * pair_dispersion_correction
hole_bare     += 0.5 * pair_dispersion_correction
```

(`reference/teib_calibrated_hf_bcs_20260728_1755_unpacked/teib_calibrated_hf_bcs_20260728_1755/src/excitonic_bcs.py`, lines 964–979).

Because `hole_bare` is a **hole excitation energy**, this raises `epsilon_a+epsilon_b`, i.e. it changes the E1–H1 pair detuning. In an ordinary-electron `(E1,H1)` basis the same operation corresponds to opposite shifts of `E_c` and `E_v`, hence a `tau_z`-type correction—not a common BHZ `C*k^2` term.

## 2. Source authority

### Du main paper and Supplementary Note 2

The main paper reports `m_e≈0.032 m0`, `m_h≈0.136 m0`, `epsilon≈15`, and `d≈10 nm`, but uses the masses explicitly for a Bohr-radius estimate. It does not certify them as the exact Fig. 2 scalar-band masses. Supplementary Note 2 gives generic `epsilon_a(k)` and `epsilon_b(k)` but no numerical arrays, quadratic coefficient, cutoff, or chemical potential.

See `docs/du2017_paper_logic.md`, lines 7–25.

### Du thesis

The thesis describes an isotropic parabolic two-band ancestor and prints the Littlewood/Zhu equations. Its physical pair kinetic energy is `E_e(k)+E_h(k)`, which yields the mass-derived `c_pair*k^2`. No second quadratic correction is printed.

See `docs/equations.md`, the “Du-thesis zero-temperature backbone” and pair-curvature paragraphs.

### Supplementary Note 6 and cited Kane method

Supplementary Note 6 supplies an 8-band Kane–Poisson model with material parameters `A_c`, `gamma_i`, `E_g`, `Delta`, etc. It does not supply BHZ `B,C,D`, nor a scalar post-Kane `C*k^2` correction. If the Kane E1/H1 bands are used directly, their nonparabolicity and all higher powers of `k` are already present; adding a separate low-energy curvature without a downfolding subtraction would double count.

The controlled reconstructed low-k E1/H1 diagonal curvatures are

```text
E1 ordinary-electron curvature = -85.75 meV nm^2
H1 hole curvature              = 1365.81 meV nm^2
pair sum                         = 1280.06 meV nm^2
```

(`docs/du2017_paper_logic.md`, lines 83–101). Their pair sum is already of the same order as the mass-derived `1470.77 meV nm^2`; it is not missing a `16609.46 meV nm^2` term.

A target-free nested-window trace audit now makes this sharper. On `kmax=0.012...0.060 nm^-1`, quartic fits give stable leading coefficients

```text
c_xi  = +1.28332 ... +1.29029 eV nm^2
c_eta = -1.46177 ... -1.45480 eV nm^2.
```

At `kmax=0.06 nm^-1`, `c_xi,4=-22.1992 eV nm^4`; its contribution is about `-0.288 meV`, versus the quadratic `+4.620 meV`. The quartic term resolves smooth source nonparabolicity but is already contained in the full Kane bands. Receipt: `runs/du2017_lowdin_trace_curvature_20260807/SCIENTIFIC_STATUS.md`.

The publication does not state that Note 6 exported these bands to Note 2. Thus direct Kane use is a transparent reconstruction hypothesis, not a recovered author interface.

## 3. Preserved TEIB/senior lineage

### No-scale lineage

The July-20 preserved source

```text
reference/teib_20260720_unpacked/teib/src/excitonic_bcs.py
SHA-256 929ccdaa918582b06674de0a9a3c1f28c0ff4c152c64b94938a2b82bedf32a20
```

contains no `pair_curvature` or empirical pair-dispersion correction. Its direct Kane path consumes converted Kane bands.

A later `current_uncalibrated_fixed_density` run is preserved by log and validation, although its exact source bytes are not present in the restored archive. Its logged source hash for `src/excitonic_bcs.py` is

```text
fbb8e3f61f0ad5c47559290611e4535d8fa7ac23999c7c624a2ced92be1ffb5d
```

(`direct_current_uncalibrated_20260728_1750.log`, lines 1–11). The run used unit Coulomb, `epsilon_r=15`, no calibrated parameters, and produced

```text
E(0)       = 8.604589 meV
E_min      = 3.626822 meV
k_min      = 0.0598336 nm^-1
Delta_max  = 3.741073 meV
```

(`validation_current_uncalibrated_20260728_1750.json`, lines 1–12 and 104–134).

The archive-level `SOURCE_SHA256SUMS.txt` is stale relative to both later run logs and the active restored calibrated source, so it is not authority for either final execution.

### Restored calibrated lineage

The restored source defines

\[
\delta\epsilon_{\rm pair}(k)=16.60946k^2
+0.01840197\left[e^{-(k/0.02869179)^2}-1\right]
-0.00015R_{0.025,0.008}(k)
\]

(`src/excitonic_bcs.py`, lines 270–311) and adds half to each electron/hole excitation branch (lines 964–979).

The calibrated execution log records, simultaneously,

```text
pair mu              = -37.61102 meV
intralayer scale      = 4.2
interlayer scale      = 2.09
diagonal scale        = 0.782
pair curvature        = 16.60946 eV nm^2
Gaussian/ring terms   = enabled
pairing vertex        = enabled
```

(`direct_calibrated_historical_20260728_1800.log`, lines 12–34).

This is a multi-parameter calibrated reconstruction. The separate quadratic coefficient has no independent Du-source provenance.

## 4. Existing discriminating calculations

No additional solver run is needed to decide whether a currently source-backed quadratic term was omitted:

| source-derived normal dispersion | closure/interaction | result | status |
|---|---|---|---|
| mass-derived parabolic `c_pair*k^2` | fixed density, unit Coulomb, infinite domain | `E(0)=9.396`, `E_min=3.230`, `Delta_max=3.296 meV` | converged, disagrees |
| same, hard cutoff | `kmax=1.10 kF` | `E(0)=9.456`, `E_min=1.708`, `Delta_max=1.762 meV`; active/incomplete boundary | forensic cutoff cannot reproduce full curve |
| same, fixed pair `mu` plus cutoff | best paired physical-axis case | `E(0)=6.515`, `E_min=2.778`, `Delta_max=2.909 meV` | wrong anti-correlated tradeoff |
| direct no-scale Kane bands | fixed density, unit Coulomb | `E(0)=8.605`, `E_min=3.627`, `Delta_max=3.741 meV` | converged, disagrees |
| legacy direct Kane bands | dynamic two-channel `mu`, scale 1 | `E(0)=8.448`, `E_min=3.578`, `Delta_max=3.701 meV` | converged, disagrees |
| fixed-common-`mu` scalar controls | unit Coulomb | normal or `Delta_max=5.202`, `E_min=4.606 meV` depending truncation | no Fig. 2 root |
| split-zero Löwdin trace channels | no interaction or target; nested `k^2+k^4` fit | `c_xi≈1.283–1.290 eV nm^2`; stable Kane nonparabolicity | already included in direct Kane route |

Primary receipts:

- `results/du2017_inas_gasb_excitonic_insulator/runs/du2017_thesis_hard_cutoff_unit_coulomb_20260807/SCIENTIFIC_STATUS.md`
- `results/du2017_inas_gasb_excitonic_insulator/runs/du2017_thesis_fixed_mu_hard_cutoff_unit_coulomb_20260807/SCIENTIFIC_STATUS.md`
- `reference/.../validation_current_uncalibrated_20260728_1750.json`
- `results/teib_reference_reproduction_20260721/SENIOR_FIG2_NONFIT_AUDIT_20260722.md`, lines 23–41
- `results/du2017_inas_gasb_excitonic_insulator/runs/du2017_lowdin_trace_curvature_20260807/SCIENTIFIC_STATUS.md`

These paths already include the leading physical quadratic dispersion or the full Kane dispersion. A new polynomial fitted to Fig. 2 would not be a missing source calculation.

## 5. Decision

### Rejected as a physical next step

- adding the senior `16.60946 eV nm^2` coefficient;
- relabeling that coefficient as a generic high-energy/BHZ `C` term;
- fitting `c2`, `c4`, Gaussian, or ring corrections to Fig. 2;
- extracting a polynomial from the full Kane bands and then adding it back on top of those same bands;
- further cutoff/`mu` scans without a new source contract.

### Allowed next step

Only one of the following can reopen a dispersion-correction calculation:

1. author code/data specifying the actual `epsilon_a(k)`, `epsilon_b(k)`, or a post-Kane correction;
2. an author-lineage derivation specifying the projection/downfolding reference and showing a non-double-counted residual operator;
3. a source-bound statement that Fig. 2 used a different retained-subband model, with enough parameters to compute it without using Fig. 2 as a fit target.

The author request now asks explicitly whether a `c2*k^2`, `c4*k^4`, or tabulated post-Kane pair-dispersion correction was used, and whether it acted in the common-energy or E1–H1 detuning channel.

## Final classification

There is no currently justified “physical `C*k^2` fix” left to test. The mass/detuning curvature is already present; the BHZ identity curvature cannot fix the balanced pair spectrum; and the only separate large quadratic term in the local lineage is calibrated. Fig. 2 reproduction remains underdetermined pending author/source clarification.
