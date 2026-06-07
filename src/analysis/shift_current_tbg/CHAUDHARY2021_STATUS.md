# Chaudhary 2021 reproduction status

## Scope

Reference:

```text
reference/Chaudhary 等 - 2021 - Shift-current response as a probe of quantum geometry and electron-electron interactions in twisted.pdf
```

Current focus: quantum geometry first.  The `|A|^2 S` shift-vector integrand is now validated against an independent gauge-invariant finite-difference link formula; only after this audit should spectra/maps be compared with the paper.  The selected-pair response path now uses the common WannierBerri-style gauge-safe derivative helper.  Hartree Fig. 4 remains a model-convention problem, likely involving chemical-potential/edge, screening, or SCF conventions rather than the response derivative formula.

## Important corrections made

1. **Band model correction**

   The initial `atmg` shell-gauge TBG adapter gave a wrong Fig. 2(a)-style band structure.  It has been superseded by the repository's previous b0 BM model:

   ```text
   mean_field.systems.tbg.zero_field
   ```

   Corrected path:

   ```text
   kappa' -> gamma -> kappa -> mu -> kappa
   ```

2. **FD direct-transition correction**

   The first corrected-b0 response used `fd_mode=all`, mixing same-side and cross-gap flat--dispersive direct transitions.  This gave a spurious large FD signal at charge neutrality.

   Paper-like Fig. 2 FD direct transitions use:

   ```text
   fd_mode = same_side
   lower dispersive -> valence flat
   conduction flat -> upper dispersive
   ```

   This makes the FD response Pauli-blocked at neutrality, as stated around Fig. 2(e).

## Current tracked code surface

The Chaudhary/TBG shift-current workspace now keeps reusable code in modules and retires the historical per-panel `run_*` / `plot_*` scripts from the tracked package surface.  Historical commands and output paths remain in this status file, reports, result metadata, and git history; they should not be restored as standalone scripts unless the workflow is promoted into a dispatcher-backed reusable tool.

Current reusable modules:

```text
chaudhary2021.py
hartree.py
../response_derivative_gauge.py
```

The validated reusable response piece is the common WannierBerri-style derivative path in `../response_derivative_gauge.py`. Future production response workflows should call that common layer and should be exposed through `scripts/mean_field_tools.py`, `src/mean_field/cli.py`, or a curated devtool instead of adding another tracked paper-panel script.

## Current outputs after cleanup

See the concise index:

```text
results/shift_current_tbg/CURRENT_RESULTS.md
```

Active top-level result directories are now limited to:

```text
results/shift_current_tbg/chaudhary2021_quantum_geometry_audit_lg7_m55/
results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3/
results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_edgegap_mu_lg7_m20_c3/
results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3_wannierberri_sceta{5,10,20,40}/
results/shift_current_tbg/chaudhary2021_hartree_bands_paperfill_lg7_m9_eps{10,15,20,30}_T15K/
results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps{10,15,20,30}_T15K/
results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K_firststar_linearpm_wannierberri_sceta{5,10,20,40}/
results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15states_T0occ_firststar_linearpm_{sumrule,wannierberri_sceta*}/
results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m12_c3_eps10_T15K_edgegap_mu/
results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps{10,15,20}_edgegap_lg7_m12_c3/
results/shift_current_tbg/chaudhary2021_paperstyle_comparison/
results/shift_current_tbg/chaudhary2021_paperstyle_comparison_wannierberri_sceta40/
```

Older convergence scans, tensor/delta diagnostics, first-generation comparisons, and older integrand-map attempts were moved to:

```text
results/shift_current_tbg/_archived_tests_and_diagnostics_20260601_qg_cleanup/
```

The earlier sharp-occupation `T=0` Hartree production diagnostics did not converge on the coarse mesh and remain archived at:

```text
results/shift_current_tbg/_archived_tests_and_diagnostics_20260531/hartree_T0_nonconverged/
```

## Quantum geometry audit

Current authoritative output:

```text
results/shift_current_tbg/chaudhary2021_quantum_geometry_audit_lg7_m55/
```

This run validates the actual paper integrand before any visual paper comparison:

```text
R^{xxy}_{mn} = |A^x_{mn}|^2 S^{yx}_{mn}
             = Im[ A^x_{mn} (A^x_{nm})_{;y} ]
```

The gauge-free Hamiltonian-derivative sum rule matches an independent gauge-invariant finite-difference link formula for the shift vector.  Selected-pair calls in `analysis.shift_current_htg.response` route through `analysis.response_derivative_gauge`, so current TBG/HTG production selected-transition weights use the same validated helper.  The helper now also exposes the WannierBerri/Wannier90 `sc_eta` principal-value regularizer for intermediate denominators, needed for near-degenerate FD diagnostics.  Audit error on deterministic FF/FD points:

```text
max_abs_error ~4.7e-3 nm
median_abs_error ~2.0e-4 nm
```

Important unit convention: the direct link derivative is naturally in the old b0 dimensionless momentum; multiply by graphene lattice constant `a=0.246 nm` before comparing to the sum-rule shift vector in nm.

Paper Fig. 2(d,e) comparison file:

```text
paper_vs_ours_quantum_geometry_maps.png
```

Current map conclusion: FF sign structure is qualitatively reasonable; FD is Pauli-blocked at neutrality and has the expected electron/hole sign reversal, but the paper's broader triangular Gamma pocket is not yet quantitatively reproduced.  A mechanism audit now shows this is not mainly Pauli blocking or transition-energy contour size: the explicit `mu=±30 meV` Pauli mask is active, and the resonant transition-energy region is broad enough.  The mismatch is localized to near-Gamma quantum geometry with near-degenerate dispersive doublets; exact selected-pair `y;xx` is dominated by a Gamma spike.  WannierBerri-style `sc_eta` principal-value regularization and FD doublet summation broaden/suppress the spike, but the production `sc_eta=5,10,20,40 meV` spectra show a tradeoff: damping the 26 meV low-energy FD feature also damps the 45--80 meV high-energy Hartree FD feature.  Changing only the response occupation from 15 K to 0 K also does not fix Fig. 4(c).  This is not a complete Fig. 2(e)/Fig. 4(c) fix.

## Main noninteracting findings

### FF contribution

For `Delta1=Delta2=5 meV`, `theta=0.8 deg`, corrected b0 model:

```text
FF nu=0 peak ~1.3e3 microA nm V^-2 near 9--10 meV
```

This matches the intended Fig. 2 scale/order.

### FD filling-labelled same-side contribution

With `fd_mode=same_side`, FD is zero at neutrality:

```text
FD nu=0 = 0
```

For finite flat-band filling, same-side FD appears at tens of meV.  With `fd_bands=10`, peaks are around `~80 meV` and `~5--6e3 microA nm V^-2`; with nearest dispersive `fd_bands=1`, peaks are `~0.8--1.1e3 microA nm V^-2` around `~54--71 meV`.

### Explicit chemical-potential FD cuts

For Fig. 2(e)-style chemical potentials:

```text
mu=-30 meV: FD peak ~ -1.74e5 microA nm V^-2 at 21.2 meV
mu=0 meV:   FD = 0
mu=+30 meV: FD peak ~ +1.53e5 microA nm V^-2 at 21.6 meV
```

The **shape and energy** match the expected Fig. 2(e) logic: opposite-sign peaks for electron/hole chemical potentials and zero at neutrality.  A postprocessing audit shows that using per-flavor response degeneracy `1` and a wider Lorentzian `eta~10 meV` gives peaks of order `~9e3 microA nm V^-2`, consistent with the plotted Fig. 2(e) scale.  The filling-to-chemical-potential map should still use total spin/valley degeneracy `4` for the paper's `nu` labels.  The b0 runner now separates these as `--degeneracy` (response multiplier, default `1`) and `--filling-degeneracy` (default `4`).

## Delta scan / Supplement S1-style trend

For `Delta=5,10,20 meV` with FF at `mu=0` and nearest same-side FD at `mu=+30 meV`:

```text
Delta=5 meV:  FF peak ~1.30e3 at 9.6 meV;  FD peak ~2.33e5 at 22.2 meV
Delta=10 meV: FF peak ~8.70e2 at 17.0 meV; FD peak ~1.43e5 at 21.2 meV
Delta=20 meV: FF peak ~3.63e2 at 29.6 meV; FD peak ~1.12e5 at 18.2 meV
```

The qualitative trend matches the Supplement S1 discussion: increasing `Delta` shifts the FF response to higher frequency and suppresses it, while the FD peak shifts lower and is also suppressed.  The FD absolute amplitude remains larger than the paper plot and still needs a normalization/broadening audit.

## Hartree prototype

Implemented a full-continuum Hartree-only prototype in reusable module form:

```text
hartree.py
```

The historical Hartree band/response launch scripts were retired from the tracked package surface during cleanup. Treat archived command lines as provenance only; future reruns should use a dispatcher-backed tool or recover reusable logic into `hartree.py` / `chaudhary2021.py`.

This is intentionally separate from the repository's older active-band projected HF code.  The prototype constructs Fourier components of the Hartree potential from the two flat-band wavefunctions, with the density measured relative to charge neutrality, and applies the resulting scalar potential in the full local continuum basis.  This keeps `dH/dk` equal to the continuum kinetic derivative, so the existing shift-current formula remains usable.

A local `lg5/m5` smoke test already showed the expected Chaudhary Fig. 3 trend:

```text
nu=+2: electron flat bandwidth decreases ~16.5 -> ~6.9 meV, hole flat bandwidth broadens ~16.2 -> ~25.8 meV
nu=-2: hole flat bandwidth decreases ~16.2 -> ~6.7 meV, electron flat bandwidth broadens ~16.5 -> ~26.0 meV
nu=0: Hartree potential is zero by construction because density is measured relative to CNP
```

The smoke test was archived under `_archived_tests_and_diagnostics_20260531/hartree_smoke/`.  A converged `15 K` production diagnostic is now available at `lg7/m9` for Hartree bands and `lg7/m10/C3` for response.  The high-temperature occupation is consistent with the paper's stated regime above correlated-transition temperatures and removes the coarse-mesh sharp-Fermi-surface SCF oscillations.

`T=15 K`, `epsilon_r=15`, `lg7/m9` band diagnostics:

```text
nu=-3: hole flat bandwidth 16.25 -> 3.98 meV; electron flat bandwidth 16.49 -> 29.52 meV
nu=-2: hole flat bandwidth 16.25 -> 6.72 meV; electron flat bandwidth 16.49 -> 26.01 meV
nu=-1: hole flat bandwidth 16.25 -> 11.25 meV; electron flat bandwidth 16.49 -> 21.49 meV
nu=+1: hole flat bandwidth 16.25 -> 21.25 meV; electron flat bandwidth 16.49 -> 11.49 meV
nu=+2: hole flat bandwidth 16.25 -> 25.81 meV; electron flat bandwidth 16.49 -> 6.92 meV
nu=+3: hole flat bandwidth 16.25 -> 29.36 meV; electron flat bandwidth 16.49 -> 3.95 meV
```

All finite-filling `15 K` Hartree SCF runs converged to `final_error < 1e-5` within `25--26` iterations.

`T=15 K` response diagnostics show the expected interaction enhancement and low-energy FD peak transfer:

```text
FF |nu|=2: nonint ~1.0e3 at 16 meV -> Hartree ~6.0e3 at 13.5 meV
FD |nu|=2: nonint ~1.4e3 at 80 meV -> Hartree ~3.8e3 at 25 meV
FD |nu|=3: Hartree ~2e4 at 26--27 meV, plus higher-energy residual structure
```

These are strong qualitative Fig. 1/Fig. 4 signatures, but not final quantitative reproduction until dielectric/screening and FD transition-set conventions are audited.

Response convergence and epsilon scans now exist:

```text
m10 -> m12 response mesh:
  FF peaks stable; largest FD peaks shift little in energy but drop by ~20--30%.

fd_bands=1 vs 10 at m12:
  nearest-dispersive fd_bands=1 captures most of the strongest low-energy Hartree FD peak for |nu|=2,3,
  but loses broader remote-band/high-energy structure.

epsilon_r scan (10,15,20,30):
  |nu|=2 same-side flattened bandwidth grows from ~4 meV at eps=10 to ~11.5 meV at eps=30;
  FF enhancement decreases with weaker Hartree interaction;
  FD has competing low/high-energy peaks, so global max-abs signs/locations can switch.
```

## Tensor symmetry audit

For `Delta1=Delta2=5 meV`, FF at neutrality obeys the D3 one-coefficient pattern in the code axes:

```text
x;xy = x;yx = y;xx = - y;yy ~ 1.3e3
x;xx, x;yy, y;xy, y;yx ~ 0
```

For `Delta1=5 meV`, `Delta2=10 meV`, the second C3-allowed tensor group becomes nonzero, consistent with the D3 -> C3 symmetry lowering discussed in Appendix A.

## Slurm jobs

```text
134849 completed  corrected b0 but old fd_mode=all, archived
134856 completed  old fd_mode=all convergence, archived
134883 completed  fd_mode=same_side support runs
134902 completed  fd_bands=1 nearest-dispersive run
134909 completed  explicit mu=-30,0,+30 meV FD run
134912 completed  tensor symmetry audit
134914 completed  Delta scan
134941 completed  k-space integrand maps
134992 cancelled while pending  Hartree full-continuum bands on regular6430
134997 cancelled while pending  dependent Hartree response on regular6430
134998 completed  T=0 Hartree full-continuum bands on regular, archived as nonconverged diagnostic
134999 completed  T=0 Hartree shift-current response on regular, archived as nonconverged diagnostic
135001 completed  T=15K Hartree full-continuum bands on regular
135002 completed  T=15K Hartree shift-current response on regular
135034 completed  T=15K Hartree response m12/fd_bands convergence
135057 completed  T=15K Hartree epsilon scan
```

## Additional physics audit after paper-style comparison

A dedicated physics note has been added:

```text
CHAUDHARY2021_PHYSICS_AUDIT.md
```

Key conclusions:

- The giant noninteracting Fig. 4(b) FD edge peak is a Pauli-blocking/chemical-potential convention issue.  Literal filling-derived `mu(nu=+-3.95)` keeps the Gamma flat-band pocket occupied/unoccupied and blocks the singular Gamma FD transition.  Edge labels must be placed in the flat-dispersive gap to recover the paper-like peak.
- The Hartree FD mismatch is not fixed by FD-pair inclusion.  Pair/region decomposition shows low-energy FD is Gamma-region and nearest same-side pair, while high-energy/opposite-sign FD is outer-BZ and mostly next same-side pair.

New diagnostics:

```text
results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps10_edgegap_lg7_m12_c3/
results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps15_edgegap_lg7_m12_c3/
results/shift_current_tbg/chaudhary2021_hartree_fd_decomp_eps20_edgegap_lg7_m12_c3/
results/shift_current_tbg/chaudhary2021_paperstyle_comparison/hartree_fd_pair_region_decomposition.md
```

## Hartree density-mode audit

The first screening/density-convention audit is now done:

```text
results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K/density_mode_audit_three_modes.md
results/shift_current_tbg/chaudhary2021_hartree_density_mode_audit_lg7_m5_eps10_T15K/density_mode_audit.md
```

Implemented diagnostic modes:

```text
--density-mode full_delta_occ
--density-mode full_fixed_cnp
```

Definitions:

- `full_delta_occ`: diagonalize the full continuum Hamiltonian but subtract a same-Hamiltonian CNP occupation, so the filled remote sea cancels.  This tests carrier leakage/band-index convention without explicit remote screening.
- `full_fixed_cnp`: full occupied density in the truncated continuum basis minus a fixed noninteracting CNP reference.  This explicitly includes remote-band wavefunction polarization and may double-count screening.

Main conclusion on the coarse `m5` audit:

- `full_delta_occ` is nearly identical to the default `flat` source in both bands and response, so the doped carriers remain in the central flat pair and carrier leakage is not the issue.
- `full_fixed_cnp` is convergent and controlled, but it weakens same-side band flattening and reduces the Fig. 3(b)-style FF enhancement.
- `full_fixed_cnp` does not create the missing strong high-energy/opposite-sign Fig. 4(c) FD feature; the largest edge FD remains low-energy/Gamma-like.
- Therefore a blind replacement of the flat-only Hartree source by full fixed-CNP density is not the answer.  The paper's statement that dispersive bands renormalize the effective dielectric remains the more plausible convention.

## Ref. 65 / Choi Hartree screening-convention audit

Downloaded and inspected the Choi Ref. 65 supplementary information:

```text
tmp/pdfs/choi2021_ref65/supplement.pdf
tmp/pdfs/choi2021_ref65/supplement.txt
```

Relevant documented conventions:

- `u0=90 meV`, `u/u0=0.4` (already used here).
- Effective dielectric `epsilon=15`, fitted to STM/LL data at `theta=1.32 deg` and then kept fixed.
- Hartree Fourier components restricted to the first moiré reciprocal star: `(±1,0)`, `(0,±1)`, `(±1,±1)`.
- Ref. 65 often linearly interpolates Hartree/HF potentials from the `nu=±4` self-consistent endpoints.

Implemented/audited:

```text
--hartree-shift-mode first_star
results/shift_current_tbg/chaudhary2021_paperstyle_comparison/hartree_firststar_eps15_audit.md
```

Main conclusion:

- First-star eps15 changes bands/response moderately relative to all-shift eps15, but does not fix the simultaneous FF / low-energy-FD / high-energy-FD mismatch.
- Linear interpolation from the edge first-star potentials reduces FF enhancement and leaves the low-energy FD peak large; it is not the missing Fig. 4(c) ingredient.

## Remaining before final claims

1. The remaining Hartree mismatch is not solved by density source, first-star truncation, or Ref. 65-style edge interpolation.  Next likely suspects are exact paper chemical-potential convention for Hartree FD edge labels, possible Fock/strain/band-parameter differences, or a residual response/map convention.
2. If quantitative Hartree FD amplitudes are needed, run one higher response mesh (`m14`) or improve quadrature; `m12` still changes the largest FD peak by `~20--30%`.
3. Clarify whether paper Fig. 2(c) uses nearest dispersive bands only or a broader same-side remote-band set.
4. Refine k-space map coordinates/component convention if exact Fig. 2(d,e) visual matching is needed.
