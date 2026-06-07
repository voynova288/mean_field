# hTTG / shift-current reproduction status

Last updated: 2026-05-30

This note records what is currently reproducible after comparing the local code with reference implementations and after the hTTG symmetry audits.  It is intentionally conservative: visual alignment, post-hoc scaling, selective filtering, or fitted axis rotations are not counted as reproduction.

## Reference-code comparison

### Shift-current formula

- The local gauge-free response implementation agrees with a direct transcription of the official Wannier90/postw90 and WannierBerri shift-current internal formula for the gapped-SLG toy model.
- Historical audit script: `run_slg_toy_reference_formula_audit.py` (retired from the tracked package surface; the reusable reference-code integrand now lives in `../response_derivative_gauge.py`).
- Reference source locations inspected:
  - `reference/upstream/wannier90/src/postw90/berry.F90::berry_get_sc_klist`
  - `reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula`
  - `reference/upstream/HopTB.jl/src/optics.jl` / `src/basic.jl` for the same generalized-derivative structure.
- Representative output:
  `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_reference_formula_audit_m40/summary.json`
- Agreement: dominant components match to numerical roundoff (`~1e-14 microampere nm V^-2`).

Conclusion: the core shift-current formula in `response.py` is consistent with standard reference-code formulas.  The mismatch with Mao Fig. 10 is not fixed by switching to Wannier90/WannierBerri/HopTB conventions.

## Mao Appendix-A Fig. 10: gapped SLG toy

Status: **not honestly reproduced**.  See also `SLG_FIG10_AUDIT.md`, which compares Mao Appendix A against Hipolito, Pedersen, and Pereira, PRB 94, 045434 (2016).

What the validated calculation gives:

- Full `mu=0`, W-corrected / reference-code-consistent SLG calculation has a dominant K/K' direct-gap peak near `~3.2 eV` with `|sigma^{x;xy}|=|sigma^{y;yy}| ~ 2.3 microampere nm V^-2`.
- Mao Fig. 10 instead shows an M-point/DOS peak near `~6.2 eV` with scale `~0.1 microampere nm V^-2`.

What can mimic part of Mao Fig. 10:

- The paper-printed Eq. (4) without the tight-binding second-derivative/external completion, evaluated on a finite primitive grid, gives a blue `sigma^{x;xy}` M-point peak near `6.2 eV` with scale close to `0.1` at mesh `150`.
- This is not converged: the peak decreases from `0.313` (mesh 80) to `0.016` (mesh 300), and C3-orbit averaging makes it vanish to roundoff.
- The paper-labelled red `sigma^{y;yy}` remains zero in that calculation; the nonzero partner is `sigma^{y;xx}`.

Cross-reference paper check: Hipolito 2016 studies the same gapped honeycomb tight-binding model class and explicitly states/predicts a direct-gap onset at `hbar omega=Delta` from K-point physics.  This supports the local formal result and makes Mao's M-only-looking Fig. 10 suspicious as a full neutral tight-binding benchmark.  Local evidence figures:

- Formal W-corrected cross-reference: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_evidence/hipolito2016_crossref_evidence.png`
- Direct Hipolito Fig. 4 Re/Im reproduction is now included in the interval benchmark suite: `results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png`
- Wiggle-free transition-energy quadrature version: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_energy_quad/hipolito2016_fig4_energy_quad.png`
- Reusable Hipolito 2016 benchmark suite, including Fig. 4 and Fig. 5(b), rerun with analytic transition-energy interval denominator integration: `results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png`
- Hipolito Fig. 5(a) low-energy gap-threshold benchmark, now using analytic transition-energy interval denominator integration to remove node-crossing wiggles: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/hipolito2016_fig5a_gap_series_exact_t72/hipolito2016_fig5a_gap_series.png`
- Hipolito Fig. 5(a) full-BZ reproduction at `Gamma=1 meV`, including M-point peaks, using binned linear tetrahedra plus analytic interval denominators: `results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev/hipolito2016_fig5a_full_bz_tetra.png`

Conclusion: Mao Fig. 10 cannot currently be used as a validation gate for the response code.

## Mao Fig. 1(a): hTTG band structure / DOS

Status: **qualitatively reproduced, not yet quantitatively signed off**.

Current candidate:

- `results/shift_current_htg/bands_pathfix_132100/bands_dos/htg_bands_dos.png`

Important fixes already made:

- Mao high-symmetry path needs the extended-zone `K' = kappa_prime_m + b_m1`, not the central-zone opposite corner.
- Mao Eq. (13) is implemented with unrotated Dirac blocks by default (`zeta_rad=0.0`).

Remaining caveat:

- A digitized quantitative comparison of band energies / DOS peak locations has not yet been done, so this is a candidate rather than a final reproduction claim.

## Mao Fig. 2: central flat-band pair shift current

Status: **shape/scale are close only after a coordinate-convention diagnostic; raw code-axis paper labels are not reproduced**.

Validated raw-axis result:

- Converged shell5 central-flat calculation gives a one-coefficient C3v-like tensor:
  - C3 group1 (`x;yy = -x;xx = y;yx = y;xy`) is suppressed to `~1e-7` or below.
  - C3 group2 (`y;xx = -y;yy = x;xy = x;yx`) peaks near `~6.9e3 microampere nm V^-2` for `eta=2 meV`.
- Representative output:
  `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_132463/central_flat_allc/tensor_symmetry_audit_eta2/tensor_symmetry_audit.png`

Symmetry explanation:

- Written derivation: `HTG_SYMMETRY_AUDIT.md`.
- The implemented ABA Hamiltonian has an exact antiunitary layer-swap/conjugation symmetry:
  `H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger`, with
  `(n1,n2)->(-n2,-n1)`, layer `1<->3`, layer `2->2`, and identity sublattice map.
- Audit:
  `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_symmetry_and_convention_audits/htg_symmetry_spectrum_audit_shell3/summary.json`
- Hamiltonian error is `~8.4e-17 eV`; analytic velocity matrices transform with the exact expected signs.
- This antiunitary symmetry is distinct from the unitary physical `C2x` broken by `m sigma_z`; it nevertheless suppresses the code-axis paper-labelled `sigma^{x;yy}`.

Axis-convention diagnostic:

- Written audit: `HTG_AXIS_CONVENTION_AUDIT.md`.
- Current conclusion: Mao's published C3 / `q_j` / `T_j` / Appendix-A coordinate definitions do not derive the empirical `~4.8 deg` reflected-basis rotation; Mao/Guerci/Mora 2023 instead supports the exact combined antiunitary `C2x C2z T` symmetry of the ABA local Hamiltonian.
- Mao Fig. 2's apparent peak ratio is roughly `|sigma^{x;yy}/sigma^{y;xx}| ~ 0.25`.
- For the converged one-coefficient tensor, a reflected-basis rotation gives
  `|sigma^{x;yy}/sigma^{y;xx}| = |tan(3 alpha)|`, so this ratio corresponds to `alpha ~ 4.8 deg`.
- Diagnostic plots:
  - `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_132463/central_flat_allc/central_shell5_m16_eta2_rot4p8_reflect_y.png`
  - `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_zeta0_c3_mesh16_132372/active24_allc/active24_zeta0_m16_eta2_rot4p8_reflect_y.png`

Conclusion: Fig. 2 can be brought close in shape, sign, and order of magnitude by a small coordinate-basis rotation/reflection, but this is **not** a reproduction.  The dedicated axis audit currently finds no published-coordinate derivation of the `~4.8 deg` offset.

## Mao Fig. 1(b): full / active-window hTTG shift current

Status: **partial diagnostic only, not final reproduction**.

What is close:

- The low-energy THz signal is dominated by the central flat pair, consistent with Mao's discussion and our band-pair decomposition.
- Active-window / central-window spectra with all components reproduce the correct order of magnitude (`10^4 microampere nm V^-2`) in the THz range.

What is not final:

- Raw code-axis `sigma^{x;yy}` is suppressed by the antiunitary symmetry, as in Fig. 2.
- The available active24 all-component run is at `n_shells=3`; shell convergence showed low-cutoff artifacts in tensor labels, so active-window production should be rerun at `n_shells>=4/5` after the coordinate convention is resolved.
- The paper-labelled `sigma^{x;yy}`/`sigma^{y;xx}` comparison currently depends on the same unresolved `~4.8 deg` axis-convention diagnostic.

Conclusion: Fig. 1(b) is not accepted as reproduced yet.

## Finite-grid and mass-pattern checks

- Mirror-symmetric no-C3 shell5 mesh16 grid (job `133958`) still suppresses raw `x;yy` to `~1.5e-6` while `y;xx~6990`; group1/group2 `~4.4e-11`.
- Deliberately non-mirror-symmetric quadrature (`frac_shift=0.23,0.37`, job `133985`) activates a forbidden-component artifact: raw `x;yy~696`, `y;xx~6829`, group1/group2 `~0.046`.  This is too small to explain Mao's apparent `~0.25` ratio.
- Unequal layer mass patterns can strongly activate raw `x;yy`, but Mao Eq. (13) explicitly uses `m sigma_z \otimes I_layer`, so this is only a diagnostic of which symmetry is being broken, not a reproduction path.

## Bottom line

After reference-code comparison, the local response formula is validated.  The current reproducibility level is:

1. **Formula benchmark against reference codes:** yes, passed.
2. **Mao Fig. 1(a):** qualitative candidate, needs quantitative sign-off.
3. **Mao Fig. 2:** close only as a coordinate-convention diagnostic; raw paper labels are not reproduced.
4. **Mao Fig. 1(b):** partial order-of-magnitude/shape diagnostics; not final.
5. **Mao Fig. 10:** not honestly reproduced; likely paper Appendix-A convention/implementation issue or incomplete printed formula usage.

Do not claim reproduction of Fig. 1(b), Fig. 2, or Fig. 10 until an explicit symmetry-breaking difference in Mao's numerical setup, a documented nonstandard plotting-axis convention, or a paper/component-label issue is established from first principles.
