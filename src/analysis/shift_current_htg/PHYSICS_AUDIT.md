# Physics audit notes for Mao hTTG shift-current reproduction

Date: 2026-05-28

## Status

Do **not** claim reproduction of Mao Fig. 1(b), Fig. 2, or Fig. 10 yet.

## Paper facts used as constraints

- Mao Eq. (13): hTTG continuum model with ABA phases `phi1=-phi2=2pi/3`, bottom interface `V_{-phi1,-phi2}`, uniform mass `m sigma_z \otimes I_layer`.
- Mao text below Eq. (13): the mass preserves `C3z` but breaks PHS, `C2x`, and `C2zT`.
- Mao Eq. (15): with only `C3z`, two independent tensor groups are allowed.
- Mao Fig. 2/text: below `0.1 eV`, the THz response is accounted for by the two central flat bands.
- Mao Appendix A / Fig. 10: the paper panel is dominated by the M-point van-Hove transition around `~6 eV` and does not show the direct K/K' gap near `2m=3 eV`; this is a paper target, not yet a validated physical expectation.

## Derived formula correction

For a general nonlinear Bloch Hamiltonian,

```text
D^a_nm = <u_n|partial_a H|u_m>
W^{ab}_nm = <u_n|partial_a partial_b H|u_m>
r^b_nm = -i D^b_nm/(E_n-E_m)
```

Differentiating `r^b_nm` in a locally parallel gauge gives Mao Eq. (4) plus

```text
-i W^{ab}_nm/(E_n-E_m).
```

This term vanishes for Mao's hTTG continuum Dirac Hamiltonian but not for the SLG tight-binding benchmark.  The patched formula matches finite-difference covariant derivatives in the SLG toy to `~1e-13`.

## Current SLG toy problem

After including `W` and C3-orbit averaging the grid:

- C3 tensor identities pass at `~1e-13 microampere nm V^-2`.
- The plotted `sigma^{x;xy}` and `sigma^{y;yy}` are opposite as required by C3v.
- An independent finite-difference covariant-derivative spectrum check matches the analytic `W`-corrected implementation to `~1e-10` in the final spectra on a small mesh.
- But the dominant peak is near the K/K' direct gap (`~3.2 eV` for `m=1.5 eV`), not the M-point transition (`~6 eV`) emphasized by Mao Fig. 10.

Transition-energy filtering confirms that the low-energy contribution is physically present in the current implementation rather than just a plotting artifact.  This points to an unresolved tight-binding optical/Bloch-basis convention problem, a difference between Mao's Appendix-A implementation and the current gauge-free tight-binding implementation, or a possible issue/omission in the paper's Appendix-A calculation.

A supplied external zip (`reference/mean_field_2411_toy_fig10_corrected.zip`) was inspected but not adopted: its proposed `slg_toy_fig10.py` uses Pauli blocking (`mu=-3 eV`), an M-saddle Gaussian envelope, amplitude/unit scale factors, a digitized component ratio, DOS tapering, and visual gates.  This is useful as a note about what the paper panel visually emphasizes, but it is not a corrected full shift-current calculation and cannot validate the response code.

Cross-reference and official reference-code audit:

- Hipolito, Pedersen, and Pereira, PRB 94, 045434 (2016), studies the same gapped honeycomb tight-binding model class and explicitly predicts a direct K-point band-edge onset at `hbar omega=Delta`, with low-energy behavior `Re sigma_dc^(2) ~ -sigma_2/(4 Delta) theta(hbar omega-Delta)`.  For Mao's toy `m=1.5 eV`, `Delta=2m=3 eV`, so this supports the local K/K' direct-gap peak and makes Mao Fig. 10 unreliable as a full neutral tight-binding benchmark.
- Added local cross-reference reproduction/evidence plot `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_evidence/hipolito2016_crossref_evidence.png`.  It reproduces Hipolito's gapped-graphene low-energy K/K' onset and component-overlap check with a K-corner patch grid; the component-overlap error is `~1.6e-12` in normalized units.
- Added a direct Hipolito Fig. 4 Re/Im reproduction, now in the interval benchmark suite: `results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png`.  It uses Hipolito Eq. (25b) with the Fig. 4 parameters and fixes the remaining global convention by Eq. (31).  This independently confirms the same K/K' onset and the published Re/Im line shape.
- The small wiggles in the first fixed-k-grid Fig. 4 reproduction were numerical, not graphical: `Gamma=1 meV` is too narrow for a coarse fixed transition-energy sampling, and finite-difference covariant derivatives amplify the problem.  Added the resonance-resolved version `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_energy_quad/hipolito2016_fig4_energy_quad.png`, which uses analytic generalized derivatives and transition-energy quadrature to remove the wiggles.
- The larger oscillations in the first Hipolito Fig. 5(a) gap-series attempt had the same mathematical origin but one level subtler: even after changing variables to `Ecv`, point-sampling the narrow factors `1/(omega-Ecv+iGamma)` and `1/(omega-Ecv+iGamma)^2` at transition-energy nodes rings when the node spacing is comparable to or larger than `Gamma`.  The corrected low-energy Fig. 5(a) benchmark analytically integrates these denominator factors over each transition-energy interval and leaves only the smooth numerator to midpoint quadrature: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/hipolito2016_fig5a_gap_series_exact_t72/hipolito2016_fig5a_gap_series.png`.
- The full Hipolito Fig. 5(a) panel needs the same idea over the full BZ, including M-point saddle features.  The accepted strict run uses a linear-tetrahedron transition-energy histogram over the full primitive reciprocal cell and then analytically integrates the resonant denominators over 1 meV energy bins: `results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev/hipolito2016_fig5a_full_bz_tetra.png`.  This is a numerical integration fix, not plot smoothing.

- Cloned inspected sources under `reference/upstream/`:
  - `wannier90/src/postw90/berry.F90::berry_get_sc_klist`
  - `wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula`
- Added `run_slg_toy_reference_formula_audit.py`, a direct transcription of the official internal-term formula for the orthogonal nearest-neighbor SLG model in the TB phase convention (`AA_R=0` after subtracting Wannier centers).
- The audit agrees with the local `response.py` W-corrected spectra to numerical roundoff (`~1e-14 microampere nm V^-2` on the `mesh=40` audit) and still gives the dominant `K/K'` direct-gap peak near `3.3 eV`.
- Therefore the toy-model discrepancy is not fixed by adopting the standard Wannier90/WannierBerri shift-current formula; the remaining issue is specifically why Mao Fig. 10 displays only/dominantly the M-saddle feature near `6 eV`.

Mao Fig. 10 reproduction audit:

- Added `run_slg_toy_fig10_reproduction_audit.py` and ran it at `mesh=150`, `eta=50 meV`:
  `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_fig10_reproduction_audit_m150/slg_toy_fig10_audit.png`.
- Formal W-corrected/C3-symmetrized result: dominant `K/K'` direct-gap peak near `3.18 eV` with `|sigma^{x;xy}|=|sigma^{y;yy}|≈2.34 microampere nm V^-2`.
- Paper-printed Eq. (4)-only/no-`W` primitive-grid result: blue `sigma^{x;xy}` has an apparent M-point peak `≈-0.101` at `6.24 eV`, close to Mao Fig. 10(a), but the paper-labelled red `sigma^{y;yy}` remains zero.  The nonzero partner is instead `sigma^{y;xx}≈+0.202`.
- The Eq. (4)-only primitive-grid M peak is not converged: `|sigma^{x;xy}|` decreases from `0.313` at mesh 80 to `0.101` at mesh 150, `0.0537` at mesh 200, `0.0299` at mesh 250, and `0.0163` at mesh 300.  With C3-orbit averaging, the same Eq. (4)-only result is zero to numerical roundoff.
- Conclusion: Mao Fig. 10 cannot be honestly reproduced as a converged full shift-current result from Eq. (A1).  The blue M-point feature can be recreated only by the incomplete printed Eq. (4) without the tight-binding second-derivative/external completion and with a finite non-C3 primitive grid; the red paper component is still inconsistent.

## Current hTTG issue

Plane-wave shell convergence of the central flat-pair response shows:

- `n_shells=2` gives a spurious nonzero raw `sigma^{x;yy}`.
- `n_shells>=4` collapses raw code-axis `sigma^{x;yy}` to numerical zero.
- The group `sigma^{y;xx}=sigma^{x;xy}=sigma^{x;yx}=-sigma^{y;yy}` converges to `~6.93e3 microampere nm V^-2` for `eta=2 meV`, `~9.79e3` for `eta=1 meV`.

Thus the converged current implementation has an apparent mirror constraint (C3v-like one-scalar tensor) in its internal axes.  The dedicated derivations/audits are `HTG_SYMMETRY_AUDIT.md` and `HTG_AXIS_CONVENTION_AUDIT.md`.  The dedicated tensor audit
`results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_132463/central_flat_allc/tensor_symmetry_audit_eta2/tensor_symmetry_audit.png`
extracts the C3 Eq. (15) coefficients directly: group1 has max `~1.5e-7`, group2 has max `~6930`, so group1/group2 is `~2.2e-11` at eta `2 meV`.  Mao says the mass breaks `C2x` and only quotes `C3z` constraints, so either:

1. the paper's plotted Cartesian axes are rotated away from this mirror axis by an undocumented convention;
2. Mao's actual numerical Hamiltonian included a symmetry-breaking term not written in Eq. (13); or
3. the paper and code use different coordinate/gauge conventions for labeling tensor components.

A `y` reflection plus rotation by `alpha` maps a pure raw group-2 tensor `B` to

```text
sigma_new^{x;yy} = -B sin(3 alpha)
sigma_new^{y;xx} = -B cos(3 alpha)
```

which explains why small empirical rotations make the blue curve appear.  This is not yet a derivation of the paper convention.

## Next physics checks

1. The exact mirror-like operation of the implemented ABA Hamiltonian is now derived in `HTG_SYMMETRY_AUDIT.md`.  The implemented finite-cutoff Hamiltonian obeys
   `H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger`, with reciprocal-index map `(n1,n2)->(-n2,-n1)`, layer map `1<->3, 2->2`, and identity sublattice map.  The audit gives Hamiltonian error `~8.4e-17 eV`; velocities transform as `dH/dkx -> +U(dH/dkx)^*U^dagger`, `dH/dky -> -U(dH/dky)^*U^dagger` exactly in the analytic derivative matrices.
2. Compare that antiunitary layer-swap/conjugation symmetry with Mao's statement that `m sigma_z` breaks `C2x`.  This operation is not the unitary physical `C2x` that swaps sublattices; it is an antiunitary layer-swap/conjugation symmetry of the continuum implementation.  Since it leaves sublattice unchanged, the equal mass term `m sigma_z \otimes I_layer` survives.  For the dc shift-current tensor the antiunitary operation gives
   `sigma^{a;bc} = - R_{aa'} R_{bb'} R_{cc'} sigma^{a';b'c'}` with `R=diag(+1,-1)`.  Hence components with an even number of `y` indices (`x;xx`, `x;yy`, `y;xy`, `y;yx`) are suppressed, while the group2 combination (`y;xx = x;xy = x;yx = -y;yy`) is allowed.  This explains the observed one-coefficient C3v-like tensor.
3. Treat Mao Fig. 10 as unresolved/non-validation unless a core-equation reason is found for suppressing the formal K/K' direct-gap contribution.
4. The no-C3 shell5 mesh16 central-flat diagnostic with mirror-symmetric midpoint grid (job `133958`) shows raw `x;yy ~1.5e-6` while `y;xx ~6990` at eta 2 meV; tensor group1/group2 is `~4.4e-11`.  Finite primitive-grid artifacts do not explain the missing paper-labelled `x;yy` when the quadrature respects `k -> conj(k)`.
5. A deliberately non-mirror-symmetric quadrature (`frac_shift=0.23,0.37`, job `133985`) gives raw `x;yy~696`, `y;xx~6829`, and group1/group2 `~0.046` at shell5 mesh16.  Thus quadrature asymmetry can activate forbidden components, but not enough to explain Mao Fig. 2's apparent `~0.25` peak ratio.
6. Axis-convention diagnostic: Mao Fig. 2's apparent peak ratio `|sigma^{x;yy}/sigma^{y;xx}| ~ 0.25` would correspond to a reflected-basis rotation `alpha ~ arctan(0.25)/3 ~ 4.8 deg` for the converged one-coefficient tensor.  `HTG_AXIS_CONVENTION_AUDIT.md` checks Mao's printed C3, `q_j`/`T_j`, Appendix-A axes, and Mao/Guerci/Mora 2023 conventions; it finds no published-coordinate derivation of this angle.  Therefore the `~4.8 deg` offset must not be treated as a reproduction.
7. Only after these checks decide whether tensor-axis rotation is a documented paper-coordinate transformation, an unreported model difference, or a paper/component-label issue.
