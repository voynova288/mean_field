# Appendix-A / Fig. 10 gapped-SLG audit

Last updated: 2026-05-30

Goal: decide whether Mao et al. Fig. 10 can be used as a benchmark for the local shift-current implementation.

## External paper checked

Reference downloaded locally:

- `reference/Hipolito_Pedersen_Pereira_2016_PRB94_045434.pdf`
- F. Hipolito, T. G. Pedersen, and V. M. Pereira, *Nonlinear photocurrents in two-dimensional systems based on graphene and boron nitride*, Phys. Rev. B **94**, 045434 (2016).

This paper is directly relevant because it studies the same class of nearest-neighbor two-band honeycomb models with sublattice mass/gap and computes the intrinsic photogalvanic/shift photocurrent tensor.

## Relevant formulas / statements in Hipolito 2016

### Tensor symmetry

Hipolito Sec. III states that C3 symmetry leaves two independent in-plane rank-3 tensor components before additional mirrors; in the single-layer hBN / gapped-graphene cases they then compute the symmetry-related nonzero components and show that the expected component relations overlap in Fig. 4.

### Tight-binding model

Hipolito Eq. (28) is the same nearest-neighbor two-band gapped honeycomb model class:

```text
h(k) = [[-Delta/2, phi(k)], [phi(k)^*, +Delta/2]],
phi(k) = exp(i k_y a) + 2 exp(-i k_y a/2) cos(sqrt(3) k_x a/2).
```

Mao Appendix A uses the equivalent model with `mass m`, so `Delta = 2m`.  For Mao's toy parameters, `m=1.5 eV`, hence `Delta=3 eV`.

### Band-edge/K contribution is expected

Hipolito's text below Fig. 4 is crucial:

- The real photoconductivity should share the joint-DOS features.
- It has an onset at exactly `hbar omega = Delta` at `T=0`.
- Van-Hove singularities occur at frequencies connecting locally flat band portions.
- For small gaps, the response near the gap is much stronger and governed by virtual transitions near the `K` point.

Hipolito Eq. (31) gives the low-energy expansion near the K point:

```text
Re sigma_dc^(2)(omega) ~= sigma_2 [ -1/(4 Delta) + ... ] theta(hbar omega - Delta).
```

This explicitly predicts a nonzero band-edge contribution at the direct K/K' gap.  For Mao's `Delta=3 eV`, the threshold coefficient is not symmetry-forbidden.

### Doping can suppress the band-edge contribution

Hipolito Fig. 5(b) and the text state that at `T≈0`, a chemical potential in the conduction band blocks response for `hbar omega < 2 |mu|` by Pauli exclusion.  Mao Fig. 10, however, states the neutral toy model context and does not specify such Pauli blocking.  Therefore Pauli blocking cannot be invoked unless the benchmark is explicitly changed away from the printed Appendix-A setup.

## Local calculations

### Cross-reference figure reproduction evidence

Added two local Hipolito evidence plots:

1. Formal shift-current cross-reference plot:
   - Script: `run_slg_toy_hipolito_crossref.py`
   - Output: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_evidence/hipolito2016_crossref_evidence.png`
   - Data/summary: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_evidence/summary.json`
   - Purpose: use the W-corrected/reference-code-consistent shift-current formula and a K-corner patch grid to resolve the K/K' direct-gap onset.  The symmetry-related real components overlap to `~1.6e-12` in units of `sigma/sigma2`.

2. Hipolito Fig. 4 density-matrix reproduction:
   - Script: `run_hipolito2016_benchmark_suite.py`
   - Output: `results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png`
   - Data/summary: `results/shift_current_htg/hipolito2016_benchmark_suite_interval/summary.json`
   - Purpose: reproduce Hipolito Fig. 4's Re/Im photoconductivity shape for `gamma0=3 eV`, `Delta=0.2 eV`, `Gamma=1 meV`, `mu=0`, `T=1 K`, using the paper's two-band Eq. (25b) interband-intraband term.  The global convention is fixed by Hipolito Eq. (31), not by visual fitting.  The older fixed-grid density-matrix reproduction was archived under `results/shift_current_htg/_archived_wrong_or_superseded_20260530/hipolito_point_node_or_wiggle/`.

3. Wiggle-free resonance-resolved Hipolito Fig. 4 reproduction:
   - Script: `run_slg_toy_hipolito_fig4_energy_quad.py`
   - Output: `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_energy_quad/hipolito2016_fig4_energy_quad.png`
   - Purpose: remove the small fixed-k-grid wiggles by integrating the radial direction in transition-energy variables.  This is a numerical quadrature fix, not a plotting smoother.

Together these plots reproduce the cross-reference paper's core result: the gapped-graphene response turns on at the K/K' direct gap.  The cross-reference plot places Mao's Appendix-A audit beside it: the formal Mao-toy result peaks at `~3.18 eV` near `2m=3 eV`, whereas the paper-printed Eq.(4)-only diagnostic peak sits near the M transition at `~6.24 eV`.

### Formal / reference-code-consistent result

The local W-corrected gauge-free tight-binding calculation agrees with a direct transcription of Wannier90/postw90 and WannierBerri shift-current formulas.  Representative output:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_reference_formula_audit_m40/summary.json`
- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_formal_wcorrected_m160/slg_toy_formal_xxy_yyy.png`

It gives a dominant K/K' direct-gap response near `~3.2 eV` for `m=1.5 eV`, consistent with Hipolito's statement that a band-edge contribution exists at `hbar omega = Delta = 3 eV`.

### Paper-printed Eq. (4)-only/no-W result

If one deliberately omits the nonlinear tight-binding second-derivative/external completion and evaluates Mao's printed Eq. (4) as if the Hamiltonian were linear in k, the K/K' band-edge contribution disappears and an apparent M-point feature near `~6.2 eV` can be obtained in `sigma^{x;xy}` on a finite primitive grid.

However:

- This is not the formal shift-current calculation for a tight-binding Hamiltonian.
- The apparent M peak is mesh dependent and tends to zero with increasing primitive-grid density / C3-orbit averaging.
- The paper-labelled red component `sigma^{y;yy}` remains zero in this calculation; the nonzero partner is `sigma^{y;xx}`.

Representative audit:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_fig10_reproduction_audit_m150/slg_toy_fig10_audit.png`
- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/slg_validation_audits/slg_toy_fig10_reproduction_audit_m150/summary.json`

## Interpretation for Mao Fig. 10

Mao Fig. 10 shows a dominant M-point/DOS feature near `~6 eV` and does not show the expected neutral K/K' band-edge response near `~3 eV`.  Hipolito 2016 strongly suggests that a neutral gapped-graphene nearest-neighbor model should have a direct-gap contribution at `hbar omega=Delta`; it is not symmetry-forbidden.

Therefore Mao Fig. 10 is likely produced by one of the following:

1. the printed Eq. (4) was used outside its continuum-linear-Hamiltonian regime without the tight-binding second-derivative / position-operator completion;
2. some transition filtering or Pauli blocking was applied but not stated in the Appendix-A panel description;
3. the tensor labels/components in the panel are not the literal components of the full neutral tight-binding response;
4. the panel is a qualitative M-point/DOS illustration rather than a converged full shift-current benchmark.

## Conclusion

Mao Appendix-A Fig. 10 should **not** be used as an acceptance benchmark for the local response code.

The local result is consistent with standard reference-code formulas and with Hipolito 2016's statement that the band-edge/K contribution exists.  The visually paper-like M-only plot can be produced only by an incomplete/nonconverged diagnostic path, which is not acceptable as reproduction evidence.
