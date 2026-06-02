# hTTG shift-current analysis workspace

Purpose: start the non-interacting shift-current benchmark requested in `plan/hTTG中的Shift_Current工作文档.md`, using Mao et al. arXiv:2411.07844v2 / PRB 111, 195408 (2025) as the reference.

This folder is deliberately under `src/analysis/` rather than the stable `mean_field` package API.  It is a reproducibility workspace for formula validation and small smoke runs before any production Slurm calculation.  For the current figure-by-figure status after reference-code comparison, see `REPRODUCTION_STATUS.md`; for the Appendix-A toy-model audit against Hipolito 2016, see `SLG_FIG10_AUDIT.md`; for the reusable Hipolito 2016 benchmark suite, see `BENCHMARKS_HIPOLITO2016.md`; for the hTTG tensor-symmetry derivation, see `HTG_SYMMETRY_AUDIT.md`; for the Mao Cartesian-axis convention audit, see `HTG_AXIS_CONVENTION_AUDIT.md`.

## Paper/work-document contract captured here

- Use Mao's gauge-free formulas: `D_nm^a=<u_n|partial_{k_a}H|u_m>` in eV nm, energy differences in eV, and no intermediate `hbar` in Berry connections or generalized derivatives.  For nonlinear tight-binding benchmarks, include the additional generalized-derivative term `-i <u_n|partial_a partial_b H|u_m>/(E_n-E_m)`; it vanishes for the linear hTTG continuum Hamiltonian.
- Photon axis is photon energy `E_gamma` in eV; Lorentzian broadening is normalized as `1/eV`.
- Final conversion is applied once with `Re[-i*pi*e^2/hbar*1e6*I]`, where `I` is the BZ integral in `nm/eV`, giving `microampere nm V^-2`.
- hTTG benchmark defaults: `hbar v_F |K|=9.905 eV`, `w1=110 meV`, `r=0.8`, `m=30 meV`, single K valley, `mu=0`, `T=0`.
- Mao Eq. (13) writes the Dirac block as `hbar v_F k dot sigma` for all three layers, so the Mao adapter sets `zeta_rad=0` by default.  This differs from the Kwan-style HTG package default that rotates outer-layer Dirac cones.  Results produced before this correction (notably `active24_nshell3_132077`) should not be used for Mao comparisons.
- Stacking phases: ABA uses `(phi1, phi2)=(2pi/3,-2pi/3)` and the bottom interface uses the opposite phases; AAA uses `(0,0)`.
- Analytic `partial_k H` must pass finite-difference validation before any shift-current run is trusted.

## Files

- `response.py` - gauge-free Berry connection, generalized derivative, optional second-derivative correction for nonlinear tight-binding models, Lorentzian BZ-integral assembly, and unit conversion.
- `slg_toy.py` - Appendix-A nearest-neighbor gapped graphene toy model on the full hexagonal BZ; C3 grid symmetrization and analytic `d2H/dkdk` are included, but the comparison to Mao Fig. 10 is still unresolved.
- `htg_adapter.py` - Mao-parameter wrapper around the existing `mean_field.systems.htg` Hamiltonian, including sublattice mass, stacking phase-to-displacement conversion, and `dH/dk` validation.
- `run_slg_toy.py` - quick Appendix-A toy spectrum and C3 tensor-relation check.
- `run_slg_toy_reference_formula_audit.py` - compares the local SLG toy response against a direct transcription of the official Wannier90/postw90 and WannierBerri shift-current formulas.  This is a formula audit only, with no visual fitting or post-hoc scaling.
- `run_slg_toy_fig10_reproduction_audit.py` - honest Mao Fig. 10 audit.  It compares the formal W-corrected result with the paper-printed Eq. (4)-only/no-W finite-grid calculation and records whether any apparent M-point peak is converged.
- `BENCHMARKS_HIPOLITO2016.md` - reusable benchmark description and acceptance metrics for Hipolito 2016 gapped-graphene Fig. 4 and Fig. 5(b).
- `hipolito2016.py` - reusable Hipolito Eq. (25b) transition-energy quadrature module.
- `run_hipolito2016_benchmark_suite.py` - generates the benchmark suite plot and regression summary for Hipolito 2016 Fig. 4 and Fig. 5(b).
- `run_hipolito2016_fig5a_gap_series.py` - reproduces the low-energy K/K' threshold/gap-dependence part of Hipolito Fig. 5(a), with one global Eq. (31) calibration, no per-curve fitting, and analytic transition-energy interval integration of the narrow resonant denominators.
- `run_hipolito2016_fig5a_full_bz_tetra.py` - full-BZ Hipolito Fig. 5(a) reproduction at `Gamma=1 meV`, using a linear-tetrahedron transition-energy histogram plus analytic resonant-denominator interval integration.
- `run_hipolito2016_fig5a_full_bz_diagnostic.py` - older full-BZ diagnostic including M-point features with broader broadening; kept only as a diagnostic comparison now that the tetrahedron version is available.
- `run_slg_toy_hipolito_crossref.py` - cross-reference evidence plot for Hipolito/Pedersen/Pereira PRB 94, 045434 (2016).  It resolves the K/K' direct-gap onset of the gapped-graphene toy with a K-corner patch grid and places it next to the Mao Appendix-A audit result.
- `run_slg_toy_hipolito_fig4.py` - Hipolito Fig. 4 Re/Im photoconductivity reproduction using the paper's two-band Eq. (25b) interband-intraband term for `gamma0=3 eV`, `Delta=0.2 eV`, `Gamma=1 meV`, `mu=0`, `T=1 K`.  The default uses analytic generalized derivatives; the finite-difference derivative is retained only as a diagnostic for numerical wiggles.
- `run_slg_toy_hipolito_fig4_energy_quad.py` - resonance-resolved version of the Hipolito Fig. 4 reproduction.  It integrates the radial direction in transition-energy variables and removes the narrow-`Gamma` fixed-k-grid fluctuations without plot smoothing.
- `analyze_bandpair_tensor_symmetry.py` - load all eight hTTG tensor components, extract the two C3 Eq. (15) coefficient groups, and scan how reported `x;yy/y;xx` components change under coordinate-basis rotations.  This is an axis/symmetry diagnostic, not a fit.
- `HTG_SYMMETRY_AUDIT.md` - derivation and numerical evidence for the exact antiunitary layer-swap/conjugation symmetry of the literal Eq. (13) implementation and its tensor consequence: code-axis `sigma^{x;yy}` is forbidden while `sigma^{y;xx}` is allowed.
- `HTG_AXIS_CONVENTION_AUDIT.md` - check of Mao's printed C3, `q_j`/`T_j`, Appendix-A axes, and Mao/Guerci/Mora 2023 symmetry conventions.  Current conclusion: the empirical `~4.8 deg` rotation is not derived from published coordinate definitions.
- `run_htg_symmetry_spectrum_audit.py` - tests candidate hTTG coordinate transforms and verifies the exact antiunitary layer-swap/conjugation relation `H(k_x,-k_y)=U H(k_x,k_y)^* U^dagger` for the implemented ABA Hamiltonian, including the corresponding velocity-operator signs.
- `run_htg_layer_mass_pattern_audit.py` - diagnostic-only test of layer-dependent sublattice mass profiles.  Unequal layer masses break the layer-swap/conjugation symmetry and can activate the second C3 tensor coefficient, but Mao Eq. (13) uses equal layer mass, so this is not a reproduction claim.
- `run_htg_velocity_check.py` - fast analytic-vs-finite-difference `partial_k H` gate.
- `run_htg_shift_current_smoke.py` - tiny full-chain hTTG smoke run; not a convergence calculation.
- `run_htg_bandpair_spectra.py` - streaming selected-band-pair transition workflow for Fig. 2-style decomposition; stores compact transition events rather than full eigenvector grids. Use `--c3-symmetrize-grid` for finite-mesh tensor diagnostics; the old unsymmetrized `8x8` active-window plot is not reliable.
- `plot_bandpair_spectra.py` - plot saved band-pair spectra. For component comparison use all eight tensor components and the `--reflect-y` / `--rotation-deg` tensor-basis transformation options; plotting only internal-axis `x;yy` and `y;xx` can be misleading.
- `run_htg_bands_dos.py` - Fig. 1(a)-style band structure plus DOS diagnostic.  The high-symmetry path uses the adjacent extended-zone `K' = kappa_prime_m + b_m1`; do not replace it by the central-zone `kappa_prime_m`, which gives a visibly wrong path.

## Quick commands

From the repository root:

```bash
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_velocity_check --n-shells 1
PYTHONPATH=src python -m analysis.shift_current_htg.run_slg_toy --mesh-size 24 --no-save
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_shift_current_smoke --n-shells 1 --mesh-size 3 --no-save
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra --n-shells 1 --mesh-size 2 --eta-mev 1,2 --no-save
PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra --n-shells 1 --mesh-size 2 --pair-window central_window:-2,1 --eta-mev 2 --no-save
# Production-like smoke figures were launched through Slurm wrappers in scripts/run_shift_current_htg_*.sbatch.
```

Expected first gate: `dhdk_validation.passes_1e_minus_7 = true` for the hTTG velocity check.

## Current limitations / next steps

1. `run_htg_shift_current_smoke.py` uses very small meshes by default and is only a NaN/sign/unit smoke test.
2. Production reproduction still needs the work-document convergence table: `Nk=120,180,240` and `eta=0.5,1,2 meV`, preferably through Slurm.
3. The gapped-SLG toy now passes the internal covariant-derivative and C3 tensor checks, but it still does not match Mao Fig. 10; Hipolito 2016 supports the local K/K' band-edge result, so Mao Fig. 10 is not a reliable validation gate.
4. hTTG central-flat shell convergence shows `n_shells>=4` collapses the raw code-axis `sigma^{x;yy}` to zero while `sigma^{y;xx}` remains `~7e3` at eta=2 meV.  `HTG_SYMMETRY_AUDIT.md` explains this as an exact antiunitary symmetry of the literal Eq. (13) implementation.  `HTG_AXIS_CONVENTION_AUDIT.md` currently finds no published-coordinate derivation of the empirical `~4.8 deg` rotation/reflection that makes a paper-like blue curve.
5. Use `--pair-window name:min,max` in `run_htg_bandpair_spectra.py` for increasingly wide occupied-unoccupied windows, then document explicit band-window convergence before attempting Fig. 1.
