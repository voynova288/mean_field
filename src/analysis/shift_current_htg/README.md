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

## Current tracked surface

This workspace no longer tracks a separate launcher or plotting script for every diagnostic.  Per-run `run_*`, `plot_*`, and `analyze_*` files were retired in favor of the repository script policy: reusable code belongs in modules, while paper-panel launch recipes and historical commands belong in reports or result metadata.

Tracked reusable modules:

- `response.py` - gauge-free Berry connection, generalized derivative, optional second-derivative correction for nonlinear tight-binding models, Lorentzian BZ-integral assembly, and unit conversion.
- `slg_toy.py` - Appendix-A nearest-neighbor gapped graphene toy model on the full hexagonal BZ; C3 grid symmetrization and analytic `d2H/dkdk` are included, but the comparison to Mao Fig. 10 is still unresolved.
- `htg_adapter.py` - Mao-parameter wrapper around the existing `mean_field.systems.htg` Hamiltonian, including sublattice mass, stacking phase-to-displacement conversion, and `dH/dk` validation.
- `hipolito2016.py` - reusable Hipolito Eq. (25b) transition-energy quadrature module.
- `constants.py` - shift-current unit and broadening conversion constants.

Tracked audit/status notes remain as historical evidence and should be read before changing signs, axes, units, tensor labels, or band windows.  If a retired one-off workflow is needed again, recover the reusable logic from git history into `response.py`, `hipolito2016.py`, `htg_adapter.py`, `src/analysis/response_derivative_gauge.py`, or a dispatcher-backed devtool instead of restoring the old standalone script.

## Validation hooks

Use package-level tests and tiny module-level imports for local validation.  Heavy response integrations and production-like spectra are Slurm jobs and should go through `scripts/submit_mean_field.sbatch` or a dispatcher-backed tool, not a new tracked per-run script.

Expected first physics gate remains analytic-vs-finite-difference `dH/dk` validation in the hTTG adapter.  Historical commands and result paths are preserved in the audit notes and reports, but they are not the current tracked command surface.

## Current limitations / next steps

1. Production reproduction still needs the work-document convergence table: `Nk=120,180,240` and `eta=0.5,1,2 meV`, preferably through Slurm and a dispatcher-backed reusable command rather than a new tracked per-run script.
2. The gapped-SLG toy now passes the internal covariant-derivative and C3 tensor checks, but it still does not match Mao Fig. 10; Hipolito 2016 supports the local K/K' band-edge result, so Mao Fig. 10 is not a reliable validation gate.
3. hTTG central-flat shell convergence shows `n_shells>=4` collapses the raw code-axis `sigma^{x;yy}` to zero while `sigma^{y;xx}` remains `~7e3` at eta=2 meV. `HTG_SYMMETRY_AUDIT.md` explains this as an exact antiunitary symmetry of the literal Eq. (13) implementation. `HTG_AXIS_CONVENTION_AUDIT.md` currently finds no published-coordinate derivation of the empirical `~4.8 deg` rotation/reflection that makes a paper-like blue curve.
4. Future band-window convergence should be implemented as a reusable option in a dispatcher-backed tool or module, then documented with explicit convergence evidence before attempting Fig. 1.
