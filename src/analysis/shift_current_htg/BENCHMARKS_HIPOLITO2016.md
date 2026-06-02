# Hipolito 2016 gapped-graphene benchmark suite

Last updated: 2026-05-30

Reference:

- F. Hipolito, T. G. Pedersen, and V. M. Pereira, *Nonlinear photocurrents in two-dimensional systems based on graphene and boron nitride*, Phys. Rev. B **94**, 045434 (2016).

Purpose: make the gapped-graphene part of Hipolito 2016 a reusable benchmark for the local shift-current / photoconductivity code.  This benchmark is independent evidence that a neutral gapped honeycomb model has a K/K' direct-gap response at `hbar omega = Delta`, which is why Mao Appendix-A Fig. 10 is not a reliable full-model validation gate.

## Implemented benchmark figures

### Benchmark A: Hipolito Fig. 4

Parameters:

```text
gamma0 = 3 eV
Delta = 0.2 eV
Gamma = 1 meV
mu = 0
T = 1 K
```

Local outputs:

```text
results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_energy_quad/hipolito2016_fig4_energy_quad.png
```

What is reproduced:

- Re `sigma_dc^(2)/sigma2` turns on sharply and negatively at `hbar omega = Delta`.
- Im `sigma_dc^(2)/sigma2` has the corresponding resonant positive peak.
- The low-energy threshold follows Hipolito Eq. (31):
  ```text
  Re sigma_dc^(2) / sigma2 ~= -1/(4 Delta/gamma0)
  ```
  For `Delta=0.2 eV`, `gamma0=3 eV`, this gives `-3.75`.

### Benchmark B: Hipolito Fig. 5(a), low-energy gap series

Parameters:

```text
Delta = 0.1, 0.2, 0.5, 1.0, 2.0 eV
gamma0 = 3 eV
Gamma = 1 meV
mu = 0
T = 1 K
```

Local output:

```text
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/hipolito2016_fig5a_gap_series_exact_t72/hipolito2016_fig5a_gap_series.png
```

What is reproduced:

- The response threshold tracks `hbar omega = Delta`.
- The near-threshold magnitude decreases approximately as `1/Delta`, matching Hipolito Eq. (31).
- This is deliberately a low-energy K/K' threshold benchmark; it does not attempt the full published `0--9 eV` high-energy M/UV van-Hove structure.
- The resonant denominators are integrated analytically over transition-energy intervals.  This removes the previous node-crossing oscillations without smoothing or filtering the plotted curve.

The full-BZ Fig. 5(a) reproduction is now available:

```text
results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev/hipolito2016_fig5a_full_bz_tetra.png
```

This uses the paper's `Gamma=1 meV`, the full primitive reciprocal cell, a linear-tetrahedron transition-energy histogram, and analytic interval integration of the resonant denominators.  It includes both the low-energy K/K' thresholds and the high-energy M-point peaks, without fixed-grid shell-sampling wiggles.

### Benchmark C: Hipolito Fig. 5(b)

Parameters are as in Fig. 4, with finite chemical potential:

```text
mu = 0, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275 eV
```

Local output:

```text
results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png
```

What is reproduced:

- Pauli blocking suppresses response below approximately `hbar omega = 2 |mu|`.
- The response onset shifts to higher photon energy as `mu` increases, matching Hipolito Fig. 5(b)'s core physics.

## Scripts

Reusable module:

```text
src/analysis/shift_current_htg/hipolito2016.py
```

Benchmark suite:

```bash
PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_benchmark_suite \
  --theta-count 72 --transition-energy-intervals 900 --workers 8 \
  --output-dir results/shift_current_htg/hipolito2016_benchmark_suite_interval

PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_fig5a_gap_series \
  --theta-count 72 --transition-energy-intervals 360 --patch-radius-nm-inv 3.5 \
  --output-dir results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/hipolito2016_fig5a_gap_series_exact_t72

PYTHONPATH=src python -m analysis.shift_current_htg.run_hipolito2016_fig5a_full_bz_tetra \
  --mesh-size 720 --energy-bin-width-mev 1.0 --n-photon 1901 --gamma-mev 1.0 --workers 10 \
  --output-dir results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev
```

Standalone figure scripts:

```bash
PYTHONPATH=src python -m analysis.shift_current_htg.run_slg_toy_hipolito_fig4_energy_quad \
  --output-dir results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_energy_quad

PYTHONPATH=src python -m analysis.shift_current_htg.run_slg_toy_hipolito_crossref \
  --output-dir results/shift_current_htg/_archived_tests_and_diagnostics_20260530/hipolito_support_and_convergence/crossref_hipolito2016_fig4_evidence
```

## Numerical method

The benchmark uses Hipolito Eq. (25b), the two-band interband-intraband contribution.  For neutral time-reversal-symmetric two-band monolayer graphene, Hipolito states that the other contributions vanish for the plotted components.

Two numerical details are essential:

1. **Analytic generalized derivative.**  Finite-difference covariant derivatives produce visible noise when `Gamma=1 meV`, because the derivative of the resonant denominator contains `1/(E-Ecv+iGamma)^2`.
2. **Transition-energy denominator integration.**  A fixed k-grid produces shell-sampling wiggles for narrow broadening.  Merely sampling transition-energy nodes can still wiggle when the node spacing is larger than `Gamma`.  The robust low-energy Fig. 5(a) benchmark changes radial variables from `r` to direct transition energy `Ecv(r,theta)` and analytically integrates `1/(omega-Ecv+iGamma)` and `1/(omega-Ecv+iGamma)^2` over each energy interval.  The full-BZ Fig. 5(a) benchmark generalizes this with a linear-tetrahedron transition-energy histogram over the whole primitive reciprocal cell, then applies the same analytic interval denominator integration.

The global convention is fixed by Hipolito Eq. (31), not by visual fitting.  The Eq. (31) threshold value is used once for the `mu=0` Fig. 4 benchmark and the same scale is reused for Fig. 5(b).

## Current acceptance metrics

From

```text
results/shift_current_htg/hipolito2016_benchmark_suite_interval/summary.json
```

representative metrics:

```text
Eq.(31) target: -3.75
Re sigma at Delta+0.03 eV: -3.75
absolute error to Eq.(31) target: ~4.4e-16
generated intervals: 129600
Pauli thresholds: 2mu = 0.25, 0.30, 0.35, ..., 0.55 eV
```

These metrics should be used as regression checks before trusting future changes to the response implementation.

## Relation to Mao Appendix A

Mao Appendix A uses the same gapped-honeycomb model class but with `m=1.5 eV`, so `Delta=2m=3 eV`.  Hipolito Fig. 4/Fig. 5(b) and Eq. (31) show that a K/K' direct-gap response is expected at `hbar omega=Delta`.  The local formal Mao-toy calculation indeed peaks near `3.18 eV`; Mao Fig. 10's M-only-looking `~6 eV` panel is therefore not a reliable validation benchmark for the full neutral tight-binding response.
