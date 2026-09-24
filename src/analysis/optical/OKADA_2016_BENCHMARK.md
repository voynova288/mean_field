# Okada et al. 2016 QAH Kerr/Faraday benchmark

Reference:

```text
K. N. Okada et al.,
"Terahertz spectroscopy on Faraday and Kerr rotations in a quantum anomalous Hall state",
Nature Communications 7, 12245 (2016), DOI: 10.1038/ncomms12245.
Local PDF: reference/ncomms12245.pdf
```

## Why this is an independent cross-check

Liu-Dai 2020 tests a computed moire-band optical conductivity and an ideal
free-standing 2D sheet. Okada 2016 instead starts from measured/quantized QAH
sheet conductance and tests electromagnetic boundary conditions for a film on
an InP substrate. It therefore checks the Kerr/Faraday conversion layer without
reusing the Liu-Dai Hamiltonian or Kubo integration.

## Experimental pulse geometry

The two reported angles do not come from reflection and transmission on the
same side of a free-standing sheet:

1. Faraday: incidence from vacuum and transmission into the InP substrate.
2. Kerr: after reflection at the substrate back surface, incidence from the
   substrate side and reflection at the TI sheet.

For right/left circular polarization, Okada Eqs. (1)-(2) use

```text
t_+/- = 2 / (1 + n_s + Y_+/-)
r_+/- = (-1 + n_s - Y_+/-) / (1 + n_s + Y_+/-)
Y_+/- = Z0 (sigma_xx +/- i sigma_xy)
theta_F = (1/2) arg(t_- / t_+)
theta_K = (1/2) arg(r_- / r_+).
```

The corresponding Cartesian implementation is
`faraday_kerr_from_sheet_conductivity_on_substrate`. It obeys the physical
contract ``j_a=sum_b sigma_ab E_b``. Positive Hall conductance means

```text
sigma_xy(repo) = +sigma_H = sigma_xy(Okada)
sigma_yx(repo) = -sigma_H
E_+ = E_x - i E_y
E_- = E_x + i E_y.
```

## Quantized anchor

Using

```text
n_s = 3.47
sigma_xx = 0
sigma_xy = e^2/h
sigma_yx = -e^2/h
```

the ideal prediction is

```text
theta_F = 3.265023 mrad
theta_K = 9.173742 mrad.
```

The paper quotes approximately 3.1 and 8.7 mrad from its measured dc
conductances at 1.5 K; those values are slightly below the ideal quantized
limit because the measured state has residual longitudinal conductance and is
not perfectly quantized. The directly observed low-frequency THz spectra are
about 2.6 and 6.9 mrad and are a separate experimental quantity; this formula
benchmark does not claim to reproduce that pulse-extraction reduction.

## Universal scaling check

Okada Eq. (3) defines

```text
f(theta_F, theta_K)
  = (cot(theta_F) - cot(theta_K))
    / (cot(theta_F)^2 - 2 cot(theta_F) cot(theta_K) - 1).
```

For the dissipationless quantized sheet,

```text
f = alpha = Z0 (e^2/h) / 2 = 0.00729735257,
```

independent of substrate refractive index. More generally, with
``sigma_xy=C e^2/h``, ``sigma_yx=-C e^2/h``, and ``sigma_xx=0``, the
implemented formulas give ``f=C alpha``.

## End-to-end Chern/Kubo closure

A separate Qi-Wu-Zhang two-band Chern insulator supplies a sheet conductivity
without inserting ``e^2/h`` by hand. It is an independent toy source, not the
material model of Okada et al. At mesh 96 and ``eta=1 meV`` the Slurm benchmark
returns

```text
C_lower                       = +1.0000000000
sigma_xy / (e^2/h)            = +0.9999997656
sigma_yx / (e^2/h)            = -0.9999997656
theta_F                       = +3.26501584 mrad
theta_K                       = +9.17375448 mrad
f(theta_F, theta_K)           = +0.00729728593
alpha                         = +0.00729735257
```

New system workflows select this geometry through
`optical_response_from_kpoint_data("kerr", ..., si_prefactor=...,
geometry=Okada2016MixedPulseGeometry(substrate_refractive_index=3.47))`.
The typed geometry prevents this delayed substrate-side reflection from being
confused with ordinary vacuum-side reflection from a supported sheet.

This calculation fixes the generic Maxwell contract to the physical matrix
``(Y_1+Y_2) I + sigma``. The opposite transverse sign printed in Liu-Dai 2020
Eq. (21) is available only through an explicitly named compatibility function.
Run histories and migration reports belong in the ignored results workspace,
not in this reusable formula document.

## Numerical stability and experimental reconstruction helpers

The Eq. (3) implementation uses the equivalent tangent form

```text
p = tan(theta_F), k = tan(theta_K)
f = p (k-p) / (k-2p-p^2 k),
```

which avoids forming two very large cotangents at small angle. The module also
provides:

- `okada2016_scaling_uncertainty` for Jacobian error propagation, with an
  explicit optional Faraday/Kerr covariance;
- `okada2016_scaling_from_dc_conductivity`, implementing the exact isotropic dc
  identity `f=Z0*sigma_xy/[2*(1+Z0*sigma_xx)]`;
- `complex_polarization_from_jones`, which evaluates principal rotation and
  ellipticity from global-phase-invariant Stokes combinations and returns the
  equivalent complex angle with `eta=tanh(Im(angle))`;
- scalar and 2x2 complex normalized-field-transmission inversions. Intensity
  transmission alone is insufficient because its phase is absent.

The optical test suite covers the regular angle convention, global Jones
phase, y-linear and circular limits, transmission round trips, covariance
validation, and the ideal/QWZ chain.

## Published-figure reconstruction boundary

The OA article package and Supplementary Information contain no numerical
source tables or raw THz waveforms. Published PDF paths or raster curves may be
calibrated as source evidence, but that remains digitization rather than
independent acquisition. Per-panel provenance, residuals, Slurm job IDs, and
reproduction plots belong in ignored result reports rather than this common
formula document.

The ideal `3.2650/9.1737` mrad values, dc-derived `3.1/8.7` mrad estimates, and
directly observed `2.6/6.9` mrad spectra must always be kept distinct.

## Validation scope

This benchmark validates:

- sheet-conductivity units in Siemens;
- Hall tensor sign mapping;
- vacuum-to-substrate Faraday boundary conditions;
- substrate-side Kerr reflection;
- the exact, non-small-angle conversion on its principal, modulo-pi branch;
- the universal fine-structure scaling relation.

The direct formula benchmark does not by itself validate a system Hamiltonian
or Kubo prefactor. The separate QWZ end-to-end benchmark supplies that missing
sign/unit check. Neither benchmark validates material-specific disorder or the
experimental THz pulse-extraction procedure.

## Reusable API

```python
from analysis.optical.benchmarks.okada_2016 import okada2016_qah_benchmark

benchmark = okada2016_qah_benchmark()
print(benchmark.angles.faraday_angle_rad)
print(benchmark.angles.kerr_angle_rad)
print(benchmark.scaling_function)
```
