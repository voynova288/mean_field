# Mao hTTG retry with tetrahedron / interval spectra

Date: 2026-05-30

## Purpose

Retry the Mao hTTG shift-current spectra using the numerical lesson from the Hipolito benchmarks: do not sample narrow resonant features as isolated k-grid events.  The new workflow bins transition energies with linear triangles over the moire primitive cell and integrates the Lorentzian denominator analytically over each transition-energy bin.

## New code

- `run_htg_bandpair_spectra_tetra.py`

Numerical method:

```text
primitive moire cell -> linear triangular transition-energy histogram
optional C3-rotated cell average
Lorentzian interval integral:
  int_E0^E1 eta/pi / ((omega-E)^2+eta^2) dE
```

This is a quadrature fix, not a plot smoother.

## Slurm runs

- `134370`: central flat-band and central-window retry.
- `134379`: active-24 retry, shell 3 mesh 12.
- `134382`: central-flat shell-5 mesh convergence (`m=24,28`).
- `134385`: active-24 retry, shell 4 mesh 12.

## Outputs

Central flat pair, shell 5, C3 averaged:

```text
results/shift_current_htg/mao_retry_tetra_central_flat_shell5_m28/
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_retry_tetra_support/mao_retry_tetra_central_flat_shell5_m20/
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_retry_tetra_support/mao_retry_tetra_central_flat_shell5_m24/
```

Important figures:

```text
central_flat_eta2_code_axes.png
central_flat_eta2_rot4p8_reflect_y.png
tensor_symmetry_audit_eta2/tensor_symmetry_audit.png
```

Central window `[-4,3]`, shell 3, mesh 18, C3 averaged:

```text
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_retry_tetra_support/mao_retry_tetra_central_window_shell3_m18/
```

Active-24 window `[-12,11]`, mesh 12, C3 averaged:

```text
results/shift_current_htg/mao_retry_tetra_active24_shell4_m12/
results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_retry_tetra_support/mao_retry_tetra_active24_shell3_m12/
```

## Main numerical findings

The narrow-broadening wiggles are removed.  The dominant smooth peak remains near `0.030--0.031 eV`.

Representative eta=2 meV peaks:

```text
central_flat shell5 m28:
  raw code-axis sigma^{y;xx}: 6.92e3 microA nm V^-2 at 0.0302 eV
  raw code-axis sigma^{x;yy}: 6.6e-1 microA nm V^-2
  empirical reflect_y+rot4.8: sigma^{y;xx}=-6.71e3, sigma^{x;yy}=-1.72e3 at 0.0302 eV

central_flat convergence at eta=2 meV, empirical reflect_y+rot4.8, relative to mesh 28:
  mesh20: x;yy 0.74%, y;xx 0.69%
  mesh24: x;yy 0.36%, y;xx 0.32%

central_window shell3 m18:
  raw code-axis sigma^{y;xx}: 6.98e3 microA nm V^-2 at 0.0302 eV
  raw code-axis sigma^{x;yy}: 5.47 microA nm V^-2

active24 shell4 m12:
  raw code-axis sigma^{y;xx}: 7.02e3 microA nm V^-2 at 0.0304 eV
  raw code-axis sigma^{x;yy}: 3.0e-3 microA nm V^-2
  empirical reflect_y+rot4.8: sigma^{y;xx}=-6.80e3, sigma^{x;yy}=-1.75e3 at 0.0304 eV
```

C3 / symmetry audit at eta=2 meV:

```text
central_flat group1/group2 ratio: 2.63e-6 (m20), dropping to 6.63e-7 (m24); m28 remains below 1e-4 with raw x;yy tiny
central_window group1/group2 ratio: 1.63e-4
active24 group1/group2 ratio: 1.21e-4 (shell3), and shell4 raw x;yy is again tiny
```

Thus the exact symmetry conclusion remains: in the published Eq. (13) convention with equal layer mass, the raw code-axis `sigma^{x;yy}` channel is symmetry-forbidden / numerically tiny, while the other C3 tensor coefficient is large.

## Interpretation

The new tetrahedron/interval spectra solve the numerical wiggle problem.  They do **not** solve the Mao component-label / axis-convention problem.  Applying the previously diagnosed empirical `reflect_y + rotation ~4.8 deg` again mixes the large tensor coefficient into both plotted channels and gives a smooth paper-like two-component shape, but this rotation is still not derived from a published Mao coordinate convention and therefore is not an honest final reproduction.

Status after this retry:

- Numerical wiggles: fixed for selected hTTG band-pair/window spectra.
- Peak scale: smooth THz-scale response, around `7e3 microA nm V^-2` for eta=2 meV, close to Mao's order of magnitude.
- Fig. 1(b)/Fig. 2 final reproduction: still blocked by symmetry/component-label convention, not by spectral wiggles.
