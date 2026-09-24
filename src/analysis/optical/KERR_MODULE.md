# Generic normal-incidence Kerr/Faraday module

## Scope

The module models one local, zero-thickness 2D conducting sheet between two
scalar nonmagnetic media at normal incidence. The sheet response is a complex
Cartesian tensor in Siemens:

```text
j_a = sum_b sigma[a,b] E_b
sigma.shape = (..., 2, 2)
```

It is independent of a material Hamiltonian. System adapters may supply an SI
sheet tensor directly or obtain it through the common Kubo/k-point layer.

## Fixed conventions

```text
complex fields                exp(-i omega t)
sample axes                   right-handed fixed (x,y,+z)
Jones basis                   (x,y)
matrix axes                   (output polarization,input polarization)
local interface solve         medium 1 -> medium 2
reflected Jones components    fixed sample x/y, not viewer-reoriented
positive Hall tensor          sigma_xy > 0, sigma_yx < 0
```

Every scattering result records whether local incidence propagates along
sample `+z` or `-z`. The Okada delayed pulse therefore carries `-1`, while its
vacuum-to-substrate transmission carries `+1`.

## Maxwell solve

For `Y_i=n_i/Z0`, tangential-field continuity and the sheet-current jump give

```text
[(Y1+Y2) I + sigma] E_t = 2 Y1 E_i
E_r = E_t - E_i
```

Thus the Jones matrices are

```text
T = solve((Y1+Y2) I + sigma, 2 Y1 I)
R = T - I
```

Use:

```python
from analysis.optical import sheet_scattering_at_normal_incidence

scattering = sheet_scattering_at_normal_incidence(
    sigma_siemens,
    incident_refractive_index=n1,
    transmitted_refractive_index=n2,
)
T = scattering.transmission_matrix
R = scattering.reflection_matrix
Et = scattering.transmitted_field(incident_jones)
Er = scattering.reflected_field(incident_jones)
```

Conductivity and medium leading dimensions broadcast. The low-level API can
therefore carry parameter/frequency batches. The high-level spectral Kerr API
is intentionally stricter: refractive indices must be scalar or pointwise
arrays with the same one-dimensional frequency shape as the conductivity.

## Polarization and ellipticity

`complex_polarization_from_jones` evaluates normalized Stokes invariants rather
than dividing by `Ex` to determine orientation:

```text
S0 = |Ex|^2 + |Ey|^2
S1 = |Ex|^2 - |Ey|^2
S2 = 2 Re(Ex* Ey)
S3 = 2 Im(Ex* Ey)
theta = 1/2 atan2(S2,S1)       modulo pi
eta = S3 / (S0 + sqrt(S1^2+S2^2))
```

The returned masks distinguish:

- nonzero noncircular fields: polarization and orientation defined;
- circular fields: ellipticity `+/-1`, orientation undefined;
- zero fields: polarization undefined.

Use `zero_policy="nan"` for spectra with reflection zeros.

## Continuous spectral branches

Principal Kerr/Faraday angles are modulo pi. To unwrap an ordered spectrum:

```python
theta_continuous = response.unwrapped_kerr_angle_rad()
```

or call `unwrap_polarization_angle` directly. It applies
`0.5*unwrap(2*theta)` separately to each contiguous valid segment. It never
bridges zero/circular gaps. Adjacent valid samples must resolve physical angle
changes below `pi/2`; otherwise no discrete unwrap can determine the branch.

## High-level generic API

New workflows should connect through the typed common k-point dispatcher:

```python
from analysis.optical import (
    NormalIncidenceSameSideGeometry,
    optical_response_from_kpoint_data,
)

response = optical_response_from_kpoint_data(
    "kerr",
    photon_energies_ev,
    kpoint_payloads,
    si_prefactor=si_sheet_conductivity_prefactor,
    geometry=NormalIncidenceSameSideGeometry(
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    ),
)
```

Canonical k-point kinds are `linear_conductivity` and `kerr_faraday`; aliases
`linear`, `conductivity`, `kerr`, and `faraday` are accepted. `kerr` and
`faraday` both return the paired `KerrFaradayResponse`.

The `si_prefactor` is mandatory. It must convert the adapter's `dH/dk` and
k-weight units to a 2D sheet conductivity in Siemens. Geometry is represented
by `NormalIncidenceSameSideGeometry` or `Okada2016MixedPulseGeometry`, so
ordinary supported-film reflection cannot be silently confused with Okada's
vacuum-to-substrate Faraday plus substrate-to-vacuum Kerr pulse. Invalid
frequency/media shapes are rejected before an expensive k-point iterator is
consumed.

The current common Kubo kernel is interband-only. The explicitly named
Liu-Dai printed-sign helpers remain compatibility-only and are intentionally
not selectable through the generic dispatcher.

## Validation invariants

The focused test suite checks:

1. bare-interface Fresnel amplitudes;
2. scalar-sheet analytic amplitudes;
3. full matrix boundary residuals;
4. circular/Hall eigenvalues for both signs;
5. lossless Hall-sheet power conservation;
6. joint conductivity/media/Jones broadcasting;
7. scale-safe Stokes recovery and zero/circular masks;
8. modulo-pi unwrapping with invalid gaps;
9. Okada, Liu-Dai compatibility, and QWZ anchors;
10. immutable result arrays and generator-safe selected-band accumulation;
11. generic `linear`/`kerr` dispatch equivalence, typed geometry routing, and validation before consuming k-point iterators.

## Deferred physics

Not yet provided:

- oblique-incidence TE/TM-Hall mixing;
- coherent multilayers or backside propagation;
- finite-thickness bulk films;
- anisotropic or magnetic surrounding media;
- Drude/intraband/contact conductivity.

These require separate derivations and should not be emulated with plotting or
post-processing corrections.
