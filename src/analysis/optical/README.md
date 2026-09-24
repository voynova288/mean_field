# Unified optical-response API

This package is the front door for future optical-response workflows. A system
adapter supplies the k-point data

```python
energies, evecs = diagonalize(H(k))
dhdk = np.stack([dHdkx(k), dHdky(k)])
d2hdk = optional_second_derivatives
```

and then calls the generic API instead of importing a paper-specific response
module directly.

## Supported formula families

```text
Transition-table surface:
kind="shift_current"       -> analysis.shift_current
kind="injection_current"   -> analysis.injection_current
kind="cpge"                -> alias for injection_current

K-point tensor surface:
kind="linear_conductivity"       -> interband Kubo sheet tensor
kind="kerr_faraday"              -> Kubo tensor + physical Maxwell sheet optics
kind="second_harmonic_generation" -> Passos finite-band velocity-gauge SHG
kind="third_harmonic_generation"  -> Passos finite-band velocity-gauge THG
kind="linear"                    -> alias for linear_conductivity
kind="kerr"/"faraday"            -> aliases returning the paired KerrFaradayResponse
kind="shg"/"thg"                 -> harmonic-generation aliases
```

The two surfaces are deliberately separate because Kerr/linear conductivity
use the full complex tensor rather than positive-transition tables. Passing a
linear/Kerr kind to a transition helper raises an error pointing to
`optical_response_from_kpoint_data`.

The common module does not implement system physics. It dispatches to:

- `analysis.shift_current` for generalized-derivative/shift-vector response;
- `analysis.injection_current` for length-gauge injection-current / CPGE kernels
  `Delta v * r * r`;
- `analysis.optical.linear` for interband optical conductivity;
- `analysis.optical.kmesh` for linear accumulation and named Liu--Dai compatibility helpers;
- `analysis.optical.magneto` for low-level quasi-2D sheet Kerr/Faraday formulas;
- `analysis.optical.kerr` for the typed generic
  `optical_response_from_kpoint_data` front door and lower-level direct helpers;
- `analysis.optical.passos` for the production finite-band SHG/THG recursion;
- `analysis.optical.harmonic` for its typed generic formulation/front door;
- `analysis.optical.shg` for the older Liu--Dai Eq.(4) compatibility scaffold.

All Berry-connection response families reuse the gauge-safe Hamiltonian-gauge
infrastructure from `analysis.response_derivative_gauge`. The k-mesh optical
payload helper `optical_kpoint_data_from_eigensystem` also routes `dH/dk` through
that same layer before storing `velocity_h`, so system adapters should pass raw
basis-space Hamiltonian derivatives rather than pre-rotated ad hoc matrices. Do
not differentiate raw eigenvector phases or raw `np.angle(A_mn)` values in
system adapters.

## Minimal usage

```python
from analysis.optical import (
    precompute_optical_tensors,
    positive_optical_transition_terms,
    accumulate_optical_spectrum,
    cpge_part,
)

# Shift current
shift = precompute_optical_tensors(
    "shift_current", energies, evecs, dhdk, d2hdk=d2hdk,
    denominator_cutoff_ev=1e-10,
)
transitions, weights = positive_optical_transition_terms(shift, "x;yy")
sigma = accumulate_optical_spectrum(
    "shift_current", omega_ev, transitions, weights, k_weight=k_weight,
    eta_ev=eta_ev,
)

# Injection current / CPGE
inj = precompute_optical_tensors("cpge", energies, evecs, dhdk)
transitions, weights = positive_optical_transition_terms(inj, "y;xy")
complex_response = accumulate_optical_spectrum(
    "injection_current", omega_ev, transitions, weights, k_weight=k_weight,
    eta_ev=eta_ev,
)
cpge = cpge_part(complex_response)
```

For backward-compatible transition-table output, the optical front door uses
the historical Joya/legacy shift-current conversion
`SHIFT_CURRENT_PREFAC_UA_NM_PER_V2` with phase `-1j` unless the caller overrides
`shift_prefactor` and `shift_prefactor_phase`. This is not an inferred
WannierBerri convention. The lower-level shift-current conversion helpers
always require an explicit prefactor and phase.

## Kerr / Faraday route

The generic route computes a physical sheet tensor
``j_a=sum_b sigma_ab E_b`` and solves the normal-incidence Maxwell boundary
conditions. Liu--Dai 2020 (`reference/s41524-020-0299-4.pdf`) supplies the Kubo
and quasi-2D formulas, while Okada 2016 below independently closes the signed
Hall/Fresnel convention.

The production low-level API is
`sheet_scattering_at_normal_incidence`. For scalar nonmagnetic media with
``Y_i=n_i/Z0`` it solves

```text
[(Y1+Y2) I + sigma] T = 2 Y1 I
R = T - I
```

and returns full 2x2 Jones matrices with axes
``(output polarization,input polarization)``. Conductivity/media leading axes
are broadcast explicitly, and arbitrary incident Jones fields can be applied
with ``transmitted_field`` or ``reflected_field``. The convenience
``faraday_kerr_from_sheet_conductivity`` API selects unit x-polarized incidence
and defaults to vacuum/vacuum, but also accepts generic incident/transmitted
refractive indices.

Conventions are fixed, not guessed:

```text
complex fields: exp(-i omega t)
sample axes: fixed right-handed (x,y,+z)
local solve: medium 1 -> medium 2
sample propagation: stored per solve as +z or -z
reflected Jones components: fixed sample x/y axes, not viewer-reoriented
sigma[a,b] = j_a/E_b
```

``complex_polarization_from_jones`` returns principal orientation, signed
ellipticity, and masks distinguishing zero fields from circular fields with
undefined orientation. For ordered spectra,
``unwrap_polarization_angle`` uses ``0.5*unwrap(2*theta)`` independently on
each contiguous valid segment; callers must sample densely enough that the
physical orientation changes by less than ``pi/2`` between adjacent valid
points. No branch is inferred across a zero/circular gap.

```python
from analysis.optical import (
    NormalIncidenceSameSideGeometry,
    optical_response_from_kpoint_data,
)

response = optical_response_from_kpoint_data(
    "kerr",
    omega_ev,
    kpayloads,
    eta_ev=eta_ev,
    si_prefactor=si_prefactor,
    selected_bands=selected_bands,
    geometry=NormalIncidenceSameSideGeometry(
        incident_refractive_index=n1,
        transmitted_refractive_index=n2,
    ),
)
theta_f = response.faraday_angle_rad
theta_k = response.kerr_angle_rad
sigma_siemens = response.conductivity
```

`linear_conductivity_tensor_from_gauge_data` leaves the overall unit prefactor to
callers.  For Kerr/Faraday, the conductivity passed to `faraday_kerr_*` must be a
2D sheet conductivity in Siemens.  Liu--Dai Eq. (20) uses the Kubo prefactor
`e^2/hbar`; with the common `(2*pi)^-2` BZ-integral convention this is
`E2_OVER_HBAR_S = 2*pi*E2_OVER_H_S`, while quantized Hall plateaus are still
reported in units of `E2_OVER_H_S = e^2/h`.

For grid integrations, system adapters yield `OpticalKPointData` and use the
same generic k-point dispatcher:

```python
from analysis.optical import (
    optical_kpoint_data_from_eigensystem,
    optical_response_from_kpoint_data,
)

kpayloads = [
    optical_kpoint_data_from_eigensystem(
        energies_k, evecs_k, dhdk_k, weight=w_k, mu_ev=mu,
    )
    for energies_k, evecs_k, dhdk_k, w_k in system_k_iterator
]
linear = optical_response_from_kpoint_data(
    "linear", omega_ev, kpayloads, si_prefactor=si_prefactor,
)
```

The lower-level `linear_conductivity_from_kpoint_data`,
`kerr_faraday_from_kpoint_data`, and
`faraday_kerr_from_sheet_conductivity` functions remain available for formula
unit tests, compatibility workflows, and diagnostics. New physical workflows
should normally use `optical_response_from_kpoint_data`. Its `si_prefactor` is
required and has no unit-unsafe default.

Liu--Dai Eq. (21) prints the opposite transverse-field sign from the one derived
from its own Eqs. (17)-(20) and ``j=sigma E``. The generic API follows Maxwell.
Only workflows reproducing the paper's signed figures should explicitly call
`kerr_faraday_from_kpoint_data_liu_dai_2020_printed`; this named compatibility
route leaves the conductivity unchanged and flips only the transverse fields.

Migration note: generic Kerr/Faraday artifacts produced before the Okada/QWZ
sign audit on 2026-07-12 used the Liu-Dai printed transverse sign. Their saved
conductivity tensors remain valid, but non-Liu signed angles should be rerun
through the physical generic solver. Liu-Dai paper-comparison workflows are not
silently changed; they now select the named compatibility route explicitly.

Okada et al. 2016 (`reference/ncomms12245.pdf`) provides an independent
substrate-backed QAH cross-check. Its pulse geometry is not the same as the
free-standing helper above: Faraday transmission is from vacuum into the
substrate, while the delayed Kerr pulse reflects from the conducting sheet on
its substrate side. Use
`faraday_kerr_from_sheet_conductivity_on_substrate` for an existing sheet
tensor, or the typed generic route

```python
from analysis.optical import Okada2016MixedPulseGeometry

response = optical_response_from_kpoint_data(
    "kerr",
    omega_ev,
    kpayloads,
    si_prefactor=si_prefactor,
    geometry=Okada2016MixedPulseGeometry(
        substrate_refractive_index=3.47,
    ),
)
```

The older `kerr_faraday_from_kpoint_data_on_substrate` direct helper remains
available for explicit compatibility workflows. Both routes require an
explicit SI prefactor so adapter-unit conductivities cannot silently enter the
Fresnel solve. The physical tensor
mapping is direct: ``sigma_xy(repo)=sigma_xy(Okada)``. In the repository's
default FHS/Kubo orientation, positive Chern number is stored as
``sigma_xy=+C e^2/h`` and ``sigma_yx=-C e^2/h``. For the ideal QAH limit
``sigma_xy=e^2/h``, ``sigma_xx=0``, and InP
`n_s=3.47`, the prediction is

```text
theta_F = 3.2650 mrad
theta_K = 9.1737 mrad
f(theta_F, theta_K) = alpha = 0.00729735
```

consistent with the paper's finite-dc-conductance estimates of about 3.1 and
8.7 mrad. The reusable formula benchmark is
`analysis.optical.benchmarks.okada_2016.okada2016_qah_benchmark`.

## SHG/THG production route: Passos 2018

The production harmonic-generation path implements Passos et al., Phys. Rev. B
97, 235446 (2018), Eqs. (13), (31), and (32). An order-`n` response requires the
complete covariant Hamiltonian derivative tower through order `n+1`: SHG uses
`G1,G2,G3`, and THG uses `G1,G2,G3,G4`. All current/contact sectors are retained;
they must not be replaced by a three-velocity transition sum.

For a fixed, complete, k-independent orthonormal basis, construct each point
with `passos_kpoint_data_from_fixed_basis_derivatives`. The grid must cover one
primitive BZ exactly. `PassosPrimitiveBZGrid` supports a uniform midpoint rule
and exact replacement of selected base cells by equal refined subcells, without
adding or double-counting BZ weight.

```python
from analysis.optical import (
    IndependentInputAdiabaticSwitching,
    PassosFiniteBandVelocityGauge,
    harmonic_response_from_kpoint_data,
)

formulation = PassosFiniteBandVelocityGauge(
    switching=IndependentInputAdiabaticSwitching(gamma_ev=gamma_ev),
)
shg = harmonic_response_from_kpoint_data(
    "shg",
    omega_ev,
    passos_kpoints,
    formulation=formulation,
)
thg = harmonic_response_from_kpoint_data(
    "thg",
    omega_ev,
    passos_kpoints_order4,
    formulation=formulation,
)
```

The existing `optical_response_from_kpoint_data(...,
adiabatic_gamma_ev=gamma_ev)` form is retained as a compatibility facade and
constructs the same typed formulation. Supplying both forms is rejected before
consuming the k-point iterator.

The Passos payload owns the Cartesian reciprocal-coordinate length and BZ
measure, so the dispatcher derives the SI prefactor. For a 2D sheet, SHG output
is A m/V^2 and THG output is A m^2/V^3. `passos_shg_susceptibility` converts SHG
to the sheet susceptibility in m^2/V; it returns an effective-bulk m/V value
only when the caller supplies an explicit physical thickness. `si_prefactor`, `eta_ev`,
`include_bz_factor`, `denominator_cutoff_ev`, `selected_bands`, and Kerr
`geometry` are rejected on this route. A selected window of a larger model or a
local continuum patch is not a valid substitute for rebuilding the complete
finite model and all induced derivative vertices.

The implementation is checked against literal Eq. (31)/(34) matrix oracles, an
independent fixed-basis Liouvillian solve, intrinsic permutation symmetry,
U(1) and degenerate-subspace covariance, Rice--Mele inversion-forbidden SHG,
and the published full-BZ graphene THG curves in Passos Fig. 3. The distinct
scattering-term curves in Passos Figs. 1/2 are reproduced by the
Mikhailov-2016 analytical benchmark, with one `i*gamma` in each cumulative
length-gauge density-matrix pole. That paper-specific oracle is not exposed as
a generic scattering solver and must not be emulated by changing the poles in
`passos.py`. See `HARMONIC_GENERATION.md` for the formulation boundary and
`PASSOS_2018_BENCHMARK.md` for parameters, provenance, convergence, and limits.

## Liu--Dai Eq.(4) SHG compatibility route

```python
from analysis.optical import (
    LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2,
    LIU_DAI_2020_EQ4_SHG_PREFAC_S,
    shift_current_velocity_gauge_from_kpoint_data,
    shg_conductivity_from_kpoint_data,
    shg_susceptibility_from_conductivity,
)

# Liu-Dai Eq.(4) dc shift-current velocity-gauge form.
shift_vg = shift_current_velocity_gauge_from_kpoint_data(
    omega_ev,
    kpayloads,
    eta_ev=eta_ev,
    prefactor=LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2,
    selected_bands=selected_bands,
)

# Liu-Dai Eq.(4) SHG velocity-gauge form.
shg = shg_conductivity_from_kpoint_data(
    omega_ev,
    kpayloads,
    eta_ev=eta_ev,
    prefactor=LIU_DAI_2020_EQ4_SHG_PREFAC_S,
    selected_bands=selected_bands,
)
chi_pm_per_v = shg_susceptibility_from_conductivity(
    shg.conductivity, omega_ev, output="pm_per_v",
)
```

These velocity-gauge Eq.(4) helpers are formula-level compatibility scaffolds for Liu--Dai Fig.4, not the production generic SHG implementation.
The exported positive-magnitude prefactors assume this repo's common TBG/TDBG
convention `velocity_h=dH/dk` in eV nm and k weights in nm^-2.  Liu--Dai Eq.(4)
is odd in the electron charge, so paper-style Fig.4 workflows should use the
signed `LIU_DAI_2020_EQ4_*` prefactors, not the positive magnitudes.  The strict
SI 2D coefficient is A m/V^2, while the paper labels the 2D shift-current
response as microA/V^2 with the sheet length suppressed; report this convention
explicitly before claiming a paper reproduction.  Also audit degeneracy/window
handling and any system-specific symmetrization factor.

By default, the Eq.(4) helpers skip terms with static energy denominators below
`denominator_cutoff_ev`.  If a target formula should instead keep diagonal/small
denominators and let the explicit `i eta` broadening regularize them, pass
`skip_degenerate_energy_denominators=False`.  This is a paper-convention choice,
not a numerical stabilization knob.

## Benchmark notes

See `KERR_MODULE.md` for the production normal-incidence API, conventions,
branch handling, invariants, and deliberately deferred electromagnetic scope.

See `HARMONIC_GENERATION.md` for the typed production SHG/THG API and explicit
causal-prescription boundaries.

See `PASSOS_2018_BENCHMARK.md` for the production finite-band SHG/THG formula,
full-BZ graphene reproduction, Fig.1/2 analytical benchmark, unit contract, and
independent SHG gates.

See `LIU_DAI_2020_BENCHMARK.md` for compatibility formula anchors and expected
benchmark scales for Kerr/Faraday, SHG, and shift-current comparisons.

## Boundaries

- The generic electromagnetic solver currently covers one zero-thickness 2D sheet at normal incidence between scalar nonmagnetic media.
- Oblique-incidence TE/TM mixing, finite-thickness films, coherent multilayers, anisotropic/magnetic bulk media, and backside propagation are deferred to separate derived scattering-matrix layers.
- Linear conductivity is interband-only; metallic low-frequency work requires a separately audited intraband/Drude/contact contribution.
- Momentum units are inherited from the system adapter's `dH/dk`.
- Passos SHG/THG derives SI units from its typed payload; other response prefactors, spin/valley multiplicity, scattering time, paper signs, plotting labels, colorbars, and Slurm orchestration remain explicit adapter/workflow choices.
- External Berry-position terms are not exposed here until their generalized
  derivatives are derived and tested.
- Broad optical grid integrations are compute jobs and should run through Slurm,
  not on login nodes.
