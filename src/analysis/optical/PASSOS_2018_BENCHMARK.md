# Passos 2018 finite-band SHG/THG benchmark

## Scope

This module implements D. J. Passos *et al.*, Phys. Rev. B **97**, 235446
(2018), arXiv:1712.04924v2, as the production finite-band velocity-gauge
harmonic-response route. The paper demonstrates THG; the same recursion at
order two supplies the generic SHG implementation.

Primary local evidence:

```text
reference/1712.04924v2.pdf
SHA256 5a51fc9b2af3b7c1d205d62e9178d99eac0bea0c7c792789358c71963b042dc9
```

The official arXiv TeX was inspected for Eqs. (13), (31), (32), (34), and
(43)-(50). Published Fig. 3 vectors were extracted from a page-9 SVG without
smoothing, amplitude rescaling, coordinate alignment, or interpolation across
clipping.

## Scaled recursion

The implementation stores

```text
G^(r) = hbar^r h^(r)
R^(r) = hbar^r rho^(r)
z     = E + i gamma
```

where `G^(r)` has numerical units `eV * coordinate_length^r`. For equal-input
harmonic generation, Eq. (31) becomes

```text
R^(a1...ar) = [sum_m [G^(a1...am), R^(a(m+1)...ar)] / m!]
               / [r z - (epsilon_s-epsilon_s')].
```

Eq. (32) then contains every current/contact sector

```text
sum_p Tr[G^(beta a1...ap) R^(a(p+1)...an)] / p!
```

and the intrinsic permutation average over input `(axis,frequency)` pairs. For
SHG this reproduces the three terms printed in Eq. (34): `G1*R2`, `G2*R1`, and
`G3*R0/2`. Dropping the latter terms is not a finite-band gauge-complete
implementation.

The causal broadening used in paper Fig. 3 is applied to each input frequency:
`E_j -> E_j+i*gamma`. Therefore an r-photon pole has `r*gamma`; a constant
width in every pole is a different phenomenology.

For electron charge `q=-e`, eV input energies, and Cartesian reciprocal
coordinates, the automatic SI prefactor is

```text
i^n (-1)^(n+1) (e^2/hbar) length_unit_m^(n+1-d).
```

For a 2D sheet, SHG is in A m/V^2 and THG is in A m^2/V^3. The sheet SHG
susceptibility is in m^2/V. An m/V effective-bulk value requires an explicit
physical thickness.

## Payload and BZ contract

`PassosKPointData` contains all bands of the declared finite model and the
covariant derivative tower through order `n+1`. The fixed-basis constructor requires a square complete eigensystem, checks that
it diagonalizes the supplied Hamiltonian, and requires the explicit declaration
`fixed_basis_connection_is_zero=True`. A moving/projected/embedded/
nonorthogonal basis must supply a separately derived covariant tower.

`PassosPrimitiveBZGrid` covers one primitive reciprocal parallelogram. Uniform
base cells use midpoint quadrature. Selected base cells may be *replaced* by
equal refined subcells; this preserves the exact BZ measure and avoids overlap.
Every grid index must be supplied exactly once in increasing index order so the
compensated reduction is deterministic. A selected band window, repeated
k point with compensating weight, or local continuum patch is rejected as a
production Passos input.

Model/source certificates are explicit caller declarations and consistency
metadata, not formal proofs that an external Hamiltonian implementation is
correct.

## Paper Figs. 1 and 2: scattering-term benchmark

Figs. 1/2 use a different finite-scattering phenomenology from Fig. 3.  The
length-gauge equation of motion contains one `+i*gamma` in every cumulative
density-matrix pole, rather than adding `i*gamma` to each input.  With
`Omega=hbar*omega/mu` and `Gamma=gamma/mu`, the equal-input poles are

```text
O1   = (Omega   + i Gamma)/2
O12  = (2 Omega + i Gamma)/2
O123 = (3 Omega + i Gamma)/2
```

`benchmarks/mikhailov_2016.py` implements Mikhailov, PRB 93, 085403 (2016),
Eqs. (59)-(78), including the `(3/0)`, `(2/1)`, `(1/2)`, and `(0/3)` sectors.
The closed forms are used away from coincident arguments; for
`Omega < 0.1`, where their removable `(a-b)`/`(a-c)` denominators become
ill-conditioned, the implementation evaluates the defining Eqs. (69)-(73)
instead, requires mapped Gauss-Legendre agreement between successive orders
(default 512 and 1024), and records the fallback count/order.
For `gs=gv=2` and `vF=sqrt(3)*a*t/(2*hbar)`, its Eq. (61) normalization is
exactly the Passos normalization `3*q^4*a^2*t^2/(16*pi*hbar*mu^4)`.

The official standalone source-panel PDFs `E1.pdf`, `E2.pdf`, and `E3.pdf`
were extracted from the Passos arXiv source tar.  Their vector paths were
calibrated only from printed ticks; the clipped Fig. 1b paths remain separate
segments.  No smoothing, coordinate alignment, amplitude scaling, or
interpolation across clipping was used.

```text
results/optical/CURRENT_PASSOS2018/fig12_source_vector_digitization/
results/optical/CURRENT_PASSOS2018/passos2018_fig12_current_best_mikhailov_20260716/
```

Direct residuals at every visible vector point are:

```text
Fig. 1a Re RMS/max = 0.001381 / 0.01215 sigma0
Fig. 1a Im RMS/max = 0.001701 / 0.01376 sigma0
Fig. 1b Re RMS/max = 0.003054 / 0.05165 sigma0
Fig. 1b Im RMS/max = 0.008105 / 0.16036 sigma0
Fig. 2  Re RMS/max = 0.008554 / 0.04623 sigma0
Fig. 2  Im RMS/max = 0.009791 / 0.06554 sigma0
```

Primary provenance:

```text
Passos E1.pdf  56ca345df39ffd0bded08182c78ff193d3b9d6525685482f1a2f72fc3103ea3b
Passos E2.pdf  55bde8e003fc92bae52c8aff19aad0e0ba570152f00fcde88cd6ee24887e1599
Passos E3.pdf  b6d02adeba40c64f647011f0d06fdb6781146189735a1100fec6c06849b4e715
Mikhailov TeX d2632736c895786291827c6f59b81d5a31761a9fee0c01392dce4e1fa3bbc631
```

These residuals validate the analytical scattering-term target used by Passos.
The bound current source is accepted for the optical scope by Slurm jobs
`184926`, `184927`, `184928`, and atomic artifact job `184929`; the exact
repository-wide caveat is tracked in
`results/optical/CURRENT_PASSOS2018/PASSOS_FIG12_AND_GENERIC_HARMONIC_STATUS_20260716.md`.
This does not promote a generic non-Abelian length-gauge scattering solver: changing
`r*gamma` to `gamma` in the velocity-gauge recursion would omit the
field-dependent equilibrium-density vertices required by gauge equivalence.

## Paper Fig. 3 reproduction

The benchmark uses the complete nearest-neighbour graphene model of Eqs.
(43)-(50):

```text
t = 1 eV
mu/t = 0.1
T = 0
explicit spin copies = 2
full primitive BZ (both valleys included; no extra valley factor)
sigma0 = 3 q^4 a^2 t^2 / (16 pi hbar mu^4)
```

Accepted artifacts:

```text
results/optical/CURRENT_PASSOS2018/passos2018_fig3_current_best_refined_20260715/
```

Panel a (`gamma/t=0.011`) uses Slurm run 183192, an exact base-192 grid with
Dirac-cell refinement factor 8 (69,120 points). Panel b (`gamma/t=0.001`) uses
run 183189, base 384 with refinement factor 16 (669,696 points). Source-vector
residuals exclude only points clipped at the printed axis boundary:

```text
Fig. 3a Re RMS/max = 0.05416 / 0.20409  sigma0
Fig. 3a Im RMS/max = 0.07759 / 0.19880  sigma0
Fig. 3b Re RMS/max = 0.02265 / 0.08991  sigma0
Fig. 3b Im RMS/max = 0.02253 / 0.09261  sigma0
```

At fixed local fine spacing, base-96 -> base-192 changes panel a by complex RMS
`0.02086 sigma0`; base-192 -> base-384 changes the narrower panel b by
`0.14054 sigma0` over `0.02 <= hbar*omega/t <= 0.2`. Separately, panel-b local
refinement 32 -> 64 at base 96 changes the curve by RMS `0.02466 sigma0`.
These are direct quadrature differences, not post-hoc fitting. The final source
RMS values are well below the approximate `0.24 sigma0` graphical half-line
scale of the published vector strokes. The `gamma/t=0.001` case requires both
local Dirac refinement and outer-BZ refinement; a uniform 384x384 mesh visibly
retains comb-like quadrature structure.

Published vectors are evidence, not raw author arrays. The original numerical
integration code and source arrays are unavailable.

## Independent SHG gates

The order-two route is tested separately from the graphene THG plot:

- literal Eq. (31)/(34) matrix-loop equality;
- an independent fixed-basis Liouvillian solve;
- intrinsic input-index permutation symmetry;
- random band U(1) and exact spin-degenerate U(2) covariance;
- explicit Eq. (32) sector and BZ-cancellation diagnostics;
- exact refined-cell weight oracle including `(2*pi)^-d`;
- inversion-broken Rice--Mele SHG is nonzero;
- restoring Rice--Mele inversion makes full-BZ electric-dipole SHG vanish;
- direct Passos calls and `optical_response_from_kpoint_data("shg"/"thg")`
  agree.

## Public API

```python
formulation = PassosFiniteBandVelocityGauge(
    switching=IndependentInputAdiabaticSwitching(gamma_ev=gamma_ev),
)
response = harmonic_response_from_kpoint_data(
    "shg",  # or "thg"
    photon_energies_ev,
    passos_kpoints,
    formulation=formulation,
)
```

The legacy `optical_response_from_kpoint_data(...,
adiabatic_gamma_ev=gamma_ev)` form constructs the same formulation internally.

The generic Passos route rejects `si_prefactor`, `eta_ev`,
`include_bz_factor`, `denominator_cutoff_ev`, `selected_bands`, and Kerr
`geometry`. This prevents Liu--Dai compatibility conventions from entering the
production recursion silently.

The older `shg_conductivity_velocity_gauge` and
`shg_conductivity_from_kpoint_data` functions remain named Liu--Dai Eq. (4)
compatibility scaffolds. They use three first-derivative velocity matrices and
must not be cited as validation of the Passos finite-band SHG chain.

## Limits

- Independent-particle, electric-dipole response only.
- No excitons, electron-hole interactions, phonon assistance, or nonlinear
  intraband collision model.
- The adiabatic broadening is a specified phenomenology, not a microscopic
  relaxation theory.
- A projected/moving basis requires its own audited covariant derivative tower.
- Sheet-to-bulk susceptibility conversion requires an explicit thickness.
