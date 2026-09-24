# Liu-Dai 2020 optical formula anchors

Reference:

```text
reference/s41524-020-0299-4.pdf
Jianpeng Liu and Xi Dai,
"Anomalous Hall effect, magneto-optical properties, and nonlinear optical
properties of twisted graphene systems",
npj Computational Materials 6, 57 (2020).
```

This document records only reusable formula and convention anchors for
`analysis.optical`. Paper-run history, Slurm jobs, panel scores, and unresolved
reproduction details belong under `results/optical/`; the archived ledger is

```text
results/optical/LIU_DAI_2020_REPRODUCTION_LEDGER_20260713.md
```

## Linear conductivity and Kerr/Faraday

### Kubo input

Liu-Dai Eq. (20) is the interband optical sheet-conductivity kernel:

```text
sigma_ab(omega)
  = i (e^2/hbar) integral_k sum_{m != n}
      v^a_mn v^b_nm (f_n-f_m)
      / (E_n-E_m)
      / (E_n-E_m-hbar*omega-i*eta).
```

Repository tensor contract:

```text
sigma[a,b] = j_a / E_b
```

For the default ordered `(kx,ky)` FHS/Kubo orientation,

```text
C > 0  ->  sigma_xy = +C e^2/h
           sigma_yx = -C e^2/h.
```

The rendered paper prints a `(2*pi)^3` integration denominator although the
system is two-dimensional. The common implementation uses `(2*pi)^2`; the
independent QWZ benchmark recovers the quantized `e^2/h` plateau with this
measure.

`linear_conductivity_tensor_from_gauge_data` leaves the final unit prefactor to
the caller. For `dH/dk` in energy-times-length and a two-dimensional reciprocal
weight, the audited Kerr route uses

```python
prefactor = 1j * E2_OVER_HBAR_S
```

so the accumulated tensor is a sheet conductivity in Siemens.

### Physical Maxwell conversion

For a sheet between media with wave admittances `Y1` and `Y2`, the generic API
solves

```text
[(Y1+Y2) I + sigma] E_t = 2 Y1 E_i.
```

For x-polarized incidence this gives a transverse field proportional to
`-sigma_yx`. Liu-Dai Eq. (21), and consequently its Eq. (23), prints the
opposite transverse sign from the one derived from its own Eqs. (17)-(20) and
`j=sigma E`.

Therefore the API separation is explicit:

```python
# Generic physical Maxwell convention for new workflows
optical_response_from_kpoint_data(
    "kerr", ..., si_prefactor=1j * E2_OVER_HBAR_S,
    geometry=NormalIncidenceSameSideGeometry(),
)

# Lower-level physical diagnostics
kerr_faraday_from_kpoint_data(...)
faraday_kerr_from_sheet_conductivity(...)

# Signed Liu-Dai paper-figure compatibility only
kerr_faraday_from_kpoint_data_liu_dai_2020_printed(...)
faraday_kerr_from_sheet_conductivity_liu_dai_2020_printed(...)
```

The compatibility route leaves the Kubo conductivity unchanged and flips only
the transverse reflected/transmitted fields and signed angles. It must not be
used as a generic tensor convention.

The generic k-point dispatcher requires an explicit SI `si_prefactor`; direct
Kerr helpers require `prefactor`. Neither route has a unit-unsafe default, and
the Liu-Dai printed sign is intentionally not selectable through the generic
dispatcher.

### Independent cross-check

Okada et al. 2016 supplies the independent substrate-backed QAH check. See:

```text
src/analysis/optical/OKADA_2016_BENCHMARK.md
```

Its QWZ end-to-end chain validates

```text
Chern -> Kubo sigma_xy -> sheet Maxwell solve -> Kerr/Faraday -> alpha.
```

## Second-order response anchors

### C3 tensor structure

Liu-Dai Eq. (3) leaves two independent in-plane components. The reusable
helpers are:

```python
from analysis.optical.benchmarks.liu_dai_2020 import (
    c3_inplane_second_order_tensor,
    decompose_c3_inplane_second_order_tensor,
)
```

For tensor ordering `sigma[current, optical_a, optical_b]`, they implement

```text
sigma_xxx = -sigma_xyy = -sigma_yxy = -sigma_yyx
sigma_yxx =  sigma_xxy =  sigma_xyx = -sigma_yyy.
```

### Velocity-gauge Eq. (4)

The paper uses the same three-velocity transition structure for dc shift
current and SHG. The named Liu--Dai compatibility scaffolds are:

```python
shift_current_velocity_gauge_from_kpoint_data(...)
shg_conductivity_from_kpoint_data(...)
shg_susceptibility_from_conductivity(...)
```

These functions are retained only for Liu--Dai/Ref. 74 compatibility. The
generic `kind="shg"` front door uses the Passos 2018 finite-band recursion and
requires `G1,G2,G3`, including current/contact sectors; see
`PASSOS_2018_BENCHMARK.md`.

The Liu--Dai compatibility susceptibility relation is

```text
chi^c_ab(2 omega) = i sigma^c_ab(2 omega) / (2 epsilon0 omega).
```

Ref. 74, Zhang et al. PRB 97, 241118 (2018), establishes the relevant formula
conventions:

- `l=n` and `m=n` do not contribute;
- `l=m` is the two-band sector and `l!=m` the three-band sector;
- the dc and SHG frequency branches used by the common implementation agree
  with the cited derivation;
- no additional contact/diamagnetic term is supplied by that reference.

Thus the Liu-Dai/Ref. 74 default is

```python
skip_degenerate_energy_denominators = True
```

rather than treating diagonal denominators as an adjustable numerical knob.

### Units and scope

The strict two-dimensional second-order coefficient carries a sheet length.
For production Passos output, `passos_shg_susceptibility` labels the resulting
sheet susceptibility as m^2/V and requires an explicit thickness before
reporting an effective-bulk m/V value.
The paper labels shift current in `microA/V^2`, while the common eV-nm
implementation naturally exposes `microA nm/V^2` (equivalently `A m/V^2`).
This convention must be stated explicitly; it cannot be repaired by visual
rescaling.

Paper-order scales such as `O(10^3)` shift current and `O(10^6) pm/V` SHG are
only sanity checks. They are not evidence of panel reproduction, gauge
completeness, or a justified finite-band truncation.

## Gauge and derivative route

System adapters pass basis-space `dH/dk` to
`optical_kpoint_data_from_eigensystem`, which delegates to
`analysis.response_derivative_gauge.hamiltonian_gauge_data` and stores

```text
velocity_h[a,n,m] = <u_n | dH/dk_a | u_m>.
```

No optical workflow should differentiate raw eigenvector phases or raw
`angle(A_mn)` values.

## Validation boundary

Formula-level tests must distinguish:

1. analytic tensor and boundary-condition identities;
2. independent toy-model/Kubo validation;
3. saved-result consistency;
4. full system recomputation;
5. paper-panel reproduction.

Detailed Liu-Dai Fig. 3/4/5 status remains under:

```text
results/optical/CURRENT_FIG3/
results/optical/CURRENT_FIG4/
results/optical/CURRENT_FIG5/
```

Do not move job ledgers, overlay scores, or unresolved paper-specific diagnostics
back into the common analysis package.
