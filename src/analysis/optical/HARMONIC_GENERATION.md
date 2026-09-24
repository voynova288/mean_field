# Generic harmonic-generation API

## Production formulation

The production SHG/THG backend is the complete finite-band velocity-gauge
recursion of Passos et al. (2018), implemented in `passos.py`.  Its gauge and
causal prescription are coupled explicitly:

```python
from analysis.optical import (
    IndependentInputAdiabaticSwitching,
    PassosFiniteBandVelocityGauge,
    harmonic_response_from_kpoint_data,
)

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

The legacy front door remains valid:

```python
response = optical_response_from_kpoint_data(
    "shg",
    photon_energies_ev,
    passos_kpoints,
    adiabatic_gamma_ev=gamma_ev,
)
```

It constructs the same typed formulation internally.  Supplying both
`formulation` and `adiabatic_gamma_ev` is rejected before consuming the k-point
iterator.

## Physical contract

- `shg` requires a complete derivative tower through third order.
- `thg` requires a complete derivative tower through fourth order.
- Every band of the declared finite model is included.
- The grid covers one primitive full BZ exactly once in increasing
  `grid_index` order; refined cells replace base cells rather than being added
  to them. The canonical order makes compensated reduction deterministic.
- `fixed_basis_complete` means a complete, k-independent orthonormal basis
  with zero basis connection. Moving/embedded/nonorthogonal bases must provide
  an independently derived externally covariant tower.
- A selected-band window or local continuum patch is not a valid production
  input.
- Independent-input adiabatic switching means
  `E_j -> E_j + i*gamma`; an `r`-photon pole therefore contains `r*gamma`.
- For a 2D sheet, SHG conductivity is in `A m / V^2` and THG conductivity is
  in `A m^2 / V^3`.
- Native SHG susceptibility is a sheet response in `m^2/V`; an effective-bulk
  value in `m/V` requires an explicit positive thickness.

Results report:

```text
gauge_kind = passos_finite_band_velocity_gauge
causal_prescription_kind = independent_input_adiabatic
```

## Formulation boundaries

| Formulation | Status |
|---|---|
| Complete finite-band Passos velocity gauge + independent-input adiabatic switching | generic production |
| Liu-Dai Eq. (4) three-velocity SHG | named compatibility only |
| Mikhailov graphene `xxxx` THG + density-matrix scattering | analytical benchmark only |
| Non-Abelian length gauge + density-matrix scattering | not implemented |

The Mikhailov benchmark used for Passos Figs. 1/2 has one `+i*gamma` in every
*cumulative density-matrix pole*.  It must not be emulated by replacing
`r*gamma` with `gamma` in `passos.py`: the velocity-gauge scattering equation
also requires the field-dependent equilibrium density matrix.  A future generic
length-gauge backend needs a mesh-level non-Abelian transport/derivative payload
and independent gauge-equivalence validation.

## Susceptibility

```python
from analysis.optical import shg_susceptibility_from_response

sheet = shg_susceptibility_from_response(response)
bulk_like = shg_susceptibility_from_response(
    response,
    effective_thickness_m=thickness_m,
)
```

With the package convention `exp(-i*omega*t)`,

```text
chi_sheet^(2) = i sigma_sheet^(2) / (2 epsilon0 omega).
```

## Current validation

The final 53-entry input manifest was verified by Slurm jobs `184926` and
`184927` (29 focused and 93 optical tests), followed by job `184928` (666
repository tests passed with only two explicitly deselected, unrelated RLG-hBN
failures previously demonstrated in job `184834`). Atomic Fig.1/2 artifact job
`184929` records the validation chain and result hashes. See
`results/optical/CURRENT_PASSOS2018/PASSOS_FIG12_AND_GENERIC_HARMONIC_STATUS_20260716.md`.
