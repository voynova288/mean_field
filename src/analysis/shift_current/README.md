# Generic shift-current module

This package is the common API for shift-current calculations. A physical
system should provide only the k-point adapter data:

```python
energies, evecs = diagonalize(H(k))      # evecs columns are eigenvectors
dhdk = np.stack([dHdkx(k), dHdky(k)])    # shape (ndim,basis,basis)
d2hdk = optional_second_derivatives      # shape (ndim,ndim,basis,basis)
```

The derivative calculation is not reimplemented here: all Berry connection and
generalized-derivative work is delegated to `analysis.optical_response.gauge`,
which follows the WannierBerri Hamiltonian-gauge convention. This
`analysis.shift_current` package is a compatibility path that re-exports the
common optical-response API.

```python
from analysis.optical_response import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    parse_component,
    positive_transition_terms,
    precompute_shift_current_tensors,
)

tensors = precompute_shift_current_tensors(
    energies,
    evecs,
    dhdk,
    d2hdk=d2hdk,
    denominator_cutoff_ev=1e-10,
)
transitions, weights = positive_transition_terms(
    tensors,
    parse_component("x;yy"),
    convention=JOYA_EQ7_GEOMETRIC_CONVENTION,
)
```

For large finite-cutoff bases, avoid constructing the full generalized-derivative
tensor and use `component_kernel_from_gauge_pair(...)` or
`component_transition_weight_from_gauge_pair(...)` for selected transitions.
This still sums virtual/intermediate bands over the full supplied basis.

Current lightweight system/workflow adapters:

- TDBG/Joya: `mean_field.systems.tdbg.shift_current` supplies analytic lab-frame
  `dH/dk`, zero `d2H/dk2`, one-k gauge data, full tiny-k tensors,
  selected-pair kernels, Gamma-centered cell helpers, and K- mirror-x
  tensor-sign helpers.

Retired Chaudhary b0 TBG and hTG legacy shift-current wrappers should stay in
ignored local reports/internal workspaces or git history. New system-specific
response work should connect Hamiltonians and derivatives directly to this
common API instead of restoring those paper-audit surfaces.

## Named conventions

- `JOYA_EQ7_GEOMETRIC_CONVENTION`: ordered optical product (`none`), geometric
  sign `+1`, unnormalized optical Lorentzian. This matches the Joya 2025
  Eq.(7) point audit before omitted global conductivity prefactors.
- `WANNIERBERRI_INTERNAL_IMN_CONVENTION`: symmetrized optical product (`sum`) and
  geometric sign `-1`, matching WannierBerri `ShiftCurrentFormula` internal Imn.
  For same-polarization components this encodes the audited relation
  `Imn = -2 * ordered_pair_kernel`.
- `HTG_LEGACY_CONVENTION`: old hTG workspace convention (`sum`, sign `+1`,
  normalized Lorentzian).

## Boundaries

- Momentum units are inherited from the system adapter's `dH/dk`; callers must
  document them before applying unit prefactors.
- `sc_eta` / principal-value regularization is separate from the optical
  Lorentzian broadening.
- Final paper sign, spin degeneracy, SI prefactor, colorbar normalization, and
  panel layout remain workflow/report-layer choices, not generic formula code.

## Workflow boundary

The common module owns the response math (`JOYA_EQ7_GEOMETRIC_CONVENTION`,
selected-pair/full-virtual-band kernels, Fermi-window and Lorentzian heatmap
accumulation).  System adapters, such as `mean_field.systems.tdbg.shift_current`,
own system data and coordinate conventions: analytic lab-frame `dH/dk`, optional
`d2H/dk2`, reciprocal-cell shifts, valley/mirror conventions, and paper-specific
labels.

Paper scans, mesh choices, plotting, Slurm orchestration, and saved-output
evidence are workflow/report-layer concerns.  They should import this common
module and the relevant system adapter rather than reviving retired
`analysis.shift_current_htg` / `analysis.shift_current_tbg` workspaces.
