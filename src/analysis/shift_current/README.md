# Generic shift-current module

This package is the common API for shift-current calculations. A physical
system should provide only the k-point adapter data:

```python
energies, evecs = diagonalize(H(k))      # evecs columns are eigenvectors
dhdk = np.stack([dHdkx(k), dHdky(k)])    # shape (ndim,basis,basis)
d2hdk = optional_second_derivatives      # shape (ndim,ndim,basis,basis)
```

The derivative calculation is not reimplemented here: all Berry connection and
generalized-derivative work is delegated to `analysis.response_derivative_gauge`,
which follows the WannierBerri Hamiltonian-gauge convention.

```python
from analysis.shift_current import (
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
- Chaudhary b0 TBG: `mean_field.systems.tbg.chaudhary2021` exposes
  `b0_shift_current_point_data`, `b0_shift_current_tensors_at_k`, and
  `b0_component_kernel_at_k` around the old b0 model.  Chaudhary audit docs
  are archived under `docs/shift_current/tbg/`.
- hTG/Mao: `mean_field.systems.htg.shift_current` preserves the hTG legacy
  response surface as wrappers around this generic API; `mean_field.systems.htg.mao2025`
  owns Mao-specific model parameters, stacking phases, sublattice mass, and
  `dH/dk` validation helpers.  Mao paper figures are still not reproduced.

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

## Joya/TDBG 50x50 evidence and workflow boundary

The common module owns the response math (`JOYA_EQ7_GEOMETRIC_CONVENTION`,
selected-pair/full-virtual-band kernels, Fermi-window and Lorentzian heatmap
accumulation).  The TDBG adapter in `mean_field.systems.tdbg.shift_current`
owns only system data and Joya coordinate conventions: analytic lab-frame
`dH/dk`, zero affine-model `d2H/dk2`, Gamma-centered fractional shifts, and the
K- mirror-x tensor sign.

The remaining Joya 2025 pipeline glue is intentionally script/workflow level:
`tmp/joya2025/run_tdbg_response_eq7_fullc_scout.py` chooses stackings, Delta
sweep, valleys, mesh, transition window, output schema, plotting, and Slurm
orchestration.  It imports the common module and TDBG adapter rather than the
retired `analysis.shift_current_htg` / `analysis.shift_current_tbg` workspaces.

Current saved-output evidence: Slurm job `140728` ran the generic-API `50x50`
Joya scout into `results/TDBG/joya2025_generic_api_50x50_20260609/`; dependent
postcheck job `140759` wrote `postcheck_20260609.{json,md}` with `passes=True`,
`point_count=40000`, `pair_errors_total=0`, `dH_route=analytic`, and
`d2H_route=zero_affine_tdbg_continuum`.  This validates the migrated generic
pipeline against the saved corrected bundle, not final paper-level units/signs.
