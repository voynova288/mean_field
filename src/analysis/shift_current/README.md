# Generic shift-current module

This package is the common formula API for shift-current calculations. New
optical workflows should usually enter through `analysis.optical` and choose
`kind="shift_current"`; import this package directly only when shift-current
specific controls are needed.

A physical system should provide only the k-point adapter data:

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

## Exact-degenerate U(N) contract

The ordinary pair API is band resolved. For nondegenerate bands its labeled
kernel is U(1)-invariant. Inside an exactly degenerate manifold, however, a
label such as `n -> m` is a frame choice and its individual weight is not
required to survive a U(N) rotation.

For a complete exactly degenerate initial group `I` and final group `F`, use

```python
transition = exact_degenerate_group_transition_weight(
    tensors,
    initial_group=occupied_doublet,
    final_group=empty_doublet,
    component="x;yy",
    convention=WANNIERBERRI_INTERNAL_IMN_CONVENTION,
)
```

The underlying ordered product is

```text
Tr_I[(A^c_;a)_IF A^b_FI],
```

implemented by `component_group_trace_amplitude(...)`. Under independent
rotations in `I` and `F`, the two rectangular blocks transform oppositely and
the product changes only by similarity inside `I`; its trace is invariant.
This is the matrix-product interpretation of
`WannierBerri ShiftCurrentFormula.trace_ln`, which sums its complete pair block.

`exact_degenerate_group_transition_weight(...)` deliberately requires each
group to be the complete maximal eigenspace, bitwise identical energies and
occupations inside it, and scalar intragroup first-derivative blocks (up to an
explicit absolute diagnostic tolerance). It therefore rejects a partial
multiplet and any point with first-order intragroup splitting; persistence of
the exact degeneracy beyond that local condition remains the caller's
responsibility. It does not average a near-degenerate cluster and does not
authorize arbitrary rotations of nondegenerate bands. A
near-degenerate cluster spectrum needs an additional matrix-valued
spectral/occupation contract and is not implemented by this milestone.

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

## WannierBerri positive-frequency conductivity conversion

`WANNIERBERRI_INTERNAL_IMN_CONVENTION` fixes only the local geometric tensor.
For a positive-energy occupied-to-empty list that contains each transition once,
use the separate named conductivity conversion:

```python
from analysis.shift_current import (
    wannierberri_positive_transition_conductivity_from_imn,
)

# real_imn_integral includes (f_initial-f_final) * Imn[initial,final]
sigma = wannierberri_positive_transition_conductivity_from_imn(real_imn_integral)
```

The fixed WannierBerri dynamic calculator uses `+pi e^2/(4 hbar)`, evaluates
both ordered band-group pairs, uses `f(E2)-f(E1)`, and includes both delta
orientations. Since its implemented Hermitian tensors give
`Imn[final,initial] = -Imn[initial,final]`, the two ordered pairs reduce at
positive frequency to

```text
[-pi e^2/(2 hbar)]
* (f_initial-f_final)
* Imn[initial,final]
* delta(E_final-E_initial-hbar*omega).
```

Accordingly, `WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2` is
**negative one half** of the historical unsigned
`SHIFT_CURRENT_PREFAC_UA_NM_PER_V2`. With a finite-width Lorentzian, include the
fixed calculator's resonant and antiresonant broadened orientations before
applying the helper if exact finite-width parity is required; a resonant-only
positive-transition spectrum is the positive-frequency delta-limit convention.

The old `SHIFT_CURRENT_PREFAC_UA_NM_PER_V2` remains exported for callers that
explicitly choose the Joya/legacy conversion. It is an unsigned historical
scale, not a complete WannierBerri convention. The low-level
`conductivity_from_integral` and `spectra_from_transition_table` APIs require
explicit prefactor and phase arguments; they never infer a conversion from the
kernel alone.

The response coefficient follows the field convention of Ibañez-Azpiroz et al.,
PRB 97, 245143 (2018), Eqs. (6)--(8):
`E(t)=E(omega)e^{-i omega t}+E(-omega)e^{+i omega t}` with
`E(-omega)=E(omega)*`, and
`j^a=2 sigma^{abc}(0;omega,-omega) Re[E_b(omega)E_c(-omega)]`.
The helper returns that `sigma(0;omega,-omega)` convention; it does not convert
to a coefficient defined using a real-field peak amplitude.

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
