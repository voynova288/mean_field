# `analysis.topology`

This directory intentionally contains only the common FHS/Wilson-link topology
pipeline:

```text
FHSState -> generic boundary sewing -> FHS links -> Berry plaquette flux -> Chern
```

The public computation entry is `compute_lattice_topology(state)` and accepts
only `FHSState`. Systems should only build `FHSState` objects with band/flavor metadata and, when needed,
`BlockSewingSpec` basis metadata for generic seam sewing. Physical adapters with
a validated non-Euclidean or k-dependent endpoint-overlap convention may call
`link_variable_from_overlap_matrix(...)`; Berry plaquettes and Chern integration
still remain in this common module. Zero, singular, and non-finite endpoint
overlaps fail closed; no identity regularizer is permitted.

FHS plaquette flux is principal-wrapped to `(-π, π]`. Use
`result.minimum_plaquette_branch_margin` as a caller-thresholded admissibility
diagnostic. For exact direct sums, compare determinant links and plaquette
fluxes modulo `2π`; separately wrapped component and union Chern sums need not
match if a plaquette approaches the branch cut. An admissible conserved-sector
sum can resolve that same-mesh alias, but it does not prove continuum or
basis-cutoff convergence.

Out of scope here and in system modules: projector-QGT finite differences,
Fubini-Study metric, paper-target registries, saved-result validators,
system-wrapper factories, system-private plaquette loops, system-private Chern
integration, and system-private topology sewing transforms when a
`BlockSewingSpec` can describe the basis.
