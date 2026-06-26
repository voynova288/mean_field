# `analysis.topology`

This directory intentionally contains only the common FHS/Wilson-link topology
pipeline:

```text
FHSState -> generic boundary sewing -> FHS links -> Berry plaquette flux -> Chern
```

The public computation entry is `compute_lattice_topology(state)`.  Systems
should only build `FHSState` objects with band/flavor metadata and, when needed,
`BlockSewingSpec` basis metadata for generic seam sewing.

Out of scope here and in system modules: projector-QGT finite differences,
Fubini-Study metric, paper-target registries, saved-result validators,
system-wrapper factories, system-private plaquette loops, system-private Chern
integration, and system-private topology sewing transforms when a
`BlockSewingSpec` can describe the basis.
