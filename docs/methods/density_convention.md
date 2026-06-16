# Density convention method note

The canonical implementation is `mean_field.core.hf.density`.

## Definitions

Let `P_ab(k) = <c_a^† c_b>` be the stored-orientation physical occupied-state projector.

Supported public conventions:

- `projector`: the physical projector `P`.
- `stored_delta`: `D = P - P_ref`.
- `half_shifted`: the special case `D = P - 1/2 I`.

All public HF density arrays use `axis_order = "abk"`, shape `(n_state, n_state, n_k)`.

## Orientation

The stored projector orientation follows archive/HF density convention `P_ab = <c_a^† c_b>`.  Some wavefunction contractions need ket-space matrices.  Use:

```python
DensityBundle(...).as_projector(orientation="ket")
```

or:

```python
stored_density_to_projector(..., convention="ket")
```

Do not open-code the transpose in system adapters unless the adapter is deliberately translating a system-local archive format at the boundary.

## Reference density

A stored delta is not meaningful without its reference density.  Public code must either:

1. load the explicit `reference_density` array;
2. use an explicit `ReferenceDensity.average(nt, nk, value=0.5)` policy for historical half-filled archives; or
3. declare `reference_policy="none"` for true projector-like arrays stored as deltas from zero.

Use `reference_policy="require"` when a missing reference would change the physics conclusion.

## Migration rule

Existing helpers such as `stored_density_to_projector`, `conventional_projector_to_stored`, and `stored_projector_to_conventional` may remain as compatibility names, but their implementation should delegate to `mean_field.core.hf.density`.
