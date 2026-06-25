# Bands API contract

The stable band entrypoint is:

```python
from mean_field.api import make_model, compute_bands

model = make_model("htg", theta_deg=1.8, n_shells=3)
bands = compute_bands(model, n_bands=4, points_per_segment=80)
```

`compute_bands()` delegates to existing system methods such as `standard_kpath`, `bands_along_path`, and `bands_on_grid`, then normalizes the result into a `BandBundle`.

## BandBundle

`BandBundle` contains:

- `k`: path or grid k-points;
- `energies`: sampled band energies;
- `eigenvectors`: optional eigenvectors;
- `basis_metadata`: labels, node indices, band indices, system metadata, and `component_groups` records when the model declares them;
- `convention`: public units/axis/gauge metadata;
- `source`: `path`, `grid`, or `raw`.

## Scope

This is a non-interacting/model-band façade.  HF quasiparticle bands should eventually be exposed through `HFResult.quasiparticle_bands(path)`, because they may need saved HF state, projected basis metadata, and microscopic wavefunction reconstruction.

## Compatibility rule

System-specific special cases such as HTG Chern-basis labels or archived optional mapped-spectrum adapters may remain in `basis_metadata` until a second tracked system needs the same public field.  Do not force one-system paper outputs into generic fields prematurely.
