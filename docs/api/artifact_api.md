# Artifact API contract

The public artifact helper is:

```python
from mean_field.api import ArtifactManifest, ConventionBundle, load_result, update_artifact_manifest, write_contract_artifacts
```

## Required result files

A complete workflow result should contain these files at the result root:

```text
manifest.json
model.json
config.yaml
conventions.json
environment.json
validation.json
observables.json
```

Use `write_contract_artifacts(...)` to write these sidecars without rewriting large numerical arrays.  The helper stores JSON-compatible YAML in `config.yaml` to avoid an additional YAML dependency.

For derived postprocessing outputs that share an existing result root, use `update_artifact_manifest(...)` to add files/metadata to `manifest.json` without overwriting the existing sidecars.

Large numerical arrays should be referenced from `manifest.json`, not inlined in JSON.  Recommended names are:

```text
hf_state.npz
bands_path.npz
plots/
logs/
```

## conventions.json

`ConventionBundle` serializes the minimum convention metadata:

- `energy_unit`
- `length_unit`
- `momentum_unit`
- `density_convention`
- `density_axis_order`
- `hamiltonian_axis_order`
- `wavefunction_axis_order`
- `valley_labels`
- `spin_labels`
- `gauge`

See `docs/conventions.md` for semantics.

## Compatibility

Legacy outputs may not contain the full schema.  `load_result(path)` is tolerant and returns `None` for missing optional files.  New workflows should write the full schema before being treated as public/stable.
