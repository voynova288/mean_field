# Artifact API contract

The public artifact helper is:

```python
from mean_field.api import ArtifactManifest, ConventionBundle, load_result, update_artifact_manifest, write_contract_artifacts
from mean_field.core.io import write_npz_artifact
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

Large numerical arrays should be referenced from `manifest.json`, not inlined in JSON.  Public JSON sidecars are written as strict JSON: `NaN`/`Infinity` tokens are rejected rather than emitted.  Use `write_npz_artifact(...)` for new dense-array payloads that need atomic writes; it rejects object-dtype arrays so downstream readers can keep `np.load(..., allow_pickle=False)`.  Recommended names are:

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

Legacy outputs may not contain the full schema.  `load_result(path)` is tolerant when an optional sidecar key is absent and returns `None` for `ResultDirectory.canonical_hf_run_result` in that case.  If a manifest explicitly references `canonical_hf_run_result`, the sidecar must be a relative path inside the result root, must exist, and must match the canonical HF run sidecar schema; missing files, absolute paths, `..` escapes, malformed schema, or non-standard JSON numeric tokens are rejected.  For new contract outputs, `load_result(path)` reads `model.json`, JSON-compatible `config.yaml`, `conventions.json`, `environment.json`, `validation.json`, and `observables.json`.  New workflows should write the full schema before being treated as public/stable.
