# HF API contract

The stable public import path is:

```python
from mean_field.api import make_model, HFConfig, run_hf
```

`mean_field.api` is a façade.  In the first phase it freezes call shapes and delegates to existing system modules; it does not change physical formulas or rewrite solvers.

## Model construction

```python
model = make_model(
    "rlg_hbn",
    layers=5,
    xi=1,
    theta_deg=0.77,
    displacement_mev=64.0,
)
```

Supported façade names currently include `htg`, `rlg_hbn`, `tdbg`, `tmbg`, and `atmg`.

## HFConfig

`HFConfig` records the public method-level contract:

- filling and mesh;
- active band window or explicit active band indices;
- interaction scheme;
- density convention;
- dielectric/gate/kernel knobs;
- iteration controls and seeds.

Not every system adapter implements every option yet.  A system must fail explicitly rather than silently reinterpret unsupported fields.

## run_hf

`run_hf(model, cfg)` is intentionally strict.  It requires a system adapter exposing a `run_hf(config, **kwargs)` hook, or an explicitly documented façade adapter.

Current adapter coverage:

- TDBG projected HF can be dispatched with `run_hf(model, cfg, tdbg_config=TDBGProjectedHFConfig(...), init_mode=...)`.  The public `HFConfig` must match the explicit TDBG config for mesh, filling, iteration limit, and precision, and must set `density_convention="projector"`.  Generic `HFConfig -> TDBGProjectedHFConfig` inference is intentionally not implemented.
- Other systems still fail explicitly until a system-owned adapter is added.

Existing paper runners remain valid internal workflows, but new public code should target this API.

## HFResult

`HFResult` records:

- `model`: serializable `ModelRecord`;
- `config`: `HFConfig`;
- `state`: the raw system result/state, preserved for backward compatibility;
- `observables`: scalar/order-parameter metadata;
- `artifacts`: optional `ArtifactManifest`;
- `canonical_run_result`: optional canonical `mean_field.core.contracts.HFRunResult` I/O view.

For TDBG explicit projected HF, `state` remains the raw `TDBGProjectedHFResult` and `canonical_run_result` is populated by `tdbg_projected_hf_result_to_hf_run_result(...)`.  The canonical view maps the raw projector density to `DensityState(density_delta=P-R)`, records `ProjectedBasis`, `HamiltonianParts`, iteration history, metadata, and marks `supports_crpa=False`.  It does not re-run HF, does not reconstruct final active eigenvectors, and does not claim cRPA compatibility.

`HFResult.save(...)` writes the normal public sidecars plus `canonical_hf_run_result.json` when `canonical_run_result` is present.  This sidecar is metadata/shape-only: it records contract class names, array shapes, density/reference schemes, Hamiltonian component metadata, iteration count, and manifest keys, but does not serialize large density/Hamiltonian/wavefunction arrays.  `load_result(...)` reads this sidecar into `ResultDirectory.canonical_hf_run_result` when the manifest references it.

`HFResult.reconstruct_micro_wavefunctions()` is part of the public contract because topology, shift current, Fubini-Study metric, and TDHF often need microscopic wavefunctions rather than only active-subspace eigenvectors.  System adapters should implement it before claiming those downstream workflows are fully supported.
