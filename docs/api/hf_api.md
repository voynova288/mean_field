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

`run_hf(model, cfg)` is intentionally strict.  It requires a system adapter exposing a `run_hf(config, **kwargs)` hook.  Until each system has that hook, existing paper runners remain valid internal workflows, but new public code should target this API.

## HFResult

`HFResult` records:

- `model`: serializable `ModelRecord`;
- `config`: `HFConfig`;
- `state`: system or core HF state;
- `observables`: scalar/order-parameter metadata;
- `artifacts`: optional `ArtifactManifest`.

`HFResult.reconstruct_micro_wavefunctions()` is part of the public contract because topology, shift current, Fubini-Study metric, and TDHF often need microscopic wavefunctions rather than only active-subspace eigenvectors.  System adapters should implement it before claiming those downstream workflows are fully supported.
