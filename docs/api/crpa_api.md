# cRPA API contract

The stable cRPA façade is:

```python
from mean_field.api import CRPAConfig, compute_crpa
```

`compute_crpa(model_or_solution, config)` freezes the public call shape.  Existing production logic remains in `mean_field.crpa` and system/devtool adapters until each object exposes a uniform `compute_crpa(config)` hook.

## CRPAConfig

`CRPAConfig` records:

- q mesh or q-grid selection;
- dielectric environment (`epsilon_bn`, `ds_angstrom`);
- broadening `eta_mev`;
- occupation mode;
- form-factor convention.

## Density and interaction warning

cRPA must not blindly reinterpret a saved HF density.  Convert through `mean_field.core.hf.density` and record `density_convention` in artifacts.  If a workflow uses a split interaction scheme such as a Zhang-style flat/remote cRPA correction, that scheme must be explicit in `HFConfig.interaction_scheme` and in `conventions.json`/result metadata.

## Current status

The façade currently raises `NotImplementedError` unless the object supplies `compute_crpa(config)`.  This is intentional: API shape is frozen before moving existing chunk/merge code behind it.
