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

Current run coverage:

- TDBG projected HF can be dispatched with `run_hf(model, cfg, tdbg_config=TDBGProjectedHFConfig(...), init_mode=...)`.  The public `HFConfig` must match the explicit TDBG config for mesh, filling, iteration limit, and precision, and must set `density_convention="projector"`.  Generic `HFConfig -> TDBGProjectedHFConfig` inference is intentionally not implemented.
- Other systems still fail explicitly with `Unified run_hf is frozen at the API level, but this model has no run_hf(config) adapter yet` until a system-owned config runner is added.

Existing paper runners remain valid internal workflows, but new public code should target this API.

## Registered post-run HF adapters

Stable post-run canonical adapters are discoverable from `mean_field.api.hf` without implying `run_hf(config)` support:

```python
from mean_field.api.hf import list_hf_adapters, resolve_hf_adapter

for info in list_hf_adapters(adapter_type="canonical_hf_run_result"):
    print(info.name, info.system_name, info.supports_run_hf_config)

adapter = resolve_hf_adapter("htg_supercell_hf_run_to_hf_result")
```

Registered conversion surfaces currently cover:

- TDBG projected HF: `tdbg_projected_hf_result_to_hf_run_result(...)` for an existing `TDBGProjectedHFResult`.
- HTG primitive HF: `htg_hf_run_to_hf_run_result(...)` and `htg_hf_run_to_hf_result(...)` for an existing primitive-cell run.
- HTG folded-supercell HF: `htg_supercell_hf_run_to_hf_run_result(...)` and `htg_supercell_hf_run_to_hf_result(...)` for an existing supercell run.
- TBG zero-field HF: `tbg_zero_field_hf_run_to_hf_run_result(..., grid_solution=...)` or `b0_hf_benchmark_run_to_hf_run_result(...)`; the grid solution is required and is not fabricated.
- RnG/hBN HF: `rlg_hbn_hf_run_to_hf_run_result(...)` for an existing RnG/hBN run.
- TMBG Polshyn-Wang bundle: `polshyn_wang_hf_bundle_to_hf_run_result(basis, state, info, ...)` for an explicit saved bundle.

These adapters are I/O/public-surface bridges only: they wrap already-computed system artifacts, preserve system density conventions in the canonical contract, and do not rerun SCF, infer missing configs, touch cRPA, or change physics.

## HFResult

`HFResult` records:

- `model`: serializable `ModelRecord`;
- `config`: `HFConfig`;
- `state`: the raw system result/state, preserved for backward compatibility;
- `observables`: scalar/order-parameter metadata;
- `artifacts`: optional `ArtifactManifest`;
- `canonical_run_result`: optional canonical `mean_field.core.contracts.HFRunResult` I/O view.

For TDBG explicit projected HF, `state` remains the raw `TDBGProjectedHFResult` and `canonical_run_result` is populated by `tdbg_projected_hf_result_to_hf_run_result(...)`.  The canonical view maps the raw projector density to `DensityState(density_delta=P-R)`, records `ProjectedBasis`, `HamiltonianParts`, iteration history, metadata, and marks `supports_crpa=False`.  It does not re-run HF, does not reconstruct final active eigenvectors, and does not claim cRPA compatibility.

For HTG folded-supercell HF, `htg_supercell_hf_run_to_hf_run_result(...)` wraps an existing `HTGSupercellHartreeFockRun` as a canonical `HFRunResult`.  HTG supercell densities are already stored as `P-R`, so the adapter uses `density_state_from_delta(...)`, preserves `(n_state,n_state,n_k)` arrays, records folded-basis metadata, and uses collapsed Hamiltonian parts (`fixed=total-h0`, `hartree=fock=0`) unless a future run surface exposes component splits.

`HFResult.save(...)` writes the normal public sidecars plus `canonical_hf_run_result.json` when `canonical_run_result` is present.  This sidecar is metadata/shape-only: it records contract class names, array shapes, the explicit `density_delta_definition="P-R"`, density/reference schemes, Hamiltonian component metadata, iteration count, and manifest keys, but does not serialize large density/Hamiltonian/wavefunction arrays.  Adapter-provided metadata and the last iteration row are sanitized before JSON output: dense arrays are summarized, non-finite values are rejected, and public JSON is written without `NaN`/`Infinity` tokens.  `load_result(...)` reads this sidecar into `ResultDirectory.canonical_hf_run_result` when the manifest references it, and rejects missing or path-escaping referenced sidecars.

`HFResult.reconstruct_micro_wavefunctions()` is part of the public contract because topology, shift current, Fubini-Study metric, and TDHF often need microscopic wavefunctions rather than only active-subspace eigenvectors.  System adapters should implement it before claiming those downstream workflows are fully supported.
