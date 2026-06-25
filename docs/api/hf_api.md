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

Supported façade names in the tracked core profile currently include `htg`, `rlg_hbn`, `tbg`, `tdbg`, and `tmbg`. ATMG and HTQG are archived optional systems under `local_archive/optional_features/`.

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
- HTG primitive-cell HF can be dispatched with `run_hf(model, cfg, htg_config=HTGRunHFConfig(...))`.  The public `HFConfig` must match the explicit HTG config for filling, square mesh, iteration limit, precision, dielectric/gate scalars, `interaction_scheme="average"`, `coulomb_kernel="2d_gate"`, and `density_convention="stored_delta"`.  The projected window comes from `HTGRunHFConfig.projected_band_count`; generic active-window inference is intentionally not implemented.  The former HTG folded-supercell/fractional-filling adapter is archived under `local_archive/optional_features/htg_supercell_20260625/` and is not part of the current tracked core profile.
- RnG/hBN HF can be dispatched with `run_hf(model, cfg, rlg_hbn_config=RLGhBNRunHFConfig(...))`.  The public `HFConfig` must match the explicit RnG/hBN config for filling, square mesh, iteration limit, precision, interaction scheme, dielectric/gate scalars, Coulomb-kernel family, and `density_convention="stored_delta"`.  Screening, active-window, valley, and projection options remain explicit system-owned inputs and are not inferred from generic `HFConfig` fields.
- TBG zero-field restricted HF can be dispatched with `run_hf(model, cfg, tbg_zero_field_config=TBGZeroFieldRunHFConfig(grid_solution=...))`.  The matching `BMSolution` is required because the canonical basis needs the exact B0 grid and BM micro-wavefunctions used by SCF.  `HFConfig.mesh` is the B0 grid point count `(lk+1, lk+1)`, `HFConfig.dsc_nm` records the resolved screening length used by the overlap kernel, and no `HFConfig -> BMSolution` inference is performed.
- TMBG Polshyn-Wang projected HF can be dispatched with `run_hf(model, cfg, tmbg_polshyn_config=PolshynRunHFConfig(...))`.  The projected indices, target primitive band, doubled-cell mesh, interaction shifts, initialization policy, seed, interaction scalars, and optional `PolshynH0SubtractionConfig(mode="active-reference" | "minus-full-p0")` remain explicit Polshyn config inputs; generic `HFConfig` fields are checked for consistency rather than translated into hidden paper-window choices.
- Systems without a safe config-to-run adapter still fail explicitly with `Unified run_hf is frozen at the API level, but this model has no run_hf(config) adapter yet` until a system-owned config runner is added.

Existing paper runners remain valid internal workflows, but new public code should target this API.

## Registered HF boundary adapters

Stable HF boundary adapters are discoverable from `mean_field.api.hf`; entries with `supports_run_hf_config=False` remain post-run-only and must not be treated as `run_hf(config)` support:

```python
from mean_field.api.hf import list_hf_adapters, resolve_hf_adapter

for info in list_hf_adapters(adapter_type="canonical_hf_run_result"):
    print(info.name, info.system_name, info.supports_run_hf_config, info.run_hf_config_reason)

adapter = resolve_hf_adapter("htg_hf_run_to_hf_run_result")
```

Registered boundaries currently cover:

- TDBG projected HF: `tdbg_projected_hf_result_to_hf_run_result(...)` for an existing `TDBGProjectedHFResult`.
- HTG primitive HF: `htg_hf_run_to_hf_run_result(...)` and `htg_hf_run_to_hf_result(...)` for an existing primitive-cell run; `htg_explicit_primitive_run_hf` records the explicit `HTGRunHFConfig` run adapter.
- TBG zero-field HF: `tbg_zero_field_hf_run_to_hf_run_result(..., grid_solution=...)`, `tbg_zero_field_hf_run_to_hf_result(..., grid_solution=...)`, or `b0_hf_benchmark_run_to_hf_run_result(...)`; `tbg_zero_field_explicit_run_hf` records the explicit `TBGZeroFieldRunHFConfig(grid_solution=...)` run adapter.  The grid solution is required and is not fabricated.
- RnG/hBN HF: `rlg_hbn_hf_run_to_hf_run_result(...)` and `rlg_hbn_hf_run_to_hf_result(...)` for an existing RnG/hBN run; `rlg_hbn_explicit_run_hf` records the explicit `RLGhBNRunHFConfig` run adapter.
- TMBG Polshyn-Wang: `polshyn_wang_hf_bundle_to_hf_run_result(basis, state, info, ...)` for an explicit saved bundle; `tmbg_polshyn_explicit_run_hf` records the explicit `PolshynRunHFConfig` run adapter.  `PolshynH0SubtractionConfig` exposes only the reviewed Polshyn-owned one-body conventions `none`, `active-reference`, and `minus-full-p0`; signs are fixed by mode and no Fig. S1 window/plot/Slurm workflow is inferred.

Post-run adapters are I/O/public-surface bridges only: they wrap already-computed system artifacts and preserve system density conventions in the canonical contract.  Registered `run_hf` adapters call existing system-owned runners from explicit system configs.  Neither adapter class infers missing configs, touches cRPA, or changes physics.

## HFResult

`HFResult` records:

- `model`: serializable `ModelRecord`;
- `config`: `HFConfig`;
- `state`: the raw system result/state, preserved for backward compatibility;
- `observables`: scalar/order-parameter metadata;
- `artifacts`: optional `ArtifactManifest`;
- `canonical_run_result`: optional canonical `mean_field.core.contracts.HFRunResult` I/O view.

For TDBG explicit projected HF, `state` remains the raw `TDBGProjectedHFResult` and `canonical_run_result` is populated by `tdbg_projected_hf_result_to_hf_run_result(...)`.  The canonical view maps the raw projector density to `DensityState(density_delta=P-R)`, records `ProjectedBasis`, `HamiltonianParts`, iteration history, metadata, and marks `supports_crpa=False`.  It does not re-run HF, does not reconstruct final active eigenvectors, and does not claim cRPA compatibility.

For HTG primitive public `run_hf` calls, `HFResult.state` remains the raw HTG run object and `canonical_run_result` is populated by the existing HTG post-run adapter.  This adapter does not infer missing physics: the caller must pass `HTGRunHFConfig`, and the public `HFConfig` is validated as a matching contract rather than translated into hidden runner options.

For RnG/hBN public `run_hf` calls, `HFResult.state` remains the raw `RLGhBNHartreeFockRun` and `canonical_run_result` is populated by `rlg_hbn_hf_run_to_hf_run_result(...)`.  The adapter requires explicit `RLGhBNRunHFConfig` because screening/projection/valley choices are system-owned physics inputs.  It preserves stored `P-R` density semantics, records collapsed Hamiltonian parts, and marks `supports_crpa=False`.

For TBG zero-field public `run_hf` calls, `HFResult.state` remains the raw `RestrictedHartreeFockRun` and `canonical_run_result` is populated by `tbg_zero_field_hf_run_to_hf_run_result(...)` using the explicit matching `BMSolution`.  The adapter does not construct the BM grid from generic `HFConfig`; callers must provide `TBGZeroFieldRunHFConfig(grid_solution=...)`, and the public config is validated as a matching contract.

For TMBG Polshyn-Wang public `run_hf` calls, `HFResult.state` remains the explicit `PolshynWangHFState` and `canonical_run_result` is populated by `polshyn_wang_hf_bundle_to_hf_run_result(...)`.  The adapter requires `PolshynRunHFConfig` because the projected indices, target primitive band, doubled-cell embedding, shifts, initialization policy, h0-subtraction convention, and interaction values are system-owned physics inputs and must not be inferred from a generic `HFConfig`.  When `PolshynH0SubtractionConfig` is enabled, metadata records the mode, fixed application sign, P0 reference, and q=0 policy.

`HFResult.save(...)` writes the normal public sidecars plus `canonical_hf_run_result.json` when `canonical_run_result` is present.  The default `canonical_payload="metadata_only"` sidecar is metadata/shape-only: it records contract class names, array shapes, the explicit `density_delta_definition="P-R"`, density/reference schemes, Hamiltonian component metadata, iteration count, and manifest keys, but does not serialize large density/Hamiltonian/wavefunction arrays or write `canonical_hf_arrays.npz`.  Adapter-provided metadata and the last iteration row are sanitized before JSON output: dense arrays are summarized, non-finite values are rejected, and public JSON is written without `NaN`/`Infinity` tokens.  `load_result(...)` reads this sidecar into `ResultDirectory.canonical_hf_run_result` when the manifest references it, and rejects missing or path-escaping referenced sidecars.  Dense canonical arrays are explicit opt-in only via `canonical_payload="arrays"`, which writes `canonical_hf_arrays.npz` plus `canonical_hf_arrays.schema.json` and records those keys in the manifest.

`HFResult.reconstruct_micro_wavefunctions()` is part of the public contract because topology, shift current, Fubini-Study metric, and TDHF often need microscopic wavefunctions rather than only active-subspace eigenvectors.  `mean_field.core.hf.reconstruction` provides the system-independent array contraction helper from a canonical projected microscopic basis and explicit active HF eigenvectors.  TDBG, HTG primitive, and RLG-hBN expose system-owned reconstruction adapters with explicit ordering/selection/guard metadata.  TMBG/Polshyn exposes `mean_field.systems.tmbg.polshyn_supercell.reconstruct_polshyn_wang_hf_micro_wavefunctions` as a public flat-k diagnostic API only; returned bundles remain topology-ineligible (`topology_eligible=False`) and must be rejected by common topology bundle guards.  Polshyn topology must use `mean_field.systems.tmbg.topology.compute_polshyn_projected_hf_topology`, which owns doubled-cell mesh reshape/sewing before delegating to common FHS.  These adapters are software/API plumbing only; paper/topology claims still require target-specific Slurm validation.
