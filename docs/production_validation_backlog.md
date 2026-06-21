# Production validation backlog runbook

This runbook is for the post-cleanup validation backlog.  It distinguishes software/API readiness from numerical or paper-level validation.  Non-TDHF/non-cRPA validation may proceed after local self-checks; TDHF and cRPA work still requires a separate explicit request.

## Common rules

- Run heavy HF, topology, TDHF, response, or eigensolver workloads on a compute node or via Slurm, not on login nodes.
- Keep runs self-describing: save config, exact command, git commit, environment variables, stdout/stderr, manifest, and summary JSON/MD.
- A passing unit test is software validation only.  Paper-level validation requires saved artifacts and a quantitative comparison to the target panel/quantity.
- TDHF and cRPA validation/code paths remain out of scope unless explicitly requested; do not modify `src/mean_field/crpa/*` or TDHF modules while advancing the other gates.
- Non-TDHF/non-cRPA validation and cleanup gates may proceed after local self-checks of command, output path, and expected runtime.

Recommended lightweight gate before production jobs:

```bash
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MEAN_FIELD_HF_DISABLE_NUMBA=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m compileall -q src scripts
pytest -q $(git ls-files tests)
```

## 1. Historical canonical HF sidecars

Goal: produce staged canonical `HFRunResult` sidecars for historical archives that contain complete raw run data.

Status after the current cleanup:

- RLG/hBN: eligible when `hf_ground_state.npz` and compatible cache manifests are present.
- TDBG: eligible only when full `hf_state.npz`, `projected_basis.npz`, `state_labels.json`, `projected_hf_summary.json`, `config.json`, and model metadata satisfy the loader contract.
- HTG primitive/supercell: eligible only when full raw state/basis archives and exact model/interaction metadata satisfy the loader contract.
- Summary-only archives remain blocked; do not fabricate micro-wavefunctions, model objects, or run history.

Dry-run inventory example (roots are positional; no historical mutation):

```bash
python -m mean_field.devtools.backfill_canonical_hf_sidecars \
  results \
  --report-json tmp/backfill_inventory.json \
  --report-md tmp/backfill_inventory.md
```

Fast metadata-only dry-run when you do not want NPZ archive scanning:

```bash
python -m mean_field.devtools.backfill_canonical_hf_sidecars \
  results \
  --no-archives \
  --report-json tmp/backfill_inventory_no_archives.json \
  --report-md tmp/backfill_inventory_no_archives.md
```

Staged write example (no historical mutation):

```bash
python -m mean_field.devtools.backfill_canonical_hf_sidecars \
  results \
  --write \
  --target-root tmp/staged_canonical_backfill \
  --allow-target-root tmp \
  --report-json tmp/backfill_write_inventory.json \
  --report-md tmp/backfill_write_inventory.md
```

Current staged sidecars note:

- `/data/home/ziyuzhu/tmp/mean_field_canonical_backfill_staged_229e06e_20260621_180741` contains 48 written RLG/hBN sidecars.
- A self-check found all 48 written entries are `rlg_hbn_archive` and use the existing `mean_field.systems.RnG_hBN.tdhf.load_rlg_hbn_tdhf_run_from_archive` reconstruction path.
- Because TDHF remains out of scope, these stay staged and are not applied to historical `results/` in the current continuation.

Acceptance:

- `historical_results_mutated` is `false`.
- staged sidecars validate with `mean_field.api.load_result(...)` metadata-only loading.
- manifest patches are reviewed separately before any application to `results/`.

## 2. TMBG Polshyn fresh `run_hf(config...)`

Goal: run explicit Polshyn-Wang projected HF through the public API without inferring paper windows from generic `HFConfig`.

Software entrypoint:

```python
from mean_field.api import HFConfig, run_hf
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
from mean_field.systems.tmbg.polshyn_supercell import PolshynRunHFConfig

model = TMBGModel.from_config(theta_deg=1.25, n_shells=0, params=TMBGParameters.minimal())
polshyn = PolshynRunHFConfig(
    mesh_size=1,
    projected_indices=(2,),
    target_band_index=2,
    shifts=(),
    v0=0.0,
    max_iter=1,
)
config = HFConfig(
    filling=3.5,
    mesh=(1, 1),
    active_band_indices=(2,),
    density_convention="stored_delta",
    max_iter=1,
    precision=polshyn.precision,
)
result = run_hf(model, config, tmbg_polshyn_config=polshyn)
```

Latest software preflight (not paper-level validation):

```text
commit: 92f4e98
output: /data/home/ziyuzhu/tmp/mean_field_validation_tmbg_polshyn_92f4e98_20260622_003023/summary.json
result_model: tmbg_polshyn
has_canonical_run_result: true
best_seed: 5
workflow metadata: tmbg.polshyn_wang.explicit_config
```

Latest software-gate result on `test001`:

```text
commit: 8f86abe
command: pytest -q tests/test_tmbg_polshyn_hf_readiness.py
result: 10 passed
coverage added: metadata-only `HFResult.save(...)` writes `canonical_hf_run_result.json`, remains loadable via `mean_field.api.load_result(...)`, and does not write `canonical_hf_arrays.npz`.
```

Production-scale validation still needs explicit physics choices:

- target primitive band and topology evidence, e.g. C=2 band;
- projected window and remote-band reference convention;
- interaction shell/cutoff and screening values;
- candidate initial states (`bm_wang`, `cdw`, and any physically relevant competitors);
- convergence and energy comparison across candidates.

Acceptance:

- canonical sidecar saves with `canonical_payload="metadata_only"` cheaply;
- dense array payload is opt-in;
- exact SCF-grid bands/order parameters are saved before any topology or paper overlay claim.

## 3. HTG fractional filling

Goal: validate primitive/supercell fractional-filling HF using explicit supercell adapters, not fractional occupations in a primitive cell.

Preconditions:

- use `HTGSupercellRunHFConfig`/supercell path for fractional fillings;
- choose the minimal physically intended supercell denominator by default (`Nc=2` for half, `Nc=3` for thirds);
- record primitive filling, reference diagonal, supercell matrix, occupation counts, and exact SCF grid.

Software gate:

```bash
pytest -q tests/test_htg_supercell.py tests/test_htg_supercell_contract_adapter.py
```

Latest software-gate result on `test001`:

```text
commit: 9d045c5
command: pytest -q tests/test_htg_supercell.py tests/test_htg_supercell_contract_adapter.py
result: 10 passed
```

Additional public-HF gate:

```text
commit: 4ce02cb
command: pytest -q tests/test_api_hf_adapters.py
result: 16 passed
coverage added: explicit HTG supercell `run_hf(...)` metadata-only save/load writes `canonical_hf_run_result.json`, remains loadable via `mean_field.api.load_result(...)`, and does not write `canonical_hf_arrays.npz`.
```

This is software readiness only, not a converged fractional-filling production run.

Production acceptance:

- self-consistent run converges for relevant candidate states;
- exact SCF-grid path/bands saved, not nearest-grid/flattened-index plots;
- energy components and filling from density match the intended convention;
- if comparing to a paper panel, axes/path/energy reference and target bands are documented.

## 4. RnG/hBN finite-q TDHF

Goal: validate finite-q intraflavor/shortcut TDHF assembly and lowest-branch stability masks before production maps.

Software gates:

```bash
pytest -q tests/test_rlg_hbn_tdhf_adapter.py tests/test_rlg_hbn_hf_contract_adapter.py tests/test_api_imports.py
```

Production prerequisites:

- start from a saved canonical-ready HF archive with compatible RLG/hBN basis/overlap cache manifests;
- record q-shift sector, channel, pair counts, structure residuals, and raw TDHF eigenvalues;
- mask complex/unstable sectors explicitly instead of replacing them with a higher positive branch.

Acceptance:

- matrix-structure residuals are within tolerance;
- raw eigenvalue pairing and metric signs are reported;
- finite-q target maps include a manifest linking HF source archive, q grid, channel, and stability policy.

## 5. HTQG projected HF / Fig. 1

Goal: validate HTQG projected-HF and Fig. 1 style band outputs without confusing software readiness with paper reproduction.

Preconditions:

- identify the target Hamiltonian parameters, projected band/subspace, reference density, and path convention;
- use common HF/problem machinery where possible and keep HTQG-specific physics in `systems/htqg`;
- save exact SCF-grid spectra and path metadata.

Software gates:

```bash
pytest -q tests/test_htqg_model.py tests/test_api_imports.py
```

Latest software-gate result on `test001`:

```text
commit: 9d045c5
command: pytest -q tests/test_htqg_model.py tests/test_api_imports.py
result: 14 passed
```

Current public-HF status:

```text
HTQG has no registered HF adapter yet.
`run_hf(HTQGModel, HFConfig(...))` is tested to fail explicitly with "no run_hf(config) adapter yet" instead of inferring projected-HF physics.
```

This is band/model API readiness only; projected-HF production validation still needs the explicit HTQG HF adapter or workflow described below.

Production acceptance:

- converged HF state archive with density, hamiltonian, h0, energies, k grid, convergence history, and provenance;
- quantitative comparison to target Fig. 1 axes/energy reference/band features;
- unresolved parameter/path ambiguity explicitly reported, not hidden in plotting.

## 6. TMBG Polshyn production validation

Goal: move from explicit API smoke to paper-level Polshyn-Wang validation.

Required evidence before claiming reproduction:

- target band selected by topology/config evidence, not by visual position alone;
- filling computed from occupation counts minus reference (`nu=7/2` convention);
- convergence across relevant initial states;
- exact SCF-grid bands/order parameters saved;
- optional Chern/topology postprocessing only after the target band/subspace is explicitly selected.

Production job template should be created in ignored scratch or a reviewed durable workflow, then submitted only after explicit approval.
