# Mean_Field cleanup plan: next structural pass

This plan tracks the next cleanup pass after the large-file/facade split work through `origin/main` `229e06e`.

## Hard constraints

- Preserve system-specific physics. In particular, keep RnG/hBN layer-dependent Coulomb, layer-dependent form factors, q=0 internal screening, and average/CN schemes in system code.
- cRPA is no longer part of the tracked package surface for the current cleanup direction; archived cRPA code remains in ignored `local_archive/retired_surface/crpa_untracked_20260622/`.
- Do not modify TDHF code/validation flows without a separate explicit instruction.
- Do not run heavy HF/topology/response numerics on login nodes. Use a compute node or Slurm after checking command/output paths.
- Do not mutate historical `results/` without explicit approval. Canonical backfill staging remains under allowlisted `/data/home/ziyuzhu/tmp` unless separately approved.
- Keep package code/tests/scripts/docs independent of ignored `local_archive/`.

## Completed status

- Phase 0 git hygiene: completed in `ddebb13`.
- Phase 1 common order-parameter module: completed in `ddebb13`.
- Phase 2 optical-response package boundary/facade: completed in `9b374cc`; implementation moved into `analysis.optical_response`; shift-current and gauge derivative implementations split into package-local modules in the current continuation.
- Phase 3 cRPA/HF bridge split: completed in `ef8ba6f`, then cRPA was archived/untracked from git in the current simplification pass because it is not a near-term development target.
- Phase 4 public API registries and workflow extraction: completed across `e6eab50`, `a00803f`, `9b8202a`, and `faa706e`.
- Devtools cleanup follow-up: RLG/hBN retired sidecar/archive helpers moved out of command files; canonical HF backfill scanner was split and later archived/untracked with the rest of `src/mean_field/devtools` because devtools are no longer part of the minimal public git surface.

## Phase 0 — git surface hygiene

1. Keep only minimal smoke/contract tests tracked. Most detailed regression tests are local/internal artifacts and should live in ignored archive/workspaces unless explicitly requested for public git.
2. Ignore local/generated/internal test payloads:
   - `tests/local/`
   - `tests/internal/`
   - `tests/slow/`
   - `tests/data/generated/`
   - generated arrays: `tests/**/*.npz`, `tests/**/*.npy`
3. Ignore local examples/scratch only:
   - `examples/local/`
   - `examples/scratch/`
   - `examples/oneoff/`
4. If an `examples/` tree is introduced, keep only short public API smoke examples and document keep/drop decisions in `examples_inventory.md`.

## Phase 1 — common order-parameter module

Add `src/analysis/order_parameters/` with reusable schema and density/coherence/translation/classification helpers. Route existing system functions through backward-compatible wrappers:

- `systems/tdbg/projected_hf_state.py::_numeric_order_parameters`
- `systems/tdbg/projected_hf_state.py::tdbg_order_parameters`
- `systems/tmbg/_polshyn_wang.py::translation_order_parameters`
- `core/hf/_finite_field_kernel.py::calculate_valley_spin_order_parameters`

Acceptance:

- old/new outputs matched the archived local `tests/test_order_parameters.py` equivalence checks before the broad test suite was untracked;
- system modules keep only label/adaptation wrappers;
- no formulas or thresholds are changed except through explicitly tested delegation.

## Phase 2 — optical response package boundary

Create `src/analysis/optical_response/` and migrate implementation from:

- `src/analysis/response_derivative_gauge.py`
- `src/analysis/shift_current/core.py`

Keep old paths as compatibility shims/re-exports. Keep system adapters thin: they prepare Hamiltonian/eigenpairs/derivatives/convention signs and call the common package.

Acceptance:

- response/shift-current behavior was covered by the archived local regression tests before broad tests were untracked;
- tracked smoke tests keep the public import/API surface alive;
- retired hTG/TBG paper workspaces remain retired.

## Phase 3 — cRPA archive/untrack

Earlier cRPA/HF bridge cleanup split the implementation behind a facade. The current simplification pass retires cRPA from public git because it is not a near-term development target.

Archived local copy:

```text
local_archive/retired_surface/crpa_untracked_20260622/
```

Tracked package policy:

- no `src/mean_field/crpa/` package;
- no `mean_field.api.compute_crpa` facade;
- no cRPA devtool commands in `scripts/mean_field_tools.py`;
- no tracked `src/mean_field/devtools/` package in the minimal public surface;
- cRPA tests/docs remain archived locally, not tracked.

## Phase 4 — public API registries and workflow extraction

- TDHF adapter registry remains in `api/tdhf.py`, initially for RnG/hBN q=0 and finite-q safe adapters.
- Convert `api/models.py` to a model adapter registry to reduce hard-coded imports.
- Move TDBG projected-HF config parsing/save workflow out of `cli.py` into `workflows/tdbg_projected_hf.py`.

Acceptance:

- public API docs updated;
- focused API tests added;
- CLI remains thin and backwards compatible.

## Lower priority

- Bands wrappers remain; topology wrappers/common topology helpers are archived out of the tracked public surface for now.
- Devtools cleanup: tracked devtools are archived locally for now; future durable commands should be reintroduced only through a small reviewed public surface.
