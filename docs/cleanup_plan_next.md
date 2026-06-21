# Mean_Field cleanup plan: next structural pass

This plan tracks the next cleanup pass after the large-file/facade split work through `origin/main` `229e06e`.

## Hard constraints

- Preserve system-specific physics. In particular, keep RnG/hBN layer-dependent Coulomb, layer-dependent form factors, q=0 internal screening, and average/CN schemes in system code.
- Preserve cRPA/HF conventions: `D = P - 1/2 I`, `Delta_HI` uses bare Coulomb, dynamic flat-band self-energy uses cRPA, and ODA delta-H on `delta_D` must not add `+1/2 I`.
- Do not run heavy HF/topology/response numerics on login nodes. Use the test node or Slurm only after explicit authorization.
- Do not mutate historical `results/` without explicit approval. Canonical backfill staging remains under allowlisted `/data/home/ziyuzhu/tmp` unless separately approved.
- Keep package code/tests/scripts/docs independent of ignored `local_archive/`.

## Completed status

- Phase 0 git hygiene: completed in `ddebb13`.
- Phase 1 common order-parameter module: completed in `ddebb13`.
- Phase 2 optical-response package boundary/facade: completed in `9b374cc`.
- Phase 3 cRPA/HF bridge split: completed in `ef8ba6f`.
- Phase 4 public API registries and workflow extraction: completed across `e6eab50`, `a00803f`, `9b8202a`, and `faa706e`.
- Devtools cleanup follow-up: RLG/hBN retired sidecar/archive helpers moved to `src/mean_field/workflows/rlg_hbn.py` and retired command files deleted in the current continuation.

## Phase 0 — git surface hygiene

1. Stop ignoring the entire `tests/` tree. Public contract tests must be tracked by default.
2. Ignore only local/generated test payloads:
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

- old/new outputs match existing tests and new `tests/test_order_parameters.py` equivalence checks;
- system modules keep only label/adaptation wrappers;
- no formulas or thresholds are changed except through explicitly tested delegation.

## Phase 2 — optical response package boundary

Create `src/analysis/optical_response/` and migrate implementation from:

- `src/analysis/response_derivative_gauge.py`
- `src/analysis/shift_current/core.py`

Keep old paths as compatibility shims/re-exports. Keep system adapters thin: they prepare Hamiltonian/eigenpairs/derivatives/convention signs and call the common package.

Acceptance:

- current response/shift-current tests keep passing;
- new smoke tests cover the new package import path and one TDBG one-k call;
- retired hTG/TBG paper workspaces remain retired.

## Phase 3 — cRPA/HF bridge split

Split `src/mean_field/crpa/hf_interface.py` into:

- `src/mean_field/crpa/hf_bridge/density.py`
- `src/mean_field/crpa/hf_bridge/split_scheme.py`
- `src/mean_field/crpa/hf_bridge/kernels.py`
- `src/mean_field/crpa/hf_bridge/energy.py`
- `src/mean_field/crpa/hf_bridge/runner.py`

Keep `crpa/hf_interface.py` as a compatibility re-export facade, target `<180` lines.

Acceptance:

- no algorithmic cRPA change;
- bare split identity test error `<1e-10`;
- density and split-scheme conventions explicitly tested.

## Phase 4 — public API registries and workflow extraction

- Replace `api/crpa.py` NotImplemented facade with a registry-based entry to the TBG cRPA workflow.
- Add TDHF adapter registry in `api/tdhf.py`, initially for RnG/hBN q=0 and finite-q safe adapters.
- Convert `api/models.py` to a model adapter registry to reduce hard-coded imports.
- Move TDBG projected-HF config parsing/save workflow out of `cli.py` into `workflows/tdbg_projected_hf.py`.

Acceptance:

- public API docs updated;
- focused API tests added;
- CLI remains thin and backwards compatible.

## Lower priority

- Bands/topology wrappers are already thin enough; do not spend the next pass there unless a concrete duplicate blocks the above phases.
- Devtools cleanup: RLG/hBN retired runner helper extraction is complete; canonical backfill scan split remains a possible later cleanup if it blocks maintainability.
