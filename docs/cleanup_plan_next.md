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

- Bands wrappers remain. Thin TMBG, TDBG, RLG-hBN, and HTG topology wrappers are restored; paper workflows remain archived, while the tracked surface includes the common FHS topology core and generic wavefunction/system adapters under `src/analysis/topology`.
- Devtools cleanup: tracked devtools are archived locally for now; future durable commands should be reintroduced only through a small reviewed public surface.

## Phase 5 — 35k tracked-core profile cleanup

New target after the Polshyn/topology/API cleanup through `origin/main` `dec4fa2`:

- The 35k budget counts only tracked Python source under `src/`.
- Tests, docs, and scripts are not part of the 35k budget, but tests should not be deleted for line count. Heavy or historical tests may move only to ignored `tests/local/` or `tests/slow/` when their feature surface is archived.
- Do not keep splitting already-small files just to appear cleaner; all tracked Python files are now below 1000 lines. The next reduction must come from reducing tracked feature surface.
- Every optional-feature retirement must first copy the tracked source into ignored `local_archive/optional_features/<feature>_<date>/`, then remove the tracked files with `git rm`.
- Tracked code, tests, scripts, and docs must not import from ignored `local_archive/`.

### Core profile to preserve

Keep the tracked core profile focused on:

- `mean_field.api` public facade and adapters for the retained systems.
- `mean_field.core` generic lattice/HF/contracts/I/O/reconstruction helpers.
- `analysis.order_parameters`.
- `analysis.optical_response`.
- `analysis.topology` minimal FHS/link/plaquette/Chern core, wavefunction-grid helpers, and system-facing adapter.
- Retained system surfaces: RnG/hBN HF, TDBG projected HF, TBG zero-field, primitive HTG, and minimal TMBG/Polshyn adapters.

### Optional features to archive first

1. **TBG finite-field lane**
   - Archive `src/mean_field/systems/tbg/finite_field/`.
   - Archive generic finite-B helpers: `src/mean_field/core/hf/finite_field.py`, `src/mean_field/core/hf/_finite_field_*.py`, and `src/mean_field/core/magnetic_field.py`.
   - Remove public exports/tests/docs references to these modules.
   - Expected reduction: roughly 3.1k lines.
2. **HTG supercell lane**
   - Archive `src/mean_field/systems/htg/supercell.py`, `src/mean_field/systems/htg/supercell_contracts.py`, and `src/mean_field/systems/htg/_supercell_*.py`.
   - Keep primitive HTG model/HF.
   - Expected reduction: roughly 2.4k lines.
3. **Exploratory optional systems**
   - Archived `src/mean_field/systems/atmg/` and `src/mean_field/systems/htqg/` to ignored optional-feature storage.
   - Expected reduction: roughly 2.0k lines combined.
4. **Topology quantum geometry and public TDHF façade**
   - Archived `src/analysis/topology/quantum_geometry.py` while keeping FHS topology core/system/wavefunction helpers.
   - Archived the small public `mean_field.api.tdhf` façade and docs; retained lower-level core/system TDHF code is still source-budget-visible and should only be interpreted as internal/optional scope unless reopened.
5. **Contract/sidecar compression**
   - Continue extracting shared metadata/sidecar helpers out of system contract files without changing system physics.
6. **RLG/hBN cache compression**
   - Move generic cache/hash/path mechanics into a reusable core I/O helper while preserving RLG/hBN-specific cache keys and physics checks.

### Non-negotiable safety constraints

- Do not delete or simplify RnG/hBN layer-dependent Coulomb, layer-dependent form factors, q=0 internal screening limits, or average/CN interaction schemes for line count.
- Do not move topology FHS math back into system modules.
- Do not mutate historical `results/` during source-surface cleanup.
- Do not submit Slurm jobs for these archive phases; source compile/import/tests are sufficient unless a retained physical code path changes.
- cRPA remains archived unless explicitly reopened. If restored later, preserve the split convention: bare Coulomb for remote-reference/Hartree subtraction and screened cRPA only for the intended dynamic self-energy channel.

### Validation after each phase

Run from a compute-safe development node such as `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
python -m pip install -e . --dry-run --no-deps --no-build-isolation
python scripts/line_budget.py --root src --suffix .py --max-lines 35000
```

The line-budget tool is report-only by default; use `--fail-on-over` only after the codebase is expected to be under budget.
