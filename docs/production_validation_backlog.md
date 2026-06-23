# Production validation backlog runbook

This runbook is for the post-cleanup validation backlog.  It distinguishes software/API readiness from numerical or paper-level validation.  cRPA is archived out of the tracked package surface for now; TDHF work still requires a separate explicit request.

## Common rules

- Run heavy HF, topology, TDHF, response, or eigensolver workloads on a compute node or via Slurm, not on login nodes.
- Keep runs self-describing: save config, exact command, git commit, environment variables, stdout/stderr, manifest, and summary JSON/MD.
- A passing unit test is software validation only.  Paper-level validation requires saved artifacts and a quantitative comparison to the target panel/quantity.
- TDHF validation/code paths remain out of scope unless explicitly requested.
- cRPA validation is not part of the tracked package surface in this cleanup direction; archived code/docs/tests are under ignored `local_archive/retired_surface/crpa_untracked_20260622/`.
- Non-TDHF validation and cleanup gates may proceed after local self-checks of command, output path, and expected runtime.

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

The backfill scanner/writer devtool is archived out of the tracked package surface for now:

```text
local_archive/retired_surface/devtools_untracked_20260622/devtools/canonical_hf_backfill/
```

Historical application requires explicit authorization because it mutates `results/` and may cross TDHF-named loader boundaries. Current applied sidecar status:

- User authorized historical sidecar application on 2026-06-22.
- Staging root: `/data/home/ziyuzhu/tmp/mean_field_canonical_backfill_staged_229e06e_20260621_180741`.
- Apply log: `/data/home/ziyuzhu/tmp/mean_field_canonical_backfill_apply_7bede75_20260622.json`.
- Minimal manifest creation log: `/data/home/ziyuzhu/tmp/mean_field_canonical_backfill_manifest_create_7bede75_20260622.json`.
- Metadata-only contract audit: `/data/home/ziyuzhu/tmp/mean_field_canonical_backfill_metadata_validation_7bede75_20260622_141528.json`.
- Result: 48/48 RLG/hBN historical roots have `canonical_hf_run_result.json`, 48/48 have minimal `manifest.json`, 48/48 are discoverable by `mean_field.api.load_result(root)`, 48/48 source archives exist, issue count 0.
- No dense `canonical_hf_arrays.npz` payload was backfilled; historical sidecars are metadata-only summaries.

Public HF metadata-only sidecar coverage:

```text
coverage: tests/test_api_hf_adapters.py
adapters covered: tbg_zero_field, tdbg, htg, htg_supercell, rlg_hbn
validation: 16 passed on test001
assertion shape: HFResult.save(..., canonical_payload="metadata_only") writes canonical_hf_run_result.json, remains loadable via mean_field.api.load_result(...), and does not write canonical_hf_arrays.npz.
```

TMBG Polshyn has the same metadata-only sidecar coverage in `tests/test_tmbg_polshyn_hf_readiness.py`.

Acceptance after authorized application:

- `historical_results_mutated` is `true` only for the 48 authorized RLG/hBN sidecar/manifest writes above.
- Applied sidecars validate with `mean_field.api.load_result(...)` metadata-only loading.
- No overwrites were performed; future sidecar applications still require separate target-root review.

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
coverage commit: e308aa0
validation: pytest -q tests/test_tmbg_polshyn_hf_readiness.py
result: 10 passed on test001
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
pytest -q tests/test_htg_supercell.py tests/test_api_hf_adapters.py -k htg
```

Latest software-gate result on `test001`:

```text
archived historical gate: pytest -q tests/test_htg_supercell.py tests/test_htg_supercell_contract_adapter.py
current tracked gate: pytest -q tests/test_htg_supercell.py tests/test_api_hf_adapters.py -k htg
```

Additional public-HF gate:

```text
coverage commit: 834b132
validation: pytest -q tests/test_api_hf_adapters.py
result: 16 passed on test001
coverage added: explicit HTG supercell `run_hf(...)` metadata-only save/load writes `canonical_hf_run_result.json`, remains loadable via `mean_field.api.load_result(...)`, and does not write `canonical_hf_arrays.npz`.
```

Latest software preflight (not production validation):

```text
commit: 7d8ac74
output: /data/home/ziyuzhu/tmp/mean_field_validation_htg_supercell_7d8ac74_20260622_100932/summary.json
result_model: htg_supercell
has_canonical_run_result: true
loaded_canonical_sidecar: true
metadata_only_arrays_absent: true
primitive_nu: 3.5
supercell_area_ratio: 2
filling_from_density: 3.5000000000000018
workflow metadata: htg.supercell.explicit_config.preflight
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
# TDHF detailed tests are archived locally with the broad test suite.
pytest -q tests/test_api_imports.py
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

## 7. Current non-Slurm metadata-only preflight coverage

These checks were run on `test001` at commit `7bede75` and do not constitute paper-level physics validation.

Historical canonical sidecars:

```text
summary: /data/home/ziyuzhu/tmp/mean_field_canonical_backfill_metadata_validation_7bede75_20260622_141528.json
records: 48
manifest_exists: 48
sidecar_exists: 48
source_archive_exists: 48
load_result_ok: 48
canonical_loaded: 48
issue_count: 0
basis_systems: RnG_hBN
density_reference_schemes: CN, average
```

Fresh public HF adapter preflights:

```text
summary: /data/home/ziyuzhu/tmp/mean_field_production_validation_7bede75_20260622_134024/summary.json
cases: tmbg_polshyn_mesh1_metadata_only, htg_supercell_primitive_nu_3p333333_mesh1_metadata_only, htg_supercell_primitive_nu_3p666667_mesh1_metadata_only
all_have_canonical_run_result: true
all_loaded_canonical_sidecar: true
all_metadata_only_arrays_absent: true
```

```text
summary: /data/home/ziyuzhu/tmp/mean_field_production_validation_followup_7bede75_20260622_134844/summary.json
cases: tbg_zero_field_explicit_mesh2_metadata_only, rlg_hbn_explicit_mesh1_metadata_only, tdbg_projected_explicit_mesh1_metadata_only
all_have_canonical_run_result: true
all_loaded_canonical_sidecar: true
all_metadata_only_arrays_absent: true
```

## 8. Prepared but not submitted paper-level Slurm validation plan

No Slurm jobs have been submitted in this continuation. If paper-level validation is requested next, prepare reviewed scratch wrappers that:

1. run on a Slurm CPU partition selected from live `sinfo`/`scontrol` state and account `hmt03`;
2. save output under a new timestamped result root outside tracked git;
3. write `manifest.json`, config, environment, validation, observables, and canonical HF sidecars;
4. compare against an explicit target quantity/panel rather than claiming reproduction from software smoke tests.

Recommended first paper-level candidates, in increasing physics-risk order:

- TMBG Polshyn: choose target band/window and candidate initial states (`bm_wang`, `cdw`, competitors); run convergence/energy comparison before any topology claim.
- HTG fractional filling: use `HTGSupercellRunHFConfig`, exact SCF-grid path/bands, and document denominator/supercell choice for `3+1/3` and `3+2/3`.
- RLG/hBN canonical sidecar audit: use the newly discoverable historical sidecars as metadata inputs only; do not run TDHF finite-q validation unless TDHF scope is explicitly reopened.
- TDBG projected HF: run explicit projected config only after target window/filling/interaction choices are fixed.

Blocked or out-of-scope unless separately authorized:

- cRPA validation (tracked surface archived).
- topology/Chern/QGT paper-level validation beyond the common system-independent core and thin TMBG/TDBG/ATMG/RLG-hBN/HTG wrappers; reintroduce reviewed concrete workflows for each target system before claiming system or paper reproduction.
- RLG/hBN finite-q TDHF production maps (TDHF remains a separate scope boundary).
