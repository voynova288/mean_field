# Script surface policy

This repository should not grow by adding a new standalone script for every diagnostic, Slurm run, paper panel, or parameter sweep.  The target command surface is intentionally small and generic.

## Current audit

The 2026-06-02 cleanup removed the historical tracked pile of one-off launchers and paper-panel scripts.  The remaining tracked surface is deliberately small:

- `../scripts/`: `AGENTS.md`, `mean_field_tools.py`, and `submit_mean_field.sbatch`.
- `../src/mean_field/devtools/`: `AGENTS.md` plus five maintained cRPA/TPT implementation modules behind the developer-tool dispatcher or lightweight tests.

Treat deleted one-off wrappers as historical debt, not as templates to restore.  If a retired workflow is needed again, recover the reusable logic from git history into an existing dispatcher rather than restoring the old per-run wrapper as-is.

## Preferred command surfaces

Keep durable entrypoints concentrated in:

- package CLI groups in `../src/mean_field/cli.py` for stable user-facing workflows;
- `../scripts/mean_field_tools.py` only for maintained developer utilities;
- `../scripts/submit_mean_field.sbatch` for generic Slurm execution.

The package CLI and developer-tool dispatcher are separate command surfaces: neither forwards commands to the other. Invoke package workflows with `python -m mean_field.cli` (or the installed `mean-field` console entry), and invoke only registered developer tools through `mean_field_tools.py`.

Historical B0 Julia exporters and paper-benchmark commands are archived provenance, not maintained entrypoints. Independent numerical fixtures may remain under `benchmarks/`, but generating or plotting those fixtures is not a package workflow.

New work should first try to call an existing command through these surfaces, for example:

```bash
PYTHONPATH=../src python -m mean_field.cli <package-command> ...
python ../scripts/mean_field_tools.py <developer-tool-command> ...
sbatch ../scripts/submit_mean_field.sbatch python -m mean_field.cli <package-command> ...
```

## Rules for new scripts/devtools

Before adding any file under `scripts/` or `src/mean_field/devtools/`, check whether an existing command can accept one more option, subcommand, config file, or input artifact.

Add a new tracked script/devtool only if all of the following are true:

1. The workflow is expected to be reused beyond the current conversation or one paper-panel attempt.
2. It cannot be cleanly expressed as arguments to an existing command.
3. It has one clear owner surface: package CLI or `mean_field_tools.py`, with the generic Slurm wrapper used only for execution and no cross-forwarding between command owners.
4. It has a small validation path such as `--help`, dry-run, syntax check, saved-result check, or a tiny smoke test.
5. It does not bypass existing login-node guards for HF, topology, eigensolver, or response-function compute.

Do not commit per-run `.sbatch` files, timestamped launchers, ad hoc plotting scripts, or narrow parameter sweeps unless the user explicitly asks to preserve them as durable project assets.  Put scratch launchers in ignored locations such as `tmp/` or `scripts/local/`.

## cRPA devtool fail-closed contract

The retained `run_tbg_crpa_chunk` and `merge_tbg_crpa_chunks` commands remain developer tools; this hardening does not migrate them into the public API, change cRPA physics, or remove either command. Both commands must apply the shared login-node guard after parsing and before creating output directories, writing workflow artifacts, loading the BM cache, or initializing plotting.

Their stricter failures are intentional compatibility breaks at invalid or ambiguous inputs: chunk selectors no longer clip out-of-bounds ranges, q-range and chunk selectors cannot be combined, chunk index/count must be paired and nonempty, malformed scalar/shell requests fail before compute, and merge now requires exact duplicate-free `lk^2` q coverage unless `--allow-partial` explicitly requests a nonempty valid subset. Merge inputs must match the writer's five-file/NPZ-label/shape/integer-coordinate contract before normal serialization begins.

The merge boundary is fail-closed. It reads `crpa_params.json` with `read_json_artifact`, rejects duplicate keys and nonstandard constants, requires exact top-level and Coulomb schemas with non-boolean scalar types and finite/domain-valid values, and requires canonical `build_q_shift_table(q_lg)[1]` ordering in every NPZ and chunk. Physical arrays are dtype- and finiteness-checked before any conversion; real, complex, and index arrays cannot cross categories. Writer-redundant q-vector, q-norm, unit-conversion, effective-epsilon, and `epsilon_times_bn` fields must agree before merged scientific files are created. In addition, `effective_epsilon` must tightly equal `real(diag(epsilon))`; the canonical q-shift inventory must contain exactly one `(0, 0)` column; and `q_tilde_real`/`q_tilde_imag` must tightly equal `q_real`/`q_imag` in that column.

The three genuinely chunk-local metadata fields `k_periodic_max_wrap_shell`, `periodic_roll_required_lg`, and `periodic_roll_no_alias` form one per-chunk all-missing/all-present triple. Present values have exact non-boolean integer/integer/boolean types, satisfy nonnegative maximum wrap, the owner relation `periodic_roll_required_lg == q_lg + 2*k_periodic_max_wrap_shell`, and `periodic_roll_no_alias == (lg >= periodic_roll_required_lg)`. Valid chunk values merge by `max`, `max`, and logical `all`. Static metadata uses recursive, type-preserving comparison: dictionary keys are exact; booleans, strings, and integers are exact with booleans distinct from integers; finite floats use `rtol=1e-12`, `atol=1e-14`; and lists/tuples recurse without cross-type coercion. `merge_coverage` is merge-local, is ignored as an input comparison field, and is removed then regenerated from current coverage so valid partial outputs can be recursively merged.

The merged `crpa_params.json` metadata and validation report both record actual partial/full coverage without changing top-level schema, base filenames, or NPZ keys. Workflow input jobs are explicitly provenance-only, marked non-executable, and remain pending/awaiting validation until every chunk has loaded and validated. Replay commands use the active `sys.executable`, the absolute repository `scripts/mean_field_tools.py`, the dispatcher command, exact `repr` float arguments, and `.resolve()` absolute BM-solution/output/chunk arguments. Manifest metadata records the true launch cwd and the replay-command owner rather than claiming the repository root was the launch cwd.

Merge-owner consolidation is complete: `mean_field.crpa.chunk_merge` uniquely owns strict chunk validation, coverage/canonical sorting, metadata aggregation, and the five-file merged base-artifact writer, while `merge_tbg_crpa_chunks` retains only CLI, guard, workflow state, presentation, diagnostics, and stdout orchestration. Preserve the existing NPZ keys, dtypes/numerical serialization behavior, and base artifact filenames; do not route this devtool through `mean_field.api.compute_crpa` or change `compute_crpa_from_solution`. The public `mean_field.api.compute_crpa` path remains disabled, and production cRPA physics remains unvalidated. The dedicated contracts construct chunks through real `CRPAResult -> write_crpa_outputs` serialization, compare all restored scalar/Coulomb/array fields supported by `load_crpa_result`, cover multi-chunk `q_lg > 1`, and exercise recursive partial merging. These engineering contracts have run under focused and full pytest gates in addition to compile/import/manual golden parity checks; no production cRPA numerical calculation or physics validation has run.

## Devtool runtime-owner policy

`mean_field.devtools._runtime` is retired rather than retained as a compatibility facade. Its four migrated callers—`merge_tbg_crpa_chunks`, `run_tbg_crpa_chunk`, `qualify_tpt_bulk_parent_wavefunctions`, and `tpt_bulk_paw_oracle`—must import the login-node guard directly from `mean_field.runtime`. The separately maintained `tpt_wannier_q0_hf` command also imports that canonical guard directly and was never an `_runtime` caller. TPT JSON artifacts must use `mean_field.core.io.write_json_artifact(payload, path)` directly. The retired CSV parsers and complex-array converter had no maintained callers and must not be copied into another owner.

The TPT parent qualifier and PAW oracle each own a versioned selected-provenance inventory. These inventories are intentionally selected file sets, **not** complete Python import closures. Parent-qualification reports use schema v5, version `parent_qualification_selected_provenance_v2`, and exact scope `parent_qualification_v2_selected_set_not_complete_python_import_closure`. Active/full PAW producers and all four postflight reports use version `paw_oracle_selected_provenance_v2` and exact scope `paw_oracle_v2_selected_set_not_complete_python_import_closure`. The inventories include `mean_field.runtime`, `mean_field.core.io`'s export and implementation owners, and the relevant TPT source owner; they exclude the retired devtool runtime module.

PAW v2 preparation records use exactly `selected_provenance_version`, `selected_provenance_scope`, and `selected_provenance_hashes_at_preparation`. Postflight reports add `selected_provenance_hashes_at_validation` and `selected_provenance_drift`; the retired `runtime_source_*` names are legacy input and are not accepted. Source seals, active handoff records, submission/revalidation records, prior reports, and related JSON trust inputs are read through `mean_field.core.io.read_json_artifact`, require a top-level object, and reject duplicate keys and `NaN`/`Infinity` constants.

PAW handoff and postflight validation is fail closed: legacy or unversioned seals, missing/unknown scopes, unknown versions, malformed or uppercase hashes, and missing/extra inventory keys raise domain-specific `ValueError`. A single postflight provenance helper performs record/keyset/content validation for all four validators; version, scope, and keyset failures occur before the external authorization loader, while only same-version, same-scope, exact-keyset content drift may reach the existing external artifact-revalidation path. Historical result trees and seals are not rewritten, are not silently or automatically upgraded, and remain fail-closed under this schema. This policy is engineering/provenance authority only and is not results-byte proof or production-physics validation.

## Consolidation target

Long term, `../scripts/` should contain only a few generic entrypoints plus possibly a small number of stable language/reference bridges.  `../src/mean_field/devtools/` should contain reusable implementation modules behind those entrypoints, not a flat collection of one-off runners.

When cleaning up existing files:

- migrate reusable logic into system modules, common analysis modules, or an existing devtool;
- preserve durable public design details in `docs/`, and keep run-specific diagnostics in ignored local `reports/` directories or result metadata rather than in many near-duplicate launch scripts;
- when retiring substantial system-specific HF/topology/bands/plotting code, copy it first into ignored `local_archive/retired_surface/...` if it may be useful for future repair; archived code must not be imported by tracked package code and does not count as maintained surface;
- delete or untrack obsolete wrappers after confirming no current documentation or tests depend on them;
- keep heavy validation on Slurm; keep hard-coded saved-result artifact audits in ignored reports/internal workspaces rather than public package modules.
