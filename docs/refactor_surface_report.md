# Refactor surface report

This Phase 2 report measures legacy surface area and tracks cleanup slices that delete or thin old paths.

## Summary

- Tracked text lines: 71028
- Tracked Python lines: 67005
- Tracked Julia lines: 826
- `src` Python files: 191
- `src` Python lines: 60878
- Files over 1000 lines: 13
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5

## Completed cleanup slices

### remove_atmg_fig3_devtool

- Moved reusable build_khalaf_fig3_path helper from one-off devtool to systems/atmg/bands.py, removed run_atmg_fig3_band_plot dispatcher command, and deleted tracked paper-panel devtool.
- Deleted files: `src/mean_field/devtools/run_atmg_fig3_band_plot.py`.
- Gross legacy LOC removed/thinned: 665.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 20 -> 18.

### remove_rlg_hbn_band_plot_devtool

- Moved reusable RnG/hBN Fig.6 HF path and band-plot manifest helpers into systems/RnG_hBN/bands.py, removed plot_rlg_hbn_paper_hf_bands dispatcher command, and deleted tracked paper-panel plotting devtool.
- Deleted files: `src/mean_field/devtools/plot_rlg_hbn_paper_hf_bands.py`.
- Gross legacy LOC removed/thinned: 739.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 18 -> 17.

### remove_htqg_fig1_bands_devtool

- Removed one-off HTQG Fig.1 first-pass band plotting devtool from tracked command surface. The reusable HTQG band/path/domain APIs remain in systems/htqg.
- Deleted files: `src/mean_field/devtools/run_htqg_fig1_bands.py`.
- Gross legacy LOC removed/thinned: 213.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 17 -> 13.

### remove_htqg_projected_hf_devtool

- Removed HTQG projected-HF paper-scan CLI glue from tracked devtools. The core HTQG projected-HF solver remains in systems/htqg/hf.py; future durable reproduction should be reintroduced under workflows with explicit validation gates.
- Deleted files: `src/mean_field/devtools/run_htqg_projected_hf.py`.
- Gross legacy LOC removed/thinned: 400.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 13 -> 10.

### remove_htg_hf_devtool

- Removed duplicate HTG projected-HF CLI glue from devtools after the public API gained explicit HTGRunHFConfig dispatch and the system runner remains available in systems/htg.
- Deleted files: `src/mean_field/devtools/run_htg_hf.py`.
- Gross legacy LOC removed/thinned: 463.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 10 -> 9.

### dedupe_htg_htqg_central_band_metrics

- Moved shared central two-band bandwidth/gap diagnostics into mean_field.core.bands. HTG and HTQG keep stable estimate_central_band_metrics wrappers but no longer duplicate the metric implementation.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 49.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_selected_band_indices

- Moved centered/selected band-index resolution into mean_field.core.bands. HTG and HTQG bands now use the core resolver, while their hamiltonian modules preserve centered_band_indices as a compatibility alias.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 39.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_tdbg_tmbg_band_diagonalizers

- Collapsed duplicate path/grid diagonalizer closures inside TDBG and TMBG band adapters while preserving their public compute_bands_* APIs.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 12.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_rlg_hbn_band_diagonalizer

- Collapsed duplicate path/grid diagonalizer closures inside the RnG/hBN band adapter without touching the dirty HF/TDHF implementation files.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 8.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_htqg_validation_types

- Reused core ValidationCheck/ValidationReport for HTQG validation, adding optional tolerance support to the core validation record instead of maintaining a second local dataclass pair.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 49.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### drop_validation_status_aliases

- Removed tiny ATMG/TDBG local status alias functions and call core status_from_bool directly.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 6.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### drop_tmbg_validation_status_alias

- Removed the tMBG local status alias and call core status_from_bool directly throughout validation.py.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 3.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### drop_rlg_hbn_validation_status_alias

- Removed the RnG/hBN local status alias and call core status_from_bool directly, without touching dirty HF/TDHF implementation files.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 3.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### drop_htg_validation_status_alias

- Removed the HTG local status alias and call core status_from_bool directly from its validation helper.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 3.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_htg_validation_check_helper

- Moved HTG's generic condition/value/tolerance ValidationCheck constructor into core.validation.make_validation_check and kept HTG output detail strings compatible.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 8.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### dedupe_htqg_validation_check_helper

- Reused core.validation.make_validation_check from HTQG validation while preserving its system-specific detail strings and tolerance payloads.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 2.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 9.

### thin_rlg_hbn_fig6_prereq_devtool

- Moved the reusable RnG/hBN Fig. 6 screened-U checkpoint helper behind mean_field.api.validation and thinned the devtool to CLI argument parsing plus login-node guard.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 88.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 9 -> 8.

### lazy_load_rlg_hbn_q0_tdhf_devtool

- Removed the top-level RnG/hBN system import from the q=0 TDHF devtool; the heavy system adapter is now imported lazily only after dry-run/config validation and login-node guards.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 6.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 8 -> 7.

### lazy_load_rlg_hbn_backfill_adapters

- Removed direct RnG/hBN system imports from the canonical sidecar backfill defaults; write-mode loaders/adapters are now imported lazily via import_module.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 2.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 7 -> 5.

### dedupe_tmbg_validation_check_helper

- Reused core.validation.make_validation_check for the simple tMBG validate_physics checks, reducing repeated condition/status/value/detail boilerplate without changing thresholds or diagnostics.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 14.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_tmbg_validation_append_checks

- Reused core.validation.make_validation_check for the remaining simple tMBG append-time validation checks, leaving skipped/fail diagnostics explicit.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 10.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_htqg_band_diagonalizer

- Collapsed duplicate HTQG path/grid selected-band diagonalizer setup into one system-local helper while preserving public compute_bands_* APIs and metadata.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 10.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_htg_band_diagonalizer

- Collapsed duplicate HTG path/grid selected-band diagonalizer setup into one system-local helper while preserving public compute_bands_* APIs.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 14.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_atmg_band_diagonalizer_call

- Collapsed repeated ATMG direct diagonalization calls used by plain and mapped band paths into one system-local helper without changing mapped-spectrum logic.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 16.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### thin_rlg_hbn_band_helpers

- Thinned RnG/hBN band helper payload construction without changing Fig. 6 path semantics, manifest keys, or public exports.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 14.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_tmbg_ktilde_validation_checks

- Reused core.validation.make_validation_check for tMBG Ktilde diagnostic pass/fail checks while preserving detail strings.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 4.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_tmbg_checkpoint_validation_checks

- Reused core.validation.make_validation_check across tMBG paper-checkpoint pass/fail records while preserving checkpoint names, values, and detail strings.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 16.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### dedupe_rlg_hbn_validation_checks

- Reused core.validation.make_validation_check for RnG/hBN validation pass/fail records while preserving re-exported validation types and diagnostics.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 7.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_hipolito_shift_current_audit_model

- Retired the unexported Hipolito 2016 shift-current paper-audit toy model from the public analysis package; reusable shift-current math and the lightweight SLG toy benchmark remain in analysis.shift_current.
- Deleted files: `src/analysis/shift_current/toy_models/hipolito2016.py`.
- Gross legacy LOC removed/thinned: 778.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_chaudhary_hartree_diagnostic_module

- Retired the unexported Chaudhary 2021 Hartree diagnostic module from the TBG system package; the maintained Chaudhary shift-current adapter remains in mean_field.systems.tbg.chaudhary2021.
- Deleted files: `src/mean_field/systems/tbg/chaudhary2021_hartree.py`.
- Gross legacy LOC removed/thinned: 733.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tdbg_hf_plotting_helper

- Retired the non-exported TDBG projected-HF plotting helper module and its lone helper-specific test; the public TDBG path plotting surface remains in mean_field.systems.tdbg.plot.
- Deleted files: `src/mean_field/systems/tdbg/hf_plotting.py`.
- Gross legacy LOC removed/thinned: 349.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_full_flavor_ivc_scaffold

- Retired the unexported Polshyn full-flavor IVC array-contract scaffold; no tracked production code imports it and the maintained Polshyn/Wang HF adapter remains in tmbg.polshyn_supercell.
- Deleted files: `src/mean_field/systems/tmbg/full_flavor_ivc.py`.
- Gross legacy LOC removed/thinned: 687.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_legacy_b0_julia_inspect_helpers

- Retired legacy B0 Julia inspect-only helpers from scripts/_julia_impl while preserving the Julia dispatcher and export/reference commands.
- Deleted files: `scripts/_julia_impl/inspect_b0_grid_overlap_julia.jl`, `scripts/_julia_impl/inspect_b0_hf_first_iteration_julia.jl`, `scripts/_julia_impl/inspect_b0_hf_iteration_trace_julia.jl`, `scripts/_julia_impl/inspect_b0_hf_shift_metrics_julia.jl`, `scripts/_julia_impl/inspect_b0_overlap_reference_julia.jl`.
- Gross legacy LOC removed/thinned: 699.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_paper_checkpoint_runner

- Retired the heavy tMBG Park paper-checkpoint reproduction runner from system validation and CLI; lightweight validate_physics and Ktilde diagnostics remain available.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 600.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_ktilde_diagnostic_cli

- Retired the remaining paper-specific tMBG Ktilde diagnostic CLI/export surface; core tMBG validate_physics and Hamiltonian cross-checks remain in system validation.
- Deleted files: `tests/test_cli_tmbg_artifacts.py`.
- Gross legacy LOC removed/thinned: 350.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_cutoff_validation_path

- Retired the optional tMBG cutoff-convergence path from lightweight validation; C9 remains a skipped report check pointing users to dedicated Slurm convergence workflows.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 150.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_atmg_khalaf_checkpoint_helper

- Retired the unreferenced ATMG Khalaf paper-checkpoint helper from the public system surface; ATMG validate_physics remains as the lightweight validation entry point.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 27.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_duplicate_hamiltonian_cross_check

- Retired the unexported tMBG duplicate Hamiltonian cross-check builder and removed the C11 validation hook; direct Hamiltonian tests and lightweight validate_physics remain.
- Deleted files: `src/mean_field/systems/tmbg/cross_check.py`.
- Gross legacy LOC removed/thinned: 345.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_optional_validation_diagnostics

- Retired the optional tMBG node-exchange and C3 diagnostic branches from validate_physics while keeping compatibility kwargs and skipped report entries.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 50.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_topology_saved_artifact_validator

- Retired the hard-coded analysis.topology saved-result validator for dated results trees; reusable topology primitives and system adapters remain, while historical artifact audits move to ignored internal workspaces.
- Deleted files: `src/analysis/topology/validate_existing_results.py`.
- Gross legacy LOC removed/thinned: 263.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### thin_tmbg_polshyn_legacy_helpers

- Removed old standalone Polshyn path/grid/projected-basis/manual-SCF helper paths while keeping the registered Polshyn-Wang canonical bundle adapter and tested Wang problem/filling utilities.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 948.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tbg_zero_field_supercell_legacy_workflow

- Replaced the unexported TBG zero-field supercell BM/SCF workflow module with the small Zhang sqrt(3) filling-convention helpers that are actually referenced.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 879.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_topology_paper_target_registry

- Retired the hard-coded analysis.topology paper/artifact target registry from the common topology package; dated reproduction inventories belong in ignored reports/internal workspaces.
- Deleted files: `src/analysis/topology/targets.py`.
- Gross legacy LOC removed/thinned: 157.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_htg_paper_plot_writers

- Retired HTG Fig. 3b/Fig. 7/Fig. 8a paper-panel plotting writers from the public system surface while keeping generic HTG path and HF path band plotting helpers.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 285.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_paper_band_figure_writer

- Retired the tMBG Fig. 2-like multi-panel paper-band composer and panel dataclass while keeping ordinary band, lattice, Berry-curvature, and flat-band-index plot helpers.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 159.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_htqg_private_plot_helpers

- Deleted the unexported HTQG private plotting helper module; reusable plotting remains in mean_field.core.plotting and active system plot adapters.
- Deleted files: `src/mean_field/systems/htqg/plot.py`.
- Gross legacy LOC removed/thinned: 81.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_htqg_charge_density_diagnostic

- Deleted the unexported HTQG real-space charge-density diagnostic helper module after confirming no package, docs, or tracked tests reference it.
- Deleted files: `src/mean_field/systems/htqg/density.py`.
- Gross legacy LOC removed/thinned: 102.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### thin_slg_toy_point_reference

- Removed unreferenced full-BZ SLG toy integration and C3 diagnostic helpers while keeping the point-level Hamiltonian, derivatives, and diagonalizer used by gauge-safe response tests.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 186.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### prune_devtool_runtime_selectors

- Removed unused band-window selector helpers from the devtool runtime module while preserving JSON, CSV, complex-pair, and login-node guard utilities used by tracked devtools.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 38.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### prune_tmbg_polshyn_unused_diagnostics

- Removed unreferenced Polshyn/Wang target-Hamiltonian and sector diagnostic helpers while preserving the tested Wang HF problem builder, bundle adapter, filling helpers, and CDW order diagnostics.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 71.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

## Top 30 Python files under `src`

| Lines | Path |
|---:|---|
| 2361 | `src/mean_field/systems/RnG_hBN/hf.py` |
| 2305 | `src/mean_field/systems/htg/mean_field_adapter.py` |
| 1640 | `src/mean_field/systems/RnG_hBN/tdhf.py` |
| 1566 | `src/mean_field/devtools/backfill_canonical_hf_sidecars.py` |
| 1563 | `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` |
| 1348 | `src/mean_field/systems/tbg/zero_field/runners.py` |
| 1347 | `src/mean_field/systems/htg/supercell.py` |
| 1306 | `src/mean_field/api/hf.py` |
| 1267 | `src/mean_field/crpa/hf_interface.py` |
| 1258 | `src/mean_field/systems/tbg/zero_field/hf.py` |
| 1149 | `src/mean_field/core/hf/finite_field.py` |
| 1127 | `src/mean_field/systems/tmbg/polshyn_supercell.py` |
| 1012 | `src/mean_field/systems/tbg/finite_field/spectrum.py` |
| 925 | `src/mean_field/systems/htqg/hf.py` |
| 873 | `src/analysis/topology/quantum_geometry.py` |
| 805 | `src/analysis/response_derivative_gauge.py` |
| 786 | `src/mean_field/systems/RnG_hBN/cache.py` |
| 772 | `src/analysis/shift_current/core.py` |
| 736 | `src/mean_field/cli.py` |
| 735 | `src/mean_field/systems/tbg/zero_field/hf_contracts.py` |
| 710 | `src/mean_field/core/hf/tdhf.py` |
| 686 | `src/mean_field/systems/RnG_hBN/hf_contracts.py` |
| 683 | `src/mean_field/systems/tbg/chaudhary2021.py` |
| 643 | `src/mean_field/systems/htg/supercell_contracts.py` |
| 627 | `src/mean_field/benchmarks.py` |
| 617 | `src/mean_field/core/hf/overlap.py` |
| 565 | `src/analysis/topology/core.py` |
| 528 | `src/mean_field/systems/tbg/zero_field/artifacts.py` |
| 498 | `src/mean_field/crpa/validation.py` |
| 492 | `src/mean_field/systems/RnG_hBN/screening.py` |

## Direct private-system imports in workflow surfaces

| Path | Line | Import |
|---|---:|---|
| `src/mean_field/devtools/prepare_tbg_crpa_bm.py` | 9 | `from mean_field.systems.tbg.params import TBGParameters` |
| `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` | 35 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` | 50 | `from mean_field.systems.RnG_hBN.hf import RLG_HBN_FORM_FACTOR_CONVENTION_VERSION` |
| `src/mean_field/devtools/run_rlg_hbn_tdhf_finite_q.py` | 16 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/validate_tbg_crpa_artifact.py` | 22 | `from mean_field.systems.tbg import TBGParameters` |

## Repeated module-family line counts

### `bands.py`

Total lines: 691

| Lines | Path |
|---:|---|
| 174 | `src/mean_field/systems/atmg/bands.py` |
| 155 | `src/mean_field/systems/RnG_hBN/bands.py` |
| 130 | `src/mean_field/systems/htqg/bands.py` |
| 113 | `src/mean_field/systems/htg/bands.py` |
| 64 | `src/mean_field/systems/tmbg/bands.py` |
| 55 | `src/mean_field/systems/tdbg/bands.py` |

### `validation.py`

Total lines: 923

| Lines | Path |
|---:|---|
| 201 | `src/mean_field/systems/tmbg/validation.py` |
| 193 | `src/mean_field/systems/htqg/validation.py` |
| 156 | `src/mean_field/systems/RnG_hBN/validation.py` |
| 138 | `src/mean_field/systems/atmg/validation.py` |
| 126 | `src/mean_field/systems/tdbg/validation.py` |
| 109 | `src/mean_field/systems/htg/validation.py` |

### `topology.py`

Total lines: 1362

| Lines | Path |
|---:|---|
| 380 | `src/mean_field/systems/RnG_hBN/topology.py` |
| 338 | `src/mean_field/systems/htg/topology.py` |
| 304 | `src/mean_field/systems/htqg/topology.py` |
| 167 | `src/mean_field/systems/tdbg/topology.py` |
| 96 | `src/mean_field/systems/tmbg/topology.py` |
| 77 | `src/mean_field/systems/atmg/topology.py` |

## Repeated symbol names

### `bands.py`

| Symbol | Count | Paths |
|---|---:|---|
| `_diagonalize` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `_make_diagonalizer` | 4 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `_prepare_band_diagonalizer` | 2 | `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py` |
| `compute_bands_along_path` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `compute_bands_on_grid` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `estimate_central_band_metrics` | 2 | `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py` |

### `validation.py`

| Symbol | Count | Paths |
|---|---:|---|
| `_check` | 2 | `src/mean_field/systems/htg/validation.py`, `src/mean_field/systems/htqg/validation.py` |
| `validate_lattice` | 2 | `src/mean_field/systems/htg/validation.py`, `src/mean_field/systems/htqg/validation.py` |
| `validate_physics` | 4 | `src/mean_field/systems/RnG_hBN/validation.py`, `src/mean_field/systems/atmg/validation.py`, `src/mean_field/systems/tdbg/validation.py`, `src/mean_field/systems/tmbg/validation.py` |

### `topology.py`

| Symbol | Count | Paths |
|---|---:|---|
| `ChernBasisResult` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `_central_eigensystem` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `_normalize_band_indices` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `_reciprocal_translation` | 2 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/htg/topology.py` |
| `_topology_adapters` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `apply` | 3 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `boundary_sewing_transforms` | 2 | `src/mean_field/systems/htqg/topology.py`, `src/mean_field/systems/tdbg/topology.py` |
| `compute_chern_basis_on_grid` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `compute_topology_from_eigenvectors` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `compute_topology_from_grid_result` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `compute_topology_on_grid` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `grid_builder` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `integer_residual_a` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `integer_residual_b` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `sublattice_sigma_z` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `to_dict` | 2 | `src/mean_field/systems/htg/topology.py`, `src/mean_field/systems/htqg/topology.py` |
| `transform` | 2 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/tdbg/topology.py` |

## Phase 2 cleanup gates

- No PR should claim cleanup if it only adds wrappers without deleting or thinning legacy paths.
- Each slice must update this report with before/after LOC and remaining old entry points.
- Physics-heavy migrations require focused parity tests before deleting old implementations.
- cRPA/HF bridge changes are deferred until the known cRPA bug is isolated.
