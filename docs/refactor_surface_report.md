# Refactor surface report

This Phase 2 report measures legacy surface area and tracks cleanup slices that delete or thin old paths.

## Summary

- Tracked text lines: 62671
- Tracked Python lines: 59348
- Tracked Julia lines: 826
- `src` Python files: 238
- `src` Python lines: 53245
- Files over 1000 lines: 1
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0

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

### prune_unused_lattice_path_helpers

- Removed unreferenced generic uniform-lattice and TBG path-sample segment helpers while preserving the active LatticeGrid dataclass and path projection API.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 22.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_topology_pilot_sewn_adapters

- Retired unexported HTQG pilot topology and tMBG sewn topology adapter modules from the public package surface after owner approval; maintained topology wrappers still delegate to analysis.topology.
- Deleted files: `src/mean_field/systems/htqg/topology.py`, `src/mean_field/systems/tmbg/topology_sewn.py`.
- Gross legacy LOC removed/thinned: 537.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_shift_current_paper_adapters

- Retired the Chaudhary b0 TBG and hTG legacy shift-current paper-adapter modules after owner approval; maintained response math remains in analysis.shift_current and response_derivative_gauge, with TDBG/Joya as the active system adapter.
- Deleted files: `src/mean_field/systems/htg/shift_current.py`, `src/mean_field/systems/tbg/chaudhary2021.py`.
- Gross legacy LOC removed/thinned: 1025.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_tmbg_plot_validation_tails

- Retired ignored-test-only tMBG plotting and lightweight validation tail modules, keeping the maintained tMBG model, bands, topology, and Polshyn-Wang bundle surfaces.
- Deleted files: `src/mean_field/systems/tmbg/plot.py`, `src/mean_field/systems/tmbg/validation.py`.
- Gross legacy LOC removed/thinned: 434.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_htg_paper_helper_tails

- Retired ignored-test-only HTG paper plotting, strong-coupling classification, Mao2025 response, and lightweight validation helper modules while keeping the maintained HTG model, HF, supercell, and topology surfaces.
- Deleted files: `src/mean_field/systems/htg/plot.py`, `src/mean_field/systems/htg/strong_coupling.py`, `src/mean_field/systems/htg/mao2025.py`, `src/mean_field/systems/htg/validation.py`.
- Gross legacy LOC removed/thinned: 781.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_atmg_tdbg_validation_tails

- Retired ignored-test-only ATMG and TDBG lightweight validation modules while keeping the maintained model, bands, topology, HF, artifact, and shift-current surfaces.
- Deleted files: `src/mean_field/systems/atmg/validation.py`, `src/mean_field/systems/tdbg/validation.py`.
- Gross legacy LOC removed/thinned: 264.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### thin_rlg_hbn_hf_overlap_density_helpers

- Thinned RnG/hBN HF helper code after old-vs-new characterization: the self-overlap block builder now delegates to the between-basis builder, zero-fill grid shifts use the core HF helper, and average reference density uses the core density helper.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 84.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### inline_rlg_hbn_hf_problem_callables

- Removed exported RnG/hBN HF initializer and density-builder wrapper classes that were only used by build_rlg_hbn_hf_problem; equivalent initializer and density-update closures now feed the core HF problem directly.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 31.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### archive_retire_htg_topology_surface

- Archived system topology files to ignored local_archive and retired the tracked HTG topology surface; kept only the HTG sublattice basis operator in the Hamiltonian layer and thinned ATMG/TMBG topology wrappers to direct generic-adapter calls.
- Deleted files: `src/mean_field/systems/htg/topology.py`.
- Gross legacy LOC removed/thinned: 401.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### thin_rlg_hbn_topology_sewing_adapter

- Archived the original RnG/hBN topology module and split HF microstate sewing into a small system gauge bridge, leaving topology.py as a thin boundary-sewing adapter around analysis.topology.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 81.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### archive_retire_system_plot_helpers

- Archived and removed RnG/hBN and TDBG system-local band-plot helper modules; tracked plotting should go through core plotting or workflow-level adapters rather than system paper/helper surfaces.
- Deleted files: `src/mean_field/systems/RnG_hBN/plot.py`, `src/mean_field/systems/tdbg/plot.py`.
- Gross legacy LOC removed/thinned: 160.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### archive_thin_atmg_band_adapter

- Archived ATMG bands.py and retired the mapped-spectrum audit payload from path/grid band APIs; tracked ATMG bands now expose only the thin generic path/grid adapter.
- Deleted files: `tests/test_atmg_fig3_path.py`.
- Gross legacy LOC removed/thinned: 158.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### archive_retire_rlg_hbn_paper_band_helpers

- Archived RnG/hBN bands.py and retired paper Fig.6 path, plot-manifest, and neutrality-energy helper exports; tracked RnG/hBN bands now expose only generic path/grid band adapters.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 133.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### retire_system_central_band_metric_wrappers

- Retired HTG/HTQG system-local central-band metric wrappers; callers now use the generic core band metric helper directly.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 31.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 5.

### archive_retire_rlg_hbn_paper_hf_runner

- Archived the RnG/hBN paper-HF runner to ignored local_archive and thinned the tracked devtool to metadata sidecar/archive compatibility helpers only; removed the dispatcher command so the paper runner is no longer active command surface.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1248.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 5 -> 3.

### archive_retire_rlg_hbn_parallel_merge_devtool

- Archived the RnG/hBN parallel paper-HF merge workflow and thinned the tracked devtool to metadata sidecar compatibility only; removed its dispatcher command.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 231.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 3 -> 3.

### archive_retire_rlg_hbn_finite_q_tdhf_devtool

- Archived and removed the RnG/hBN finite-q TDHF devtool command surface while leaving system adapter code/tests intact; finite-q production policy remains a system-layer task, not a tracked one-off runner.
- Deleted files: `src/mean_field/devtools/run_rlg_hbn_tdhf_finite_q.py`.
- Gross legacy LOC removed/thinned: 408.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 3 -> 2.

### archive_retire_crpa_direct_import_devtools

- Archived and removed the cRPA prep/validation wrappers that directly imported private TBG system modules, leaving the tracked tested cRPA chunk/merge devtools intact and avoiding cRPA algorithm changes.
- Deleted files: `src/mean_field/devtools/prepare_tbg_crpa_bm.py`, `src/mean_field/devtools/validate_tbg_crpa_artifact.py`.
- Gross legacy LOC removed/thinned: 368.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 2 -> 0.

### thin_system_band_adapters

- Thinned archived system bands modules to compact generic path/grid adapter shims; systems still supply Hamiltonians, grids, and diagonalizers while core.bands owns the loops/result containers.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 361.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### thin_system_topology_adapters

- Thinned archived system topology modules to compact generic adapter shims; systems retain only boundary-sewing bridges and metadata routing while analysis.topology owns Berry/Chern calculations.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 356.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### archive_retire_htqg_projected_hf

- Archived and removed the HTQG projected-HF implementation from tracked system surface; HTQG remains a noninteracting model with public run_hf explicitly unsupported.
- Deleted files: `src/mean_field/systems/htqg/hf.py`.
- Gross legacy LOC removed/thinned: 972.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### archive_retire_tdbg_hf_facade

- Archived and removed the legacy TDBG hf.py facade after projected-HF functionality had been split into maintained projected_hf_* modules; package exports now point directly at the maintained split facade.
- Deleted files: `src/mean_field/systems/tdbg/hf.py`.
- Gross legacy LOC removed/thinned: 480.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### thin_system_validation_adapters

- Archived historical system validation bodies and thinned tracked HTQG/RLG-hBN validation modules to cheap structural/Hermitian smoke checks only, avoiding grid/HF/paper-checkpoint recomputation in system surface.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 277.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### archive_retire_misc_devtools

- Archived and removed leftover one-off cRPA comparison and RLG/hBN Fig.6 prerequisite devtools from the tracked command surface; maintained tested chunk/merge and sidecar tools remain.
- Deleted files: `src/mean_field/devtools/compare_tbg_crpa_fig1e.py`, `src/mean_field/devtools/validate_rlg_hbn_fig6_prereqs.py`.
- Gross legacy LOC removed/thinned: 308.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### archive_retire_rlg_hbn_q0_tdhf_devtool

- Archived the dense RnG/hBN q=0 TDHF devtool runner and thinned the tracked module to schema sidecar and single-flavor shortcut compatibility helpers only; dispatcher command was removed.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 328.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### archive_retire_benchmark_sync_devtool

- Archived and removed the untested benchmark copy/sync devtool and dispatcher aliases from tracked command surface.
- Deleted files: `src/mean_field/devtools/sync_benchmarks.py`.
- Gross legacy LOC removed/thinned: 55.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### resolve_rlg_hbn_intraflavor_finite_q_tdhf_lane

- Resolved the RnG/hBN TDHF dirty lane by adding intraflavor finite-q assembly, overlap-shift closure helper, q/-q partner-structure residual reporting, package exports, and focused adapter tests; unrelated RnG/hBN HF ODA-control hunks were kept for a separate slice.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 0.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### resolve_rlg_hbn_oda_lambda_cap_lane

- Resolved the remaining RnG/hBN HF dirty lane by threading optional max_oda_lambda through run_rlg_hbn_hartree_fock and scan_rlg_hbn_ground_state into the shared core HF runner.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 0.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### retire_generated_refactor_report_json

- Removed the tracked generated refactor_surface_report.json snapshot; the Markdown report remains the durable tracked status artifact and JSON can be regenerated locally when needed.
- Deleted files: `docs/refactor_surface_report.json`.
- Gross legacy LOC removed/thinned: 1016.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_canonical_hf_backfill_devtool

- Split the canonical HF sidecar backfill devtool into scan/write/report/CLI modules while keeping the original import path and dispatcher command as a thin compatibility shim.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1559.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tbg_zero_field_runner_surface

- Split the TBG zero-field benchmark runner facade into focused helper/BM/B0/artifact/suite modules while preserving the public runners.py API and artifact monkeypatch hooks.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1345.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_htg_primitive_hf_adapter_surface

- Split the HTG primitive HF adapter facade into typed/reference/initialization/basis/interaction/runner/contract modules while preserving public mean_field_adapter imports and HF adapter registry paths.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 2301.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_public_hf_api_facade

- Split the public HF API module into private type, registry, sidecar, result, and dispatch modules while keeping mean_field.api.hf as the stable facade and preserving lazy adapter import paths.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1274.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tmbg_polshyn_hf_helper_surface

- Split the TMBG Polshyn doubled-cell HF helper into typed, canonical-contract, filling, and Wang-engine modules while keeping polshyn_supercell.py as the registered public facade.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1121.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tbg_zero_field_hf_helper_surface

- Split the TBG zero-field HF helper into basis/overlap, restricted, full, and diagnostics modules while keeping zero_field.hf as the public facade and leaving finite-field code untouched.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1249.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_htg_supercell_hf_surface

- Split the HTG folded-supercell HF helper into typed, geometry, basis/overlap, runner, and path I/O modules while preserving the supercell.py public facade and contract imports.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1342.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_rlg_hbn_hf_surface

- Split the RnG/hBN projected-HF surface into shared, typed, reference-density, basis/remote-average, interaction/path, and runner modules while preserving hf.py public facade imports for HF, cache, contracts, and TDHF adapters.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 2211.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_rlg_hbn_tdhf_surface

- Split the RnG/hBN TDHF adapter into support/type/orbital/pair/archive/q0/finite-q/dispatch modules while preserving tdhf.py public facade imports for package exports and archive loaders.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 2029.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### route_finite_field_hf_through_projected_core_apis

- Adapted full finite-B magnetic overlaps to generic HFOverlapBlockSet/build_projected_interaction_hamiltonian and build_projected_hf_kernel, then split finite_field.py into focused core finite-field modules while preserving the public facade.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1106.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tbg_finite_field_spectrum_surface

- Reused the core magnetic_r_orbit_positions helper for magnetic-translation orbit bookkeeping and split the TBG finite-field spectrum adapter into parameter, sweep, LL-matrix, Hamiltonian, and overlap modules while preserving spectrum.py facade imports.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 983.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

## Top 30 Python files under `src`

| Lines | Path |
|---:|---|
| 1267 | `src/mean_field/crpa/hf_interface.py` |
| 873 | `src/analysis/topology/quantum_geometry.py` |
| 805 | `src/analysis/response_derivative_gauge.py` |
| 788 | `src/mean_field/devtools/canonical_hf_backfill/_scan.py` |
| 786 | `src/mean_field/systems/RnG_hBN/cache.py` |
| 772 | `src/analysis/shift_current/core.py` |
| 736 | `src/mean_field/cli.py` |
| 735 | `src/mean_field/systems/tbg/zero_field/hf_contracts.py` |
| 710 | `src/mean_field/core/hf/tdhf.py` |
| 699 | `src/mean_field/systems/RnG_hBN/_hf_basis.py` |
| 686 | `src/mean_field/systems/RnG_hBN/hf_contracts.py` |
| 684 | `src/mean_field/api/_hf_sidecars.py` |
| 643 | `src/mean_field/systems/htg/supercell_contracts.py` |
| 627 | `src/mean_field/benchmarks.py` |
| 617 | `src/mean_field/core/hf/overlap.py` |
| 590 | `src/mean_field/systems/htg/_hf_contracts.py` |
| 565 | `src/analysis/topology/core.py` |
| 553 | `src/mean_field/systems/RnG_hBN/_hf_interaction_path.py` |
| 543 | `src/mean_field/systems/RnG_hBN/_hf_runner.py` |
| 528 | `src/mean_field/systems/tbg/zero_field/artifacts.py` |
| 498 | `src/mean_field/crpa/validation.py` |
| 492 | `src/mean_field/systems/RnG_hBN/screening.py` |
| 485 | `src/mean_field/systems/RnG_hBN/_tdhf_finite_q.py` |
| 456 | `src/mean_field/systems/tbg/zero_field/model.py` |
| 454 | `src/mean_field/crpa/workflow.py` |
| 445 | `src/mean_field/systems/tdbg/projected_hf_state.py` |
| 435 | `src/mean_field/systems/tmbg/_polshyn_wang.py` |
| 434 | `src/mean_field/crpa/diagnostics.py` |
| 431 | `src/mean_field/systems/htg/_hf_initialization.py` |
| 427 | `src/mean_field/systems/tbg/zero_field/_hf_full.py` |

## Direct private-system imports in workflow surfaces

| Path | Line | Import |
|---|---:|---|

## Repeated module-family line counts

### `bands.py`

Total lines: 178

| Lines | Path |
|---:|---|
| 35 | `src/mean_field/systems/RnG_hBN/bands.py` |
| 35 | `src/mean_field/systems/htqg/bands.py` |
| 28 | `src/mean_field/systems/atmg/bands.py` |
| 28 | `src/mean_field/systems/htg/bands.py` |
| 26 | `src/mean_field/systems/tdbg/bands.py` |
| 26 | `src/mean_field/systems/tmbg/bands.py` |

### `validation.py`

Total lines: 144

| Lines | Path |
|---:|---|
| 93 | `src/mean_field/systems/htqg/validation.py` |
| 51 | `src/mean_field/systems/RnG_hBN/validation.py` |

### `topology.py`

Total lines: 172

| Lines | Path |
|---:|---|
| 61 | `src/mean_field/systems/RnG_hBN/topology.py` |
| 61 | `src/mean_field/systems/tdbg/topology.py` |
| 25 | `src/mean_field/systems/atmg/topology.py` |
| 25 | `src/mean_field/systems/tmbg/topology.py` |

## Repeated symbol names

### `bands.py`

| Symbol | Count | Paths |
|---|---:|---|
| `_diagonalize` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `_make_diagonalizer` | 4 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `_prepare_band_diagonalizer` | 2 | `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py` |
| `compute_bands_along_path` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `compute_bands_on_grid` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |

### `validation.py`

| Symbol | Count | Paths |
|---|---:|---|
| `_finite` | 2 | `src/mean_field/systems/RnG_hBN/validation.py`, `src/mean_field/systems/htqg/validation.py` |

### `topology.py`

| Symbol | Count | Paths |
|---|---:|---|
| `compute_topology_from_eigenvectors` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `compute_topology_from_grid_result` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `compute_topology_on_grid` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |
| `grid_builder` | 4 | `src/mean_field/systems/RnG_hBN/topology.py`, `src/mean_field/systems/atmg/topology.py`, `src/mean_field/systems/tdbg/topology.py`, `src/mean_field/systems/tmbg/topology.py` |

## Phase 2 cleanup gates

- No PR should claim cleanup if it only adds wrappers without deleting or thinning legacy paths.
- Each slice must update this report with before/after LOC and remaining old entry points.
- Physics-heavy migrations require focused parity tests before deleting old implementations.
- cRPA/HF bridge changes are deferred until the known cRPA bug is isolated.

## Update: cleanup plan / order-parameter / optical-response / cRPA bridge / API registry pass

Commits in this pass:

- `ddebb13 Extract common order-parameter helpers`
- `9b374cc Add optical response package facade`
- `ef8ba6f Split cRPA HF bridge surface`
- `e6eab50 Add explicit CRPA and TDHF adapter registries`

### Current summary after this pass

- Tracked text lines: 65236
- Tracked Python lines: 61583
- Tracked Julia lines: 826
- `src` Python files: 267
- `src` Python lines: 54917
- Files over 1000 lines: 0

### git hygiene / examples policy

- Removed the dangerous bare `tests/` ignore rule from `.gitignore`.
- Public tests are now trackable by default.
- Ignored only local/generated test payloads: `tests/local/`, `tests/internal/`, `tests/slow/`, `tests/data/generated/`, `tests/**/*.npz`, `tests/**/*.npy`.
- Added local examples ignores: `examples/local/`, `examples/scratch/`, `examples/oneoff/`.
- No tracked `examples/` directory exists in this worktree, so no examples inventory was generated in this slice.

### Order-parameter extraction

Added common package:

- `src/analysis/order_parameters/`
  - `schema.py`
  - `density.py`
  - `flavor.py`
  - `coherence.py`
  - `translation.py`
  - `classification.py`
  - `adapters.py`

Routed existing wrappers through the common helpers while preserving import paths:

- `mean_field.systems.tdbg.projected_hf_state._numeric_order_parameters`
- `mean_field.systems.tdbg.projected_hf_state.tdbg_order_parameters`
- `mean_field.systems.tmbg._polshyn_wang.translation_order_parameters`
- `mean_field.core.hf._finite_field_kernel.calculate_valley_spin_order_parameters`

Equivalence validation:

- `tests/test_order_parameters.py`
- focused gate: `25 passed` for order-parameter/TDBG/tMBG/API tests.

### Optical-response package boundary

Added forward-facing package:

- `src/analysis/optical_response/`
  - `gauge.py`
  - `components.py`
  - `conventions.py`
  - `occupations.py`
  - `transitions.py`
  - `shift_current.py`
  - `heatmap.py`
  - `toy_models/`

Compatibility status:

- Historical paths `analysis.response_derivative_gauge` and `analysis.shift_current` remain valid.
- TDBG shift-current adapter now imports the common API through `analysis.optical_response.*`.
- Retired hTG/TBG paper response workspaces were not restored.

Validation:

- `tests/test_optical_response_api.py`
- focused gate: `13 passed` for optical-response/TDBG/API tests.

### cRPA/HF bridge split

Split the only remaining >1000-line tracked Python file:

| Before | After |
|---:|---:|
| `src/mean_field/crpa/hf_interface.py`: 1267 lines | `src/mean_field/crpa/hf_interface.py`: 15-line compatibility facade |

New implementation modules:

- `src/mean_field/crpa/hf_bridge/density.py`
- `src/mean_field/crpa/hf_bridge/split_scheme.py`
- `src/mean_field/crpa/hf_bridge/kernels.py`
- `src/mean_field/crpa/hf_bridge/energy.py`
- `src/mean_field/crpa/hf_bridge/runner.py`

Compatibility helpers restored in the TBG zero-field facade for cRPA imports:

- `_hex_shell_contains`
- `_precompute_overlap_screening`
- `_screened_coulomb_matrix`
- `_with_tbg_overlap_screening`

Convention validation added:

- `tests/test_crpa_hf_bridge_split.py`
- verifies `D = P - 1/2 I` roundtrip and bare component sum identity against the generic projected interaction builder with tolerance `1e-12`.
- focused gate: `28 passed` for cRPA bridge/API tests.

### Public API registry connections

cRPA:

- Added `CRPAAdapterInfo` and registry helpers:
  - `list_crpa_adapters`
  - `get_crpa_adapter_info`
  - `resolve_crpa_adapter`
- Registered `tbg_workflow`, delegating to `mean_field.crpa.workflow.compute_crpa` only when caller provides explicit TBG inputs.
- `compute_crpa(...)` still preserves object-hook dispatch and refuses silent production parameter inference.

TDHF:

- Added `TDHFAdapterInfo` and registry helpers:
  - `list_tdhf_adapters`
  - `get_tdhf_adapter_info`
  - `resolve_tdhf_adapter`
- Registered RLG/hBN adapters:
  - `rlg_hbn_q0`
  - `rlg_hbn_finite_q`
- `run_tdhf(...)` requires explicit raw HF run + canonical HF state/result for RLG/hBN adapters; it does not load archives implicitly.

Validation:

- `tests/test_public_api_registries.py`
- focused gate: `29 passed` for API registry tests.

### Full gate

After the API registry slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result on `test001`: `206 passed`.

## Update: model registry and TDBG workflow extraction

Commits in this continuation:

- `a00803f Convert model API to adapter registry`
- `9b8202a Extract TDBG projected HF CLI workflow`

### Current summary after this continuation

- Tracked text lines: 65532
- Tracked Python lines: 61735
- Tracked Julia lines: 826
- `src` Python files: 268
- `src` Python lines: 55058
- Files over 1000 lines: 0

### Model API registry

- Converted `mean_field.api.models.make_model` from one long hard-coded branch list into a small adapter registry.
- Added public registry helpers:
  - `ModelAdapterInfo`
  - `list_model_adapters`
  - `get_model_adapter_info`
  - `resolve_model_adapter`
- Preserved existing public names/aliases and constructor defaults for HTG, HTQG, RLG/hBN, TBG, TDBG, tMBG, and ATMG.
- Updated `docs/api/model_api.md` and `tests/test_public_api_registries.py`.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/api tests/test_public_api_registries.py
PYTHONPATH=src pytest -q tests/test_public_api_registries.py tests/test_api_imports.py tests/test_api_hf_adapters.py
```

Result: `30 passed`.

### TDBG projected-HF workflow extraction

- Moved TDBG projected-HF JSON parsing, validation, dry-run normalization, output freshness check, run dispatch, and artifact save logic from `src/mean_field/cli.py` to `src/mean_field/workflows/tdbg_projected_hf.py`.
- `cli.py` now keeps the command-line parser and a thin `cmd_tdbg_projected_hf(...)` delegating to `run_tdbg_projected_hf_workflow(...)`.
- The CLI still dependency-injects `make_model`, `run_hf`, and the compute guard so existing tests and monkeypatches remain compatible.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/cli.py src/mean_field/workflows tests/test_cli_tdbg_projected_hf.py
PYTHONPATH=src pytest -q tests/test_cli_tdbg_projected_hf.py tests/test_api_imports.py
```

Result: `15 passed`.

### Full gate

After the workflow extraction slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result on `test001`: `207 passed`.

## Update: public bands API KGrid/KPath inputs

Commit in this continuation:

- `faa706e Accept explicit KGrid and KPath band inputs`

### Current summary after this continuation

- Tracked text lines: 65717
- Tracked Python lines: 61858
- Tracked Julia lines: 826
- `src` Python files: 268
- `src` Python lines: 55135
- Files over 1000 lines: 0

### Bands API change

- `mean_field.api.compute_bands` now accepts explicit public `KGrid` and `KPath` dataclass inputs.
- `KGrid` with explicit `kvec`/`frac` is evaluated by direct model `diagonalize(...)`, so non-square explicit grids no longer go through the old square-only `bands_on_grid(mesh_size)` facade path.
- `KPath` is converted to the core `KPath` shape and evaluated through the generic path-band loop.
- Existing `grid_mesh=int` and square `grid_mesh=(n,n)` behavior is preserved.
- Non-square tuple inputs now produce a clear instruction to pass an explicit `KGrid`.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/api/bands.py tests/test_api_bands.py
PYTHONPATH=src pytest -q tests/test_api_bands.py tests/test_api_imports.py
```

Result: `13 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: retired RLG/hBN devtool helper extraction

Commit in this continuation:

- `6414558 Move retired RLG hBN helpers to workflows`

### Current summary after this continuation

- Tracked text lines: 65745
- Tracked Python lines: 61836
- Tracked Julia lines: 826
- `src` Python files: 266
- `src` Python lines: 55113
- Files over 1000 lines: 0

### Devtool surface change

Moved lightweight compatibility helpers from retired command-surface modules into `src/mean_field/workflows/rlg_hbn.py` and updated tests to import the workflow helpers directly.

Deleted retired tracked command files:

- `src/mean_field/devtools/run_rlg_hbn_paper_hf.py`
- `src/mean_field/devtools/merge_rlg_hbn_parallel_hf.py`
- `src/mean_field/devtools/run_rlg_hbn_tdhf_q0.py`

Preserved helper functionality under explicit workflow names:

- `write_rlg_hbn_paper_hf_contract_sidecars`
- `save_rlg_hbn_paper_hf_state_archive`
- `load_rlg_hbn_paper_hf_archive_density`
- `write_rlg_hbn_parallel_hf_merge_contract_sidecars`
- `rlg_hbn_tdhf_q0_shortcut_decision`
- `write_rlg_hbn_tdhf_q0_contract_sidecars`

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/workflows src/mean_field/devtools tests/test_artifact_schema.py tests/test_rlg_hbn_tdhf_adapter.py
PYTHONPATH=src pytest -q tests/test_artifact_schema.py tests/test_rlg_hbn_tdhf_adapter.py
```

Result: `43 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: canonical HF backfill scanner split

Commit in this continuation:

- `abc1d94 Split canonical HF backfill scanner`

### Current summary after this continuation

- Tracked text lines: 65824
- Tracked Python lines: 61863
- Tracked Julia lines: 826
- `src` Python files: 270
- `src` Python lines: 55140
- Files over 1000 lines: 0

### Scanner split

`src/mean_field/devtools/canonical_hf_backfill/_scan.py` was reduced from a 797-line implementation module to a 54-line compatibility facade plus public `scan_backfill_candidates(...)` entrypoint.

New split modules:

- `src/mean_field/devtools/canonical_hf_backfill/_scan_utils.py`
- `src/mean_field/devtools/canonical_hf_backfill/_scan_contracts.py`
- `src/mean_field/devtools/canonical_hf_backfill/_scan_discovery.py`
- `src/mean_field/devtools/canonical_hf_backfill/_scan_classify.py`

This was a mechanical no-algorithm-change refactor.  Existing imports from `_scan`, including `_mapping` used by `_cli.py`, remain supported.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/devtools/canonical_hf_backfill tests/test_backfill_canonical_hf_sidecars.py
PYTHONPATH=src pytest -q tests/test_backfill_canonical_hf_sidecars.py
```

Result: `15 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: optical-response implementation move

Commit in this continuation:

- `b9c92aa Move optical response implementations into package`

### Current summary after this continuation

- Tracked text lines: 65870
- Tracked Python lines: 61863
- Tracked Julia lines: 826
- `src` Python files: 270
- `src` Python lines: 55140
- Files over 1000 lines: 0

### Optical-response implementation move

The earlier `analysis.optical_response` package was upgraded from facade-only to owning the implementations:

- `src/analysis/optical_response/gauge.py` now contains the gauge-safe derivative implementation formerly in `src/analysis/response_derivative_gauge.py`.
- `src/analysis/optical_response/shift_current.py` now contains the common shift-current implementation formerly in `src/analysis/shift_current/core.py`.
- Historical import paths remain as compatibility shims:
  - `src/analysis/response_derivative_gauge.py`
  - `src/analysis/shift_current/core.py`

No formulas were changed; this was a mechanical import-boundary migration.  `analysis.optical_response` internal submodules now import the package-local implementation rather than routing through the old shims.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis/optical_response src/analysis/response_derivative_gauge.py src/analysis/shift_current tests/test_optical_response_api.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py tests/test_api_bands.py
```

Result: `15 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: optical-response shift-current module split

Commit in this continuation:

- `1139cfa Split optical response shift-current modules`

### Current summary after this continuation

- Tracked text lines: 65977
- Tracked Python lines: 61925
- Tracked Julia lines: 826
- `src` Python files: 270
- `src` Python lines: 55202
- Files over 1000 lines: 0

### Shift-current module split

After moving the implementation into `analysis.optical_response`, the shift-current implementation was split so the package module names now match their responsibilities:

- `components.py`: component/axis parsing and labels.
- `conventions.py`: response convention bundles and physical constants.
- `occupations.py`: Fermi occupation helper.
- `heatmap.py`: Lorentzian broadening, spectrum accumulation, and Fermi/omega heatmap helpers.
- `shift_current.py`: tensor precomputation and transition/pair kernel aggregation; still re-exports the public symbols for compatibility.

Line counts after split:

- `shift_current.py`: 523
- `components.py`: 86
- `conventions.py`: 76
- `occupations.py`: 22
- `heatmap.py`: 158

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis/optical_response src/analysis/shift_current tests/test_optical_response_api.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py tests/test_api_bands.py
```

Result: `15 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: optical-response gauge derivative module split

Commit in this continuation:

- `18e12d7 Split optical response gauge derivative modules`

### Current summary after this continuation

- Tracked text lines: 66073
- Tracked Python lines: 61970
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55247
- Files over 1000 lines: 0

### Gauge derivative module split

`analysis.optical_response.gauge` was reduced from an 805-line implementation module to a 43-line public facade/re-export module.  The implementation now lives in smaller package-local modules:

- `gauge_data.py`: dataclasses for Hamiltonian-gauge and generalized-derivative payloads.
- `gauge_primitives.py`: energy denominators, degeneracy groups, random block gauges, covariant gauge rotations, subspace trace, eigenbasis rotation.
- `gauge_hamiltonian.py`: Hamiltonian-gauge ingredient construction.
- `gauge_derivatives.py`: covariant/generalized derivative formulas and WannierBerri internal `Imn` port.
- `gauge_shift.py`: shift-vector/integrand and Wilson-link phase derivative helpers.

No formulas, signs, cutoffs, or denominator policies were changed; this was a mechanical module-boundary split.  Historical `analysis.response_derivative_gauge` remains a shim through `analysis.optical_response.gauge`.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis/optical_response src/analysis/response_derivative_gauge.py src/analysis/shift_current tests/test_optical_response_api.py tests/test_api_imports.py tests/test_api_bands.py tests/test_rlg_hbn_tdhf_adapter.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py tests/test_api_bands.py tests/test_rlg_hbn_tdhf_adapter.py
```

Result: `37 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `210 passed`.

## Update: optical-response package exports and breadcrumbs

Commits in this continuation:

- `59cf51e Update optical response breadcrumbs`
- `d11a3f9 Complete optical response package exports`

### Current summary after this continuation

- Tracked text lines: 66096
- Tracked Python lines: 61937
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55198
- Files over 1000 lines: 0

### Breadcrumb and export updates

- Updated root and `src/analysis/` AGENTS breadcrumbs plus architecture/response docs to name `analysis.optical_response` as the implementation owner and old `response_derivative_gauge.py` / `shift_current/` paths as compatibility shims.
- Expanded `analysis.optical_response.__all__` to aggregate public symbols from split component, convention, gauge, heatmap, occupation, and shift-current modules.
- Added smoke coverage for split-module package exports including degeneracy grouping, random block gauges, WannierBerri group trace, and named conventions.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis/optical_response tests/test_optical_response_api.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py tests/test_api_bands.py
```

Result: `16 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `211 passed`.

## Update: optical-response toy-model ownership

Commit in this continuation:

- `b16e9e5 Move optical toy models into response package`

### Current summary after this continuation

- Tracked text lines: 66151
- Tracked Python lines: 61952
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Toy-model boundary update

The gapped-SLG optical-response toy model implementation now lives under `analysis.optical_response.toy_models`.  Historical `analysis.shift_current.toy_models` paths remain compatibility shims.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis/optical_response/toy_models src/analysis/shift_current/toy_models tests/test_optical_response_api.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py
```

Result: `14 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `212 passed`.

## Update: production validation backfill CLI examples

Commit in this continuation:

- `4382dc1 Fix backfill validation runbook CLI examples`

### Current summary after this continuation

- Tracked text lines: 66216
- Tracked Python lines: 61970
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Runbook/API mismatch fixed

`docs/production_validation_backlog.md` now matches the current canonical sidecar backfill CLI: roots are positional arguments, not `--root`.  The runbook also includes an explicit `--no-archives` dry-run example for fast metadata-only inventory.

A regression test covers the documented positional-root plus `--no-archives` dry-run shape and verifies report files are written without historical mutation.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/devtools/canonical_hf_backfill tests/test_backfill_canonical_hf_sidecars.py
PYTHONPATH=src pytest -q tests/test_backfill_canonical_hf_sidecars.py
```

Result: `16 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `213 passed`.

## Update: response compatibility docs

Commit in this continuation:

- `21242f7 Update response compatibility docs`

### Current summary after this continuation

- Tracked text lines: 66296
- Tracked Python lines: 61970
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Compatibility docs update

- Updated README and `src/mean_field/systems/AGENTS.md` to point new response work at `analysis.optical_response`, while documenting `analysis.response_derivative_gauge` / `analysis.shift_current` as compatibility paths.
- Updated `analysis.shift_current` README and package docstring to describe it as a compatibility re-export surface.
- Updated optical-response internal docstrings to reference `analysis.optical_response.gauge` directly.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/analysis src/mean_field/systems tests/test_optical_response_api.py
PYTHONPATH=src pytest -q tests/test_optical_response_api.py tests/test_api_imports.py
```

Result: `14 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `213 passed`.

## Update: non-TDHF/cRPA validation boundary and TMBG Polshyn preflight

Commit in this continuation:

- `c89fafc Record non TDHF CRPA validation boundary`

### Current summary after this continuation

- Tracked text lines: 66309
- Tracked Python lines: 61970
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Validation boundary update

The active validation boundary is now documented as:

- TDHF and cRPA validation/code paths remain out of scope unless explicitly requested.
- Non-TDHF/non-cRPA validation and cleanup gates may proceed after local self-checks of command, output path, and expected runtime.

### TMBG Polshyn software preflight

A 1x1 explicit-config TMBG Polshyn public `run_hf(...)` preflight was run on `test001` with no writes to historical `results/`.

Output:

```text
/data/home/ziyuzhu/tmp/mean_field_validation_tmbg_polshyn_92f4e98_20260622_003023/summary.json
```

Summary:

```text
result_model: tmbg_polshyn
has_canonical_run_result: true
best_seed: 5
workflow metadata: tmbg.polshyn_wang.explicit_config
```

This is software/API readiness evidence only, not paper-level production validation.

## Update: HTG and HTQG software validation gates

Commit in this continuation:

- `9195526 Record HTG and HTQG software gates`

### Current summary after this continuation

- Tracked text lines: 66372
- Tracked Python lines: 61970
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Software gates recorded

Non-TDHF/non-cRPA lightweight gates run on `test001`:

```text
pytest -q tests/test_htg_supercell.py tests/test_htg_supercell_contract_adapter.py
# 10 passed

pytest -q tests/test_htqg_model.py tests/test_api_imports.py
# 14 passed
```

These are software/API readiness gates only, not production/paper-level validation.

## Update: TMBG Polshyn metadata-only HF save/load gate

Commit in this continuation:

- `e308aa0 Test TMBG Polshyn metadata-only HF save`

### Current summary after this continuation

- Tracked text lines: 66472
- Tracked Python lines: 61988
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### TMBG Polshyn acceptance covered

Added a public test that runs the explicit TMBG Polshyn `run_hf(...)` smoke, saves the result with `canonical_payload="metadata_only"`, reloads it through `mean_field.api.load_result(...)`, and verifies that no `canonical_hf_arrays.npz` payload is written.

Also documented the canonical sidecar staging self-check: the existing 48 RLG/hBN staged sidecars rely on the `RnG_hBN.tdhf` archive loader, so they remain staged while TDHF is out of scope.

Focused validation:

```bash
PYTHONPATH=src pytest -q tests/test_tmbg_polshyn_hf_readiness.py
```

Result: `10 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `214 passed`.

## Update: HTQG explicit missing-HF-adapter gate

Commit in this continuation:

- `d325acf Document HTQG missing HF adapter gate`

### Current summary after this continuation

- Tracked text lines: 66529
- Tracked Python lines: 61999
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### HTQG public HF boundary

Added a public HF-adapter gate that documents the current HTQG state: `HTQGModel` has no registered HF adapter, and `run_hf(HTQGModel, HFConfig(...))` must fail explicitly with the frozen public-API message instead of inferring projected-HF physics.

This preserves the boundary until an explicit HTQG projected-HF adapter/workflow is designed with target Hamiltonian parameters, projected band/subspace, reference density, and path convention.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/api src/mean_field/systems/htqg tests/test_api_hf_adapters.py tests/test_htqg_model.py
PYTHONPATH=src pytest -q tests/test_api_hf_adapters.py tests/test_htqg_model.py
```

Result: `20 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `215 passed`.

## Update: HTG supercell metadata-only HF save/load gate

Commit in this continuation:

- `834b132 Test HTG supercell metadata-only HF save`

### Current summary after this continuation

- Tracked text lines: 66584
- Tracked Python lines: 62008
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### HTG supercell canonical sidecar acceptance

Extended the explicit HTG supercell public `run_hf(...)` gate to save the result with `canonical_payload="metadata_only"`, reload it through `mean_field.api.load_result(...)`, and verify that no dense `canonical_hf_arrays.npz` payload is produced.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/api src/mean_field/systems/htg tests/test_api_hf_adapters.py
PYTHONPATH=src pytest -q tests/test_api_hf_adapters.py
```

Result: `16 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `215 passed`.

## Update: public HF metadata-only sidecar coverage

Commit in this continuation:

- `f094c75 Cover public HF metadata-only sidecars`

### Current summary after this continuation

- Tracked text lines: 66648
- Tracked Python lines: 62016
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Metadata-only sidecar coverage

Extended the public HF adapter tests to reuse one metadata-only save/load assertion across explicit non-TDHF/non-cRPA `run_hf(...)` adapters:

- `tbg_zero_field`
- `tdbg`
- `htg`
- `htg_supercell`
- `rlg_hbn`

The shared assertion verifies that `HFResult.save(..., canonical_payload="metadata_only")` writes `canonical_hf_run_result.json`, remains loadable via `mean_field.api.load_result(...)`, and does not write `canonical_hf_arrays.npz`.

Focused validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/api tests/test_api_hf_adapters.py
PYTHONPATH=src pytest -q tests/test_api_hf_adapters.py
```

Result: `16 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `215 passed`.

## Update: HF API docs for Polshyn and sidecar payloads

Commit in this continuation:

- `1ee5b1b Update HF API docs for Polshyn and sidecars`

### Current summary after this continuation

- Tracked text lines: 66692
- Tracked Python lines: 62016
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### HF API documentation sync

Updated `docs/api/hf_api.md` to match the current public HF surface:

- added `htqg` to the model-facade list while preserving the explicit missing-HF-adapter boundary;
- documented `run_hf(..., tmbg_polshyn_config=PolshynRunHFConfig(...))` and `tmbg_polshyn_explicit_run_hf`;
- documented TMBG Polshyn `HFResult` semantics;
- clarified that `HFResult.save(...)` defaults to `canonical_payload="metadata_only"`, does not write `canonical_hf_arrays.npz`, and writes dense canonical arrays only with explicit `canonical_payload="arrays"`.

Focused validation:

```bash
PYTHONPATH=src pytest -q tests/test_api_hf_adapters.py tests/test_api_imports.py tests/test_tmbg_polshyn_hf_readiness.py
```

Result: `36 passed`.

Full gate on `test001` after this slice:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
```

Result: `215 passed`.

## Update: HF API model facade list

Commit in this continuation:

- `f1f7172 Fix HF API model facade list`

### Current summary after this continuation

- Tracked text lines: 66711
- Tracked Python lines: 62016
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### Documentation consistency

`docs/api/hf_api.md` now lists the public model facades consistently with the model registry, including `tbg` as well as `htqg`.

## Update: HTG supercell preflight and artifact payload docs

Commits in this continuation:

- `51db277 Record HTG supercell software preflight`
- `a79c0db Document HF canonical artifact payload modes`

### Current summary after this continuation

- Tracked text lines: 66773
- Tracked Python lines: 62016
- Tracked Julia lines: 826
- `src` Python files: 275
- `src` Python lines: 55205
- Files over 1000 lines: 0

### HTG supercell software preflight

Ran a 1x1 explicit HTG supercell public `run_hf(...)` preflight on `test001` with output only under `/data/home/ziyuzhu/tmp`:

```text
/data/home/ziyuzhu/tmp/mean_field_validation_htg_supercell_7d8ac74_20260622_100932/summary.json
```

Summary:

```text
result_model: htg_supercell
has_canonical_run_result: true
loaded_canonical_sidecar: true
metadata_only_arrays_absent: true
primitive_nu: 3.5
supercell_area_ratio: 2
filling_from_density: 3.5000000000000018
workflow metadata: htg.supercell.explicit_config.preflight
```

This is software/API readiness evidence only, not a converged fractional-filling production run.

### Artifact payload documentation

Updated `docs/api/artifact_api.md` to document the `HFResult.save(...)` canonical payload modes: `canonical_payload="metadata_only"` is the default and writes only `canonical_hf_run_result.json`, while dense canonical arrays require explicit `canonical_payload="arrays"`.

## Update: untrack unused HTQG system helpers

Commit in this continuation:

- `36e655b Untrack unused HTQG system helpers`

### Current summary after this continuation

- Tracked text lines: 66377
- Tracked Python lines: 61576
- Tracked Julia lines: 826
- `src` Python files: 272
- `src` Python lines: 54765
- `src/mean_field/systems` Python lines: 30582
- Files over 1000 lines: 0

### Systems surface cleanup

Removed three HTQG helper modules from the tracked package surface after AST/import audit showed they were not public package dependencies:

- `src/mean_field/systems/htqg/chiral.py`
- `src/mean_field/systems/htqg/symmetry.py`
- `src/mean_field/systems/htqg/validation.py`

The removed files were copied to ignored local archive before `git rm`:

```text
local_archive/retired_surface/htqg_unused_helpers_20260622/
```

`tests/test_htqg_model.py` now keeps the small Hamiltonian/time-reversal smoke inline instead of importing a package-level validation helper. This preserves public HTQG model coverage without keeping test-only validation code in `systems/`.

Validation:

```bash
PYTHONPATH=src python -m compileall -q src/mean_field/systems/htqg tests/test_htqg_model.py tests/test_api_hf_adapters.py
PYTHONPATH=src pytest -q tests/test_htqg_model.py tests/test_api_hf_adapters.py -k "htqg or model_registry"
# 5 passed, 15 deselected

PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 215 passed
```

## Update: archive cRPA, broad tests, devtools, and benchmark workflow surface

Commit in this continuation:

- `fb2ca54 Archive cRPA tests and workflow surfaces`

### Scope archived to ignored local surface

The user explicitly authorized making cRPA and most tests non-tracked for the current cleanup direction. The following tracked surfaces were copied to ignored local archives before `git rm`:

```text
local_archive/retired_surface/crpa_untracked_20260622/
local_archive/retired_surface/tests_untracked_20260622/
local_archive/retired_surface/devtools_untracked_20260622/
local_archive/retired_surface/benchmark_workflow_untracked_20260622/
```

Removed from tracked package/API surface:

- `src/mean_field/crpa/`
- `src/mean_field/api/crpa.py`
- `docs/api/crpa_api.md`
- cRPA chunk/merge devtools
- `src/mean_field/devtools/`
- package CLI/workflow/benchmark runner glue:
  - `src/mean_field/cli.py`
  - `src/mean_field/benchmarks.py`
  - `src/mean_field/workflows/`
- TBG zero-field benchmark runner/artifact/plotting command surface:
  - `src/mean_field/systems/tbg/zero_field/runners.py`
  - `src/mean_field/systems/tbg/zero_field/hf_runners.py`
  - `src/mean_field/systems/tbg/zero_field/artifacts.py`
  - `src/mean_field/systems/tbg/zero_field/plotting.py`
  - `src/mean_field/systems/tbg/zero_field/path_advisor.py`
  - split `_runners_*` helper modules
- broad local regression tests, keeping only a minimal tracked smoke/contract set.

The remaining tracked tests are:

```text
tests/test_api_hf_adapters.py
tests/test_api_imports.py
tests/test_core_contracts.py
tests/test_core_hf_layering.py
tests/test_htg_supercell.py
tests/test_htqg_model.py
tests/test_public_api_registries.py
tests/test_tmbg_polshyn_hf_readiness.py
```

### Public surface adjustments

- `mean_field.api` no longer exports cRPA facade symbols.
- `mean_field.crpa`, `mean_field.devtools`, `mean_field.workflows`, `mean_field.cli`, and `mean_field.benchmarks` are absent from the tracked package surface.
- `scripts/mean_field_tools.py` is a placeholder dispatcher with no Python commands registered.
- `scripts/mean_field_tools.jl` and `scripts/submit_mean_field.sbatch` remain tracked.
- `src/mean_field/systems/tbg/zero_field/__init__.py` now exports only core model/HF/path/overlap/adapter helpers rather than benchmark runner/artifact/plotting helpers.

### Current summary after this continuation

- Tracked text lines: 48641
- Tracked Python lines: 43922
- Tracked Julia lines: 826
- `src` Python files: 214
- `src` Python lines: 42380
- `tests` Python lines: 1481
- `src/mean_field/systems` Python lines: 27694
- Files over 1000 lines: 0

Validation:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 55 passed
```

Import boundary smoke:

```text
mean_field imports successfully
mean_field.api has no compute_crpa export
mean_field.crpa absent as expected
mean_field.devtools absent as expected
mean_field.workflows absent as expected
mean_field.cli absent as expected
mean_field.benchmarks absent as expected
```

## Update: archive topology/Berry helper surface to reach 4w src target

Commit in this continuation:

- `edfa818 Archive topology helper surface`

### Scope archived to ignored local surface

The previous unified topology/Berry-geometry package and remaining system topology convenience wrappers were copied to ignored local archive before `git rm`:

```text
local_archive/retired_surface/topology_untracked_20260622/
```

Removed from tracked package/docs surface:

- `src/analysis/topology/`
- `docs/topology_framework.md`
- system topology convenience wrappers:
  - `src/mean_field/systems/RnG_hBN/topology.py`
  - `src/mean_field/systems/atmg/topology.py`
  - `src/mean_field/systems/tdbg/topology.py`
  - `src/mean_field/systems/tmbg/topology.py`
- `topology_on_grid(...)` model convenience methods and topology re-exports from system facades.

Kept in tracked code:

- RnG/hBN sewing helpers, with a local `SewingTransform` callable type alias instead of importing `analysis.topology`.
- TDBG projected-HF `translation_srcmap(...)`, moved locally into `projected_hf_geometry.py` because projected-HF geometry still uses that q-site shift map independently of topology/Chern wrappers.

### Current summary after this continuation

- Tracked text lines: 46135
- Tracked Python lines: 41474
- Tracked Julia lines: 826
- `src` Python files: 205
- `src` Python lines: 39932
- `tests` Python lines: 1481
- `src/mean_field/systems` Python lines: 27432
- Files over 1000 lines: 0

Validation:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 55 passed
```

## Update: archive zero-reference utility surface after 4w pass

Commit in this continuation:

- `121b8e9 Archive zero-reference utility surface`

### Scope archived to ignored local surface

After the cRPA/tests/devtools/topology cleanup reached the 4w `src` target, a fresh AST/import audit found a small set of tracked modules with no tracked Python import references. These were copied to ignored local archive before `git rm`:

```text
local_archive/retired_surface/zero_ref_surface_20260622/
```

Removed from tracked package surface:

- `src/mean_field/runtime.py`
- `src/mean_field/paths.py`
- `src/mean_field/plotting.py`
- `src/mean_field/core/plotting/`
- `src/mean_field/systems/tdbg/artifacts.py`
- `src/mean_field/systems/tbg/zero_field/supercell.py`
- `src/analysis/optical_response/transitions.py`

Also updated local guidance/docs so the tracked surface no longer points at archived topology or plotting helpers.

### Current summary after this continuation

- Tracked text lines: 45331
- Tracked Python lines: 40678
- Tracked Julia lines: 826
- `src` Python files: 197
- `src` Python lines: 39136
- `tests` Python lines: 1481
- `src/mean_field/systems` Python lines: 27234
- Files over 1000 lines: 0

Validation:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 55 passed
```

## Update: archive facade-only helper surface

Commit in this continuation:

- `0b70ace Archive facade-only helper surface`

### Scope archived to ignored local surface

A follow-up audit identified helpers that were only re-exported through facades or covered by now-local tests, with no tracked package caller. These were copied to ignored local archive before `git rm`:

```text
local_archive/retired_surface/facade_only_helpers_20260622/
```

Removed from tracked package surface:

- `src/mean_field/api/validation.py` (`validate_fig6_screening_checkpoints` heavy RLG/hBN checkpoint helper)
- `src/mean_field/systems/tdbg/shift_current.py` (TDBG/Joya system response adapter; common `analysis.optical_response` remains tracked)
- `src/mean_field/systems/htqg/commensurate.py` (paper-local commensurate-geometry helper)
- `src/mean_field/systems/RnG_hBN/validation.py` (smoke/paper-checkpoint validation facade)
- `src/mean_field/systems/htg/chiral.py` (small chiral-limit convenience helper)

Updated public/system facades and smoke tests accordingly. HF contracts, explicit `run_hf(...)` adapters, TMBG Polshyn facade, and TDHF/core code were left intact.

### Current summary after this continuation

- Tracked text lines: 44621
- Tracked Python lines: 39928
- Tracked Julia lines: 826
- `src` Python files: 192
- `src` Python lines: 38420
- `tests` Python lines: 1447
- `src/mean_field/systems` Python lines: 26612
- Files over 1000 lines: 0

Validation:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 53 passed
```

## Update: thin system package facades

Commit in this continuation:

- `58d7876 Thin system package facades`

### Scope

Reduced system package-root `__init__.py` files to the symbols actually needed by the current tracked API/tests. Implementation modules remain tracked and importable by explicit module path; only large package-root re-export surfaces were thinned.

Touched facades:

- `src/mean_field/systems/RnG_hBN/__init__.py`
- `src/mean_field/systems/atmg/__init__.py`
- `src/mean_field/systems/htg/__init__.py`
- `src/mean_field/systems/tdbg/__init__.py`
- `src/mean_field/systems/tmbg/__init__.py`

### Current summary after this continuation

- Tracked text lines: 44122
- Tracked Python lines: 39386
- Tracked Julia lines: 826
- `src` Python files: 192
- `src` Python lines: 37878
- `tests` Python lines: 1447
- `src/mean_field/systems` Python lines: 26070
- Files over 1000 lines: 0

Validation:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 53 passed
```

## Update: reintroduce minimal common topology FHS core

Commit in this continuation:

- `ca92e10 Reintroduce minimal topology FHS core`

### Scope restored

Topology scope was explicitly reopened by the user. Restored only a small, system-independent common API:

```text
src/analysis/topology/__init__.py
src/analysis/topology/core.py
```

The restored surface provides Fukui-Hatsugai-Suzuki link variables, plaquette Berry flux, Chern integration, state/subspace selection, direct-gap grouping, optional boundary sewing transforms, and metadata records for selected wavefunction columns.

Still archived / not restored:

- system topology wrappers under `src/mean_field/systems/*/topology.py`
- `analysis.topology.quantum_geometry` QGT/quantum-metric helpers
- topology workflow/report/plotting adapters
- paper-level topology validation jobs

### Validation

A small QWZ two-band smoke test was added:

```text
tests/test_analysis_topology.py
```

It verifies lower/upper band Chern numbers in topological and trivial mass regions, total two-band subspace Chern cancellation, and direct-gap grouping.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 56 passed
```

### Current summary after this continuation

- Tracked text lines: 44916
- Tracked Python lines: 40076
- Tracked Julia lines: 826
- `src` Python files: 194
- `src` Python lines: 38493
- `tests` Python lines: 1522
- `src/mean_field/systems` Python lines: 26070
- Files over 1000 lines: 0

## Update: full health check and topology wavefunction layout helper

Commit in this continuation:

- `bb1c820 Remove archived CLI console entry point`
- `8e80c39 Add topology wavefunction layout helpers`

### Health check finding

A full code-health pass after the topology core restore found one real packaging issue: `pyproject.toml` still declared the archived console entry point `mean-field = "mean_field.cli:main"` even though `src/mean_field/cli.py` is no longer tracked. The stale console-script entry was removed. Editable metadata dry-run now succeeds with local build isolation disabled:

```bash
python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

Network-isolated build dependency resolution still cannot fetch `setuptools>=69`, so build-isolation dry-run is not a code/package-config signal on the cluster.

### Topology follow-up

Added a generic topology wavefunction layout helper without restoring system wrappers or QGT helpers:

```text
src/analysis/topology/wavefunction.py
```

This helper canonicalizes already-built wavefunction arrays to `(mesh_1, mesh_2, basis_dim, n_state)`, preserves flattened state labels, reshapes flat k axes to 2D grids, and builds `WavefunctionIndex` metadata from state labels. It does not reconstruct projected-HF microscopic wavefunctions or infer sewing conventions.

Updated tests:

```text
tests/test_analysis_topology.py
```

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 58 passed

python import-boundary smoke
# import boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 45233
- Tracked Python lines: 40343
- Tracked Julia lines: 826
- `src` Python files: 195
- `src` Python lines: 38712
- `tests` Python lines: 1570
- `src/mean_field/systems` Python lines: 26070
- Files over 1000 lines: 0

## Update: add minimal topology system-facing adapter

Commit in this continuation:

- `b1aca58 Add minimal topology system adapter`

### Scope

Added a small system-facing adapter without restoring concrete `mean_field.systems.*.topology` wrappers:

```text
src/analysis/topology/system.py
```

The adapter accepts already-built eigenvector grids or grid-result objects with `eigenvectors` and optional `k_grid_frac`, attaches system/valley/label metadata, delegates all FHS link/plaquette/Chern calculations to `analysis.topology.core`, and returns a compact `TopologyResult` for callers that need a historical-style result shape.

Still not restored:

- concrete system `topology.py` modules
- retrying grid builders or model methods such as `topology_on_grid(...)`
- QGT/quantum-metric helpers
- paper-level topology workflows or Slurm jobs

### Validation

`tests/test_analysis_topology.py` now covers the system adapter with QWZ eigenvector grids, orientation sign, metadata propagation, grid-result input, and the missing-eigenvector error path.

Focused validation on `test001`:

```bash
PYTHONPATH=src pytest -q tests/test_analysis_topology.py
# 7 passed
```

### Current summary after this continuation

- Tracked text lines: 45539
- Tracked Python lines: 40590
- Tracked Julia lines: 826
- `src` Python files: 196
- `src` Python lines: 38910
- `tests` Python lines: 1619
- `src/mean_field/systems` Python lines: 26070
- Files over 1000 lines: 0

## Update: restore system-independent quantum-geometry core

Commit in this continuation:

- `420e8d7 Restore topology quantum geometry core`

### Scope restored

Restored the system-independent projector QGT / quantum metric / Fubini-Study trace helpers:

```text
src/analysis/topology/quantum_geometry.py
```

The restored module depends only on already-built wavefunction meshes plus optional boundary sewing transforms and the common FHS core. It provides projector QGT finite differences, metric/Berry decomposition, Fubini-Study trace helpers, normalized map helpers, reciprocal-vector coordinate transforms, and optional FHS comparison through `compute_quantum_geometry(..., include_fhs=True)`.

Still not restored:

- concrete `mean_field.systems.*.topology` wrappers
- projected-HF microscopic wavefunction reconstruction helpers
- topology plotting/report workflows
- paper-level topology/QGT Slurm validation jobs

### Validation

`tests/test_analysis_topology.py` now includes quantum-geometry smoke coverage:

- constant wavefunction has zero QGT/metric/Berry curvature and zero Chern;
- QWZ two-band model has projector finite-difference Chern consistent with FHS Chern within finite-difference tolerance;
- Fubini-Study trace and normalized-map helper shape/integration behavior is checked.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 62 passed

python import-boundary smoke
# import boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 46535
- Tracked Python lines: 41541
- Tracked Julia lines: 826
- `src` Python files: 197
- `src` Python lines: 39820
- `tests` Python lines: 1660
- `src/mean_field/systems` Python lines: 26070
- Files over 1000 lines: 0

## Update: restore thin TMBG topology wrapper

Commit in this continuation:

- `1ffe7dd Restore thin TMBG topology wrapper`

### Scope restored

Restored the first concrete system topology wrapper as a thin delegation layer:

```text
src/mean_field/systems/tmbg/topology.py
```

The wrapper exposes `compute_topology_from_eigenvectors`, `compute_topology_from_grid_result`, and `compute_topology_on_grid`. It contains no FHS/link/plaquette implementation and delegates all topology math to `analysis.topology`; the on-grid helper only builds one explicit eigenvector grid through the existing TMBG band API.

Still not restored:

- TDBG/ATMG/RLG-hBN concrete topology wrappers
- `model.topology_on_grid(...)` package-root convenience exports
- retrying grid-builder workflows
- projected-HF microscopic wavefunction reconstruction helpers
- paper-level topology workflows or Slurm jobs

### Validation

Added:

```text
tests/test_tmbg_topology.py
```

The tests use QWZ/fake-grid inputs and monkeypatching to prove metadata/delegation behavior without running TMBG production physics.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 66 passed

python import-boundary smoke
# topology wrapper boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 46815
- Tracked Python lines: 41765
- Tracked Julia lines: 826
- `src` Python files: 198
- `src` Python lines: 39928
- `tests` Python lines: 1776
- `src/mean_field/systems` Python lines: 26178
- Files over 1000 lines: 0

## Update: restore thin TDBG topology wrapper

Commit in this continuation:

- `00d79b9 Restore thin TDBG topology wrapper`

### Scope restored

Restored the second concrete system topology wrapper as a thin delegation layer:

```text
src/mean_field/systems/tdbg/topology.py
```

The wrapper exposes `compute_topology_from_eigenvectors`, `compute_topology_from_grid_result`, `compute_topology_on_grid`, `boundary_sewing_transforms`, and reuses `translation_srcmap` from projected-HF geometry. It delegates all FHS link/plaquette/Chern calculations to `analysis.topology` and keeps TDBG q-site boundary sewing as system metadata/gauge plumbing.

Still not restored:

- ATMG/RLG-hBN concrete topology wrappers
- `model.topology_on_grid(...)` package-root convenience methods
- retrying grid-builder workflows
- projected-HF microscopic wavefunction reconstruction helpers
- paper-level topology workflows or Slurm jobs

### Validation

Added:

```text
tests/test_tdbg_topology.py
```

The tests use QWZ/fake-grid inputs and monkeypatching to prove metadata/delegation behavior, explicit grid construction parameters, n-band guard behavior, and q-site boundary sewing shape/mapping behavior without running TDBG production physics.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 71 passed

python import-boundary smoke
# TMBG/TDBG topology wrapper boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47061
- Tracked Python lines: 41952
- Tracked Julia lines: 826
- `src` Python files: 199
- `src` Python lines: 39972
- `tests` Python lines: 1919
- `src/mean_field/systems` Python lines: 26222
- Files over 1000 lines: 0

## Update: restore thin ATMG topology wrapper

Commit in this continuation:

- `c67de40 Restore thin ATMG topology wrapper`

### Scope restored

Restored the third concrete system topology wrapper as a thin delegation layer:

```text
src/mean_field/systems/atmg/topology.py
```

The wrapper exposes `compute_topology_from_eigenvectors`, `compute_topology_from_grid_result`, and `compute_topology_on_grid`. It contains no FHS/link/plaquette implementation and delegates all topology math to `analysis.topology`; the on-grid helper only builds one explicit eigenvector grid through the existing ATMG band API.

Still not restored:

- RLG-hBN concrete topology wrapper
- `model.topology_on_grid(...)` package-root convenience methods
- retrying grid-builder workflows
- projected-HF microscopic wavefunction reconstruction helpers
- paper-level topology workflows or Slurm jobs

### Validation

Added:

```text
tests/test_atmg_topology.py
```

The tests use QWZ/fake-grid inputs and monkeypatching to prove metadata/delegation behavior, explicit grid construction parameters, and n-band guard behavior without running ATMG production physics.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 75 passed

python import-boundary smoke
# TMBG/TDBG/ATMG topology wrapper boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47239
- Tracked Python lines: 42071
- Tracked Julia lines: 826
- `src` Python files: 200
- `src` Python lines: 39978
- `tests` Python lines: 2032
- `src/mean_field/systems` Python lines: 26228
- Files over 1000 lines: 0

## Update: restore thin RLG-hBN topology wrapper

Commit in this continuation:

- `286ee1f Restore thin RLG-hBN topology wrapper`

### Scope restored

Restored the RLG-hBN concrete system topology wrapper as a thin delegation layer:

```text
src/mean_field/systems/RnG_hBN/topology.py
```

The wrapper exposes `compute_topology_from_eigenvectors`, `compute_topology_from_grid_result`, `compute_topology_on_grid`, and `rlg_hbn_boundary_sewing_transforms`. It contains no FHS/link/plaquette implementation and delegates all topology math to `analysis.topology`; the RLG-hBN-specific part is reciprocal-translation boundary sewing and explicit orientation metadata.

Still not restored:

- HTG concrete topology wrapper
- `model.topology_on_grid(...)` package-root convenience methods
- retrying grid-builder workflows
- projected-HF microscopic wavefunction reconstruction helpers
- paper-level topology workflows or Slurm jobs

### Validation

Added:

```text
tests/test_rlg_hbn_topology.py
```

The tests use QWZ/fake-grid inputs and fake reciprocal-lattice sewing to prove metadata/delegation behavior, orientation-sign metadata, explicit grid construction parameters, n-band guard behavior, and reciprocal-translation sewing shape/mapping behavior without running RLG-hBN production physics.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 80 passed

python import-boundary smoke
# TMBG/TDBG/ATMG/RLG-hBN topology wrapper boundary ok

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47397
- Tracked Python lines: 42170
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39991
- `tests` Python lines: 2118
- `src/mean_field/systems` Python lines: 26321
- Files over 1000 lines: 0

## Update: codify system topology wrapper boundary

Commit in this continuation:

- `1396390 Add topology wrapper boundary test`

### Scope

Added a tracked boundary test for the current concrete topology wrapper surface:

```text
tests/test_system_topology_wrapper_boundary.py
```

The test asserts that thin wrappers are importable for TMBG, TDBG, ATMG, and RLG-hBN, and that `mean_field.systems.htg.topology` remains absent pending a separate API decision. HTG was not restored in this step because there was no archived `systems/htg/topology.py`, and the current HTG band-grid API selects absolute band windows via `band_indices` / `central_band_count`, so its wrapper semantics should be reviewed before adding a public topology surface.

### Validation

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 85 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47486
- Tracked Python lines: 42200
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39991
- `tests` Python lines: 2148
- `src/mean_field/systems` Python lines: 26321
- Files over 1000 lines: 0

## Update: harden topology wrapper endpoint and sewing guards

Commit in this continuation:

- `a01b23f Harden topology wrapper grid guards`

### Scope

Hardened the restored thin topology wrappers based on wrapper review findings:

- `compute_topology_on_grid(..., endpoint=True)` now raises in TMBG, TDBG, ATMG, and RLG-hBN wrappers. FHS torus meshes must use one representative per periodic direction, not duplicate the endpoint seam.
- `RLG_hBN.compute_topology_from_grid_result(..., use_boundary_sewing=True)` now requires either explicit `sewing_transforms` or both `lattice` and `params`; it no longer silently falls back to no sewing when the default boundary-sewing request cannot be fulfilled.

No paper-level physics validation, Slurm jobs, projected-HF reconstruction, or model convenience methods were added.

### Validation

Updated wrapper tests now cover endpoint rejection and the RLG-hBN boundary-sewing input guard.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 90 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47598
- Tracked Python lines: 42231
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39998
- `tests` Python lines: 2172
- `src/mean_field/systems` Python lines: 26328
- Files over 1000 lines: 0

## Update: unify topology band-index semantics and add model convenience

Commit in this continuation:

- `b9be514 Unify topology band indices and model convenience`

### Scope

Unified system topology band-index semantics and added optional model convenience methods:

- `analysis.topology.compute_system_topology_from_grid_result(...)` now treats `band_indices` as grid-result/system band labels. If `grid_result.band_indices` is present, those labels are mapped to returned eigenvector columns; otherwise a full/prefix grid uses the natural labels `0..n_columns-1`.
- The common adapter records `band_indices_semantics`, `absolute_band_indices`, `column_indices`, and `grid_result_band_indices` in result metadata.
- TMBG, TDBG, ATMG, and RLG-hBN `compute_topology_on_grid(...)` now default to full-grid eigenvectors when `n_bands=None`; explicit `n_bands` is only a prefix guard and must include the requested absolute band labels.
- TMBG and ATMG wrappers now accept explicit `sewing_transforms`, forwarding them to the common topology API instead of hard-locking no-sewing behavior. No automatic TMBG/ATMG sewing convention was fabricated.
- Added optional lazy `model.topology_on_grid(...)` delegates for TMBG, TDBG, ATMG, and RLG-hBN. HTG remains absent pending its separate absolute-band-window API decision.

### Validation

Added/updated tests for common grid-result band-label mapping, TMBG/ATMG explicit sewing-transform forwarding, wrapper full-grid default semantics, and model convenience delegation.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 98 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47740
- Tracked Python lines: 42361
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39998
- `tests` Python lines: 2302
- `src/mean_field/systems` Python lines: 26342
- Files over 1000 lines: 0

## Update: make topology orientation sign self-consistent

Commit in this continuation:

- `288035c Make topology orientation sign self-consistent`

### Scope

Hardened the common topology orientation convention:

- `orientation_sign` is now restricted to `+1` or `-1`; arbitrary curvature scaling is rejected.
- For `orientation_sign=-1`, returned FHS links are conjugated and the returned Berry connection is computed from the conjugated links, so Berry connection, Berry curvature, and Chern number are internally consistent.
- Added a toy QWZ test asserting `link_1`, `link_2`, Berry connection, curvature, and rounded Chern all flip self-consistently under `orientation_sign=-1`.

No system-specific sewing convention, paper workflow, projected-HF reconstruction, Slurm job, or physical-result claim was added.

### Validation

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 99 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47839
- Tracked Python lines: 42378
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39999
- `tests` Python lines: 2318
- `src/mean_field/systems` Python lines: 26342
- Files over 1000 lines: 0

## Update: add TMBG/ATMG reciprocal boundary sewing helpers

Commit in this continuation:

- `bda781b Add TMBG ATMG topology sewing helpers`

### Scope

Added system-owned reciprocal-basis boundary sewing helpers for the thin TMBG and ATMG topology wrappers:

- `mean_field.systems.tmbg.topology.boundary_sewing_transforms(lattice)` uses G-index reciprocal translation with local block size 6.
- `mean_field.systems.atmg.topology.boundary_sewing_transforms(lattice, params)` uses the same G-index reciprocal translation with local block size `2 * params.n_layers`.
- TMBG/ATMG `compute_topology_on_grid(...)` now defaults to these boundary sewing transforms unless callers pass explicit `sewing_transforms` or `boundary_sewing=False`.

This restores the expected torus seam relabeling for the continuum plane-wave basis, but does not claim paper-level Chern validation or convergence. No Slurm jobs, paper workflows, projected-HF reconstruction, or result mutations were performed.

### Validation

Added toy shape/mapping tests for the TMBG/ATMG reciprocal translation transforms and kept fake-grid topology tests software-only.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 101 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 47892
- Tracked Python lines: 42389
- Tracked Julia lines: 826
- `src` Python files: 201
- `src` Python lines: 39987
- `tests` Python lines: 2341
- `src/mean_field/systems` Python lines: 26330
- Files over 1000 lines: 0

## Update: restore thin HTG topology wrapper

Commit in this continuation:

- `517840d Restore thin HTG topology wrapper`

### Scope

Restored `mean_field.systems.htg.topology` as a thin wrapper over `analysis.topology` for ordinary HTG single-particle GridBandsResult topology:

- HTG already used the common `GridBandsResult` band/grid container; the new topology wrapper maps requested absolute HTG band labels through `grid_result.band_indices` using the common topology adapter.
- `compute_topology_on_grid(...)` requests the contiguous absolute band window needed by the HTG scipy diagonalizer, then selects the requested labels through the common grid-result mapping.
- Added HTG reciprocal boundary sewing using the same G-index relabeling as TMBG with local block size 6.
- Added optional `HTGModel.topology_on_grid(...)` lazy delegate.
- Updated current docs and topology boundary tests to include HTG in the restored thin-wrapper surface.

Still not restored: HTG Chern-basis workflows, projected/supercell-HF topology, paper workflows, plotting/report scripts, or paper-level claims.

### Validation

Added `tests/test_htg_topology.py` with QWZ/fake-grid/monkeypatch tests for delegation, absolute band-label mapping, contiguous-window grid requests, endpoint rejection, and sewing transform shape/mapping.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 106 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 48091
- Tracked Python lines: 42544
- Tracked Julia lines: 826
- `src` Python files: 202
- `src` Python lines: 39997
- `tests` Python lines: 2486
- `src/mean_field/systems` Python lines: 26340
- Files over 1000 lines: 0

## Update: add pure projected-HF micro reconstruction helper

Commit in this continuation:

- `c7b45b1 Add projected HF micro reconstruction helper`

### Scope

Added a small system-independent projected-HF reconstruction helper under `mean_field.core.hf`:

- `canonicalize_projected_micro_basis(...)` canonicalizes a projected microscopic basis to `(k, microscopic_basis, active_basis)`.
- `reconstruct_projected_micro_wavefunctions(...)` contracts a canonical projected microscopic basis with explicit active HF eigenvectors shaped `(active_basis, hf_state, k)` and returns a `MicroscopicWavefunctionBundle`.

The helper is pure array plumbing. It does not infer system-specific band/flavor ordering, does not build sewing transforms, does not fill missing active HF eigenvectors in system adapters, and does not restore projected-HF topology or paper workflows.

### Validation

Added `tests/test_core_hf_reconstruction.py` for identity/rotation reconstruction, noncanonical axis handling, unitarity guard, bad grid shape, bad labels, and bad basis rank.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 109 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 48239
- Tracked Python lines: 42642
- Tracked Julia lines: 826
- `src` Python files: 203
- `src` Python lines: 39992
- `tests` Python lines: 2598
- `src/mean_field/systems` Python lines: 26340
- Files over 1000 lines: 0

## Update: add guarded HFResult micro reconstruction fallback

Commit in this continuation:

- `f5b41ad Add guarded HFResult micro reconstruction fallback`

### Scope

Added a guarded public fallback for `HFResult.reconstruct_micro_wavefunctions()`:

- Existing system-state adapters still take precedence.
- The fallback only runs when `canonical_run_result.final_state.basis.micro_wavefunctions`, `final_state.eigenvectors_active`, and `basis.kvec` are present and non-empty.
- The fallback requires `basis.micro_wavefunctions` to be rank-3 and explicitly tagged with `basis.metadata["wavefunctions_axis_order"] == "k,microscopic_basis,active_basis"`.
- Noncanonical/raw system arrays such as current TDBG raw 4D projected wavefunctions are rejected with `NotImplementedError` rather than guessed.
- The fallback uses the existing system-independent `core.hf.reconstruction.reconstruct_projected_micro_wavefunctions(...)` helper.

System-specific TDBG/HTG/RLG-hBN/TMBG reconstruction adapters remain deferred; subagent reports for those lanes were written under ignored `tmp/subagents/reconstruction_next/`.

### Validation

Added `tests/test_api_hf_result_reconstruction.py` covering canonical dense fallback, missing/empty array errors, raw-rank rejection, missing axis metadata rejection, and state-adapter precedence.

Validation on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 112 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 48430
- Tracked Python lines: 42842
- Tracked Julia lines: 826
- `src` Python files: 203
- `src` Python lines: 39993
- `tests` Python lines: 2788
- `src/mean_field/systems` Python lines: 26340
- Files over 1000 lines: 0

## Update: connect projected-HF reconstruction adapters across systems

Commit in this continuation:

- `2da1ea1 Connect projected HF reconstruction adapters`

### Scope

Connected projected-HF microscopic reconstruction to the common `mean_field.core.hf.reconstruction` helper through reviewed system-owned adapters:

- **TDBG**: `TDBGProjectedHFResult.reconstruct_micro_wavefunctions(...)` expands raw `(state,k,q_site,local)` projected data to canonical `(k,microscopic_basis,active_basis)` with explicit `spin,valley,q_site,local` row metadata, uses Hermitian final-HF eigensystems only, supports selected-state reconstruction, and records that sewing/topology eligibility is unavailable until a TDBG torus sewing convention is implemented.
- **HTG primitive/supercell**: added HTG-local reconstruction helpers with selected-state output guards. Primitive reconstruction remains no-sewing/topology-ineligible; supercell reconstruction can attach the validated full-boundary sewing transforms and has nontrivial row-order tests.
- **RLG-hBN/RnG-hBN**: exported explicit reconstruction helpers from `mean_field.systems.RnG_hBN.hf`, with selected-state guards, final-HF eigensystem wrapper, direct-sum row metadata, and projected-micro sewing transforms. Validation remains software/toy level until the saved Fig.6 target is rerun/rechecked on Slurm.
- **TMBG/Polshyn**: added a public flat-k diagnostic reconstruction adapter with Hermiticity/off-sector/stored-energy checks and selected-state guards. It is exported from the public Polshyn facade, but its returned flat bundles remain explicitly sewing-disabled and topology-ineligible; topology must go through the separate doubled-cell sewing/reshape adapter in `mean_field.systems.tmbg.topology` and still requires Slurm/paper validation.

These changes do not restore paper workflows, do not submit Slurm jobs, and do not claim physical Chern/QGT/paper reproduction.

### Validation targets located by subagents

Subagent report `tmp/subagents/reconstruction_next2/validation_targets.md` identified the strongest future physical checks:

- RLG-hBN Fig.6 xi=1, V=60 meV: saved HF state + cached projected basis + historical reconstructed-micro Chern JSON/NPZ.
- HTG minimal supercell `nu=3.5`: saved supercell HF state plus isolated flat-band topology artifacts; micro basis/eigenvectors must be rebuilt/rediagonalized under Slurm.
- TMBG/Polshyn Fig. S1: good Chern-number targets, but cleaned artifacts lack full state/micro-basis/eigenvector payloads.
- TDBG: no suitable projected-HF reconstruction target found in `results/`; existing Fig.3 artifacts are non-HF topology references.

### Software validation

Focused reconstruction gate on `test001`:

```bash
PYTHONPATH=src python -m compileall -q \
  src/mean_field/systems/tdbg src/mean_field/systems/htg \
  src/mean_field/systems/RnG_hBN src/mean_field/systems/tmbg \
  tests/test_tdbg_projected_hf_reconstruction.py \
  tests/test_htg_supercell.py \
  tests/test_rlg_tmbg_reconstruction_adapters.py

PYTHONPATH=src pytest -q \
  tests/test_tdbg_projected_hf_reconstruction.py \
  tests/test_htg_supercell.py \
  tests/test_rlg_tmbg_reconstruction_adapters.py \
  tests/test_api_hf_result_reconstruction.py \
  tests/test_api_hf_adapters.py::test_public_run_hf_tdbg_explicit_config_dispatches_without_guessing \
  tests/test_api_hf_adapters.py::test_public_run_hf_tdbg_explicit_config_attaches_canonical_contract_result \
  tests/test_api_hf_adapters.py::test_public_run_hf_htg_primitive_explicit_config_attaches_canonical_contract_result \
  tests/test_api_hf_adapters.py::test_public_run_hf_htg_supercell_explicit_config_attaches_canonical_contract_result
# 28 passed
```

Full gate on `test001` before commit:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 128 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 51623
- Tracked Python lines: 45917
- Tracked Julia lines: 826
- `src` Python files: 206
- `src` Python lines: 42057
- `tests` Python lines: 3799
- `src/mean_field/systems` Python lines: 28404
- Files over 1000 lines: 0

The previous soft `src` Python line budget is exceeded by the multi-system reconstruction adapter surface; this should be cleaned up later via a separate refactor/compaction pass if preserving the 4w target remains mandatory.

## Update: finalize projected-HF topology/reconstruction API guards

Commit in this continuation:

- `2b5c174 Finalize projected HF topology API guards`

### Scope

Finalized the aggressive common-API wiring requested for the remaining projected-HF reconstruction/topology surface:

- `HFResult.reconstruct_micro_wavefunctions(...)` accepts selected `state_indices`/`band_indices` plus `max_dense_elements`, forwards supported kwargs to state adapters, and marks the canonical dense fallback explicitly topology-ineligible because it is algebraic contraction without system sewing/grid topology adapters.
- `analysis.topology.compute_system_topology_from_bundle(...)` now fails early for flat reconstructed bundles with a clear instruction to use a system topology adapter or validated grid reshape/sewing path.
- `mean_field.systems.tdbg.topology.compute_projected_hf_topology(...)` is the TDBG projected-HF topology API: it reconstructs through the TDBG system adapter, preserves sewing transforms, reshapes flat source-grid wavefunctions to `(mesh, mesh, basis, state)`, and delegates only FHS math to the common topology adapter. The public flat `TDBGProjectedHFResult.reconstruct_micro_wavefunctions(...)` remains topology-ineligible by itself because `WavefunctionBundle` does not carry sewing transforms.
- `mean_field.systems.tmbg.polshyn_supercell.reconstruct_polshyn_wang_hf_micro_wavefunctions(...)` is accepted as a public flat-k diagnostic API. Its returned flat bundles remain explicitly sewing-disabled and topology-ineligible; use `mean_field.systems.tmbg.topology.compute_polshyn_projected_hf_topology(...)` for the reviewed doubled-cell sewing/reshape path, with physical validation still pending.
- Common direct-sum reconstruction helpers gained focused tests and duplicate-index rejection.

### Validation

Focused software/API gate on `test001`:

```bash
PYTHONPATH=src pytest -q \
  tests/test_core_hf_reconstruction.py \
  tests/test_analysis_topology.py \
  tests/test_api_hf_result_reconstruction.py \
  tests/test_tdbg_projected_hf_reconstruction.py \
  tests/test_tdbg_topology.py \
  tests/test_rlg_tmbg_reconstruction_adapters.py \
  tests/test_system_topology_wrapper_boundary.py
# 50 passed
```

Full software/API gate on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 138 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Current summary after this continuation

- Tracked text lines: 52571
- Tracked Python lines: 46809
- Tracked Julia lines: 826
- `src` Python files: 206
- `src` Python lines: 42636
- `tests` Python lines: 4112
- `src/mean_field/systems` Python lines: 28557
- Files over 1000 lines: 0

This remains software/API validation only. HTG/RLG physical parity scripts were prepared under ignored `tmp/subagents/reconstruction_remaining/validation_prep/`; Slurm validation is separate.

## Update: add reviewed Polshyn doubled-cell topology adapter

Commit in this continuation:

- `ec23fdd Add Polshyn doubled cell topology adapter`

### Scope

After user correction, topology validation conventions were clarified:

- RLG-hBN topology should follow the paper/physical sewing convention; matching an old no-sewing artifact is not physical validation.
- HTG topology should use the common moiré FHS/sewing framework; old local Berry/min-link artifacts should not drive ad-hoc implementation changes.
- TDBG projected-HF topology remains software/API-only until a real reproduced HF target exists.
- TMBG/Polshyn topology can proceed through a derived doubled-cell sewing/reshape adapter.

Implemented `mean_field.systems.tmbg.topology.compute_polshyn_projected_hf_topology(...)` as the separate topology-ready path for TMBG/Polshyn projected-HF states:

- Keeps `polshyn_supercell.reconstruct_polshyn_wang_hf_micro_wavefunctions(...)` as a public flat-k diagnostic API whose returned bundles remain `topology_eligible=False`.
- Adds Polshyn doubled-cell `B1/B2` boundary sewing over row order `spin_major,valley_inner,basis_F(local=6,embed_x,embed_y)`.
- Reshapes flat Polshyn k order (`iy/f2` outer, `ix/f1` inner) to `(mesh_B1, mesh_B2, basis, state)` with `order='F'` before delegating to common FHS.
- Rejects no-sewing Polshyn topology by default; `diagnostic_no_sewing=True` records a non-physical diagnostic path with `topology_eligible=False`.
- Requires explicit/validated flat k-grid order; no `sqrt(n_k)` topology-grid fallback.
- Nests caller metadata under `caller_metadata` so safety/provenance fields cannot be overwritten.

### Validation

Focused gate on `test001`:

```bash
PYTHONPATH=src pytest -q \
  tests/test_tmbg_topology.py \
  tests/test_rlg_tmbg_reconstruction_adapters.py \
  tests/test_analysis_topology.py \
  tests/test_core_hf_reconstruction.py
# 35 passed
```

Full software/API gate on `test001`:

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src pytest -q $(git ls-files tests)
# 142 passed

python -m pip install -e . --dry-run --no-deps --no-build-isolation
# Would install mean-field-0.1.0
```

### Metrics after this continuation

- Tracked text lines: 53249
- Tracked Python lines: 47487
- Tracked Julia lines: 826
- `src` Python files: 206
- `src` Python lines: 43085
- `tests` Python lines: 4341
- `src/mean_field/systems` Python lines: 29006
- Files over 1000 lines: 0

This is still software/API validation only. Paper-level TMBG/Polshyn Chern validation remains a Slurm target, preferably starting from the S1b no-remote k18 folded subspace/HF split-band checkpoints.

## Update: promote reviewed Polshyn h0-subtraction API

Commit in this continuation:

- pending: promote Polshyn h0-subtraction API

### Scope

After Slurm validation of the Polshyn doubled-cell topology path, promoted the two reviewed Polshyn one-body conventions into a small public, system-owned API:

- `PolshynH0SubtractionConfig(mode="none" | "active-reference" | "minus-full-p0")`
- `PolshynH0SubtractionResult`
- `polshyn_reference_projector_blocks(...)`
- `compute_polshyn_active_reference_h0_correction(...)`
- `compute_polshyn_minus_full_p0_h0_correction(...)`
- `basis_with_polshyn_h0_correction(...)`
- `apply_polshyn_h0_subtraction(...)`

The API lives under `mean_field.systems.tmbg.polshyn_supercell` and remains a TMBG/Polshyn system adapter. It does not add Fig. S1 workflows, plotting, Slurm submission, topology target selection, or generic core-HF behavior. Application signs are fixed by mode: `active-reference` applies `+1`; `minus-full-p0` applies `-1`; callers cannot override the sign.

`PolshynRunHFConfig` now accepts an explicit `h0_subtraction` field. When enabled, the public `run_hf(...)` adapter prebuilds Polshyn overlap blocks, applies the h0 correction to the projected basis, then still runs the shared Wang/Xiaoyu core-HF engine. Metadata records mode, sign, P0 reference, and q=0 policy in observables, artifact metadata, and canonical archive manifest.

### Validation evidence

Fresh Slurm validation before this API promotion, at `7d3fc7a`, established the selected-target conventions used here:

- S1b noninteracting folded-subspace topology: job `159192`, PASS, summary `tmp/subagents/reconstruction_remaining/validation_prep/runs/polshyn_s1b_noninteracting_159192/summary.json`.
- S1b no-remote HF split target with `minus-full-p0`: job `159214`, PASS, summary `tmp/subagents/reconstruction_remaining/validation_prep/runs/polshyn_s1b_hf_159214/summary.json`.
- S1c remote-window HF split target with `active-reference`: job `159217`, PASS, summary `tmp/subagents/reconstruction_remaining/validation_prep/runs/polshyn_s1c_hf_159217/summary.json`.

Post-API software gate on `test001`:

```bash
PYTHONPATH=src pytest -q \
  tests/test_tmbg_polshyn_hf_readiness.py \
  tests/test_tmbg_topology.py \
  tests/test_rlg_tmbg_reconstruction_adapters.py \
  tests/test_api_imports.py
# 44 passed
```

These tests cover h0-subtraction config normalization/fixed signs, reference projector blocks, active-reference correction against the common interaction builder, q=0 policy, basis correction shape/Hermiticity guards, minus-full-P0 empty-overlap safety, public facade exports, and public `run_hf(...)` metadata for `none`, `active-reference`, and `minus-full-p0` smoke configurations.

### Caveat

The Slurm summaries above validated the ignored harness logic before the API promotion. After this public API commit, rerun the S1b/S1c Slurm validation through the public `PolshynH0SubtractionConfig` path before upgrading documentation from software/API validation to post-API physical validation.
