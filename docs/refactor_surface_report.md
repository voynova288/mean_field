# Refactor surface report

This Phase 2 report measures legacy surface area and tracks cleanup slices that delete or thin old paths.

## Summary

- Tracked text lines: 61776
- Tracked Python lines: 58516
- Tracked Julia lines: 826
- `src` Python files: 190
- `src` Python lines: 52413
- Files over 1000 lines: 9
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

## Top 30 Python files under `src`

| Lines | Path |
|---:|---|
| 2256 | `src/mean_field/systems/RnG_hBN/hf.py` |
| 2058 | `src/mean_field/systems/RnG_hBN/tdhf.py` |
| 1347 | `src/mean_field/systems/htg/supercell.py` |
| 1306 | `src/mean_field/api/hf.py` |
| 1267 | `src/mean_field/crpa/hf_interface.py` |
| 1258 | `src/mean_field/systems/tbg/zero_field/hf.py` |
| 1149 | `src/mean_field/core/hf/finite_field.py` |
| 1127 | `src/mean_field/systems/tmbg/polshyn_supercell.py` |
| 1012 | `src/mean_field/systems/tbg/finite_field/spectrum.py` |
| 873 | `src/analysis/topology/quantum_geometry.py` |
| 805 | `src/analysis/response_derivative_gauge.py` |
| 788 | `src/mean_field/devtools/canonical_hf_backfill/_scan.py` |
| 786 | `src/mean_field/systems/RnG_hBN/cache.py` |
| 772 | `src/analysis/shift_current/core.py` |
| 736 | `src/mean_field/cli.py` |
| 735 | `src/mean_field/systems/tbg/zero_field/hf_contracts.py` |
| 710 | `src/mean_field/core/hf/tdhf.py` |
| 686 | `src/mean_field/systems/RnG_hBN/hf_contracts.py` |
| 643 | `src/mean_field/systems/htg/supercell_contracts.py` |
| 627 | `src/mean_field/benchmarks.py` |
| 617 | `src/mean_field/core/hf/overlap.py` |
| 590 | `src/mean_field/systems/htg/_hf_contracts.py` |
| 565 | `src/analysis/topology/core.py` |
| 528 | `src/mean_field/systems/tbg/zero_field/artifacts.py` |
| 498 | `src/mean_field/crpa/validation.py` |
| 492 | `src/mean_field/systems/RnG_hBN/screening.py` |
| 456 | `src/mean_field/systems/tbg/zero_field/model.py` |
| 454 | `src/mean_field/crpa/workflow.py` |
| 445 | `src/mean_field/systems/tdbg/projected_hf_state.py` |
| 434 | `src/mean_field/crpa/diagnostics.py` |

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
