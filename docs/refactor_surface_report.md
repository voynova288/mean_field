# Refactor surface report

This Phase 2 report preserves its historical tracked baseline and tracks cleanup slices that delete or thin old paths. The current local-tree snapshot below includes retained untracked/optional source and therefore must not be compared directly with the historical tracked-only baseline.

## Historical tracked baseline summary

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

- Retired the unexported Polshyn full-flavor IVC array-contract scaffold; no tracked production code imports it and the maintained Polshyn/Wang canonical adapter now belongs to `tmbg._polshyn_contracts`.
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

- Deleted the unexported HTQG private plotting helper module; reusable plotting remains in `mean_field.core.plotting.bands` and active system plot adapters.
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

- Lifecycle: the one-off RnG/hBN finite-q TDHF entrypoint was retired early, temporarily restored on a later debug branch, and is now finally retired after the orchestration surface was superseded by typed system APIs and later source-bound validation. The 2026-07-16 handoff checkpoint itself was blocked and incomplete; subsequent typed work completed the 12x12 three-channel full mesh and established **validated nonreproduction**, not Fig. S45 reproduction.
- Archived and removed the maintained devtool and dispatcher command without a shim or replacement CLI. The reusable core/system typed finite-q TDHF capabilities, their physics tests, and the later validated-nonreproduction evidence remain; this surface cleanup does not alter numerical owners, formulas, or the authority of those later results.
- Preserved all `tmp/tdhf/**` historical scratch, oracles, and evidence unchanged as read-only/unsupported provenance. The incomplete steps recorded in the 2026-07-16 handoff are not maintained workflow instructions; later completion is documented separately in `docs/tdhf_core_contract.md`, `docs/debug/rlg_hbn_fixed_quotient_hf_response_plan_20260716.md`, and `docs/debug/FIXED_QUOTIENT_DERIVATION.md`.
- This retirement itself adds no physics authority and must not be interpreted as Fig. S45 reproduction. It also does not revoke the later source-bound conclusion of validated nonreproduction.
- Deleted files: `src/mean_field/devtools/run_rlg_hbn_tdhf_finite_q.py`.
- Gross legacy LOC removed/thinned: 957 (956-line devtool plus one dispatcher registration).
- Direct `mean_field.systems.*` import statements in devtools/scripts/workflows: 7 -> 4.
- Focused retained-owner/script-surface gate: `86 passed`. The separate fixed-quotient anchor file retains its two known baseline failures. Full repository gate: `1447 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Exact-byte retirement archive: `local_archive/retired_surface/rlg_hbn_finite_q_tdhf_entrypoint_20260916/`, manifest SHA-256 `077f3d824ce167a8e4964ec6ff9681794e487f3560105d18e00011818f1d7b10`.
- Status-clarification follow-up archive: `local_archive/retired_surface/rlg_hbn_finite_q_tdhf_entrypoint_status_followup_20260916/`, manifest SHA-256 `b86bf022d3159837855960f155011ac0a33e58cfd3abf2ab9fd1d182a450050e`.

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

- Split the HTG primitive HF adapter façade into typed/reference/initialization/basis/interaction/runner/contract modules; the temporary `mean_field_adapter` compatibility path was subsequently retired after registry and provenance migration to `_hf_contracts`.
- At that split phase, deleted files: none; the later aggregate-façade retirement and wildcard hardening are recorded below.
- Gross legacy LOC removed/thinned: 2301.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_public_hf_api_facade

- Split the public HF API module into private type, registry, sidecar, result, and dispatch modules while keeping mean_field.api.hf as the stable facade and preserving lazy adapter import paths.
- Deleted files: none; this slice thinned duplicated implementations in place.
- Gross legacy LOC removed/thinned: 1274.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tmbg_polshyn_hf_helper_surface

- Split the TMBG Polshyn doubled-cell HF helper into typed, canonical-contract, filling, and Wang-engine modules; the temporary `polshyn_supercell.py` compatibility façade was subsequently retired after registry migration to `_polshyn_contracts`.
- At that split phase, deleted files: none; the later façade/shared retirement is recorded below.
- Gross legacy LOC removed/thinned: 1121.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 0 -> 0.

### split_tbg_zero_field_hf_helper_surface

- Split the TBG zero-field HF helper into basis/overlap, restricted, full, and diagnostics modules; the temporary `zero_field.hf` compatibility façade was subsequently retired after all callers moved to those owners.
- At that split phase, deleted files: none; the later compatibility-façade retirement is recorded below.
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

### retire_tbg_test_only_paper_preflights_20260915

- Archived and removed non-exported, test-only `khalaf_fig3.py`, `kumar2010.py`, and `wang2025.py` preflight contracts plus their dedicated tests.
- Archive: `local_archive/retired_surface/tbg_zero_field_paper_preflights_20260915`.
- Gross Python LOC removed: 3,854 (2,657 source + 1,197 tests).
- Retirement is not a validation or refutation of the referenced papers.

### retire_unused_crpa_hf_bridge_20260915

- Archived and removed the uncalled, unvalidated cRPA HF bridge (`hf_interface.py`, `hf_validation.py`, and `screened_coulomb.py`) and its 32 stale package exports.
- Retained cRPA dielectric/chunk/merge code is unchanged; `flat_remote.py` now imports private overlap helpers from their real owner.
- Archive: `local_archive/retired_surface/crpa_unused_hf_bridge_20260915`.
- Gross source LOC removed: 1,710.

### remove_private_dead_helpers_20260915

- Removed 128 lines of unreferenced private helpers and temporary non-exported aliases across HTG, RLG/hBN, TBG, InAs/GaSb, CLI, devtools, and distributed-eigensolver support.
- Redirected TBG overlap re-exports to the common-HF owner instead of retaining compatibility-only imports.

### retire_non_api_abc_tbg_oracles_20260915

- Retired the entire unregistered ABC/Vituri system island and seven TBG companion diagnostic/oracle modules after confirming that none participate in `mean_field.api`, root exports, durable dispatchers, or other maintained systems.
- Preserved the exact source, dedicated tests, mixed-test predecessor, fixtures, scratch callers, and documentation snapshots in `local_archive/retired_surface/non_api_abc_tbg_oracles_20260915/`.
- Removed 43,054 source lines, 20,043 dedicated/companion test lines, 3,377 fixture text lines, and 129 scratch-script lines. Direct imports of the retired system-local modules are intentionally unsupported; common HF/TDHF APIs remain.

### inas_gasb_api_centered_cleanup_20260916

- Added explicit retained APIs for Xue/Zeng BHZ modeling, common `compute_bands`, typed Xue Q=0 mean field, exact saved-grid Kane4 bands, and the existing source-bound matrix-EI solver. No paper/material parameters are guessed.
- Archived and removed the standalone plane-wave qualification chain: 5,975 source lines and 4,159 dedicated test lines.
- Archived and removed obsolete Cartesian/split-zero operational authority workflows, publication sealers/projections, two standalone devtools, and their dedicated tests: 4,485 source lines and 1,317 test lines.
- Retired the now-unreferenced 152-line Cartesian workflow diagnostic helper. Signed radial carrier accounting was restored after retained matrix-EI tests proved it remained part of the live API.
- Gross retirement for this slice: 10,612 source lines and 5,476 dedicated test lines. Added API/model contract code is reported separately from retirement rather than hidden in a net line count.

### inas_gasb_api_centered_cleanup_followup_20260916

- Archived and removed the experimental imposed-E1/H1 fixed-pair density/Poisson/archive branch: 3,787 module lines, 66 fixed-pair-only adapter lines, and 1,949 dedicated test lines. The canonical Kane-Poisson and matrix-EI APIs remain.
- Archived and removed pair-replay artifact/publication authority after its operational workflow was retired: 989 source lines and 339 lines from the mixed kdotpy test. Pair selection numerical logic remains.
- Archived and removed the standalone angular-stability diagnostic and synthetic suite: 732 source lines and 937 test lines. Angular/axial interaction operators and matrix-EI HF remain.
- Archived and removed package-root-only Du2017 material attestation, full-parent reference, and UV-trial qualification helpers: 1,851 source lines and 1,059 dedicated test lines.
- Follow-up gross retirement: 7,425 source lines, 3,945 dedicated test lines, and 399 mixed-test lines.
- Cumulative InAs/GaSb API-centered retirement: 18,037 source lines, 9,421 dedicated test lines, and 410 mixed-test lines.

### inas_gasb_thin_package_facade_20260916

- Replaced the 164-symbol package-root compatibility facade with 51 maintained model, band, canonical Kane-Poisson builder/solver, archive-I/O, normal-reference, interaction-operator, and matrix-EI entrypoints.
- Retained implementation tests now import formula helpers and diagnostics directly from their owner modules instead of making those private surfaces package-root compatibility promises.
- Facade reduction: 237 source lines and 113 package-root exports; owner-import and exact-export contract tests expanded by 98 lines for explicitness.
- Cumulative InAs/GaSb API-centered source retirement: 18,274 lines.

### inas_gasb_owner_boundary_cleanup_20260916

- Moved same-directory staged no-replace publication into `mean_field.core.io`; deleted the 194-line system-local `workflow_publication.py` and its two zero-caller workflow claim/failure functions. Current Kane4 and canonical Kane-Poisson codecs retain identical publication semantics.
- Removed 123 lines of zero-caller helpers: the Cartesian boundary-restricted-mu diagnostic, `conventions.block_unitary`, duplicate kdotpy source hashes, and the unused equal-area radial mesh.
- Removed 39 lines of kdotpy historical facade identity: adapter lazy builder re-exports, forged builder `__module__` values, and historical dependency-injection wrappers. `kdotpy_window_builder` is now the sole builder owner; selection and homotopy algorithms are unchanged.
- Added direct `core.io` no-replace publication tests and owner-bound kdotpy seam tests.
- This pass retired 360 InAs/GaSb source lines; cumulative InAs/GaSb API-centered source retirement is 18,634 lines.

### inas_gasb_legacy_artifact_adapter_retirement_20260916

- Archived and removed the orphaned Cartesian Kane4 artifact adapter module and its three package-root exports.
- Removed the diagnostic-only nonzero-split radial kdotpy loader; the retained split-zero loader now always requires exact XML and wavefunction-manifest hash attestations and no longer exposes an attestation bypass.
- Removed schema-less canonical Kane-Poisson decoding and its nonauthorizing legacy wrapper. The maintained archive loader accepts only the typed current schema.
- Package-root API narrowed from 51 to 47 typed entrypoints while retaining model, bands, Xue HF, canonical Kane-Poisson, strict split-zero Kane4, typed bundle I/O, interaction operators, and matrix-EI.
- This pass retired 532 InAs/GaSb source lines; cumulative InAs/GaSb API-centered source retirement is 19,166 lines.

### historical_migration_devtools_retirement_20260916

- Current HF producers/loaders already use typed canonical artifact APIs, so the seven-module historical canonical-sidecar backfill scanner/stager family and its dispatcher command were archived and removed.
- Removed the endpoint-inclusive B0 density migration devtool; typed production/export APIs reject that legacy mesh convention.
- Removed the already-retired RLG/hBN paper-HF shim and its compatibility-only test; callers use the system-owned `save_rlg_hbn_hf_archive` API directly.
- Retired 1,819 source lines and 413 dedicated/compatibility test lines. Current canonical artifact roundtrips and the RLG archive owner roundtrip remain covered.

### inas_api_route_and_pair_only_cleanup_20260916

- Narrowed the InAs/GaSb package root from 47 low-level exports to the six high-level model/bands/HF entrypoints owned by `api.py`; Kane-Poisson, archive, normal-reference, interaction, and matrix-EI capabilities remain available from explicit numerical-owner modules.
- Made `KdotpyE1H1PairSelectionSpec` mandatory, removed anchor-only/auto-nearest quartet selection and the `require_pair_resolved` bypass, and made every maintained kdotpy call return a pair-resolved receipt before canonical-window unwrapping.
- Removed `carriers.py`; matrix-EI microscopic neutrality now uses the `normal_reference` owner. Test-only frozen-detuning and energy-sorted pocket calibration diagnostics were retired, while the distinct signed-radial `band_carriers.py` capability remains.
- Together with the historical migration-devtool wave, this pass retired 2,324 gross source lines and about 500 compatibility/diagnostic test lines. The InAs/GaSb root API is now six symbols and the package is 24 files / 16,393 lines.
- Focused gate: `282 passed`. Full repository gate: `1318 passed, 6 skipped, 4 failed`; all four failures are the pre-existing generic TDHF typed-invariant ordering, two RLG/hBN TDHF fixed-quotient/C3 checks, and TBG finite-field mixed-convergence acceptance.

### retire_tbg_zero_field_b0_benchmark_workflow

- Retired the TBG zero-field paper-benchmark execution surface: `runners.py`, all `_runners_*` modules, plotting/report/sidecar writers, off-grid HF path reconstruction/parity, B0 benchmark-result adapters, benchmark/bm/hf CLI branches, and the Julia reference-export dispatcher/helpers.
- Retained the BM/HF numerical owners, typed `TBGZeroFieldRunHFConfig` adapter, strict source-bound complete-state archive, exact saved-SCF-grid band selector, path-advisor geometry, and immutable external Julia/reference fixtures through the internal `mean_field.reference_oracles` loader.
- The strict writer now requires explicit `hf_run=` and `grid_solution=` inputs. The loader validates archive schema, issuer, typed source lineage, provenance/diagnostic cross-fields, density-derived filling, and real ODA history. The saved-grid selector unconditionally requires typed source lineage, rejects non-finite coordinates, and has no caller-adjustable nearest-grid tolerance.
- Gross task-owned source/script reduction: **4,733 lines**. Gross task-owned test reduction after replacement archive/lineage tests: **1,608 lines**.
- Retirement archives:
  - `local_archive/retired_surface/tbg_b0_benchmark_workflow_20260916/`, manifest SHA-256 `71befcfd61aa2366a16785f054ef262b936f5712add6be5852c610b768165ef4`.
  - `local_archive/retired_surface/tbg_b0_reference_loader_pruning_20260916/`, manifest SHA-256 `ff660d31c1a064539ad9b42febc3bac084efe895db041720e02554a3ee6a3351`.
- Focused TBG/API/archive gate: `205 passed`.
- Broad gate excluding an unrelated concurrent TPT-bulk collection failure: `1272 passed, 6 skipped, 4 failed`; the four failures remain the pre-existing generic TDHF ordering, two RLG/hBN fixed-quotient/C3 checks, and TBG finite-field mixed-convergence acceptance.
- Read-only reviewer verdict after four fail-closed review/fix rounds: **GO**.

### split_inas_gasb_projected_owners_and_bind_matrix_ei_interactions

- Replaced the six-class `isinstance` ladder inside `matrix_ei.py` with one exact-type, fail-closed interaction binding owner in `matrix_ei_interaction.py`. Unknown duck types and subclasses cannot self-assert reference policy, Hartree/Fock decomposition, electrostatic ensemble, model label, or authority.
- `MatrixEIResult` now preserves `interaction_kind`, `interaction_model_label`, and `interaction_authority`; mixed electrostatics, diagnostic reductions, and explicit precomputed tensors retain their lower authority in `physics_status`.
- Split and deleted the 1,037-line mixed `projected_model.py` owner with no compatibility façade:
  - `kane4_bundle.py`: source-bound bundle, canonical I/O, frame lifting;
  - `projected_vertices.py`: local/integrated density vertices and reciprocity;
  - `projected_fock.py`: Green kernels, direct/tensor Fock actions, energy/gauge/order helpers.
- Migrated all live source, `ei_mf`, tests, and current API docs to the new canonical owners. The InAs/GaSb package-root façade remains exactly six high-level symbols.
- Focused validation: `tests/test_inas_gasb*.py` — `144 passed`; cross-package `ei_mf` plus API import gate — `15 passed`. Broad gate excluding the unrelated concurrent TPT-bulk collection failure: `1273 passed, 6 skipped, 4 failed`, with the same four pre-existing unrelated failures.
- Read-only reviewer verdicts: interaction Protocol **GO**; projected-owner split **GO** after current-snapshot cleanup.
- Exact-byte archives:
  - `local_archive/retired_surface/inas_gasb_matrix_ei_concrete_dispatch_20260916/`, manifest SHA-256 `Dedb27c64a04521970524477e5ec2ef280f1def33883179815a1b3dc07b8a7fe`;
  - `local_archive/retired_surface/inas_gasb_projected_model_owner_split_20260916/`, manifest SHA-256 `5ab7ae478e8b6a39a0cc6863eed26c837c640269d1f4a0b753b86f5f6fe0cf54`.

### unify_hf_registry_and_retire_historical_handoffs

- Made the typed HF adapter registry the sole public `run_hf` execution authority. `_hf_dispatch.py` now enumerates `adapter_type="run_hf"` entries in registry order and contains no system-name tuple, TDBG special branch, or `model.run_hf` duck hook.
- Moved TDBG config validation/execution into `systems/tdbg/projected_hf_contracts.py`; every run entry points directly to a system owner rather than back to the public dispatcher.
- Public `HFConfig`, all five registered model types, and all five explicit system config types require exact concrete types. Unregistered subclasses cannot inherit execution authority.
- Retired the `projected_config` TDBG alias and ten zero-logic converter forwarding aliases from `mean_field.api.hf`; conversion capabilities remain available through `resolve_hf_adapter(...)` and their canonical system owners.
- `_hf_dispatch.py` shrank from 162 to 65 lines; control-plane logic is now single-path even though TDBG owner validation moved intact into its system module.
- Archived and removed 25 tracked historical session handoffs, superseded translation/review documents, and retired B0 Julia-port notes: 5,106 documentation lines. Current architecture/API documents remain authoritative.
- Removed the unreferenced 454-line Haldane Checkpoint-A CLI/plot/artifact workflow from the generic injection-current package. The independent toy Hamiltonian, analytic derivatives, BZ quadrature, spectra, and local response kernels remain; the Checkpoint-A-specific zone-average scan retired with the workflow.
- Focused registry/TDBG gate: `72 passed`; extended API/system-contract gate: `89 passed`; injection/optical gate: `15 passed`; full repository gate: `1306 passed, 6 skipped, 4 failed`, with the same four pre-existing unrelated failures.
- Read-only reviewer verdicts: HF registry **GO**; Haldane workflow retirement **GO**.
- Exact-byte archives:
  - `local_archive/retired_surface/hf_registry_multi_dispatch_compat_20260916/`, manifest SHA-256 `720739bef8e7d1d8a2e30d779987d8a6c12aa4e69988cd50342f9bb81c776643`;
  - `local_archive/retired_surface/hf_registry_exact_model_types_20260916/`, manifest SHA-256 `53d6751f7238e864fccc14492be2e248e144df1868bee6f8200fa8190f2559aa`;
  - `local_archive/retired_surface/historical_handoffs_and_review_docs_20260916/`, manifest SHA-256 `532f1c30f414560b658ab46897a5e230a9facd191adac201686921fc3a4de8a8`;
  - `local_archive/retired_surface/haldane_checkpoint_workflow_20260916/`, manifest SHA-256 `F1e6e0d7afbe4980a7ec5c1b837a446c531ba35e7464b195c5cbc825a1a1cfb6`.

### retire_engineering_compatibility_aliases

- Corrected plotting ownership by moving the exact 232-line implementation from `mean_field.plotting` into canonical `mean_field.core.plotting.bands` and deleting the 38-line reverse façade.
- Deleted the 13-line PtSe2 distributed-eigensolver compatibility module; the PtSe2 package root now imports the exact generic core objects directly.
- Deleted three zero-logic PtSe2 2x2 builder aliases. Tests now call the generic builders with explicit `ptse2_2x2_supercell()`; physical-Q oracles and 2x2 initialization/run policy owners remain.
- Removed the historical `ValidationCheck(val=...)` / `.val` alias in favor of the canonical `value` field.
- Removed the ignored `fd_step_nm_inv` keyword from the TDBG shift-current adapter chain; the independent finite-difference derivative oracle remains, while response production continues to use analytic `dH/dk`.
- Required explicit prefactor and phase at the low-level shift-current conversion boundary. The optical front-door historical defaults remain numerically unchanged and are now passed explicitly; the named WannierBerri positive-transition conversion remains separate.
- Retained the unique Arora/Liu–Dai TBG Hamiltonian and optical-payload capability, but removed three zero-caller Fig.2 path/band/energy-zero postprocessing leaves. Removed three zero-caller HTQG validators whose implementations did not test the chiral, symmetry, or decoupled-Dirac properties named; the real Hamiltonian/chiral/symmetry owners and lightweight checks remain.
- Gross task-owned reduction: 201 physical source lines and two source modules.
- Focused gates: engineering/response `69 passed`; Arora/Liu–Dai/HTQG `20 passed`; final HTQG `10 passed`. Full repository gate: `1311 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only reviewer verdicts: engineering aliases **GO**; shift-current conversion boundary **GO**; Arora leaves/false validators **GO**.
- Exact-byte archives:
  - `local_archive/retired_surface/engineering_compat_aliases_20260916/`, manifest SHA-256 `3710d78fd3fdc8b6c8d81cc53ed7b63a7475361a2ad63b2e888a2a2ccf978138`;
  - `local_archive/retired_surface/shift_current_implicit_prefactor_20260916/`, manifest SHA-256 `444324fab6e2d2a67321147c8716948f321fe87e2be084a7d598e10ccf5b661d`;
  - `local_archive/retired_surface/unowned_paper_path_and_false_validation_helpers_20260916/`, manifest SHA-256 `02a2129f643cef19c91973fd2d00b03c7be977e7e570359b49aab89ed4289265`.

### narrow_facades_and_retire_unowned_workflows

- Narrowed the PtSe2 package root from 104 low-level re-exports to nine typed primitive/supercell HF front-door symbols. All PAO, reciprocal, form-factor, physical-Q, symmetry, solver-variant, and distributed-eigensolver owners remain available from their explicit modules.
- Retired the zero-caller RLG/hBN path-band cache and cache-manifest workflow while retaining screening, projected-basis and overlap caches required by TDHF archive replay.
- Retired the uninstalled `ei_mf` argparse/plot/artifact workflow (`cli.py`, `io.py`, `plots.py`, 1,001 source lines). The complete numerical API, generalized CNP channel conventions, Coulomb kernels, Kane helpers, thermodynamics and Zhu1995 owners remain. Active and historical documentation now marks the CLI and script layout as archived.
- Reduced `mean_field.runtime` to the live login-node guard; Slurm wrappers remain the authority for thread configuration.
- Retired the unconnected workflow scheduler-prediction layer while retaining manifests, run-state/failure reports, Slurm metadata and artifact writers.
- Retired the 118-line `mean_field.core` broad re-export façade. Stable high-level access remains under `mean_field.api`; low-level capabilities are imported from explicit owner modules.
- Gross task-owned reduction: 1,750 physical source lines and three source modules.
- Focused gates: PtSe2/RLG `94 passed`; `ei_mf` plus InAs/GaSb `219 passed`; runtime/workflow/core `50 passed`; explicit core-owner import `32 passed`.
- Full repository gate: `1315 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only reviewer verdicts: PtSe2 root **GO**; RLG cache **GO**; `ei_mf` workflow **GO** after documentation closure; runtime **GO**; workflow scheduler **GO**; core root **GO** under the accepted low-level compatibility-breaking policy.
- Exact-byte archives:
  - `local_archive/retired_surface/ptse_root_rlg_path_cache_20260916/`, manifest SHA-256 `E258b965d17ace06eb27ccc5fddb0656e3edcb3a503dece464e8a906be7347d4`;
  - `local_archive/retired_surface/ei_mf_cli_workflow_20260916/`, manifest SHA-256 `Cca318357d320065b27567aa1ef449c229d60dbf14e1e48a8c83dd8d8473b0a6`;
  - `local_archive/retired_surface/ei_mf_cli_workflow_doc_followup_20260916/`, manifest SHA-256 `Cd03337a4f9d7271ff1b64c2f11678a139d9ae99c858e4335908acdb07c8f9b6`;
  - `local_archive/retired_surface/runtime_scheduler_core_root_facades_20260916/`, manifest SHA-256 `B3776b43d2f17afe280c31c85081d975178877c603c31b70add067a6c7a778cd`.
### narrow_remaining_system_roots

- Narrowed six remaining broad system roots to typed front doors while retaining every numerical owner:
  - RLG/hBN: about 157 exports → `RLGhBNModel`, `RLGhBNParams`, `RLGhBNInteractionParams`, `RLGhBNRunHFConfig`;
  - HTG: 106 exports → model, parameters, interaction parameters and primitive/supercell HF configs;
  - TDBG: 57 exports → model, parameters, interaction settings, projected-HF config and projected window;
  - TMBG and ATMG: model/parameter pairs;
  - HTQG: model, parameters, domain and explicit-displacement typed inputs.
- Migrated the registered RLG finite-q TDHF devtool, common API validation and all affected tests to explicit physical owner modules. This changes import ownership only; TDHF archive/provenance authority and its two existing fixed-quotient failures remain a separate unresolved issue.
- Removed the TDBG HF dispatcher’s reverse package-root dependency; it now imports build/run operations directly from `projected_hf_data` and `projected_hf_run`.
- Removed the zero-caller RLG `reproduce_paper_checkpoints` stub, which only returned a retired/skipped report. The live structural validation and screened-U scientific checkpoint remain.
- Archived and removed four stale tracked workflow documents whose advertised HTG/TMBG commands no longer exist; numerical Hamiltonian/topology/HF owners and historical archive copies remain.
- Gross task-owned reduction: 777 physical source lines. Additional tracked-document retirement: 335 lines across four files.
- Focused gates: five-root migration `216 passed, 6 skipped`; RLG migration `141 passed, 2 failed` with the same two fixed-quotient baselines.
- Full repository gate: `1318 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only reviewer verdicts: five-root façade migration **GO**; TDBG owner routing **GO**; RLG root migration **GO** when strictly scoped to import/API ownership; stale-doc/RLG stub scientific review **GO** after this report closure.
- Exact-byte archives:
  - `local_archive/retired_surface/system_root_facades_and_stale_workflow_docs_20260916/`, manifest SHA-256 `F05468ba6acbb16c771d2cc76f5258b93e5aea7b70387add5843881182c61719`;
  - `local_archive/retired_surface/system_root_facades_test_migration_followup_20260916/`, manifest SHA-256 `Dcc6e56e2af726162bc5f135925b882eadedc17620fa48bb1a919ed553bbec8c`;
  - `local_archive/retired_surface/rlg_hbn_root_facade_20260916/`, manifest SHA-256 `205e161f2b1f271e40032d908b5c39bc672bbfa772682b2974dcf52502b1c05a`.

### narrow_tbg_nested_roots

- Narrowed `mean_field.systems.tbg.zero_field` from 126 re-exports to nine typed model/config/data/state/build symbols. Artifact, TDHF, overlap, path/advisor and solver internals remain in explicit owner modules.
- Narrowed `mean_field.systems.tbg.finite_field` from 72 re-exports to 19 typed spectrum/HF lifecycle symbols. LL formulas remain in `spectrum`, TBG spectrum-to-core adapters remain in `hf`, generic magnetic bookkeeping remains in `core.magnetic_field`, and generic finite-B HF remains in `core.hf.finite_field`.
- The parent TBG root now imports its model directly from `zero_field.model`; exact parent/nested `__all__` order and owner identity are locked in `tests/test_system_root_apis.py`.
- Repaired a now-invalid HF-contract guidance string to name the retained `solve_bm_model_on_torus` API.
- Archived and removed three stale HTG workflow plans describing nonexistent scripts, retired system-local topology, and retired shift-current workspaces. Their 1,427 lines remain available in the exact-byte local archive but no longer claim current authority.
- Gross task-owned source reduction: 361 physical source lines. Additional historical-plan retirement: 1,427 lines across three files.
- Focused gate: `225 passed, 1 failed`; the failure is the existing TBG finite-field mixed-convergence baseline. Root/guidance follow-up: `7 passed`.
- Full repository gate: `1319 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only reviewer verdicts: zero-field root **GO** after guidance repair; finite-field root **GO**; stale HTG plans **GO**.
- Exact-byte archives:
  - `local_archive/retired_surface/tbg_nested_root_facades_and_stale_htg_plans_20260916/`, manifest SHA-256 `Fb88e41398cf607096bbf67ec63d66354d039caa556e44db782e5c375d906dee`;
  - `local_archive/retired_surface/tbg_nested_root_contract_guidance_followup_20260916/`, manifest SHA-256 `1cb7ce19dd31f684f2be40f7abb4250d43414f588c00c25139874b9579a3f4bd`.

### retire_core_hf_flat_root

- Retired the 733-line, 369-symbol `mean_field.core.hf` aggregate façade. The package remains an empty namespace with `__all__ == ()`; stable user-facing execution remains under `mean_field.api`.
- Migrated all tracked absolute and relative flat-root imports to their exact original owner modules. This includes archive-vs-density compatibility bindings, finite-field public owners, overlap/occupation/interaction/SCF owners, signed-q TDHF, finite-temperature TDHF, scalar curvature, and the explicit `tdhf_scalar_functional` ABI.
- No numerical implementation, function body, default, class definition, `__module__` normalization, density orientation, ODA rule, or TDHF formula was changed.
- Added a fail-closed layering gate that parses absolute and relative imports, rejects parent-package `hf` imports, and locks the root AST to a docstring plus empty `__all__` assignment.
- Updated README, architecture and the path-scoped HF breadcrumb to describe the empty namespace and explicit-owner policy. The final stale finite-field docstring now names `engine`/`problem` owners rather than the retired flat root.
- Gross task-owned source reduction: 632 physical source lines.
- Focused affected gate: `697 passed, 6 skipped, 4 failed`; boundary/API follow-up: `46 passed`.
- Full repository gate: `1323 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only reviewer verdicts: import/ABI closure **GO**; owner/physics-path parity **GO**; breadcrumb/archive/process review **GO** after gate and migration-note closure.
- Exact-byte archives:
  - `local_archive/retired_surface/core_hf_flat_root_facade_20260916/`, manifest SHA-256 `9ab642f0b9c633bfe6ee35d16a8207358cc638b22cf4e35c3268388b4330cb2b`;
  - `local_archive/retired_surface/core_hf_relative_import_followup_20260916/`, manifest SHA-256 `68ad5615e4d1c2996c8627b3ba045da72b02c97b83feedad2aec7c6c2290e4e2`;
  - `local_archive/retired_surface/core_hf_owner_guidance_followup_20260916/`, manifest SHA-256 `759331bed7d9b80e36456a2974081736ea5fbe4dc36b7d6fa04338a93babf3de`;
  - `local_archive/retired_surface/core_hf_runner_guidance_followup_20260916/`, manifest SHA-256 `61aadb222e038fe453f7279f39fa50597601bc683dd83633a5e78ffb699facb8`.

### retire_internal_hf_and_plotting_facades

- Deleted the 108-line `mean_field.systems.tdbg.projected_hf` aggregate compatibility module. Public dispatch continues through `projected_hf_contracts -> projected_hf_data/projected_hf_run`; the five-symbol TDBG package root is unchanged. The uncalled multi-run `scan_tdbg_projected_hf_states` workflow was not promoted to an owner.
- Deleted the 121-line `mean_field.systems.tbg.zero_field.hf` pure re-export module. State/provenance, restricted/full solvers and diagnostics now come directly from `_hf_basis_overlap`, `_hf_restricted`, `_hf_full`, and `_hf_diagnostics`; common overlap/interaction capabilities come from their core owners.
- Reduced `mean_field.core.plotting` from an 11-symbol aggregate to an empty namespace. Shared plotting remains in `mean_field.core.plotting.bands`.
- Removed five pure façade-identity test files and only the compatibility assertions from three mixed owner-behavior tests. Functional owner, public-dispatch, artifact, exact-grid, TDHF and plotting tests remain.
- Added absence contracts for both retired HF modules and an AST-locked empty plotting-root contract.
- Gross task-owned reduction: 243 source lines and 64 test lines.
- Focused gate: `226 passed`.
- Full repository gate: `1319 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only review: TDBG owner/dispatch closure **GO**; TBG zero-field owner/identity closure **GO**; plotting/docs/archive closure **GO** after documentation and AST-gate follow-up.
- Exact-byte archive: `local_archive/retired_surface/internal_hf_and_plotting_facades_20260916/`, manifest SHA-256 `de3c7ce9518e9957a19baf70c4531047a67a867b4e0ba7514e4f8923ee54e67d`.

### narrow_tbg_finite_hf_and_retire_htg_adapter

- Reduced `mean_field.systems.tbg.finite_field.hf` from 421 to 271 lines and from 50 exports to six TBG-owned builders/paper-flux helpers. Generic types and operations now come directly from `mean_field.core.hf.finite_field` and `mean_field.core.magnetic_field`.
- Retired the two system-level `build_tl_symmetric_finite_field_hf_inputs_from_*` forwarding wrappers. The unified builders retain the same reduced path through explicit `reduced_translation=True`.
- Deleted the 13-line wildcard `mean_field.systems.htg.mean_field_adapter` aggregate. Registry authority and all primitive HTG provenance strings now name `_hf_contracts`; raw execution remains in `_hf_runner`.
- Replaced transitive wildcard imports and dynamic `globals()` exports across all seven primitive HTG HF owner modules with explicit dependencies and owner-only `__all__` contracts. Numerical function bodies were unchanged.
- Compatibility-surface reduction was 162 source lines before explicit-owner hardening; the explicit dependency lists add 158 lines, for a net task-owned source reduction of four lines. The change is primarily an authority-boundary reduction rather than a line-count reduction.
- Added exact gates for the six-symbol TBG adapter, retired-wrapper absence, all seven HTG owner export lists, wildcard absence, registry import paths, and five primitive HTG provenance owner strings.
- Final focused gate after owner hardening: `135 passed, 1 failed`; the sole failure was the established `test_finite_field_problem_preserves_author_mixed_density_convergence_rule` baseline.
- Full repository gate: `1320 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Read-only review: TBG finite-field API **GO**; primitive HTG owner/registry/provenance closure **GO**; documentation/archive/test-contract closure **GO** after the final focused-gate update.
- Exact-byte archives:
  - `local_archive/retired_surface/tbg_finite_hf_reexports_and_htg_adapter_20260916/`, manifest SHA-256 `eda8f359535e9f5e507d68bf5dbe855bf67c3a733e0974ca2b9cafa205f6c83b`.
  - `local_archive/retired_surface/htg_wildcard_owner_hardening_followup_20260916/`, manifest SHA-256 `6696d73f8bfdd22e9f461d756c3a130b6f33ce8a26afb00c2cef9443b9bb87ab`.
  - `local_archive/retired_surface/tbg_htg_contract_gate_followup_20260916/`, manifest SHA-256 `1da7b2d39ff04c23b1e0471247c2ccea6d21cbc497e024fc1572f94abbc791cf`.

### harden_htg_supercell_owner_boundaries

- Replaced transitive wildcard imports and dynamic `globals()` exports across `_supercell_types`, `_supercell_geometry`, `_supercell_basis`, `_supercell_runner`, and `_supercell_path_io` with explicit canonical-owner dependencies and literal owner-only `__all__` contracts.
- Reduced `_supercell_shared` from a 53-line transitive aggregate to an empty three-line namespace. All consumers now import directly from the numerical or type owner.
- Preserved every local class/function definition and body byte-for-byte relative to the retirement archive. The 36-symbol `mean_field.systems.htg.supercell` façade, all object identities, and the private `_supercell_reference_density_blocks` bridge remain unchanged.
- `supercell_contracts` now imports required objects directly from their owners while preserving its four-symbol API, registry paths, schemas, and five provenance strings.
- Added exact gates for six owner export lists, wildcard/dynamic-export absence, the 36 façade identities, the private bridge, all three registry paths, and five provenance owner strings.
- Focused gate: `68 passed`.
- Full repository gate: `1439 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- Across the archived task source slice, explicit dependencies increased source by a net 81 lines; the focused layering test increased by 90 lines. This is authority hardening, not LOC reduction.
- Exact-byte archive: `local_archive/retired_surface/htg_supercell_wildcard_owner_hardening_20260916/`, manifest SHA-256 `8246f4f22533da8ffe426c56a445f3e86783c2ed1f012a7fc067f2750ba4fd60`.
- Contract-gate/report follow-up archive, recovered from verified source authority v23: `local_archive/retired_surface/htg_supercell_contract_gate_followup_20260916/`, manifest SHA-256 `becd2c94d377c5d41d718079344bc93648e89b30ecd709d4f8d1016c7f7fc311`.

### retire_tmbg_polshyn_facade_and_harden_owners

- Replaced wildcard/transitive dependencies and dynamic `globals()` exports in `_polshyn_types`, `_polshyn_filling`, `_polshyn_wang`, and `_polshyn_contracts` with explicit canonical-owner imports and literal owner-only export lists (5, 5, 17, and 1 symbols respectively).
- Retired the empty aggregate `_polshyn_shared.py` and pure re-export `polshyn_supercell.py` façade. The registered canonical adapter now resolves directly to `mean_field.systems.tmbg._polshyn_contracts:polshyn_wang_hf_bundle_to_hf_run_result` with owner identity preserved.
- Preserved all local class/function bodies byte-for-byte against the retirement archive except the three requested new-artifact guidance/source/adapter strings in `_polshyn_contracts`; the Wang stored-projector orientation, `stored_delta`, `P_store - R`, `stored_abk`, axis declarations, `supports_crpa=False`, and system identity metadata remain unchanged.
- Added static owner-layering, deleted-module, exact registry-path, and resolved-owner-identity gates. Stdlib AST/symtable checks found no wildcard imports, dynamic `globals()`, or unresolved globals; owner and scoped-test import smoke plus `compileall` passed.
- Migrated the final `tests/test_core_supercell.py` caller directly to `_polshyn_filling`; no live source or test imports the retired façade/shared modules.
- Task-owned net change: `-34` source lines and `+86` test lines; the extra tests lock owner exports, module retirement, registry identity, and metadata boundaries.
- Focused gate: `68 passed`.
- Full repository gate: `1440 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures.
- The worker archive `local_archive/retired_surface/tmbg_polshyn_facade_and_wildcards_20260916/` preserves 11 exact payloads but uses a legacy lowercase layout. The superseding project-format archive is `local_archive/retired_surface/tmbg_polshyn_facade_and_wildcards_contract_followup_20260916/`, manifest SHA-256 `36a6a664d9458a8faa068aab1f9cdb7a8e27b9b0fb3d97359781e20728d488fc` (12 payload files, including the final caller).

## Current local-tree snapshot (2026-09-16)

- All `src/**/*.py`: 347 files, 134,889 lines, 31 files over 1,000 lines.
- `src/mean_field/**/*.py` + `src/analysis/**/*.py`: 326 files, 129,976 lines.
- Current `tests/test_*.py`: 138 files, 51,933 lines.
- Current InAs/GaSb package: 27 Python files, 16,607 lines; its 13 focused test files contain 6,379 lines.
- Gross task-owned retirement is used instead of whole-tree net attribution because unrelated concurrent work continued.

## Top 30 Python files under `src` (current local tree)

| Lines | Path |
|---:|---|
| 3960 | `src/mean_field/devtools/tpt_bulk_paw_oracle.py` |
| 2822 | `src/mean_field/systems/htqg/microscopic_supercell.py` |
| 2102 | `src/mean_field/systems/htqg/microscopic_hf.py` |
| 2094 | `src/mean_field/core/hf/tdhf_scalar_curvature.py` |
| 2060 | `src/mean_field/systems/tbg/zero_field/_hf_basis_overlap.py` |
| 2026 | `src/mean_field/systems/RnG_hBN/_tdhf_fixed_quotient.py` |
| 2001 | `src/mean_field/systems/inas_gasb/kdotpy_window_builder.py` |
| 1989 | `src/mean_field/core/hf/tdhf_scalar_functional.py` |
| 1933 | `src/mean_field/systems/htqg/hf.py` |
| 1928 | `src/mean_field/systems/inas_gasb/kdotpy_poisson_adapter.py` |
| 1920 | `src/mean_field/systems/ptse2_openmx/supercell.py` |
| 1487 | `src/mean_field/systems/tbg/zero_field/tdhf.py` |
| 1481 | `src/mean_field/systems/tpt_bulk/microscopic_hf.py` |
| 1475 | `src/mean_field/systems/tbg/zero_field/model.py` |
| 1468 | `src/mean_field/core/hf/tdhf_finite_temperature.py` |
| 1454 | `src/mean_field/systems/tbg/zero_field/hf_contracts.py` |
| 1428 | `src/mean_field/systems/RnG_hBN/_hf_basis.py` |
| 1426 | `src/mean_field/systems/inas_gasb/kane_poisson.py` |
| 1402 | `src/mean_field/core/hf/tdhf_signed.py` |
| 1373 | `src/mean_field/systems/tise2/folded_hf.py` |
| 1335 | `src/mean_field/systems/RnG_hBN/_hf_response_finite_q.py` |
| 1207 | `src/mean_field/core/hf/density_vertex.py` |
| 1195 | `src/mean_field/systems/RnG_hBN/_tdhf_finite_q.py` |
| 1167 | `src/analysis/optical/passos.py` |
| 1157 | `src/mean_field/systems/tpt_bulk/wavefunctions.py` |
| 1148 | `src/mean_field/systems/inas_gasb/axial_fock.py` |
| 1140 | `src/mean_field/core/hf/tdhf.py` |
| 1080 | `src/ei_mf/gap_solver_generalized.py` |
| 1073 | `src/mean_field/systems/tbg/zero_field/artifacts.py` |
| 1073 | `src/analysis/shift_current/core.py` |

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

### harden_analysis_namespace_exports

- Replaced wildcard package initialization in `analysis.shift_current`, `analysis.injection_current`, `analysis.optical`, and `analysis.optical.benchmarks` with explicit leaf imports and literal ordered `__all__` contracts of 45, 11, 91, and 37 symbols respectively. Intentional wildcard-compatibility break: package-level `from ... import *` no longer binds leaf module handles such as `core`, `kmesh`, or benchmark modules; those handles are deliberately omitted from `__all__` and the star namespace, although normal Python package attributes and direct leaf import paths remain available. Maintained function/class/constant exports retain identity with their leaf owners. No response formula, normalization, convention, or numerical implementation changed.
- Migrated the Liu--Dai benchmark test and documentation example to direct leaf-owner imports. Added `tests/test_analysis_namespace_exports.py` to lock exact export order/counts, owner identity, star-import namespaces, and the AST prohibition on wildcard imports or computed `__all__` values.
- Task-owned net change: `+395` source lines and `+359` test lines. This is deterministic API-authority hardening rather than LOC reduction; the added lines make every maintained export and identity gate explicit.
- Static `compileall` and clean-process import/star/identity smoke passed. Focused response/API gate: `128 passed`; full repository gate: `1443 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures. No response-grid production calculation was run.
- Exact-byte archive: `local_archive/retired_surface/analysis_explicit_namespace_exports_20260916/`, MANIFEST SHA-256 `1d0a0e4ac5177b04a78f17e22f033e29fd0ef52e213777fa14096369ec078f57`; `COMPLETE` present.

### retire_tdbg_package_cli_forwarding

- Removed the duplicate TDBG package-CLI forwarding path from `scripts/mean_field_tools.py`. The script is now a developer-tool dispatcher only; canonical TDBG commands belong to `mean_field.cli` and are invoked with `PYTHONPATH=src python -m mean_field.cli tdbg ...` or the installed `mean-field` entry point.
- Intentional compatibility break: `python scripts/mean_field_tools.py tdbg ...` now fails with `Unknown command` rather than forwarding into the package parser. No TDBG parser, dispatch handler, numerical owner, or physics behavior changed.
- `scripts/mean_field_tools.py`: 102 -> 67 lines (`-35` net). Remaining old package-CLI forwarding entries: 0. The exact four retained developer commands are `merge_tbg_crpa_chunks`, `run_tbg_crpa_chunk`, `qualify_tpt_bulk_parent_wavefunctions`, and `tpt_bulk_paw_oracle`.
- README and script-surface policy now distinguish the package CLI from the developer dispatcher. Static tests lock the exact retained command set, forwarding absence, help surface, rejection of `tdbg`, and canonical package-CLI parser availability.
- Exact-byte archive: `local_archive/retired_surface/tdbg_cli_forwarding_20260916/`, manifest SHA-256 `79b08171a23794e72c9cd6e816a851a2f284a0faee0bb3b16183cc28ac4d6355`.
- Focused dispatcher/package-CLI gate: `10 passed`; full repository gate: `1448 passed, 6 skipped, 4 failed`, with the same four unrelated baseline failures. The archive receipt correctly records that pytest had not yet run at seal time.

### harden_crpa_devtools_fail_closed

- Retained both cRPA developer commands and their existing computational owners while adding the shared login-node guard before cache loading, output creation, workflow-artifact writes, or plotting initialization. No public-API migration, cRPA formula/form-factor/occupation change, command deletion, `mean_field.api.compute_crpa` enablement, or `compute_crpa_from_solution` change was made.
- Added pure request/selector validation to the chunk runner. Intentional fail-closed compatibility breaks reject mixed selectors, unpaired chunk arguments, clipped/out-of-bounds or empty selections, nonpositive/even shells, nonfinite/invalid dielectric-distance/broadening values, and Eq.19 overlap shells outside `eq19_flat_remote` mode.
- Added strict five-file, parameter, NPZ-label, integer-coordinate, shape/first-dimension, cross-file, and cross-chunk metadata/q-shift validation to merge. Canonical ordering is now the stable sort of `i + lk*j`; duplicate and out-of-range coordinates always fail. Default merge requires exact `lk^2` coverage, while explicit `--allow-partial` permits only a nonempty valid duplicate-free subset and records that coverage policy in the workflow command/metadata and merged validation report.
- Preserved the existing merged NPZ keys, key order, dtype conversions, shapes, numerical values, coverage metadata, base artifact filenames, and normal successful serialization/diagnostic path. The authorized merge-owner consolidation is complete: `src/mean_field/crpa/chunk_merge.py` now uniquely owns validation, coverage/canonical sorting, metadata aggregation, and the five-file base writer; the devtool retains CLI/workflow/presentation orchestration. Public `mean_field.api.compute_crpa` remains disabled and production cRPA physics remains unvalidated.
- Owner extraction evidence: the merge devtool was reduced from 1016 to 328 lines; the new package owner is 719 lines. Pre/post semantic golden parity passed for exact JSON bytes and unpacked NPZ key order, dtype, shape, and value bytes across all five base artifacts. Archive: `local_archive/retired_surface/crpa_chunk_merge_owner_extraction_20260916/`; MANIFEST SHA-256 `75ceb97c9a7d8ac7b9b5ccbcbc854fb8a7b1ea2c1c11eed076d3100d01850fae`. The pinned pre-move source-authority archive is v29, SHA-256 `f26e0186ee83572cbc0e030c8f1327513d4cd4db99d2b8ac8123edd9c6850758`. Post-extraction focused gate: `32 passed`; full repository gate: `1475 passed, 7 skipped, 4 failed`, with the same four unrelated baseline failures. No production cRPA calculation or new physics validation was run.
- Added `tests/test_crpa_devtool_contracts.py` with synthetic/monkeypatch contracts for static arguments, pure selection, guard-before-write/cache/run ordering, duplicate/gap/out-of-range/partial coverage, NPZ label/shape/coordinate mismatch, canonical sorting, and workflow coverage metadata. Per task restriction, pytest and numerical cRPA were not run; targeted `compileall`, import/parser/selector/manifest helper smoke, and a temporary synthetic full-coverage merge-validation smoke passed.
- Exact-byte pre-edit archive: `local_archive/retired_surface/crpa_devtool_fail_closed_hardening_20260916/`, MANIFEST SHA-256 `58f3c825bee0b735e1e578c784025703981a1b52afb198a1bcfe7acc2f694237`; receipt records no git, pytest, or results use, and `COMPLETE` is present.

### harden_crpa_devtools_fail_closed_followup

- Closed the remaining merge-boundary fail-open cases without changing cRPA formulas, results, public API, top-level parameter schema, base filenames, or NPZ keys. Merge now uses repository strict JSON parsing, exact typed/finiteness/domain validation for parameter and Coulomb schemas, canonical q-shift inventory validation, pre-cast dtype/finiteness checks, nonnegative q norms, and tight writer-semantic cross-NPZ redundancy checks.
- Fixed valid multi-chunk metadata handling: the three q-subset-dependent periodic-roll fields are aggregated by `max`, `max`, and logical `all`; every other metadata field remains exact. Actual partial/full coverage is written into merged `crpa_params.json` metadata and the validation report.
- Workflow manifests now record replayable dispatcher commands using `sys.executable`, the absolute repository `scripts/mean_field_tools.py`, `repr` float arguments, and the repository working directory. Input-chunk jobs are marked provenance-only and remain pending/awaiting validation until all loaders and validators succeed; failed validation no longer reports inputs as present/succeeded.
- The earlier hardening tests used a hand-authored synthetic five-file loader fixture only. The follow-up fixture now constructs chunks through the real `CRPAResult -> write_crpa_outputs` writer, checks `load_crpa_result` roundtrip, and adds a tiny `_run_merge` success contract for the five base files plus canonical q shifts, strict JSON, scalar/array dtype and NaN rejection, redundant-field tampering, dynamic metadata aggregation, coverage metadata, and pending input state.
- Validation run in this follow-up: targeted `compileall`, imports/command construction, and one manual temporary real-writer synthetic full-coverage `_run_merge` smoke passed. Pytest was deliberately not run, so the newly added and retained tests remain unexecuted. No production cRPA numerical calculation or physics validation was run or claimed.
- Exact-byte pre-edit follow-up archive: `local_archive/retired_surface/crpa_devtool_fail_closed_followup_20260916/`, MANIFEST SHA-256 `06787cd1cca320fbfdc5a7c24101b1468a45b07872da6914761b4a41977a9956`; receipt records no git, pytest, or results modification, and `COMPLETE` is present.

### harden_crpa_devtools_fail_closed_followup2

- Closed the second-round cRPA devtool review blockers without changing cRPA formulas, computational owners, public API, result trees, top-level parameter schema, base artifact filenames, or NPZ labels. Cross-NPZ validation now tightly binds `effective_epsilon` to `real(diag(epsilon))` and binds `q_tilde_real`/`q_tilde_imag` to the unique canonical `(0, 0)` q-shift column of `q_real`/`q_imag`.
- Dynamic periodic-roll metadata is now a per-chunk all-missing/all-present typed triple. Present values are checked against the existing owner formula `required_hf_periodic_lg(q_lg, max_wrap_shell)`, nonnegative wrap, and the exact no-alias predicate before the existing `max`/`max`/`all` reduction. Merge-local `merge_coverage` is excluded from static comparison and regenerated from current coverage, permitting recursively merged valid partial artifacts.
- Static metadata comparison is recursive and type-preserving, with exact dictionary keys and boolean/string/integer values, finite-float tolerance `rtol=1e-12`, `atol=1e-14`, and recursive list/tuple handling. Incompatible scalar or sequence types do not compare equal.
- Chunk and merge replay commands retain `sys.executable` and the absolute dispatcher while resolving BM-solution, output, and chunk arguments to absolute paths. Manifest metadata records the actual launch cwd and replay-command owner. Input chunk records are explicitly provenance-only/non-executable rather than claimed executable jobs.
- Expanded the dedicated contracts with epsilon-diagonal and zero-shift tampering, malformed dynamic triples and relations, non-repository-cwd absolute command construction, recursive partial-output merging, fuller real-writer/loader field parity, literal schema expectations independent of tested constants, and successful multi-chunk `q_lg=3` composition.
- Validation: targeted `compileall` passed; a temporary manual `CRPAResult -> write_crpa_outputs -> load_crpa_result -> _load_and_validate_chunks` two-chunk `q_lg=3` smoke passed. A second temporary manual `_run_merge` partial-to-full recursive writer smoke and non-repository-cwd manifest smoke passed, including nested finite-float metadata tolerance, stale `merge_coverage` replacement, and absolute replay arguments. Subsequent focused engineering gate: `30 passed`; full repository gate: `1473 passed, 7 skipped, 4 failed`, with the same four unrelated baseline failures. No production cRPA calculation or new physics validation was run.
- Exact-byte pre-edit archive: `local_archive/retired_surface/crpa_devtool_fail_closed_followup2_20260916/`, MANIFEST SHA-256 `bb9ddae73f92d22bf5ea047eecdecd0f9de3980850f0599ef13182f046f9834c`; receipt records no git, pytest, or results modification, and `COMPLETE` is present.

### retire_devtool_runtime_owner

- Retired `mean_field.devtools._runtime` without a compatibility facade and deleted its helper-only test. Exact pre-edit bytes for the retired owner, all four maintained callers, affected tests, `devtools/AGENTS.md`, policy, and this report are sealed at `local_archive/retired_surface/devtool_runtime_owner_migration_20260916/`; MANIFEST SHA-256 `a663aef7ca5b2e8e6300c8ec6aef3f72a0b8f126acc952dc014430d16cf9524d`, 13/13 archived files rehashed successfully, and `COMPLETE` matches the manifest hash.
- `merge_tbg_crpa_chunks`, `run_tbg_crpa_chunk`, `qualify_tpt_bulk_parent_wavefunctions`, and `tpt_bulk_paw_oracle` now import the guard directly from `mean_field.runtime`. The two migrated TPT modules call `mean_field.core.io.write_json_artifact(payload, path)` directly; the retired CSV/complex helpers were not migrated. These four mappings were unchanged by the migration. The current dispatcher additionally contains the independently added `tpt_wannier_q0_hf` command, which also uses the canonical runtime and core-I/O owners and was never an `_runtime` caller.
- The TPT qualifier now emits report schema v5 with independent `parent_qualification_selected_provenance_v2` version/scope/hash fields. Its selected evidence includes `runtime.py`, `core/io/__init__.py`, and `core/io/artifacts.py` and explicitly is not a complete import closure.
- The TPT PAW oracle independently uses `paw_oracle_selected_provenance_v2`, including the three direct owners above and `systems/tpt_bulk/source.py`. Active/full seals and postflight reports write the inventory version; historical result trees and seals were not modified and legacy/unversioned seals fail closed.
- One private validator gates active handoff and all four postflights. It enforces domain-specific container/version/map/lowercase-64-hex/current-file/exact-keyset checks, reports complete sorted missing/extra lists, and prevents malformed keysets from reaching external revalidation. Only same-version, same-keyset content drift retains the pre-existing external revalidation route.
- Tests now lock owner imports and retired-module absence, direct JSON ownership/newline behavior, qualifier/oracle inventory versions and exact keys, malformed/legacy/unknown/missing/extra failures, same-key content drift, keyset-drift nonauthorization, active handoff, and the current five-command dispatcher. The old empty-tuple provenance fixture was removed.
- Initial validation was intentionally engineering-only: explicit `compileall`, clean imports, the then-current four dispatcher `--help` paths, direct-owner/inventory inspection, JSON newline smoke, exact/same-key-drift/malformed selected-provenance smoke, and active-handoff fail-closed smoke passed. Subsequent focused and full pytest gates are recorded in follow-up 4 below. Git, Slurm, results, and production physics were not run or modified.
- LOC impact across the 13 archived task paths: `7977 -> 8339` lines (`+362` net), including 62 retired legacy/test lines; source caller/oracle/qualifier growth is contract/provenance hardening rather than physics. No numerical formula or result changed. Remaining uncertainty: pytest was prohibited, so the updated tests are compiled but unexecuted.

### retire_devtool_runtime_owner_followup

- Before this follow-up changed any tracked task file, exact pre-edit bytes for the PAW oracle, parent qualifier, affected tests, script policy, and this report were copied to `local_archive/retired_surface/devtool_runtime_owner_migration_followup_20260916/`. Its six payloads rehashed 6/6; `MANIFEST.tsv` SHA-256 is `23eec1cc59226564fae3d850aaadccaba79316a552d55b03fe1d79ba0a107145`, and `COMPLETE` binds that hash.
- `tpt_bulk_paw_oracle` now reads every JSON trust input used by active handoff, four postflights, submission receipts, external authorization, prior reports, pinned certificates, and shell preflight through `mean_field.core.io.read_json_artifact`; a shared wrapper requires a top-level object. Duplicate object keys and nonstandard `NaN`/`Infinity` constants therefore raise `ValueError` rather than being silently accepted.
- PAW v2 uses exact scope `paw_oracle_v2_selected_set_not_complete_python_import_closure`; qualifier v5 independently uses exact scope `parent_qualification_v2_selected_set_not_complete_python_import_closure`. Both strings explicitly describe selected sets rather than complete Python import closures. PAW preparation fields are `selected_provenance_version`, `selected_provenance_scope`, and `selected_provenance_hashes_at_preparation`; reports add `selected_provenance_hashes_at_validation` and `selected_provenance_drift` at top level.
- The old PAW `runtime_source_*` fields are explicitly classified as legacy and rejected. Missing/unknown scope, missing/unknown version, malformed or uppercase SHA-256, and missing/extra selected keys fail closed. Existing historical artifacts and result trees were not changed and cannot be silently or automatically upgraded to PAW v2.
- One postflight provenance helper now owns record validation, current-inventory hashing, same-version/same-scope/same-keyset content drift, and the call to the existing external revalidation authorization loader. All four actual validators call it exactly once; scope/version/keyset failures occur before that loader. Active handoff accepts the exact current selected set and rejects content drift pending regeneration.
- Behavioral contracts now cover duplicate version/inventory keys, `NaN`/`Infinity`, non-object roots, scope failures, uppercase hashes, legacy fields, all four real validator entrypoints reaching authorization only for same-keyset drift, keyset-drift nonauthorization, active handoff acceptance/rejection, qualifier-v5 and PAW field builders, active/full source-seal field locations, postflight report field locations, JSON trailing newlines, and all four dispatcher command help subprocesses.
- Engineering-only validation performed without pytest: focused `compileall` passed; a manual temporary-directory synthetic smoke passed strict parsing, exact/malformed/drift provenance, all-four-validator authorization routing, active/full producer seals, active handoff drift rejection, qualifier v5 serialization, exact report keys, and trailing newlines; direct source inspection found no `json.loads` and one common-helper call in each validator; all four `<command> --help` subprocesses returned zero. This is not results-byte proof, scientific-result validation, or production-physics evidence. No git command, pytest, results modification, Slurm action, or physics calculation was performed. Remaining uncertainty is that the expanded pytest contracts were compiled but intentionally not executed.
- Follow-up LOC before this report entry, across the five non-report archived paths, is `5488 -> 5996` (`+508`), primarily behavioral contracts; this count excludes this self-modifying report and makes no claim about unrelated repository state.

### retire_devtool_runtime_owner_followup2

- Exact pre-edit bytes for the producer-behavior test follow-up are sealed at `local_archive/retired_surface/devtool_runtime_owner_migration_followup2_20260916/`; MANIFEST SHA-256 `846663fad8fe73df15cca293f61dd74649d58f62abaf13c36176c588e8a71567`, with hash-bound receipt and `COMPLETE`.
- Tiny deterministic backends now drive the real `qualify_parent` report builder/writer and real `prepare_active_wavecar` seal builder/writer. Tests verify qualifier-v5 and PAW-v2 version/scope/exact selected-key placement, strict reread, trailing newline, and read-only active seal without running production-sized PAW or VASP work.
- The follow-up is engineering provenance evidence only. It does not validate physical PAW overlaps, VASP execution, production Slurm behavior, or historical result bytes.

### retire_devtool_runtime_owner_followup3

- Before this test-only follow-up changed either task file, the exact pre-edit bytes of `tests/test_tpt_bulk_paw_oracle.py` and this report were archived under `local_archive/retired_surface/devtool_runtime_owner_migration_followup3_20260916/` using the standard `files/`, `MANIFEST.tsv`, `receipt.json`, and hash-bound `COMPLETE` layout. The `MANIFEST.tsv` SHA-256 is `b3d484a50c98842596f40d5ec439b894ccf86fcefaf59f7be62987cce33ae7aa`; both payloads were rehashed before edits.
- `tests/test_tpt_bulk_paw_oracle.py` now parameterizes all four actual postflight validators over raw malformed `SOURCE_SEAL.json` inputs (duplicate key, `NaN`, `Infinity`, and non-object root), missing/unknown provenance scope, and uppercase 64-hex inventory values. Every case requires endpoint rejection, and the malformed/source-metadata matrices prove the authorization loader was not called.
- The four actual endpoints also exercise the authorization matrix: content drift with no authorization, no drift with an unexpected supplied authorization, and same-keyset drift with strict duplicate-key authorization JSON. These are fail-closed boundary contracts, not authorization success simulations.
- Each actual validator additionally reaches its final real `_write_json_new` through tiny temporary output files. Only science/Slurm boundary readers are replaced; `_validate_postflight_provenance` and `_write_json_new` remain unpatched. The contracts require the returned report to equal a strict reread, the endpoint-specific schema to match, exactly five selected-provenance fields to occur at final-report top level, and the artifact to have a trailing newline, read-only mode, and no `NaN`/`Infinity` token.
- Initial validation was limited to `compileall` and a direct manual temporary-directory smoke of all 36 new parameterized endpoint cases. These are tiny engineering endpoints only: no production code, git command, results tree, Slurm action, numerical physics, or production-physics claim was made. The successful-path science payloads are deliberately tiny monkeypatched boundary values rather than physical PAW data.

### retire_devtool_runtime_owner_followup4

- Final endpoint contracts were pinned before this closure update at `local_archive/retired_surface/devtool_runtime_owner_migration_followup4_20260916/`; MANIFEST SHA-256 `8f29b83d3202b50d3c5caf6c043f4917667d70e276b17dd9cae41cf7672b0407`. The archive contains the final TPT behavior tests, current five-command script-surface contract, policy, and pre-closure report.
- The final-report test no longer patches `_write_json_new`; all four validators publish through the real strict writer. The drift-without-authorization matrix now passes `revalidation_seal_path=None` rather than a nonexistent path. Current dispatcher tests include the separately added `tpt_wannier_q0_hf` command without conflating it with the four `_runtime` migration callers.
- Focused runtime/TPT/cRPA/script/core-I/O gate: `113 passed`. Full repository gate: `1540 passed, 7 skipped, 4 failed`. The four established unrelated failures are `tests/test_core_hf_tdhf_scalar_curvature.py::test_public_factory_rejects_y_lane_source_pair_and_callback_mutations`, `tests/test_rlg_hbn_tdhf_fixed_quotient_anchor.py::test_rlg_hbn_fixed_quotient_is_canonical_anchor_independent`, `tests/test_rlg_hbn_tdhf_fixed_quotient_anchor.py::test_rlg_hbn_fixed_pair_sewing_has_independent_c3_cubed_representation`, and `tests/test_tbg_finite_field_hf.py::test_finite_field_problem_preserves_author_mixed_density_convergence_rule`. The seven skips remain explicit HTQG Slurm-only tests. No production TPT PAW calculation, numerical physics validation, results modification, git operation, or Slurm action was performed.
- Source authority v31: `local_archive/source_authority/mean_field_local_source_20260916_devtool_runtime_retired_v31/Mean_Field_local_source.tar.gz`, SHA-256 `d538aa1a1f88e41c0d44c6dc8b1fa63f6203d8ab1cfbc7bcd9393bac654f78e3`. Exact extraction and extracted-tree `compileall` succeeded; a machine-readable superseding runtime receipt is recorded by the final closure follow-up.

### retire_devtool_runtime_owner_closure_followup5

- Superseding source authority v32: `local_archive/source_authority/mean_field_local_source_20260916_devtool_runtime_closure_v32/Mean_Field_local_source.tar.gz`, SHA-256 `723a027abe854cc77c80b4923c07e599098e9f1472e035aaee9d843b053547b0`.
- Machine-readable closure: `local_archive/retired_surface/devtool_runtime_owner_migration_closure_followup5_20260916/`, MANIFEST SHA-256 `211ad44091fd3eb9a2e6005c4f18d326a2751b24d813163d860d1b496eaffe94`. Its runtime receipt binds the exact source archive, extraction and extracted-tree `compileall` commands, interpreter/runtime identity, output hashes, and zero return codes. Its test summary records the focused/full commands, counts, exact four failed node IDs, and HTQG Slurm-only skip scope.
- This evidence closes engineering/provenance authority only. It does not promote legacy TPT artifacts, validate production PAW overlaps, or establish production physics authority.

## Phase 2 cleanup gates

- No PR should claim cleanup if it only adds wrappers without deleting or thinning legacy paths.
- Each slice must update this report with before/after LOC and remaining old entry points.
- Physics-heavy migrations require focused parity tests before deleting old implementations.
- cRPA/HF bridge changes are deferred until the known cRPA bug is isolated.

### 2026-09-16 — dangling paper scratch-wrapper retirement

- Retired **8** ignored, formula-free scratch wrappers (**353 LOC**): `tmp/htqg_fig1_shell5.sbatch`, `tmp/htqg_fig1_shell5_realistic.sbatch`, `tmp/htqg_fig1_shell6_realistic.sbatch`, `tmp/run_htg_fractional_fillings_20260616.sh`, `tmp/run_htg_fractional_nu3p5_seed2_20260616.sh`, `tmp/run_htg_fractional_nu3p666_20260616.sh`, `tmp/run_waters2024_shell6_hf_chern_test.sh`, `tmp/run_waters2024_screenedbasis_delta_task_20260607.sh`.
- Removed legacy calls to nonexistent dispatcher command(s): `run_htg_hf`, `run_htqg_fig1_bands`, `run_waters2024_band_reproduction`. No dispatcher command was recreated.
- Capability is preserved in the existing HTG/HTQG/Waters numerical owners and oracles; no numerical owner, oracle, result, or log was deleted or modified. Only the eight wrappers were removed.
- Exact evidence is archived at `local_archive/retired_surface/dangling_paper_scratch_wrappers_20260916/`; manifest SHA-256 `cd5afe83d64d5b2be0250bbe97a509fcab9cd759d573eddb61e037da035f405d`. `files/` includes the eight wrappers and the pre-edit bytes of `docs/refactor_surface_report.md`, while `MANIFEST` records mode, SHA-256, line, and byte metadata. Archived wrapper copies pass `bash -n`, and static `rg` found no live nonarchive `.sh`/`.sbatch`/`.py` reference to the retired filenames.
