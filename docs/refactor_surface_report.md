# Refactor surface report

This Phase 2 report measures legacy surface area and tracks cleanup slices that delete or thin old paths.

## Summary

- `src` Python files: 200
- `src` Python lines: 68892
- Files over 1000 lines: 15
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 10

## Completed cleanup slices

### remove_atmg_fig3_devtool

- Moved reusable build_khalaf_fig3_path helper from one-off devtool to systems/atmg/bands.py, removed run_atmg_fig3_band_plot dispatcher command, and deleted tracked paper-panel devtool.
- Deleted files: `src/mean_field/devtools/run_atmg_fig3_band_plot.py`.
- Gross legacy LOC removed: 665.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 20 -> 18.

### remove_rlg_hbn_band_plot_devtool

- Moved reusable RnG/hBN Fig.6 HF path and band-plot manifest helpers into systems/RnG_hBN/bands.py, removed plot_rlg_hbn_paper_hf_bands dispatcher command, and deleted tracked paper-panel plotting devtool.
- Deleted files: `src/mean_field/devtools/plot_rlg_hbn_paper_hf_bands.py`.
- Gross legacy LOC removed: 739.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 18 -> 17.

### remove_htqg_fig1_bands_devtool

- Removed one-off HTQG Fig.1 first-pass band plotting devtool from tracked command surface. The reusable HTQG band/path/domain APIs remain in systems/htqg.
- Deleted files: `src/mean_field/devtools/run_htqg_fig1_bands.py`.
- Gross legacy LOC removed: 213.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 17 -> 13.

### remove_htqg_projected_hf_devtool

- Removed HTQG projected-HF paper-scan CLI glue from tracked devtools. The core HTQG projected-HF solver remains in systems/htqg/hf.py; future durable reproduction should be reintroduced under workflows with explicit validation gates.
- Deleted files: `src/mean_field/devtools/run_htqg_projected_hf.py`.
- Gross legacy LOC removed: 400.
- Direct `mean_field.systems.*` imports in devtools/scripts/workflows: 13 -> 10.

## Top 30 Python files under `src`

| Lines | Path |
|---:|---|
| 2361 | `src/mean_field/systems/RnG_hBN/hf.py` |
| 2305 | `src/mean_field/systems/htg/mean_field_adapter.py` |
| 2151 | `src/mean_field/systems/tmbg/polshyn_supercell.py` |
| 1640 | `src/mean_field/systems/RnG_hBN/tdhf.py` |
| 1567 | `src/mean_field/devtools/backfill_canonical_hf_sidecars.py` |
| 1563 | `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` |
| 1348 | `src/mean_field/systems/tbg/zero_field/runners.py` |
| 1347 | `src/mean_field/systems/htg/supercell.py` |
| 1306 | `src/mean_field/api/hf.py` |
| 1267 | `src/mean_field/crpa/hf_interface.py` |
| 1258 | `src/mean_field/systems/tbg/zero_field/hf.py` |
| 1202 | `src/mean_field/cli.py` |
| 1149 | `src/mean_field/core/hf/finite_field.py` |
| 1104 | `src/mean_field/systems/tmbg/validation.py` |
| 1012 | `src/mean_field/systems/tbg/finite_field/spectrum.py` |
| 936 | `src/mean_field/systems/tbg/zero_field/supercell.py` |
| 925 | `src/mean_field/systems/htqg/hf.py` |
| 873 | `src/analysis/topology/quantum_geometry.py` |
| 805 | `src/analysis/response_derivative_gauge.py` |
| 786 | `src/mean_field/systems/RnG_hBN/cache.py` |
| 778 | `src/analysis/shift_current/toy_models/hipolito2016.py` |
| 772 | `src/analysis/shift_current/core.py` |
| 735 | `src/mean_field/systems/tbg/zero_field/hf_contracts.py` |
| 733 | `src/mean_field/systems/tbg/chaudhary2021_hartree.py` |
| 710 | `src/mean_field/core/hf/tdhf.py` |
| 687 | `src/mean_field/systems/tmbg/full_flavor_ivc.py` |
| 686 | `src/mean_field/systems/RnG_hBN/hf_contracts.py` |
| 683 | `src/mean_field/systems/tbg/chaudhary2021.py` |
| 643 | `src/mean_field/systems/htg/supercell_contracts.py` |
| 627 | `src/mean_field/benchmarks.py` |

## Direct private-system imports in workflow surfaces

| Path | Line | Import |
|---|---:|---|
| `src/mean_field/devtools/backfill_canonical_hf_sidecars.py` | 1141 | `from mean_field.systems.RnG_hBN.tdhf import load_rlg_hbn_tdhf_run_from_archive` |
| `src/mean_field/devtools/backfill_canonical_hf_sidecars.py` | 1146 | `from mean_field.systems.RnG_hBN.hf_contracts import rlg_hbn_hf_run_to_hf_run_result` |
| `src/mean_field/devtools/prepare_tbg_crpa_bm.py` | 9 | `from mean_field.systems.tbg.params import TBGParameters` |
| `src/mean_field/devtools/run_htg_hf.py` | 21 | `from mean_field.systems.htg import (` |
| `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` | 35 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/run_rlg_hbn_paper_hf.py` | 50 | `from mean_field.systems.RnG_hBN.hf import RLG_HBN_FORM_FACTOR_CONVENTION_VERSION` |
| `src/mean_field/devtools/run_rlg_hbn_tdhf_finite_q.py` | 16 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/run_rlg_hbn_tdhf_q0.py` | 21 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/validate_rlg_hbn_fig6_prereqs.py` | 10 | `from mean_field.systems.RnG_hBN import (` |
| `src/mean_field/devtools/validate_tbg_crpa_artifact.py` | 22 | `from mean_field.systems.tbg import TBGParameters` |

## Repeated module-family line counts

### `bands.py`

Total lines: 820

| Lines | Path |
|---:|---|
| 190 | `src/mean_field/systems/atmg/bands.py` |
| 176 | `src/mean_field/systems/htqg/bands.py` |
| 173 | `src/mean_field/systems/RnG_hBN/bands.py` |
| 150 | `src/mean_field/systems/htg/bands.py` |
| 74 | `src/mean_field/systems/tmbg/bands.py` |
| 57 | `src/mean_field/systems/tdbg/bands.py` |

### `validation.py`

Total lines: 1932

| Lines | Path |
|---:|---|
| 1104 | `src/mean_field/systems/tmbg/validation.py` |
| 248 | `src/mean_field/systems/htqg/validation.py` |
| 164 | `src/mean_field/systems/atmg/validation.py` |
| 163 | `src/mean_field/systems/RnG_hBN/validation.py` |
| 129 | `src/mean_field/systems/tdbg/validation.py` |
| 124 | `src/mean_field/systems/htg/validation.py` |

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
| `_resolve_band_indices` | 2 | `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py` |
| `compute_bands_along_path` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `compute_bands_on_grid` | 6 | `src/mean_field/systems/RnG_hBN/bands.py`, `src/mean_field/systems/atmg/bands.py`, `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py`, `src/mean_field/systems/tdbg/bands.py`, `src/mean_field/systems/tmbg/bands.py` |
| `estimate_central_band_metrics` | 2 | `src/mean_field/systems/htg/bands.py`, `src/mean_field/systems/htqg/bands.py` |

### `validation.py`

| Symbol | Count | Paths |
|---|---:|---|
| `_check` | 2 | `src/mean_field/systems/htg/validation.py`, `src/mean_field/systems/htqg/validation.py` |
| `_status` | 2 | `src/mean_field/systems/atmg/validation.py`, `src/mean_field/systems/htg/validation.py` |
| `_status_from_bool` | 3 | `src/mean_field/systems/RnG_hBN/validation.py`, `src/mean_field/systems/tdbg/validation.py`, `src/mean_field/systems/tmbg/validation.py` |
| `reproduce_paper_checkpoints` | 2 | `src/mean_field/systems/RnG_hBN/validation.py`, `src/mean_field/systems/tmbg/validation.py` |
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
