# Architecture

## Why a rewrite instead of transpilation

The Julia codebase is optimized around script-driven workflows and shared mutable structs. That is effective for exploratory physics work, but it is a poor fit for a Python package meant to grow to more systems and more users. The Python version should preserve the physics and benchmarks while improving:

- package boundaries
- user-facing interfaces
- testability
- benchmark reproducibility
- performance isolation

## Proposed package structure

```text
src/mean_field/
  paths.py
  core/
    lattice.py
    hf/
      engine.py
      problem.py
      flavors.py
      occupations.py
  systems/
    tbg/
      params.py
      zero_field/
        model.py
        overlap.py
        hf.py
        hf_contracts.py
        path.py
```

## Layering rules

- `core/` must not depend on a specific physical system.
- `core/hf/` owns reusable Hartree-Fock bookkeeping and SCF iteration logic that should survive a future move from TBG to multilayer or other graphene stackings.
- `core/hf/problem.py` owns the reusable HF problem-definition surface: state initialization, interaction builders, projected-density solvers, and run composition.
- `systems/tbg/` contains TBG-specific physics.
- `systems/tbg/zero_field/hf.py` owns TBG zero-field state, initialization policy, and Hartree/Fock construction, but should consume reusable helpers from `core/hf/` instead of redefining them.
- TBG zero-field benchmark orchestration, artifact export, plotting, and runner helpers are archived out of the tracked public surface. Tracked zero-field TBG keeps the core BM model, overlaps, HF kernels, path helpers, and HF contract adapters.
- Benchmark metadata loaders and package CLI glue are archived out of the tracked public surface for now.

## Retirement archive policy

Cleanup may retire system-specific implementation surfaces even when they could be useful as debugging references later. Before deleting or replacing substantial system-specific HF, topology, bands, band-plot, or Berry-curvature plotting code, copy the old tracked file or code slice into an ignored local archive such as `local_archive/retired_surface/<date-or-commit>/...`. The archive is intentionally not pushed to git and must not be imported by package code, tests, scripts, or docs examples. It is a recovery/reference stash only; the package surface and LOC metrics count only files tracked by git.

After archival, keep only thin tracked adapters that connect system-owned Hamiltonian/basis/gauge/window choices to generic APIs such as `mean_field.api`, `mean_field.core.hf`, and `mean_field.core.bands`. Do not keep paper-panel plot writers, duplicated Berry/Chern loops, or system-local SCF/problem loops in tracked code merely as a backup; use the local archive or git history if a retired implementation must be consulted later.

## Topology / Berry-geometry status

The previous unified topology/Berry-geometry helper package and system wrappers are archived out of the tracked public surface for now:

```text
local_archive/retired_surface/topology_untracked_20260622/
```

If topology becomes a near-term target again, reintroduce a small reviewed common API rather than restoring all historical wrappers. Do not duplicate `_unit_link`, `_subspace_link`, determinant-link, or plaquette loops in system modules without first deciding the maintained public topology boundary.

## Common plotting surface

Shared band/path plotting helpers are exposed under:

```text
src/mean_field/core/plotting/bands.py
```

System plot modules should import generic helpers such as `load_plot_backend`, `format_kpath_axis`, `plot_band_columns`, `save_figure_pair`, and k-path TSV writers from this core namespace.  Systems still own paper-specific labels, panel layouts, colors, overlays, and default filenames.

## Gauge-safe response derivative layer

Gauge-safe Berry-connection generalized derivatives and shift-vector helpers are centralized in:

```text
src/analysis/optical_response/gauge.py
src/analysis/optical_response/gauge_*.py
```

The historical `src/analysis/response_derivative_gauge.py` path is a compatibility shim.  The optical-response gauge modules mirror the WannierBerri/Wannier90 covariant-derivative convention for Hamiltonian-gauge matrices and are reusable beyond the current shift-current workspaces.  They should be the common place for:

- Hamiltonian-gauge derivative ingredients;
- Berry-connection generalized derivatives;
- selected-pair and subspace-trace helpers;
- Wilson-link validation of shift vectors;
- random phase/block-unitary gauge-covariance tests.

Do not implement response derivatives by differentiating raw eigenvector phases or raw `np.angle(A_mn)` in a system module.  If a response calculation needs more common derivative capability, extend `analysis.optical_response` first and then call it from the system or analysis adapter.  See `src/analysis/RESPONSE_DERIVATIVE_GAUGE.md` for the local contract and validation notes.

## Shift-current workspace status

The old directories `src/analysis/shift_current_htg` and `src/analysis/shift_current_tbg` have been retired.  Reusable response mathematics lives in `src/analysis/optical_response/`; compatibility import paths remain under `src/analysis/response_derivative_gauge.py` and `src/analysis/shift_current/`; physical-system Hamiltonians, derivatives, basis/gauge conventions, and paper compatibility adapters belong under `src/mean_field/systems/<system>/`.  Historical audits and reproduction notes should stay in ignored local reports/internal workspaces rather than the public docs surface.

When future systems need optical-response or shift-current analysis, connect the system model through a thin adapter that supplies Hamiltonians, derivatives, energies/eigenvectors, occupation data, units, and conventions to the common analysis helpers.  Keep paper-specific scans, plotting, and unresolved reproduction diagnostics out of the common framework until the relevant formula and convention gates have passed.

## Current reusable HF split

The zero-field TBG port now has three explicit layers instead of one large `hf.py` bucket:

- `core/hf/`: flavor-sector indexing, band labeling, occupation helpers, and generic convergence utilities.
- `core/magnetic_field.py`: system-agnostic finite-magnetic-field bookkeeping such as rational fluxes, magnetic mesh/orbit indexing, reciprocal-shell shifts, and Streda/Diophantine filling helpers. System layers should import/re-export these helpers rather than redefining them.
- `core/hf/engine.py`: generic SCF iteration, ODA mixing, convergence-rule handling, and density-update plumbing. This layer should be reusable across moire systems even when the Coulomb kernel or projected basis changes.
- `core/hf/problem.py`: generic HF problem definitions that let each physical system swap in its own non-interacting model, Coulomb kernel, projected basis, and initialization policy without rewriting the SCF loop.
- `systems/tbg/zero_field/hf.py`: the TBG-specific interaction kernels, density builders, and initialization semantics that still depend on BM overlaps, Coulomb conventions, and Julia B0 benchmark rules.
- `systems/tbg/finite_field/spectrum.py`: the finite-magnetic-field BM/LL spectrum adapter ported from the author `bmLL*.jl` modules for arXiv:2310.15982v3. It keeps author finite-B parameter conventions, LL translation matrix elements, magnetic-BZ Hamiltonian construction, central `2q` Hofstadter subbands, projected `PΣz`, and optional `Λ_(m,n)` overlaps in the TBG system layer.
- `core/hf/finite_field.py`: the reusable finite-magnetic-field HF framework. It owns finite-B HF state/input bundles, stored-projector initialization and density updates, screened Coulomb kernels, full magnetic-BZ and magnetic-translation-reduced interaction contractions, SCF problem/run helpers, and summaries. It is system-agnostic: systems provide projected Hofstadter spectra, overlap blocks, k-vectors, normalization counts, and physical parameters.
- `systems/tbg/finite_field/hf.py`: a thin TBG adapter. It computes/validates TBG K/K′ `MagneticSpectrumResult` objects, expands TBG valley overlaps into the generic spin/valley HF basis, supplies TBG magnetic k-vectors/normalization, and exposes paper/Fig.6 convenience APIs. It must not own the finite-B HF calculation itself; new finite-B HF capabilities should be added to `core/hf/finite_field.py` and then connected here.
- Archived local reference: `local_archive/retired_surface/benchmark_workflow_untracked_20260622/` contains the former benchmark-facing orchestration, path-band diagnostics, plotting, CLI, and workflow helpers.

This is the intended direction for future systems. A new graphene stacking should first try to reuse `core/hf/`, then add its own `systems/<name>/...` physics layer. Benchmark or CLI workflows should stay local/ignored unless they become reviewed durable public commands.

## Script and devtool surface

The command surface should stay small. Use `scripts/mean_field_tools.py`, `scripts/mean_field_tools.jl`, and `scripts/submit_mean_field.sbatch` as the durable entrypoint placeholders. Package CLI/devtools are archived out of the minimal public surface for now.

Before adding a new tracked script or reintroducing devtools, check whether the workflow can remain in ignored local scratch/archive space. Per-run `.sbatch` files, timestamped launchers, narrow plotting scripts, and temporary parameter sweeps should normally stay ignored. See `script_surface_policy.md` for the detailed policy and cleanup target.

## Performance strategy

The right performance target is not "Python everywhere"; it is "Python orchestration with optimized kernels where needed".

Recommended sequence:

1. Establish a numerically correct NumPy/SciPy baseline.
2. Profile full benchmark runs and identify dominant kernels.
3. Move stable hot loops to Numba or another compiled backend.
4. Add cluster-aware parallelism that uses the full allocated CPU budget by default without oversubscription.
5. Preserve API compatibility so kernels can be swapped without changing workflow code.

Expected hot spots for the zero-field port:

- Hamiltonian assembly on the k-grid
- overlap and Coulomb-form-factor construction
- repeated eigendecompositions inside SCF
- Hartree/Fock tensor contractions
- benchmark-level orchestration that can be parallelized across cases or twist angles

Candidate acceleration backends:

- `Numba` for Hamiltonian assembly and overlap loops
- `SciPy` LAPACK drivers for eigensolvers
- process-level parallel runners for independent benchmark cases, plus explicit BLAS/thread control inside a case
- `JAX` or `CuPy` later if GPU or accelerator support becomes useful
- compiled extensions only if profiling shows Python-side orchestration is no longer the bottleneck

Cluster execution assumptions:

- Heavy BM/HF benchmarks and any other nontrivial compute-side validation should run on CPU compute nodes through Slurm, not on the login nodes.
- `login001` and `login002` are submission / inspection entry points only. They must not be used to run numerical tests, benchmark commands, SCF solves, eigensolvers, BLAS-heavy scripts, or other compute work.
- The short-task development route is `login002 -> test001`; login nodes are reserved for editing, file inspection, parameter checks, queue checks, and non-compute script validation.
- `long` is also a valid CPU partition for these development benchmarks when the short-task route is not appropriate.
- Formal runs should be designed as Slurm jobs first, not as interactive-node work.
- Jobs expected to exceed the `test001` envelope should be escalated deliberately into an appropriate Slurm CPU workflow rather than silently moved to a login-node run.
- Slurm usage should stay conservative on the shared cluster: unless the user says otherwise, use at most 5 nodes and prefer serial task lists within a node when several cases can be packed together.
- When a job is assigned a single CPU node, its per-node resources should be used as fully as practical. CPU nodes are heterogeneous; many expose `56` cores while some expose `64`, so `cpus-per-task` should be chosen intelligently to saturate the target node type instead of being hard-coded to `28`.
- Memory requests for single-node CPU jobs should likewise be sized to use most of the node when appropriate, while still leaving a defensible safety margin rather than under-requesting by habit.
- Runtime benchmark records should include allocated CPU count, BLAS thread count, process count, and whether JIT warm-up time is included.
- Default runtime settings should honor environment variables such as `SLURM_CPUS_PER_TASK` before falling back to `os.cpu_count()`.

## User-facing design goals

- stable Python package API
- reproducible configuration objects
- benchmark-aware runners
- command-line entry points for common tasks
- output formats that are easy to inspect and diff

## Initial non-goals

- immediate port of every magnetic-field production workflow/script beyond the reusable B-SCHF module
- exact reproduction of every script in the Julia repo
- premature backend complexity before benchmark parity
