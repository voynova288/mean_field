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
  cli.py
  paths.py
  reference_oracles.py
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
        _hf_basis_overlap.py
        _hf_restricted.py
        _hf_full.py
        _hf_diagnostics.py
        hf_runners.py
        path.py
        artifacts.py
```

## Layering rules

- `core/` must not depend on a specific physical system.
- `core/hf/` owns reusable Hartree-Fock bookkeeping and SCF iteration logic that should survive a future move from TBG to multilayer or other graphene stackings. Its package root is an empty namespace: framework code imports explicit owner modules, while stable user-facing execution goes through `mean_field.api`.
- `core/hf/problem.py` owns the reusable HF problem-definition surface: state initialization, interaction builders, projected-density solvers, and run composition.
- `systems/tbg/` contains TBG-specific physics.
- `systems/tbg/zero_field/_hf_basis_overlap.py`, `_hf_restricted.py`, `_hf_full.py`, and `_hf_diagnostics.py` own TBG zero-field state, overlap/interaction construction, initialization policy, solvers, and diagnostics. Callers import the required explicit owner rather than a compatibility aggregate.
- `systems/tbg/zero_field/hf_runners.py` selects exact intersections with saved typed SCF grids; it must not reconstruct off-grid Hamiltonians, interpolate, or substitute nearest points.
- `systems/tbg/zero_field/artifacts.py` owns only the strict typed complete-state archive. Benchmark sidecars, paper plots, and suite orchestration are retired.
- `reference_oracles.py` is an internal loader for immutable external numerical fixtures; it is not a package-root API or workflow runner.
- CLI commands call maintained typed APIs; historical B0 benchmark orchestration is not a CLI surface.

## Retirement archive policy

Cleanup may retire system-specific implementation surfaces even when they could be useful as debugging references later. Before deleting or replacing substantial system-specific HF, topology, bands, band-plot, or Berry-curvature plotting code, copy the old tracked file or code slice into an ignored local archive such as `local_archive/retired_surface/<date-or-commit>/...`. The archive is intentionally not pushed to git and must not be imported by package code, tests, scripts, or docs examples. It is a recovery/reference stash only; the package surface and LOC metrics count only files tracked by git.

After archival, keep only thin tracked adapters that connect system-owned Hamiltonian/basis/gauge/window choices to generic APIs such as `mean_field.api`, explicit `mean_field.core.hf.<owner>` modules, `mean_field.core.bands`, `mean_field.core.plotting.bands`, and `analysis.topology`. Do not keep paper-panel plot writers, duplicated Berry/Chern loops, or system-local SCF/problem loops in tracked code merely as a backup; use the local archive or git history if a retired implementation must be consulted later.

## Unified topology / Berry-geometry layer

Berry connection, Berry curvature / plaquette flux, and Chern-number calculations are unified under:

```text
src/analysis/topology/
```

The architectural rule is:

```text
system eigenstate grid
  -> FHSState + band/flavor metadata + BlockSewingSpec
  -> common generic sewing and FHS/Wilson links
  -> Berry plaquette flux
  -> Chern number
```

System modules may construct the eigenstate mesh, select/map state labels, and
provide basis metadata needed to instantiate `BlockSewingSpec`. They must return
`FHSState` and must not expose topology-result calculators or private seam
transforms. Only `analysis.topology.compute_lattice_topology(FHSState)` builds
links, plaquette flux, and Chern numbers. Extend the common FHS core only when
the shared algorithm itself needs a system-independent capability. See
`docs/topology_framework.md` for conventions and examples.

## Common plotting surface

Shared band/path plotting helpers are exposed under:

```text
src/mean_field/core/plotting/bands.py
```

System plot modules should import generic helpers such as `load_plot_backend`, `format_kpath_axis`, `plot_band_columns`, `save_figure_pair`, and k-path TSV writers from this core namespace.  Systems still own paper-specific labels, panel layouts, colors, overlays, and default filenames.

## Gauge-safe response derivative layer

Gauge-safe Berry-connection generalized derivatives and shift-vector helpers are centralized in:

```text
src/analysis/response_derivative_gauge.py
```

This module mirrors the WannierBerri/Wannier90 covariant-derivative convention for Hamiltonian-gauge matrices and is reusable beyond the current shift-current workspaces.  It should be the common place for:

- Hamiltonian-gauge derivative ingredients;
- Berry-connection generalized derivatives;
- selected-pair and subspace-trace helpers;
- Wilson-link validation of shift vectors;
- random phase/block-unitary gauge-covariance tests.

Do not implement response derivatives by differentiating raw eigenvector phases or raw `np.angle(A_mn)` in a system module.  If a response calculation needs more common derivative capability, extend `analysis.response_derivative_gauge` first and then call it from the system or analysis adapter.  See `src/analysis/RESPONSE_DERIVATIVE_GAUGE.md` for the local contract and validation notes.

## Optical-response API status

Reusable optical-response math is now organized as four layers:

```text
src/analysis/response_derivative_gauge.py   # Hamiltonian-gauge derivatives, Berry connection, generalized derivative
src/analysis/shift_current/                # shift-current formula family
src/analysis/injection_current/            # injection-current / CPGE formula family
src/analysis/optical/                      # front-door dispatcher for future optical workflows
```

New system/workflow code should generally connect through `analysis.optical`. Transition-table workflows choose `kind="shift_current"` or `kind="injection_current"`/`"cpge"`; full k-point tensor workflows call `optical_response_from_kpoint_data` with `kind="linear_conductivity"`, `kind="kerr_faraday"`, `kind="shg"`, or `kind="thg"`. Kerr requires an explicit `si_prefactor` and typed normal-incidence geometry. Production SHG/THG uses `harmonic_response_from_kpoint_data` with a `PassosFiniteBandVelocityGauge(IndependentInputAdiabaticSwitching(...))` formulation, the complete finite-band covariant derivative tower through order `n+1`, and one exact full primitive-BZ grid; its SI conversion is derived from that typed payload, and selected-band windows are rejected. The Mikhailov one-gamma-per-cumulative-pole result used for Passos Figs. 1/2 remains a graphene analytical benchmark, not a generic scattering option. Never obtain it by changing only `r*gamma` in the velocity-gauge recursion; a generic scattering backend first needs a mesh-level non-Abelian length-gauge derivative/transport contract. Physical systems remain responsible for Hamiltonians, derivative provenance, basis/gauge choices, model multiplicity, and paper conventions. Plotting labels, colorbar normalization, and Slurm orchestration remain workflow-layer choices, not generic formula code.

## Shift-current workspace status

The old directories `src/analysis/shift_current_htg` and `src/analysis/shift_current_tbg` have been retired.  Reusable response mathematics lives in `src/analysis/response_derivative_gauge.py`, `src/analysis/shift_current/`, `src/analysis/injection_current/`, and the front-door `src/analysis/optical/`; reference/toy benchmarks live under the relevant `toy_models/` subpackages; physical-system Hamiltonians, derivatives, basis/gauge conventions, and paper compatibility adapters belong under `src/mean_field/systems/<system>/`.  Historical audits and reproduction notes should stay in ignored local reports/internal workspaces rather than the public docs surface.

When future systems need optical-response, shift-current, injection-current, or CPGE analysis, connect the system model through a thin adapter that supplies Hamiltonians, derivatives, energies/eigenvectors, occupation data, units, and conventions to `analysis.optical` or its underlying common analysis helpers.  Keep paper-specific scans, plotting, and unresolved reproduction diagnostics out of the common framework until the relevant formula and convention gates have passed.

## Generic TDHF/RPA layer

Reusable signed-momentum TDHF/RPA algebra lives in
`src/mean_field/core/hf/tdhf_signed.py` and is exposed through
`mean_field.api.run_tdhf` / `run_tdhf_typed`.  Physical-system modules must be
thin providers: they build the HF-basis pair inventories and independent A/B
blocks, preserve raw momentum/carry and gauge/sewing provenance, and return a
typed generic or self-conjugate sector.  They must not implement their own
Liouvillian eigensolver, Wang norm/sign assignment, static/dynamic classifier,
or Ward acceptance logic.

The framework intentionally distinguishes generic non-TRIM `{q,-q}` orbits
from q=0 and sewn self-conjugate/Nyquist sectors.  Generic sectors carry
independent `A(q)`, `A(-q)`, `B(q)`, and `B(-q)` plus an explicit anti-linear
Nambu sewing.  Self-conjugate sectors require a canonical sewn A/B payload;
raw boundary aliases are diagnostic inputs and are never averaged into one.
Static-Hessian authority is typed separately from projected signed-A/B
authority so a system adapter cannot silently promote a response regulator to
a global scalar curvature.

All new TDHF paper benchmarks must call the public API.  Benchmark-specific
Hamiltonians, form factors, screening, HF sources, symmetry generators, and
paper comparison remain under `systems/<system>/` or benchmark fixtures; the
common solver and validation logic remain in `core/hf`.

## Current reusable HF split

The zero-field TBG port now has three explicit layers instead of one large `hf.py` bucket:

- `core/hf/`: flavor-sector indexing, band labeling, occupation helpers, and generic convergence utilities.
- `core/magnetic_field.py`: system-agnostic finite-magnetic-field bookkeeping such as rational fluxes, magnetic mesh/orbit indexing, reciprocal-shell shifts, and Streda/Diophantine filling helpers. System layers should import these helpers directly rather than redefining or re-exporting them.
- `core/hf/engine.py`: generic SCF iteration, ODA mixing, convergence-rule handling, and density-update plumbing. This layer should be reusable across moire systems even when the Coulomb kernel or projected basis changes.
- `core/hf/problem.py`: generic HF problem definitions that let each physical system swap in its own non-interacting model, Coulomb kernel, projected basis, and initialization policy without rewriting the SCF loop.
- `systems/tbg/zero_field/_hf_basis_overlap.py`, `_hf_restricted.py`, `_hf_full.py`, and `_hf_diagnostics.py`: explicit owners for the TBG-specific state, interaction kernels, density builders, initialization semantics, solver variants, and diagnostics that still depend on BM overlaps and Coulomb conventions.
- `systems/tbg/finite_field/spectrum.py`: the finite-magnetic-field BM/LL spectrum adapter ported from the author `bmLL*.jl` modules for arXiv:2310.15982v3. It keeps author finite-B parameter conventions, LL translation matrix elements, magnetic-BZ Hamiltonian construction, central `2q` Hofstadter subbands, projected `PΣz`, and optional `Λ_(m,n)` overlaps in the TBG system layer.
- `core/hf/finite_field.py`: the reusable finite-magnetic-field HF framework. It owns finite-B HF state/input bundles, stored-projector initialization and density updates, screened Coulomb kernels, full magnetic-BZ and magnetic-translation-reduced interaction contractions, SCF problem/run helpers, and summaries. It is system-agnostic: systems provide projected Hofstadter spectra, overlap blocks, k-vectors, normalization counts, and physical parameters.
- `systems/tbg/finite_field/hf.py`: a thin TBG adapter. It computes/validates TBG K/K′ `MagneticSpectrumResult` objects, expands TBG valley overlaps into the generic spin/valley HF basis, supplies TBG magnetic k-vectors/normalization, and exposes paper/Fig.6 convenience APIs. It must not own the finite-B HF calculation itself; new finite-B HF capabilities should be added to `core/hf/finite_field.py` and then connected here.
- `systems/tbg/zero_field/hf_runners.py`: exact saved-SCF-grid band selection only; historical B0 benchmark orchestration and off-grid HF path reconstruction are retired.

The former TBG companion parity/Stage7A diagnostic exceptions were never public APIs and have been retired to `local_archive/retired_surface/non_api_abc_tbg_oracles_20260915/`. New work must use the common HF/TDHF APIs rather than restoring system-local duplicate solvers.

This is the intended direction for future systems. A new graphene stacking should first try to reuse `core/hf/`, then add its own `systems/<name>/...` physics layer, and only after that add benchmark or CLI workflows.

## Script and devtool surface

The command surface should stay small.  Use `scripts/mean_field_tools.py`, `scripts/submit_mean_field.sbatch`, and package CLI subcommands as the durable entrypoints.  `src/mean_field/devtools` should provide reusable implementation modules behind those entrypoints, not a growing collection of one-off runners. Historical B0 Julia exporters are archived provenance rather than maintained commands.

Before adding a new tracked script or devtool, try to extend an existing command with an option, subcommand, or config input.  Per-run `.sbatch` files, timestamped launchers, narrow plotting scripts, and temporary parameter sweeps should normally stay in ignored scratch space.  See `script_surface_policy.md` for the detailed policy and cleanup target.

## Full-projector scalar-functional qualification boundary

The reusable public ABI is `mean_field.core.hf.tdhf_scalar_functional`. It uses
conventional dense `complex128` matrices
`P_ij = <c_j^dagger c_i>` and raw, unweighted identities
`dE[P+tD]/dt = Tr(F[P]D)` and `dF[P+tD]/dt = dF[P,D]`. Every registered
Hermitian direction is normalized internally to unit Frobenius norm and must
clear the locked pre-normalization signal floor. A receipt distinguishes:

- `registered_probe_functional_consistency`: all preregistered E/F/dF probes
  passed;
- `full_projector_functional_consistency`: additionally, the registered
  inventory is exactly the internally generated normalized `N^2` Hermitian
  basis for a supported small dimension.

An incomplete production inventory must never report full-projector
consistency. A system adapter may make optional generic dF informativeness
mandatory for its own authority boundary. Exact-unitary support is not an
asserted Boolean: the plan contains explicit same-trace idempotent projector
values, and the receipt records actual finite, nonmutating E and F execution
for each value. These gates do not compare or promote TDHF A/B or H+ authority.

The former system-local Vituri storage/evidence adapter was not part of the
common API and is preserved only in
`local_archive/retired_surface/non_api_abc_tbg_oracles_20260915/`.

Callback, dependency, source-byte, input-manifest, and verifier snapshots are
a trusted-provider drift boundary only. Python tracing and hashes are not a
sandbox, hostile-code proof, or global completeness proof. No full-space
functional receipt by itself establishes A/B scalar-Hessian equality,
production readiness, or paper reproduction.

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
