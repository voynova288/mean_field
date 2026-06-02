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
  benchmarks.py
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
        hf_runners.py
        path.py
        plotting.py
        runners.py
```

## Layering rules

- `core/` must not depend on a specific physical system.
- `core/hf/` owns reusable Hartree-Fock bookkeeping and SCF iteration logic that should survive a future move from TBG to multilayer or other graphene stackings.
- `core/hf/problem.py` owns the reusable HF problem-definition surface: state initialization, interaction builders, projected-density solvers, and run composition.
- `systems/tbg/` contains TBG-specific physics.
- `systems/tbg/zero_field/hf.py` owns TBG zero-field state, initialization policy, and Hartree/Fock construction, but should consume reusable helpers from `core/hf/` instead of redefining them.
- `systems/tbg/zero_field/{hf_runners,path,plotting,runners}.py` is the B0 workflow layer: path reconstruction, artifact export, plotting, and benchmark orchestration.
- `benchmarks.py` knows how to load benchmark metadata, not how to solve physics.
- CLI commands call high-level runners, not low-level kernels directly.

## Unified topology / Berry-geometry layer

Berry connection, Berry curvature / plaquette flux, and Chern-number calculations are unified under:

```text
src/analysis/topology/
```

The architectural rule is that topology is system-independent after wavefunctions have been generated and selected.  System modules should provide:

- a wavefunction mesh with shape `(mesh_1, mesh_2, basis_dim, n_states)`;
- selected state/subspace indices;
- `WavefunctionIndex` metadata that labels band, Chern-basis, flavor, valley, and system meaning;
- optional boundary sewing transforms for non-periodic plane-wave gauges.

The common framework then builds FHS link variables, Berry-connection phases, plaquette flux, and Chern numbers.  Do not duplicate `_unit_link`, `_subspace_link`, determinant-link, or plaquette loops in future system modules; extend `analysis.topology` instead.  See `docs/topology_framework.md` for conventions, validation status, and examples.

## Current reusable HF split

The zero-field TBG port now has three explicit layers instead of one large `hf.py` bucket:

- `core/hf/`: flavor-sector indexing, band labeling, occupation helpers, and generic convergence utilities.
- `core/hf/engine.py`: generic SCF iteration, ODA mixing, convergence-rule handling, and density-update plumbing. This layer should be reusable across moire systems even when the Coulomb kernel or projected basis changes.
- `core/hf/problem.py`: generic HF problem definitions that let each physical system swap in its own non-interacting model, Coulomb kernel, projected basis, and initialization policy without rewriting the SCF loop.
- `systems/tbg/zero_field/hf.py`: the TBG-specific interaction kernels, density builders, and initialization semantics that still depend on BM overlaps, Coulomb conventions, and Julia B0 benchmark rules.
- `systems/tbg/zero_field/runners.py` and `hf_runners.py`: benchmark-facing orchestration and path-band diagnostics.

This is the intended direction for future systems. A new graphene stacking should first try to reuse `core/hf/`, then add its own `systems/<name>/...` physics layer, and only after that add benchmark or CLI workflows.

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

- immediate port of all magnetic-field workflows
- exact reproduction of every script in the Julia repo
- premature backend complexity before benchmark parity
