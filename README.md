# Mean Field

Python code for continuum-model and projected Hartree-Fock calculations in graphene moire systems.

The package started as a benchmark-driven rewrite of a Julia `TBG_HartreeFock` workflow and now contains reusable numerical infrastructure plus system-specific implementations for TBG zero-field HF, tMBG, TDBG, alternating-twist multilayer graphene, and helical trilayer graphene.

## Scope

- `mean_field.core.hf`: reusable projected Hartree-Fock machinery, including occupations, ODA iteration, Coulomb kernels, overlap contractions, flavor-sector helpers, and the generic TDHF/RPA core.
- `mean_field.systems.tbg`: zero-field TBG/BM core model and HF adapters.
- `mean_field.systems.tmbg`: twisted monolayer-bilayer graphene continuum model and validation checks.
- `mean_field.systems.tdbg`: twisted double bilayer graphene continuum model and projected-HF helpers.
- `mean_field.systems.atmg`: alternating-twist multilayer graphene continuum-model utilities.
- `mean_field.systems.htg`: helical trilayer graphene continuum model and projected-HF adapter for Kwan et al. style calculations.
- `analysis.topology`: minimal system-independent FHS link/plaquette/Chern helpers, wavefunction-grid canonicalization helpers, a small system-facing adapter for already-built eigenvector grids, and projector QGT/quantum-metric helpers. Thin `mean_field.systems.{tmbg,tdbg,atmg,RnG_hBN}.topology` wrappers and matching model `topology_on_grid(...)` convenience methods delegate to this common API; HTG topology wrapper, projected-HF reconstruction helpers, and paper workflows remain archived under `local_archive/retired_surface/topology_untracked_20260622/` until reviewed separately.
- `analysis.optical_response`: reusable WannierBerri-style, gauge-safe derivative and shift-current helpers for Berry-connection generalized derivatives, shift vectors, response components, named conventions, occupations, Lorentzian/heatmap accumulation, and one-k-point tensor helpers.
- `analysis.response_derivative_gauge` and `analysis.shift_current`: historical compatibility import paths that re-export the common optical-response API. System-specific Hamiltonians/derivatives live under `mean_field.systems.*`; historical shift-current audit notes stay in ignored local reports/internal workspaces.

Large generated outputs, local benchmark bundles, PDFs, Slurm logs, local tests, historical reports, planning notes, and code-agent work documents are intentionally not versioned. They should be regenerated, kept in an internal workspace, or stored separately.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional Numba acceleration:

```bash
python -m pip install -e ".[dev,perf]"
```

## Validation

This public snapshot keeps the core package, stable dispatchers, durable docs,
and a small set of public contract tests for the API/density/artifact schema.
Broad local regression tests and benchmark data are not included in the
published repository.

A lightweight syntax check can be run from the repository root:

```bash
python -m compileall -q src scripts
```

Core conventions are documented in `docs/conventions.md`. Public API contracts live under `docs/api/`. The generic TDHF/RPA core contract is documented in `docs/tdhf_core_contract.md`.

Heavy self-consistent HF calculations, topology-grid eigensolver recomputations, and broad numerical pytest runs should be submitted to a compute node through Slurm rather than run on a login node.

## Command Surface

The desired public script surface is intentionally small.  Prefer existing dispatchers over adding new standalone scripts; see `docs/script_surface_policy.md`.

- `scripts/mean_field_tools.py`: minimal Python dispatcher placeholder; durable commands should be reintroduced only after review.
- `scripts/mean_field_tools.jl`: Julia helper dispatcher for benchmark-reference exports.
- `scripts/submit_mean_field.sbatch`: generic Slurm wrapper for numerical jobs.

Examples:

```bash
python scripts/mean_field_tools.py help
sbatch scripts/submit_mean_field.sbatch python -m compileall -q src scripts
```

## Repository Layout

- `src/mean_field/`: installable Python package.
- `scripts/`: stable/generic entrypoints only; one-off wrappers should stay in ignored scratch paths.
- `docs/`: stable architecture, framework/API contracts, script policy, and durable migration maps only.

The following local directories are ignored by design:

- `benchmarks/`
- `results/`
- `logs/`
- `reference/`
- `reports/`
- `plan/`
- `tests/local/`, `tests/internal/`, `tests/slow/`, generated test data, generated arrays, and broad local regression tests; only a small smoke/contract subset remains tracked under bare `tests/`
- Python/Jupyter/cache artifacts
- task-specific work documents and temporary handoff notes
