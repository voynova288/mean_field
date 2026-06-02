# Mean Field

Python code for continuum-model and projected Hartree-Fock calculations in graphene moire systems.

The package started as a benchmark-driven rewrite of a Julia `TBG_HartreeFock` workflow and now contains reusable numerical infrastructure plus system-specific implementations for TBG zero-field HF, tMBG, TDBG, alternating-twist multilayer graphene, and helical trilayer graphene.

## Scope

- `mean_field.core.hf`: reusable projected Hartree-Fock machinery, including occupations, ODA iteration, Coulomb kernels, overlap contractions, and flavor-sector helpers.
- `mean_field.systems.tbg`: zero-field TBG/BM benchmark and HF adapters.
- `mean_field.systems.tmbg`: twisted monolayer-bilayer graphene continuum model and paper-checkpoint helpers.
- `mean_field.systems.tdbg`: twisted double bilayer graphene continuum model and band/topology tools.
- `mean_field.systems.atmg`: alternating-twist multilayer graphene continuum-model utilities.
- `mean_field.systems.htg`: helical trilayer graphene continuum model and projected-HF adapter for Kwan et al. style calculations.
- `analysis.topology`: local unified Berry-geometry framework used by system topology adapters for Berry connection, plaquette flux, Chern numbers, wavefunction-index metadata, and boundary sewing.
- `analysis.response_derivative_gauge`: reusable WannierBerri-style, gauge-safe derivative helpers for Berry-connection generalized derivatives, shift vectors, and subspace/gauge validation.
- `analysis.shift_current_htg` and `analysis.shift_current_tbg`: active shift-current reproduction and diagnostic workspaces. The shift-current framework is not yet considered stable; reusable derivative logic should live in `analysis.response_derivative_gauge` rather than in these workspaces.

Large generated outputs, local benchmark bundles, PDFs, Slurm logs, and code-agent work documents are intentionally not versioned. They should be regenerated or stored separately.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional Numba acceleration:

```bash
python -m pip install -e ".[dev,perf]"
```

## Validation

This public snapshot keeps the core package, stable dispatchers, and a small
set of durable docs. Local regression tests and benchmark data are not included
in the published repository.

A lightweight syntax check can be run from the repository root:

```bash
python -m compileall -q src scripts
```

The unified topology framework has its durable design note in `docs/topology_framework.md`.  Existing saved topology artifacts can be checked without rerunning Hamiltonian solves via:

```bash
python -m analysis.topology.validate_existing_results --root /path/to/Mean_Field
```

Heavy self-consistent HF calculations, topology-grid eigensolver recomputations, and broad numerical pytest runs should be submitted to a compute node through Slurm rather than run on a login node.

## Command Surface

The public script surface is intentionally small:

- `scripts/mean_field_tools.py`: Python command dispatcher for stable benchmark and reproduction tools.
- `scripts/mean_field_tools.jl`: Julia helper dispatcher for benchmark-reference exports.
- `scripts/submit_mean_field.sbatch`: generic Slurm wrapper for numerical jobs.

Examples:

```bash
python scripts/mean_field_tools.py help
python scripts/mean_field_tools.py run_htg_hf --help
sbatch scripts/submit_mean_field.sbatch python scripts/mean_field_tools.py run_htg_hf --help
```

## Repository Layout

- `src/mean_field/`: installable Python package.
- `scripts/`: stable entrypoints only.
- `docs/`: stable architecture and migration notes.

The following local directories are ignored by design:

- `benchmarks/`
- `results/`
- `logs/`
- `reference/`
- `tests/`
- Python/Jupyter/cache artifacts
- task-specific work documents and temporary handoff notes
