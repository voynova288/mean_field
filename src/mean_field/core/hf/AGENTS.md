# AGENTS.md

## Scope

Applies to the reusable Hartree-Fock framework in this directory.

## Source of Truth

- Architecture and layering: `../../../../docs/architecture.md`.
- Public export surface: `__init__.py`.
- Generic SCF/problem contracts: `engine.py`, `problem.py`, `interaction.py`.

## Local Guidance

- Keep this directory system-agnostic. Do not import from `mean_field.systems.*` or encode TBG/RnG/HTG-specific basis conventions here.
- Generic SCF iteration, ODA, occupation bookkeeping, projected-overlap contraction, Coulomb helpers, and projected-HF kernel assembly belong here when they are reusable across systems.
- Physical-system choices belong in `../../systems/<system>/`: Hamiltonian construction, basis labels, valley/flavor conventions, projected window selection, screening model, sewing/gauge conventions, and paper-specific runners.
- If a needed hook is missing, add a protocol/callback/adapter surface here instead of copying the SCF loop into a system module.

## Workflow

Lightweight syntax validation from the repository root:

```bash
python -m compileall -q src/mean_field/core/hf
```

Do not run heavy HF self-consistency or BLAS/eigensolver benchmarks on login nodes; submit those through Slurm.
