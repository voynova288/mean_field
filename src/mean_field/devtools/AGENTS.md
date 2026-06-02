# AGENTS.md

## Scope

Applies to developer-facing modules under `src/mean_field/devtools`.

## Source of Truth

- Script policy: `../../../docs/script_surface_policy.md`.
- Public dispatcher: `../../../scripts/mean_field_tools.py`.
- Runtime safeguards/helpers: `_runtime.py`.

## Local Guidance

- This package is not a dumping ground for one-off paper-panel runners. Prefer a small set of reusable devtools behind `scripts/mean_field_tools.py` or the package CLI.
- Before adding a new module, first try to extend an existing devtool with an option/subcommand or move reusable logic into `mean_field.systems.*`, `mean_field.core.*`, or `analysis.*`.
- If a devtool is kept, it should expose `main()`, support `--help`, reuse `_runtime.py` for JSON writing and login-node guards, and be callable through the dispatcher when it is a durable command.
- Paper-specific plotting, parameter sweeps, and timestamped diagnostics should usually live in ignored scratch space or result metadata, not as tracked devtools.

## Safety

Any devtool that can diagonalize grids, run HF, compute topology, or integrate response functions must call `ensure_not_running_compute_on_login_node` or be Slurm-only by construction.
