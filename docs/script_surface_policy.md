# Script surface policy

This repository should not grow by adding a new standalone script for every diagnostic, Slurm run, paper panel, or parameter sweep.  The current public command surface is intentionally minimal.

## Current audit

The cleanup removed the historical tracked pile of one-off launchers, paper-panel scripts, package CLI glue, and devtool implementation modules.  The remaining tracked surface is deliberately small:

- `../scripts/`: `AGENTS.md`, `mean_field_tools.py`, `mean_field_tools.jl`, `submit_mean_field.sbatch`, and the Julia reference helper implementations used by `mean_field_tools.jl`.
- No tracked `../src/mean_field/devtools/` package in the minimal public surface.
- No tracked `../src/mean_field/cli.py` package CLI in the minimal public surface.

Archived local references:

```text
local_archive/retired_surface/devtools_untracked_20260622/
local_archive/retired_surface/benchmark_workflow_untracked_20260622/
```

Treat deleted one-off wrappers as historical debt, not as templates to restore.  If a retired workflow is needed again, recover only the reusable logic from the local archive/git history into a reviewed small public command.

## Preferred command surfaces

Keep durable entrypoints concentrated in:

- `../scripts/mean_field_tools.py` as a placeholder for reviewed Python commands;
- `../scripts/mean_field_tools.jl` for Julia reference-export helpers;
- `../scripts/submit_mean_field.sbatch` for generic Slurm execution.

Current lightweight examples:

```bash
python ../scripts/mean_field_tools.py help
sbatch ../scripts/submit_mean_field.sbatch python -m compileall -q src scripts
```

## Rules for new scripts/devtools

Before adding any file under `scripts/` or reintroducing `src/mean_field/devtools/`, check whether the workflow can stay in ignored local scratch/archive space.

Add a new tracked script/devtool only if all of the following are true:

1. The workflow is expected to be reused beyond the current conversation or one paper-panel attempt.
2. It cannot stay as a local ignored command/config in `tmp/`, `local/`, or `local_archive/`.
3. It has a clear owner surface: a reviewed `mean_field_tools.py` command, `mean_field_tools.jl`, or the generic Slurm wrapper.
4. It has a small validation path such as `--help`, dry-run, syntax check, saved-result check, or a tiny smoke test.
5. It does not bypass existing login-node guards for HF, topology, eigensolver, or response-function compute.

Do not commit per-run `.sbatch` files, timestamped launchers, ad hoc plotting scripts, or narrow parameter sweeps unless the user explicitly asks to preserve them as durable project assets.  Put scratch launchers in ignored locations such as `tmp/`, `scripts/local/`, or `local/`.

## Consolidation target

Long term, `../scripts/` should contain only a few generic entrypoints plus possibly a small number of stable language/reference bridges.  `../src/mean_field/devtools/` should remain absent unless a reviewed durable command needs a small implementation package.

When cleaning up existing files:

- migrate reusable logic into system modules or common analysis modules; keep command glue local unless it becomes durable API;
- preserve durable public design details in `docs/`, and keep run-specific diagnostics in ignored local `reports/` directories or result metadata rather than in many near-duplicate launch scripts;
- when retiring substantial system-specific HF/topology/bands/plotting code, copy it first into ignored `local_archive/retired_surface/...` if it may be useful for future repair; archived code must not be imported by tracked package code and does not count as maintained surface;
- delete or untrack obsolete wrappers after confirming no current documentation or tests depend on them;
- keep heavy validation on Slurm; keep hard-coded saved-result artifact audits in ignored reports/internal workspaces rather than public package modules.
