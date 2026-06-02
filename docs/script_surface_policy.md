# Script surface policy

This repository should not grow by adding a new standalone script for every diagnostic, Slurm run, paper panel, or parameter sweep.  The target command surface is intentionally small and generic.

## Current audit

The 2026-06-02 cleanup removed the historical tracked pile of one-off launchers and paper-panel scripts.  The remaining tracked surface is deliberately small:

- `../scripts/`: `AGENTS.md`, `mean_field_tools.py`, `mean_field_tools.jl`, `submit_mean_field.sbatch`, and the Julia reference helper implementations used by `mean_field_tools.jl`.
- `../src/mean_field/devtools/`: `AGENTS.md` plus a curated set of reusable implementation modules behind `mean_field_tools.py` or lightweight tests.

Treat deleted one-off wrappers as historical debt, not as templates to restore.  If a retired workflow is needed again, recover the reusable logic from git history into an existing dispatcher rather than restoring the old per-run wrapper as-is.

## Preferred command surfaces

Keep durable entrypoints concentrated in:

- `../scripts/mean_field_tools.py` for Python commands and stable developer utilities;
- `../scripts/mean_field_tools.jl` for Julia reference-export helpers;
- `../scripts/submit_mean_field.sbatch` for generic Slurm execution;
- package CLI groups in `../src/mean_field/cli.py` when the command is part of the stable package API.

New work should first try to call an existing command through these surfaces, for example:

```bash
python ../scripts/mean_field_tools.py <existing-command> ...
sbatch ../scripts/submit_mean_field.sbatch python ../scripts/mean_field_tools.py <existing-command> ...
```

## Rules for new scripts/devtools

Before adding any file under `scripts/` or `src/mean_field/devtools/`, check whether an existing command can accept one more option, subcommand, config file, or input artifact.

Add a new tracked script/devtool only if all of the following are true:

1. The workflow is expected to be reused beyond the current conversation or one paper-panel attempt.
2. It cannot be cleanly expressed as arguments to an existing command.
3. It has a clear owner surface: package CLI, `mean_field_tools.py`, or the generic Slurm wrapper.
4. It has a small validation path such as `--help`, dry-run, syntax check, saved-result check, or a tiny smoke test.
5. It does not bypass existing login-node guards for HF, topology, eigensolver, or response-function compute.

Do not commit per-run `.sbatch` files, timestamped launchers, ad hoc plotting scripts, or narrow parameter sweeps unless the user explicitly asks to preserve them as durable project assets.  Put scratch launchers in ignored locations such as `tmp/` or `scripts/local/`.

## Consolidation target

Long term, `../scripts/` should contain only a few generic entrypoints plus possibly a small number of stable language/reference bridges.  `../src/mean_field/devtools/` should contain reusable implementation modules behind those entrypoints, not a flat collection of one-off runners.

When cleaning up existing files:

- migrate reusable logic into system modules, common analysis modules, or an existing devtool;
- preserve reproducibility details in `docs/`, `reports/`, or result metadata rather than in many near-duplicate launch scripts;
- delete or untrack obsolete wrappers after confirming no current documentation or tests depend on them;
- keep heavy validation on Slurm and use saved-result validators when available.
