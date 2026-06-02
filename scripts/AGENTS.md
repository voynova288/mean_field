# AGENTS.md

## Scope

Applies to tracked repository entrypoints under `scripts/`.

## Source of Truth

- Script policy: `../docs/script_surface_policy.md`.
- Preferred Python dispatcher: `mean_field_tools.py`.
- Preferred Julia bridge: `mean_field_tools.jl`.
- Preferred Slurm wrapper: `submit_mean_field.sbatch`.

## Local Guidance

- Do not add a new standalone script, timestamped launcher, or one-off `.sbatch` file by default.
- Prefer extending `mean_field_tools.py`, `mean_field_tools.jl`, `../src/mean_field/cli.py`, or an existing devtool module.
- Use `submit_mean_field.sbatch` for Slurm execution instead of creating a custom wrapper for each parameter set.
- Put scratch launchers and temporary sweeps in ignored locations such as `../tmp/` or `local/`, not in the tracked script surface.

## Safety

HF, topology-grid recomputation, response integration, eigensolvers, and broad numerical tests are compute workloads. Scripts that can run them must be Slurm-safe and must not encourage login-node execution.
