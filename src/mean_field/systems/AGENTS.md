# AGENTS.md

## Scope

Applies to physical-system implementations and adapters under `src/mean_field/systems`.

## Source of Truth

- Repository architecture: `../../../docs/architecture.md`.
- Topology framework contract: `../../../docs/topology_framework.md`.
- Reusable HF framework: `../core/hf`.
- Common analysis helpers: `../../analysis/topology` and `../../analysis/response_derivative_gauge.py`.

## Local Guidance

- Each system owns its physical model: Hamiltonian, lattice/basis labels, parameters, gauge/sewing convention, screening choices, projected windows, and paper-specific compatibility wrappers.
- Reuse `../core/hf` for generic HF iteration and projected-HF plumbing. Do not fork the SCF loop unless the generic framework is demonstrably insufficient.
- Reuse `../../analysis/topology` for the full `FHSState -> generic boundary sewing -> Berry plaquette flux -> Chern` path. System topology modules should only build `FHSState` objects with band/flavor metadata and `BlockSewingSpec` basis labels; do not keep system-private topology sewing transforms, Chern calculators, or historical topology result dataclasses.
- Reuse `../../analysis/response_derivative_gauge.py` for gauge-safe response derivatives when a system needs shift-vector or generalized-derivative logic.
- Keep reproduction scripts and Slurm orchestration out of system core when possible; prefer `scripts/` or `src/mean_field/devtools/` for workflow glue.

## Safety

System validation often diagonalizes large grids or runs HF. Treat it as Slurm work unless it is an explicitly tiny syntax/smoke check.
