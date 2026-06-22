# AGENTS.md

## Scope

Applies to physical-system implementations and adapters under `src/mean_field/systems`.

## Source of Truth

- Repository architecture: `../../../docs/architecture.md`.
- Topology framework contract: `../../../docs/topology_framework.md`.
- Reusable HF framework: `../core/hf`.
- Common analysis helpers: `../../analysis/topology` and `../../analysis/optical_response`.

## Local Guidance

- Each system owns its physical model: Hamiltonian, lattice/basis labels, parameters, gauge/sewing convention, screening choices, projected windows, and paper-specific compatibility wrappers.
- Reuse `../core/hf` for generic HF iteration and projected-HF plumbing. Do not fork the SCF loop unless the generic framework is demonstrably insufficient.
- Reuse `../../analysis/topology` for Berry links, plaquette flux, and Chern numbers. System topology modules should generate wavefunction meshes, choose/label states, supply sewing transforms, and map results to historical dataclasses.
- Reuse `../../analysis/optical_response` for gauge-safe response derivatives when a system needs shift-vector or generalized-derivative logic; old `../../analysis/response_derivative_gauge.py` is only a compatibility shim.
- Keep reproduction scripts and Slurm orchestration out of system core. Current public surface does not track `src/mean_field/devtools/`; workflow glue should stay in ignored local scratch/archive unless reintroduced through a reviewed `scripts/` entrypoint.

## Safety

System validation often diagonalizes large grids or runs HF. Treat it as Slurm work unless it is an explicitly tiny syntax/smoke check.
