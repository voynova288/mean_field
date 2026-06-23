# AGENTS.md

## Scope

Applies to physical-system implementations and adapters under `src/mean_field/systems`.

## Source of Truth

- Repository architecture: `../../../docs/architecture.md`.
- Minimal common topology APIs: `../../analysis/topology` (`core.py`, `quantum_geometry.py`, `wavefunction.py`, `system.py`); archived concrete system topology wrappers/projected-HF reconstruction/paper workflows, if explicitly needed: `../../../local_archive/retired_surface/topology_untracked_20260622/`.
- Reusable HF framework: `../core/hf`.
- Common response helpers: `../../analysis/optical_response`.

## Local Guidance

- Each system owns its physical model: Hamiltonian, lattice/basis labels, parameters, gauge/sewing convention, screening choices, projected windows, and paper-specific compatibility wrappers.
- Reuse `../core/hf` for generic HF iteration and projected-HF plumbing. Do not fork the SCF loop unless the generic framework is demonstrably insufficient.
- Concrete topology/Berry-geometry system wrappers should stay thin and delegate to `../../analysis/topology`; currently `tmbg/topology.py` and `tdbg/topology.py` are restored. Use the common topology API for FHS link/plaquette/Chern calculations, projector QGT/quantum metric, wavefunction-grid canonicalization, and thin metadata packaging on already-built eigenvector grids; reintroduce any additional concrete system wrapper only through a small reviewed API instead of restoring all historical wrappers.
- Reuse `../../analysis/optical_response` for gauge-safe response derivatives when a system needs shift-vector or generalized-derivative logic; old `../../analysis/response_derivative_gauge.py` is only a compatibility shim.
- Keep reproduction scripts and Slurm orchestration out of system core. Current public surface does not track `src/mean_field/devtools/`; workflow glue should stay in ignored local scratch/archive unless reintroduced through a reviewed `scripts/` entrypoint.

## Safety

System validation often diagonalizes large grids or runs HF. Treat it as Slurm work unless it is an explicitly tiny syntax/smoke check.
