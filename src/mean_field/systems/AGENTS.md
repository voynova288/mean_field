# AGENTS.md

## Scope

Applies to physical-system implementations and adapters under `src/mean_field/systems`.

## Source of Truth

- Repository architecture: `../../../docs/architecture.md`.
- Minimal common topology APIs: `../../analysis/topology` (`FHSState`, `BlockSewingSpec`, FHS/Wilson link, plaquette flux, Chern). Archived QGT/paper workflows, if explicitly needed: `../../../local_archive/retired_surface/topology_untracked_20260622/` or `../../../local_archive/optional_features/topology_quantum_geometry_20260625/`.
- Reusable HF framework: `../core/hf`.
- Common response helpers: `../../analysis/optical_response`.

## Local Guidance

- Each system owns its physical model: Hamiltonian, lattice/basis labels, parameters, gauge/sewing convention, screening choices, projected windows, and paper-specific compatibility wrappers.
- Reuse `../core/hf` for generic HF iteration and projected-HF plumbing. Do not fork the SCF loop unless the generic framework is demonstrably insufficient.
- Concrete topology modules should stay as state builders and delegate all FHS link/plaquette/Chern calculations to `../../analysis/topology`; currently `tmbg/topology.py`, `tdbg/topology.py`, `RnG_hBN/topology.py`, and `htg/topology.py` expose `fhs_state_*` builders only. They may attach band/flavor metadata and `BlockSewingSpec` basis labels, but must not expose `compute_topology_*`, `topology_on_grid`, system-local sewing transforms for ordinary block bases, or topology result dataclasses.
- Reuse `../../analysis/optical_response` for gauge-safe response derivatives when a system needs shift-vector or generalized-derivative logic; old `../../analysis/response_derivative_gauge.py` is only a compatibility shim.
- Keep reproduction scripts and Slurm orchestration out of system core. Current public surface does not track `src/mean_field/devtools/`; workflow glue should stay in ignored local scratch/archive unless reintroduced through a reviewed `scripts/` entrypoint.

## Safety

System validation often diagonalizes large grids or runs HF. Treat it as Slurm work unless it is an explicitly tiny syntax/smoke check.
