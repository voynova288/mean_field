# AGENTS.md

## Scope

Applies to common analysis helpers under `src/analysis`.

## Source of Truth

- Minimal FHS/Wilson topology conventions: `../../docs/topology_framework.md` and `topology/README.md`.
- Gauge-safe generalized-derivative conventions: `RESPONSE_DERIVATIVE_GAUGE.md` and `response_derivative_gauge.py`.
- Generic shift-current API: `shift_current/README.md` and `shift_current/core.py`.
- Unified optical-response front door: `optical/README.md` and `optical/core.py`.

## Local Guidance

- `topology/` is only the common `FHSState -> generic boundary sewing -> FHS/Wilson-link -> plaquette-flux -> Chern-number` framework. System adapters should feed it unified state objects with band/flavor metadata and `BlockSewingSpec` basis labels; QGT/FS metric, paper-target helpers, system-private sewing transforms, and system-private Chern calculators do not belong there.
- `response_derivative_gauge.py` is a reusable WannierBerri-style derivative calculator for Berry connections, generalized derivatives, shift vectors, and gauge/subspace checks.
- `shift_current/` is the common shift-current layer: components, named WannierBerri/Joya conventions, Fermi occupations, Lorentzian conventions, transition tables, heatmap accumulation, one-k-point tensor APIs, and lightweight reference/toy checks under `shift_current/toy_models/`. System-specific code should provide Hamiltonian eigenpairs and derivatives, then call this API.
- `injection_current/` is the common injection-current / CPGE formula layer. It reuses the Hamiltonian-gauge velocity/Berry connection infrastructure and adds the length-gauge `Delta v * r * r` kernel.
- `optical/` is the front-door API for future optical workflows. New workflow/system code should usually call `analysis.optical` first, choosing `kind="shift_current"` or `kind="injection_current"`, instead of branching manually between the two formula packages.
- The old `shift_current_htg/` and `shift_current_tbg/` analysis workspaces have been retired. Put hTG/TBG system adapters under `mean_field.systems`, reusable math under `optical/`, `shift_current/`, `injection_current/`, or `response_derivative_gauge.py`, and historical paper audits in ignored local reports/internal workspaces.
- Never differentiate raw eigenvector phases or raw `np.angle(A_mn)` values. Use covariant/generalized derivatives or Wilson-link validation.

## Safety

Topology-grid recomputation, response-function grid integration, and broad numerical validation are compute jobs. Use Slurm rather than login-node execution.
