# AGENTS.md

## Scope

Applies to common analysis and quantum-geometry helpers under `src/analysis`.

## Source of Truth

- Unified topology/Berry-geometry conventions: `../../docs/topology_framework.md` and `topology/README.md`.
- Gauge-safe generalized-derivative conventions: `RESPONSE_DERIVATIVE_GAUGE.md` and `response_derivative_gauge.py`.
- Generic shift-current API: `shift_current/README.md` and `shift_current/core.py`.

## Local Guidance

- `topology/` is the common Berry connection, plaquette flux, and Chern-number framework. System adapters should feed it wavefunctions, selected indices, metadata, and optional boundary sewing transforms.
- `response_derivative_gauge.py` is a reusable WannierBerri-style derivative calculator for Berry connections, generalized derivatives, shift vectors, and gauge/subspace checks.
- `shift_current/` is the common shift-current layer: components, named WannierBerri/Joya conventions, Fermi occupations, Lorentzian conventions, transition tables, heatmap accumulation, one-k-point tensor APIs, and lightweight reference/toy checks under `shift_current/toy_models/`. System-specific code should provide Hamiltonian eigenpairs and derivatives, then call this API.
- The old `shift_current_htg/` and `shift_current_tbg/` analysis workspaces have been retired. Put hTG/TBG system adapters under `mean_field.systems`, reusable math under `shift_current/` or `response_derivative_gauge.py`, and historical paper audits in ignored local reports/internal workspaces.
- Never differentiate raw eigenvector phases or raw `np.angle(A_mn)` values. Use covariant/generalized derivatives or Wilson-link validation.

## Safety

Topology-grid recomputation, response-function grid integration, and broad numerical validation are compute jobs. Use Slurm rather than login-node execution.
