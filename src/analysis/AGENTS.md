# AGENTS.md

## Scope

Applies to common analysis and quantum-geometry helpers under `src/analysis`.

## Source of Truth

- Unified topology/Berry-geometry conventions: `../../docs/topology_framework.md` and `topology/README.md`.
- Gauge-safe generalized-derivative conventions: `RESPONSE_DERIVATIVE_GAUGE.md` and `optical_response/gauge.py`.
- Generic shift-current API: `optical_response/shift_current.py` and compatibility shims under `shift_current/`.

## Local Guidance

- `topology/` is the common Berry connection, plaquette flux, and Chern-number framework. System adapters should feed it wavefunctions, selected indices, metadata, and optional boundary sewing transforms.
- `optical_response/gauge.py` is the reusable WannierBerri-style derivative facade for Berry connections, generalized derivatives, shift vectors, and gauge/subspace checks; implementation is split across `optical_response/gauge_*` modules. `response_derivative_gauge.py` is only a compatibility shim.
- `optical_response/` is the common shift-current/optical-response layer: components, named WannierBerri/Joya conventions, Fermi occupations, Lorentzian conventions, transition tables, heatmap accumulation, one-k-point tensor APIs, and lightweight reference/toy checks. Historical `shift_current/` paths re-export this API for compatibility.
- The old `shift_current_htg/` and `shift_current_tbg/` analysis workspaces have been retired. Put hTG/TBG system adapters under `mean_field.systems`, reusable math under `optical_response/`, and historical paper audits in ignored local reports/internal workspaces.
- Never differentiate raw eigenvector phases or raw `np.angle(A_mn)` values. Use covariant/generalized derivatives or Wilson-link validation.

## Safety

Topology-grid recomputation, response-function grid integration, and broad numerical validation are compute jobs. Use Slurm rather than login-node execution.
