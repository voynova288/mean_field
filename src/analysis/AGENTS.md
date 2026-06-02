# AGENTS.md

## Scope

Applies to common analysis and quantum-geometry helpers under `src/analysis`.

## Source of Truth

- Unified topology/Berry-geometry conventions: `../../docs/topology_framework.md` and `topology/README.md`.
- Gauge-safe generalized-derivative conventions: `RESPONSE_DERIVATIVE_GAUGE.md` and `response_derivative_gauge.py`.

## Local Guidance

- `topology/` is the common Berry connection, plaquette flux, and Chern-number framework. System adapters should feed it wavefunctions, selected indices, metadata, and optional boundary sewing transforms.
- `response_derivative_gauge.py` is a reusable WannierBerri-style derivative calculator for Berry connections, generalized derivatives, shift vectors, and gauge/subspace checks.
- `shift_current_htg/` and `shift_current_tbg/` are active reproduction/diagnostic workspaces, not a finished stable shift-current framework. Promote reusable math from those workspaces back into common analysis modules before other systems depend on it.
- Never differentiate raw eigenvector phases or raw `np.angle(A_mn)` values. Use covariant/generalized derivatives or Wilson-link validation.

## Safety

Topology-grid recomputation, response-function grid integration, and broad numerical validation are compute jobs. Use Slurm rather than login-node execution.
