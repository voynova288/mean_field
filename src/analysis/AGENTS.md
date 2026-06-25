# AGENTS.md

## Scope

Applies to common analysis and response helpers under `src/analysis`.

## Source of Truth

- Gauge-safe generalized-derivative conventions: `RESPONSE_DERIVATIVE_GAUGE.md` and `optical_response/gauge.py`.
- Generic shift-current API: `optical_response/shift_current.py` and compatibility shims under `shift_current/`.
- Minimal topology APIs: FHS core `topology/core.py`; wavefunction-grid canonicalization `topology/wavefunction.py`; system-facing adapter `topology/system.py`; archived concrete system wrappers/projected-HF reconstruction/QGT helpers/paper workflows, if explicitly needed: `../../local_archive/retired_surface/topology_untracked_20260622/` or `../../local_archive/optional_features/topology_quantum_geometry_20260625/`.

## Local Guidance

- `optical_response/gauge.py` is the reusable WannierBerri-style derivative facade for Berry connections, generalized derivatives, shift vectors, and gauge/subspace checks; implementation is split across `optical_response/gauge_*` modules. `response_derivative_gauge.py` is only a compatibility shim.
- `optical_response/` is the common shift-current/optical-response layer: components, named WannierBerri/Joya conventions, Fermi occupations, Lorentzian conventions, heatmap accumulation, one-k-point tensor APIs, and lightweight reference/toy checks. Historical `shift_current/` paths re-export this API for compatibility.
- `topology/` is a reviewed common API for FHS link/plaquette/Chern calculations, generic wavefunction-grid layout canonicalization, and a thin system-facing adapter for already-built eigenvector grids. Do not restore concrete system wrappers, projected-HF reconstruction helpers, QGT helpers, or paper-specific topology workflows wholesale; reintroduce them only after deciding their public boundary and validation target.
- The old `shift_current_htg/` and `shift_current_tbg` analysis workspaces have been retired. Put hTG/TBG system adapters under `mean_field.systems`, reusable math under `optical_response/`, and historical paper audits in ignored local reports/internal workspaces.
- Never differentiate raw eigenvector phases or raw `np.angle(A_mn)` values. Use covariant/generalized derivatives or Wilson-link validation.

## Safety

Response-function grid integration and broad numerical validation are compute jobs. Use Slurm rather than login-node execution.
