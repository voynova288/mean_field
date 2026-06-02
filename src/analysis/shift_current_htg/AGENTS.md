# AGENTS.md

## Scope

Applies to the hTTG shift-current reproduction workspace.

## Source of Truth

- Local status and limitations: `README.md`, `REPRODUCTION_STATUS.md`.
- Formula and convention audits: `PHYSICS_AUDIT.md`, `HTG_SYMMETRY_AUDIT.md`, `HTG_AXIS_CONVENTION_AUDIT.md`.
- Reusable derivative conventions: `../RESPONSE_DERIVATIVE_GAUGE.md`.

## Local Guidance

- This directory is not a completed general shift-current framework. It is a reproducibility and diagnostic workspace.
- Visual agreement after coordinate rotation/reflection, post-hoc scaling, selective band filtering, or plotting adjustments is not evidence of reproduction unless the underlying formula/convention issue is resolved.
- Reusable derivative or gauge-covariance logic should be moved to `../response_derivative_gauge.py`; keep this directory for hTTG-specific adapters, diagnostics, and paper checks.
- Before changing signs, axes, units, broadening, tensor labels, or band windows, read the relevant audit note and state what convention is being changed.

## Safety

Production-like spectra and convergence scans are Slurm jobs. Only tiny analytic/finite-difference gates should run directly on a login node.
