# AGENTS.md

## Scope

Applies to the Chaudhary-2021/TBG shift-current reproduction workspace.

## Source of Truth

- Local plan/status: `CHAUDHARY2021_PLAN.md`, `CHAUDHARY2021_STATUS.md`, `CHAUDHARY2021_PHYSICS_AUDIT.md`.
- Reusable derivative conventions: `../RESPONSE_DERIVATIVE_GAUGE.md`.
- TBG model inputs live in `../../mean_field/systems/tbg/zero_field`.

## Local Guidance

- This directory is still an active reproduction/diagnostic workspace, not a stable shift-current API.
- The validated reusable piece is the WannierBerri-style, gauge-safe derivative path in `../response_derivative_gauge.py`; selected-pair production code should use that rather than differentiating phases directly.
- Keep TBG Hamiltonian/model conventions in `../../mean_field/systems/tbg/zero_field`; this directory should orchestrate response diagnostics and paper comparisons.
- Do not fix Fig. 2/Fig. 4 mismatches by visual scaling, hand-tuned transition filters, or cosmetic plotting. Isolate the model, occupation, screening, or response-formula issue first.

## Safety

Hartree response runs, dense k-grid scans, and broad convergence checks must go through Slurm. Login-node work should be limited to file inspection and very small syntax/smoke checks.
