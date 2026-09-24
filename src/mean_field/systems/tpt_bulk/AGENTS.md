# TPT bulk system guidance

## Scope

Applies only to the periodic 40-atom/320-Wannier bulk TPT lineage in this directory.

## Source of Truth

- Physical input gates: `../../../../results/tpt_bulk_mean_field/DATA_REQUIREMENTS.md`.
- Rejection of the former interaction model: `../../../../results/tpt_bulk_mean_field/RETRACTION_UNSUPPORTED_G0_HF_20260917.md`.

## Safety

- `source.py` has noninteracting HR authority only. Do not infer microscopic Coulomb density vertices from HR eigenvector coefficients.
- The former cell-monopole/RK/Fock-only `hf.py` model is rejected and removed. Do not restore, extend, or rerun it. The only permitted interacting path is the source-bound `microscopic_hf.py` adapter, which must require an immutable same-parent `psi`/`rho(k,q+G)` bundle and remain unusable while that bundle is absent.
- Missing density vertices, parent/gauge evidence, screening, filling, Hartree, boundary conditions, or double counting must stop the calculation; never replace them with identity/point-centre form factors, `.mmn` neighbour overlaps, borrowed parameters, or zero terms.
- Do not expose or scan a global interaction-strength multiplier. Hartree is mandatory; remove only a rigorously derived uniform background mode.
- Do not create an HF band figure until the interacting Hamiltonian passes the physical-input gate and exact saved-grid validation.
