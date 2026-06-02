# HTG Fig. 7 Physics Audit 2026-05-01

## Context

The interrupted session was checking why the existing HTG Fig. 7 reproduction differed visibly from Kwan, Ledwith, Lo, and Devakul (2023). The Hartree-Fock convention should be checked against `reference/Kwan 等 - 2023 - Strong-coupling topological states and phase transitions in helical trilayer graphene.pdf` and, where useful for the generic HF contraction convention, `/data/home/ziyuzhu/TBG_HartreeFock`.

Note: `reference/2310.15982v3.pdf` is a different finite-field TBG paper and should not be used as the HTG Fig. 7 reference.

## Findings

1. The stored-projector convention is not the immediate bug.

   The TBG reference code uses the background-subtracted projector convention in the HF contractions, and the Python core's stored-projector path is intentionally aligned with that convention.

2. The previous HTG Fig. 7 run was not at the paper parameter point.

   The existing output used `w_ev = 0.110` but still used the older HTG default `v_F = 1.03e6 m/s`, giving `alpha = 0.321163` for the Fig. 7a point. The Kwan 2023 continuum model states `v_F = 8.8e5 m/s` and fixes `w_AB = 110 meV`, giving `alpha = 0.375907` at `theta = 1.70 deg`.

3. The HTG interaction default kept a finite exact `q = 0` Coulomb value.

   The validated TBG HF reference implementation has `V(q=0) = 0` by default. Keeping the finite double-gate limit at the exact discrete zero point changes the finite-grid HF self-energy, especially the Fock cache. HTG `InteractionParams()` now defaults to `finite_zero_limit = false`, while the core Coulomb helper can still evaluate the finite limit when explicitly requested.

4. The old quick Fig. 7 run was a fixed-sector representative run, not the full paper-scale phase search.

   The corrected rerun below intentionally preserves that narrow scope: it recomputes the two Fig. 7 representative sectors with corrected physical inputs. It does not replace a >300-seed phase-diagram search.

## Code Changes

- `src/mean_field/systems/htg/params.py`
  - Added `KWAN_2023_FERMI_VELOCITY_M_PER_S = 8.8e5`.
  - Added `KWAN_2023_TUNNELING_EV = 0.110`.
  - Added `HTGParams.kwan2023()`.
  - Changed HTG `InteractionParams.finite_zero_limit` default to `False`.

- `src/mean_field/devtools/run_htg_hf.py`
  - Added `--fermi-velocity-m-per-s`.
  - Changed HF runner default `--w-ev` to `0.110`.
  - Records `fermi_velocity_m_per_s`, `vf_ev_nm`, `finite_zero_limit`, `drop_q0_coulomb`, and `zero_cutoff_nm_inv` in `hf_params.json`.

- `scripts/submit_mean_field.sbatch`
  - Fixed repo-root detection under Slurm by preferring `SLURM_SUBMIT_DIR` when it is a Mean_Field checkout. Without this, Slurm's copied script path can make `REPO_ROOT=/var/spool/slurmd`, causing `mkdir /var/spool/slurmd/logs` failures.

- Historical one-off Fig. 7 launch scripts were retired during script-surface cleanup.
  - Future reruns should use `scripts/submit_mean_field.sbatch` with `scripts/mean_field_tools.py run_htg_hf ...` instead of adding a new tracked wrapper per sector.

## Verification

Local-safe checks:

```bash
PYTHONPATH=/data/home/ziyuzhu/Mean_Field/src python3 -m pytest \
  tests/test_core_hf_coulomb.py \
  tests/test_htg_lattice.py \
  tests/test_htg_mean_field_adapter.py
```

Result: `16 passed`.

The runner correctly refused a direct login-node smoke run on `login001`.

Slurm rerun:

- Submission host: `login002`
- Job: `83827`
- Node: `node015`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:07:56`
- Output root: `results/HTG/htg_fig7_corrected_20260501_203016`

Corrected Fig. 7a representative:

- Path: `results/HTG/htg_fig7_corrected_20260501_203016/fig7a_d2b2_theta170_w75_w110_nk18`
- Class: `[D2 B2]`
- `v_F = 8.8e5 m/s`
- `w_AB = 0.110 eV`
- `w_AA = 0.075 eV`
- `finite_zero_limit = false`
- `drop_q0_coulomb = true`
- `alpha = 0.3759069757`
- Converged: `true`
- Iterations: `29`
- HF gap: `22.8805 meV`
- Path gap: `22.8308 meV`
- Hermitian residual: `7.10e-18`
- Projector residual: `1.22e-15`
- Figure: `fig7_spin_resolved_bands.png`

Corrected Fig. 7b representative:

- Path: `results/HTG/htg_fig7_corrected_20260501_203016/fig7b_d3_theta180_w75_w110_nk18`
- Class: `[D3]`
- `v_F = 8.8e5 m/s`
- `w_AB = 0.110 eV`
- `w_AA = 0.075 eV`
- `finite_zero_limit = false`
- `drop_q0_coulomb = true`
- `alpha = 0.3550248320`
- Converged: `true`
- Iterations: `1`
- HF gap: `20.0774 meV`
- Path gap: `25.0620 meV`
- Hermitian residual: `3.50e-18`
- Projector residual: `1.11e-15`
- Figure: `fig7_spin_resolved_bands.png`

## Remaining Risk

The corrected figures are materially closer to the paper parameter point and no longer hide the wrong velocity or `q=0` convention in metadata. They are still fixed-sector representative calculations. A final paper-grade claim should add either paper-data digitization or a direct comparison workflow against extracted Fig. 7 curves, and should run the full seed/state search used for the paper phase selection if the goal is to reproduce the phase diagram rather than the two representative band plots.
