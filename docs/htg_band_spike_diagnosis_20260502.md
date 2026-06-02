# HTG band spike diagnosis, 2026-05-02

## Context

The interrupted Codex session was comparing the apparently smooth
`results/HTG/htg_fig8a_nu4_theta180_w75_w110_nk18_20260430_v2/hf_bands_path.png`
against the later Fig. 7/Fig. 8 companion band plots under
`results/HTG/htg_fig7_corrected_20260501_203016`.

## Findings

- The saved `hf_bands_path.npz` arrays do not contain large discontinuities.
  For the 2026-05-01 Fig. 7(a)/(b) outputs, the 48-point path data had maximum
  adjacent energy steps of about `1.83 meV` and `1.66 meV`, and maximum second
  differences of about `3.32 meV` and `3.27 meV`.
- The Fig. 8-style `E_Fock` spikes in the 2026-05-01 and 2026-05-02 `nu=+2`
  potential plots were not a valid physical feature and not just a plotting
  window issue. They came from evaluating the path Fock potential with
  `finite_zero_limit=False`: when a path point coincides with an SCF-grid point,
  the exact `q=0` screened Coulomb entry was set to zero, while neighboring
  points used the finite `q -> 0` limit. This creates isolated upward Fock
  cusps.
- Job `84171` recomputed only the high-symmetry path from the saved 2026-05-01
  SCF states at `120 points/segment`. It did not rerun HF self-consistency.
  The new 120-point path data reduced maximum adjacent energy steps to about
  `1.17 meV` and `1.16 meV`, and maximum second differences to about
  `2.22 meV` and `2.29 meV`.
- Job `84226` recomputed the Fig. 8-style Hartree/Fock path potentials from
  the saved SCF states with `finite_zero_limit=True` and updated the default
  `fig8a_hartree_fock_potentials.png/pdf` and `fig8a_potential_path.npz` in:
  `results/HTG/htg_fig7_corrected_20260501_203016/fig7a_*`,
  `results/HTG/htg_fig7_corrected_20260501_203016/fig7b_*`,
  `results/HTG/htg_fig7_kwan_zeta0_20260502_002/fig7a_*`, and
  `results/HTG/htg_fig7_kwan_zeta0_20260502_002/fig7b_*`.
  The old q0-zero spiky files were preserved with `_q0zero_spiky` suffixes.

## Artifacts

- Fig. 7(a), 120-point bands:
  `results/HTG/htg_fig7_corrected_20260501_203016/fig7a_d2b2_theta170_w75_w110_nk18/hf_bands_path_p120.png`
- Fig. 7(b), 120-point bands:
  `results/HTG/htg_fig7_corrected_20260501_203016/fig7b_d3_theta180_w75_w110_nk18/hf_bands_path_p120.png`
- 120-point NPZ files:
  `hf_bands_path_p120.npz` in the same two result directories.
- Fig. 8(a) 2026-04-30 SCF-grid-only path diagnostic:
  `results/HTG/htg_fig8a_nu4_theta180_w75_w110_nk18_20260430_v2/hf_scf_grid_path_bands.png`
- Fig. 7(a)/(b) 2026-05-01 SCF-grid-only path diagnostics:
  `hf_scf_grid_path_bands.png` and `hf_scf_grid_path_bands.tsv` in each
  `results/HTG/htg_fig7_corrected_20260501_203016/fig7*` leaf directory.
- Fig. 8(a) 2026-04-30 p120 visual check:
  `results/HTG/htg_fig8a_nu4_theta180_w75_w110_nk18_20260430_v2/hf_bands_path_p120_offset_aligned.png`.
  This is only a visual alignment check; the raw p120 recomputation has a
  legacy absolute-offset mismatch with the 2026-04-30 metadata.
- Corrected finite-q0 Fig. 8-style potential plots:
  `fig8a_hartree_fock_potentials.png` in each corrected 2026-05-01/2026-05-02
  Fig. 7 leaf directory. The explicit finite-q0 copies are also saved as
  `fig8a_hartree_fock_potentials_finiteq0.png`.

## Code changes

Historical one-off scripts used during this diagnosis were retired during script-surface cleanup.  Future reruns should use the generic dispatcher/Slurm wrapper (`scripts/mean_field_tools.py` plus `scripts/submit_mean_field.sbatch`) and the reusable HTG HF entrypoint `mean_field.devtools.run_htg_hf` rather than restoring per-case scripts.

The durable implementation changes from this diagnosis live in the HTG system modules and retained devtools:

- old `hf_params.json` compatibility for missing `zeta_rad`, `fermi_velocity_m_per_s`, `finite_zero_limit`, or `zero_cutoff_nm_inv` should be handled in reusable loaders/runners;
- corrected reruns should pass explicit path resolution, band-window, and finite-q0 settings through a generic command rather than a timestamped launcher;
- saved-SCF-grid diagnostics should be implemented as reusable plotting/post-processing commands when needed, not as one-off tracked scripts.

## 2026-05-02 Gamma-m-Gamma path correction

The residual Fig. 7(b) mismatch on the gamma-m-gamma part was not the same
issue as the earlier spike.  `build_paper_hf_kpath()` previously used
`Gamma -> M -> Gamma` with the final Gamma equal to the original central-zone
Gamma, which simply retraced the gamma-m segment.  The paper path should pass
through the mBZ edge midpoint and continue to the reciprocal-lattice-equivalent
Gamma across that edge, i.e. `Gamma -> M -> Gamma + b_m1`.

Code updates:

- `src/mean_field/systems/htg/lattice.py` now uses `Gamma + b_m1` for the
  final Gamma in `build_paper_hf_kpath()`.
- `tests/test_htg_lattice.py` now checks that `M` is the midpoint between the
  two Gamma copies and that the final Gamma differs from the initial one by
  `b_m1`.
- `docs/htg_fig7_physics_audit_20260501.md` now points to the local Kwan HTG
  PDF; `reference/2310.15982v3.pdf` is a different finite-field TBG paper.

Verification:

- Local geometry test: `/data/home/ziyuzhu/miniconda3/bin/python3 -m pytest
  tests/test_htg_lattice.py -q` -> `4 passed`.
- Slurm job `84286` recomputed the four saved Fig. 7 states with suffix
  `_p160_crossm_finiteq0`; state `COMPLETED`, elapsed `00:03:38`, exit `0:0`.
- New Fig. 7 outputs:
  `fig7_spin_resolved_bands_p160_crossm_finiteq0.png/pdf` and
  `hf_bands_path_p160_crossm_finiteq0.png/pdf/npz` in each 2026-05-02 Fig. 7
  leaf directory.
- The D3 cases now have a non-retraced gamma-m-gamma segment; the last-segment
  mirror-difference diagnostic is about `15.9 meV` instead of a folded-back
  duplicate.
