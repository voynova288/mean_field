# TBG finite-field module port validation (2026-06-07)

Reference: `/data/home/ziyuzhu/TBG_HartreeFock/2310.15982v3.pdf`, especially SI Sec. III.

Author source paths:

- `/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/bmLL.jl`
- `/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/bmLL_IKS.jl`
- `/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/MagneticFieldHF*.jl`
- `/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/initP_helpers.jl`

## Ported modules

Mean_Field paths:

- `src/mean_field/systems/tbg/finite_field/spectrum.py`
- `src/mean_field/systems/tbg/finite_field/hf.py`
- `src/mean_field/systems/tbg/finite_field/__init__.py`
- `src/mean_field/systems/tbg/finite_field/README.md`

## Spectrum port status

Implemented from `bmLL*.jl`:

- finite-B BM parameter convention (`FiniteFieldBMParameters`);
- LL indexing (`in_gamma`) and geometric projections (`projector_para`, `projector_norm`);
- associated-Laguerre LL translation matrix;
- `_tLL_v1` and `_tLL_v1_valleyKprime` equivalents via `tll_matrix(..., valley=...)`;
- magnetic lattice coordinates, including optional K' q0 shift;
- LL Hamiltonian and `Σz` construction;
- central `2q` Hofstadter subband diagonalization;
- magnetic-translation orbit generation for `Vec`;
- projected `PΣz`;
- direct and symmetry-reduced `Λ_(m,n)` overlap generation: `compute_coulomb_overlap` and `compute_coulomb_overlap_fast`.

## HF port status

Implemented from `MagneticFieldHF*.jl` and `initP_helpers.jl`:

- finite-B density convention `P=<d†d>-I/2` in the stored-projector contraction convention;
- full magnetic-BZ Hartree/Fock contraction;
- tL-symmetric / IKS-reduced Hartree/Fock contraction with `phi` phase;
- finite-B filling, density update, and initialization helpers;
- `expand_valley_overlap_data_to_flavors`, matching the author expansion of valley-resolved metadata into both spin sectors;
- integration with reusable `mean_field.core.hf` SCF/ODA machinery.

## Validation run

Lightweight tests on `test001`:

```bash
cd /data/home/ziyuzhu/Mean_Field
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  /data/home/ziyuzhu/miniconda3/bin/python -m pytest -q \
  tests/test_tbg_finite_field_spectrum.py \
  tests/test_tbg_finite_field_hf.py \
  tests/test_core_hf_engine.py \
  tests/test_core_hf_api.py
```

Result after the direct dual-code fix, paper-style magnetic-spectrum sweep helpers, full/tL HF input/helper additions, tiny assembled-input SCF smokes, and the unified finite-B HF input/kernel/run API refactor below: `43 passed`.

Default paper-style magnetic-spectrum API smoke on `test001` also passed: `compute_magnetic_spectrum_sweep` with `theta=1.20°`, `q<=12`, `phi<=1/2`, author `nLL=25*q/p`, and author `nq` produced `case_count=23`, `row_count=892`, `red_point_count=137`, and energy range about `[-17.19044, 17.19044] meV`.

Refactor check: before the unified API cleanup, the finite-field/core subset was `43 passed`; after consolidating full and tL-symmetric input/kernel/run convenience paths through `build_finite_field_hf_inputs_from_spectra`, `build_finite_field_hf_inputs_from_parameters`, `build_finite_field_hf_kernel_from_inputs`, and `run_finite_field_hartree_fock_from_inputs`, the same subset remained `43 passed` on `test001`.

Added reusable no-I/O HF assembly helpers for the next interacting B-SCHF step:

- `magnetic_shell_shifts` mirrors the author interaction-shell loop/order.
- `build_finite_field_hf_state_from_spectra` builds `FiniteFieldHartreeFockState` from K/K′ `MagneticSpectrumResult` objects.
- `build_full_flavor_overlap_data_from_spectra` computes K/K′ overlaps and expands them to full spin/valley flavor basis.
- `paper_hofstadter_fluxes`, `author_landau_cutoff`, `red_chern_minus_one_group_mask`, and `compute_magnetic_spectrum_sweep` provide the no-I/O core API for paper Fig.3(a)-style non-interacting magnetic spectra. `MagneticSpectrumSweepResult.as_point_table()` returns flattened arrays for plotting/comparison.
- `build_finite_field_hf_inputs_from_spectra` / `build_finite_field_hf_inputs_from_parameters` now provide the unified finite-B HF assembly API. The default returns a full magnetic-BZ `FiniteFieldHartreeFockInputs`; `reduced_translation=True` returns a reduced tL-symmetric/IKS `FiniteFieldTLSymmetricHartreeFockInputs`. The older `build_tl_symmetric_*` names remain compatibility wrappers.
- `build_finite_field_hf_kernel_from_inputs` and `run_finite_field_hartree_fock_from_inputs` now dispatch on either input-bundle type, keeping workflow code on one API while preserving separate full/tL physics contractions internally.
- `build_finite_field_hf_inputs_from_parameters` computes both valley spectra from BM parameters and then assembles the same no-I/O input bundle; it requires explicit `n_landau` and `shifts`/`shell_ng`.
- `build_finite_field_hf_kernel_from_inputs` and `run_finite_field_hartree_fock_from_inputs` start from a `FiniteFieldHartreeFockInputs` bundle while still reusing the generic `core/hf` SCF/ODA loop.
- `FiniteFieldTLSymmetricHartreeFockInputs`, `build_tl_symmetric_finite_field_hf_inputs_from_spectra`, `build_tl_symmetric_finite_field_hf_inputs_from_parameters`, `build_tl_symmetric_finite_field_hf_kernel_from_inputs`, and `run_tl_symmetric_finite_field_hartree_fock_from_inputs` cover the reduced tL-symmetric/IKS path while preserving full magnetic-strip overlap data.
- `summarize_finite_field_hartree_fock` provides no-I/O checkpoint summaries: filling, energy per moire unit cell, chemical potential, finite-system single-particle gap, final norm, iteration count, convergence flag, and exit reason. The single-particle gap is not labeled as a many-body charge gap.
- Tiny non-production HF smoke tests now run assembled inputs through `build_finite_field_hf_kernel` and the reusable `core/hf` SCF/ODA loop for a few iterations, including a `p/q=1/2` case that exercises the full magnetic-strip shape `(16, 2, 16, 2)` and a reduced tL/IKS input smoke where state `nk=1` while overlap/full-k keeps `q*nk=2`.

Direct author-code comparison found and fixed one port bug:

- `src/mean_field/systems/tbg/finite_field/spectrum.py::_hermitian_from_upper` must mirror Julia `Hermitian(H, :U)` by keeping the diagonal once and mirroring only the strict upper triangle.
- The earlier Python helper added the diagonal twice, doubling LL kinetic energies and giving Hofstadter spectra at the wrong energy scale.
- `bmLL_IKS.jl` also uses a random `constructLatticeIKS` mesh shift; direct array comparisons must either use `bmLL.jl` or record/reuse the same `mesh_shift` in Python.

Longer direct `test001` validation:

- script: `tmp/validate_tbg_finite_field_spectrum_sweep_20260607.py`
- log: `logs/tbg_ff_spectrum_validation_20260607.log`
- summary: `results/tbg_finite_field_port_validation_20260607/finite_field_spectrum_sweep_summary.json`
- tmux session exited after completion.

Result summary:

- `status=pass`
- `case_count=15`
- elapsed about `28.1 s`
- fluxes include `1/q` for `q=2..12` with author cutoff `nLL=25*q/p`, plus `2/5`, `3/7`, `4/9`, `5/12`.
- maximum `PΣz` Hermiticity error: `9.04e-16`.
- maximum first-k vector orthonormality error: `4.09e-15`.
- extra finite-overlap checks passed for small `1/2` and `1/3` cases.

## Direct dual-code paper-figure validation

Target: paper `2310.15982` Fig. 3(a), non-interacting Hofstadter spectrum at `theta=1.20°`, valley `K`, one spin component, unstrained BM parameters.

Commands were run on `test001` with author Julia and Mean_Field Python using the same author `bmLL_IKS.jl` random mesh shifts:

- author runner: `tmp/author_fig3a_hofstadter_20260607.jl`
- Python runner: `tmp/python_fig3a_hofstadter_20260607.py`
- comparison/plotter: `tmp/compare_tbg_fig3a_dualcode_20260607.py`
- log: `logs/tbg_fig3a_dualcode_20260607.log`
- result directory: `results/tbg_finite_field_fig3a_dualcode_20260607/`

Result:

- `status=pass`
- `case_count=23` fluxes with `q<=12` and `phi<=1/2`
- rows: author `892`, Python `892`
- max `|ΔE| = 7.372e-13 meV`
- RMS `ΔE = 1.681e-13 meV`
- tolerance: `1e-9 meV`

Key artifacts:

- `results/tbg_finite_field_fig3a_dualcode_20260607/dualcode_fig3a_comparison.md`
- `results/tbg_finite_field_fig3a_dualcode_20260607/dualcode_fig3a_comparison.json`
- `results/tbg_finite_field_fig3a_dualcode_20260607/dualcode_overlay_fig3a_hofstadter.png`
- `results/tbg_finite_field_fig3a_dualcode_20260607/dualcode_fig3a_energy_differences.png`

## Caveats / next production step

This now validates the ported non-interacting finite-B Hofstadter spectrum against the author Julia code for the paper Fig. 3(a) checkpoint. It is still not a full interacting B-SCHF phase-point reproduction: a complete HF phase point with all production overlap shells and SCF settings should be run via Slurm or a deliberate test-node job.
