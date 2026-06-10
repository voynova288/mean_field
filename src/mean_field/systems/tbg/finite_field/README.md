# Finite-field TBG magnetic spectrum and Hartree-Fock adapter

This package ports the finite-field magnetic spectrum and Hartree-Fock parts of the author code:

```text
/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/bmLL*.jl
/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock（作者原始代码）/libs/MagneticFieldHF*.jl
```

into the `Mean_Field` layered framework.

Reference: `/data/home/ziyuzhu/TBG_HartreeFock/2310.15982v3.pdf`, SI Sec. III.

Generic finite-magnetic-field bookkeeping — rational fluxes, magnetic meshes, reciprocal-shell shifts, and Streda/Diophantine fillings — lives in `mean_field.core.magnetic_field`. Generic finite-B Hartree-Fock calculation — state/input bundles, stored-projector initialization, density updates, screened interaction contractions, full and tL/IKS reduced kernels, SCF/run helpers, and summaries — lives in `mean_field.core.hf.finite_field`. This TBG package re-exports those helpers for backward compatibility while keeping only BM/LL spectrum construction, TBG projected-overlap assembly, spectrum-to-core adapter APIs, and Fig. 6 paper bookkeeping in the TBG system layer.

## Layering

- `mean_field.core.hf` owns the generic SCF/ODA loop.
- `mean_field.core.magnetic_field` owns system-agnostic finite-B flux/mesh/shell/Diophantine helpers.
- `mean_field.core.hf.finite_field` owns the generic finite-B HF calculation: input bundles, stored-projector conventions, finite-B interaction contractions, full/tL-reduced kernels, initialization, run helpers, and summaries.
- `mean_field.systems.tbg.finite_field.spectrum` owns the non-interacting finite-B BM/LL spectrum:
  - author-code `Params`/`initParamsWithStrain` conventions;
  - LL translation matrix elements `_tLL_v1` / `_tLL_v1_valleyKprime`;
  - magnetic-BZ Hamiltonian construction and central `2q` subband diagonalization;
  - paper-style Hofstadter sweep helpers for Fig.3(a)-like spectra: `paper_hofstadter_fluxes`, `author_landau_cutoff`, `red_chern_minus_one_group_mask`, and `compute_magnetic_spectrum_sweep`;
  - magnetic-translation orbit generation for `Vec`;
  - projected `PΣz` and optional `Λ_(m,n)` overlap blocks, including the author `computeCoulombOverlap_v2` symmetry reduction as `compute_coulomb_overlap_fast`.
- `mean_field.systems.tbg.finite_field.hf` is a thin adapter into the generic finite-B HF core:
  - validates K/K′ `MagneticSpectrumResult` pairs;
  - maps TBG Hofstadter metadata to generic `H0`/`Σz` state inputs;
  - expands valley-resolved `bmLL` overlaps into the generic spin/valley HF basis via the core `expand_valley_overlap_data_to_flavors`;
  - supplies TBG magnetic k-vectors and normalization counts from `mean_field.core.magnetic_field`;
  - provides no-I/O assembly helpers for finite-B HF inputs from K/K′ spectra or BM parameters: `build_finite_field_hf_state_from_spectra`, `build_full_flavor_overlap_data_from_spectra`, and `build_finite_field_hf_inputs_from_spectra` / `build_finite_field_hf_inputs_from_parameters`;
  - keeps only TBG/paper Fig. 6 Streda-line conveniences (`paper_fig6_finite_b_fluxes`, `paper_fig6_branch_cases`).
  Use `reduced_translation=True` on the unified builders for the reduced tL-symmetric/IKS path; the older `build_tl_symmetric_*` names are compatibility wrappers.
- JLD2 metadata production remains outside this module; adapters should save/load the returned arrays in workflow code rather than putting file I/O in the core physics layer.

## Formula map

Paper / author code convention:

```text
P_ab(k) = <d†_a,k d_b,k> - δ_ab/2                         (S54)
E[P] = tr(T P^T) + 1/(2A) Σ_q V_q tr(Λ_q P^T) tr(Λ_-q P^T)
       - 1/(2A) Σ_q V_q tr(Λ_q P^T Λ_-q P^T) + const       (S61)
H_MF(P) = T + 1/A Σ_q V_q tr(Λ_-q P^T) Λ_q
            - 1/A Σ_q V_q Λ_q P^T Λ_-q                    (S64)
```

The Python implementation keeps the same stored-projector contraction used by the existing zero-field port.

## Main entry points

```python
from mean_field.systems.tbg.finite_field import (
    FiniteFieldBMParameters,
    MagneticFlux,
    compute_magnetic_spectrum,
    compute_magnetic_spectrum_sweep,
    paper_hofstadter_fluxes,
    red_chern_minus_one_group_mask,
    compute_coulomb_overlap,
    compute_coulomb_overlap_fast,
    FiniteFieldHartreeFockInputBundle,
    FiniteFieldHartreeFockInputs,
    FiniteFieldHartreeFockState,
    FiniteFieldTLSymmetricHartreeFockInputs,
    MagneticOverlapData,
    magnetic_shell_shifts,
    finite_field_diophantine_filling,
    paper_fig6_finite_b_fluxes,
    paper_fig6_branch_cases,
    build_h0_from_hofstadter_metadata,
    build_finite_field_hf_state_from_spectra,
    build_full_flavor_overlap_data_from_spectra,
    build_finite_field_hf_inputs_from_spectra,
    build_finite_field_hf_inputs_from_parameters,
    build_finite_field_hf_kernel,
    build_finite_field_hf_kernel_from_inputs,
    build_tl_symmetric_finite_field_hf_kernel,
    run_finite_field_hartree_fock,
    run_finite_field_hartree_fock_from_inputs,
    summarize_finite_field_hartree_fock,
)
```

Use `compute_magnetic_spectrum` for a single non-interacting Hofstadter spectrum corresponding to `bmLL.jl` / `bmLL_IKS.jl`. Use `compute_magnetic_spectrum_sweep` for Fig.3(a)-style magnetic spectra over the paper flux grid (`q<=12`, `phi<=1/2` by default) with author `nLL=25*q/p`, author `nq`, and the red C=-1 subband-group mask. The returned `MagneticSpectrumSweepResult.as_point_table()` gives flattened arrays ready for paper-style scatter plots without doing file I/O.

`compute_coulomb_overlap(result, m, n)` returns one projected `Λ_(m,n)` block in the shape expected by `MagneticOverlapData`; `compute_coulomb_overlap_fast` is the production-style symmetry-reduced path corresponding to author `computeCoulombOverlap_v2`.

Use `paper_fig6_finite_b_fluxes` and `paper_fig6_branch_cases(s,t)` to reproduce the selected Fig. 6 finite-B branch grid and its fillings `nu=s+t*phi/phi0` without hard-coded decimal fillings. Use `magnetic_shell_shifts` plus `build_finite_field_hf_inputs_from_spectra` to assemble an HF input bundle from already computed K/K′ spectra. Use `build_finite_field_hf_inputs_from_parameters` when the no-I/O helper should also compute the two valley spectra. These helpers require explicit `shifts` or `shell_ng`, so expensive overlap generation is never accidental.

The same builder API covers both finite-B HF variants:

- default `reduced_translation=False`: full magnetic-BZ path corresponding to `MagneticFieldHF.jl` / `MagneticFieldHF_IKS.jl` and returns `FiniteFieldHartreeFockInputs`;
- `reduced_translation=True`: reduced IKS/tL-symmetric path corresponding to `MagneticFieldHF_tLSymmetric*_IKS*.jl` and returns `FiniteFieldTLSymmetricHartreeFockInputs`.

For assembled bundles, use `build_finite_field_hf_kernel_from_inputs` / `run_finite_field_hartree_fock_from_inputs` for both paths; they dispatch on the input-bundle type. The explicit `build_tl_symmetric_*_from_inputs` and `run_tl_symmetric_*_from_inputs` names remain as compatibility wrappers.

Use `summarize_finite_field_hartree_fock` for no-I/O checkpoint summaries.  Its `single_particle_gap` is the occupied/unoccupied HF eigenvalue gap of the finite calculation, not a many-body charge gap.

## Validation

Lightweight software/formula tests:

```bash
PYTHONPATH=src pytest -q tests/test_tbg_finite_field_spectrum.py tests/test_tbg_finite_field_hf.py tests/test_core_hf_engine.py tests/test_core_hf_api.py
```

Passing these tests validates paper-style flux/red-group sweep helpers, LL helper identities, zero-tunneling central LLs, magnetic-orbit vector shapes, `Λ(q=0)=I`, fast-vs-direct `Λ_(m,n)` equivalence, valley-overlap expansion, array ordering, finite-B normalization, projector filling, full and tL-symmetric spectrum-to-HF input assembly, tiny assembled-input HF/SCF smokes through `core/hf` including a `q=2` magnetic-strip case, and the full/tL-reduced Hartree-Fock contractions against independent author-loop equivalents.

A longer direct `test001` validation is recorded in `reports/tbg_finite_field_module_port_validation_20260607.md`, with summary JSON under `results/tbg_finite_field_port_validation_20260607/`. It validates `1/q` fluxes through `q=12` using the author `nLL=25*q/p` cutoff. A full paper phase point still requires a Slurm or deliberate test-node B-SCHF production run.
