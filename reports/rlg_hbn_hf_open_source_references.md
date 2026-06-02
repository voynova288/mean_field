# RnG/hBN HF open-source reference scan

Date: 2026-05-27

## Bottom line

I did **not** find a public repository that directly implements the Kwan et al. MFCI-III RnG/hBN projected-HF calculation (`arXiv:2312.11617`, later PRB 112, 075109) end-to-end. The arXiv page has no GitHub/code link.

The most useful public references are nearby implementations:

1. **`ziweiwang-code/TBG-HF`** — Python projected moire HF code accompanying the Kwan et al. mean-field user guide (`arXiv:2511.21683`).
   - URL: https://github.com/ziweiwang-code/TBG-HF
   - Local copy: `reference/TBG-HF`
   - License: GPL-3.0 (`LICENSE.md`)
   - Best for: projector convention, reference projectors (`average`, `CN`, `average central`), direct/exchange contraction, ODA mixing, form-factor indexing.
   - Key files:
     - `mainProgram.py`: HF driver and ODA update.
     - `projectors.py`: `P_ref` / initial projector conventions.
     - `routines.py`: `calc_fock_matrix`, `calc_E`, `aufbau`.
     - `singleParticle.py`: form-factor generation and momentum/G-index convention.

2. **`xywang2017/TBG_HartreeFock`** — older Julia TBG HF code; likely close to the historical Julia workflow this repo rewrote.
   - URL: https://github.com/xywang2017/TBG_HartreeFock
   - Local copy: `reference/TBG_HartreeFock`
   - License: MIT (`LICENSE`)
   - Best for: legacy TBG/B0 normalization, overlap/Hartree/Fock sign, ODA logic, comparison with existing `benchmarks/b0` style artifacts.
   - Key file: `B0/libs/HF_mod.jl` (`add_HartreeFock`, `add_Hartree`, `add_Fock`, `compute_HF_energy`).

3. **`zybbigpy/R5G-AHC`** — self-contained Python/JAX/Numba HF for rhombohedral multilayer graphene, not hBN moire MFCI-III.
   - URL: https://github.com/zybbigpy/R5G-AHC
   - Local copy: `reference/R5G-AHC`
   - License: no explicit license found; use as a read-only reference unless permission is clarified.
   - Best for: rhombohedral multilayer layer index convention, layer-resolved Coulomb tensor, simple SCF loop, SK tight-binding construction.
   - Key files:
     - `hf_scf.py`: builds projected Hartree/Fock tensors (`make_Lamk_hartree`, `make_Lamk_fock`, `make_Hartree`, `make_Fock`, `mean_field`).
     - `moire_hamk.py`: layer/sublattice basis ordering and displacement field.
     - `moire_lattice.py`: moire k/G grids.
   - Important limitation: it uses a simplified 2D dual-gate potential plus exponential interlayer suppression and has no hBN moire potential / screened-basis projection / average-vs-CN scheme as in MFCI-III.

4. **`PandaTurtleMan/GrapheneHartreeFock`** — generic Julia graphene HF in C4 superlattice / LL basis.
   - URL: https://github.com/PandaTurtleMan/GrapheneHartreeFock
   - Local copy: `reference/GrapheneHartreeFock`
   - Best for: DIIS/projector iteration patterns and tensor sanity checks only; physically not close to RnG/hBN.

## Mapping to current code

Current RnG/hBN implementation lives mainly in:

- `src/mean_field/systems/RnG_hBN/hf.py`
- `src/mean_field/systems/RnG_hBN/interaction.py`
- `src/mean_field/systems/RnG_hBN/screening.py`
- `src/mean_field/devtools/run_rlg_hbn_paper_hf.py`

Suggested cross-checks:

| Issue to check | Current code | Best public reference |
|---|---|---|
| `P_ref` / density-delta convention | `rlg_hbn_reference_density`, `rlg_hbn_density_delta` | `reference/TBG-HF/projectors.py` |
| Hartree/Fock signs and energy 1/2 factors | `build_rlg_hbn_interaction_components`, `compute_hf_energy` | `reference/TBG-HF/routines.py`, `reference/TBG_HartreeFock/B0/libs/HF_mod.jl` |
| form-factor momentum convention | `calculate_layer_projected_overlap_between`, `_source_grid_shift_without_wrap` | `reference/TBG-HF/singleParticle.py::gen_form_factors` |
| layer-resolved Coulomb and layer traces | `interaction.py`, `build_rlg_hbn_interaction_components` | `reference/R5G-AHC/hf_scf.py` |
| screened-basis projection / physical h0 | `build_rlg_hbn_projected_basis`, `_assert_average_remote_hamiltonian_contract` | no exact public implementation found; use MFCI-III paper equations/checkpoints |
| remote-valence average correction | `_prepare_remote_average_source`, `_remote_average_hamiltonian_from_source` | no exact public implementation found; compare against MFCI-III Appendix B |

## Recommendation

For debugging the current RnG/hBN HF, use `TBG-HF` and `TBG_HartreeFock` as the authoritative **projected-HF convention** references, and use `R5G-AHC` only for **rhombohedral layer/tensor intuition**. The likely RnG/hBN-specific fragile points remain the ones not covered by public code: screened-basis projection, physical vs screened `h0`, q=0 layer Hartree screening, and remote-band average correction.
