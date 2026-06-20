# TDHF core interface contract

Reference papers and planning notes are local/internal inputs; this public note records only the durable TDHF/RPA core API contract.

## Scope

The reusable TDHF/RPA implementation lives in `src/mean_field/core/hf/tdhf.py`. It is system agnostic and adds only the layer needed after a converged HF calculation:

1. fixed collective-momentum particle-hole basis construction;
2. dense debug assembly of `A`, `B`, and `L = [[A, B], [-B*, -A*]]`;
3. ordinary non-Hermitian diagonalization of `L`;
4. eta-metric normalization with `eta = diag(+1, -1)`;
5. flavor-channel grouping helpers.

System-specific gauge choices, form factors, layer Coulomb kernels, screening schemes, Umklapp sums, saved HF-state loading, and paper runners remain in `src/mean_field/systems/<system>/`. The first RLG/hBN bridge lives in `src/mean_field/systems/RnG_hBN/tdhf.py`.

## Required inputs

For a fixed momentum sector `q`, the core expects:

- converged HF eigenvalues `E[alpha]`;
- particle-hole labels `ParticleHolePair(particle, hole, ...)` already filtered to that `q` sector;
- HF-basis two-body matrix elements supplied as one of:
  - a small dense debug tensor `V[a,b,c,d]`,
  - a sparse mapping keyed by `(a,b,c,d)`, or
  - a production callable `V_hf(a,b,c,d)`.

The single-particle Hamiltonian `T_ij` is not an input: kinetic and interaction-scheme details must already be encoded in the converged HF spectrum and HF-basis matrix elements.

## Two-body tensor convention

`V[a,b,c,d]` is the un-antisymmetrized coefficient of

```text
c_b^† c_a^† c_c c_d
```

which is equivalent to `c_a^† c_b^† c_d c_c`. The core formulas are

```text
A[p h, p' h'] = (E[p] - E[h]) delta[p,p'] delta[h,h']
                + V[p,h',h,p'] - V[p,h',p',h]
B[p h, p' h'] = V[p,p',h,h'] - V[p,p',h',h]
```

Do not pass an already antisymmetrized tensor; direct and exchange subtraction is done explicitly here.

## Momentum-sector rule

For translation-invariant HF states, production code must build a separate TDHF block for each collective momentum `q`:

```text
phi = (k+q, particle; k, hole)
```

`build_momentum_sector_particle_hole_pairs(...)` enforces this shape through a system-provided `add_momentum(k, q)` callback. `build_all_particle_hole_pairs(...)` is only for toy models or for lists that have already been filtered to one fixed sector.

## Current status

Implemented and unit-tested in the core layer:

- V-convention smoke test for `A`, `B`, and particle-hole symmetry;
- ordinary non-Hermitian solve with positive eta-metric branch extraction;
- eta-Gram normalization for degenerate subspaces;
- fixed-`q` ph-pair helper;
- intraflavor / intervalley / interspin / inter-spin-valley grouping;
- legality check for the conduction-only fully spin-valley polarized shortcut.

Implemented in the RLG/hBN system adapter:

- extraction of per-k HF orbitals and energies from `RLGhBNHartreeFockState` with the same flavor-block occupation ordering used by the HF density builder;
- q=0 particle-hole pair construction with particle and hole constrained to the same mBZ grid point;
- on-demand `V_hf(a,b,c,d)` backed by layer-resolved form factors and the full transfer-momentum Coulomb tensor stored in `RLGhBNLayerOverlapBlockSet`;
- dense q=0 TDHF matrix construction for small smoke tests and guarded checkpoint pilots;
- loading historical HF archives formerly written by the retired `run_rlg_hbn_paper_hf` workflow through `load_rlg_hbn_tdhf_run_from_archive(...)`, using cached projected basis / layer-overlap blocks rather than rerunning HF, and rejecting archives marked with the diagnostic `MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1` convention;
- command-surface access via `python scripts/mean_field_tools.py run_rlg_hbn_tdhf_q0 --hf-archive ...`, with login-node guard for the actual dense TDHF solve and `--dry-run` for configuration validation;
- vectorized q=0 dense assembly via `build_rlg_hbn_tdhf_q0_matrices_from_pairs(..., assembly="vectorized")`, grouping ph pairs by k and using NumPy/BLAS compiled kernels for layer form-factor contractions instead of calling `V_hf` element-by-element in Python;
- q=0 runner dense-memory guard (`--max-pairs`, `--max-dense-memory-gb`) and shortcut guard so the fully polarized simplification is not applied to mixed `--channel all` blocks;
- local lightweight regression coverage for fixed-q pair construction, dense q=0 smoke assembly, vectorized-vs-generic assembly parity (including multi-k synthetic blocks), direct HF-basis form-factor contraction against a manual expression, distinct Umklapp/full-Q kernel contributions, momentum conservation, all-channel shortcut blocking, and q=0 Fock-diagnostic env/archive guard;
- finite-q support introspection via `rlg_hbn_tdhf_finite_q_mode_support(...)`, which states that the canonical bridge only supplies parity-checked orbitals and that RLG/hBN `V_hf`, finite-q wrapping, direct/B terms, and q/-q pair-sector policy remain system-layer responsibilities.

Not yet implemented:

- full finite-q RLG/hBN direct/B/intraflavor TDHF assembly: the existing legacy system path covers only the conduction-only fully spin-valley-polarized flavor-flip exchange shortcut, and no canonical bridge should fabricate the missing `V_hf`/q/-q pair-sector policy;
- a finite-difference Fock/Hartree derivative test tying the stored-projector HF Hamiltonian response directly to `V_hf`;
- iterative block/matvec eigensolver for large 12x12 R5G/hBN sectors beyond dense channel pilots;
- MA-TBG system adapter and Goldstone-counting workflow;
- Slurm-scale Checkpoint A/B/C reproductions.
