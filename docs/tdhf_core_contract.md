# TDHF core interface contract

Reference papers and planning notes are local/internal inputs; this public note records only the durable TDHF/RPA core API contract.

## Scope

The reusable TDHF/RPA implementation lives in `src/mean_field/core/hf/tdhf.py`. It is system agnostic and adds only the layer needed after a converged HF calculation:

1. fixed collective-momentum particle-hole basis construction;
2. dense debug assembly of `A`, `B`, and q=0 `L = [[A, B], [-B*, -A*]]`; system adapters must use the signed-q partner form `[[A(q),B(q)],[-B(-q)*,-A(-q)*]]` at nonzero q;
3. ordinary non-Hermitian diagonalization of `L`;
4. eta-metric normalization with `eta = diag(+1, -1)`;
5. flavor-channel grouping helpers.

System-specific gauge choices, form factors, layer Coulomb kernels, screening schemes, Umklapp sums, saved HF-state loading, and paper runners remain in `src/mean_field/systems/<system>/`. The first RLG/hBN bridge lives in `src/mean_field/systems/RnG_hBN/tdhf.py`.

## Typed signed-q public API

New system adapters must enter through `mean_field.api.run_tdhf(...)` or
`run_tdhf_typed(...)` and implement
`TDHFSectorProviderProtocol.build_tdhf_sector(config, **kwargs)`.  The adapter
returns one of two deliberately distinct core types:

- `TDHFGenericSignedQSector` for a non-TRIM orbit with distinct canonical
  `q` and `-q`;
- `TDHFSelfConjugateQSector` for literal q=0 or an explicitly sewn
  self-conjugate/Nyquist momentum.

The typed contract version is `typed_signed_q_v1`.  System adapters still own
HF-state loading, form factors, interaction contractions, momentum wrapping,
periodic gauge, and pair sewing.  The core owns the following common steps:

1. A/B block shape and source/pair-provenance validation;
2. static Hessian and Liouvillian construction;
3. Hermiticity, pseudo-Hermiticity, Nambu-sewing, and signed spectral gates;
4. raw eigensystem retention and Wang norm/sign assignment;
5. independent static, dynamic, and zero-mode-origin statuses;
6. source/matrix-bound Ward certificate validation.

### Generic non-TRIM orbit

The adapter supplies independent blocks on explicit pair spaces:

```text
A_plus          : P_q    -> P_q
B_plus_minus    : P_-q^* -> P_q
A_minus         : P_-q   -> P_-q
B_minus_plus    : P_q^*  -> P_-q
```

After adapter-declared pair alignment,

```text
A(q) = A(q)^dagger
A(-q) = A(-q)^dagger
B(q) = B(-q)^T

H(q) = [[ A(q),       B(q)      ],
        [ B(-q)^*, A(-q)^* ]]
L(q) = eta H(q),  eta = diag(+I_q, -I_-q)
```

No same-sector `B(q)=B(q)^T` check is made.  An explicit anti-linear Nambu
sewing `w_- = S w_+^*` must bind the source and exact ordered pair inventories.
The core checks both directions of
`L(-q) S + S L(q)^* = 0` and the metric anti-covariance relation.

For a real, non-null metric mode, Wang Appendix A assigns

```text
metric_sign    = sign(w^dagger eta w)
assigned_q     = metric_sign * q
assigned_energy = metric_sign * Re(eigenvalue)
```

Negative assigned energies remain visible.  Complex, zero-eigenvalue, and
null-metric modes are retained raw and are not assigned by this rule.  The
null-metric classification tolerance and the selected-basis eta-Gram
orthogonality tolerance are separate controls; the latter must not be inferred
from the former in a large near-degenerate sector.

### Self-conjugate momentum

A self-conjugate adapter must provide a canonical sewn pair basis and canonical
`A=A^dagger`, `B=B^T`.  Raw `+M/-M` blocks may be attached as diagnostics, but
they are never averaged or promoted to a scalar Hessian.  Exact-M static
status remains `not_established` unless the adapter supplies an independently
certified canonical scalar Hessian.

### Status and Ward semantics

The new API does not collapse all conclusions into one precedence enum:

- static: `positive_definite`, `positive_semidefinite`, `indefinite`,
  `invalid`, or `not_established`;
- dynamic: `real`, `complex`, or `invalid`;
- zero origin: ordinary dynamic zero, uncertified static null, or
  `ward_static_null`.

A static null is not called Goldstone without a Ward certificate.  A passing
certificate is bound to source and interaction fingerprints, exact H/L bytes,
scalar-Hessian authority, stationarity, generator provenance, action residual,
and static-null overlap.  It cannot be reused for another sector.

For several broken generators, `core/hf/tdhf_goldstone.py` independently
certifies the full generator tangent subspace and the physical real
antisymmetric commutator matrix `rho_ab`.  Following the
Watanabe--Murayama theorem it reports type-A
`n_A = n_BG-rank(rho)` and type-B `n_B = rank(rho)/2`; therefore five
Ward-certified static null directions may represent only three dynamic
branches.  Mapping type-A/type-B to linear/quadratic dispersion requires the
system's regular spatial-gradient structure and remains adapter-local.  The
core never antisymmetrizes a supplied `rho`, drops dependent generators, or
infers Goldstone type from eigenvalue sorting.  A typed analysis accepts the
certificate only after rebinding source/interaction/sector and exact H/L,
generator-basis, and `rho` fingerprints and recomputing its action/null gates.

The first typed consumer is the diagnostic-only Kwan companion adapter in
`systems/tbg/zero_field/companion_tdhf.py`.  Its unit canary independently
assembles `K(q)` and `K(-q)` on a 2x3 fixture, exercises a true generic
`q=(0,1)` orbit and a separate exact-boundary `q=(1,0)` orbit, consumes the
explicit transition conjugation map, and requires matrix parity with the core
H blocks.  Exact-boundary aliases are never averaged: distinct raw kernels
must be byte-identical before one branch can be certified as the canonical
self-conjugate payload.  The saved N=10 Fig. 8(a) q=+/-1 triplet and singlet
matrices were replayed through commit `9faf9b8` in sealed Slurm job `205051`.
Both sectors retained 400 modes at each assigned sign, remained dynamically
real and statically positive definite, and matched saved raw/assigned spectra
at about `1e-15 eV`.  This is saved-matrix/API parity only: it does not rebuild
BM, HF, or the Eq. (90) contractions and does not upgrade the single-seed
K-IVC diagnostic to a global-ground-state claim.

The first independent exact oracle is
`systems/hubbard_1d/tdhf_alavirad_sau.py`.  It projects the literal periodic
one-dimensional Hubbard Hamiltonian of Alavirad--Sau Eq. (14) with explicit
fermionic bitstrings onto the Eq. (13) one-magnon basis
`|psi_q> = sum_p phi_qp c^dagger_{p+q,down} c_{p,up}|FM>`, independently
constructs the saturated-FM interspin TDHF action, and requires entrywise
matrix and spectrum equality.  Its gates cover the q=0 spin-flip Goldstone
mode, computed HF stationarity, negative finite-q mode, and large-U
`rho_s -> -2t^2/U`.  This is authority only for a half-filled saturated SU(2)
ferromagnet with `B=0`; it is not a full mixed-sector RPA, K-IVC, or general
moire soft-mode benchmark.

The older `TDHFMatrices`, `solve_tdhf_liouvillian`, and
`analyze_tdhf_signed_stability` APIs remain for compatibility and existing
q=0/toy workflows.  New generic-q system adapters and all new paper benchmarks
must use the typed API.

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
- historical command-surface access via the retired `run_rlg_hbn_tdhf_q0` devtool; tracked code now keeps only sidecar/shortcut compatibility helpers, while dense q=0 TDHF workflow code is archived under ignored `local_archive/`;
- vectorized q=0 dense assembly via `build_rlg_hbn_tdhf_q0_matrices_from_pairs(..., assembly="vectorized")`, grouping ph pairs by k and using NumPy/BLAS compiled kernels for layer form-factor contractions instead of calling `V_hf` element-by-element in Python;
- q=0 runner dense-memory guard (`--max-pairs`, `--max-dense-memory-gb`) and shortcut guard so the fully polarized simplification is not applied to mixed `--channel all` blocks;
- local lightweight regression coverage for fixed-q pair construction, dense q=0 smoke assembly, vectorized-vs-generic assembly parity (including multi-k synthetic blocks), direct HF-basis form-factor contraction against a manual expression, distinct Umklapp/full-Q kernel contributions, momentum conservation, all-channel shortcut blocking, and q=0 Fock-diagnostic env/archive guard;
- finite-q support introspection via `rlg_hbn_tdhf_finite_q_mode_support(...)`; RLG/hBN form factors, wrapping, quotient branches, and signed q/-q policy remain system-layer responsibilities;
- typed-provenance variational-v2 finite-q assembly via `build_rlg_hbn_tdhf_finite_q_quotient_context(...)`, `build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs(...)`, and the +q-only compatibility wrapper `build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs(...)`: ordinary wrapped legs use analytic periodic relabelling, fixed endpoints use three same-puncture-copy branches with tangent weight `1/3`, and the two Hessian legs are independently summed;
- independently returned +q/-q Liouvillians without post-hoc Hermitization/symmetrization, with legacy pair assembly still fail-closed for typed quotient archives;
- complete raw non-Hermitian diagnostics in `TDHFSpectrum`: raw eigenvalues, eta norms, and eigensolver residuals, in addition to selected positive-metric modes;
- actual 12x12 validation for intraflavor, intervalley, and interspin channels: q=0 strict HF-response parity, independent generic-q C3 anchors, signed fixed self sectors, q/-q particle-hole pairing, eta norms, solver residuals, interspin SU(2) Goldstone/Ward checks, and deterministic 26-orbit C3+inversion expansion to 144 points per channel.

Not yet implemented:

- a generic canonical-HF bridge for the RLG/hBN finite-q quotient (the validated API currently requires the system-specific typed archive/context);
- iterative block/matvec eigensolver for large 12x12 R5G/hBN sectors beyond dense channel pilots;
- MA-TBG system adapter and Goldstone-counting workflow;
- Slurm-scale Checkpoint A/B/C reproductions.
