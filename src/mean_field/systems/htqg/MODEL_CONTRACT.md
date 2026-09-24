# HTQG 4×1 full-microscopic inhomogeneous HF v2 model contract

Status: **LOCAL IMPLEMENTATION-COVARIANCE GATES PASS; GOVERNING-PLAN READINESS INCOMPLETE — Batch B NO-GO**

This contract governs the first small-system implementation only. It separates
verified algebra and basis conventions from physical choices that have not yet
been authorized. `UNRESOLVED` means the code must refuse a physical run; it is
not permission to inherit a projected-HF or frozen-wall convention silently.

## Authority boundary

The following sources establish narrow conventions, not a complete 4×1
self-consistent model. The acceptance source of truth is
`plan/HTQG_realspace_HF_small_system_plan_v2.md`; the local A0–A8 gate names
below are a narrower implementation-covariance decomposition and must not be
identified with that plan's T0–T8 without an explicit crosswalk:

- `src/mean_field/systems/htqg/hamiltonian.py`: primitive microscopic H0 and
  `H_K′(k)=H_K(-k)*`.
- `src/mean_field/core/hf/density.py`: repository stored density orientation.
- `results/HTQG_Fujimoto2025_hf/active8_hf_eps5_nu_m3_reduced_v1/source_capsule_v14_controlplane_manifest_key_fix/{configs/production.json,source_snapshot/src/mean_field/{core/hf/coulomb.py,core/hf/interaction.py,systems/htqg/hf.py}}`:
  source-manifest-bound reciprocal-space HF convention used by completed
  shell-6/mesh-18 runs. It authorizes the interaction/reference convention
  port, not the prior projected active-window approximation itself.
- `results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/source_capsule_v12_full_parent_wall_construction_utilization_qualifier_mesh18_regular6430/FORMULA_CONTRACT.json`:
  qualified `Nx=18` frozen-endpoint wall construction only.
- `results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/runtime_publication_v12_full_parent_wall_construction_utilization_qualifier_mesh18_regular6430/wall_construction_utilization_qualifier_j0_v12/summary.json`:
  construction/performance qualification only; not self-consistent wall HF.
- `results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/source_capsule_v13r3_full_parent_wall_spectrum_solver_qualification_mesh18_regular6430/inputs/active2/branches/{ABA5,ABG1}/state.npz`:
  historical projected active2 interaction metadata only.
- `results/HTQG_Fujimoto2025_hf/active2_aba_abg_boundary_eps5_nu_m3_v1/source_capsule_v13r3_full_parent_wall_spectrum_solver_qualification_mesh18_regular6430/AUTHORITY.md`:
  explicitly forbids promotion to a self-consistent full-parent wall solution.

## Resolved microscopic geometry

- Real-space supercell: `IntegerSupercell(4,0,0,1)`, area ratio 4.
- Primitive finite mesh: `(8,4)`.
- Reduced supercell mesh: `(2,4)`.
- Exact fold map:
  `(r1,r2,m) -> (r1+2m mod 8,r2)`, `m=0,1,2,3`.
- Plane-wave shell for the first requested calculation: shell 1, `N_G=7`.
- Per-reduced-K basis order:
  `(fold,g,layer,sublattice,valley,spin)`.
- Dimension: `4×7×4×2×2×2 = 896`.
- Valley order: physical `(K,K′)=(+1,-1)`.
- K′ construction: only `H_K′(k)=H_K(-k)*`; no additional reciprocal-label
  reversal.
- Common-denominator plane-wave labels:
  `n1=m+eta*4*g1`, `n2=eta*4*g2`.
  They mean
  `Delta k=(n1/4)b_M1+(n2/4)b_M2`.
- Reduced momenta are
  `K=(r1/8)b_M1+(r2/4)b_M2`.
- A density-transfer label `(q1,q2)` means
  `Q=(q1/8)b_M1+(q2/4)b_M2`.
- Canonical new-module density:
  `P_ab=<c_b^dagger c_a>` with K-block shape `(nk,nb,nb)`.
- Repository stored density orientation:
  `Pstored[a,b,k]=Pket[k,b,a]`; conversion must be explicit.
- Framework Hamiltonian orientation is different:
  `Hframework[a,b,k]=Hket[k,a,b]`; only the K axis moves and the matrix is not
  transposed. Energies use the established band-major shape `(nb,nk)`.
- The currently validated occupation path is the zero-temperature reusable
  `core/hf/occupations.py` global canonical solver over all reduced K points
  using one chemical potential. An unresolved Fermi degeneracy is rejected
  rather than filled independently per K. This does **not** implement the
  governing plan's Batch-B `kBT_ev=5e-4` fractional Fermi occupation.
- Fold-sector off-diagonal density matrices are permitted.

## Resolved H0 assembly

The varying layer-3/layer-4 interface is assembled channel by channel as

`H34 = sum_c (M_c A_c + A_c^dagger M_c^dagger)`.

`A_c` is the no-displacement one-way layer-4 to layer-3 parent edge. The
multiplier is applied on the left. ABA is `s_ABG=0`; ABG is `s_ABG=1`.
Both uniform endpoints must replay the exact primitive `build_hamiltonian`
blocks at the same finite shell and mesh. Agreement with a projected active-band
HF model is not this test.

## Resolved interaction and reference convention

The implementation uses intravalley plane-wave density vertices `Gamma_Q`
that preserve layer, sublattice, valley, and spin, matching the prior
reciprocal-space HTQG HF transfer policy (no intervalley Coulomb form factor
and no additional microscopic orbital-position phase). It requires:

- exact `Gamma_0=I`;
- `Gamma_-Q=Gamma_Q^dagger`;
- one fixed reduced-torus K displacement for every entry under a transfer
  label, with the exact physical momentum difference rechecked from the HTQG
  fold/plane-wave labels (finite plane-wave projection may make a vertex
  partial in orbital coverage);
- the source-bound screened-Coulomb weight for every physical Q;
- a bounded literal four-index Wick oracle for tiny tests only.

The selected interaction is the symmetric double-gate kernel

`V(Q) = 2*pi*(e^2/4*pi*epsilon0)/(epsilon_r*|Q|) * tanh(|Q|*d_sc)`,

with `Q` in `nm^-1`, `V` in `eV nm^2`, `epsilon_r=5`, `d_sc=25 nm`, and the
finite exact-zero limit
`V(0)=2*pi*(e^2/4*pi*epsilon0)*d_sc/epsilon_r`. For the 4×1 cell,

`w_Q = V(Q)/(A_supercell*N_K)`,

where `A_supercell=4*A_primitive` and `N_K=2*4=8`. This equals
`V(Q)/(A_primitive*N_k_primitive)` because `N_k_primitive=8*4=32`.
There is no extra transfer-side factor of one half.

The completed reciprocal-space runs establish this convention inside a
projected active window. Extending it to the complete shell-1 microscopic
cutoff is an explicit convention-continuity choice for this task—not a claim
that the old active-window result proves cutoff convergence or removes
remote-sea exchange. At the selected finite cutoff, the historical per-k
reference semantics are extended by convention to the projector onto the
lowest half of the same shell-1 primitive H0 at each primitive `(8,4)` k. For a
uniform 4×1 replay, its authorized representation is the exact fold of those
32 primitive projectors. Slurm T8 job `503838` demonstrated that independently
taking the lowest 448 states of each 896-dimensional reduced-K block is **not**
the same reference (maximum matrix-element discrepancy about `0.322`), so that
candidate is rejected rather than hidden by numerical tolerance. For the first
inhomogeneous-wall trial, the user now authorizes **Scheme A**: construct the wall
`H0`, occupy its globally lowest 3584 states under one chemical potential, and
use that projector as the fixed Hartree reference. This is not the rejected
lowest-448-at-each-reduced-K construction: Scheme-A reduced-K ranks may
redistribute. It is also not claimed to be the exact lift of the historical
primitive reference. It is a new explicit wall-background convention that must
pass neutral-gap, projector, total-trace, endpoint-comparison, and
width-sensitivity gates before physical promotion. The strict Batch-A
constructor still fails closed unless the resulting projector and provenance
are supplied explicitly.

- `Q_H=P-P_ref` for every Hartree transfer, including Q=0;
- `Q_F=P` (absolute Fock), including Q=0;
- the exact screened Q=0 weight is not set to zero in either contraction.

Thus the neutral background is removed by `P-P_ref`, but the doped uniform
capacitor term remains. Hartree Q=0 removal and Fock Q=0 removal are not
conflated.

For supplied reference-adjusted densities, the action is

`H_H = sum_Q w_Q Gamma_Q Tr(Gamma_-Q (P-P_ref))`,

`H_F = -sum_Q w_Q Gamma_Q P Gamma_Q^dagger`.

Primitive filling `nu` changes the global occupied count by
`nu*N_k_primitive`. At shell 1, `nb=896`, `N_K=8`, and the neutral reference
contains `8*448=3584` states. Therefore `nu=-3` means
`N_occ,total=3584-3*32=3488`. The occupation solver still uses one global
chemical potential; `436` per reduced K is only the average count, not an
independently imposed K-sector filling. The governing workflow document already
uses this single global chemical potential. The rank-109-per-primitive-k
neutral/reference construction must not be reinterpreted as a separate
occupation constraint. T8 therefore requires the correct total count and exact
primitive/supercell occupation covariance; per-k ranks are recorded diagnostics
and are not constrained to 109.

The scalar is the extensive finite-torus quantity

`E = Tr(H0 P) + 1/2 Tr(H_H Q_H) + 1/2 Tr(H_F Q_F)`.

Any reported per-reduced-K, per-supercell, per-primitive-cell, or per-area
energy must state and apply that additional divisor explicitly.

## UNRESOLVED physical inputs

The reciprocal-space HF audit, pinned authority hashes checked by
`build_batch_a_screened_coulomb_hf_inputs`, and explicit
convention-continuity decision resolve the interaction normalization, transfer
policy, zero mode, historical primitive neutral reference, and `nu=-3` count
above. They do not prove that any selected inhomogeneous reference is equivalent
to the historical primitive reference. Full-space absolute Fock remains a
finite-cutoff model whose UV reliability must be studied separately; it is not
silently presented as a converged continuum sea. The uniform T8 execution gate
(item 3) is now resolved by source-bound job `503888` and its global-mu post-run
reclassification. The wall profile family and a Scheme-A reference trial are
now authorized and implemented. Source-hash-bound shell-1 Slurm jobs `505788`,
`505789`, `505793`, `505810`, and `505815` completed the numerical gates below.
They expose material width and endpoint-reference sensitivity, so selection of
a physical width and item 4 still block physical Batch B; they do not undo the
uniform Batch-A implementation-covariance result:

1. **NUMERICALLY QUALIFIED FAMILY; PHYSICAL WIDTH UNSELECTED — wall profile:**
   the user authorizes an `N=4` periodic smoothstep double wall.
   `wall_width_cells` is an explicit parameter satisfying
   `0 < wall_width_cells < N/2=2`; it is never inferred from the old qualified
   `Nx=18`, `xi=1.5` wall. At `w_normal/L_M=(0.35,0.50,0.75)`, the planned
   `M_u=256,512,1024` sequence converged monotonically. The `512 -> 1024` maximum
   H0 changes were `(8.00e-8,1.76e-8,1.12e-8) eV`, and the corresponding
   relative projector differences were `(7.26e-8,1.50e-8,8.78e-9)`. The
   converged neutral gaps were `(5.9280e-4,6.0982e-6,2.1143e-4) eV`. Thus
   Fourier sampling is not the source of the central near-closure, but the
   width choice is physically consequential and remains unselected.
2. **IMPLEMENTED AND NUMERICALLY QUALIFIED TRIAL; NOT PHYSICALLY PROMOTED —
   inhomogeneous reference:** Scheme A uses the projector onto the globally
   lowest 3584 states of the inhomogeneous wall `H0`, with one chemical
   potential and variable reduced-K ranks. It fails closed on an ambiguous
   neutral Fermi boundary. The uniform endpoint covariance test agrees with the
   exact fold of the *same global-neutral primitive convention* to below
   `3.0e-14` in maximum matrix element for both ABA and ABG. It is nevertheless
   a new wall-background convention, not a hidden lift of the historical
   rank-109-per-primitive-k reference and not the rejected
   lowest-448-per-reduced-K construction. At the ABG endpoint it happens to
   equal the historical folded reference to roundoff; at ABA it differs by
   `0.378` in maximum projector matrix element and `0.0528` in relative
   Frobenius norm, with reduced-K ranks
   `[450,446,446,448,452,446,446,450]`. Through the selected Hartree kernel that
   ABA difference produces an induced one-body Hartree eigenvalue range
   `[-2.185,+1.472] meV` (maximum matrix element `0.814 meV`), while the ABG
   shift is below `1.1e-12 eV`. These endpoint differences are physical
   consequences of the convention and must be reported rather than aligned
   away.
3. **RESOLVED — uniform full-HF replay authority:** job `503888` executed the
   primitive `(8,4)` versus supercell `(2,4)` comparison using the exact folded
   historical reference, physical vertices, global filling 3488, normalization,
   and finite cutoff. Its only failing assertion was the unauthorized `109/k`
   discriminator; the source-bound post-run reclassification records T8 PASS.
4. **SCF branch meaning:** independent inhomogeneous SCF branch versus an action
   initialized or constrained by frozen ABA/ABG endpoint states.
5. **Finite-temperature canonical map and free energy:** the generic HF layer
   now provides an offset-stable, explicitly k-weighted one-global-mu Fermi
   map, fractional ket density, fermionic entropy/density diagnostics, and
   `F=E-kBT*S` bookkeeping. Focused `test001` tests pass, including asymmetric
   weights, nonzero mu, weight-rescaling, complex ket orientation, energy-shift
   invariance, and endpoint cases. The HTQG microscopic adapter now has an
   explicit finite-temperature mode: it uses the same global Fermi map for
   initialization and every SCF update, labels the interaction scalar as
   internal energy, and separately records the Helmholtz objective. The adapter
   currently accepts only equal positive k weights because its interaction
   vertices embed uniform quadrature; a common weight rescales particle number,
   internal energy, and entropy consistently without changing the HF map.
   `kBT_ev=5e-4` is therefore an available explicit input. The adapter also
   supports explicit Helmholtz-change and consecutive-iteration criteria, but
   their physical threshold/count and an HTQG run remain without source-bound
   authority.
6. **Seed and ansatz contract:** `seed` is currently ignored; the required
   Aufbau, branch-informed, and perturbed legal densities and the
   spin/valley-coherence ansatz are not source-bound.
7. **Fail-closed final acceptance:** the generic engine rejects a false
   mixed-step convergence when the recomputed final raw fixed-point residual
   fails and exposes protected iteration-convergence and final-acceptance
   hooks. In explicit finite-temperature mode, the HTQG adapter requires a
   caller-selected number of consecutive iterations satisfying both the raw
   density threshold and an absolute Helmholtz free-energy-change threshold.
   It then rechecks the actual terminal state's Helmholtz change and adds
   particle-number, density-spectrum/Hermiticity, commutator, and final
   `rho=f(H-mu)` gates. Internal energy, entropy, Helmholtz objective, latest
   objective change, and the consecutive-pass streak remain separately
   labelled. These mechanisms are locally tested but do not select physically
   authoritative tolerances or establish an unrestricted wall-SCF result.
8. **Plan-level unrestricted checks and observables:** the uniform folded fixed
   point has not been released into an unrestricted supercell SCF, and the
   plan's V=0, spin-rotation, restricted-versus-full, and real-space wall-profile
   outputs remain unimplemented or unexecuted.

The first Batch-A physical configuration selects `theta=2.25 deg`, `w=0.110`
eV, `kappa=0.6`, `lambda_MDT=-0.23 nm`, Dirac rotations enabled,
`epsilon_r=5`, screening distance `25 nm`, and primitive `nu=-3`; each remains
an explicit source-bound input rather than a hidden library default.

## Local implementation-covariance A0–A8 gates

The following are fail-closed local gates. They were historically labelled
T0–T8 in the test file, but they are not a complete implementation of the
governing plan's separately defined T0–T8. Passing them supports only the
finite-cutoff implementation-covariance scope stated here.

- **A0 — contract/configuration:** exact 4×1 meshes and shell; every physical
  choice explicit and provenance-bearing; unresolved choices rejected; global
  Fermi tie rejected.
- **A1 — folding/basis/density ABI:** fold bijection and inverse; basis
  index/unindex; K/K′ physical labels; exact ket/stored density roundtrip.
- **A2 — profile/Fourier convention:** direct Fourier oracle, physical-label
  differences, both valleys, and a complex nonuniform profile.
- **A3 — microscopic H0:** matrix-level Hermiticity and exact uniform ABA/ABG
  replay, including realistic nonzero MDT and Dirac rotations; explicit
  discrimination of `M A + A† M†` from wrong ordering.
- **A4 — density vertices:** physical Q-label construction, exact integer
  labels, Gamma0 identity, reverse closure, fixed K mapping, finite weights,
  and zero-mode policy consistency.
- **A5 — optimized interaction oracle:** sparse Hartree and Fock actions and
  scalar agree with a literal four-index Wick oracle on a tiny complex,
  nonzero-Q/cross-K case.
- **A6 — limiting cases:** one-particle self-interaction cancellation and a
  two-spin Hubbard/density-density limit with known energy.
- **A7 — variational derivative:** finite-difference derivative of the complete
  scalar equals `H0+H_H+H_F`, including complex nonzero-Q/cross-K vertices and
  the selected references.
- **A8 — full uniform primitive/supercell replay:** H0, interaction action,
  energy, global occupation, and a uniform full-HF fixed point agree after
  exact fold/unfold mapping for the same finite-cutoff physical model.

Current lightweight tests establish the algebraic parts of local A0–A7, including a
complex cross-K Wick/derivative oracle and a generic-engine orientation/global-
occupation adapter check. Sealed Slurm job `503820` passed the independent
nonuniform shell-1 T3 direct-DFT ordered-assembly oracle. T8 job `503836` failed
closed before interaction construction on an undersampled profile fixture;
its 32-sample successor `503838` reached and rejected the representation-
dependent reduced-sector neutral reference before HF contraction. Thus no T8
interaction, occupation, or fixed-point verdict exists. Successor job
`503879` passed the source manifest, H0 replay, folded-reference validation,
and physical vertex construction, then failed before contraction because the
vertex-equivalence validator used nonexistent `IntegerSupercell.determinant`
instead of `area_ratio`. Job `503888` ran the repaired path through physical
vertex, Hartree/Fock action, scalar energy, global occupation, chemical-
potential/Fermi-gap, and folded fixed-point equivalence. Those primitive/
supercell implementation-covariance checks passed. With the document-prescribed
single global chemical potential, the primitive-k occupied ranks ranged from
106 to 110 and summed to the correct global target 3488. That redistribution is
valid global-ensemble behavior. The former demand that every k retain rank 109
was an erroneous extra discriminator introduced during implementation review,
not a requirement of the governing document; it must not be used to reject the
job `503888` physics path or to force equal per-k filling. The source-bound post-run correction at
`postrun_job_503888_global_mu_reclassification_v1` reclassifies the already
executed evidence as local A8 PASS without a duplicate heavy HF run. This does
not close the governing plan's unrestricted-release and dynamical T7/T8 gates.

## Batch B gate and reporting

Batch B may run only after a source-bound readiness receipt crosswalks and
passes every governing-plan requirement, including the unresolved items 1–8
above. The former `batch_a_aggregate_v1_global_mu_corrected` is valid only as a
local finite-cutoff implementation-covariance aggregate; its broader
`BATCH_A_T0_T8_PASS` label is superseded by the scope correction receipt and
must not authorize Batch B. The first allowed size remains exactly shell 1,
reduced mesh `(2,4)`, and a four-cell smooth ABA|ABG profile. No shell-6
auto-submission is allowed.

A successful small pilot would establish implementation behavior at that
cutoff. It would not by itself establish model-approximation reliability,
isolated-wall physics, four net chiral channels, topology, complete wall
spectrum, complete degeneracy, or full-space convergence.
