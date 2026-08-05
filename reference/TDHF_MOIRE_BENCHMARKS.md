# Moire TDHF/RPA cross-validation references

Date: 2026-08-01

This index separates equation/tutorial sources from numerical reproduction
targets. PDFs are immutable local evidence; rendered or digitized figures are
not author raw numerical arrays.

## Tutorial and convention source

### Kwan et al., *Mean-field Modelling of Moire Materials: A User's Guide with Selected Applications to Twisted Bilayer Graphene*

- Local PDF: `reference/2511.21683v1.pdf`
- arXiv: <https://arxiv.org/abs/2511.21683>
- SHA256: `2354caaa3c5fddbdc7c5caaacbc9dcfa94c45dfc855d930b10372daabf6fd8a6`
- Relevant material:
  - Sec. 3.5, Eqs. (64), (70)-(90): TDHF equation, Thouless/TDVP
    derivation, quadratic static Hessian, RPA equivalence, and MA-TBG finite-q
    equation.
  - Eq. (81): Hermitian static quadratic form `[[A,B],[B*,A*]]`.
  - Eq. (84): non-Hermitian Liouvillian `[[A,B],[-B*,-A*]]`.
  - Sec. 6.1-6.4 and Fig. 8: Goldstone counting and numerical MA-TBG TDHF.
- Companion HF code already present at `reference/TBG-HF`, commit
  `0d2a3d742aa901fa45ce46690c1385887165f58c`. The public release does not
  contain its TDHF routines, so it is an HF/source-convention reference rather
  than an independent TDHF implementation.

## Primary numerical TDHF targets

### Alavirad and Sau, *Ferromagnetism and its stability from the one-magnon spectrum in twisted bilayer graphene*

- Local PDF: `reference/1907.13633v1.pdf`
- arXiv: <https://arxiv.org/abs/1907.13633v1>
- Journal: Phys. Rev. B 102, 235123 (2020)
- SHA256: `1ed85a7cec13d96428856768c88a6f6b4dacf847843acc9d9278f338beb612fd`
- Why it is unusually useful:
- the single-spin-flip Hilbert space is diagonalized exactly at fixed q,
  independently of a TDHF implementation;
- for a saturated SU(2) ferromagnetic Slater state, this is the exact
  one-magnon oracle for the interspin TDHF sector;
- it exposes the q=0 spin Goldstone mode, quadratic stiffness, finite-q
  negative modes, and the transition from positive to negative stiffness.
- Scope limitation:
- the paper uses a screened Hubbard-like interaction and saturated
  spin/valley-polarized states near the chiral flat-band limit, not the K-IVC
  source used by the Kwan benchmark;
- it validates interspin algebra and stability diagnostics, not the complete
  intraflavor K-IVC spectrum.

### Khalaf et al., *Soft modes in magic angle twisted bilayer graphene*

- Local PDF: `reference/2009.14827v2.pdf`
- arXiv: <https://arxiv.org/abs/2009.14827v2>
- SHA256: `991c348090db78d25d2d06e69858f9df58ff35c411bc5a199478b70326280730`
- Why it is the primary benchmark:
  - direct TDHF calculation plus nonlinear sigma-model interpretation;
  - Appendix A gives the microscopic TDHF contractions;
  - Fig. 3 supplies unusually strong K-IVC numerical checkpoints, but not a
    complete executable parent/subtraction/cutoff specification.
- Fig. 3 parameters:
  - `nu=0`: `(Nx,Ny)=(18,18)`;
  - `nu=-2`: `(Nx,Ny)=(18,12)` (an `18x18` run would be a separate square-grid control, not the published Fig. 3 mesh);
  - `theta=1.08 deg`, `kappa=w0/w1=0.75`;
  - `epsilon_r=12.5`, dual-gate distance `d_s=40 nm`;
  - two remote valence and two remote conduction bands retained per
    spin/valley in the numerical projection.
- Paper checkpoints:
  - mean-field gaps approximately `25 meV` (`nu=0`) and `14 meV` (`nu=-2`);
  - `nu=0`: 16 soft modes, fourfold degeneracy, four Goldstone modes;
  - `nu=-2`: 12 modes, including two quadratic and one linear Goldstone mode.
- Framework acceptance gates, not reported Fig. 3 numbers:
  - independent q/-q RPA pairing;
  - scalar static-Hessian positivity for the stable source;
  - source-bound Ward certificates and exact-boundary sewing.

### Kumar, Xie, and MacDonald, *Lattice Collective Modes from a Continuum Model of Magic-Angle Twisted Bilayer Graphene*

- Local PDF: `reference/2010.05946v2.pdf`
- arXiv: <https://arxiv.org/abs/2010.05946v2>
- Journal: Phys. Rev. B 104, 035119 (2021)
- SHA256: `4839d31fd922923875e800a5e5208a94aab018a6714c6b4ee148decff2d4e9a4`
- Independent cross-check:
  - Eqs. (5)-(8) use the same signed-q TDHF structure and identify the
    Hermitian matrix as the Hessian of the SCHF energy;
  - Appendix A derives TDHFA from linear response;
  - Appendix B derives the static Hessian;
  - Appendix C derives the q=0 spin-rotation Goldstone vector.
- Main numerical target:
  - `nu=-3`, `theta=1.1 deg`, `epsilon_r^{-1}=0.06`;
  - low-energy spin-, valley-, and intraflavor collective modes below the HF
    particle-hole gap;
  - exactly gapless q=0 spin wave and seven low-energy collective branches;
  - Fig. 4 provides dispersion-scale and degeneracy checks.

### Vituri et al., *Incommensurate inter-valley coherent states in ABC graphene: collective modes and superconductivity*

- Local PDF: `reference/2408.10309v1.pdf`
- arXiv: <https://arxiv.org/abs/2408.10309v1>
- Journal: Phys. Rev. B 111, 075103 (2025)
- SHA256: `ec761a2b494a8e5983ff3fb6cfb842e114526cc0ba8b3e7cdc7c128f5d204bc8`
- Cross-system finite-q instability benchmark:
- Supplement Sec. III, Eqs. (12)-(24), gives the generalized-RPA/TDHF
  susceptibility and the Hermitian H / non-Hermitian `Sigma_z H` construction;
- Eqs. (22)-(23) give explicit finite-q intervalley excitation and
  de-excitation contractions;
- Figs. 3 and 8 connect C3-related finite-q susceptibility maxima, mode
  softening, and the onset of imaginary frequencies;
- the intravalley magnon supplies a separate small spin-stiffness checkpoint.
- Scope limitation: this is dual-gated ABC trilayer graphene with a single
  active band, not MA-TBG. Use it to validate generic finite-q IVC
  susceptibility/instability logic, not numerical equality to the TBG spectrum.

### Wang et al., *Putting a new spin on the incommensurate Kekule spiral: from spin-valley locking and collective modes to fermiology and implications for superconductivity*

- Local PDF: `reference/2509.12320v1.pdf`
- arXiv: <https://arxiv.org/abs/2509.12320v1>
- SHA256: `1e648ad06731c46815d3216ff2da9f3dbedd65dd61ab1c237cd944de06a01d94`
- Strong signed-q structural benchmark:
- Appendix A, Eqs. (A2)-(A5), defines A, B, the Hermitian stability matrix,
  and the Liouvillian;
- Eqs. (A6)-(A8) explicitly distinguish TRIM from non-TRIM q and combine
  independent q and -q sectors;
- Eq. (A8) gives the outer signed-q block and the paper's norm/sign-based
  spectral assignment;
- the numerical IKS spectra test Goldstone counting and linear versus
  quadratic modes on 10x10 and 20x20 meshes.
- Priority interpretation: this is P1 for our non-TRIM signed-q algebra,
  because it directly targets the convention that was hardest to validate. It
  remains P2 as a full numerical reproduction target because that requires a
  strained IKS HF source, perturbations, and its own reference conventions.

### Lin et al., *Collective excitations of the Chern-insulator states in commensurate double moire superlattices of twisted bilayer graphene on hexagonal boron nitride*

- Local PDF: `reference/2301.05359v1.pdf`
- arXiv: <https://arxiv.org/abs/2301.05359v1>
- Journal: Phys. Rev. B 107, 195434 (2023)
- SHA256: `fce1462ff7b4a30edf2f35f7cd4ceef937ce07569a8a297164a5c86cfad0753d`
- Later-stage target: comparison of active-band and full-HF TDHF in TBG/hBN,
  including zero-gap spin waves, valley-wave gaps above about `2.5 meV`, and
  intraflavor excitons reaching about `20 meV`. This should be attempted only
  after the pristine MA-TBG Goldstone/counting benchmarks pass.

## Ordered benchmark ladder

1. **Equation/algebra gate**
   - reproduce tutorial Eqs. (81), (84), and (90) from one scalar HF
     functional;
   - validate `B(q)=B(-q)^T`, `spec L(q)=-conj(spec L(-q))`, raw residuals,
     eta norms, and static Hessian inertia.
2. **Tutorial Fig. 8(a), reduced faithful MA-TBG benchmark**
   - K-IVC at `nu=0`, no strain, `10x10` mesh;
   - `theta=1.05 deg`, `wAA=80 meV`, `wAB=110 meV`, `epsilon_r=10`;
   - require four gapless modes and twelve gapped low-energy modes, all
     fourfold degenerate, before comparing detailed energies.
3. **Exact interspin one-magnon oracle — implemented**
   - `systems/hubbard_1d/tdhf_alavirad_sau.py` independently projects the
     real-space Eq. (14) bitstring Hamiltonian into the Eq. (13) `psi_q` basis
     and constructs the interspin TDHF spin-flip action;
   - entrywise matrices, q=0 Ward/Goldstone, the first negative mode, and the
     large-U stiffness limit are gated in the test suite.
4. **Khalaf Fig. 3 production benchmark — authority closure required first**
   - fresh K-IVC sources at `nu=0` on `18x18` and `nu=-2` on `18x12`;
   - do not submit production work until absolute `w1`, `v_F`, parent/transfer
     cutoffs, projected-out remote-valence self-energy, and double-counting
     subtraction are bound to an executable source;
   - compare mode counts, degeneracies, Goldstone dispersion type, HF gaps,
     and only then the plotted dispersion.
5. **Independent Kumar-Xie-MacDonald benchmark**
   - reproduce q=0 spin Ward/Goldstone and the seven low-energy branches at
     `nu=-3` under that paper's reference and interaction conventions.
6. **Wang non-TRIM signed-q benchmark**
   - the typed core now validates independent A(q), A(-q), B(q), B(-q), the
     Appendix-A outer-block construction, eta-degenerate normalization, and
     norm/sign spectral assignment;
   - a full numerical strained-IKS target remains separate and requires its own
     HF source, perturbations, and reference conventions.
7. **Vituri ABC finite-q instability benchmark**
   - reproduce susceptibility enhancement at three C3-related q vectors,
     real-mode softening, and the onset of imaginary frequency using one shared
     HF/TDHF interaction source.
8. **TBG/hBN benchmark**
   - compare active-band and full-HF collective excitations after pristine TBG
     passes.

## Integrity boundary

- Do not tune scale/offset, select modes by proximity to a raster, copy symmetry
  sectors, or average assembled A/B matrices.
- Each benchmark must use the same executable scalar/provider for HF energy,
  SCF Hamiltonian, q=0 response, and finite-q Hessian.
- A Goldstone pass alone does not validate finite-q stiffness; mode count,
  signed q/-q pairing, and at least one independent scalar-curvature gate are
  required.
- Paper figures are visual evidence only unless author raw arrays are obtained.
