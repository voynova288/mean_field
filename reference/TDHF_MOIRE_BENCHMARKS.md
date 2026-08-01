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

### Khalaf et al., *Soft modes in magic angle twisted bilayer graphene*

- Local PDF: `reference/2009.14827v2.pdf`
- arXiv: <https://arxiv.org/abs/2009.14827v2>
- SHA256: `991c348090db78d25d2d06e69858f9df58ff35c411bc5a199478b70326280730`
- Why it is the primary benchmark:
  - direct TDHF calculation plus nonlinear sigma-model interpretation;
  - Appendix A gives the microscopic TDHF contractions;
  - Fig. 3 supplies a fully stated K-IVC benchmark.
- Fig. 3 parameters:
  - `(Nx,Ny)=(18,18)`;
  - `nu=0` and `nu=-2`;
  - `theta=1.08 deg`, `kappa=w0/w1=0.75`;
  - `epsilon_r=12.5`, dual-gate distance `d_s=40 nm`;
  - two remote valence and two remote conduction bands retained per
    spin/valley in the numerical projection.
- Checkpoints:
  - mean-field gaps approximately `25 meV` (`nu=0`) and `14 meV` (`nu=-2`);
  - `nu=0`: 16 soft modes, fourfold degeneracy, four Goldstone modes;
  - `nu=-2`: 12 modes, including two quadratic and one linear Goldstone mode;
  - q/-q RPA pairing and static-Hessian positivity for the stable source.

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
3. **Khalaf Fig. 3 production benchmark**
   - fresh `18x18` K-IVC sources at `nu=0,-2` with the paper parameters;
   - compare mode counts, degeneracies, Goldstone dispersion type, HF gaps,
     and only then the plotted dispersion.
4. **Independent Kumar-Xie-MacDonald benchmark**
   - reproduce q=0 spin Ward/Goldstone and the seven low-energy branches at
     `nu=-3` under that paper's reference and interaction conventions.
5. **TBG/hBN benchmark**
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
