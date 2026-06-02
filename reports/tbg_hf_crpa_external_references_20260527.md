# TBG HF+cRPA External Reference Search

Date: 2026-05-27

Status: reference search completed; no fully equivalent public Zhang-style continuum HF+cRPA implementation found.

## Search Conclusion

I did not find an open-source implementation that matches the full target stack:

- continuum BM flat-band basis
- remote-band Hartree-Fock potential
- cRPA dielectric from remote bands
- self-consistent unrestricted HF at integer fillings
- Zhang PRL 128, 247402 / supplementary Fig. 4 band benchmark

The useful external references split into two categories:

1. `TBG-HF` is the best open-source continuum HF implementation for formula and convention comparison.
2. Davydov/Choo/Fischer/Neupert Materials Cloud is the best open cRPA-screened TBG data/code reference, but it is a Wannier-Hubbard workflow rather than our continuum HF band solver.

Therefore the next productive route is a formula audit, not a parameter sweep: use `TBG-HF` to audit BM/HF/projector/form-factor conventions, and use the Materials Cloud cRPA archive plus cRPA papers to audit screening scale and matrix-element normalization.

## Reference 1: TBG-HF

Repository:

- https://github.com/ziweiwang-code/TBG-HF

Local clone:

- `/data/home/ziyuzhu/Mean_Field/reference/TBG-HF`
- commit: `0d2a3d742aa901fa45ce46690c1385887165f58c`

Companion paper:

- Y. H. Kwan, Z. Wang, G. Wagner, N. Bultinck, S. H. Simon, S. A. Parameswaran, "Mean-field Modelling of Moire Materials: A User's Guide with Selected Applications to Twisted Bilayer Graphene", arXiv:2511.21683.
- https://arxiv.org/abs/2511.21683

Why it is useful:

- It is a Python code for self-consistent Hartree-Fock on the continuum model of TBG.
- It includes BM diagonalization, intravalley form factors, reference projectors, Hartree and Fock contractions, and a full self-consistency loop.
- The README explicitly says the release accompanies the mean-field tutorial/review and is a simplified version of codes used in Kwan/Wagner/Bultinck/Simon/Parameswaran TBG HF papers.

Why it is not a direct Zhang replacement:

- The README states that the release does not include more general interaction schemes and other advanced routines used in the publication code family.
- It does not appear to implement the Zhang remote-band cRPA dielectric pipeline.
- It is still the most useful public code anchor for our no-cRPA/HF convention audit.

Important local convention anchors from the cloned code:

- Form factor in `singleParticle.py::gen_form_factors` is documented as `<u(k)|u(k+q+G)>` in periodic gauge.
- The form-factor construction uses periodic k wrapping and explicit G shifting; this is directly relevant to our `hf_periodic` convention and the `Q + wrap` alias issue.
- `projectors.py::gen_projector` has an `average central` reference projector: remote valence filled, remote conduction empty, central bands half-filled.
- `mainProgram.py` uses the HF matrix input as `P - P_ref`.
- `routines.py::calc_fock_matrix` separates Hartree and Fock terms; the Fock contraction includes the minus sign.

Immediate use in our audit:

- Compare our `Delta = P - P_ref` construction against `TBG-HF/mainProgram.py`.
- Compare our HF projector index order against `TBG-HF/projectors.py`.
- Compare our form factor convention against `TBG-HF/singleParticle.py::gen_form_factors`.
- Compare our Hartree/Fock contraction index order against `TBG-HF/routines.py::calc_fock_matrix`.
- Do this first on the already validated no-cRPA branch, then map the cRPA replacement into the same contraction slots.

## Reference 2: Davydov/Choo/Fischer/Neupert Materials Cloud cRPA Archive

Dataset:

- "Four- and twelve-band low-energy symmetric Hamiltonians and Hubbard parameters for twisted bilayer graphene using ab-initio input"
- https://archive.materialscloud.org/record/680
- newer Materials Cloud page: https://archive.materialscloud.org/records/axch9-vsd17

Paper:

- A. Davydov, K. Choo, M. H. Fischer, T. Neupert, "Construction of low-energy symmetric Hamiltonians and Hubbard parameters for twisted multilayer systems using ab-initio input", Phys. Rev. B 105, 165153 (2022), arXiv:2012.12942.
- https://arxiv.org/abs/2012.12942

Why it is useful:

- The arXiv abstract states that they construct low-energy four-band and twelve-band Hamiltonians and compute extended Hubbard parameters within cRPA.
- Materials Cloud provides `codes.tar.gz` and `data.tar.gz`.
- The archive states that `codes.tar.gz` contains custom TBX code and a modified Wannier90, and `data.tar.gz` contains input/output data including cRPA-screened Hubbard parameters.

Why it is not a direct Zhang replacement:

- This is an ab-initio/Slater-Koster/Wannier-Hubbard workflow.
- It gives screened Hubbard matrix elements, not a continuum BM remote-band dielectric matrix ready to plug into our HF solver.

Immediate use in our audit:

- Use it to sanity-check cRPA screening magnitude and matrix-element normalization.
- Do not use it to judge the final Zhang Fig. 4 band topology directly.
- If our cRPA dielectric or screened interaction is off by an obvious factor, compare definitions of unit-cell area, Fourier normalization, spin/valley degeneracy, and excluded target subspace against this archive.

## Reference 3: Zhang/Lu/Liu Target Paper

Paper:

- S. Zhang, X. Lu, J. Liu, "Correlated insulators, density wave states, and their nonlinear optical response in magic-angle twisted bilayer graphene", Phys. Rev. Lett. 128, 247402 (2022), arXiv:2109.11441.
- https://arxiv.org/abs/2109.11441

Why it remains the target:

- The abstract says their correlated-insulator and density-wave calculations use extended unrestricted Hartree-Fock including Coulomb screening effects from remote bands.
- This is the closest target for our current HF+cRPA reproduction.

Code availability:

- I did not find a public companion code repository for this Zhang PRL in the web/GitHub search.
- Therefore the paper and supplement remain formula targets, not executable references.

Immediate use in our audit:

- Treat direct-gap and SCF-grid band structure as the hard acceptance gates.
- Continue using the Zhang supplement equations for the cRPA formula chain.
- Do not infer correctness from dielectric-only plots if the HF bands fail.

## Reference 4: cRPA Physics Benchmarks

Vanhala/Pollet:

- T. I. Vanhala and L. Pollet, "Constrained random phase approximation of the effective Coulomb interaction in lattice models of twisted bilayer graphene", arXiv:1909.09556.
- https://arxiv.org/abs/1909.09556

Use:

- Qualitative benchmark for static screening behavior and effective interaction trends.
- Not a direct HF band solver.

Goodwin/Kennes/Rubio/Wehling family:

- Useful for cRPA screening scale and device-geometry dependence.
- I did not find a directly reusable public continuum HF+cRPA code in the current search.

## Audit Checklist For Our Code

This is the concrete external-reference-driven checklist before more Slurm sweeps:

1. BM basis and reciprocal lattice
   - Compare our BM plane-wave indexing, valley labels, and moire reciprocal vectors against `TBG-HF/singleParticle.py`.

2. Periodic form factor
   - Compare our `hf_periodic` form factor to `TBG-HF`'s `<u(k)|u(k+q+G)>` periodic-gauge form.
   - Re-check the `Q + wrap` requirement and ensure the cRPA artifact grid cannot alias the actual transferred G.

3. Reference projector
   - Compare our `P_ref` / `Delta = P - P_ref` against `TBG-HF`'s `average central` and `mainProgram.py` usage.
   - Keep Zhang CNP occupation logic separate from HF filling projectors.

4. Hartree contraction
   - Compare our Hartree matrix contraction order against `TBG-HF/routines.py::calc_fock_matrix`.
   - This is the correct place to settle `hartree_dimless[q1_index, q2_index]` vs transposed indexing by derivation.

5. Fock contraction
   - Compare sign and index order against `TBG-HF`'s explicit negative Fock contraction.
   - Then map bare Coulomb versus cRPA-screened interaction replacement into the same slot.

6. cRPA normalization
   - Use Zhang supplement as the primary formula.
   - Use Davydov Materials Cloud as a secondary check for area factors, spin/valley factors, and screened-interaction units.

7. Acceptance diagnostic
   - Prefer saved SCF-grid point line plots and exact-grid summaries.
   - Reconstructed high-symmetry path bands remain presentation-only until the SCF-grid direct gap and bandwidth are correct.

## Current Next Step

The next code-facing action should be a narrow formula audit against the cloned `TBG-HF` implementation, starting with no-cRPA HF conventions that are already known to be validated locally. Only after that mapping is clean should the cRPA-screened interaction be inserted and tested on small legal artifacts.
