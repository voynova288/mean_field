# Draft request for Fig. S45 raw data / implementation details

**Subject:** Request for raw Fig. S45 TDHF data and finite-cutoff convention (Phys. Rev. B 112, 075109)

Dear Prof. Bernevig and coauthors,

I am independently reproducing the R5G/hBN Hartree-Fock and TDHF calculation in Fig. S45 of Phys. Rev. B 112, 075109 (arXiv:2312.11617), using the stated `(0+4)` screened basis projection, 12x12 mesh, `xi=1`, `theta=0.77 deg`, `V=64 meV`, and `epsilon_r=5`.

I have isolated a finite-cutoff convention ambiguity. A branch-quotient implementation is exactly C3 covariant and satisfies the signed q/-q RPA identities, but forces nodes at the two nonzero C3-fixed torus points and shifts the nearly uniform Fig. S44 intervalley mode upward. A matching fixed-|G| single-representative HF/TDHF functional restores the near-uniform mode, nonzero fixed-point weight, and paper-close energies: at q=0 the intervalley mode is `2.16695 meV` with `IPR*Nk=1.006`, and at the three smallest C3-related finite q points the intraflavor/intervalley/interspin raster RMSEs are `0.057/0.049/0.0037 meV`.

However, when those q sectors are assembled independently, the single-representative regulator has a `1.20 meV` maximum C3 defect in the converged HF band spectrum and `1.36--1.71 meV` in the full TDHF A spectrum, even though the lowest-branch C3 spreads are only `0.0027--0.029 meV`. I therefore did not copy or average sectors. Instead I ran all 144 q points independently as a finite-cutoff hypothesis test. The intraflavor white-region classifier recovered 35 of the 45 raster-white q points with no false positives; the stable-point RMSEs were `0.559/0.130/0.050 meV` for intraflavor/intervalley/interspin. This is substantially closer than the exact quotient but is not a completed reproduction.

The HF interaction, ODA, scalar energy, q=0 response, and finite-q projected response are now bound to one executable provider. Every one of 432 A/B columns in both signed sectors agrees with the explicit D19 contractions to below `5.8e-16 meV` at `q=(1,0)` and exact M. A generic-q exact unitary-projector finite-difference gate agrees with the static Hessian to `2.8e-8 meV/cell`. At exact M, a canonical raw `-M` role-resolved scalar also has a negative unitary-projector curvature (`-0.0130632 meV/cell`), driven by a positive A contribution and larger negative B contribution. But raw `-M` and `+M` are torus-equivalent while their finite-cutoff A blocks differ by up to `0.00559 meV`; I have kept this signed-boundary ambiguity explicit rather than averaging it away.

Would you be willing to share either the raw 12x12 lowest-mode arrays used for the kappa_hBN=1 bottom row, or the following implementation details?

1. How is `k+q` folded when it leaves the stored torus representative, and how is periodic gauge imposed at that boundary?
2. Is the finite interaction cutoff applied to `|G|` or to the physical transfer `|q+G|` (or `|q_WS+G|`)?
3. How are exactly shortest-vector boundary ties handled?
4. How are the two nonzero C3-fixed torus points represented in the active and filled-remote-band sectors?
5. Is the finite plane-wave parent dimension kept constant over the torus? The 19-point shell is C3 closed at Gamma but its affine closure has 27 points at the nonzero fixed sectors; conversely, a nearest-27 rule cuts a Gamma boundary orbit (including all exact ties gives 31 points). Is your implementation variable-rank, tie-averaged/multi-representative, or based on another support prescription?
6. Are C3/inversion-related HF k points and TDHF q sectors assembled independently, or is the HF source/response computed only in an irreducible wedge and copied from symmetry representatives? If copied, what sewing/representation is used at the two nonzero C3-fixed torus points?
7. On the even 12x12 mesh, how are the torus-equivalent raw aliases `q_i=-6` and `q_i=+6` treated at exact M? Is one half-open representative selected for both particle-hole orientations, or do the ph and hp legs retain opposite repeated-zone lifts?
8. How is the filled-valence average-scheme Fock contribution evaluated with the 19-RLV plane-wave cutoff?
9. If available, could you share the raw complex eigenvalues, static-Hessian inertia, or negative/complex classification used for the white intraflavor regions?

The distinction is numerically important: an exact branch quotient changes the response Hilbert space and fixed-point mode weight, whereas a literal single-representative finite regulator gives paper-close low modes but a measurable unaveraged C3 defect in the converged source. I would be happy to share the raw signed-q spectra, source/Ward closure metrics, and convention audit.

Thank you for considering the request.
