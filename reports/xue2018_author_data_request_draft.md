# Draft request for Xue--MacDonald 2018 Fig. 2 numerical details

**Subject:** Request for numerical details or raw data for PRL 120, 186802 (2018), Fig. 2

Dear Dr. Xue and Prof. MacDonald,

I am independently reproducing the Hartree--Fock calculation in “Time-Reversal Symmetry-Breaking Nematic Insulators near Quantum Spin Hall Phase Transitions,” with particular attention to the higher-energy time-reversal-preserving blue branch in Fig. 2.

Using the public equations, the ordinary-electron reference relative to filled valence bands, and a complete momentum-dependent TRS self-energy, I find a stationary TRS branch that is continuous from Fig. 2 points 1 through 26 but folds before point 27. It therefore does not reproduce the connected blue curve. Independent roots at all 62 points and several source-motivated constrained/seed calculations also do not recover the published branch.

Could you share any of the following, if still available?

1. The raw arrays behind Fig. 2: the 62 `(E_g,A)` points, black order parameter, red ground-state gap, and blue higher-energy TRS gap.
2. The momentum domain and quadrature used for the main four-band calculation: cutoff, grid shape/size, endpoint convention, and weights.
3. The treatment of the `q=0` Coulomb term or singular source cell.
4. The self-consistent iteration method: mixing/ODA/DIIS parameters, tolerances, finite-temperature smearing if any, and continuation direction.
5. The initial self-energy used for the TR-preserving solution: amplitude and momentum profile associated with
   `H_{up,down}^{vc}(0)=-H_{up,down}^{cv}(0)`.
6. Whether time reversal or a reduced Pauli/channel ansatz was enforced during iteration, or whether the relation was used only as an initial seed in the unrestricted `4x4` Fock equation.
7. Whether the plotted gap was minimized only on the self-consistent grid or with a separate continuous-`k` search.
8. Any surviving source code, input files, checkpoints, or notes sufficient to reconstruct this branch.

For provenance, I am using arXiv:1710.00410v3 and the 2018 dissertation “Interactions and topology in two-dimensional semiconductor and semimetal” (DOI `10.15781/t2sb3xh72`). I will keep any digitized figure data separate from independent calculations and will not fit interaction or momentum scales to the published curve.

Thank you for any guidance or archived material you can provide.
