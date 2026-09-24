# InAs/GaSb projected matrix-EI model contract

## Status and purpose

This is a new-model research path for explaining the InAs/GaSb experiment. It is not a claim to reproduce the unpublished Du/Kai-Chang workflow. Digitized Fig. 2 curves, fitted `pairing_scale`, scalar `xi/Delta`, and current vertices from a different Hamiltonian are forbidden as model inputs.

The hard credibility target is now a fresh split-zero full 8-band Kane–Poisson normal reference, a systematically converged multiband matrix BCS/HF calculation, and a chemical potential defined in one explicit electrostatic ensemble. The detailed equations and fail-closed gates are in `plan/DU2017_FULL8_KANE_MATRIX_BCS_WORKPLAN.md`.

`Kane4Bundle` remains an implementation milestone: the same fresh `split=0` Kane calculation supplies the four-state low-energy Hamiltonian, eight-orbital microscopic amplitudes, projected density vertices, and current numerator. It is a projected four-state model, not full-8-band interacting HF, and cannot by itself authorize a material prediction. Every downstream scaffold artifact must still store the same bundle fingerprint.

`normal_reference.py` now provides an arbitrary selected-window Note-6 carrier/chemical-potential contract. It independently projects the first-two and last-six microscopic Kane orbital sectors, separates canonical neutrality roots from fixed-reservoir evaluations, binds the exact input arrays by hash, records root diagnostics, and tests energy-zero covariance. Its claim is only a window-restricted numerical root until fixed-gate and window/momentum/z convergence gates pass.

`kane_poisson.py` and `kdotpy_poisson_adapter.py` now provide a fresh
canonical-periodic `split=0` fixed-point path with actual full-parent `+U I_8`
attestation, variable-dielectric Gauss closure, final replay, and typed archive
reload. The 2026-07-19 W4/W8/W12/W16 ladder passed every numerical gate but
failed band-window convergence: both the neutral carrier density and chemical
potential changed strongly as remote mixed confinement subbands entered. This
is evidence that the current raw orbital-transfer/window closure is not yet a
production multiband electrostatic reference. It does not authorize fixed-gate
physics or matrix HF; the next required derivation is a remote-complement or
explicitly reference-subtracted carrier closure.

The 2026-07-20 follow-up separates three different counts that must not be
conflated. The active Poisson density uses W4: two E1 Kramers branches and two H1
Kramers branches, four bands total. A candidate-28 eigensystem is used only for
robust overlap selection, and the complete parent has `8*nz=1056` states; neither
28 nor 1056 is the number of explicit charge-carrying bands. Exact full-parent
completion is algebraically consistent but UV dominated and is prohibited from
SCF. Fresh W4 jobs at `(nr,kmax)=(24,0.14)` and `(40,0.18)` both converge
numerically but fail outer-k stability, with a `69.94%` density change between
cutoff setups. No further Poisson refinement or matrix HF is released by those
runs.

## 2026-07-21 TEIB-derived fixed-density diagnostic

The supplied TEIB code was reproduced unchanged before any port. Its Poisson
stage inputs `n_e=n_h=5.5e10 cm^-2`, uses one E1 and one H1 branch with declared
Kramers degeneracy two, and does not solve a chemical potential. Its default Fig.
2 path is a digitized paper reconstruction and is not a physics source.

A historical `experimental_fixed_pair_density` diagnostic used explicit
rank-two E1/H1 projectors, TEIB midpoint-annular weights, all eight Kane
components, periodic variable-dielectric Poisson, and independent cold replays.
It had no `mu_mev`, was neither canonical nor fixed-gate, and never authorized
Hartree-Fock. That diagnostic implementation and its dedicated tests are now
archived under `local_archive/retired_surface/inas_gasb_fixed_pair_diagnostic_20260916/`;
it is not part of the maintained API. The canonical Kane-Poisson and matrix-EI
APIs are unchanged in meaning.

GaSb `gamma2=0.08` is available only as an explicit, source-hash-pinned inferred
material policy; the printed `8.18` path remains available as a causal control.
It is not safe to silently adopt either value. At matched `nk=81` fixed density,
corrected/printed kdotpy give Gamma E1-H1 separations `+20.4045/-13.8235 meV`
and potential ranges `4.76969/3.41545 meV`, while supplied TEIB `N=7` gives
`-29.8614 meV` and `3.33774 meV`. Exact annular-weight parity is below
`1e-19 nm^-2`, so this mismatch is in the parent/profile chain, not the density
normalization.

The named TEIB `M` sign and kdotpy `rr` sign are related by the fixed basis phase
`diag(1,1,-i,1,i,-1,-i,-1)`; homogeneous bulk matrices and spectra agree to
about `1e-7 meV`. No kdotpy sign change is justified. Conversely, TEIB's default
`N=7` plane-wave parent is not z-basis converged: `N=31` changes E1-H1 to
`-33.8573 meV` and the potential range to `2.95981 meV`. A kdotpy `dz=0.25 nm`
check also changes the physical candidate positions, and Slurm `192320` stops
fail-closed because adjacent-k and previous-potential pair assignments disagree.
Therefore no fixed-density result authorizes HF. Full evidence is in
`results/teib_reference_reproduction_20260721/TEIB_MEAN_FIELD_PHYSICS_CHAIN_AUDIT.md`.

## Basis and representation

Use the ordinary electron-band basis

\[
\Psi_{\bf k}=(c_{E1,+},c_{E1,-},c_{H1,+},c_{H1,-})^T .
\]

The E1 and H1 Kramers pairs are explicit. No extra spin degeneracy is applied. Excitonic order is interband electron coherence in this representation; it is not an additional superconducting BdG term.

For a momentum-dependent active-basis rotation

\[
\Phi_{\bf k}\rightarrow \Phi_{\bf k}G_{\bf k},\qquad
G_{\bf k}\in U(2)_{E1}\times U(2)_{H1},
\]

all local matrices transform as

\[
M_{\bf k}\rightarrow G_{\bf k}^\dagger M_{\bf k}G_{\bf k}.
\]

Only block traces, singular values, spectra, densities, energies, and response traces that obey this covariance may be interpreted physically.

## Kane4 bundle

The microscopic active frame is

\[
\Phi_{\bf k}(z)\in\mathbb C^{N_{\rm micro}\times4},\qquad
\int dz\,\Phi_{\bf k}^\dagger(z)\Phi_{\bf k}(z)=I_4.
\]

The bundle stores

- Cartesian momenta and exact integration weights for \(d^2k/(2\pi)^2\);
- the z grid and integration weights;
- \(h_0^{(4)}({\bf k})\);
- \(\Phi_{\bf k}(z)\);
- the projected full-Kane derivative numerator \(\Phi^\dagger\partial_{k_i}H_{\rm Kane}\Phi\), when available;
- source structure, potential, kdotpy version/options, active-state selection, split value, completeness, and a content hash.

The existing historical split-zero Lowdin artifact saves only `h_eff`; it does not save the lifted microscopic frame and therefore is insufficient by itself for a matrix interaction.

## Density vertex

The z-resolved projected density vertex is

\[
\Lambda_{{\bf k}{\bf p}}(z)=
\Phi_{\bf k}^\dagger(z)\Phi_{\bf p}(z),
\]

with identities

\[
\Lambda_{{\bf k}{\bf p}}^\dagger(z)=
\Lambda_{{\bf p}{\bf k}}(z),\qquad
\int dz\,\Lambda_{{\bf k}{\bf k}}(z)=I_4.
\]

The scalar Gamma6/heavy-hole probability profiles in `ei_mf.wavefunction_form_factor` are diagnostic charge profiles only; they discard the phases, cross-orbital matrix elements, and k dependence needed here.

## Normal-ordered projected Phase 1

Define the ket-oriented density matrix

\[
P_{ab}({\bf k})=\langle c_{{\bf k}b}^\dagger c_{{\bf k}a}\rangle
\]

The production reference is the ordinary-electron E1-empty/H1-filled state

\[
P_{\rm ref}=\operatorname{diag}(0,0,1,1),\qquad
D({\bf k})=P({\bf k})-P_{\rm ref}.
\]

`noninteracting_fermi_state` remains available only as an explicitly selected historical incremental diagnostic; it is no longer the production default. `MatrixEIConfig` also defaults to `momentum_policy=explicit_cartesian_2d`: the mesh must have rank two, positive weights, and exact closure to the sampled area divided by `(2*pi)^2`. Radial/axial and analytic tests must explicitly select `analytic_or_legacy_diagnostic`.

Two policies are implemented. `normal_ordered_exchange_only` applies only `Sigma_F[D]`. `normal_ordered_hartree_fock` applies Hartree and Fock to the same reference-subtracted density only when `H0` does not already contain that Hartree field. For a saved self-consistent Kane--Poisson parent, the correct continuation is `H0 + Sigma_H[P-P_parent] + Sigma_F[P-P_ref]`; applying `Sigma_H[P-P_ref]` would double count the parent Poisson potential and is rejected fail-closed. Until the split-reference Hartree functional is implemented, such parents use frozen-Poisson exchange-only diagnostics. The saved Phase-1 runs used a differential periodic, zero-mean layered Hartree together with uniform-dielectric exchange; they remain explicitly mixed-kernel scaffolds. The code now supports two builder-attested common response families: (i) an *open, ungated* layered family with neutral `q=0` zero-exterior-field Hartree and finite-`q` exchange, and (ii) an ideal one-sided-gate family with a homogeneous Dirichlet metal plane half a z-grid cell beyond one endpoint and a transparent semi-infinite dielectric at the other. Both use one dielectric grid, one boundary discretization, and one content-bound Hartree/Fock attestation. The second family matches the topology of a front-gated/semiconductor-substrate device, but the numerical box face is not a measured Wafer-B gate position. Neither builder generates the nonzero gate potential in `H0` or updates the Kane envelopes.

For Fig. 2, the final article specifies a semi-insulating-GaAs Wafer B with a 1 micrometre buffer, 11.5 nm InAs / 8 nm GaSb wells, Al0.8Ga0.2Sb barriers, and a 5 mm x 5 mm semitransparent front gate. The thesis identifies the optical gate as 10 nm Pd over Al2O3, but gives no Al2O3 thickness. Its separate *typical-stack* description conflicts with the final article's Wafer-B thickness/doping labels, so the quoted 50 nm upper barrier and 3 nm + 3 nm caps are not promoted to the final device. Until the final cap/oxide thickness, work function, fixed charge, CNP voltage, and z-orientation are supplied, the ideal one-sided gate is a topology-correct response scaffold rather than a fixed-voltage device source.

For a supplied mixed-representation Coulomb Green function \(G_q(z,z')\), the exchange self-energy is

\[
\Sigma_F({\bf k})=-\int_{\bf p}\int dz\,dz'\,
G_{|{\bf k}-{\bf p}|}(z,z')
\Lambda_{{\bf k}{\bf p}}(z)D({\bf p})
\Lambda_{{\bf p}{\bf k}}(z').
\]

Three different E1--H1 quantities are saved and must not be conflated:

\[
\chi_{EH}=D_{EH}=\langle H1^\dagger E1\rangle,
\qquad
\Phi_{EH}=-\Sigma^F_{EH},
\qquad
\Delta_{{\rm Du},EH}=-2\Sigma^F_{EH}.
\]

The factor of two is the normalization needed for Du's
\(E^2=\xi^2+|\Delta|^2\) convention. The normal Kane hybridization
\((h_0)_{EH}\), interaction field \(\Phi_{EH}\), Du-normalized gap, and total
\((h_{\rm MF})_{EH}\) are stored separately. If \((h_0)_{EH}\ne0\), the total
E1--H1 block is not called spontaneous excitonic order. The interaction energy is

\[
E_F[D]=\frac12\int_{\bf k}{\rm Tr}[\Sigma_F({\bf k})D({\bf k})].
\]

There is no second gap equation using the same interaction: adding one would double count exchange/pairing.

At finite temperature the reported thermodynamic diagnostic is

\[
\Delta F=\Delta E-T\,[S(P)-S(P_{\rm ref})],
\]

with the matrix Fermi entropy evaluated from the eigenvalues of each density matrix. Internal energy and canonical free-energy difference are saved separately.

The initial Phase-1 SCF uses the global CNP active-space constraint

\[
\int_{\bf k}{\rm Tr}P({\bf k})
=2\int_{\bf k}1,
\]

which fixes only the total active occupation. A later gate-voltage calculation must replace this simplified constraint with a profile-dependent electrostatic functional. In the strongly hybridized continuum model, raw Gamma6/valence weights, fixed-reference E1/H1 weights, and occupied-upper/vacant-lower energy-pair counts are inequivalent.

The N95 exact-bundle audit makes the ensemble distinction explicit. If the complementary microscopic orbital weights are reused as the interacting BCS number equation, then

\[
 n_{\Gamma_6}-n_{\rm valence\ holes}
 =\int_{\bf k}{\rm Tr}P
 -\int_{\bf k}{\rm Tr}P_{\rm valence}.
\]

For this quartet the second term corresponds to an average active occupation `2.515094065063523`, not two. The resulting coherent solution is necessarily metallic because its chemical potential lies in the upper active bands. This orbital-weight equation remains the declared Supplementary-Note-6 Kane--Poisson charge closure; it must not be silently promoted to the E1/H1 BCS number equation.

At E1/H1 half filling, the unmodified projected Kane Hamiltonian is already insulating because of normal one-body E1--H1 hybridization, and the spontaneous exchange block collapses. To isolate the paper-style BCS mechanism, the code therefore provides a separately labelled diabatic reduction that zeros only the normal E1--H1 blocks before solving at active half filling. On the exact N95 grid its normal state has indirect overlap `-2.2150406617712406 meV`. One cold-replayed nphi=32 basin has maximum excitonic singular value `1.3159291770637007 meV`, indirect gap `1.5783409346735056 meV`, and no saved-grid Fermi crossings, but it is metastable. Four independent nphi=16 seed channels converged to the same lower-free-energy metallic basin; its nphi=32 cold replay has `Delta F=-0.0003425688461355203 meV nm^-2`, maximum excitonic singular value `0.5282292562406592 meV`, and 108 Fermi-crossing brackets on 32/32 rays. A later ensemble audit showed that active half filling enforces charge neutrality but does not fix the common electron--hole pair density. The lower metal is driven mainly by diagonal H--H exchange while the phase-2 parent Poisson potential remains frozen. The user subsequently fixed the physical ensemble more strongly: the one chemical potential produced by Kane--Poisson must be frozen during BCS/HF, with no density re-root and no independent electron/hole chemical potentials. Therefore all active-half rerooted branches in this paragraph are forensic and not source-consistent equilibrium candidates. Neither a gapped basin nor a visually similar JDOS may be selected because it resembles Fig. 2, and no Coulomb channel may be multiplied by an unsourced empirical scale. This reduction is not the full projected Kane Hamiltonian and must remain labelled accordingly.

The fixed-mu interface is now explicit. `MatrixEIConfig(constraint_policy="kane_poisson_fixed_mu", fixed_mu_mev=...)` evaluates every density update at the supplied same-gauge Kane--Poisson ordinary-electron mu and never calls a number or neutrality root. This `mu_el` must not be confused with the pair-density Lagrange multiplier denoted `mu` in the Du thesis and Supplementary-Note-2 scalar BCS equations. In an ordinary-electron-to-hole transformation, `xi_pair=E_c-E_v` while `eta_charge=E_c+E_v-2*mu_el`; the absolute Kane--Poisson mu cancels from the pair-energy channel. `electron_hole_vacuum` is now the production default and uses the ordinary-electron E1-empty/H1-filled projector so that `D=P-P_ref` retains carrier Fock terms. `noninteracting_fermi_state` is an explicitly selected historical incremental diagnostic only. For fixed mu the result also reports `Delta Omega=Delta E-T Delta S-mu_el Delta N_el`, rather than using the canonical free energy for branch ranking. A saved-grid N95 preflight at `mu_el=113.8688022886634 meV` gives microscopic `n_e/n_h=0.00304706/0.00309574 nm^-2` for the unmodified projected Kane quartet at `0.1 K`, close but not identical to the parent common density `0.00303040 nm^-2`. Applying the no-normal-E/H reduction at the same mu changes these to `0.00313699/0.00250557 nm^-2`, a large `0.000631424 nm^-2` imbalance. A one-step unscaled radial full-carrier Fock map is also nonperturbative: `max|Sigma_F|=10.8834 meV`, relative weighted density change `0.58798`, and microscopic `n_e/n_h` move from `0.00302265/0.00303514` to `0.00353875/0.00481748 nm^-2`. A subsequent two-radial-node core-path SCF converges in 1198 iterations at unchanged `mu_el`, proving that `D=P-Pvac` full-carrier Fock and the fixed-mu grand potential are numerically executable; its sparse-cell densities are not physical estimates. A full `nr=160,nphi=512` radial run and four independent seed probes then all converged to the same insulating basin (`Delta Omega=-0.1013491 meV nm^-2`, indirect gap `7.3483 meV`, `max|Sigma_F|=14.8277 meV`, maximum E-H singular value `2.58405 meV`). E-E and H-H Fock energies, `-0.02679` and `-0.03215 meV nm^-2`, dominate the E-H contribution `-0.004257 meV nm^-2`. The microscopic density imbalance is `-0.00238831 nm^-2` even though the active insulator has equal E1-electron/H1-hole counts. This branch fails cutoff closure: its endpoint `||D||_F`, `||Sigma||_F`, and E-H singular value are `0.3928`, `7.6421 meV`, and `1.2648 meV`, and the outer `k>=0.20 nm^-1` window carries `22.1%/30.9%` of weighted D/Sigma norms. Decomposing `D_final=(P_nonint-Pvac)+(P_final-P_nonint)` does not isolate the problem to a removable normal-state constant: the interaction-induced term alone has endpoint `||Sigma||_F=4.1577 meV`, E-H singular value `0.7609 meV`, and `34.3%` of its weighted Sigma norm above `0.20 nm^-1`. Therefore fixed `mu_el` does not by itself preserve the Kane--Poisson density once diagonal exchange is iterated, and this strong frozen-parent basin is not a physical phase result. The public Du equations do include diagonal exchange, so pairing-only BCS is not source-faithful; nevertheless their printed chemical-potential algebra does not define a unique mapping to `mu_el`. Earlier UV audits already rejected both extending the continuum cutoff and choosing a generic finite-BZ stencil as predictive physics. By user scope, atomistic/Wannier/Tan replacements are also excluded: Du2017 work must remain inside the paper/supplement/thesis/author-code model family. Du-only forensic tests may examine undocumented or result-selected momentum cutoffs, retained-subband/reference windows, quadrature, chemical-potential, diagonal-Fock, and electrostatic conventions, but any unsupported Fig. 2 match remains forensic/uncontrolled and cannot authorize exact-off-axis, optical, or physical-phase claims.

### Signed-band carrier diagnostic versus reference-subtracted charge

The source-consistent kdotpy radial IDOS convention transports nonzero signed band indices from the charge-neutral gap. At zero temperature it defines

\[
 n_e=\sum_{b>0}\int_{\bf k} f_b({\bf k}),\qquad
 n_h=\sum_{b<0}\int_{\bf k}[1-f_b({\bf k})].
\]

For a radial interval, kdotpy linearly interpolates the occupied fraction between its endpoint energies and multiplies it by \((k_{i+1}^2-k_i^2)/(4\pi)\). `band_carriers.py` implements this convention and has exact numerical parity with the installed kdotpy IDOS. On the frozen-potential source at `kmax=0.12 nm^-1`, all remote signed bands contribute zero: the active indices `(-2,-1,+1,+2)` give the entire balanced count, about `2.318e10 cm^-2`. The Supplementary-Note-6 Gamma6/valence-orbital weighting on the same source gives `4.041e10 cm^-2`, so the signed-band value is only `0.5736` of it; this is a carrier-definition difference, not quadrature noise. Moreover, the final radial interval alone contributes about `4.42e9 cm^-2`, or `19.1%` of the signed electron count. Fresh `split=0` diagonalization changes the total by only about `9.8e4 cm^-2`; the artificial split is therefore not the closure failure. A subsequent source-anchored all-band alignment scan extended the split-zero calculation to `kmax=0.22 nm^-1`: the balanced signed count grew from `2.318e10` to `2.488e11 cm^-2`, while the absolute final-shell contribution grew from `4.42e9` to `1.394e10 cm^-2`. The active quartet supplied exactly 100% of the count at every cutoff, so neither remote carrier bands nor energy-column relabeling repairs the closure. Neither carrier convention currently supplies a cutoff-independent gate calibration.

The UV-safe quantity already used by the differential Hartree operator is instead the reference-subtracted electron-number density

\[
 \delta n(z)=\int_{\bf k}\mathrm{Tr}[\Lambda_{{\bf k}{\bf k}}(z)D({\bf k})],
 \qquad D=P-P_0,
\]

which obeys

\[
 \int dz\,\delta n(z)=\int_{\bf k}\mathrm{Tr}D({\bf k}).
\]

It is invariant under local active-space gauge rotations and needs no electron/hole quasiparticle decomposition. At fixed active number its periodic Poisson source is neutral. It closes the *response relative to a declared reference*, but it does not determine that reference or the external gate alignment; those require a common fixed-gate Kane-Poisson functional.

## Coulomb and self cell

For a uniform dielectric,

\[
G_q(z,z')=
\frac{2\pi e^2}{4\pi\epsilon_0\epsilon_r q}
 e^{-q|z-z'|}.
\]

Production code must integrate the \(q=0\) momentum cell using the declared 2D mesh. An arbitrary `q_floor` is not an acceptable self-cell prescription. A repeated-periodic z Green function cannot be reused naively for exchange: its constant-z mode scales as `1/q^2`, so the two-dimensional self-cell has a logarithmic infrared divergence. `layered_green.py` therefore makes the boundary family explicit. Its open discrete-transparent kernel has the physical `1/q` infrared limit and a finite circular-cell average; its neutral `q=0` limit supplies the matching zero-exterior-field Hartree operator. The ideal one-sided-gate kernel instead has a finite `q->0` response because induced metal charge balances even a nonneutral source; its homogeneous-Dirichlet response at the gate and transparent condition at the opposite endpoint are used identically by Hartree and Fock. Uniform discrete image-charge oracles, an independent heterogeneous cumulative-flux `q=0` oracle, and an absolute circular-cell integral fix its signs and normalization. The implementation canonicalizes endpoint exterior dielectrics, owns and freezes all action arrays, and binds both direct and compressed Fock storage to a common attestation. Mixed or unattested Hartree/Fock specifications are rejected unless an explicitly labeled scaffold opts in. The physical cap/gate dielectric, semi-insulating GaAs and buffer, fixed charges, work function, nonzero gate source, fixed-voltage thermodynamic reference, and paired-state screening remain missing model layers.

## Axial radial exchange reduction and mesh status

For an axially sewn radial density matrix, the projected exchange action is reduced exactly at finite `nphi` by pulling both source legs through the active-space rotation,

\[
K^{\mathrm{eff}}_{ij;adbc}
=\frac{1}{N_\phi}\sum_{\ell rs}
K^{2D,\ell}_{ij;adrs}(U_\ell)_{rb}(U_\ell)^*_{sc},
\qquad
\Sigma_{i;ad}=-\sum_{jbc}w_jK^{\mathrm{eff}}_{ij;adbc}D_{j;bc}.
\]

The circular self cell uses the full-mesh point weight `w_j/nphi`, not the radial weight. `AxialProjectedFockOperator` is builder-attested and source-bound, and the actual Kane4 `nphi=4` oracle agrees with the full two-dimensional action to `1.24e-14 meV` in the self-energy and `6.94e-18 meV nm^-2` in exchange energy.

A fixed-`nr` limit with `nphi -> infinity` is **not** a valid polar-mesh convergence path for the present midpoint quadrature. It makes same-annulus sectors arbitrarily thin while keeping their radial width fixed, exposing the point-sampled `1/q` singularity logarithmically. The `nr=24` scan through `nphi=4096` therefore drifts rather than converges; it is a diagnosed anisotropic-mesh artifact, not evidence for a larger gap. Subsequent scans keep the outer annular-sector arc/radial aspect ratio near one: `(nr,nphi)=(12,144),(24,296),(48,596),(96,1200)`.

At `delta=12 meV`, `epsilon_r=15`, and `T=0.1 K`, the balanced meshes contain two reproducible axial+TR basins. On `nr=48`, the weak/semimetal branch has `(F,gap,order)=(-0.001611687 meV nm^-2,-4.865712 meV,0.112176 meV)`, while the strong/gapped branch has `(-0.001437827,+1.766003,1.121705)`. On `nr=96`, these become `(-0.001622308,-4.853338,0.102955)` and `(-0.001392167,+1.764636,1.122899)`. Thus the gapped branch is radially stable but metastable; the constrained same-mesh free energy favors the weak semimetal by about `2.30e-4 meV nm^-2` on the finest mesh.

Thermodynamic, gap/order, and E1-H1 off-diagonal changes are small from `nr=48` to `96`, but the original predeclared full-matrix gate remains failed. A subsequent clean `nr=48,nphi=596` audit removed the radial TR projector while retaining axial covariance. TR-odd seeds found a much lower Stoner-like branch, `F=-0.003095817 meV nm^-2`, with nearly complete E1 Kramers polarization (`|sigma_z|≈0.9953`); its opposite-polarization solution is the explicit time-reversal partner. The result was then continued to `nr=96,nphi=1200`, where `F=-0.003069106 meV nm^-2`, the indirect gap is `-8.707256 meV`, and `|sigma_z(E1)|=0.995377`. The `48/596 -> 96/1200` comparison passes all original full-matrix thresholds: `Delta F=2.67e-5 meV nm^-2`, total `D/Sigma` RMS `0.01068/0.08825 meV`, and E1-H1 block RMS `0.000212/0.000900 meV` (D values dimensionless). Thus even the weak TR-even semimetal is not the radial-converged unrestricted axial minimum of the unscreened exchange-only scaffold.

Matched-resolution cutoff jobs continued the polarized pair on both sides of `kmax=0.14,nr=96,nphi=1200`. At `kmax=0.12,nr=82,nphi=1024`, Slurm `184413` converges to `F=-0.002502449 meV nm^-2`, gap `-7.170047 meV`, order `0.174165 meV`, and `|sigma_z(E1)|=0.992945`, but fails `F`, gap, total-`D`, and total-`Sigma` gates (`|Delta F|=5.67e-4 meV nm^-2`, gap change `1.54 meV`, `D/Sigma` RMS `0.882/2.73 meV`). At `kmax=0.18,nr=124,nphi=1552`, Slurm `184414` converges independently from endpoint-held and linearly tapered outer seeds to `F=-0.002809108`, gap `-10.132683`, order `0.118404 meV`, and `|sigma_z(E1)|=0.773185`; it fails the predeclared branch-identity gate and the common-window `F/gap/D/Sigma` gates. Its `k>0.14` globally weighted outer response is `D=0.63659` and `Sigma=3.92081 meV`, far above `0.01/0.05 meV`. Both jobs therefore terminate fail-closed with exit 3: the current bare-exchange active-quartet polarized branch is not cutoff closed even though it is balanced-mesh converged at fixed `kmax=0.14`.

For non-axisymmetric stability work, `CoRotatingHarmonicFockOperator` implements the exact finite-`nphi` uniform-dielectric linear block in the convention `D_tilde(phi)=sum_m exp(+i m phi) D_m` and `K_m=(1/nphi)sum_l exp(+i m theta_l) C_l`. G0 tests cover the frozen pre-change `m=0` tensor/action, general physical `m/-m` pairs against the full two-dimensional action and Fock energy, the self-conjugate Nyquist channel, weighted self-adjointness, aliases, and `nphi=1200` negative-mode stability.

Exact off-axis microscopic frames generally require the coupled action `Sigma_m=sum_q K_mq[D_q]`. `direct_full_2d_co_rotating_fock_modes` and `direct_full_2d_co_rotating_fock_mode_contributions` now provide a direct projected full-2D oracle for this identity. A deliberately non-axial toy frame confirms nonzero off-diagonal mode coupling and reconstructs the full self-energy/energy to `2e-12`; an axial control is mode diagonal. Reduced faithful N95 checks at fixed ordinary-electron `mu_el=113.8688022886634 meV`, `T=1.4 K`, and unscaled `epsilon_r=15` close the coupled reconstruction below `6e-16 meV`. The C4-only `nphi=4` sample is exactly mode diagonal, while generic `nphi=8/16` samples show off-diagonal contribution fractions `1.9e-4` to `2.4e-4`. For the tested normal density the diagonal axial closure differs by `0.028-0.035%` in the full self-energy, `0.36-0.45%` in the E1-H1 block, and below `0.01%` in Fock energy. Separate `nphi=16` mode-content stress tests using the saved lower-metal and gapped densities give full-self-energy errors `0.0241%/0.0233%` and E1-H1 errors `0.0095%/0.0119%`; these do not rehabilitate the forensic ensembles. Exact decoupling is false, but large exact-frame coupling is excluded as the current mismatch mechanism on all reduced checkpoints tested.

The generic finite-temperature G1 layer now constructs the Fermi-matrix Frechet response `chi`, the SCF Jacobian `J=chi F`, and the self-adjoint stability kernel `C=(-chi)^(1/2)(-F)(-chi)^(1/2)`. Full-2D finite differences validate the nonzero-mode canonical response and show that its first-order chemical-potential shift vanishes. A log-domain sandwich bound controls two deep-occupied float64 square-root underflows without pretending that their stored zeros are mathematical zeros.

Slurm `185785` then materialized and fully diagonalized every `768 x 768` float64 `C` block for the saved `nr=48,nphi=596` polarized TR pair and `m=1..12`. All-column B/S/C parity, additive linearity, exhaustive dense TR intertwining, full +/- spectrum parity, reconstruction, and artifact gates pass. The largest dense value is the `m=1` plus result `0.998125921075798`, with runner diagnostic upper `0.998125921075886 < 0.999`; all 24 branch-mode blocks are labeled `finite_scaffold_dense_below_threshold`. This is exhaustive only inside the stated finite float64 `m=1..12` scaffold. It excludes `m=0`, `m>=13`, nonlinear angular basins, continuum and cutoff closure, screening, remote bands, and device physics, and therefore is not a physical unrestricted-stability claim.

For the TR-even branch comparison, the weak branch has total `D/Sigma` interpolation RMS values `0.289` and `0.530 meV` (E1-H1 blocks `0.0156` and `0.0120 meV`), reflecting its sharply occupied semimetal pockets; the strong branch has total `0.0107` and `0.125 meV` (E1-H1 blocks `0.00092` and `0.00271 meV`). This distinction must not be erased post hoc. In particular, no dielectric scan or claim of an equilibrium gapped EI is authorized by these results; carrier/electrostatic closure, untested angular sectors, nonlinear non-axisymmetric basins, and physical unrestricted stability remain open.

## Current and optical status

The authoritative bare current numerator is projected from the same microscopic source:

\[
\mathcal J_i^{(0)}({\bf k})=
\Phi_{\bf k}^\dagger
\frac{\partial H_{\rm Kane}}{\partial k_i}
\Phi_{\bf k}.
\]

The physical velocity divides by \(\hbar\), and current includes charge. A bare transition spectrum from the final matrix-HF eigenstates is an optical diagnostic only. A conserving conductivity additionally requires the self-consistent kernel \(\delta\Sigma/\delta P\), diamagnetic/contact terms, and Ward/f-sum checks.

## Validation gates

1. **Source gate:** fresh `split=0`; no digitized/scalar/fitted inputs; common bundle hash.
2. **Bundle gate:** Hermiticity, selected-spectrum parity, active/remote isolation, microscopic normalization, time reversal.
3. **Vertex gate:** reciprocity, positivity, local density parity, random local `U(2)_E x U(2)_H` covariance.
4. **Interaction gate:** Hermiticity, analytic uniform-dielectric and delta-layer limits, mesh-derived Coulomb self cell, finite-difference `dE/dD=Sigma`.
5. **SCF gate:** interaction-off normal reference, scalar/two-level limit, seed/continuation stability, reference and energy consistency.
6. **Current gate:** direct full-Kane projection versus covariant derivative of the same Kane4 bundle.
7. **Experiment gate:** only after the earlier gates, compare spectra, density, temperature, field, capacitance, and optical response jointly. Fig. 2 is validation-only.

## Current implementation boundary

`mean_field.systems.inas_gasb` currently provides the basis/TR/axial contract, source-hashed Kane4 bundles, z-resolved vertices, mesh-derived circular self cells, projected exchange, differential periodic/open/ideal-one-sided-gate Hartree operators, builder-attested common open and one-sided-gate Hartree/Fock response builders, carrier-closure diagnostics, matrix entropy/free energy, symmetry projection, and a source-bound generic-HF-engine adapter. The shared engine accepts a custom convergence metric. Remaining production gates include a measured Wafer-B z stack and front-gate coordinate, a fully split-zero fixed-voltage Kane-Poisson source, external charge/work-function terms, fixed-gate envelope feedback, controlled BZ/UV closure, paired-state screening, unrestricted symmetry stability, and conserving optical response. Current results must remain labeled frozen-potential Phase-1 scaffolds.
