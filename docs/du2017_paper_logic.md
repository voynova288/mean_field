# Du et al. 2017: logic and numerical-interface audit

This note separates what the paper explicitly connects from what a reproduction must infer. It is based on the main text, Methods, Supplementary Note 2, Supplementary Note 6, Supplementary Figs. 5, 12, and 13, and the cited Li-Yang-Chang Kane/Poisson method.

## 1. Three distinct evidence and calculation chains

### A. Fig. 2: two-band excitonic-insulator mean field

The main text introduces a two-band electron-hole Hamiltonian with point-layer interactions

```text
V_ee(q) = V_hh(q) = e^2/(2 epsilon |q|)
V_eh(q) = V_ee(q) exp(-q d)
```

and says that a mean-field calculation produces `Delta(k)`, the pair-breaking energy `E(k)`, and the JDOS. Supplementary Note 2 gives the generalized BCS equations for `Delta`, `xi`, and `eta`.

However, neither the main text nor Supplementary Note 2 specifies:

- the numerical `epsilon_a(k)` and `epsilon_b(k)` used for Fig. 2;
- a number equation or a requirement that the interacting BCS occupations integrate to the experimentally estimated `n0`;
- the numerical chemical potential, bare band overlap, momentum cutoff, or solver details;
- that the Fig. 2 two-band dispersions are exported from Supplementary Note 6's 8-band calculation.

The effective masses `m_e≈0.032 m0` and `m_h≈0.136 m0` are stated immediately before the theoretical-model section, but the text explicitly uses them to estimate the Bohr radius. It does not unambiguously state that they are the Fig. 2 band masses.

### B. Fig. 3 and Supplementary Fig. 5: transport/capacitance gap versus temperature

The measured activation gap is about 2 meV. Capacitance data show the gap beginning to collapse around 5-6 K and disappearing near 10 K. Supplementary Fig. 5 plots the calculated maximum of `Delta(k)` versus temperature, starting near 27 K in energy units and vanishing near 10 K.

The capacitance experiments are on devices from Wafer A and Wafer C. Fig. 2 THz spectroscopy is on Wafer B. The paper does not document whether Supplementary Fig. 5 and Fig. 2 use identical single-particle dispersions or layer geometry.

### C. Fig. 4 and Supplementary Note 6: 8-band Kane/Poisson hybridization diagnostic

Supplementary Note 6 defines a separate 8-band Kane Schrödinger-Poisson calculation. Its role in the main text is the in-plane-field hybridization diagnostic in Fig. 4f.

The method:

- expands envelope functions in plane waves over total length `L`;
- builds electron density from occupied Gamma6 components;
- builds hole density from unoccupied valence components;
- fixes the Fermi level by integrated charge neutrality;
- solves Poisson self-consistently.

No target density or gate boundary condition is specified. The natural reading, consistent with the cited method, is a neutral periodic supercell whose carrier density emerges from the broken-gap overlap. Thus a fixed target `n_e=n_h=5.5e10 cm^-2` is an additional reproduction constraint, not the only paper-defined Kane/Poisson procedure.

## 2. Meaning of the experimental CNP density

Methods and Supplementary Figs. 12-13 estimate `n0` from magnetotransport, capacitance integration, and extrapolated gate-density rates. At `Vb=0`, `n0≈5.5e10 cm^-2` with uncertainty `5e9 cm^-2`.

This is an experimental per-layer equilibrium-density estimate, not the sum `n_e+n_h`. The paper does not give an equation that forces the occupations of the separate Supplementary-Note-2 calculation to reproduce this value. Therefore the density inferred from digitized Fig. 2 (`~9.1e9 cm^-2` if its minimum is interpreted as the standard spin-degenerate 2D `kF`) demonstrates an undocumented mapping between experiment and theory, but by itself does not prove an algebraic contradiction.

Numerically, the Fig. 2 minimum `0.02354 nm^-1` tracks the inverse intralayer spacing `sqrt(n0)=0.02345 nm^-1`, a distance convention already used in the 2015 preprint and thesis. It does not equal the standard circular-pocket Fermi momentum `sqrt(2*pi*n0)=0.05879 nm^-1`. The safe conclusion is therefore “inverse-spacing or undocumented model momentum scale,” not a proven missing-`sqrt(2*pi)` plotting error. A standard Fermi-momentum interpretation would require an unsupported effective degeneracy near `4*pi`.

## 3. Reproduced periodic Kane/Poisson checkpoint

A paper-local active-subspace implementation of Supplementary Note 6 Eq. (5) was constructed using kdotpy wavefunctions:

- occupied Gamma6 components contribute electron density;
- unoccupied Gamma8/Gamma7 components contribute hole density;
- the Fermi level is solved from charge neutrality at 0.1 K;
- a variable-dielectric periodic Poisson equation is solved with zero mean potential.

For Wafer B geometry and the documented Kane parameters, the converged checkpoint is:

- `n_e=n_h=4.04117e10 cm^-2`;
- charge imbalance below `3e-4 cm^-2`;
- electron/hole centroid separation `6.0994 nm`;
- periodic potential peak-to-peak `2.33814 meV`;
- second-iteration fixed-point residual `7.1e-6 meV`;
- directly integrated Poisson energy `0.000284180 meV nm^-2 = 0.703212 meV` per pair;
- fixed-profile pair-chemical shift `1.406424 meV`, equal to the density-weighted layer-potential difference in magnitude.

The smaller potential than the 8-14 meV cited in Li-Yang-Chang is physically consistent with the lower carrier density here; the cited example has about `1.61e11 cm^-2`.

Artifacts:

- Slurm job `172408`
- `runs/kdotpy_nc_ei_stageB_20260711_note6_iter02_slurm/`
- `runs/kdotpy_nc_ei_stageB_20260711_note6_iter02_slurm/note6_analysis/metrics.json`
- `runs/du2017_capacitor_audit_20260712/`

## 4. Controlled E1/H1 downfolding result

The full 1056-dimensional kdotpy Hamiltonian was reconstructed from the production XML and periodic potential. At all 61 radial momenta its 28 computed eigenvalues reproduce the saved XML to approximately `3e-12 meV`.

The Kane-parameter plumbing was also rechecked against Li-Yang-Chang 2009. Supplementary Table 1 supplies already modified `A_c` and `gamma_i`, while `E_p=22.5 eV` comes from the cited method paper. kdotpy consumes `F`, `gamma_i`, and `P` directly and only renormalizes when an explicit `renormalize_*` material option is present; none was used. The conversion `A_c=hbarm0*(2F+1)` is therefore direct, not double-renormalized. Li-Yang-Chang explicitly neglect strain because the mismatch is below 1%, consistent with the zero-strain calculation. Missing unpublished parameters remain a caveat, but double renormalization is not the source of the observed crossing.

A fixed-k=0 four-state E1/H1 reference subspace was then mapped to each exact active quartet using the unitary polar/Löwdin overlap map. This retains the complete 4x4 E1/H1 Hamiltonian, including off-diagonal hybridization, and reports the squared principal-angle cosines rather than normalizing away leakage.

Core results:

- minimum subspace completeness for `k<=0.06 nm^-1`: `0.9421`;
- global minimum completeness through `0.12 nm^-1`: `0.8575`;
- exact selected-spectrum reconstruction error: below `6e-15 meV`;
- low-k E1 diagonal curvature: `-85.75 meV nm^2`;
- low-k H1 hole curvature: `1365.81 meV nm^2`;
- diabatic E1/H1 crossing: `k=0.1174 nm^-1`;
- at the natural SP `kF=0.05039 nm^-1`, the diagonal E1/H1 separation is `11.44 meV` and hybridization RMS is `0.0939 meV`.

Thus the negative E1 curvature is not caused by atomic-character switching or a defective scalar projector. It survives a controlled full-Hamiltonian downfolding in a well-conditioned low-energy subspace. Supplementary Note 6 therefore does not supply the simple semimetallic two-band input needed by Fig. 2 at the natural SP density.

Artifacts: Slurm jobs `172544`, `172550`; `results/nc_ei_stageB_kane_ep225_note6_lowdin_full_20260711/`.

## 5. Temperature-ensemble audit

Two explicit temperature continuations quantify the missing ensemble/electrostatic convention:

1. **Frozen background**: freeze the 0.1 K balanced-root bands, chemical potential, and relative alignment, then solve only the three Note-2 equations. This loses charge balance immediately and the fixed-background normal instability ends near `4.63 K`.
2. **Reoptimized CNP diagnostic**: enforce `n_e=n_h=4.04117e10 cm^-2` at every temperature by numerically varying the two effective channels. The finite-Delta branch converges through 6 K, is numerically disconnected at 7-9 K, and falls onto the normal root by 9.5 K. Because the relative channel is non-identifiable on the low-T unblocked plateau and no electrostatic functional is supplied, this path is not a uniquely defined physical ensemble. The separately constructed balanced normal root has `lambda_max<1` at all sampled temperatures, so the branch is not selected by a documented continuous normal-state instability.

Neither path is a documented reproduction of Supplementary Fig. 5. The first does not remain at experimental CNP; the second changes an electrostatic alignment without assigning its capacitor/Poisson energy. A thermodynamic temperature scan requires a coupled electrostatic functional or an author-provided ensemble convention.

Artifacts:

- `results/nc_ei_stageB_kane_ep225_note6_sp_temperature_scan_20260711/`
- `results/nc_ei_stageB_kane_ep225_note6_sp_balanced_temperature_scan_20260711/`

## 6. Finite-temperature equation audit

Before thermodynamics, Supplementary Note 2 already has a chemical-potential ambiguity. Eq. (1) subtracts the same `mu` from both electron and hole diagonal energies, which in the doubled pair convention implies `xi_bare=epsilon_P-2*mu` and `eta_bare=epsilon_M`. The later printed equations instead subtract one `mu` from both `epsilon_P` and `epsilon_M`. These can only be reconciled through hidden redefinitions (or `mu=0`), so the current solver's `literal_paper` mode follows the later printed equations but is not labeled as uniquely derived from Eq. (1).

The corrected core now uses explicit `mu_sum=mu_a+mu_b` and `mu_diff=mu_a-mu_b`. Slurm `176495` proves that the old and explicit-channel periodic-SP roots agree in `Delta`, `xi`, and `E_pair` to about `1e-8 meV`, so the reported `3.721 meV` gap and `7.732 meV` k≈0 energy were not artifacts of the single-`mu` naming. However, `eta` and `mu_diff` can shift oppositely while `eta+mu_diff` stays invariant. The 0.1 K root has blocking margin `0.293 meV≈34 kBT`; on this unblocked plateau charge balance does not uniquely determine the relative channel. The legacy `eps_M_offset` is therefore one representative of a thermally frozen manifold, not a measured electrostatic alignment.

The literal finite-temperature `xi` equation printed in Supplementary Note 2 is not jointly integrable with the printed `Delta` equation in the natural BdG/Hartree-Fock sense. With

```text
F = 1 - f((E-eta)/2) - f((E+eta)/2),
```

the cross-derivative mismatch is

```text
partial_xi[-(Delta/E)F] - partial_Delta[(1-xi/E)F]
= -(Delta/E) partial_E F,
```

which is generically nonzero at finite temperature. The standard thermodynamic BdG bracket is `1-(xi/E)F`, differing only when `F != 1`. This may be a parenthesis/algebra error in the supplement, but the publication does not provide code sufficient to decide whether Supplementary Fig. 5 used the printed or thermodynamic form.

Consequences:

- the previously found disconnected finite-temperature roots cannot be ranked with a standard grand potential while retaining the literal printed `xi` equation;
- at zero temperature in an unblocked regime the two forms coincide, so the converged 0.1 K Stage-B checkpoint is largely unaffected;
- `src/ei_mf/thermodynamics.py` records the integrability diagnostic and the alternative thermodynamic functional, but the literal equation remains the default.

For the same frozen external bands, alignment, and chemical potential, the thermodynamically integrable diagnostic finds only one converged nonzero root among the sampled temperatures. At `0.1 K` it has

```text
Omega_EI - Omega_normal = +0.0110654 meV nm^-2,
```

so it is metastable and not thermodynamically selected. There is no sampled free-energy crossing. This diagnostic is not a paper reproduction, but it excludes the proposal that a missing standard BdG free-energy comparison would rescue the current SP-driven branch for the same frozen external bands and shifts. It does not establish the authors' hidden ensemble or rule out different wafer/band inputs.

Artifact: Slurm job `172580`; `results/nc_ei_stageB_kane_ep225_note6_sp_thermodynamic_temperature_scan_20260711/`.

## 7. Numerical convention red-team

The 2D Fourier prefactor and radial measure were independently rechecked: `V(q)=2*pi*[e^2/(4*pi*eps0)]/(eps_r*q)`, `K=(1/2*pi) int dphi V`, and the radial weights implement `int k dk/(2*pi)`. Absolute-oracle tests now guard the prefactor and disk measure.

A Slurm convergence scan over `nk=200..800`, `nphi=180..1440`, and `q_floor=(0.125..1.0)*dk` changes the unscaled Stage-A minimum by at most `0.08 meV`. This is far below the multi-meV Stage-B discrepancy. Slurm `176451` passed all `36` tests. Artifact: `runs/nc_ei_logic_redteam_20260712/`, Slurm `176450`.

The JDOS required a separate observable-level correction. The old one-delta-per-radial-cell quadrature produced `80` maxima although `E_pair(k)` has one internal stationary point. Conservative PCHIP-refined annular integration removes this alias comb and is stable across refinement, origin-boundary conventions, and independent `nk=120/180/240` roots. At the production `eta=0.1 meV` there is no line-B local maximum; the old line-B `pass` is withdrawn. For `eta<=0.05 meV` narrow maxima reappear, but their positions and multiplicities vary strongly across `nk=120/180/240`; only the high-energy `E(k≈0)` scale—not a robust JDOS peak—is currently supported. Artifact: `runs/nc_ei_jdos_quadrature_audit_20260712/`.

Paper-vector digitization, inverse plotting, target-derived forward JDOS, target-window peak selection, and paper-calibrated reconstruction are now strictly forbidden and their result directories have been deleted. They are not calculations and must never be shown as reconstructed results. If Fig. 2 must be viewed, use a direct crop from the published PDF only. Any future `E_pair(k)`, `Delta(k)`, JDOS, or absorption figure must be generated solely from an independently specified Hamiltonian and self-consistent saved state, with no paper curve entering equations, parameters, branch selection, broadening selection, coordinates, or postprocessing. Cleanup receipt: `results/du2017_inas_gasb_excitonic_insulator/TARGET_DERIVED_RECONSTRUCTION_CLEANUP_20260807.json`.

## 8. Independent Zhu1995 backbone benchmark

The cited Zhu et al. radial equations were independently implemented in their native spinless atomic units with `k=tan(beta)` Gaussian quadrature and cell-integrated logarithmic Coulomb singularities. Slurm `176609` reproduces every public textual Fig. 2 checkpoint at `rs=2.66/5.90,d=1`; Slurm `176610` passes all `46` EI tests. This excludes the radial measure, angular Coulomb average, and Zhu Eq.-4–6 factors as explanations for the Du mismatch.

A non-predictive Zhu→Du audit (`176619`) gives a sharper split result. With the reported masses/density/geometry, the equivalent Zhu parameters are `rs=2.2206`, `d/a*=0.6527`, and the inverse-spacing momentum unit is `0.026039 nm^-1`. The nearest scanned case independently fits `0.026021 nm^-1`, within `0.071%` of the inverse-spacing value. But it yields all-curve RMSE `1.084 meV` for `E` and `1.013 meV` for `Delta`; unconstrained `rs,d` fitting reaches scan boundaries and retains `1.872 meV` held-out `E` error. Thus Zhu is a credible solver ancestor and momentum-conversion explanation, not a complete quantitative author model.

## 9. Pair-curvature channel closure

The phrase `C*k^2` is ambiguous. In an ordinary-electron two-band Hamiltonian, a common identity-channel curvature changes `eta_charge=E_c+E_v-2*mu_el` but cancels from `xi_pair=E_c-E_v`; only a `tau_z*k^2` mass/detuning term changes the balanced pair spectrum. The latter is already present either through the mass-derived coefficient `1.4707651566 eV nm^2` or through the direct Kane E1/H1 dispersions. The restored senior bundle's additional `16.60946 eV nm^2` correction acts in this detuning channel and is explicitly calibrated alongside Gaussian/ring, vertex, chemical-potential, and Coulomb-channel knobs; it is not found in the Du paper, supplement, thesis, or cited Kane method.

Unit-Coulomb hard-cutoff and fixed-pair-`mu` scans cannot simultaneously match `E(0)`, `E_min`, and `Delta_max`; the direct uncalibrated Kane route likewise gives `E(0)=8.605`, `E_min=3.627`, and `Delta_max=3.741 meV`. A target-free nested-window fit of the saved split-zero Löwdin traces gives a stable Kane pair curvature `c_xi=1.283–1.290 eV nm^2` plus a subleading `k^4` correction already contained in the direct parent. Therefore no currently source-backed extra quadratic term remains to test. Full derivation and source hashes: `docs/debug/du2017_pair_curvature_channel_audit_20260807.md`; numerical trace receipt: `runs/du2017_lowdin_trace_curvature_20260807/SCIENTIFIC_STATUS.md`.

## 10. Consequences for the next calculations

- A density scan with the unscreened point-layer kernel is not a paper-level prediction. The measured density trend is attributed to increased screening, but no density-dependent screening model is specified.
- The balanced `fixed_cnp_root` remains useful as a stringent numerical and physical audit, but it should not be described as the unique author procedure.
- A paper-level temperature curve requires the authors' actual two-band input/equation convention or code/data; additional continuation of the Note-6-driven branch cannot resolve the mismatch.
- If further independent work is desired, the next defensible route is a separate paper-mass/two-band thermodynamic model explicitly classified as a model study, not as an inferred Note-6 interface.

A consolidated authority table covering all published, inherited, local, and rejected numerical conventions is maintained at `docs/debug/du2017_numerical_source_lineage_table_20260807.md`. Values from different evidence labels must not be combined into a synthetic “author model.”
