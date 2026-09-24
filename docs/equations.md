# InAs/GaSb EI mean-field equation conventions

This note records the formula provenance for `src/ei_mf`, the reproduction module for Du et al., Nature Communications 8, 1971 (2017), `reference/s41467-017-01988-1.pdf`.

## Sources

- Main paper: `reference/s41467-017-01988-1.pdf`.
- Supplementary information: `reference/supplement/41467_2017_1988_MOESM1_ESM.pdf`, Supplementary Note 2, pages 16-17.
- Lingjie Du PhD thesis (2016): `reference/du_thesis_2016.pdf`, thesis pages 37-40 and Sec. 6.5/Fig. 6.22.
- Zhu et al. Keldysh-Kozlov implementation: `reference/zhu_littlewood_1995_exciton_droplet.pdf`, Eqs. 4-6.
- Local implementation plan: `plan/NC_EI_mean_field_code_agent_workdoc_v2.md`.

## Paper convention

The code follows the paper/Supplementary Note 2 convention

```text
E_pair(k)^2 = Delta_order(k)^2 + xi(k)^2
```

so the implemented pair-breaking excitation is

```python
E_pair = sqrt(xi**2 + Delta_order**2)
```

Do **not** multiply this by two.  If a later implementation uses ordinary superconducting half-quasiparticle energies internally, it must explicitly convert back to this paper convention before saving or plotting.

## Du-thesis zero-temperature backbone

The thesis explicitly prints the isotropic parabolic point-layer model used as the methodological ancestor of Fig. 6.22:

```text
V_ee(q)=V_hh(q)=2*pi/q,
V_eh(q)=2*pi*exp(-q*d)/q,
Delta_i = sum_j w_j V_eh(i,j) Delta_j/E_j,
xi_i = epsilon_pair_i-mu-sum_j w_j V_hh(i,j) [1-xi_j/E_j],
E_i^2 = xi_i^2+Delta_i^2,
n = sum_j w_j [1-xi_j/E_j]/2.
```

`src/ei_mf/zhu1995.py` implements this spinless effective-atomic-unit system with `k=tan(beta)` Gaussian quadrature and explicit diagonal logarithmic-cell integration. The mass unit is `2*m_reduced`; consequently its atomic length is one half of the conventional reduced-mass exciton Bohr radius.

The thesis has two internal printing issues that prevent blind literal transcription: its pair kinetic term repeats `E_h+E_h`, where the physically consistent expression is `E_e+E_h`, and its first-quantized electron-hole sign conflicts with the attractive sign in the second-quantized Hamiltonian. The source-corrected executable interpretation follows the cited Zhu equations.

For the quoted particle distance, Du defines `L=2*r_avg` and `1=n*pi*r_avg^2`; therefore

```text
n = 4/(pi*L^2),
```

not `1/L^2`. With `L=50 nm`, `n=5.093e10 cm^-2`. The physical effective-a.u. momentum is `k_au/a_Zhu`; an additional division by `sqrt(2*pi)` is retained only as a forensic `sqrt(n)` axis diagnostic, never as a physical unit conversion.

The quadratic pair dispersion is already intrinsic to this source model:

```text
c = hbar^2/(2*m0) * (1/m_e + 1/m_h),
epsilon_pair(k) = c*k^2.
```

For the Du2017 masses `m_e=0.032 m0`, `m_h=0.136 m0`, this gives `c=1.4707651566 eV nm^2`. It must not be confused with the senior calibrated bundle's fitted additional `16.60946 eV nm^2` correction. Adding that correction on top of the physical kinetic dispersion would be an unsupported double modification.

`mapped_beta_grid(..., k_max=...)` and `solve_zhu1995_fixed_rs(..., k_max=...)` expose a finite hard momentum domain only for explicit historical forensics; the default remains the source `k=tan(beta)` infinite domain. `solve_zhu1995_fixed_mu` implements the thesis gap/exchange equations with externally fixed `mu`, making density an output as in the underdetermined Supplementary-Note-2 setup. A cutoff or `mu` selected by Fig. 2 agreement must be labelled target-informed/uncontrolled and cannot be promoted as a physical regulator or parameter prediction.

## Supplementary Note 2 variables

The supplementary equations define

```text
eps_P(k) = eps_a(k) + eps_b(k)
eps_M(k) = eps_a(k) - eps_b(k)
E(k)^2 = Delta(k)^2 + xi(k)^2
```

where `eps_a` is the conduction electron energy and `eps_b` is the valence-band **hole** energy.  A raw valence-electron energy from an 8-band calculation must not be used as `eps_b` unless it has first been converted to the hole-energy convention and documented.

The band CSV interface in `src/ei_mf/bands.py` therefore requires

```text
k_nm_inv,epsilon_a_meV,epsilon_b_meV
```

with `epsilon_b_meV` already in the hole convention.

## Generalized BCS fixed-point equations

This entire radial scalar interface is explicitly isolated as
`physics_mode=paper_literal_legacy` with
`physics_status=source_literal_with_known_internal_inconsistencies`. It is a
historical diagnostic, not the production physics mode. The production mode is
`derived_ordinary_electron_hf_bcs` in
`mean_field.systems.inas_gasb`, which iterates the matrix
`D=P-diag(0_E1,1_H1)` on an explicit 2D mesh.

The update kernel in `src/ei_mf/gap_solver_generalized.py` implements the Supplementary Note 2 equations in radial discretized form:

```text
Delta_i = sum_j w_j K_ab(i,j) Delta_j/E_j
          * [1 - f((E_j - eta_j)/2) - f((E_j + eta_j)/2)]

xi_i = eps_P_i - mu
       - sum_j w_j K_aa(i,j) [1 - xi_j/E_j]
         * [1 - f((E_j - eta_j)/2) - f((E_j + eta_j)/2)]

eta_i = eps_M_i - mu
        - sum_j w_j K_ab(i,j)
          * [1 - f((E_j - eta_j)/2) + f((E_j + eta_j)/2)]
```

Here `w_j` are the two-dimensional radial integration weights for `int d^2k/(2*pi)^2`, and `K_ab/K_aa` are angular averages of the positive Coulomb-kernel magnitudes.  The electron-hole attraction sign is not stored inside `K_ab`.

For unscreened `1/q`, fixed `q_floor` is a legacy regulator, not part of the paper model. `angular_averaged_kernel_cell_integrated()` instead uses the exact off-diagonal elliptic average and integrates the logarithmic diagonal over its annular `k dk/(2*pi)` cell, as in the source-corrected Du-thesis/Zhu treatment. It evaluates `(exp(-q*d)-1)/q` for point layers and `(F_xy(q)-1)/q` for source-derived Kane profiles. This target-independent quadrature must be converged in `n_k` and quadrature order, not selected by Fig. 2 agreement.

The default point-layer model uses `F_aa(q)=1` and `F_ab(q)=exp(-q*d)`. For source-derived profiles, load `KaneLayerProfiles.from_npz(...)` from `ei_mf.wavefunction_form_factor` and pass it explicitly as `form_factor_profiles=` to the Coulomb-kernel API. This replaces the point-layer factors without an adjustable scale by Kane-wavefunction projector form factors

```text
F_xy(q) = int dz dz' rho_x(z) rho_y(z') exp(-q |z-z'|)
```

where `rho_e` is the normalized active-quartet Gamma6 projector density and `rho_h` the normalized Gamma8-heavy-hole projector density. The source wavefunction files and profile NPZ are recorded in result metadata.

The stricter momentum-resolved diagnostic instead defines rank-two E1/H1 projectors in one Gamma-point reference frame, transports the complete active quartet by a polar/Lowdin isometry, and forms normalized full-eight-orbital pair traces `rho_e(k,z)` and `rho_h(k,z)`. Because diagonal endpoint densities are not microscopic k-to-k transition densities, the scalar e-h form factor is explicitly the Hermitian approximation

```text
F_eh^dens(k,k',q) = 1/2 [rho_e(k) E_q rho_h(k')
                            + rho_e(k') E_q rho_h(k)]
```

with analogous symmetric `F_ee` and `F_hh`. The single same-species kernel required by the balanced scalar Note-2 interface is typed as

```text
K_aa^balanced = (K_ee + K_hh) / 2.
```

This preserves a symmetric pair-space operator and reduces exactly to the frozen-profile kernel when the profiles are k independent. It is a declared endpoint-density closure, not the microscopic projected Kane vertex, which would require complex transition densities `psi_s(k)^* psi_s(k')`, and it is not attributed to the Du authors.

The maintained `ei_mf.gap_solver_generalized` API separates the corresponding fixed-point, fixed-band-root, fixed-density, and explicit-channel CNP-root problems through `solve_generalized_fixed_mu`, `solve_generalized_fixed_mu_root`, `solve_generalized_fixed_density`, and `solve_generalized_cnp_root`. The former `python -m ei_mf.cli` plotting/scan/artifact workflow is archived and is not a maintained command surface.

`solve_generalized_fixed_mu_root` solves only the three printed Supplementary-Note-2 equations and is the closest **literal-publication** implementation of that note, which gives a chemical potential but no number equation. It is not yet a Hamiltonian-derived thermodynamic convention because the supplement's chemical-potential algebra is internally ambiguous.

From Eq. (1), if the electron and hole diagonal energies are `A=epsilon_a-mu_a` and `B=epsilon_b-mu_b`, the doubled pair convention requires

```text
xi_bare  = A+B = epsilon_P - (mu_a+mu_b)
eta_bare = A-B = epsilon_M - (mu_a-mu_b).
```

For the common `mu_a=mu_b=mu` printed in Eq. (1), this becomes `xi_bare=epsilon_P-2*mu` and `eta_bare=epsilon_M`. The later printed equations instead subtract the same single `mu` from both `epsilon_P` and `epsilon_M`; algebraically that corresponds to `mu_a=mu, mu_b=0`, contradicting Eq. (1), unless `mu=0` or hidden redefinitions are used. The current solver intentionally preserves the later printed equations and labels them literal; it must not be described as the unique convention derived from Eq. (1).

The Du thesis resolves the role of one symbol but not the full inconsistency. It defines the reference vacuum as valence full/conduction empty and minimizes `f=<h_e-h>-mu<n>` with `<n>=sum_k v_k^2`, so its `mu` is a pair-density Lagrange multiplier. It is not the absolute Kane--Poisson ordinary-electron Fermi energy. For ordinary-electron conduction/valence energies `E_c,E_v` and electronic chemical potential `mu_el`, the electron/hole excitation energies are `A=E_c-mu_el` and `B=mu_el-E_v`. Hence

```text
xi_pair = A+B = E_c-E_v,
eta_charge = A-B = E_c+E_v-2*mu_el.
```

Thus the Kane--Poisson `mu_el` cancels from the pair-energy channel and enters only the charge-asymmetry/blocking channel. Directly inserting `113.8688022886634 meV` as the scalar BCS pair `mu` is a category error. The matrix fixed-mu path instead evaluates the original ordinary-electron Hamiltonian at that same-gauge `mu_el` and is a distinct grand-canonical model, not a literal transcription of Supplementary Note 2.

`fixed_cnp_root` adds two reproduction constraints, `n_e=n0` and `n_h=n0`, and now solves explicit `mu_sum` and `mu_diff` channels. Legacy `(mu, eps_M_offset)` restart values map to `mu_sum=mu` and `mu_diff=mu-eps_M_offset`. On a fully gapped unblocked branch, both densities are equal and independent of the relative channel to exponential accuracy; the eta equation fixes only `eta+mu_diff`. Therefore `mu_diff`/the legacy alignment is not uniquely selected by charge balance on this plateau. The root remains a useful two-channel equation/density audit, but selecting a physical relative alignment requires an author-defined chemical-potential convention or an electrostatic/capacitor functional.

For `fixed_density`, the code evaluates electron and hole occupations separately:

```text
u^2 = (1 + xi/E)/2,  v^2 = (1 - xi/E)/2
f_minus = f((E-eta)/2),  f_plus = f((E+eta)/2)
n_e(k) = u^2 f_plus  + v^2 (1-f_minus)
n_h(k) = u^2 f_minus + v^2 (1-f_plus)
n_pair = (n_e+n_h)/2
```

including the spin degeneracy and radial `d^2k/(2*pi)^2` weights.  On a valid CNP branch, `n_e=n_h=n_pair`; therefore every production result reports the charge-imbalance checkpoint `n_e-n_h` in addition to the average density.  At balanced T=0 this reduces to `g_s * int v_k^2`.  A paper-quality Stage-B claim still requires both a documented 8-band/self-consistent band input and simultaneous density/charge balance.

For a broken-gap input, scalar continuation can jump between electron- and hole-polarized normal branches.  The constrained `fixed_cnp_root` mode therefore solves the full unknown vector

```text
(Delta[0:N], xi[0:N], eta[0:N], mu_sum, mu_diff)
```

with a matrix-free Krylov root. The first `3N` residuals are the three Supplementary-Note-2 equations; the final two are `(n_e-n0)/n0` and `(n_h-n0)/n0`. `eps_M` is stored in the input-band frame, while legacy `eps_M_offset=mu_sum-mu_diff` is retained only for restart/provenance compatibility. A positive blocking margin warns that the relative channel is thermally frozen and not identifiable from charge balance. A 1 K root is used only as a continuation seed for the 0.1 K solve. The normal branch is audited separately through the largest eigenvalue of the linearized gap kernel.

## Minimal Note-6-to-scalar alignment gate

In the fixed k=0 E1/H1 Löwdin reference basis, define the diabatic signed normal detuning

```text
g(k) = Tr[H_E1,E1(k)]/2 - Tr[H_H1,H1(k)]/2.
```

An energy-sorted direct gap cannot replace `g` once normal E1/H1 hybridization or pair splitting is present. A constant relative alignment is represented by

```text
H_normal -> H_normal + delta*(P_E1-P_H1)/2,
g -> g+delta.
```

The minimal scalar bridge would require

```text
xi_scalar(k) = g(k)+delta,
E_bridge(k) = sqrt((g(k)+delta)^2+Delta_scalar(k)^2).
```

Here `delta` changes the xi/tau-z channel; it is not `mu_diff`, which changes eta. On the controlled split-zero Note-6 grid, the direct final-xi comparison has a `2.640 meV` minimax residual. Its decomposition is

```text
xi_scalar-g
= [(epsilon_P_scalar-mu_sum)-g]
+ [xi_scalar-(epsilon_P_scalar-mu_sum)].
```

The first bare-interface term has a smaller but nonzero `0.530 meV` minimax residual; the second scalar Fock correction varies from `-9.054` to `-4.834 meV`. Thus most total k dependence is interaction-generated. The gate excludes an exact constant mapping but does not exclude a self-consistently re-solved Kane-band HF theory.

The trace-reduced follow-up uses

```text
epsilon_a = Tr[H_E1,E1]/2,
epsilon_b = -Tr[H_H1,H1]/2,
```

then solves the same scalar generalized equations afresh. Its linearized normal-state test uses `D_j=w_j F_j/|xi_j|` and the symmetric operator `sqrt(D) K_ab sqrt(D)`, which is similar to the unsymmetrized gap map. At 0.1 K on the current discrete trace model, `lambda_max≈6.376>1`; this establishes pairing-map supercriticality, not a thermodynamic free-energy Hessian or full 4x4 stability.

## Pair-breaking absorption and minimal optical diagnostic

The THz process removes one electron-hole pair from the excitonic condensate and creates individual electron and hole final states at opposite in-plane momenta. Its golden-rule spectrum is

```text
Gamma(Omega) proportional to
 integral d^2k/(2*pi)^2 |M_eh(k)|^2 F(k)
 delta_gamma[Omega-E_pair(k)].
```

For an isotropic model this becomes

```text
Gamma(Omega) proportional to
 sum_{k_i:E_pair(k_i)=Omega}
 k_i |M_eh(k_i)|^2 F(k_i) / |dE_pair/dk|_{k_i}.
```

All roots, stationary points, and peak energies in this expression must come from the independently solved `E_pair(k)` on its declared physical momentum grid. Paper-vector digitization, inverse plotting, target-window peak selection, and fitted coordinate transforms are forbidden. If the published curve must be viewed, use a direct crop from the paper and do not convert it into numerical solver or plotting input. A correct 2D `k dk` endpoint is generally a finite step rather than a divergent peak; any claimed peak must be supported by the calculated stationary-point topology and the same-Hamiltonian matrix element.

For the diagnostic Nambu spinor `Psi_k=(a_k,b^dagger_-k)`, the doubled-paper Hamiltonian is

```text
h_k = [eta*tau_0 + xi*tau_z + Delta*tau_x]/2,
```

whose interbranch splitting is `E_pair`. One controlled but incomplete vertex candidate is the radial diagonal-band charge current

```text
J_r = diag(-v_e,+v_h)
 = (v_h-v_e)*tau_0/2 - (v_e+v_h)*tau_z/2.
```

Because the identity term does not connect the two quasiparticle branches and `|<+|tau_z|->|^2=Delta^2/E_pair^2`, the isotropic x-polarized angular-average shape is

```text
W_current = F*(d epsilon_P/dk)^2*Delta^2/(8*E_pair^2),
F = 1-f((E-eta)/2)-f((E+eta)/2).
```

The bare-bubble conductivity shape adds the on-shell `1/E_pair` Kubo factor. `src/ei_mf/pair_breaking.py` implements these weights and direct matrix-element tests. A split-zero full-Kane projection can supply an interband current numerator, but it may be combined only with transition energies and excitonic eigenstates from that same Hamiltonian; cross-model splicing is forbidden. A complete THz response still requires same-Hamiltonian remote-band dipoles, gauge-restoring vertex corrections, collective modes, tunneling current, disorder, finite-width geometry, and absolute prefactors.

## Finite-temperature thermodynamic consistency

Supplementary Note 2 prints the `xi` Hartree-Fock bracket as

```text
(1 - xi/E) * F,  F = 1 - f((E-eta)/2) - f((E+eta)/2).
```

Together with the printed `Delta` bracket `-(Delta/E)F`, these local derivatives fail the scalar-potential integrability condition at finite temperature:

```text
partial_xi[-(Delta/E)F] - partial_Delta[(1-xi/E)F]
= -(Delta/E) partial_E F.
```

The right-hand side is generically nonzero for `T>0`. Therefore no natural BdG/Hartree-Fock finite-temperature grand potential whose stationarity gives all three literal printed equations has been established. This is a fixed-`eta` local one-form test; unequal symmetric interaction kernels cannot cancel its local exterior derivative. It does not rule out artificial residual-squared objectives or integrating factors, which are not physical grand potentials. At zero temperature, or on any region where `F` is locally constant away from a blocking edge, the mismatch vanishes.

The thermodynamically integrable BdG bracket is instead

```text
1 - (xi/E) * F.
```

It follows from the local grand-potential term

```text
phi = xi + eta - E
      - 2 kBT log(1 + exp[-(E-eta)/(2 kBT)])
      - 2 kBT log(1 + exp[-(E+eta)/(2 kBT)]).
```

`src/ei_mf/thermodynamics.py` implements the cross-derivative diagnostic and this alternative grand potential. The numerical solver keeps `literal_paper` as the default and accepts `xi_thermal_convention="thermodynamic"` only as an explicit fixed-band/fixed-mu diagnostic. This may represent a missing parenthesis in the supplement, but it must not be silently relabeled as the authors' implementation without their code or data.

A free-energy comparison is valid only for roots with the same external `eps_P`, `eps_M`, and `mu`. Roots with different free `eps_M_offset` additionally require an electrostatic/capacitor energy; the current fixed-CNP alignment variable has no such thermodynamic term.

## Supplementary Note 6 periodic Schrödinger-Poisson convention

Supplementary Note 6 is a distinct 8-band calculation used explicitly for the in-plane-field hybridization analysis. It expands envelope functions in a periodic plane-wave supercell and defines

```text
rho_e(z) = -e sum_s int d^2k/(2*pi)^2 sum_{n=1,2} |phi_n^s(z)|^2 f(E_s)
rho_h(z) = +e sum_s int d^2k/(2*pi)^2 sum_{n=3,...,8} |phi_n^s(z)|^2 [1-f(E_s)]
```

with the Fermi level fixed by integrated charge neutrality. The paper does not impose a target density or gate boundary in this calculation. `src/ei_mf/kane_charge_density.py` implements this active-subspace density and a neutral, zero-mean periodic Poisson solve. It must not be replaced by a uniform full-spectrum reference-density subtraction.

## Stage-A reduction

For the current Stage-A reproduction, `eta=0` and `xi` is not Hartree/Fock-renormalized.  The parabolic detuning is

```text
xi0(k) = HBAR2_OVER_2M0_MEV_NM2 * (1/m_e + 1/m_h) * (k^2 - kF^2)
```

again with no extra factor `1/2`.  The gap equation uses the thermal factor

```text
tanh(E_pair / (4 k_B T))
```

not `tanh(E_pair / (2 k_B T))`.
