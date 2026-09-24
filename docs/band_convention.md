# NC EI Stage-B band-input convention

## Required CSV convention

`ei_mf` consumes radial band data with columns

```text
k_nm_inv,epsilon_a_meV,epsilon_b_meV
```

- `epsilon_a_meV` is the conduction-electron energy.
- `epsilon_b_meV` is the **hole** energy, not the valence-electron energy.
- For a valence-electron branch `E_v(k)` referenced to its Fermi crossing, the exported hole energy is `epsilon_b(k) = -E_v(k)`.

The generalized equations form

```text
epsilon_P = epsilon_a + epsilon_b
epsilon_M = epsilon_a - epsilon_b
```

and save `Delta_order`, `xi`, `eta`, and `E_pair=sqrt(xi^2+Delta_order^2)` in the paper convention.

## kdotpy 8-band Kane route

The current non-digitized audit uses kdotpy 1.4.1 with the Wafer-B active structure and thick numerical barriers:

```text
AlSb(23 nm) / InAs(11.5 nm) / GaSb(8 nm) / AlSb(23 nm)
```

Supplementary Note 6 is implemented as a neutral periodic active-subspace Schrödinger-Poisson problem, not through kdotpy's generic `cardens=0` classifier. Occupied Gamma6 components define electron density, unoccupied valence components define hole density, and the Fermi level is fixed by integrated charge neutrality. The production checkpoint is

```text
results/kdotpy_nc_ei_stageB_20260711_note6_iter02_slurm/
```

with `n_e=n_h=4.0411715653e10 cm^-2`, potential peak-to-peak `2.33814 meV`, and potential residual `7.07e-6 meV`.

### Controlled E1/H1 downfolding

Atomic Gamma6/Gamma8 weighting and maximum-character tracking are retained only as diagnostics. They are not projectors onto confined E1/H1 subbands.

`ei_mf.kane_downfolding.lowdin_effective_hamiltonian` instead uses exact full-Hamiltonian eigenvectors. If `V0` is the fixed k=0 E1/H1 reference quartet, `Vk` the exact active quartet, and `A=V0^dagger Vk`, it constructs

```text
G = A^dagger A
W = A G^(-1/2)
H_eff = W diag(E_active) W^dagger.
```

The eigenvalues of `G` are squared principal-angle cosines and expose remote-subspace leakage. The method is invariant under active-eigenvector phases/permutations, keeps full E1-H1 off-diagonal hybridization, and exactly retains the selected spectrum.

For the production Note-6 state, the full 1056-dimensional Hamiltonian reproduces the saved XML spectrum to `~3e-12 meV`. The downfolding has completeness at least `0.942` for `k<=0.06 nm^-1`, but yields E1 curvature `-85.75 meV nm^2`, an E1/H1 crossing at `0.1174 nm^-1`, and an `11.44 meV` diagonal separation at the natural SP `kF=0.05039 nm^-1`. Thus the Note-6 quartet does not furnish the simple Fig. 2 semimetallic two-band input.

The optical/current interface was independently checked after taking kdotpy `split→0`, adding negative k, and converging central derivative steps. Direct projection of the full 1056D `dH/dkx` gives lower/upper RMS `1.788105 meV nm` at k=0 and `2.942058 meV nm` at k=0.05. These agree with the fixed-reference Lowdin derivative to `1.4e-8` relative at k=0 and `0.6%` at k=0.05. The finite current channel is real within Note 6, but its direct gaps (`14.553/11.277 meV`) are incompatible with the separate scalar BCS transition energies (`7.732/3.721 meV`). No mixed optical spectrum is valid without a common author-defined Hamiltonian, pairing self-energy, basis map, and current vertex.

For scalar-interface tests, use the diabatic signed block trace

```text
g(k)=Tr[H_E1,E1(k)]/2-Tr[H_H1,H1(k)]/2,
```

not an energy-sorted direct gap. A relative operator `delta*(P_E1-P_H1)/2` changes `g` by exactly `delta`. On the complete split-zero grid, mapping this `g` directly to the saved interacting scalar `xi` has a `2.640 meV` minimax residual. Decomposing the mismatch shows a `0.530 meV` bare-band minimax residual plus a scalar Fock correction varying from `-9.054` to `-4.834 meV`. Thus a constant exact E1/H1 alignment is not the missing Note-6-to-Note-2 interface, but most total momentum dependence is interaction-generated rather than bare-band deformation. This delta acts in the xi/tau-z channel and is not `mu_diff`, which acts in eta.

A trace-reduced follow-up re-solves the generalized equations with `epsilon_a=Tr(H_E1)/2` and hole energy `epsilon_b=-Tr(H_H1)/2`. At the natural SP density it gives `E(k≈0)=7.003 meV`, `E_min=3.876 meV` at `0.05167 nm^-1`, and `Delta_max=4.027 meV`; radial `nk=120..240` variation is below about `0.04 meV`. This is a fresh scalar equation root, not a full 4×4 result: normal E1/H1 hybridization and spin splitting are discarded, the EI pairing map is scalar, and no Kane/Poisson feedback is performed. A staged normal root is supercritical under the same scalar gap map (`lambda_max≈6.376` at 0.1 K), but no free-energy or global-uniqueness claim follows.

Artifacts: Slurm jobs `172544`, `172550`, `177470`, `177971`, `177975`, `178385`; `results/nc_ei_stageB_kane_ep225_note6_lowdin_full_20260711/`, `runs/nc_ei_stageB_kane_ep225_note6_lowdin_full_split0_20260712/`, `runs/nc_ei_kane_scalar_constant_alignment_audit_20260712/`, and `runs/nc_ei_trace_lowdin_split0_cnp_root_audit_20260712/`.

### Density alignment

Older CSV exports shifted scalar branches to cross at the experimental `5.5e10 cm^-2`. That operation is now classified as a constrained audit, not as author-defined electrostatics. The periodic SP density emerges as `4.04117e10 cm^-2`; a free additional `epsilon_M` offset requires an explicit capacitor/Poisson energy before thermodynamic comparisons.

## Parameter provenance and current caveat

Supplementary Table 1 supplies `Eg`, `Ev`, spin-orbit splitting, `Ac`, modified Luttinger parameters, and dielectric constants for InAs, GaSb, and AlSb. Its cited method source, Li, Yang, and Chang, Phys. Rev. B 80, 035303 (2009), sets `Ep=22.5 eV` for every layer material; the reproduction uses that value. The supplement does not print all lattice/elastic, Zeeman, or BIA parameters. Strain and BIA terms are disabled consistently with the adopted cited-method approximation.

The Note-6 calculation is now an auditable periodic Kane/SP reconstruction, but the publication still does not define a Note-6-to-Note-2 Fig. 2 interface.

## Completion criteria

A result may be called a paper-level Fig. 2 reproduction only when all hold:

1. the generalized fixed point converges;
2. electron and hole densities and their ensemble convention are documented separately;
3. electrostatic alignment has a derived Poisson/capacitor functional rather than a free fit;
4. the E1/H1 two-band input is author-provided or derived by a controlled, validated projection;
5. the finite-temperature equation/free-energy convention is thermodynamically consistent and source-backed;
6. Fig. 2 `Delta_order(k)`, `E_pair(k)`, and JDOS agree without digitized curves entering the solver.
