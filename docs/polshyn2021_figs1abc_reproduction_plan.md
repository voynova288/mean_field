# Polshyn 2021 Supplementary Fig. S1(a-c) reproduction plan

Target PDF:
`reference/Polshyn 等 - 2021 - Topological charge density waves at half-integer filling of a moiré superlattice.pdf`

Rendered reference page:
`tmp/pdfs/polshyn2021_pages/page-13.png`

## Target panels

Supplementary Fig. S1:

- **S1a**: non-interacting tMBG band structure with the target `C = 2` flat band labeled.
  Path shown in the paper: `Gamma - K_-^M - M - K_+^M - Gamma - M`.
- **S1b**: Hartree-Fock band structure for the `nu = 7/2` SBCI along the doubled-cell
  rectangular BZ line `kx = 0`, **without remote bands**.  The paper's yellow branch is
  the spin-up valley `K+` `C = 2` BM band split into two `C = 1` bands by translation
  breaking.
- **S1c**: same line plot, but with five remote bands kept: two above and three below the
  `C = 2` band.

Paper method facts extracted from the Supplementary Information:

- System: twisted monolayer-bilayer graphene (tMBG), not TBG.
- HF interaction: double-gate screened Coulomb,
  `V(q) = tanh(d q) / (2 epsilon_0 epsilon_r q)`.
- Parameters stated explicitly: `epsilon_r = 20`, gate distance `d = 120 nm`.
- Target state: `nu = 7/2` SBCI near `D = 0.4 V/nm`.
- Allowed translation breaking: translations along `y` preserved; the other primitive
  translations are preserved only at order two.
- Translation-breaking wavevector: `Q = (2 pi / (sqrt(3) a_M)) xhat = G_M1 / 2` in the
  code's tMBG convention.
- The doubled BZ used for S1(b,c) is rectangular with axes approximately
  `kx a_M in [-pi/sqrt(3), pi/sqrt(3)]` and `ky a_M in [-pi, pi]`.

## Supercell convention to reuse from the Zhang-Fig.-10 work

Use the same supercell logic as the recent Zhang implementation, but with an area-2 tMBG
cell rather than Zhang's area-3 TBG cell.

In the existing tMBG code, primitive reciprocal vectors are

```text
b1 = G_M1 = (4 pi / (sqrt(3) a_M), 0)
b2 = G_M2 = (2 pi / (sqrt(3) a_M), -2 pi / a_M)
```

A doubled rectangular cell compatible with the paper is

```text
B1 = b1 / 2                         # x direction, CDW wavevector Q
B2 = b2 - b1 / 2 = (0, -2 pi / a_M) # y direction
```

This corresponds to real-space integer matrix

```text
R1 = 2 r1 + r2
R2 = r2
Nc = 2
n11 = 2, n12 = 1, n21 = 0, n22 = 1
```

and inverse folding relations

```text
b1 = 2 B1
b2 = B1 + B2
Q  = B1
```

This is the central geometry check before doing HF.

## Filling convention

Project to the `C = 2` primitive band in all four spin/valley flavors, then fold by `Nc=2`.

For the **no-remote** S1b calculation, each flavor has two folded bands per doubled-cell k.
The `nu = 7/2` occupation is

```text
K+ up:   1 / 2  # half-filled target C=2 band after folding
K+ down: 2 / 2
K- up:   2 / 2
K- down: 2 / 2
```

For the **remote-band** S1c calculation, keep primitive bands ordered as

```text
3 lower remote bands + target C=2 band + 2 upper remote bands
```

After folding, each flavor has 12 bands.  Occupations should be

```text
K+ up:   3*2 + 1 = 7
K+ down: 3*2 + 2 = 8
K- up:   3*2 + 2 = 8
K- down: 3*2 + 2 = 8
```

The centered/reference density must not use the charge-neutral TBG convention.  Polshyn's
`nu=7/2` counts electrons added into the target conduction `C=2` band, so the target band
reference is empty.  For the remote-band calculation, the subtraction-method reference fills
lower remote bands and leaves the target and upper remote bands empty, flavor by flavor.
This gives `(7 - 0)/2 = 7/2` for the no-remote folded cell and `(31 - 24)/2 = 7/2` for the
remote folded cell.

## Implementation phases

### Phase 0 - parameter and non-interacting anchor

1. Add a Polshyn-specific devtool, tentatively
   a reusable tMBG system/devtool entrypoint if this workflow is revived.  The old one-off `run_tmbg_polshyn_figs1_abc.py` runner was retired during script-surface cleanup.
2. Implement S1a first using existing `mean_field.systems.tmbg` non-interacting model.
3. Default parameters after the Polshyn-2020 convention correction:
   - `theta_deg = 1.29`, matching the experimental device.
   - `parameter_set = polshyn2020` with `t0=-2.61 eV`, `wAB=117 meV`, `wAA=0.7 wAB`,
     `t1=361 meV`, `t3=283 meV`, `t4=138 meV`, and `delta=0`.
   - `interlayer_potential = -0.033 eV` as the code-convention proxy for paper `D=+0.4 V/nm`.
   - `n_shells = 5` initially; allow `6` as convergence diagnostic.
4. Validate the target band identity by computing/recording the flat-band index and, if
   affordable, the Chern number on a Slurm-backed mesh.  Do not claim the label `C=2`
   from plotting alone.

### Phase 1 - doubled-cell projected basis

Implement a tMBG counterpart of the Zhang/TBG supercell machinery:

1. Build an area-2 doubled-cell k mesh in `(B1, B2)`.
2. For each doubled-cell k, fold primitive momenta `k` and `k + Q`.
3. Diagonalize primitive tMBG at these two momenta for both valleys and selected bands.
4. Embed primitive plane-wave eigenvectors into a doubled-cell reciprocal grid using
   integer coordinates

```text
primitive G = n1 b1 + n2 b2
fold f      = 0 or 1
super coords = (2 n1 + n2 + f, n2) in (B1, B2)
```

5. Reuse `core.hf.ProjectedWavefunctionBasis` / overlap contraction machinery where
   possible, as the HTG adapter already does through rectangular G embedding.

### Phase 2 - projected HF density

Use the generic projected-HF engine rather than a phenomenological CDW potential.

1. Build active overlap blocks with double-gate screened Coulomb (`epsilon_r=20`,
   `d_sc_nm=120`).
2. Run flavor-diagonal projected HF at fixed sector occupation counts above.
3. Use the validated Wang/Xiaoyu stored-projector ODA framework as the default SCF engine
   (`--hf-engine wang`).  The earlier compact fixed-mixing prototype is retained as
   `--hf-engine legacy` only for diagnostics because it produced nonuniform order and wavy
   bands on S1b.
4. Start with random initial density in the `K+ up` folded target sector so the solver can
   break primitive translation; keep a BM/folded init as a diagnostic only.
4. Save per-iteration error/energy and the translation-breaking order parameter
   `O(k) = sqrt(sum_nn' |<c^dag_{n,k} c_{n',k+Q}>|^2)`.
5. Stop only after convergence or report explicitly as an unconverged qualitative check.

### Phase 3 - S1(b,c) line plots

1. Build a rectangular-BZ line `kx=0`, `ky a_M in [-pi, pi]`, using the doubled-cell basis.
2. Reconstruct the HF Hamiltonian on that line from the saved SCF density.
3. For S1b plot the no-remote run; for S1c plot the remote-band run.
4. Color convention:
   - yellow: `K+ up`
   - purple: `K+ down`
   - pink: `K-`, spin-degenerate if numerically degenerate
5. Save raw NPZ/TSV plus PNG/PDF.  Keep energy reference explicit; likely align the
   occupied/unoccupied split to the dashed zero line as in the paper.

## 2026-05-30 configuration/convention corrections

Two target-identification mistakes were found during the physical/folding audit:

1. The Polshyn driver must not use the old Park-checkpoint AB/upper-flat default.  That plot
   was a plausible band structure for another configuration/band, not the project target.
2. Polshyn 2020 Supplement Eq. (S5) writes `T_Bernal` as the middle-bottom block in
   `(top, middle, bottom)` order.  Since this code stores `(bottom, middle, top)`, the
   Polshyn bottom-middle block must be `T_Bernal^dagger`, not the historical Park/JM block.

The current Polshyn defaults are therefore:

```text
parameter_set = polshyn2020
blg_stacking = BA
interlayer_potential = -0.033  # code convention for paper D ~= +0.4 V/nm
auto_target_role = upper-flat # the C=+2 conduction band
```

A quick sewn-Chern probe at `theta=1.29`, `D_code=-33 meV`, `mesh=12` found the upper/conduction
central band has `C=+2` and the lower/valence band has `C=-1`, matching S1a labels.  The latest
S1a anchor is close to the paper, but S1b/S1c still need HF/order/convergence and possibly the
full subtraction-method renormalized `h0` before claiming reproduction.  A first-pass active-projected
`HF[-P0_projected+P_ref]` diagnostic did not solve S1b and worsened order, so the next implementation
must include the full active-remote off-block structure of `Q_remote P0 Q_remote - P0`, not just the
active-active projection.  Separately, when remote bands are kept (S1c), the filled kept remote bands must
enter the active HF potential: the current default `--h0-subtraction active-reference` adds `HF[P_ref]`
so the `P-P_ref` SCF convention is equivalent to an active-band `HF[P]` potential.  This fixed the severe
S1c order failure in the SCF-grid diagnostics.

## Validation gates

- Geometry: area ratio `Nc=2`, `B1=b1/2`, `B2=b2-b1/2`, `Q=B1`, rectangular BZ axes match
  Fig. S1 labels.
- Hermiticity of projected `h0` and HF Hamiltonians.
- Occupation count gives primitive filling `nu = 7/2` for the active `C=2` band.
- With no remote bands, the two folded `K+ up` target branches should carry total Chern
  `C=2` and split into two `C=1` branches after HF (requires topology check; plot alone is
  insufficient).
- S1a band identity: the plotted target band should be the one whose valley Chern is `+2`
  in the paper's convention.
- Run numerical HF and topology checks only via Slurm, not on login nodes.

## Non-goals / pitfalls

- Do **not** reproduce S1(b,c) by inserting an ad hoc sinusoidal or constant CDW coupling
  unless the artifact is explicitly labeled as a phenomenological visual diagnostic.  The
  requested reproduction should use projected HF density.
- Do **not** use the TBG Zhang `sqrt(3) x sqrt(3)` area-3 cell; Polshyn S1 uses a doubled
  rectangular tMBG cell.
- Do **not** assume `D = 0.4 V/nm` maps uniquely to `interlayer_potential = 40 meV`; treat
  this as a parameter to verify against S1a and record in metadata.
- Do **not** mark the result as a paper-level reproduction until S1a band identity and S1b/c
  HF convergence are documented.
- Do **not** trust dense post-SCF HF path waves until the exact SCF-grid line and the Wang/Xiaoyu
  framework output have been checked; artificial folding waves are a framework/convention warning.
- Prefer `--skip-reconstructed-line` for current Polshyn HF diagnostics; the Slurm default now enables it.
  Use SCF-grid data (`polshyn_figS1*_hf_kx0_scf_grid_data.npz`) for physics conclusions; dense reconstructed
  paths are auxiliary until the target/source h0 and overlap reconstruction are audited.
- Current best assembled SCF-grid reproduction snapshot:
  `results/TMBG_Polshyn2021_figS1/polshyn_figS1abc_final_scfgrid_20260530.png`, with direct contact
  sheet `results/TMBG_Polshyn2021_figS1/polshyn_figS1abc_final_vs_paper_contact_20260530.png`.
  It uses S1b `minus-full-p0` at k18 and S1c `active-reference` at k9; this is a paper-matching
  diagnostic snapshot, not yet a single unified subtraction-method convention.
