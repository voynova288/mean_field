# Project log - hTTG shift current

Cleanup note: on 2026-05-30, known-wrong or superseded result directories were moved from the top-level `results/shift_current_htg/` directory into `results/shift_current_htg/_archived_wrong_or_superseded_20260530/`.  Valid but non-final tests/diagnostics/convergence-support runs were moved into `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/`.  Historical paths below are preserved as originally logged; check `results/shift_current_htg/CURRENT_RESULTS.md` before using any result for current conclusions.

## 2026-05-28

Created `src/analysis/shift_current_htg/` as the first implementation workspace for the Mao et al. hTTG shift-current benchmark.

### Reference points extracted from the paper

- Main formula: Mao Eq. (1), implemented with photon energy and the work-document prefactor convention.
- Gauge-free ingredients: Mao Eq. (3)-(4), implemented with `D=<u|partial_k H|u>` and energy differences so that intermediate `hbar` factors cancel.
- hTTG model: Mao Eq. (13)-(14), with `w1=110 meV`, corrugation `r=w_AA/w_AB`, hBN mass `m=30 meV`, ABA/AAA stacking phases.
- Reference targets: ABA magic angle 1.95 deg gives THz response ~1e4 microampere nm V^-2; AAA magic angle 1.75 deg gives ~1e5; chiral ABA r=0 suppresses THz response.
- Appendix-A toy model: nearest-neighbor gapped SLG with `t=2.73 eV`, `m=1.5 eV`, full hexagonal BZ; this is the first unit/sign/C3 validation target.

### Implemented

- Gauge-free response core in `response.py`.
- Appendix-A tight-binding toy in `slg_toy.py`.
- Mao hTTG adapter in `htg_adapter.py`, including:
  - `hbar v_F |K|=9.905 eV` conversion to existing HTG parameter format;
  - stacking phase `(phi1,phi2)` to HTG displacement conversion;
  - `m sigma_z` mass injection;
  - analytic `partial_k H` and finite-difference validator.
- CLI smoke gates:
  - `run_htg_velocity_check.py`
  - `run_slg_toy.py`
  - `run_htg_shift_current_smoke.py`
- Added selected-band-pair workflow:
  - vectorized pair-level generalized derivative in `response.py`, checked against the full tensor path at `n_shells=1`;
  - `run_htg_bandpair_spectra.py` streams compact transition events for specified relative band pairs, starting with the central flat-band pair `-1,0` for Fig. 2-style decomposition;
  - added `--pair-window name:min,max` to aggregate occupied-to-empty central windows for band-window convergence tests.

### Smoke validation run locally

- `PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_velocity_check --n-shells 1 --finite-step 1e-6`
  - analytic vs finite-difference `partial_k H` max error: `8.32e-11 eV nm`, passing the work-document `1e-7 eV nm` gate for this cutoff.
- `PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_shift_current_smoke --n-shells 1 --mesh-size 2 --n-energy 21 --no-save`
  - full-chain smoke completed without NaN;
  - tiny-grid ABA peak scale was `~5.5e3 microampere nm V^-2` for `x;yy`, which is only an order-of-magnitude sanity check, not a converged result.
- `PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra --n-shells 1 --mesh-size 2 --n-energy 41 --eta-mev 1,2 --no-save`
  - central flat-band pair event stream completed;
  - tiny-grid central-pair peaks were order `1e4 microampere nm V^-2`, consistent with the paper's expected THz scale but not converged.
- `PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_bandpair_spectra --n-shells 1 --mesh-size 2 --n-energy 21 --eta-mev 2 --pair-window central_window:-2,1 --no-save`
  - aggregated four central-window pairs completed; this is the first band-window-convergence hook.

### Not yet claimed

- No paper-grade hTTG figure has been reproduced yet.
- No mesh/broadening/band-window convergence has been run.
- The hTTG smoke script is intentionally tiny and must not be used as a numerical result.
- The Appendix-A SLG toy is implemented as a diagnostic, but its tensor-convention/basis-embedding comparison to Fig. 10 still needs a dedicated audit before it can serve as the acceptance test.

## 2026-05-28 visual/Slurm continuation

Submitted small CPU Slurm diagnostics from `login002` using account `hmt03`; no production-scale convergence was claimed.

### Slurm jobs

- `132069` via `scripts/run_shift_current_htg_fig1_fig2_smoke.sbatch`
  - output root: `results/shift_current_htg/fig1_fig2_smoke_132069/`
  - `n_shells=2`, band/DOS diagnostic plus central-flat and central-window spectra.
- `132076` via `scripts/run_shift_current_htg_fig1_fig2_nshell3.sbatch`
  - output root: `results/shift_current_htg/fig1_fig2_nshell3_132076/`
  - `n_shells=3`, band/DOS diagnostic plus central-flat spectra.
- `132077` via `scripts/run_shift_current_htg_active24_nshell3.sbatch`
  - output root: `results/shift_current_htg/active24_nshell3_132077/`
  - active 24-band occupied-to-empty window; this is not a full-band converged Fig. 1(b) reproduction.
- `132082` via `scripts/run_shift_current_htg_centralflat_allcomponents_nshell3.sbatch`
  - output root: `results/shift_current_htg/centralflat_allcomponents_nshell3_132082/`
  - all eight tensor components for the central flat-band pair, useful for tensor-convention checks.

### Figure reading notes

- Paper reference rendered/read: Mao Fig. 1 and Fig. 2 on PDF page 5.
- `results/shift_current_htg/fig1_fig2_nshell3_132076/bands_dos/htg_bands_dos.png`
  - visually comparable to paper Fig. 1(a): two middle bands are flat near zero and DOS has a strong central peak.  This is a qualitative band-structure checkpoint, not final because the DOS mesh and path style are still coarse.
- `results/shift_current_htg/fig1_fig2_nshell3_132076/central_flat/central_flat_eta2meV_spectra_reflect_y.png`
  - visually comparable to paper Fig. 2: after applying a y-axis reflection tensor convention, the central flat-band transition gives a large negative THz `sigma^{y;xx}` peak and a much smaller `sigma^{x;yy}` component.
  - Quantitatively still coarse: `Nk=12`, `eta=2 meV`, `n_shells=3`; peak is about `5.8e3 microampere nm V^-2`, below the paper-scale `~1e4` but in the right order.
- `results/shift_current_htg/active24_nshell3_132077/active24/active24_eta2meV_spectra_reflect_y_thz.png`
  - active-window attempt toward paper Fig. 1(b), but not successful as a Fig. 1(b) reproduction: `sigma^{x;yy}` is too small and the result is not converged over bands or k mesh.

### Path correction after visual audit

User pointed out that the band path was wrong.  Confirmed: the previous Fig. 1(a)-style path used the central-zone `kappa_prime_m` after the second `M`, which cuts across the mBZ and mislabels the high-symmetry segment.

Patch in `run_htg_bands_dos.py`:

- old wrong path: `-M -> Gamma -> M -> kappa_prime_m -> kappa_m -> Gamma`
- corrected paper-style path: `-M -> Gamma -> M -> (kappa_prime_m + b_m1) -> kappa_m -> Gamma`

The corrected `K'` is the adjacent extended-zone corner on the same mBZ edge as the second `M` and `K`, matching the label sequence `M Gamma M K' K Gamma` in Mao Fig. 1(a).

Slurm check:

- `132100` via `scripts/run_shift_current_htg_bands_pathfix.sbatch`
- output root: `results/shift_current_htg/bands_pathfix_132100/`
- corrected band/DOS figure: `results/shift_current_htg/bands_pathfix_132100/bands_dos/htg_bands_dos.png`

### Current status

- The earlier Fig. 1(a) visual comparison was based on a mislabeled/wrong high-symmetry path and should be disregarded.
- Corrected Fig. 1(a) path diagnostic is now available at `results/shift_current_htg/bands_pathfix_132100/bands_dos/htg_bands_dos.png`.
- Not yet successful: full Fig. 1(b) two-component spectrum and convergence/units acceptance.

## 2026-05-28 active24 spectrum audit

User flagged `results/shift_current_htg/active24_nshell3_132077` as visibly wrong.  Confirmed; do not use that directory for physics conclusions.

Findings:

1. `active24_nshell3_132077` computed only two displayed components (`x;yy`, `y;xx`) in the code's internal Cartesian axes.  The large response was actually sitting in other tensor components (`x;xy`, `x;yx`, `y;yy`) when all eight components were computed.
2. The finite `8x8` primitive-cell mesh was not C3-symmetrized.  This produced large finite-mesh tensor-convention artifacts.  After averaging over C3-rotated copies of the full primitive-cell mesh, the dominant C3 tensor group became consistent: `x;xy`, `x;yx`, `y;xx`, and `-y;yy` all peak around `5.35e3 microampere nm V^-2`.
3. Added `--c3-symmetrize-grid` to `run_htg_bandpair_spectra.py` and full tensor-basis rotation support to `plot_bandpair_spectra.py`.

New diagnostic runs:

- `132107`: all eight components without C3 symmetrization.
  - root: `results/shift_current_htg/active24_allcomponents_nshell3_132107/`
- `132114`: all eight components with C3-symmetrized mesh.
  - root: `results/shift_current_htg/active24_allcomponents_c3_nshell3_132114/`
  - useful diagnostic figures:
    - `active24_c3_eta2meV_rot0_reflect_y_thz.png`: shows the old plotted components after y-reflection; `x;yy` is nearly zero.
    - `active24_c3_eta2meV_rot10_reflect_y_thz.png`, `rot15`, `rot20`: demonstrate that tensor-basis rotation mixes the dominant internal component into the paper-labelled `x;yy`/`y;xx` channels.

Interpretation:

- The old active24 result was wrong mainly as a plotting/convention diagnostic: it displayed only two internal-axis components, without all-component tensor transformation and without C3 mesh symmetrization.
- Even after the fixes, this is still not a final Fig. 1(b) reproduction: the correct paper axis rotation is not yet fixed from first principles, and the calculation remains a coarse `Nk=8`, `n_shells=3`, active-24-band window run.

## 2026-05-28 Mao Dirac-convention correction

Found another important mismatch while auditing the active24 result: the first Mao runs used the existing Kwan-style HTG default `zeta_rad=None`, i.e. the outer Dirac cones were rotated in the layer blocks.  Mao Eq. (13) and Appendix B write every Dirac block as `hbar v_F k dot sigma`, without layer-rotated Pauli matrices.  Patched `MaoHTGConfig` so the default is now `zeta_rad=0.0`.

Consequences:

- Old result root `results/shift_current_htg/active24_nshell3_132077/` is invalid for Mao Fig. 1(b) for three reasons:
  1. Kwan-style rotated Dirac blocks were used instead of Mao Eq. (13)'s unrotated block.
  2. The k mesh was not C3 symmetrized.
  3. Only two internal-axis components were plotted.
- New zeta=0 diagnostic:
  - job `132340`, root `results/shift_current_htg/mao_zeta0_check_132340/`
  - all eight components + C3-symmetrized mesh at `Nk=8`.
  - `active24_zeta0_eta2meV_rot5_reflect_y_thz.png` is qualitatively closer to Mao Fig. 1(b): red `sigma^{y;xx}` dominant negative THz response and blue `sigma^{x;yy}` smaller negative response.
- Mesh-16 zeta=0 diagnostic:
  - job `132372`, root `results/shift_current_htg/mao_zeta0_c3_mesh16_132372/`
  - `Nk=16`, C3-symmetrized, active24, all components.
  - useful figures:
    - `active24_zeta0_m16_eta2meV_rot5_reflect_y_thz.png`
    - `active24_zeta0_m16_eta2meV_rot10_reflect_y_thz.png`
    - `active24_zeta0_m16_eta1meV_rot5_reflect_y_thz.png`
  - Peak scale after rotation is now `~7e3-1e4 microampere nm V^-2` depending on eta, close to Mao Fig. 1(b)'s order of magnitude, but the paper-axis rotation/domain convention still needs a first-principles fix before claiming reproduction.
- Domain flip diagnostic:
  - job `132384`, root `results/shift_current_htg/mao_zeta0_hbar_check_132384/`
  - flipping the helical domain flips the dominant C3 tensor group's sign, as expected, but does not by itself determine the paper component labels.

## 2026-05-28 physics-first audit after failed visual comparison

User correctly objected that the plotted spectra are still far from Mao Fig. 1(b)/Fig. 2 and that the physics checks in the work document should come first.  Re-reading the paper and the work document changes the status from "nearly comparable" to **not validated**.

Paper/work-document anchors:

- Mao Eq. (13): hTTG continuum Hamiltonian with unrotated `hbar v_F k dot sigma`, phases `phi1=-phi2=2pi/3` for ABA, and uniform `m sigma_z \otimes I_layer`.
- Mao Eq. (15): in the paper Cartesian convention there are two C3 tensor groups; Fig. 1(b) plots `sigma^{x;yy}` and `sigma^{y;xx}`.
- Mao Fig. 2 and text: the THz response below `0.1 eV` is accounted for by the two middle flat bands, so a central-flat-pair calculation should already have the correct THz shape if the model/tensor conventions are correct.
- Work document Step 0: Appendix-A gapped-SLG tight-binding benchmark is mandatory before trusting the hTTG response module.

New diagnostic finding and correction:

- The first Appendix-A SLG toy implementation used Mao Eq. (4) without a second-derivative term.  That is valid for the hTTG continuum Dirac Hamiltonian because `partial_a partial_b H = 0`, but it is not the covariant derivative of a nonlinear tight-binding Hamiltonian.
- Derived the general energy-difference formula:
  `r^b_{nm;a} = [Mao Eq. (4) terms] - i W^{ab}_{nm}/(E_n-E_m)`, with `W^{ab}_{nm}=<u_n|partial_a partial_b H|u_m>`.
- Patched `response.py` to accept optional `W`/`d2hdk`, and patched `slg_toy.py` to provide analytic second derivatives.  A focused finite-difference covariant-derivative check now matches the generalized derivative to `~1e-13` at random SLG k points.
- After adding `W`, the SLG toy C3 tensor relations pass when the hexagonal grid is C3-orbit averaged:
  - result root: `results/shift_current_htg/slg_toy_audit_mesh120_d2_c3/`
  - C3 relation errors are `~1e-13 microampere nm V^-2`.
- However, this corrected tight-binding toy **still does not reproduce Mao Fig. 10**: it has a large onset near the direct K/K' gap around `3.2 eV`, while Mao Fig. 10 emphasizes the M-point transition around `6 eV` with peak scale about `0.1 microampere nm V^-2`.  Therefore the Appendix-A benchmark remains unresolved; the remaining issue is likely the tight-binding optical/Bloch-basis convention or the precise convention used by Mao for Appendix A.

Implications for hTTG:

- The hTTG continuum calculation is not affected by the new second-derivative term because its Hamiltonian is linear in k; the term is exactly zero there.
- A new plane-wave-shell convergence run for the central flat pair shows that the apparent nonzero raw `sigma^{x;yy}` at `n_shells=2` was a cutoff artifact.  For `n_shells >= 4`, the code-axis raw `sigma^{x;yy}` is essentially zero, while the C3-related group `sigma^{y;xx}=sigma^{x;xy}=sigma^{x;yx}=-sigma^{y;yy}` converges to about `6.95e3 microampere nm V^-2` at `eta=2 meV`.
  - job `132459`, root `results/shift_current_htg/physics_audit_shell_central_132459/`
  - shell-5 mesh-8 raw peaks: `x;yy ~ 2.6e-7`, `y;xx ~ 6.95e3`.
  - job `132463`, root `results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/`
  - shell-5 mesh-16 raw peaks: `x;yy ~ 4e-7`, `y;xx ~ 6.93e3` for eta=2 meV; `y;xx ~ 9.79e3` for eta=1 meV.
- This reveals a physical/convention structure: in the converged code axes the hTTG tensor is effectively C3v-like with one independent nonzero scalar.  After `y` reflection plus a basis rotation by angle `alpha`, a pure raw group-2 tensor transforms as
  - `sigma_new^{x;yy} = -B sin(3 alpha)`
  - `sigma_new^{y;xx} = -B cos(3 alpha)`
  so the previously empirical `alpha ~ 5 deg` simply mixes the single converged tensor scalar into the two paper-labelled components with a ratio `tan(15 deg) ~ 0.27`, close to the visual ratio in Mao Fig. 1(b)/Fig. 2.
- This is a much better physics explanation than the earlier "all components + small rotation" story, but the rotation angle is still not derived from an explicit paper definition of the Cartesian axes.  It must not be presented as a final reproduction until that convention is fixed.

Follow-up after user pointed out the toy plot is still wrong:

- Confirmed: `results/shift_current_htg/slg_toy_audit_mesh120_d2_c3/slg_toy_xxy_yyy.png` is **not** Mao Fig. 10.  It has a dominant onset near the K/K' direct gap (`~3.2 eV`) plus a smaller M-region feature near `~6.1 eV`; Mao Fig. 10 is dominated by the M-point/DOS separation peak near `~6 eV` with scale `~0.1 microampere nm V^-2`.
- Decomposing the corrected SLG response by transition energy shows the mismatch explicitly: for `mesh=160`, `eta=50 meV`, `sigma^{x;xy}` has `max~2.30` from transitions below `4 eV` and `max~1.40` from transitions above `5 eV`.  Therefore the wrong visual comparison is not just a plotting-scale problem; the low-energy K/K' contribution is too large relative to Mao Fig. 10.
- Physical interpretation: in a purely isotropic massive-Dirac low-energy theory, a third-rank C-infinity-invariant shift-current tensor should vanish after angular integration, so Mao's claim that the M-point van Hove/DOS feature dominates is plausible.  The current tight-binding implementation produces a large low-energy contribution through the nonlinear `d2H/dkdk` term/trigonal-warping corrections.  This may indicate a Bloch-basis/optical-position convention mismatch in the toy benchmark, or possibly that Appendix A used a different numerical convention.  This remains unresolved.
- For hTTG, the converged raw tensor being effectively C3v-like is now a red flag/convention clue: either the code axes are aligned with an actual mirror axis and Mao's plotted axes are rotated away from it, or the implemented stacking/domain convention has an extra symmetry not intended by the paper.  This must be settled from the Hamiltonian/coordinate definition, not by visual fitting.

Retraction of visual-only Fig. 10 diagnostic:

- User correctly objected that visual alignment is misleading when the core calculation is wrong.  Deleted the visual-only scripts/results that Pauli-blocked, normalized, or rescaled data to resemble Mao Fig. 10.
- Rule now enforced locally: do not solve a core formula/physics/code problem in result presentation.  No post-hoc normalization, Pauli blocking, transition filtering, axis relabeling, or cosmetic plotting is allowed as evidence of reproduction.
- The honest formal status remains: the full `mu=0` W-corrected gapped-SLG calculation does not match Mao Fig. 10, and the Eq.-(4)-only paper-like calculation is not gauge/covariance consistent for a nonlinear tight-binding Hamiltonian.  Therefore Fig. 10 is not accepted as a physics validation yet.
- Independent finite-difference covariant-derivative spectrum check confirms the analytic `W`-corrected toy result: on a small mesh, finite-difference and analytic spectra agree to `~1e-10`, and both have the dominant K/K' direct-gap peak near `3.1-3.2 eV`.  The formal W-corrected reference plot is now `results/shift_current_htg/slg_toy_formal_wcorrected_m160/slg_toy_formal_xxy_yyy.png`.

External `mean_field_2411_toy_fig10_corrected.zip` audit:

- User supplied `/data/home/ziyuzhu/Mean_Field/reference/mean_field_2411_toy_fig10_corrected.zip` as a proposed GPT-5.5-Pro correction for the toy model.
- I inspected the relevant files (`src/analysis/shift_current_htg/slg_toy_fig10.py`, `run_slg_toy_fig10.py`) but did **not** import them.  The proposed implementation explicitly uses `mu_ev_for_m_channel=-3.0`, an `m_saddle_envelope_ev` Gaussian gate, `sc_length_unit_factor=0.1`, a digitized red/blue amplitude ratio, `dos_area_unit_factor=0.01`, edge tapering, and `visual_gates`.  That is an M-saddle/plot-panel construction, not a full `mu=0` shift-current benchmark.
- This violates the project rule: do not fix core physics/code mismatch via presentation, filtering, Pauli blocking, normalization, or visual gates.  Treat the zip as a quarantined external reference only, not as validated code.

Official reference-code audit after the GPT-5.5 patch rejection:

- Cloned shallow/sparse official sources to inspect formulas, not to trust labels:
  - `reference/upstream/wannier90/`
  - `reference/upstream/wannier-berri/`
- Relevant source locations:
  - `reference/upstream/wannier90/src/postw90/berry.F90::berry_get_sc_klist`
  - `reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula`
- Added `src/analysis/shift_current_htg/run_slg_toy_reference_formula_audit.py`.  It directly transcribes the official internal-term formula for the orthogonal nearest-neighbor SLG TB convention and compares it with the local `response.py` result on the same k grid.
- Validation run:
  `PYTHONPATH=src python -m analysis.shift_current_htg.run_slg_toy_reference_formula_audit --mesh-size 40 --n-energy 161 --output-dir results/shift_current_htg/slg_toy_reference_formula_audit_m40`
- Result: local W-corrected formula and official-reference transcription agree to numerical roundoff (`~1e-14 microampere nm V^-2` for the dominant components).  Both give the dominant K/K' direct-gap peak near `3.3 eV`, not Mao Fig. 10's M-saddle-only-looking peak near `6 eV`.
- Lesson captured per coding skill: after reviewing/fixing a code issue, update project docs with the root cause and reusable lesson.  Here the lesson is that official formula audits are valuable, but they strengthen the conclusion that the Mao Appendix-A Fig. 10 mismatch is not a plotting problem and not fixed by the standard Wannier90/WannierBerri shift-current formula.

Mao Fig. 10 reproduction audit:

- Added `src/analysis/shift_current_htg/run_slg_toy_fig10_reproduction_audit.py`.
- Ran:
  `PYTHONPATH=src python -m analysis.shift_current_htg.run_slg_toy_fig10_reproduction_audit --mesh-size 150 --n-energy 401 --output-dir results/shift_current_htg/slg_toy_fig10_reproduction_audit_m150`
- Output plot:
  `results/shift_current_htg/slg_toy_fig10_reproduction_audit_m150/slg_toy_fig10_audit.png`
- Formal W-corrected/C3-symmetrized result does not match Mao Fig. 10: `sigma^{x;xy}` and `sigma^{y;yy}` peak at the K/K' direct gap around `3.18 eV` with magnitude `~2.34 microampere nm V^-2`.
- The paper-printed Eq. (4)-only/no-W primitive-grid calculation gives an apparent M-point `sigma^{x;xy}` peak close to Mao's blue curve: `-0.101 microampere nm V^-2` at `6.24 eV` for mesh 150, eta 50 meV.  However, the paper-labelled `sigma^{y;yy}` is zero; the nonzero finite-grid partner is `sigma^{y;xx}=+0.202`.
- Mesh convergence exposes the apparent M peak as non-converged / grid-artifact-like: `|sigma^{x;xy}| = 0.313 (80), 0.204 (100), 0.153 (120), 0.101 (150), 0.0537 (200), 0.0299 (250), 0.0163 (300)`.  With C3-orbit averaging, the Eq. (4)-only result vanishes to roundoff.
- Honest conclusion: Mao Fig. 10 is not reproducible as a converged full shift-current calculation of Eq. (A1).  One can recreate part of the blue M-point feature only by using the incomplete printed Eq. (4) and a finite non-C3 primitive grid, which is not acceptable as validation.

hTTG tensor-axis diagnostic:

- Added `src/analysis/shift_current_htg/HTG_SYMMETRY_AUDIT.md`, a written derivation of the exact antiunitary layer-swap/conjugation symmetry and its tensor consequence.  The derivation shows that the literal Eq. (13) model with equal layer mass forces Mao Eq. (15)'s first C3 group (`sigma^{x;yy}`, `-sigma^{x;xx}`, `sigma^{y;yx}`, `sigma^{y;xy}`) to vanish in the symmetry axes, while the second group (`sigma^{y;xx}`, `-sigma^{y;yy}`, `sigma^{x;xy}`, `sigma^{x;yx}`) remains allowed.
- Added `src/analysis/shift_current_htg/analyze_bandpair_tensor_symmetry.py`.
- Ran it on the converged shell5 central-flat C3 dataset:
  `PYTHONPATH=src python -m analysis.shift_current_htg.analyze_bandpair_tensor_symmetry --input results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/central_flat_allc/htg_bandpair_spectra.npz --summary results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/central_flat_allc/summary.json --group central_flat --eta-mev 2 --reflect-y --output-dir results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/central_flat_allc/tensor_symmetry_audit_eta2`
- Output:
  `results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/central_flat_allc/tensor_symmetry_audit_eta2/tensor_symmetry_audit.png`
- Finding: the C3 Eq. (15) group1 coefficient is essentially zero (`max≈1.5e-7`), while group2 peaks at `≈6930 microampere nm V^-2`; group1/group2 ratio is `~2.2e-11`.  Thus the converged code-axis central-flat response has an extra C3v-like mirror constraint.  Rotating the coordinate basis moves weight between paper-labelled `x;yy` and `y;xx` as expected, but this is an axis/symmetry diagnostic, not a paper reproduction.
- Submitted a no-C3 shell5 mesh16 central-flat Slurm diagnostic to test whether paper-like `x;yy` can arise from finite primitive-grid artifacts:
  `scripts/run_shift_current_htg_central_shell5_mesh16_noc3.sbatch`.  Initial job `133933` was stuck pending and was cancelled; resubmission `133946` was also pending on `regular6430`; re-submitted with `--partition=regular` as job `133958`, now running on `node015`.
- Added and ran a symmetry audit:
  `PYTHONPATH=src python -m analysis.shift_current_htg.run_htg_symmetry_spectrum_audit --n-shells 3 --output results/shift_current_htg/htg_symmetry_spectrum_audit_shell3/summary.json`.
  Eigenvalues are invariant under `k -> conj(k)` to `~1e-14 eV`, but not under simple `k -> -conj(k)`, `k -> -k`, or naive C3 rotations.
- Upgraded that audit from eigenvalue-only to an exact antiunitary unitary-matrix check.  The implemented ABA Hamiltonian obeys
  `H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger`, with reciprocal-index map `(n1,n2)->(-n2,-n1)`, layer map `1<->3, 2->2`, and identity sublattice map.  The Hamiltonian error is `~8.4e-17 eV`; derivative matrices transform exactly with signs `dH/dkx -> +`, `dH/dky -> -`.  This antiunitary layer-swap/conjugation symmetry explains the one-coefficient C3v-like tensor and is distinct from the unitary `C2x` that Mao says is broken by `m sigma_z`.
- Ran the tensor audit on the older no-C3 `n_shells=3` central-flat data.  It has group1/group2 only `~9.5e-4`, and raw `x;yy` at `rot=0` is `~61` versus group2 `~5.8e3`, confirming that the large paper-like `x;yy` is not present even without C3 averaging at low cutoff.
- The no-C3 shell5 mesh16 resubmission finished as job `133958`:
  `results/shift_current_htg/physics_audit_central_shell5_mesh16_noc3_133958/central_flat_allc/summary.json`.
  Raw `x;yy` is `~1.5e-6` at eta 2 meV while `y;xx` is `~6990`, and the tensor audit gives group1/group2 `~4.4e-11`.  Thus finite primitive-grid artifacts do **not** explain the missing paper-labelled `x;yy`; the exact antiunitary symmetry suppresses it even without C3 averaging.
- Added `run_htg_layer_mass_pattern_audit.py` to test a possible physical/code hypothesis: if the hBN mass is layer-dependent rather than `m sigma_z otimes I_layer`, the layer-swap/conjugation antiunitary is broken and the second C3 tensor coefficient can turn on.  A quick low-cutoff shell2/mesh8 diagnostic (`results/shift_current_htg/htg_layer_mass_pattern_audit_shell2_m8/summary.json`) confirms that unequal layer masses produce large raw `x;yy`, but this is not a valid Mao Eq. (13) reproduction because the paper explicitly writes an equal layer mass.
- Axis-convention diagnostic: digitizing Mao Fig. 2 gives approximate peak ratio `|sigma^{x;yy}/sigma^{y;xx}| ~ 0.25`.  For the converged one-coefficient tensor, a reflected basis rotated by `alpha` gives `|sigma^{x;yy}/sigma^{y;xx}| = |tan(3 alpha)|`, implying `alpha ~ 4.8 deg`.  Generated candidate transformed plots (not a reproduction claim):
  - `results/shift_current_htg/physics_audit_central_shell5_mesh16_132463/central_flat_allc/central_shell5_m16_eta2_rot4p8_reflect_y.png`
  - `results/shift_current_htg/mao_zeta0_c3_mesh16_132372/active24_allc/active24_zeta0_m16_eta2_rot4p8_reflect_y.png`
  These match the paper-like blue/red sign and approximate low-energy amplitude ratio, but the origin of the `~4.8 deg` axis offset is not derived from Mao's definitions, so this remains only a convention diagnostic.
- Finite-grid artifact diagnostic: a deliberately non-mirror-symmetric low-cutoff run (`n_shells=3`, mesh 8, `frac_shift=0.23,0.37`) produces a sizable raw `x;yy` artifact (`~2462`) while `y;xx~6476`, showing how breaking the `k -> conj(k)` pairing in the quadrature can activate forbidden components.  This is not converged.  The shell5 mesh16 version finished as job `133985` using `scripts/run_shift_current_htg_central_shell5_mesh16_shifted_noc3.sbatch`:
  `results/shift_current_htg/physics_audit_central_shell5_mesh16_shifted_noc3_133985/central_flat_allc/summary.json`.
  At eta 2 meV, raw `x;yy~696`, `y;xx~6829`, and the C3 group1/group2 coefficient ratio is `~0.046`.  Thus a non-mirror quadrature can produce a noticeable forbidden-component artifact, but it is far smaller than the one-coefficient response and does not by itself explain Mao's paper-labelled `x;yy/y;xx` ratio.  The `~4.8 deg` axis rotation explains the plotted ratio algebraically, but remains empirical unless derived from a published convention.

hTTG axis-convention audit:

- Downloaded Mao/Guerci/Mora 2023 PRB 107, 125423 for cross-reference:
  `reference/Mao_Guerci_Mora_2023_PRB107_125423.pdf`.
- Added `HTG_AXIS_CONVENTION_AUDIT.md`.
- Key points:
  - Mao 2025's printed C3 action is exactly the +120 degree rotation that cycles the code's `q0=(0,-q)`, `q1=(sqrt(3)q/2,q/2)`, `q2=(-sqrt(3)q/2,q/2)`.
  - Appendix-A `d_j` axes differ from the hTTG `q_j` triplet by sign/relabeling, not by a small `~5 deg` rotation.
  - Mao/Guerci/Mora 2023 explicitly states that ABA local hTTG preserves the combined antiunitary `C2x C2z T`, matching the layer-swap/conjugation symmetry found numerically here.
  - No published coordinate definition found so far gives the empirical `~4.8 deg` reflected-basis rotation; that angle remains an inverse fit to the apparent Mao Fig. 2 component ratio, not a reproduction step.

SLG Fig. 10 cross-reference audit:

- Downloaded and extracted Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016):
  `reference/Hipolito_Pedersen_Pereira_2016_PRB94_045434.pdf`.
- Added `SLG_FIG10_AUDIT.md`.
- Added `run_slg_toy_hipolito_crossref.py` and generated the cross-reference evidence plot:
  `results/shift_current_htg/crossref_hipolito2016_fig4_evidence/hipolito2016_crossref_evidence.png`.
- Added `run_slg_toy_hipolito_fig4.py` and generated a direct Hipolito Fig. 4 reproduction:
  `results/shift_current_htg/crossref_hipolito2016_fig4_reproduction_m360/hipolito2016_fig4_reproduction.png`.
  This uses Hipolito Eq. (25b), `gamma0=3 eV`, `Delta=0.2 eV`, `Gamma=1 meV`, `mu=0`, `T=1 K`, and fixes the remaining global convention by Hipolito Eq. (31), not by visual fitting.
- The first fixed-k-grid Fig. 4 reproduction still had small oscillations at `Gamma=1 meV`.  Root cause: a narrow resonant denominator was being sampled on a fixed radial k grid; a finite-difference covariant derivative made component-overlap diagnostics worse.  Patched `run_slg_toy_hipolito_fig4.py` to use analytic W-corrected generalized derivatives by default, and added `run_slg_toy_hipolito_fig4_energy_quad.py`, which integrates radial momentum in transition-energy variables.  Output:
  `results/shift_current_htg/crossref_hipolito2016_fig4_energy_quad/hipolito2016_fig4_energy_quad.png`.
- Lesson for production spectra: visible wiggles at narrow broadening are numerical quadrature/derivative artifacts, not plotting issues.  Use analytic derivatives and ensure transition-energy resolution finer than the Lorentzian width, or use tetrahedron/transition-energy/adaptive quadrature for final spectra.
- Added reusable Hipolito 2016 benchmark suite:
  - module: `src/analysis/shift_current_htg/hipolito2016.py`
  - runner: `src/analysis/shift_current_htg/run_hipolito2016_benchmark_suite.py`
  - docs: `src/analysis/shift_current_htg/BENCHMARKS_HIPOLITO2016.md`
  - output: `results/shift_current_htg/hipolito2016_benchmark_suite/hipolito2016_benchmark_suite.png`
  - accepted benchmarks: Fig. 4 K/K' direct-gap Re/Im line shape and Fig. 5(b) Pauli-blocking threshold shift.  The summary records Eq. (31) target `-3.75`, local value `-3.75`, and Pauli thresholds `2mu`.
- Added `run_hipolito2016_fig5a_gap_series.py` for the low-energy K/K' threshold part of Hipolito Fig. 5(a):
  `results/shift_current_htg/hipolito2016_fig5a_gap_series_exact_t72/hipolito2016_fig5a_gap_series.png`.
  This uses one global convention factor calibrated at `Delta=0.2 eV` by Eq. (31), then reuses it for `Delta=0.1,0.2,0.5,1,2 eV`; it is not a per-curve visual normalization.  Scope is deliberately the low-energy threshold/gap-dependence part, not the full 0--9 eV high-energy M/UV van-Hove structure.
- Corrected the Fig. 5(a) low-energy quadrature after large oscillations appeared in the first version.  Root cause: with `Gamma=1 meV`, sampling `1/(omega-Ecv+iGamma)` and especially `1/(omega-Ecv+iGamma)^2` at discrete transition-energy nodes gives node-crossing/ringing artifacts when the node spacing is not much smaller than `Gamma`.  The fix is not smoothing: `hipolito_eq25b_spectrum_energy_intervals` now integrates the resonant denominators analytically over each transition-energy interval while evaluating only the smooth numerator at the interval midpoint.
- Added `hipolito_eq25b_spectrum_fixed_grid` and `run_hipolito2016_fig5a_full_bz_diagnostic.py` for a broader-broadening full-BZ diagnostic including M-point features:
  `results/shift_current_htg/hipolito2016_fig5a_full_bz_diagnostic_m140_g30/hipolito2016_fig5a_full_bz_diagnostic.png`.
  This uses `Gamma=30 meV` rather than the published `1 meV` because fixed-grid full-BZ integration otherwise produces shell-sampling artifacts; it is a diagnostic scaffold for the high-energy structure, not a strict published-figure reproduction.
- Added the strict full-BZ Hipolito Fig. 5(a) path using binned linear tetrahedra and analytic resonant-denominator interval integration:
  `results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev/hipolito2016_fig5a_full_bz_tetra.png`.
  Parameters: `Gamma=1 meV`, `mesh_size=720`, `energy_bin_width=1 meV`, `n_photon=1901`, all gaps `Delta=0.1,0.2,0.5,1,2,3,4,5,6,7 eV`.  Jobs `134317`/`134328`/`134329` completed on Slurm for mesh `360/480/720`; the m720 run is the current reference.  A convergence summary is stored at `results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra_m720_bin1mev/convergence_m360_m480_m720.json`.  This replaces the older broadening diagnostic for Fig. 5(a) claims.
- Reran the Fig. 4 / Fig. 5(b) benchmark suite with analytic transition-energy interval integration rather than point-node sampling:
  `results/shift_current_htg/hipolito2016_benchmark_suite_interval/hipolito2016_benchmark_suite.png`.
  Job `134325` completed on Slurm node004.  Eq. (31) calibration remains exact to `4.44e-16` at `Delta+0.03 eV`; Pauli thresholds remain `2mu`.
- Cleaned `results/shift_current_htg/` by moving known-wrong or superseded outputs to `results/shift_current_htg/_archived_wrong_or_superseded_20260530/` rather than deleting them.  Then moved valid but non-final tests/diagnostics/convergence-support runs to `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/`.  Archive reasons and moved directories are documented in both archive `README.md` files.
- Added hTTG tetrahedron/interval retry workflow `run_htg_bandpair_spectra_tetra.py` and comparison plotting `plot_mao_retry_tetra_comparison.py`.  Slurm jobs `134370`, `134379`, `134382`, and `134385` produced smooth Mao retry spectra.  Current top-level hTTG retry outputs:
  - `results/shift_current_htg/mao_retry_tetra_central_flat_shell5_m28/`
  - `results/shift_current_htg/mao_retry_tetra_active24_shell4_m12/`
  - `results/shift_current_htg/mao_retry_tetra_comparison_shell4active.png`
  - `results/shift_current_htg/mao_retry_tetra_summary.json`
  Supporting m20/m24/shell3/window runs were archived under `_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/mao_retry_tetra_support/`.  Conclusion: numerical wiggles are fixed, but Mao Fig. 1(b)/Fig. 2 still require an axis/component convention derivation because raw code-axis `sigma^{x;yy}` remains symmetry-forbidden/tiny.
- Key point from Hipolito 2016: the same gapped honeycomb TB model class has a direct K-point band-edge onset at `hbar omega=Delta`, with analytic threshold behavior `Re sigma_dc^(2) ~ -sigma_2/(4 Delta) theta(hbar omega-Delta)` near K.  For Mao's toy `m=1.5 eV`, `Delta=2m=3 eV`, so a K/K' direct-gap contribution is expected and is not symmetry-forbidden.
- The local Hipolito-style reproduction uses a K-corner patch grid for the `Delta=0.2 eV`, `gamma0=3 eV` benchmark.  The symmetry-related real components overlap to `~1.6e-12` in units of `sigma/sigma2`, and the onset is at the K/K' gap, matching Hipolito's physical conclusion.
- This supports the formal W-corrected/reference-code-consistent local SLG result and reinforces that Mao Fig. 10's M-only-looking panel is not a reliable full neutral tight-binding benchmark.

Reference-code comparison / reproduction status summary:

- Added `REPRODUCTION_STATUS.md`, a conservative figure-by-figure status report after comparing against Wannier90/postw90, WannierBerri, and HopTB-style formulas.
- Current status:
  1. Core shift-current formula benchmark: passed against reference-code transcription.
  2. Mao Fig. 1(a): qualitative band/DOS candidate exists, not yet quantitatively signed off.
  3. Mao Fig. 2: close in shape/scale only as a `~4.8 deg` reflected-axis convention diagnostic; raw code-axis paper labels are not reproduced because of the exact antiunitary symmetry.
  4. Mao Fig. 1(b): partial order-of-magnitude/shape diagnostics; active-window production at higher shell is still needed once axis convention is resolved.
  5. Mao Fig. 10: not honestly reproduced; formal reference-code-consistent SLG calculation disagrees with the paper panel.

Updated rule: do not claim Fig. 1(b), Fig. 2, or Fig. 10 reproduction until (i) the Appendix-A SLG benchmark convention is resolved or deliberately excluded from the hTTG validation, and (ii) the paper coordinate/domain convention / possible extra mirror symmetry is derived rather than fitted by rotation.
