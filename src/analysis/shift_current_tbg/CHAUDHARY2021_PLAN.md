# Chaudhary--Lewandowski--Refael 2021 TBG shift-current reproduction plan

Reference PDF:

```text
reference/Chaudhary 等 - 2021 - Shift-current response as a probe of quantum geometry and electron-electron interactions in twisted.pdf
```

## Core targets

1. **Noninteracting benchmark, Fig. 2**
   - TBG continuum model at `theta=0.8 deg`.
   - Paper parameters: `hbar v/a = 2.1354 eV`, `a=0.246 nm`, `u0=90 meV`, `u=0.4 u0`, layer offsets `Delta1=Delta2=5 meV`.
   - Decompose shift current into:
     - FF: direct flat-valence to flat-conduction transitions.
     - FD: transitions between central flat bands and nearby dispersive bands.
   - Main tensor in paper notation: `sigma^y;xx` / `sigma_xxy`.
   - Expected order: FF peaks around `~10 meV` with `~10^3 microA nm V^-2`; FD peaks around the flat-dispersive gap with up to `~10^4--2e4 microA nm V^-2` near `theta=0.8 deg`.

2. **Supplementary substrate/symmetry tests**
   - Vary `Delta`: `5,10,20 meV`.
   - Check tensor constraints:
     - `Delta1=Delta2`: D3 symmetry, one independent nonzero tensor coefficient.
     - `Delta1!=Delta2`: C3 symmetry, two independent nonzero tensor coefficients.

3. **Interacting/Hartree targets, Fig. 1 and Fig. 4**
   - Reproduce qualitatively only after the noninteracting implementation is validated.
   - Need a Hartree self-consistent or imported/frozen Hartree potential workflow.
   - Target physics: filling-dependent band flattening, enhancement of FD response, and a new opposite-sign peak near `~60 meV`.

## Current implementation status

Implemented corrected noninteracting scaffold plus a Hartree prototype:

```text
src/analysis/shift_current_tbg/chaudhary2021.py
src/analysis/shift_current_tbg/hartree.py
src/analysis/shift_current_tbg/run_chaudhary2021_b0_bands.py
src/analysis/shift_current_tbg/run_chaudhary2021_b0_noninteracting.py
src/analysis/shift_current_tbg/run_chaudhary2021_hartree_bands.py
src/analysis/shift_current_tbg/run_chaudhary2021_hartree_shift_current.py
src/analysis/shift_current_tbg/plot_chaudhary2021_fig2.py
src/analysis/shift_current_tbg/plot_chaudhary2021_hartree_comparison.py
```

Important correction: the initial `run_chaudhary2021_noninteracting.py` used the newer `atmg` shell-gauge adapter and produced a wrong Fig. 2(a)-style band structure.  Those outputs are archived as wrong/superseded under:

```text
results/shift_current_tbg/_archived_wrong_or_superseded_20260531/atmg_shell_wrong_bands/
```

The corrected workflow now uses the repository's previous b0 noninteracting model in `mean_field.systems.tbg.zero_field`, with momenta in the old b0 convention and energies converted from meV to eV for the response code.

Key numerical choices:

- Reuses the already validated gauge-free shift-current response core from `analysis.shift_current_htg.response`.
- Uses finite-difference `dH/dk` for the previous b0 Hamiltonian, converted to `eV nm`.
- Uses linear triangular transition-energy histograms plus analytic Lorentzian interval integration, inherited from the hTTG wiggle fix.
- Does **not** smooth spectra.

Current corrected outputs:

```text
results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd_same_lg7_m16_c3/
results/shift_current_tbg/chaudhary2021_b0_nonint_fig2_fd1_lg7_m16_c3/
```

This fixes the band structure and the FD pair-selection issue.  The paper-like FD direct-transition mode is `same_side`: lower dispersive -> valence flat and conduction flat -> upper dispersive.  This makes the FD response Pauli-blocked at charge neutrality, matching the text around Fig. 2(e).  Earlier `fd_mode=all` outputs are archived as superseded because they mixed in cross-gap flat--dispersive transitions and produced a spurious large FD signal at neutrality.

Current paper-like per-flavor metrics are summarized in:

```text
results/shift_current_tbg/chaudhary2021_nonint_convergence_table.md
```

Representative values after rescaling to response degeneracy `1`:

```text
fd_mode=same_side, fd_bands=10, lg7/m16:
  FF |nu|~2 peak ~1.0e3 microA nm V^-2 near 16 meV
  FD nu=0 = 0 (Pauli blocked)
  FD |nu|~2 peak ~1.4e3 microA nm V^-2 near 80 meV

fd_mode=same_side, fd_bands=1, lg7/m16:
  FD nu=0 = 0 (Pauli blocked)
  FD |nu|~2 peak ~0.2e3 microA nm V^-2 near 70 meV for the filling-labelled cut
```

The explicit `mu=-30,0,+30 meV` FD run (`134909`) is complete and reproduces the expected Fig. 2(e) logic: opposite-sign peaks near `21 meV` for `mu=+-30 meV` and zero FD at `mu=0`.  A normalization audit indicates that Chaudhary's plotted scale is most consistent with per-flavor response degeneracy `1` while using total-filling degeneracy `4` for the paper's `nu` labels; a wider `eta~10 meV` gives the Fig. 2(e) peak scale.  A Supplement S1-style Delta scan (`134914`) is also complete and reproduces the expected qualitative Delta trends.  Fig. 2(d,e)-style integrand maps (`134941`) are generated as diagnostics.

Hartree/interacting work now has a full-continuum Hartree potential rather than the older active-band projected HF code.  The potential is built from flavor-symmetric two-flat-band density relative to CNP and applied as a scalar Fourier potential in the local continuum basis.  The sharp-occupation `T=0` production diagnostic (`134998/134999`) showed the correct trend but did not converge on the coarse Hartree mesh and was archived.  The converged high-temperature diagnostic is:

```text
135001  T=15K Hartree bands, lg7/m9/eps15 on regular
135002  T=15K dependent Hartree response, lg7/m10/c3/eps15 on regular
```

Key outputs:

```text
results/shift_current_tbg/chaudhary2021_hartree_bands_lg7_m9_eps15_T15K/
results/shift_current_tbg/chaudhary2021_hartree_response_lg7_m10_c3_eps15_T15K/
results/shift_current_tbg/chaudhary2021_hartree_vs_nonint_T15_lg7/
results/shift_current_tbg/chaudhary2021_hartree_response_convergence_T15.md
results/shift_current_tbg/chaudhary2021_hartree_epsilon_scan_T15.md
```

The `15 K` Hartree bands reproduce the central qualitative effect: finite electron/hole filling flattens the same-side flat band and broadens the opposite-side flat band.  At `|nu|=2`, the flattened band bandwidth drops from `~16.4 meV` to `~6.8--6.9 meV`.  The response diagnostic shows FF enhancement from `~1e3` to `~6e3 microA nm V^-2` and shifts the FD weight from the noninteracting `~80 meV` scale to a strong Hartree low-energy feature near `~25 meV`.  A response-mesh/FD-band convergence table and an `epsilon_r=10,15,20,30` scan have been generated; the scan is diagnostic until the paper's fitted screening convention is established.

## Migration to trilayer graphene

The reusable part is the response pipeline:

```text
H(k), dH/dkx, dH/dky -> eigenvectors -> D_nm^a -> r_nm^a and r_nm;a^b -> transition histogram -> sigma(omega)
```

For hTTG/TBG/TTG the only model-specific pieces should be:

1. Hamiltonian builder `H(k)`.
2. Analytic derivative builder `dH/dk`.
3. Band-pair grouping rules: FF, FD, active windows.
4. Optional interaction/Hartree potential provider.
5. Tensor-axis convention audit.

The hTTG code already has items 1--3 for Mao-style runs.  The Chaudhary workflow should therefore become a cleaner template for hTTG: first validate noninteracting FF/FD decomposition in TBG, then apply identical decomposition and quadrature to hTTG active windows.

## Next validation steps

1. Audit the dielectric/screening convention (`epsilon_r`) against Chaudhary/Ref. 65; current `epsilon_r` scan is diagnostic.
2. If quantitative Hartree FD amplitudes are needed, run one higher response mesh (`m14`) or improve quadrature; `m12` still changes the largest FD peak by `~20--30%`.
3. Decide/document whether Fig. 2(c/e) is best represented by nearest-dispersive direct transitions (`fd_bands=1`) or a broader same-side remote-band sum.
4. Refine k-space integrand maps if exact visual matching of Fig. 2(d,e) is needed.
