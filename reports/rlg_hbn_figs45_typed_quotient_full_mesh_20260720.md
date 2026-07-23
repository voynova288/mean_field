# RLG/hBN Fig. S45 typed-quotient full-mesh result (2026-07-20)

## Status

The corrected `actual_node_ws_fixed_variational_copy_v2` HF/TDHF chain has now completed the full actual 12x12 finite-q calculation, including independently assembled signed q/-q matrices, raw eigenspectra, eta norms, solver residuals, and all three separated channels.

**The validated calculation does not quantitatively reproduce the published Fig. S45 bottom row.** This is a physics-level mismatch, not a plotting issue. No energy rescaling, mode replacement, smoothing, matrix symmetrization, or paper-spectrum filtering was applied.

## Accepted source and implementation

- HF archive: `results/RnG_hBN/tdhf_m2_pilot/a_v64_variational_hf_v2_185536/checkpoint_A_average_V64_hf/xi1_V064meV/runs/flavor_seed1/hf_run_state.npz`
- HF interaction: `actual_node_ws_fixed_variational_copy_v2`
- basis gauge: `c3_equivariant_reciprocal_relabel_fixedrep_v3`
- ordinary wrapped legs: `analytic_periodic_gauge_relabel_v1`
- fixed-fixed rule: `same_puncture_copy_v1`, three branches of weight 1/3
- public API: `build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs`
- mesh: 12x12; channels: intraflavor, intervalley, interspin
- orbit reduction: 26 C3+inversion representatives, expanded deterministically to all 144 torus points per channel

The repeated-zone plot coordinates are Cartesian physical q, componentwise centered, and not Wigner-Seitz folded. Periodic copies are used only to display the same finite torus in the published viewing window.

## Slurm provenance

- `190204`: production runner q=(1,0) smoke, completed in 01:05:15
- `190206`: current signed-matrix-pair q=0 smoke, completed in 00:32:09
- `190207`: 77 remaining array tasks, all completed; no failed/OOM/timed-out tasks
- `190208`: deterministic full-mesh merge, completed in 00:00:06

## Full-mesh gates

From `figs45_full_orbit_merge_summary.json`:

- maximum A/B structure residual: `7.021666937153402e-16 meV`
- maximum q/-q particle-hole assignment residual: `4.7749141014527956e-12 meV`
- maximum quartet residual: `4.7749141014527956e-12 meV`
- maximum raw eigensolver residual:
  - intraflavor: `5.986200870567597e-12`
  - intervalley: `3.130441719459884e-12`
  - interspin: `5.219305378830945e-12`
- Cartesian reciprocal-basis fit residual: `1.6653345369377348e-16 nm^-1`

Raw stability classification:

| channel | stable points | unstable points | raw minimum physical energy (meV) | raw maximum (meV) |
|---|---:|---:|---:|---:|
| intraflavor | 141 | 3 | 0.3281994304 | 11.1064629679 |
| intervalley | 144 | 0 | 2.7234582834 | 8.5246836366 |
| interspin | 144 | 0 | 2.1446636e-7 | 6.1460191623 |

The three calculated intraflavor instabilities are the C3-related M points `(-6,-6)`, `(-6,0)`, and `(0,-6)`. At representative `(-6,-6)`, the complex pair is

```text
+q: -2.89334e-7 +/- 1.18384792624 i meV, eta approximately 0
-q: +2.89334e-7 +/- 1.18384792624 i meV, eta approximately 0
```

The tiny real part is at the accepted source-stationarity scale and was not converted into an excitation energy.

## Published-raster extraction

The arXiv PDF contains six embedded 1200x1200 JPEG panel images rather than vector marker data. The bottom row was extracted with `pdfimages` and digitized by:

1. calibrating physical q to image pixels from the six mBZ-hexagon edges;
2. identifying colored versus white markers at all 144 torus sectors using periodic copies;
3. converting marker RGB values using each panel's printed colorbar tick rows;
4. checking repeated-copy consistency.

Source:

- PDF: `reference/2312.11617v1.pdf`
- PDF SHA256: `c01d496e59a8909a3d4ce57f29ca5d817e838d915f13fd7f83365692eeb04d2f`
- provenance class: published embedded JPEG raster, not raw numerical data
- maximum repeated-copy digitization spread: `0.044 meV`

The PDF text explicitly states that white regions are negative or complex modes and that the kappa_hBN=1 instability lies around the M points, with a lower-energy doubled-unit-cell HF state.

## Direct quantitative comparison

All differences below are `validated calculation - digitized paper`; there is no fitted offset or scale.

| channel | paper unstable | calculated unstable | common stable points | mean difference (meV) | RMSE (meV) | max abs difference (meV) |
|---|---:|---:|---:|---:|---:|---:|
| intraflavor | 45 | 3 | 99 | +1.64098 | 1.69112 | 2.56191 |
| intervalley | 0 | 0 | 144 | +0.71209 | 0.73014 | 0.94818 |
| interspin | 0 | 0 | 144 | +0.77015 | 0.77488 | 0.87716 |

The interspin q=0 Goldstone itself agrees: calculated `2.145e-7 meV`; the raster floor digitizes to about `0.034 meV`. The discrepancy develops at finite q.

## Legacy-source diagnostic (not accepted)

The historical untyped archive

```text
results/RnG_hBN/tdhf_m2_pilot/a_v64_formfactor_v2_148813/...
```

uses `centered_cell_reciprocal_relabel_pad1_v2`, lacks typed quotient provenance, and fails the accepted fixed-node/C3 source contract. It cannot be promoted to a physical result. Nevertheless, its old flavor-flip maps are much closer to the raster:

| channel | legacy-paper mean (meV) | legacy-paper RMSE (meV) | correlation |
|---|---:|---:|---:|
| intervalley | +0.10103 | 0.21428 | 0.99361 |
| interspin | +0.06754 | 0.15110 | 0.99572 |

Its intraflavor map still fails (`RMSE=1.42776 meV`, correlation `0.5391`) and does not reproduce the paper instability mask. This diagnostic suggests that much of the flavor-channel discrepancy is source/basis-convention dependent, but it does not justify reverting to the non-C3 archive.

## Source-level causal follow-up

The follow-up audit `reports/rlg_hbn_figs45_source_convention_discriminator_20260722.md` localizes the dominant difference to the filled-remote-band average-scheme source Hamiltonian. Between the legacy and accepted basis caches, the physical projected `h0` local spectra change by only `0.0509 meV` maximum, whereas the remote-band term changes by `12.436 meV` maximum and `3.926 meV` mean local maximum. The legacy source also has a gauge-invariant C3 block-spectrum defect of `8.876 meV`.

The paper raster itself is C3-closed to `<=0.0153 meV` digitization residual, so the legacy archive is only a proxy, not the authors' exact implementation. No public raw Fig. S45 arrays or calculation repository were found. The unresolved discriminator is the authors' finite-cutoff periodic-gauge/remote-Fock and fixed-point prescription.

Job `192593` subsequently excluded a missing fixed-target one-body chain rule: independently rebuilt physical/remote target copies satisfy `L^sharp({H_r})=sum_r S_r^dagger H_r S_r/3` with total-`h0` residual `<=3.54e-12 meV`. The paper's stated `4|q1| -> 19 RLV` convention also supports the current fixed-`|G|` shell interpretation; a dynamic `|q_WS+G|` cutoff is not justified as a paper correction without author input.

## Response-kernel physical follow-up

The mode-level audit `reports/rlg_hbn_figs45_response_kernel_physics_discriminator_20260722.md` revises the causal interpretation. On the same accepted stationary HF source, the conventional single-representative intervalley exchange kernel reduces the 26-representative paper RMSE from `0.7021` to `0.0777 meV`; its mean energy shift relative to the quotient branch is `-0.6486 meV`. The conventional q=0 intervalley mode is nearly uniform (`IPR*N_k=1.006`) with `1.16%` fixed-endpoint weight, while the typed quotient mode has essentially exact nodes at the two nonzero C3-fixed points (`2.8e-29` weight), contrary to the paper's Fig. S44 description.

This identifies quotient exclusion of the nearly uniform fixed-endpoint-coupled state as the leading intervalley mismatch. The conventional and typed lowest states undergo mode reordering; this is not a perturbative shift of one unchanged eigenvector. The conventional hybrid is still not a result: its full A-spectrum C3 residual is `0.0130 meV`, and on the quotient-stationary source it opens an interspin q=0 gap of `1.591e-3 meV`, so source/Hessian consistency is lost. A matching single-representative HF functional and fresh source must be derived before any production rerun.

## Conclusion

All current-functional, API, raw-spectrum, q/-q, C3, Goldstone, and full-mesh gates pass. The remaining failure is the paper comparison itself:

- the paper has 45 intraflavor unstable torus sectors; the corrected functional has only the three exact M sectors;
- the validated intervalley/interspin finite-q energies are systematically about 0.7-0.8 meV too high.

Therefore the generated figures are labeled **validated calculation** and **comparison**, not “Fig. S45 reproduction.” A genuine reproduction now requires an independently justified paper fixed-node/projected-basis prescription or the authors' raw code/data. Reintroducing the old single-chart/fixed-copy behavior, fitting a scale/offset, or changing the instability mask in postprocessing would violate the validated functional contract.

## Artifacts

```text
results/RnG_hBN/tdhf_m2_pilot/figs45_production_full_orbits_v1_20260719/
  orbit_manifest.json
  figs45_full_orbit_merge_summary.json
  figs45_full_orbit_merged_raw.npz
  figs45_bottom_raster_comparison.json
  figs45_bottom_raster_digitized.npz

reports/figures/
  rlg_hbn_figs45_kappa1_validated_calculation_paper_style_20260720.png
  rlg_hbn_figs45_kappa1_validated_vs_paper_raster_20260720.png
  rlg_hbn_figs45_kappa1_validated_vs_paper_raster_paper_markers_imag_20260722.png
  rlg_hbn_figs45_kappa1_validated_calculation_paper_markers_imag_20260722.png
```
