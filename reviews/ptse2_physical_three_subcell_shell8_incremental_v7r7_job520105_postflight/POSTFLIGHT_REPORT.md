# Detached postflight audit — PtSe2 shell-8 incremental V7R7 job 520105

- **Reviewed capsule:** `results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_three_subcell_shell8_incremental_v7r7`
- **Producer job:** `520105`
- **Independent numerical inputs:** published merge-checkpoint `matched_R6.npz`, `matched_R7.npz`, and `matched_R8.npz`
- **Independent execution host:** `test001` (Python 3.11.15, NumPy 1.26.4, one-thread BLAS environment)
- **Scope:** finite R6/R7/R8, `Gz=0`, fixed `384x384` analysis grid, one accepted local fixed-rank B1 branch

## Decision

**Pass for immutable publication closure and internal numerical consistency. No postflight blocker was found.** The merge/final sentinels, exact namespaces, manifests, hashes, modes, hardlink identities, and analyze-only runtime-map receipts verify. An independent NPZ-first calculation reproduces inventories, TSB RMS values, covariance eigensystems, all-pair `z2`, exact analytic regional integrals, U/L/T/M_C, and all reported pointwise shell differences to floating-point roundoff.

**This is not a shell-convergence pass.** R7-to-R8 changes are generally larger than R6-to-R7 changes, so the data do not support R8 sufficiency, monotone convergence, or infinite-shell extrapolation. The production report correctly makes none of those claims.

The detached checker did not import or execute `formula_core.py` or `analyze_shells.py`, and did not read `ANALYSIS_ARRAYS.npz` or `POINTWISE_FIELD_DIFFERENCES.npz`. It independently loaded only the three matched NPZ files, formed the formulas, reconstructed the grid, and then compared against the published CSV/JSON outputs.

## Fresh terminal accounting

A fresh `sacct` query on `login002` at `2026-09-24T05:37:19Z` records:

- parent `520105`: `COMPLETED`, `ExitCode=0:0`, elapsed `00:00:35`;
- account `hmt03`, partition `regular256`, node `node035`, 64 allocated CPUs, requested memory `250G`;
- scheduler start/end `2026-09-24T13:14:05` / `13:14:40`;
- batch step `COMPLETED 0:0`, `MaxRSS=809144K`.

Exact rows are retained in `SACCT_JOB520105.txt`. Slurm stderr is zero bytes.

## Merge/final publication closure

### Merge checkpoint

- Exact namespace: five manifest payloads plus `OUTPUT_MANIFEST.json`; no missing or foreign entry.
- Payloads and manifest are regular nonsymlink files at mode `0400`; directory mode is `0500`.
- Every payload size and SHA-256 matches the manifest.
- Merge manifest SHA-256: `5b7c1c8c6e9b75550af988ef22dbfceaa91e8b86ca6d9be423fbeef52a28d0c6`.
- Merge sentinel SHA-256: `edd2c86ea71c92494c51b121221d521fa089f65c0ee047759cd58447d83d1737`.
- `R6_R7_R8_MERGE_COMPLETE_520105` is mode `0400`, has link count 2, binds the manifest and all three matched NPZ hashes, and is inode-identical to the retained staging sentinel. The staging directory contains only that sentinel; payload staging links were removed.

### Final analysis

- Exact namespace: 20 manifest payloads plus `OUTPUT_MANIFEST.json`; no missing or foreign entry.
- Payloads and manifest are regular nonsymlink files at mode `0400`; directory mode is `0500`.
- Every payload size and SHA-256 matches the manifest.
- Final manifest SHA-256: `7ed657d89be4d2db01bc66888794e3d8613ae82187b8a3be118bfdbff0d1adf1`.
- Final sentinel SHA-256: `00f88eb98da1cc90078b9c7d548f48f40c0002940ba6ef1be0f7295c703e80cd`.
- `SHELL8_ANALYSIS_COMPLETE_520105` is mode `0400`, has link count 2, is inode-identical to its retained staging sentinel, and correctly binds the final manifest, summary, report, merge sentinel, projection sentinel, and frozen source.

The retained staging-sentinel hardlinks are consistent with the frozen direct-return sentinel-last protocol; they are not duplicate payload trees.

## Runtime-map receipts

The finalized runtime receipt has SHA-256 `f271a0ccb97c757395c8ec30b1265c8d6e91f7660ea90071915583cca31aebc0` and declares `passed=true`.

All four production map receipts verify:

| profile/stage | required core count | required-core digest | full-map digest |
|---|---:|---|---|
| analyze_merge / after imports | 51 | `5a9b8578...d08cb1` | `12cd093e...28881c` |
| analyze_merge / lazy prepublish | 51 | `5a9b8578...d08cb1` | `12cd093e...28881c` |
| analyze_final / after imports | 77 | `05bec5c9...1a039` | `de704ad9...2dd437` |
| analyze_final / lazy prepublish | 79 | `812d8d39...ce512` | `8e0c7544...9b019` |

For each receipt, the detached checker verified the external receipt hash, runtime-receipt binding, required-core count/digest, audited-extra digest, and full-map digest reconstructed from the sorted core-plus-extra records. The sole audited extra is `_csv.cpython-311-x86_64-linux-gnu.so`; its recorded size and SHA-256 still match. Host/job/rank are consistently `node035/520105/0`, exact argv distinguish merge from final, and `openmx_process_receipts=null` is correct for these analyze-only profiles.

## Independent inventory

The independently enumerated determinant-three disks reproduce:

| shell | channels | residues `(0,1,2)` | canonical TSB pairs | weighted channels |
|---|---:|---:|---:|---:|
| R6 | 385 | `(127,129,129)` | 129 | 258 |
| R7 | 535 | `(187,174,174)` | 174 | 348 |
| R8 | 691 | `(241,225,225)` | 225 | 450 |

All label sets are unique, reverse closed, and C3 closed. R6 is a strict prefix/subset of R7, and R7 of R8; the increments are 150 and 156 channels, or 306 relative to R6. Every NPZ inventory/order matches the independently generated order exactly. `spin_pauli_fourier == 2*spin_physical_fourier` is exact in every NPZ. All matched R6 coefficient arrays are exact prefixes of R7 and R8.

## TSB RMS, covariance, and all-pair z2

The coefficient convention is `f=A^-1 sum_Q f_Q exp[-2 pi i(l1 x+l2 y)]`. TSB means residue 1 or 2. Charge RMS is `sqrt(sum_TSB |f_Q|^2)/A`; physical-spin RMS also sums Cartesian components. Covariance is `C_ab=A^-2 Re sum_TSB S_aQ S_bQ*`. The principal-axis sign is fixed to positive x.

| R | electron/hole TSB RMS | physical-spin TSB RMS | covariance fractions descending | principal axis `(x,y,z)` |
|---|---:|---:|---|---|
| 6 | `0.0610285103104921` | `0.0637771610254675` | `(0.976744375655352, 0.0188298909679484, 0.00442573337669914)` | `(0.840300360010447, 0.542121115940486, -2.48546698757456e-5)` |
| 7 | `0.0618501920250041` | `0.0649072332025061` | `(0.975400673026701, 0.0191623742832477, 0.00543695269005132)` | `(0.840300345294975, 0.542121139190821, -1.18149675696684e-5)` |
| 8 | `0.0638963642566916` | `0.0674268276650740` | `(0.974106253659446, 0.0193086110735677, 0.00658513526698621)` | `(0.840300333886731, 0.542121156991224, -3.51369659735231e-6)` |

RMS values match production exactly. The maximum covariance/eigenfraction/axis discrepancy is `3.99e-16`.

All canonical pairs enter with both `+Q` and `-Q`; no exact-zero-weight channel occurs. Independent values are:

- R6: `z2 = -0.9987116872910504 - 8.503308072312686e-17 i`;
- R7: `z2 = -0.9985815859936038 - 6.389155826798994e-17 i`;
- R8: `z2 = -0.9984158647622121 - 7.897897618825025e-17 i`.

Absolute differences from production are `2.22e-16`, `3.37e-16`, and `5.55e-16`. The near-`-1` phase statistic and stable principal axis are robust finite-shell observations, but they do not establish convergence of amplitudes or fields.

## Exact regional integration and U/L/T/M_C

The detached checker independently rebuilt parallelogram and Wigner–Seitz polygons and evaluated each Fourier-mode polygon integral analytically; no raster quadrature or production regional table was used. Maximum production discrepancy over all 90 regional values is `8.88e-16`. Partition-weight closure is at most `2.22e-16`, owner-Q0 area error at most `1.67e-16`, and regional imaginary residues at most `8.94e-17`.

For the explicitly provisional assignment `A=owner0`, `B=owner2`, `C=owner1`, all vectors below are in `hbar`:

```text
R6:
 A = (-0.294614771717130, -0.157099775472589,  3.75054198310416e-05)
 B = ( 0.264574933807452,  0.203662233601321, -5.84035272724692e-05)
 C = ( 0.030434241226039, -0.047173779229773,  5.28923210275480e-05)
 U = ( 0.000394403316361, -0.000611321101041,  3.19942135861203e-05)
 L = (-0.279594852762291, -0.180381004536955,  4.79544735517554e-05)
 T = (-0.015151386726959,  0.023485002764713, -2.11137915827539e-05)

R7:
 A = (-0.298486133363816, -0.159205016275781,  3.77761495454379e-05)
 B = ( 0.268088808953249,  0.206321588111000, -5.94128233222091e-05)
 C = ( 0.030791727726927, -0.047727892936260,  5.36308873628889e-05)
 U = ( 0.000394403316361, -0.000611321101041,  3.19942135861177e-05)
 L = (-0.283287471158533, -0.182763302193391,  4.85944864338235e-05)
 T = (-0.015330129977404,  0.023762059617956, -2.14830747504248e-05)

R8:
 A = (-0.291062774290392, -0.155224189329624,  3.71491524358992e-05)
 B = ( 0.261401941551376,  0.201199182018344, -5.76010965535705e-05)
 C = ( 0.030055236055377, -0.046586313789762,  5.24461577037884e-05)
 U = ( 0.000394403316361, -0.000611321101041,  3.19942135861172e-05)
 L = (-0.276232357920884, -0.178211685673984,  4.73751244947349e-05)
 T = (-0.014961884141628,  0.023191270044707, -2.08907099208747e-05)
```

Derived values are:

| R | `|L|` | `|T|/|L|` | `|M_C|` | `M_C` parallel to A-B |
|---|---:|---:|---:|---:|
| 6 | `0.332732010465720` | `0.0839965800203623` | `0.0561392134200484` | `-2.92377026e-9` |
| 7 | `0.337126412945431` | `0.0838797006278108` | `0.0567986367513906` | `-2.23783847e-9` |
| 8 | `0.328730471233953` | `0.0839556908325662` | `0.0554400991833711` | `-2.21759723e-9` |

The maximum U/L/T/M_C production discrepancy is `4.44e-16`. `M_C` is almost purely transverse to A-B, not a zero vector. These quantities remain Wigner–Seitz-partition-specific and the owner-to-A/B/C assignment remains noncanonical.

## Pointwise shell differences

Independent `384x384` reconstruction reproduces every published maximum, RMS, and imaginary residue exactly at the saved float64 level (`maximum comparison discrepancy = 0.0`). Key real-field differences are:

| pair | observable | maximum absolute | RMS |
|---|---|---:|---:|
| R6→R7 | electron charge | `0.3626323554` | `0.1313527990` |
| R7→R8 | electron charge | `0.8866616173` | `0.2824768138` |
| R6→R8 | electron charge | `1.1217540814` | `0.3115232064` |
| R6→R7 | hole charge | `0.1871036348` | `0.0307350416` |
| R7→R8 | hole charge | `0.2377581472` | `0.0400705647` |
| R6→R7 | spin x | `0.0572035181` | `0.00987429065` |
| R7→R8 | spin x | `0.0938940258` | `0.0150845856` |
| R6→R7 | spin y | `0.0394152675` | `0.00655524238` |
| R7→R8 | spin y | `0.0647018397` | `0.00993946960` |
| R6→R7 | spin z | `0.0101656635` | `0.00222676320` |
| R7→R8 | spin z | `0.0129233833` | `0.00266463535` |

Maximum imaginary residues are at most `3.34e-16`. Electron TSB RMS rises by `1.35%` from R6→R7 and `3.31%` from R7→R8; physical-spin RMS rises by `1.77%` then `3.88%`. Likewise every listed R7→R8 pointwise maximum and RMS exceeds its R6→R7 counterpart. `|L|` and `|M_C|` also reverse direction at R8, with the final change larger in magnitude. This is clear nonmonotone finite-shell behavior and is the principal scientific limitation.

## Figure and interpretation audit

All four PNGs opened successfully, have valid PNG headers, are mode `0400`, and match the output-manifest hashes and sizes.

- `tsb_rms_convergence.png` (`1122x726`) correctly shows increasing finite-shell RMS. Electron and hole curves coincide exactly, so the orange hole curve hides the blue electron curve; the generic y-axis also places charge (`e/nm^2`) and spin (`hbar/nm^2`) on one numerical axis without explicit units. This is a presentation limitation, not a data mismatch.
- `spin_covariance_convergence.png` (`1513x686`) correctly shows a dominant first eigenfraction and nearly unchanged axis. The three axis curves overlap almost exactly, so shell differences are not visually resolvable; the CSV remains authoritative.
- `pointwise_field_differences.png` (`1853x1093`) displays the expected oscillatory shell-increment fields with no visible seam or clipping. Each panel is independently autoscaled and axes are pixel indices without physical-coordinate/unit labels, so color intensity cannot be compared quantitatively across panels. The independently reproduced CSV values provide the quantitative comparison.
- `ws_tl_mc_convergence.png` (`1583x726`) correctly shows nonmonotone `|T|/|L|` and `|M_C|`. Its tightly truncated y-ranges visually magnify small changes; it must not be read as evidence of asymptotic convergence.

The written production interpretation is appropriately conservative: it explicitly disclaims new thresholds, monotonicity, extrapolation, R8 sufficiency, unique partition, and magnetic-order labels. No unsupported positive convergence claim was found. The external job-517203 focused seven-label hole-operator crosscheck is correctly described as separate evidence, not an inline or all-new306 hole-spin calculation; no full-annulus hole-spin authority should be inferred from it.

## Reproducibility and mutation boundary

- Independent checker: `recompute_shells.py`.
- Machine-readable values, closures, and production deltas: `RECOMPUTE.json`.
- Test-node runtime identity: `RECOMPUTE_PROVENANCE.txt`.
- Fresh terminal accounting: `SACCT_JOB520105.txt`.
- No numerical work ran on a login node. The small independent recomputation ran on idle `test001` with one-thread numerical libraries.
- No V7R7 source, control, runtime, checkpoint, output, sentinel, Slurm log, or metadata byte was modified. Only this detached postflight directory was created.
