# Detached postflight audit — partition-independent TSB v7 job 516988

- **Reviewed capsule:** `results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/partition_independent_tsb_v7`
- **Producer job:** `516988`
- **Independent numerical source:** pinned recovered `matched_R6.npz` plus source `regional_integrals_exact.json`
- **Independent execution host:** `test001` (NumPy 1.26.4, one-thread coefficient-space calculation)
- **Reviewed UTC:** `2026-09-22T17:39:20Z`
- **Scope:** finite-R6, `Gz=0`, saved fixed-rank local branch; no SCF, PAO oracle, grid reconstruction, or coefficient change

## Decision

**Pass: the v7 scientific outputs, completion chain, output hashes/modes, and three figures are internally consistent within the declared scope. No blocker was found.**

The independent checker did not import or execute `analyze_tsb.py` or `formula_core.py`, did not read `analysis_arrays_v7.npz`, and did not reconstruct a real-space grid. It recomputed the requested TSB observables directly from the pinned coefficients, then read `analysis_v7.json` only to calculate production-minus-independent differences. The corrected Wigner–Seitz U/L/T values were independently formed from the source regional JSON.

This verdict does not promote finite R6 to an infinite-shell result, the local HF branch to a global ground state, the seven-mode subset to an order claim, or the task owner mapping to canonical A/B/C labels.

## Independent formulas and source boundary

For the saved convention `f(r)=A^-1 sum_Q f_Q exp(-iQ.r)`, coefficient orthogonality gives

- scalar TSB RMS: `sqrt(sum_{Q in TSB}|f_Q|^2)/A`;
- physical-spin TSB RMS: `sqrt(sum_{Q in TSB,a}|S_aQ|^2)/A`, with `S=sigma/2`;
- spin covariance: `C_ab=A^-2 Re sum_{Q in TSB} S_aQ S_bQ*`.

TSB means common determinant-three residue 1 or 2; residue 0 is primitive-periodic. The full covariance principal axis is an unoriented line; the checker reports its representative with positive x component.

For all-pair phase order, each canonical `+/-Q` pair contributes both channels with `w_Q=|n_Q||ell.S_Q|` and `exp(2i phi_Q)=(n_Q s_Q*/|n_Q s_Q*|)^2`. No positive-weight channel was removed.

For the task-defined noncanonical owner assignment `A=owner0`, `B=owner2`, `C=owner1`, the checker used exactly

- `U=A+B+C`,
- `L=(A-B)/2`,
- `T=(A+B-2C)/6`.

The regional source explicitly says its owner labels are not canonical A/B/C labels; therefore these U/L/T quantities remain partition-specific and noncanonical.

## Coefficient-space recomputation

### Residues and TSB RMS normalization

- Residue counts `(0,1,2)`: **`(127,129,129)`**; primitive `127`, TSB `258`.
- Electron TSB RMS: `0.0610285103104921 e/nm^2`; divided by `22/A`: **`0.0605013423282176`**.
- Hole TSB RMS: `0.0610285103104921 e/nm^2`; divided by `2/A`: **`0.665514765610393`**.
- Physical-spin vector TSB RMS: **`0.0637771610254675 hbar/nm^2`**.

The two normalized charge RMS values and residue counts match production exactly. The spin-RMS difference is `1.39e-17 hbar/nm^2`.

### Full spin covariance, eigenfractions, and axis

Independent full covariance:

```text
[[ 2.8277717383674863e-03,  1.7750271860100262e-03, -1.7020943515775400e-06],
 [ 1.7750271860100262e-03,  1.2216000254177527e-03,  2.4569708377413587e-06],
 [-1.7020943515775400e-06,  2.4569708377413587e-06,  1.8154504683174262e-05]]
```

- Eigenfractions descending: **`(0.976744375655352, 0.0188298909679484, 0.00442573337669914)`**.
- Positive-x principal-axis representative: **`(0.840300360010447, 0.542121115940486, -2.48546698757456e-05)`**.
- Production differences: covariance max `2.17e-18`, eigenvalue max `2.60e-18`, principal-axis max `1.11e-16`, and `lambda1/trace` difference `3.33e-16`.

The leading eigenspace is strongly isolated (`lambda1/trace≈0.977`), so the reported axis is numerically meaningful; its overall sign remains conventional.

### Seven-mode comparison

The independent subset is exactly Q0 plus six shell-1 TSB channels; TSB RMS/covariance/z2 use the six nonzero-residue channels.

- Selected/TSB counts: `7/6`.
- Seven-mode eigenfractions: **`(0.978331111623999, 0.0196109285299626, 0.00205795984603893)`**.
- Seven-mode positive-x axis: `(0.840300431192647, 0.542120993073900, -1.19189013863606e-04)`.
- Seven/full RMS ratios `(electron,hole,spin)`: **`(0.840703253703151, 0.840703253703151, 0.824251877636718)`**.
- Absolute full/seven axis dot: **`0.999999995550506`**.
- Seven-minus-full `lambda1/trace`: `+0.00158673596864634`; `lambda3/trace`: `-0.00236777353066021`.

All selected counts match exactly. Comparison-field differences versus production are at most `3.33e-16`; covariance and axis differences are at most `2.17e-19` and `1.11e-16`, respectively. These values support only the declared comparison, not a seven-mode-order claim.

### All-pair z2

- Full inventory: **129 canonical pairs, 258 positive-weight channels, zero exact-zero-weight channels**.
- Full total weight: `1.8275246682059592`.
- Independent full `z2`: **`-0.9987116872910504 - 8.503308072312686e-17 i`**.
- Production full `z2` absolute difference: **`2.22e-16`**.
- Seven-mode `z2`: `-0.9999619216248740 - 1.975353090226601e-16 i`; production difference `2.22e-16`.
- Maximum reverse-conjugacy residues are `4.34e-16` for charge and `3.39e-16` for longitudinal spin.

Thus the v7 value is genuinely an **all-pair** result: both members of every one of the 129 canonical full-R6 TSB pairs enter. The plot contract permits exact-zero rows, but this data set has none.

### Corrected Wigner–Seitz U/L/T

Independent source vectors in hbar are

```text
A = (-0.2946147717171304, -0.1570997754725885,  3.75054198310424e-05)
B = ( 0.2645749338074520,  0.2036622336013206, -5.84035272724702e-05)
C = ( 0.0304342412260389, -0.0471737792297733,  5.28923210275486e-05)
U = ( 0.0003944033163605, -0.0006113211010412,  3.19942135861209e-05)
L = (-0.2795948527622912, -0.1803810045369546,  4.79544735517563e-05)
T = (-0.0151513867269594,  0.0234850027647131, -2.11137915827542e-05)
```

- `|U| = 0.000728210885827280 hbar`.
- `|L| = 0.332732010465720 hbar`.
- `|T| = 0.0279483509424198 hbar` and `|T|/|L| = 0.0839965800203623`.
- `angle(A,B) = 170.480181321581 deg`.
- Raw sign-fixed `angle(L,axis) = 179.993166398665 deg`; sign-invariant acute line angle `0.00683360133525 deg`.
- U/L/T vectors, norms, ratio, and `angle(A,B)` match production exactly. The axis-angle difference is `1.07e-10 deg`, caused by arccos amplification near 180 degrees from a principal-axis component difference of only `1.11e-16`; it is not a physical discrepancy.
- Inverse U/L/T reconstruction max residual: `5.55e-17 hbar`.

## Output hashes, modes, and completion chain

A fresh `sacct` query records job `516988` as `COMPLETED`, exit `0:0`, on account `hmt03`, partition `regular256`, node `node034`, from `01:07:54` to `01:08:18` (24 s), with 64 allocated CPUs. The batch step is also `COMPLETED 0:0`; `MaxRSS=778460K`. Exact rows are in `SACCT_JOB516988.txt`.

Artifact closure checks pass:

- Pinned coefficient and regional JSON hashes match CONFIG, completion, and analysis bindings.
- Output directory has the exact 12-entry namespace: ten manifest payloads plus the output manifest and completion record.
- All ten manifest payload SHA-256 values and sizes match; all are regular nonsymlink files at mode `0444`.
- Output directory mode is `0555`; output manifest and completion record are both `0444`.
- Output manifest SHA-256 `d129112501964c4aee3c4a2be6c6da25a3e3755e42a94989f9bc5edd9181bf76` matches both completion and sentinel.
- Completion-record SHA-256 `7f2fe81ff2ea72b4176efa0aa7bb818df52e0ae4c11d010092ebdac536807d39` matches the sentinel.
- The eight execution bindings are identical across output manifest, completion record, completion sentinel, and analysis JSON; every referenced approval/control/runtime file independently hashes to its bound digest.
- Slurm stderr is zero bytes.
- `ANALYSIS_COMPLETE_V7` is write-bit-free at mode `0400`, points to the exact output directory, and asserts the sentinel-last namespace operation.

The sentinel's `0400` mode is less broadly readable than `0444`, but no frozen v7 contract requires a specific sentinel mode; v6 uses the same `0400` pattern. It does not break owner-side verification or immutability. If cross-user readability is desired later, a successor should explicitly chmod its sentinel after creation rather than mutating v7.

## Visual figure audit

All three PNGs were opened and inspected, and their dimensions, embedded software tag, hash, size, and `0444` mode were checked against the output manifest:

- `tsb_charge_3x3.png` — `2555x1258`, signed electron/hole panels with matching `e nm^-2` units, diverging scales, oblique exact 3x3 tiling, and the expected opposite electron/hole TSB pattern. No visible seam, stale marker, clipping, or panel mismatch.
- `tsb_total_physical_spin_3x3.png` — `1254x1558`, nonnegative `|S|` color map in `hbar nm^-2` with in-plane arrows on the same oblique 3x3 tiling. No visual discontinuity or unit/title mismatch.
- `covariance_phase_diagnostics.png` — `3168x2338`, full/seven covariance spectra and axes agree with the independently recomputed values; the phase panel is explicitly all canonical pairs with no cutoff, and the ranked panel spans all 129 pairs. Tiny weights are compressed by the linear vertical scale but not omitted.

Each embeds `Software=source-bound partition_independent_tsb_v7`; all hashes exactly match the manifest. This is visual and byte-identity QA only. Under the requested coefficient-space-only boundary, the pixel fields were not independently regenerated.

## Reproducibility and limitations

- Independent code: `recompute_coefficients.py`.
- Full machine-readable results and production deltas: `RECOMPUTE.json`.
- Test-node/runtime identity: `RECOMPUTE_PROVENANCE.txt`.
- Fresh accounting: `SACCT_JOB516988.txt`.
- The recomputation was tiny (385 coefficients and 3x3 eigensystems) and ran on `test001`; no coefficient recomputation or BLAS/eigensolver ran on a login node. It completed in seconds, so tmux was unnecessary.
- No v7 source, control, runtime, output, sentinel, or metadata byte was modified. Only this detached postflight directory was created.
