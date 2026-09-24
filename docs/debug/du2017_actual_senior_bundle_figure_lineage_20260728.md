# Du2017 actual senior bundle: calibrated Fig. 2 lineage audit

**Date:** 2026-07-28

**Bundle:** `reference/teib_calibrated_hf_bcs_20260728_1755.tar.gz`

**Outer SHA-256:**
`e6132e428051d63f7a7a81dac3e1a60b346b01a7ae424a880d24c178caef69da`

**Status:** the historical calibrated Stage-2 source and figure lineage are now
reproducible; physical/continuum certification still fails closed.

## Executive conclusion

The successful-looking Fig. 2 is not a digitized curve replay. It is a genuine
self-consistent fixed point of the archived calibrated equations. However, the
model contains explicit Fig. 2-calibrated dispersion, interaction, vertex,
branch-density, chemical-potential, and diagonal-quadrature conventions.

The final plotted curve is additionally generated from off-grid C2/PCHIP
interpolation of a coarse 71-node solution. The resulting plotted energy and
gap match the archived PDF-vector data with RMS errors of only approximately
`0.044 meV` and `0.045 meV`, respectively.

This is therefore a high-accuracy **calibrated curve reconstruction**, not a
parameter-free Kane-Poisson/HF/BCS prediction.

## Archive integrity and mixed source states

> **Source-lineage correction:** this report audits the restored historical
> calibrated source present in the supplied bundle. It must not be identified
> as the senior author's globally latest code. The user recalls a newer
> no-channel-scale version that may still match Fig. 2; that exact source is not
> present in the current workspace and remains unverified.

### No-scale routes must not be conflated

Two locally available routes omit the separate `4.2/2.09` channel scales:

1. The old `run_bcs.py --mode paper` route defaults to `paper` and calls
   `solve_paper_calibrated`. Its saved NPZ explicitly records
   `calibration_kind="digitized PDF vector paths"` and
   `prediction_status="paper-calibrated reconstruction"`. It matches Fig. 2
   without channel scales because it reads the digitized target curves. This
   route is **REJECTED / WRONG as physical reproduction, BCS/HF validation, or
   prediction**; it is allowed only as a labelled digitization/reference tool.
2. `current_uncalibrated_20260728_1750` uses unit interaction strength and a
   fixed-density closure. It is an actual solve, but gives
   `E_min=3.627 meV @ 0.0601 nm^-1` and `Delta_max=3.741 meV`, so it does not
   match Fig. 2.

The user's recalled newer no-channel-scale implementation would be a third
route. Its exact source must be obtained before deciding whether it represents
a new physical closure, a reparameterization of the same calibration, or a
plotting/reference path.

The archive is path-safe: 118 members, 101 regular files, 17 directories, no
links, devices, duplicate paths, or traversal paths.

`SOURCE_SHA256SUMS.txt` is stale. Fifteen listed source/document/test hashes do
not match the restored calibrated files. The top-level README and reference
equations describe the later uncalibrated fixed-density workflow rather than
the executable calibrated defaults.

The active restored files do match the hashes recorded by
`direct_calibrated_historical_20260728_1800.log`:

```text
run_all.sh             79ebfa6452415d784498ead87d3ea18a16f46e26b2b59c5abef92400b217e1e7
run_nonint.py          01a79322fe3354a77df79f3a17ebd98acc9b82e6f909959a9bdeb11d302e9b3d
run_bcs.py             2a3a9570f63bece1980f33435fdf97d7c14e9debdf3b07f54eb215ae6f8d9278
src/excitonic_bcs.py   91cb9a797213925a38f0fb76d2a68aab77e3ebb7c34f9f41f0fd0f50c6ab15cf
postprocess.py         eb9343c3dc8d8a1a646e749ab23b5741d7bc327141c7d90a74dcdfc11d6a80b8
```

The archived historical figure and validation also match the run log:

```text
figures/fig2ab_calibrated_historical_20260728_1800.png
32c2f44b6b9cf4a017d029d1003949bfb21b280e3d733a913f1da61849bef90f

validation_calibrated_historical_20260728_1800.json
34ff57e5b4771dbb3284d829685b2416904ec94e7611a943c1381acf4e041ea7
```

`fig2ab_reproduction_kane_bcs.png` is byte-identical to the historical
calibrated PNG.

## Test and equation replay

The bundled test suite passes on Test001:

```text
89 passed in 15.13 s
```

Using the archived 71-point bare Kane E1/H1 arrays and the restored source, the
historical Stage-2 result replays to near machine precision:

```text
max |Delta replay - archive|       6.94e-18 eV
max |Sigma replay - archive|       3.99e-17 eV
max |xi replay - archive|          2.22e-16 eV
max |E replay - archive|           2.22e-16 eV
max occupation difference          5.39e-15
```

Thus the calibrated Stage-2 numerical fixed point is real and internally
reproducible. This does not validate its parameter provenance or continuum
limit.

## How the successful figure was made

### 1. A self-consistent but uncertified N=25 Kane-Poisson parent

The historical run first produces

```text
plane-wave N                 25
Kane radial points           81
z points                     512
periodic Poisson residual    9.92e-8 eV
Kane-Poisson density         1.485e11 cm^-2
```

This parent is not the experimental `5.5e10 cm^-2` closure and is not the
source-certified UV/device completion required by the main project.

### 2. Rebuild E1/H1 on a coarse node grid

The calibrated route rebuilds the diabatic E1/H1 branches on

```text
k = 0, 0.005, ..., 0.35 nm^-1
nk = 71
```

It then uses direct radial-node weights

\[
w_j = \frac{k_j\Delta k}{2\pi},
\]

so `w0=0`. The diagonal Coulomb entries are separately multiplied by
`diagonal_scale=0.782`.

### 3. Add explicit empirical pair-dispersion corrections

The raw Kane pair dispersion is changed by

\[
C_{\rm pair}(k)
=16.60946 k^2
+0.01840197\left[e^{-(k/0.02869179)^2}-1\right]
-0.00015\,R_{0.025,0.008}(k),
\]

where the final symmetric ring term is centered at exactly
`0.025 nm^-1`, the published minimum scale.

Representative corrections are

```text
k=0.000   0.000 meV
k=0.005  -0.135 meV
k=0.025  +0.442 meV
k=0.035  +6.069 meV
```

### 4. Use an empirical momentum-dependent pairing vertex

The interlayer gap kernel is multiplied at both ends by

\[
z(k)=0.38+0.62\left[1-e^{-(k/0.019)^2}\right].
\]

Hence

```text
z(0)       = 0.380
z(0.025)   = 0.890
z(0.035)   = 0.979
```

This suppresses the gap near the origin and lets it peak near the ring. It is
not derived from archived Kane wavefunction matrix elements.

### 5. Enhance the Coulomb channels

The solver uses

```text
intralayer scale   4.2
interlayer scale   2.09
diagonal scale     0.782
```

These are not equivalent to one common dielectric constant and are explicitly
recorded in the output as effective calibration conventions.

### 6. Select a low-density branch, then fix an empirical pair-sum offset

A density homotopy at

```text
branch seed density  9.4e9 cm^-2
```

selects the excitonic branch. The density constraint is then removed, and a
fixed effective pair-sum offset

```text
mu_pair = -37.61102 meV
```

is used. The final density is an output:

```text
average density   8.8075e9 cm^-2
n_e               8.8055e9 cm^-2
n_h               8.8095e9 cm^-2
```

This is far below the experimental `5.5e10 cm^-2`; its corresponding circular
momentum scale is `0.02352 nm^-1`, almost exactly the desired Fig. 2 ring.

The code does not actually enforce `n_e=n_h` in the final fixed-mu stage. It
fixes `eta=e-h`, solves no charge-balance multiplier, and reports a final
imbalance of approximately `-3.97e-8 nm^-2`. The validation file records zero
because supplement residuals omit density/neutrality entries and the
postprocessor defaults missing residuals to zero. This is a validation bug.

### 7. Obtain a genuine fixed point of the calibrated map

The archived node solution has active equation residual

```text
8.38e-14 eV
```

and saved-node metrics

```text
E(0)               6.7741 meV
first node E(0.005) 7.0560 meV
node minimum        1.7815 meV at 0.025 nm^-1
Delta_max           1.5134 meV
```

The cancellation constructing the spectrum is visible directly:

```text
k      raw Kane pair  C_pair   Sigma_pair  -mu_pair   xi      Delta    E
       (meV)          (meV)    (meV)       (meV)      (meV)   (meV)    (meV)
0.000  -27.468         0.000   -16.896     +37.611    -6.753   0.538    6.774
0.005  -27.438        -0.135   -17.068     +37.611    -7.030   0.607    7.056
0.020  -26.984        -0.540   -12.949     +37.611    -2.861   1.365    3.170
0.025  -26.712        +0.442   -10.401     +37.611    +0.940   1.513    1.781
0.035  -25.991        +6.069    -6.859     +37.611   +10.830   1.145   10.891
```

The intralayer exchange scale, fixed offset, and pair correction jointly move
`xi` through zero near the requested ring.

### 8. Convert the coarse node solution into the displayed curve

The figure does not plot only saved SCF nodes. `postprocess.py` constructs
CubicSpline interpolants for `xi(k)` and `Delta(k)`, then plots

\[
E(k)=\sqrt{\xi_{C2}(k)^2+\Delta_{C2}(k)^2}
\]

on at least 1601 dense points. The red gap curve uses PCHIP.

This moves the displayed minimum from the saved-node value

```text
1.7815 meV at k=0.025
```

to

```text
1.5050 meV at k=0.023842
```

where the interpolated `xi` nearly vanishes.

Against the separately archived PDF-vector data, the displayed curves score

```text
C2 E(k) RMS error       0.04425 meV
C2 E(k) maximum error   0.11785 meV
PCHIP Delta RMS error   0.04523 meV
PCHIP Delta max error   0.09077 meV
```

This quantitatively confirms that the displayed figure is a very accurate
curve calibration.

### 9. Generate the JDOS from the same interpolants

The right panel is not an independently computed optical response. It uses the
radial identity

\[
J(\omega)\propto \sum_i\frac{k_i}{|dE/dk|_{k_i}}
\]

on the C2 `xi/Delta` interpolants. The denominator is regularized by

\[
\sqrt{E'^2+2|E''|\Gamma+\text{slope-floor}^2},
\qquad \Gamma=0.02\ \mathrm{meV}.
\]

The characteristic horizontal features are the interpolated stationary
energies

```text
ring minimum    1.5050 meV at 0.023842 nm^-1
k=0 endpoint    6.7741 meV
local maximum   7.0575 meV at 0.005226 nm^-1
```

The local maximum is tied to the first nonzero radial node. It is not an
independent same-Hamiltonian current-matrix-element response calculation.

## Evolution of the apparently successful figures

The bundle preserves a useful progression:

| Figure family | Closure | Minimum | Interpretation |
|---|---|---:|---|
| N=7, scale 1.0 | fixed `5.5e10 cm^-2` | `3.58 meV @ 0.0605` | fails energy and momentum |
| N=7, scale 0.6 | fixed `5.5e10 cm^-2` | `1.54 meV @ 0.0595` | fixes vertical scale only |
| current uncalibrated N=25 | fixed `5.5e10 cm^-2` | `3.63 meV @ 0.0598` | still fails |
| historical calibrated N=25 | low-density branch + fixed mu + fitted terms | plotted `1.505 meV @ 0.02384` | visually successful calibration |

The progression shows exactly what was added to obtain the final match:
first an interaction rescaling, then a low-density/fixed-mu closure and
momentum-dependent empirical dispersion/vertex terms.

## Continuum audit and unresolved same-mu question

The bundle's existing 281-point cell audit gives an essentially zero gap,

```text
Delta_max approximately 2.4e-6 meV
```

and removes the first-node local maximum. However, it is not a controlled
same-mu comparison: it uses density-selected
`mu_pair=-38.85464 meV`, whereas the historical node result fixes
`-37.61102 meV`. The 1.244 meV change is comparable to the claimed gap.

A focused same-mu 281-cell attempt using the restored source did not converge:

```text
16-step path: residual 1.26e-3 eV
40-step path: failed at step 35/40, residual 5.77e-4 eV
```

Nonconverged endpoints are rejected. This establishes that the historical
branch is not yet continuously connected to a converged same-mu cell solution;
it does not by itself prove that no disconnected cell branch exists.

## Decision

The actual source supports the following bounded statement:

> The senior code contains a real self-consistent solution of a heavily
> calibrated effective E1/H1 map, and its interpolation/JDOS pipeline produces
> a highly accurate visual reconstruction of Du et al. Fig. 2.

It does **not** support:

- a parameter-free prediction;
- the experimental-density or canonical-neutral Du2017 closure;
- a converged continuum radial quadrature;
- a source-derived Kane pairing vertex;
- a UV-certified Kane-Poisson/HF/BCS chain;
- production HF/BCS or JDOS release.

The next decisive calculation is a same-raw-band, same-pair-sum-offset,
node/cell `71 -> 141 -> 281` ladder with nonconverged cases excluded, followed
by cumulative removal of the empirical dispersion, vertex, and channel scales
on the 281-cell baseline.

## Forensic lineage figure

A four-panel postsolve-only audit figure is archived at

```text
results/inas_gasb_matrix_ei_experiment/runs/
phase0_actual_senior_calibrated_figure_lineage_20260728/
actual_senior_calibrated_figure_lineage.png
```

It overlays the calibrated C2/PCHIP curves and saved SCF nodes against the
PDF-vector curves, decomposes `xi` into raw Kane pair dispersion, empirical
`C_pair`, exchange, and fixed-offset terms, and displays the bundled node/cell
gap contrast with the different-mu caveat stated in the panel title.

## Focused two-absorption-peak ablation

A fixed-parent, fixed-pair-sum-offset, fixed-node-quadrature Test001 audit
confirms that the full radial JDOS has two dominant maxima:

```text
low peak   1.505 meV   prominence 0.9135
high peak  7.055 meV   prominence 0.4818
```

Their lineage is a ring-minimum singularity plus the merged low-k
endpoint/local-maximum structure. The latter remains tied to the coarse
71-node treatment.

Pair-family ablations give:

| Pair-family case | Delta max (meV) | JDOS maxima (meV) | Result |
|---|---:|---|---|
| full | 1.513 | 1.505, 7.055 | two calibrated peaks |
| no `pair_ring` | 1.609 | 1.605, 7.393 | two peaks survive |
| no `pair_vertex` | 3.516 | 3.268, 7.613 | two peaks survive but low peak is wrong |
| curvature + Gaussian, no ring/vertex | 3.385 | 3.178, 8.003 | minimal two-feature topology among converged pair-family tests |
| curvature only | approximately 0 | none below 10 meV | normal branch |
| Gaussian only | — | — | fixed-offset continuation did not converge; rejected |
| no pair empirical family | 6.568 | only 7.605 | low peak disappears |

Thus `pair_ring` is not required to create two peaks; it fine-tunes their
locations by roughly `-0.10 meV` and `-0.34 meV`. `pair_vertex` is not required
for the topology but is important for the low-peak energy and gap amplitude.
The tested minimal converged pair-dispersion family producing two structures is
`pair_curvature + pair_gaussian`.

This is not the complete empirical-parameter inventory. The restored active
CLI also retains `intralayer_scale=4.2`, `interlayer_scale=2.09`,
`diagonal_scale=0.782`, fixed `mu_pair`, and a low-density branch seed. Setting
either interaction channel scale individually to one collapses the paired gap
and removes the two-peak structure in this node model; setting the diagonal
scale to one preserves two peaks but moves them to approximately `0.753` and
`7.568 meV`. Therefore the two peaks cannot currently be attributed only to
the four `pair_*` families.

An exact follow-up set only `intralayer_scale=interlayer_scale=1`, while keeping
`diagonal_scale=0.782`, every historical `pair_*` term, the same parent,
fixed pair-sum offset, branch schedule, and quadrature. It converged to the
empty normal solution:

```text
Delta_max                7.83e-26 meV
n_e = n_h                0
only JDOS maximum        9.715 meV
paper E(k) RMS error     6.243 meV
paper Delta(k) RMS error 1.101 meV
```

Thus the unchanged remaining pair terms cannot reproduce the paper figure once
the two interaction-channel scales are removed.

The visual audit is at:

```text
results/inas_gasb_matrix_ei_experiment/runs/
phase0_actual_senior_pair_knob_jdos_ablation_20260728/
pair_knob_jdos_ablation.png
```
