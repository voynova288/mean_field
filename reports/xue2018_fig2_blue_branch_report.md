# Xue--MacDonald 2018 Fig. 2 blue-branch reproduction report

> **Dataset ID:** `xue2018-fig2-branch-lineage-audit-v1`
>
> **Branch:** `debug/excitonic-reproduction`
>
> **Status date:** 2026-08-28

## Scope and authority

This report addresses only Xue and MacDonald, *Physical Review Letters* **120**,
186802 (2018), Fig. 2, with emphasis on the bottom-panel blue
**time-reversal-preserving nematic branch**. Paper digitization is comparison
only and is never used as a solver input.

The historical reconstruction lane is

```text
61x61 inclusive nodes on [-3,3]^2
uniform spacing-squared weights
single q=0 Fock point omitted
```

This lane is an inferred historical discretization, not regulator-independent
continuum authority. The physical lane must instead use midpoint cells,
cell-integrated singular terms, and explicit momentum-window checks.

## First correction: the old bottom panel was provenance-inconsistent

The previous report quoted the strong-TRS ODA attractor at points 24--26,

\[
1.01620717,\quad 0.91390618,\quad 0.78088646\;Ry^*,
\]

while the tracked JSON and comparison PNG plotted an older weak-seed branch
that had collapsed to the normal-like attractor,

\[
1.81490508,\quad 1.85099293,\quad 1.88639482\;Ry^*.
\]

Those values must not be spliced into one curve. The old full-path data are now
explicitly labeled `stale_historical_branch_artifact` with branch ID
`xue2018-trs-weak-seed-normal-attractor-fullpath-v1`; the strong branch is
stored only as three separately identified anchors with branch ID
`xue2018-trs-strong-attractor-p24-p26-v1`. The active figure is
generated from one versioned branch-data file and bound by a hash manifest:

- [`data/xue2018_fig2_branch_data.json`](data/xue2018_fig2_branch_data.json)
- [`data/xue2018_fig2_artifact_manifest.json`](data/xue2018_fig2_artifact_manifest.json)

![Xue Fig. 2 branch-lineage audit](figures/xue2018_fig2_branch_lineage_audit.png)

The bottom panel intentionally shows two different calculated lineages rather
than presenting either one as the paper branch.

## Paper contract

The ordinary-electron basis is

\[
(c\uparrow,v\uparrow,c\downarrow,v\downarrow),
\]

and the interaction acts on

\[
D=P-P_{\rm ref},\qquad
P_{\rm ref}=\operatorname{diag}(0,1,0,1).
\]

The paper identifies a higher-energy TR-preserving nematic solution with the
schematic Hamiltonian

\[
H_{\rm TRS}(\mathbf k)=
\xi_{\mathbf k}s_0\tau_z+A k_xs_z\tau_x-A k_ys_0\tau_y+X s_y\tau_y,
\]

for which the gap can close when

\[
\xi_{\mathbf k}=0,\qquad A k_y=X.
\]

Because this path connects normal-insulator and QSHI limits while preserving
time reversal, a continuous branch must pass through a quasiparticle closure.

## What is already excluded

The mismatch is not explained by:

- the sign of the `s_y tau_y` seed;
- weak seed amplitude;
- a simple Hartree factor;
- the single `q=0` prescription alone;
- extra TR-compatible momentum-dependent Pauli channels;
- plotting, broadening, smoothing, or coordinate adjustment.

The saved p24 strong solution has time-reversal error approximately
`1.9e-13 Ry*`. Projecting the interaction to the paper's literal four-term
ansatz still gives a stable p24 gap near `0.9833 Ry*`.

## Why ODA is not decisive for the blue branch

The current ODA/SCF map is an energy-descent solver. It reliably finds
attractive local minima but cannot be used to rule out a higher-energy
stationary branch, separatrix, fold, or saddle.

At p24, ODA finds two attractive roots:

| root | gap / `Ry*` | reference-relative energy / `Ry*/a_B*^2` |
|---|---:|---:|
| normal-like | `1.81490554` | `-0.41183921` |
| strong TR-preserving | `1.01620637` | `-0.38010018` |

The digitized paper blue gap is approximately `0.05279 Ry*`.

A normal-to-strong density chord contains a low-gap state near
`t=0.6596748972`, with saved-grid gap `0.0681161 Ry*` and a diagnostic
off-grid estimate `0.0619796 Ry*`. However its full residual RMS is about
`0.0700`, so it is an initializer, not an HF fixed point.

Within the literal four-term constrained ansatz, Newton--Krylov finds a p24
root with residual RMS `2.9e-15` and gap `0.18836 Ry*`. A direct full-space
restart from that constrained root falls into the normal basin; this reflects
the instability and narrow basin rather than the absence of a continuation.

A self-energy constraint-release homotopy now resolves this ambiguity:

\[
\Sigma=\left[(1-\lambda)\mathcal P_{\rm TRS}
+\lambda\mathcal P_{4}\right]\Sigma_{\rm int}[D(H_0+\Sigma)],
\qquad \lambda:1\to0.
\]

Here `lambda=1` exactly reproduces the four-term constrained equation and
`lambda=0` is the complete-TRS self-energy equation. At both p24 and p25 all
14 continuation weights converged without a fold, with minimum adjacent-state
overlap above `0.99990`. The endpoints are the already identified low-gap full
stationary roots:

| point | constrained gap | released full gap | endpoint overlap with low-gap branch |
|---:|---:|---:|---:|
| 24 | `0.18836` | `0.14986` | `0.9999999999999998` |
| 25 | `0.25635` | `0.20985` | `0.9999999999999991` |

A second, higher-gap constrained p25 root from forward constrained
continuation was also released. Its gap evolves from `0.86651 Ry*` to
`0.913905 Ry*`, and its endpoint overlaps the refined strong-TRS root by
`0.9999999999999998`. Therefore the two known constrained p25 root classes
map respectively to the low-gap stationary branch and the strong-TRS branch.
Neither hides a separate paper-like branch.

## Full stationary branch now found

Using the low-gap chord only as an initializer, followed by

```text
complete-TRS residual projection
T/Ry* = 1e-2 -> 3e-3 -> 1e-3 -> 3e-4 -> 1e-4 -> 0
Newton--Krylov root refinement
pseudo-arclength continuation
full unprojected residual checks
```

we found a genuine unrestricted stationary branch with branch ID
`xue2018-trs-full-stationary-p21-p26-v1`.

| Fig. 2 point | paper blue / `Ry*` | stationary grid gap / `Ry*` | local-cell off-grid diagnostic / `Ry*` |
|---:|---:|---:|---:|
| 21 | `0.36149` | `0.05747` | `0.04468` |
| 22 | `0.28576` | `0.08341` | `0.07084` |
| 23 | `0.16825` | `0.10957` | `0.10679` |
| 24 | `0.05279` | `0.14986` | `0.14980` |
| 25 | `0.00842` | `0.20985` | `0.20899` |
| 26 | `0.13157` | `0.30075` | `0.29637` |

Every listed zero-temperature point has full residual RMS below `3e-11`, full
residual maximum below `4e-10`, and TR error below `4e-12`. The p24 root is
unstable under simple fixed-point iteration in both sectors:

- complete-TRS Jacobian spectral radius: `1.95042`;
- TR-breaking-complement spectral radius: `2.30457`.

These are fixed-point-map eigenvalues, not an energy-Hessian certificate.
The off-grid column omits the source node associated with the bounded local
mesh cell throughout minimization. It is a continuous local extension of the
historical omitted-`q=0` rule, but it is non-unique and not continuum
regulator authority. It confirms that mesh-node minimization is not the source
of the p24--p26 disagreement.

The finite-temperature horizontal pseudo-arclength branch folds at
`A = 0.24844 Ry* a_B*`, before reaching Fig. 2 point 27.

## Current conclusion

The stationary-root hypothesis was partly correct: ODA missed a real
higher-energy full-TRS stationary branch. However, the branch found from the
low-gap initializer is **not the paper blue curve**. Its p21--p26 gap trend is
opposite to the paper trend, and it folds before point 27 instead of connecting
the displayed NI-like and QSHI-like sides.

A deterministic p25 inventory from eight independent complete-TRS starts
found only three already known root classes: one normal-like root, five
strong-TRS convergences, and two low-gap stationary-branch convergences. A
full density-chord scan likewise has only the endpoint sign changes and one
interior initializer interval. This is not an exhaustive nonexistence proof,
but no additional paper-like near-zero p25 root was found. The exact four-term constrained roots have also been excluded as a separate
candidate: the low constrained root class releases into the low-gap stationary
branch, while the higher constrained root class releases into the strong-TRS
branch.

Therefore the remaining problem is no longer merely “implement a root
solver.” We must determine whether a disconnected branch exists outside the
current multistart/chord basins, or whether the paper used an unpublished
constrained/legacy closure.

## Remaining solver work

Implemented and validated:

1. full residual
   \[
   R[D;p]=D-\left[P_{\beta,\mu}(H_0(p)+\Sigma_H[D]+\Sigma_F[D])-P_{\rm ref}\right];
   \]
2. exact full-TRS projector with integer `k <-> -k` pairing;
3. Anderson/Newton--Krylov stationary solving without physical-energy descent;
4. finite-temperature homotopy to the zero-temperature rank-two map;
5. pseudo-arclength continuation through a verified fold;
6. full residual, TR, number, overlap, energy, and local fixed-map stability diagnostics;
7. exact four-term-to-full self-energy constraint release at p24 and p25.

Still required:

1. multi-start branch enumeration on both sides of the fold and from NI/QSHI
   endpoints, without selecting by paper gap;
2. regulator-safe arbitrary-k gap evaluation in the cell-integrated physical
   lane;
3. fixed-window and momentum-window continuation of the stationary branch;
4. an energy-Hessian or equivalent thermodynamic stability classification;
5. author code/data or an explicit historical constrained-solver prescription.

A historical omitted-`q=0` node rule has no unique continuous arbitrary-k
extension. Its mesh-node gaps must remain labeled historical; continuous gap
minimization must use a separately defined cell-integrated regulator.

## Promotion gate

The blue branch may be reported as solved only after at least seven adjacent
points belong to one pseudo-arclength branch and each satisfies:

```text
residual RMS < 1e-9
residual max < 1e-8
TR error < 1e-10
explicit half-filling/number closure
saved full density and immutable solver manifest
controlled finite-T -> zero-T continuation
branch continuity without silent solver or regulator changes
```

At least one point must have a converged direct gap near zero in the
cell-integrated lane, and the branch must connect NI-like and QSHI-like sides.
