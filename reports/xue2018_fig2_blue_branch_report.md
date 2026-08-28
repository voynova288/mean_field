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
root with residual RMS `2.9e-15` and gap `0.18836 Ry*`. Releasing that root to
the complete 16-Pauli HF space sends it to the normal root. It is therefore not
yet an unrestricted blue-branch solution.

## Current conclusion

The correct statement is:

> Existing energy-descent ODA finds normal-like and strong-TRS attractive
> roots. The gap-closing higher-energy TR-preserving branch required by the
> paper has not yet been tracked as a complete stationary solution. Low-gap
> chord directions and constrained roots are high-quality initializers for a
> full-TRS stationary-root continuation, not proof that the paper branch has
> been found.

It is equally incorrect to claim either that the blue saddle has been found or
that it cannot exist because ODA leaves it.

## Required solver work

The next implementation must provide separate APIs for:

1. the full residual
   \[
   R[D;p]=D-\left[P_{\beta,\mu}(H_0(p)+\Sigma_H[D]+\Sigma_F[D])-P_{\rm ref}\right];
   \]
2. an exact full-TRS projector using the integer `k <-> -k` permutation;
3. Anderson/Broyden initialization and Newton--Krylov refinement without an
   energy-decrease constraint;
4. finite-temperature homotopy toward the zero-temperature rank-two map;
5. pseudo-arclength continuation through folds, first over Fig. 2 points
   21--30;
6. residual, symmetry, energy, branch-overlap, and stability diagnostics;
7. regulator-safe arbitrary-k gap evaluation in the cell-integrated physical
   lane.

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
