# Du2017 UV follow-up: principal symbol and finite-BZ regulator family

**Date:** 2026-07-24
**Status:** completed and failed closed; no Poisson/HF/BCS release

## Core hypothesis

The W4 frozen-reference failure could have had either of two causes:

1. a simple, uniformly nondegenerate high-momentum expansion exists but the
   earlier calculations had not reached it; or
2. the full Kane principal symbol has small/vanishing E/V separations, making
   the expansion nonuniform in z-plane-wave resolution and leaving the finite
   part dependent on a UV completion.

The follow-up evidence supports the second explanation.

## Causal chain

```text
source-method plane-wave Kane Hamiltonian
-> exact in-plane polynomial coefficients
-> Gamma6/valence principal spectra
-> Sylvester denominators and resonant clusters
-> finite-BZ square-lattice regulator family
-> full-parent finite-probe density change
-> BZ-grid/edge/stencil gates
```

## 1. Exact source-method principal-symbol extraction

Slurm job `193336` extracted

\[
H=H_0+k_xB_x+k_yB_y+k_x^2A_{xx}+k_xk_yA_{xy}+k_y^2A_{yy}
\]

from the pinned plane-wave Kane implementation for

```text
N = 7, 15, 31, 47, 63, 95
angles = 0, 15, ..., 90 degrees.
```

Engineering/core checks:

```text
maximum polynomial replay error     1.78e-15 eV
maximum principal Hermiticity error 4.44e-16 eV nm^2
maximum Gamma6-valence A off-block  1.54e-33 eV nm^2
```

The finite-matrix minimum cross-sector separations were

```text
N=7   1.7089e-3 eV nm^2
N=15  6.6799e-4 eV nm^2
N=31  1.6223e-3 eV nm^2
N=47  5.4407e-5 eV nm^2
N=63  1.8486e-5 eV nm^2
N=95  1.0192e-4 eV nm^2
```

The corresponding maximum sampled Sylvester solution norms reached
`974.6 nm^-1`.  E/V convex-hull intervals were interleaved for every tested
N/direction.  Both the finite-sampled setwise-separation gate and the global
energy-ordering gate failed.

This finite sample does not prove an all-angle or N-to-infinity theorem.  It
shows that the simple expansion is severely ill-conditioned on the actual
source matrices.

## 2. Homogeneous full-3D local symbol

A reduced faithful test on `test001` formed, for each material,

\[
A_m(\hat n)=\frac{H_m(\hat n)+H_m(-\hat n)}{2}-H_m(0),
\qquad
B_m(\hat n)=\frac{H_m(\hat n)-H_m(-\hat n)}{2}.
\]

For AlSb, two continuous sign brackets were refined to exact Gamma6-valence
principal crossings.  One witness was

```text
direction       (0.661392, 0.589562, -0.463656)
principal value -0.0600000000000 eV nm^2
root residual   -1.73e-15 eV nm^2
cluster sizes   2 x 2
```

The resonant linear Kane-block singular values were

```text
0.1429620442, 0.1429620442 eV nm.
```

These singular values are invariant under independent U(2) rotations within
the complete resonant Gamma6 and valence subspaces.  Therefore the ordinary
nondegenerate Sylvester equation has no solution at that full-3D direction.

Scope qualification: the witnessed roots have nonzero k_z/k and obstruct a
joint full-3D or N-uniform argument.  They do not alone disprove a fixed-N,
purely in-plane asymptotic expansion.

The artifact field that would have certified a uniform local gap from absence
of sampled roots is fail-open in principle.  It is `false` here because a
positive AlSb witness was actually found; only the positive crossing witness,
not absence statements for InAs/GaSb, is used scientifically.

## 3. Equatorial local symbol

A separate 721-angle equatorial audit found:

```text
same-material equatorial crossings witnessed: none
cross-material spectral-union crossings:      yes
```

Minimum sampled same-material equatorial separations were

```text
InAs  0.14608 eV nm^2
GaSb  0.09381 eV nm^2
AlSb  0.04019 eV nm^2
```

Thus a pointwise local, fixed-N in-plane expansion is not disproved.  However,
the heterostructure's material-union spectra overlap—for example InAs Gamma6
versus GaSb valence—so there is no single global energy contour separating the
orbital sectors.  Interface and N-uniform control remain unresolved.

## 4. Finite-BZ regulator-family diagnostic

Because the continuum argument was nonuniform, a separately labelled hybrid
finite-BZ model was tested.  It is not zincblende tight binding and not a
Du2017 reconstruction.

Both stencils used the same square BZ, spacing `a=0.60959 nm`, full
`8(2N+1)` parent at `N=7`, average full-valence filling `6(2N+1)`, one physical
common mu preserving the frozen-reference total occupation, and a finite
`+2 meV` cosine potential-energy probe.

Nearest-neighbour stencil:

\[
k_i\to\frac{\sin aq_i}{a},\qquad
k_i^2\to\frac{2(1-\cos aq_i)}{a^2}.
\]

Fourth-order improved stencil:

\[
k_i\to\frac{8\sin aq_i-\sin2aq_i}{6a},
\]

\[
k_i^2\to
\frac{5/2-(8/3)\cos aq_i+(1/6)\cos2aq_i}{a^2}.
\]

Physical and reference density blocks were subtracted at each q before BZ
integration.  Both trace and full z-profile norms were used for the BZ-edge
gate.

Algebra checks passed:

```text
low-q continuum replay <= 4.29e-14 eV
BZ boundary periodicity <= 4.38e-16 eV
Hermiticity              <= 1.11e-16 eV
induced neutrality        ~1e-12 nm^-2 or better
```

### First ladder: job 193413

```text
nq = 16, 24, 32, 40
outcome = UNRESOLVED_BZ_GRID_OR_EDGE
```

At `nq=40`, the outer 10% BZ supplied:

```text
nearest electron profile L2 fraction  56.6%
improved electron profile L2 fraction 96.3%
```

### Predeclared final refinement: job 193419

```text
nq = 40, 48, 56, 64
outcome = UNRESOLVED_BZ_GRID_OR_EDGE
```

Successive profile changes remained:

```text
nearest: 74.5%, 69.1%, 80.6%
improved: 97.9%, 92.5%, 82.8%
```

At `nq=64`, the two stencils still differed by

```text
density 69.9%
profile 80.5%.
```

The outer 10% BZ electron-profile fractions were approximately

```text
nearest 78.5%
improved  9.1%
```

Both fail the predeclared 5% edge gate, and the difference itself is strongly
stencil dependent.  No further grid escalation is authorized.

## Discriminating conclusion

The finite BZ removes the literal infinite integral, but this is not enough.
For the present hybrid completion, the induced finite part is dominated by BZ
edge structure and does not converge on the tested grids.  The result cannot
be promoted to either `REGULATOR_SENSITIVE` universality evidence or limited
robustness; it remains an unresolved/edge-dominated regulator diagnostic.

Together with the principal-symbol evidence, this rules out the following as
controlled solutions for the present project:

- continuing the continuum cutoff ladder;
- relying on a simple frozen W4 reference to reach k^-4 at accessible k;
- declaring success merely because a generic square-lattice BZ is finite;
- selecting one square-lattice stencil from agreement with experiment.

## Code changes

Only ignored diagnostic scripts and immutable execution snapshots were added.
No production Poisson, HF, BCS, JDOS, plotting, or material-source code was
modified.

## Validation

- Principal-symbol production audit: Slurm `193336`, full 64-core node.
- Local 3D/equatorial audits: `test001`, reduced 8x8 homogeneous real path.
- Finite-BZ diagnostics: Slurm `193413` and `193419`, full 64-core nodes.
- Snapshot admission, exact source hashes, BLAS thread isolation, pointwise
  subtraction, Fourier-orientation oracle, JSON/NPZ reload, and marker-last
  publication were checked.

Engineering paths are reproducible, but the physical UV finite part remains
unverified and the tested regulator family is numerically/physically
edge-dominated.

## Decision and next defensible path

Stop this continuum/square-lattice branch before Poisson.

The remaining defensible routes are:

1. an atomistic or Wannier finite-BZ Hamiltonian whose real unit cell, basis
   multiplicity, remote bands, and electrostatic background are fixed by a
   microscopic source; or
2. author-supplied Du2017 k range, retained subbands, and reference/background.

A new microscopic model must still use one common physical chemical potential,
pointwise occupied-density differences, explicit device boundaries, and
independent z/BZ convergence.

## Evidence hashes

```text
principal-symbol audit
3c9eacb7d45032ffb1458282c085a691f5a5fd98f883c10c95ac5de8826ad04a

local full-3D symbol audit
3df2d3379daec1608ba146209492804eb3ffca32c43ab338137c1fd8285f4695

equatorial symbol audit
9d34a3322ecd74dc1dbe486c7edd2bd8880f2f0829b4a1004f63b6b58b366ec7

finite-BZ nq16-40 audit
097bb06d544fc1a54cebf5b0d6532560e73d10548702b055e7d6d0fd80a9f30f

finite-BZ nq40-64 audit
6cfb49dfa3ac5660da7cbf79fe8e5ed82a1cac66f97feef624c57f7acde53e6a
```
