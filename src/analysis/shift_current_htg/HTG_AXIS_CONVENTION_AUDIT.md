# hTTG Cartesian-axis convention audit

Last updated: 2026-05-30

Goal: decide whether the empirical `reflect-y + rotation ~4.8 deg` transformation used in diagnostics can be derived from Mao's published coordinate conventions.

## Sources checked

1. Mao et al. 2025 shift-current paper (`reference/2411.07844v2.pdf` / `tmp/pdfs/2411.07844v2.txt`):
   - Eq. (13): hTTG local continuum Hamiltonian.
   - Eq. (14): tunneling matrices `T_j`.
   - Text below Eq. (13): ABA phases `phi1=-phi2=2pi/3`, equal mass `m sigma_z \otimes I_layer`.
   - Text near Eq. (15): C3 action
     ```text
     (x,y) -> 1/2 (-x - sqrt(3)y, sqrt(3)x - y)
     ```
     i.e. a +120 degree rotation in the printed Cartesian basis.
   - Appendix A: monolayer gapped-graphene nearest-neighbor vectors
     ```text
     d_j = d[-sin(2(j-1)pi/3), cos(2(j-1)pi/3)]
     ```
     so `d_1` points along +y.
2. Mao, Guerci, Mora, PRB 107, 125423 (2023), downloaded as
   `reference/Mao_Guerci_Mora_2023_PRB107_125423.pdf`:
   - same tunneling-matrix convention `T_{j+1}=r sigma_0 + cos(2pi j/3) sigma_x + sin(2pi j/3) sigma_y`.
   - symmetry appendix states that the ABA/AAA local Hamiltonians preserve `C3z` and the combined antiunitary `C2x C2z T`.

## Code coordinate convention

The current hTTG lattice uses complex momenta `k=k_x+i k_y` and

```text
q0 = q (0, -1)
q1 = q (sqrt(3)/2,  1/2)
q2 = q (-sqrt(3)/2, 1/2)
```

with

```text
b1 = q1 - q0 = q (sqrt(3)/2, 3/2)
b2 = q2 - q0 = q (-sqrt(3)/2, 3/2).
```

The paper's printed C3 matrix

```text
R_C3 = [[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]]
```

acts on the code `q` vectors as

```text
R_C3 q0 = q1,
R_C3 q1 = q2,
R_C3 q2 = q0.
```

Therefore the code axes are already compatible with the paper's printed C3 action, provided the usual BM convention `q1` downward is used.  If one instead aligns with Appendix-A `d_1` upward, the hTTG `q` triplet is changed by a 180 degree sign / channel relabeling, not by a small `~5 deg` rotation.

## Natural convention transformations checked conceptually

The published definitions naturally allow only discrete or obvious transformations:

- channel cycling of `q_j` / `T_j`: rotations by multiples of 120 degrees;
- choosing the opposite helical domain: sign/domain flip of the stacking phases;
- using `q_j` versus `-q_j`: 180 degree rotation plus relabeling;
- switching between equivalent moire reciprocal basis choices: rotations/reflections by multiples of 30 or 60 degrees;
- including physical layer Dirac rotations: angles of order `theta` or `theta/2`, but Mao Eq. (13) explicitly uses unrotated `hbar v_F k dot sigma` blocks, and the Mao adapter now sets `zeta_rad=0`.

None of these gives a natural `alpha ~= 4.8 deg` basis rotation.

For the actual benchmark angle,

```text
theta = 1.95 deg,
theta/2 = 0.975 deg,
3 theta = 5.85 deg,
(5/2) theta = 4.875 deg,
```

but there is no published coordinate operation in Eq. (13)-(15) involving `3 theta` or `5 theta/2`.  Using such a number would be a fit, not a derivation.

## Antiunitary symmetry fixes the symmetry axes

`HTG_SYMMETRY_AUDIT.md` derives the exact finite-cutoff relation

```text
H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger,
```

with reciprocal-index map `(n1,n2)->(-n2,-n1)`, layer swap `1<->3`, and sublattice identity.  This is the code representation of the combined antiunitary product-like symmetry identified in Mao/Guerci/Mora 2023 as `C2x C2z T` for the ABA local Hamiltonian.

In the symmetry axes, the dc shift-current tensor obeys

```text
sigma^{a;bc} = - R_{aa'} R_{bb'} R_{cc'} sigma^{a';b'c'},
R = diag(+1,-1).
```

Thus the C3 Eq. (15) group containing `sigma^{x;yy}` is forbidden, and the group containing `sigma^{y;xx}` is allowed.

This symmetry is exact for the literal Mao Eq. (13) implementation with equal mass on all layers.  It is not a finite-mesh effect and not a plotting convention.

## Why the `~4.8 deg` rotation appeared

The converged tensor has one allowed C3 coefficient `B` in the antiunitary symmetry axes.  If one rotates the tensor basis by an arbitrary angle `alpha`, then, up to sign convention,

```text
sigma_new^{x;yy} ~ B sin(3 alpha),
sigma_new^{y;xx} ~ B cos(3 alpha),
|sigma_new^{x;yy}/sigma_new^{y;xx}| = |tan(3 alpha)|.
```

A rough visual/digitized Mao Fig. 2 ratio `|sigma^{x;yy}/sigma^{y;xx}| ~ 0.25` gives

```text
alpha ~= arctan(0.25)/3 ~= 4.8 deg.
```

This explains the diagnostic plots, but it does **not** derive the angle from the paper.  It is currently only an inverse fit to the apparent component ratio.

## Current conclusion

At present, I do **not** see a legitimate coordinate-convention derivation of the `~4.8 deg` reflected-basis rotation from Mao's published `q_j`, `T_j`, C3, stacking-phase, or Appendix-A axes.

The stronger conclusion is:

1. Mao's printed C3 convention is compatible with the code axes up to standard discrete relabelings.
2. Mao/Guerci/Mora 2023 indicates that ABA local hTTG preserves the combined antiunitary `C2x C2z T`; the local implementation realizes this exactly.
3. In those symmetry axes, a nonzero raw `sigma^{x;yy}` is forbidden.
4. Therefore Mao Fig. 1(b)/Fig. 2 cannot yet be honestly reproduced in paper-labelled axes from the literal Eq. (13) model.

To make the paper-labelled blue `sigma^{x;yy}` legitimate, one must still establish at least one of:

- Mao plotted axes rotated by a documented, nonstandard angle relative to the Eq. (13) axes;
- Mao's actual numerical Hamiltonian included a symmetry-breaking ingredient not written in Eq. (13), such as layer-dependent mass/substrate potential or asymmetric encapsulation;
- Mao's component labels or finite-grid procedure differ from the formal symmetry-respecting calculation.

Until one of these is shown, `rot~4.8 deg` must remain a diagnostic, not a reproduction step.
