# hTTG Eq. (13) antiunitary-symmetry audit

Last updated: 2026-05-30

Goal: explain why the implemented Mao hTTG Hamiltonian has a very small raw code-axis `sigma^{x;yy}` even though Mao Eq. (15) lists both `sigma^{x;yy}` and `sigma^{y;xx}` as C3-allowed independent components.

## Bottom line

For the symmetric ABA hTTG Hamiltonian implemented from Mao Eq. (13), with the same sublattice mass `m sigma_z` on all three layers, there is an exact antiunitary layer-swap/conjugation symmetry

```text
H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger .
```

The unitary part `U` maps

```text
reciprocal index: (n1,n2) -> (-n2,-n1)
layer:            1 <-> 3, 2 -> 2
sublattice:       identity
```

and the Cartesian operation is `R = diag(+1,-1)`, i.e. `k -> (k_x,-k_y)`.

Because the operation is antiunitary, the dc current response tensor obeys an extra minus sign relative to an ordinary mirror:

```text
sigma^{a;bc} = - R_{aa'} R_{bb'} R_{cc'} sigma^{a';b'c'} .
```

Therefore components with an even number of `y` indices vanish in the symmetry axes:

```text
sigma^{x;xx} = sigma^{x;yy} = sigma^{y;xy} = sigma^{y;yx} = 0.
```

With C3, Mao Eq. (15)'s first C3 group is therefore forced to zero, while the second group can remain finite:

```text
forbidden group: sigma^{x;yy} = -sigma^{x;xx} = sigma^{y;yx} = sigma^{y;xy} = 0
allowed group:   sigma^{y;xx} = -sigma^{y;yy} = sigma^{x;xy} = sigma^{x;yx} != 0
```

This is exactly what the converged local spectra show.  In the earlier hTTG theory paper Mao/Guerci/Mora, PRB 107, 125423 (2023), the analogous ABA local Hamiltonian is described as preserving the combined antiunitary symmetry `C2x C2z T`; the operation derived here is the finite-plane-wave representation of that product symmetry in the current code convention.

## Numerical exact check

Representative finite-cutoff exact-unitary check:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_symmetry_and_convention_audits/htg_symmetry_spectrum_audit_shell3/summary.json`

Key values:

```text
H(k_x,-k_y) - U H(k_x,k_y)^* U^dagger: max ~ 8.4e-17 eV
dH/dkx - U(dH/dkx)^*U^dagger:          0
dH/dky + U(dH/dky)^*U^dagger:          0
```

Converged shell-5 central-flat tensor audit:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_132463/central_flat_allc/tensor_symmetry_audit_eta2/tensor_symmetry_summary.json`

Key result:

```text
C3 group1 / group2 peak ratio ~ 2.2e-11
```

No-C3 but mirror-respecting primitive grid:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_noc3_133958/central_flat_allc/summary.json`

Key result at eta 2 meV:

```text
raw sigma^{x;yy} ~ 1.5e-6
raw sigma^{y;xx} ~ 6990
```

A deliberately shifted, non-mirror quadrature activates a finite-grid forbidden-component artifact, but it remains a quadrature artifact:

- `results/shift_current_htg/_archived_tests_and_diagnostics_20260530/htg_spectrum_diagnostics/physics_audit_central_shell5_mesh16_shifted_noc3_133985/central_flat_allc/tensor_symmetry_audit_eta2/tensor_symmetry_summary.json`

Key result:

```text
group1/group2 ~ 0.046
```

This is still much smaller than the rough Mao Fig. 2 ratio and disappears when the quadrature respects the antiunitary map.

## Why the symmetry exists in the implemented model

The code convention uses complex momenta `k = k_x + i k_y` and moire vectors

```text
q0 = (0, -q)
q1 = (sqrt(3) q / 2,  q / 2)
q2 = (-sqrt(3) q / 2, q / 2)
b1 = q1 - q0
b2 = q2 - q0
```

so

```text
conj(b1) = -b2,
conj(b2) = -b1.
```

Thus a plane-wave label

```text
G(n1,n2) = n1 b1 + n2 b2
```

is mapped by `k_y -> -k_y` to

```text
conj(G(n1,n2)) = G(-n2,-n1).
```

### Diagonal Dirac blocks

For valley `K` and unrotated Dirac blocks,

```text
h(p) = hbar v_F [[0, p^*], [p, 0]],
p = k + G + q_layer.
```

The layer offsets are

```text
q_layer(1) = +q0,
q_layer(2) = 0,
q_layer(3) = -q0.
```

Since `conj(q0) = -q0`, complex conjugation plus layer swap sends the top-layer momentum to the bottom-layer momentum and vice versa:

```text
conj(k + G + q0)  = k^* + G^* - q0,
conj(k + G - q0)  = k^* + G^* + q0.
```

Also `h(p^*) = h(p)^*`.  Therefore the diagonal Dirac sector is invariant under the antiunitary layer swap.

The mass term `m sigma_z \otimes I_layer` is real, diagonal, and identical on all three layers, so it is also invariant.  This point is important: a layer-dependent substrate mass would break this symmetry, but Mao Eq. (13) writes the same mass on every layer.

### Interlayer tunneling blocks

For valley `K`, the tunneling matrices have the form

```text
T_j = w1 [[r, exp(-i 2 pi j / 3)], [exp(+i 2 pi j / 3), r]],  j=0,1,2,
```

with real `r`.  They obey

```text
T_0^* = T_0,
T_1^* = T_2,
T_2^* = T_1.
```

The channel momentum shifts obey the same `1 <-> 2` channel swap under conjugation:

```text
conj(q1 - q0) = -(q2 - q0),
conj(q2 - q0) = -(q1 - q0).
```

Top-interface entries with shift sign `+` are therefore mapped to bottom-interface entries with shift sign `-`.

For ABA stacking, the displacement construction gives `d_bot = - d_top` and channel phases satisfying

```text
phase_top(0) = phase_bot(0),
phase_top(1) = phase_bot(2),
phase_top(2) = phase_bot(1).
```

Thus the tunneling sector is also exactly mapped into itself by conjugation plus layer swap.

## Consequence for the shift-current tensor

Let the antiunitary operation be `A = U K` with Cartesian action `R = diag(+1,-1)`.  Velocities transform as

```text
v_x(Rk) = + U v_x(k)^* U^dagger,
v_y(Rk) = - U v_y(k)^* U^dagger.
```

Equivalently, the response tensor transforms like a rank-3 polar tensor under `R`, but with an additional minus sign from the time-reversal/antiunitary part because the dc current is odd under time reversal whereas the optical electric fields are even:

```text
sigma^{a;bc} = - R_{aa'} R_{bb'} R_{cc'} sigma^{a';b'c'}.
```

For diagonal `R`, each component gets the sign `(-1)^{N_y}`, where `N_y` is the number of `y` indices among `(a,b,c)`.  Invariance requires

```text
sigma^{a;bc} = - (-1)^{N_y} sigma^{a;bc}.
```

Therefore:

- even `N_y`: component must vanish;
- odd `N_y`: component is allowed.

Combining this with Mao's C3 Eq. (15) leaves only the second C3 group finite.

## Relation to Mao's statement about broken `C2x` and `C2zT`

Mao 2025 text says the mass term breaks `C2x` and `C2zT` individually.  That statement is not contradicted by this audit.

The symmetry found here is **not** the usual unitary `C2x` and is **not** the usual `C2zT` separately.  It is their antiunitary product-like layer-swap/conjugation operation, consistent with Mao/Guerci/Mora 2023's statement that the ABA local Hamiltonian preserves `C3z` and `C2x C2z T`.  This product leaves sublattice unchanged in the current representation.  Since it does not exchange the A/B sublattices, `sigma_z` does not change sign and the equal mass term survives.

Thus Mao Eq. (13), as written with `m sigma_z` identical on all layers and symmetric `V_{phi1,phi2}` / `V_{-phi1,-phi2}` interfaces, contains an antiunitary product-symmetry constraint beyond the C3-only tensor statement quoted in Mao 2025 Eq. (15).

## Relation to the empirical small axis rotation diagnostic

If the tensor is expressed in axes rotated by an angle `alpha` away from the antiunitary mirror axes, the single allowed C3 coefficient mixes into both paper-labelled components.  For an allowed coefficient `B` in the mirror axes, a pure rotation gives, up to sign convention,

```text
sigma_new^{x;yy} ~ B sin(3 alpha),
sigma_new^{y;xx} ~ B cos(3 alpha),
|sigma_new^{x;yy} / sigma_new^{y;xx}| = |tan(3 alpha)|.
```

The rough Mao Fig. 2 ratio `|sigma^{x;yy}/sigma^{y;xx}| ~ 0.25` corresponds to `alpha ~ 4.8 deg`.  This explains why a `~5 deg` plotting rotation can make the spectra look paper-like.

However, unless that `~4.8 deg` is derived from Mao's published Cartesian-axis convention, it remains an empirical diagnostic and not a valid reproduction step.

## Reproduction implication

Within the literal Eq. (13) model currently implemented, a large raw code-axis `sigma^{x;yy}` is symmetry-forbidden.  Mao Fig. 1(b)/Fig. 2 can be reproduced honestly only if one of the following is established:

1. Mao's plotted `x,y` axes are rotated/reflected relative to the antiunitary symmetry axes by a derivable convention;
2. Mao's actual numerical Hamiltonian contained a symmetry-breaking term not written in Eq. (13), e.g. layer-dependent substrate mass/potential or a different basis embedding;
3. Mao's spectra contain finite-grid or component-label artifacts.

Until then, the paper-labelled nonzero `sigma^{x;yy}` should not be claimed as reproduced.
