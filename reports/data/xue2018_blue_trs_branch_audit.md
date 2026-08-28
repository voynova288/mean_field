# Xue 2018 Fig. 2 blue TR-preserving branch audit

## Core hypothesis

The blue squares are not produced by the same unrestricted Eq. (5)--(8) HF
fixed-point branch that successfully reconstructs the black order and red
ground-state gap. The remaining possibilities are an unpublished constrained
ansatz/legacy solver, a different historical interaction closure, or figure
lineage not fully described by the text.

## Causal chain

`TRS seed -> unrestricted D=P-P_ref -> Sigma_H[D]+Sigma_F[D] -> full 4x4 H(k)
-> rank-two occupied projector -> global quasiparticle gap`.

The seed is exactly the paper's `X s_y tau_y` convention:
`H_up,down^{cv}=-H_up,down^{vc}`. The saved p24 strong solution obeys global
time reversal to `1.9e-13 Ry*`.

## Discriminating checks

### 1. Stable unrestricted TRS branch

Strong seeds give:

| point | gap / Ry* |
|---:|---:|
| 24 | 1.0162 |
| 25 | 0.9139 |
| 26 | 0.7809 |

The branch collapses at point 27. The paper instead approaches zero near
points 24--25 and then reopens.

### 2. Pauli decomposition

At p24 the full solution contains the expected `0z`, `zx`, `0y`, and `yy`
terms, plus finite odd-k `x0` and `yz` components. The Hamiltonian remains
TR-invariant. Projecting the interaction onto the literal four-term paper
ansatz removes these terms but gives a converged gap `0.9833 Ry*`, so they are
not the cause of the missing closure.

### 3. Stable-root and source probes

Persistent `s_y tau_y` sources and fixed mixing find the stable normal root
and the two sign-related strong roots. An infinitesimal source selects the
aligned sign rather than revealing a small-X stable branch. Omitting Hartree,
or using exactly half the source Hartree, collapses the nematic order and gives
gaps `2.4895` and `2.1624 Ry*`; the paper's explicit Hartree term cannot be
removed to repair the blue curve.

### 4. Constrained unstable fixed points

The literal four-term ansatz has additional unstable roots. Anderson followed
by Newton--Krylov converged an interior p24 root with

- residual RMS `2.9e-15`;
- maximum residual `2.3e-14`;
- `X(Gamma)=-0.76079 Ry*`;
- gap `0.18836 Ry*`.

At p25 another interior root has gap `0.25635 Ry*`. These are much closer to a
topological closure than the stable branch, but they still do not reproduce
the paper's near-zero blue point.

When the p24 constrained root is released into the full 16-Pauli Eq. (5)
space, the full fixed-point solver converges to the normal root:

- residual RMS `8.4e-16`;
- `X(Gamma)~0`;
- gap `2.24159 Ry*`.

Thus the constrained interior solution is not an unrestricted Eq. (5) fixed
point under the validated interaction action.

## Conclusion

The blue curve cannot currently be promoted as a result of the same
unrestricted HF equations used for the strong black/red reconstruction. This
is a core source-lineage/constraint ambiguity, not a plotting, seed-amplitude,
ODA, self-cell, Hartree, or simple-mesh problem.

No empirical scale, interaction multiplier, coordinate shift, or smoothing
was used. Zeng finite-Q ground-state work should not inherit the blue branch as
an authority until author code/data or an explicit constrained-solver
prescription is found.

## Main artifacts

- `trs_seed_probe/point24_trs_amp1.0/trs_pauli_audit.json`
- `trs_ansatz_probe/point24/summary.json`
- `trs_ansatz_probe/fixed_point_chord.json`
- `trs_ansatz_probe/unstable_root_point24_v2/summary.json`
- `trs_ansatz_probe/unstable_root_point25_v3/summary.json`
- `trs_ansatz_probe/full_unstable_root_point24/summary.json`
- `trs_persistent_source_probe/`
- `trs_hartree_scale_probe/`
