# Vituri 2024 q003 same-functional detached theorem review

Date: 2026-09-16

## Scope and verdict

This review separates a conditional source-level identity from runtime q003 evidence.

For one selected-spin fixed-rank tangent inventory, the source formulas support

\[
H_{\mathrm{prod}}
=2\,\mathcal R\!\left[D_{\mathrm{prod}}+J^\dagger L_{\mathrm{prod}}J\right]
=2\,\mathcal R\!\left[D_F+J^\dagger\Sigma J\right]
=H_E,
\]

provided every assumption below is satisfied. Here \(\mathcal R\) is the complex-to-real packing used by the paired-sector Hessian. This is a conditional formula theorem, not a q003 production certificate. A registered-vector canary cannot establish a whole-operator identity.

No scalar-Hessian, eigensolver, local-stability, production, or paper authority is promoted by this review.

## Required assumptions

1. The density convention is \(P_{ij}=\langle c_j^\dagger c_i\rangle\), with the raw full trace.
2. The source density is a fixed-rank Hermitian projector and variations are exact unitary paths \(P(t)=e^{tK}P_0e^{-tK}\).
3. The selected-spin inventory contains every occupied-to-virtual transition in its declared finite-square no-wrap domain.
4. The self-conjugate \((d_x,d_y,\chi)=(0,0,0)\) sector has zero transition dimension.
5. Both signed lanes \(s\) and \(-s\), including one-empty orbits, are retained without averaging.
6. The source Fock operator is occupied/virtual block diagonal; production's pointwise gap form additionally requires the registered transition basis to diagonalize the relevant occupied and virtual blocks.
7. Production and scalar paths use the same spinors, flavor order, integer labels, no-wrap support, FFT plan, kernel, physical area, and signed-density convention.
8. The interaction map is reciprocal under the raw trace pairing.
9. Exact signed-lane covariance uses a real-even kernel. A finite residual establishes only a bounded numerical statement.
10. The interaction is divided once by physical area and is not divided by \(N_k\).

## Tangent embedding and adjoint

Let \(E_s\) inject the ordered transition vector for signed lane \(s\) into its particle-hole matrix block. For a canonical pair \(r,-r\),

\[
W_r=E_r x+(E_{-r}y)^\dagger,
\qquad
W_{-r}=W_r^\dagger.
\]

With the half-pair real Hilbert inner product,

\[
J^\dagger(C_r,C_{-r})
=\left(E_r^\dagger C_r,\ E_{-r}^\dagger C_{-r}\right).
\]

When \(C_{-r}=C_r^\dagger\), the second component is
\(E_{-r}^\dagger(C_r^\dagger)\). The previous shorthand
`conj(E_-r^dagger*C_r)` was ambiguous and has been removed from the candidate theorem string.

## One-body action

For occupied-to-virtual coordinate matrix \(X\), exact-unitary differentiation gives

\[
D_F(X)=F_vX-XF_o.
\]

If the occupied and virtual blocks are diagonal in the registered transition basis,

\[
(D_FX)_{ph}=(\epsilon_p-\epsilon_h)X_{ph}.
\]

A common identity shift cancels:

\[
(F_v+\lambda I)X-X(F_o+\lambda I)=F_vX-XF_o.
\]

Thus the physically relevant theorem is equality modulo one common identity. The current bridge uses the stronger absolute-diagonal comparison and finite tolerances. Its runtime result must therefore be reported as a bounded action equality, not an exact symbolic equality or a proof of an absolute Fock zero.

## Interaction action

The production and exact-integer scalar sources implement the same direct and exchange formulas on each admissible signed block:

- the direct term is present only for zero valley charge;
- the exchange term is the same zero-padded no-wrap convolution with the same flavor block;
- both use one division by the same physical area;
- neither introduces an additional \(N_k\) division.

The FFT orientation equivalence uses kernel evenness. The recorded asymmetric-kernel derivation remains a derivation only; the physical real-even q003 kernel cannot by itself test reversed orientation.

## Real factor two and quadratic form

For complex gradient action \(g(z)\) and real coordinates \(v=(\Re z,\Im z)\),

\[
H_Rv=2(\Re g,\Im g),
\qquad
v^TH_Rv=2\operatorname{Re}(z^\dagger g).
\]

The production canary must call the actual `PairedSectorOrbitalHessian.matvec` path to test this implementation choice. A manually packed interaction comparison is insufficient.

## Current implementation evidence boundary

`vituri2024_hf_spiral_q003_production_canary.py` is designed to:

- bind an identity-registered candidate-proof source record;
- bind the parent same-functional structural receipt and its matching source record;
- bind the independent source-Fock bytes by hash;
- independently reconstruct exact-integer support and embedding order;
- include all signed keys, including 4,081 empty partners in the exact q003 inventory;
- execute the production `matvec` factor-two path;
- compare direct, exchange, total, one-body, full action, and registered-vector quadratic form;
- require nonvacuous output in every applicable tested orbit;
- keep orientation-implementation, complete-basis, whole-operator, exact-unitary-curvature, eigensolver, stability, production, and paper authority false.

The full q003 run remains blocked until deterministic chunking/checkpointing, exact two-group aggregation without postselection, and an artifact-only certifier are implemented and reviewed. Runtime hashes from a monolithic process are not sufficient authority.
