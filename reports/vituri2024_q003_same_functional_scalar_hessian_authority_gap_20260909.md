# Vituri 2024 q003 same-functional scalar-Hessian authority gap

Date: 2026-09-09

## Verdict

The current q003 evidence is **not yet sufficient** to promote universal same-functional scalar-Hessian authority.

What is now established is narrower:

1. job `477106` produced the two no-postselection density groups and fresh-map stationary endpoints;
2. jobs `487653` and `487778` bound the same q003 source components, reconstructed the saved fresh Fock as
   \[
   F_0=h_0+\Sigma[P_0-R],
   \]
   and checked ten fixed scalar/action directions;
3. job `496930` closed the missing six-step exact-unitary ladder for those ten directions;
4. the source-bound certificate at commit `80ab602` establishes whole-inventory algebraic reciprocity of the production paired action.

Reciprocity proves that the production real action is symmetric. It does not by itself prove that this action is the second derivative of the exact-integer scalar functional for every tangent.

## New step-ladder evidence

Tracked attestation:

`reports/data/vituri2024_fig2_q003_exact_unitary_step_ladder_496930_attestation.json`

Job `496930` evaluated both density groups, all five preregistered sectors per group, and all six steps

```text
0.006, 0.004, 0.003, 0.002, 0.0015, 0.001
```

without postselection. The independent postrun checks found:

- maximum source-Fock closure residual: `4.440901974686905e-16 eV`;
- maximum selected-source Fock off-diagonal/diagonal-imaginary residual in the transition basis: `6.5478492086438755e-18 eV` (`< 1e-12 eV`);
- maximum analytic/exact-unitary curvature residual: `8.602844832339418e-09 eV` (`< 5e-08 eV`);
- maximum fine-ladder adjacent difference: `6.986184256096983e-09 eV` (`< 1e-08 eV`);
- maximum locked roundoff term `r2`: `5.6843418860808015e-08 eV` (`< 1e-07 eV`);
- minimum curvature-signal/roundoff ratio: `71251.59906224259` (`> 16`);
- all 60 `t=0` anchors were exactly zero;
- all 300 exact-unitary receipts passed unitary/projector/Hermiticity/trace gates;
- 167 focused tests passed.

This closes **fixed-probe step-ladder readiness only**. It does not convert ten directions into a universal operator identity.

## The operator identity that must still be certified

For a fixed-rank unitary path

\[
P(t)=e^{tK}P_0e^{-tK},\qquad W=[K,P_0],
\]

and a linear HF interaction map `Sigma`, the exact-integer relative functional used by the unitary oracle is

\[
\Delta E(t)=\operatorname{Tr}(F_0\Delta P(t))
+\frac12\operatorname{Tr}(\Delta P(t)\,\Sigma[\Delta P(t)]),
\]

where `Delta P(t)=P(t)-P0`. If interaction trace reciprocity holds, then

\[
E''(0)=\operatorname{Tr}(F_0[K,[K,P_0]])
+\operatorname{Tr}(W\Sigma[W]).
\]

For a general stationary source, occupied/virtual block diagonality reduces the first term only to

\[
2\operatorname{Tr}(X^\dagger F_vX-X^\dagger X F_o).
\]

The simpler expression implemented by the production gap action,

\[
2\sum_{ph}(F_p-F_h)|X_{ph}|^2,
\]

requires the registered particle/hole orbitals to diagonalize the occupied and virtual Fock blocks. The saved production source satisfies an exact valley-diagonal gate. The independently reconstructed source Fock satisfies only a tolerance-bounded transition-basis gate (`6.5478492086438755e-18 eV < 1e-12 eV`), and its diagonal entries agree with the production one-body action only to the recorded fixed-probe residuals (maximum `2.7755575615628914e-17 eV`). A universal certificate must therefore bind this basis identity explicitly and state its tolerance; it cannot infer the gap formula from stationarity alone.

The production paired Hessian will equal this scalar Hessian on the full tangent space only if all of the following are source-bound:

1. **source identity:** the two `P0` arrays, saved fresh `F0`, `h0`, authoritative normal-order reference `R`, active-band spinors, exact integer mesh labels, FFT plan, area, flavor order, and gauge;
2. **stationarity:** the fresh-map commutator/occupied-virtual block vanishes for each exact source, not merely for a separately reconstructed or aligned matrix;
3. **one-body operator identity:** on the complete selected-spin fixed-rank tangent domain, the production diagonal gap action `D_prod` equals the source-functional curvature action `D_F`; this includes the Fock-eigenbasis condition and its exact or tolerance-bounded status;
4. **interaction derivative identity:** for every supported signed block, the production response `L_prod` is exactly the same linear map `Sigma[W]` used by the scalar functional;
5. **transition geometry:** the production injection `J` implements `W=[K,P0]`, extraction is its adjoint `J†`, and lower-lane conjugation plus both `{d,-d}` sectors are retained;
6. **normalization:** the complete identity is
   \[
   H_{\rm prod}=2J^\dagger(D_{\rm prod}+L_{\rm prod})J
   =2J^\dagger(D_F+\Sigma)J=H_E,
   \]
   in raw total-energy units, with no extra interaction `1/2` and no `/Nk`;
7. **inventory completeness:** the structural identity covers the full q003 selected-spin fixed-rank tangent space: all `21,170` canonical orbits and all `15,052,040` complex (`30,104,080` real) dimensions for each source. It does not claim an unrestricted spin/global tangent space.

## Smallest non-postselected closure

An exhaustive `d^2` exact-unitary sweep is infeasible and is not necessary if an independently reviewed operator theorem is used. The minimal defensible closure is:

1. add a **candidate-only structural bridge** whose implementation fingerprint covers both independent formula implementations:
   - `vituri2024_hf_spiral_full_response.py`;
   - `vituri2024_hf_spiral_full_hessian.py`;
   - `zero_temperature_sector_stability.py`;
   - `vituri2024_tdhf_exact_integer_signed_scalar.py`;
2. independently derive both operator parts: `D_prod=D_F` in the registered Fock eigenbasis, and `L_prod=Sigma` from identical direct rank-one and exchange `-M† C_K M` factors, support, flavor blocks, area division, and no-wrap kernel orientation;
3. run a whole-inventory structural comparison over both q003 sources, with no action/eigensolver promotion and no postselection;
4. bind that comparison to the existing source-Fock closure, `477106` stationarity lineage, job `496930` step ladder, and the existing reciprocity certificate;
5. require a detached theorem review and a separate artifact-only certifier before setting a q003-specific same-functional scalar-Hessian flag.

The fixed probes and step ladder are regression/discriminator evidence for this theorem, not a substitute for it.

## Authority boundary after job 496930

Remain `true` only in their existing narrow scopes:

- exact two-source whole-inventory algebraic reciprocity;
- fixed-source exact-integer Fock closure;
- ten fixed-probe analytic/action parity;
- ten fixed-probe six-step exact-unitary curvature parity.

Remain `false`:

- universal same-functional scalar-Hessian authority;
- full-inventory exact-unitary curvature;
- literal-float full-functional parity;
- Hermitian `LinearOperator`/eigensolver authorization;
- full local stability;
- production readiness;
- paper reproduction.
