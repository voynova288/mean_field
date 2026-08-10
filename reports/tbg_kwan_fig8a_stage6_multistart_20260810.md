# TBG Kwan Fig. 8(a) Stage6 four-start HF diagnostic — 2026-08-10

## Scope

This is a **companion-source-faithful, translation-preserving, spin-identical four-start HF diagnostic** on the fixed 10×10 mesh. It tests whether the existing Eq. (99) K-IVC baseline is lower than four specified homogeneous rank-2 starts and whether those starts reach the same SCF basin.

It is not a global HF search, thermodynamic-limit result, production HF/TDHF source promotion, or new Fig. 8(a) spectrum calculation. No TDHF matrix was recomputed.

Source commit: `1683426959a009a2e31631b786199ecc13b6249d`.

## Fixed starts

Canonical basis: `(K_Zplus, K_Zminus, Kprime_Zplus, Kprime_Zminus)`. Each start uses

\[
P_{\rm can}=\frac{I+Q}{2},\qquad
P_{\rm conv}(k)=W(k)P_{\rm can}W(k)^\dagger,
\qquad P_{\rm stored}=P_{\rm conv}^T,
\]

with identical spin copies and the exact hash-bound Stage5 frame `W`.

1. `vp_K`: `Q = tau_z tensor I`.
2. `zplus_both_valleys`: `Q = I tensor sigma_z`.
3. `gammaC_plus_control`: `Q = tau_z tensor sigma_z`.
4. `sameZ_tau_x_ivc`: `Q = tau_x tensor I`; this is not Eq. (99) K-IVC.

All starts passed exact rank-2 projector, stored-orientation, spin-duplication, occupation, pairwise-distinctness, and Eq. (99)-orbit-orthogonality gates.

## Jobs and sealing

- Scientific runner/postflight job: `212127`, node `node049`, `regular128`, full 64-CPU node.
- Runner step `212127.1`: `COMPLETED`.
- Independent postflight step `212127.2`: `COMPLETED`.
- Batch state: `FAILED`, solely because the final pure-stdlib sealer used `zip(..., strict=True)` under compute-node Python 3.9.
- Zero-science seal recovery job: `212377`, node `node026`, `regular6430`.
- The sentinel was published while the sealed controller snapshot still recorded `RUNNING`. A separate post-publication `sacct` observation recorded `COMPLETED 0:0` in 11 seconds; no terminal `sacct_212377` record was added to the sealed capsule.
- Recovery scientific updates/eigensolvers: `0/0`.

The recovered sentinel preserves the original failed batch state and separately binds the completed scientific and postflight steps. It does not rewrite job `212127` as completed.

## Results

Eq. (99) baseline finite-system energy: `-3.790312076629597 eV`; gap: `24.9595827 meV`.

| Start | Iterations | Energy difference from baseline | Gap | Endpoint |
|---|---:|---:|---:|---|
| `vp_K` | 22 | `+0.4238945 meV/cell` | `8.67156 meV` | distinct higher-energy non-IVC basin |
| `zplus_both_valleys` | 22 | `+3.2407494 meV/cell` | `20.79286 meV` | distinct higher-energy non-IVC basin |
| `gammaC_plus_control` | 22 | `+3.2406997 meV/cell` | `20.79286 meV` | distinct higher-energy non-IVC basin |
| `sameZ_tau_x_ivc` | 483 | `4.03e-12 meV/cell` | `24.95955 meV` | phase-aligned baseline-equivalent |

All four endpoints passed convergence, closure, filling, positive-gap, Hermiticity, Aufbau reconstruction, eigensolver, and eigenvector-orthonormality gates.

For `sameZ_tau_x_ivc`, the phase-aligned projector Frobenius norm divided by `Nk` is `4.91e-7` under the preregistered `1e-6` threshold; the Hamiltonian residual is `7.87e-8 eV`, and the gap difference is `3.18e-8 eV`. Its strict standalone `kivc_classification_passed` remains false because residual valley polarization `1.31e-5` exceeds the `1e-8` classifier tolerance. Baseline equivalence and strict classifier status are therefore both retained rather than conflated.

## Verdict

- **Four-start sampled energetic robustness: PASS.** No valid converged endpoint is lower than the Eq. (99) baseline by more than the preregistered `1e-6 eV/cell` (`0.001 meV/cell`) tolerance.
- **Sampled basin robustness: FAIL.** Only one of four starts reaches a phase-aligned baseline-equivalent endpoint; three tested translation-preserving, spin-identical homogeneous starts remain in higher-energy invariant basins.
- **Global ground state: not proved.** The sample excludes generic random starts, spin-textured or momentum-textured projectors, IKS/finite-q order, supercells, translation breaking, and mesh/cutoff extrapolation.

Thus the appropriate wording is **“four-start sampled energetic robustness, not basin robustness.”** This strengthens the HF prerequisite behind the existing Kwan Fig. 8(a) diagnostic but does not authorize a stronger production or global-minimum claim.

## Artifacts

Original runroot:

`/data/home/ziyuzhu/.runs/Mean_Field_1683426_tbg_kwan_stage6_hf_multistart_v3_20260809`

- Aggregate summary SHA256: `857dc84cf3c1417d53c02e8b60721074dc3cca70dfec8eaf6db7eb4e9b049bba`.
- Independent postflight SHA256: `91a23ed42eb0eaf146e945ac3b80ec1c6dba7e25afbedca81f83141180c74206`.
- Recovered completion/sentinel SHA256: `470699555112cb0aa3cd8386cf63373daf43a25821fca91ffe61b1c2a3f1e179`.
- Completion and sentinel: same device/inode, mode `0444`, link count `2`.

Recovery runroot:

`/data/home/ziyuzhu/.runs/Mean_Field_1683426_tbg_kwan_stage6_hf_multistart_v3_seal_recovery_v2_20260810`

- Recovery summary SHA256: `d7ea7cf11405b2a19cfc45fa51bdb6350f0ce19a4e81e4baecd941c47ff40a0f`.
- Recovery approval SHA256: `68e2996d3d87e6a9edc2db9f964d2a682691091d620a25b8f4d2d386b998fb16`.
