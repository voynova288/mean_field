# RLG/hBN Track-P role-resolved scalar-extension audit (2026-08-01)

## Scope and authority

This report records a **diagnostic failure and contract correction** for the shell-5 Track-P finite-q response. It does not publish a new physical spectrum, repair C3, authorize exact-M stability labels, or change any Hartree/Fock/TDHF numerical formula.

Track P remains the finite-regulator hypothesis

```text
fixed-G single-representative active interaction
+ C3-repaired frozen-remote h0
```

The validated production object is now scoped as a **role-resolved projected signed-q regulator A/B matrix pair**. A role-independent global dense stored-density scalar extension is not established.

## 1. Problematic C3-orbit five-point run: job 198456

Immutable runroot:

```text
/data/home/ziyuzhu/.runs/Mean_Field_6703203_rlg_hbn_track_p_c3_scalar_fd_v2_20260801
```

Shell-5 task `198456_0` ran on controller job `198457` and failed its preregistered real-curvature gate after evaluating all three orbit members. Shell-6 task `198456_1` was cancelled while still pending after the shell-5 discriminator had failed; no shell-6 scientific result was produced.

| q | min Hstat eigenvalue (meV) | `2 v†Hstat v/Nk` (meV/cell) | all-ph five-point curvature (meV/cell) | residual (meV/cell) |
|---|---:|---:|---:|---:|
| `(-5,-4)` | `+0.08655741390994619` | `+0.00120218630430139` | `+0.00187068773508751` | `+0.00066850143078612` |
| `(4,-1)` | `-0.09512005872743334` | `-0.001321111926766532` | `-0.0015012585663498614` | `-0.0001801466395833294` |
| `(1,5)` | `-0.07405750340989878` | `-0.0010285764362430573` | `-0.001633838640676307` | `-0.0006052622044332497` |

The FD plateaus were stable, so these were not truncation-noise failures. The run, however, sent each complete finite-q density sector through one global `role="ph"` response. That convention is not the production ph/hp column convention.

Evidence:

```text
results/.../shell5/.problematic_c3_unitary_projector_curvature_v1_20260801_staging_198456_0_198457/
  track_p_shell5_problematic_c3_unitary_curvature_summary.json
SHA256 4c09947ef4ef626bde3b8c037d752d9ef37c768a30b6e4c22a2366a7ec41b9d0
```

## 2. Analytic split: job 198509

The analytic, no-FD diagnostic separated the orbital connection/A0 term from the interaction response for `q=(-5,-4)`.

Passed at numerical precision:

```text
D1 explicit x/y reconstruction                      0
D1 norm residual                                    0
K anti-Hermitian residual                           0
C_conn - C_A0                         1.11e-16 meV/cell
C_A0 formula - Hstat A0               5.55e-17 meV/cell
saved-projector - orbital-projector total           -3.31e-10 meV/cell
```

Failed:

```text
C_resp(all-ph) - C_AB                  +0.0006684905438812949 meV/cell
```

Thus the discrepancy was not the one-body gap, generator normalization, finite-SCF projector mismatch, or five-point stencil. It was localized to the response interpretation.

The job intentionally retained failed staging and did not publish canonical output:

```text
results/.../shell5/.analytic_c3_term_split_v1_20260801_staging_198509/
summary SHA256 0c0682bc9fd46b3980f68a70d81097210877d3f3ad9c2eee48a0d52137c64751
arrays  SHA256 0cbda9691fa7d9a4f3715041bc88bdb235dacc5eb8ef702804c56e1352ae0778
```

## 3. Role-resolved projected-action gate: job 198516

The next gate used the four production role components:

```text
 q/ph:   D[h(k),p(k+q)]     = x
 q/hp:   D[p(k-q),h(k)]     = y
-q/ph:   D[h(k),p(k-q)]     = conj(y)
-q/hp:   D[p(k+q),h(k)]     = conj(x)
```

The hp API blocks were explicitly repacked on the source/hole base-k fiber. Dense responses were manually projected onto all 432 output particle-hole rows and compared with both the projected response API and the saved production matrices.

All four production actions passed:

| component | saved action | max all-row residual (meV) |
|---|---|---:|
| `q/ph` | `(A-A0)x` | `7.10567214590617e-15` |
| `q/hp` | `B y` | `2.167779754565684e-15` |
| `-q/ph` | `(A_minus-A0_minus) conj(y)` | `1.1565518892014218e-14` |
| `-q/hp` | `B_minus conj(x)` | `1.790180836524724e-15` |

The reduced coefficient-space interaction action also has the required factor-of-two relation:

```text
v† H_int v / Nk = -0.21097695070272293 meV/cell
2 v† H_int v / Nk = C_AB
```

Therefore **a production A/B column-assembly bug is not supported**.

The stronger dense scalar interpretation failed:

```text
C_resp(role-split dense) - C_AB
  = +0.0005093923372181397
    -0.00010778031848469339 i  meV/cell

q/-q dense contraction conjugacy: fail
pair-sum reality: fail
```

The all-ph dense extension remained wrong by `0.0006684905438812949 meV/cell`. The current role-conditioned dense outputs are therefore not certified as restrictions of one role-independent, self-adjoint global stored-density derivative under the current bilinear pairing.

This does **not** prove that no global extension can exist. It proves that the current role-conditioned map plus current dense pairing is not one.

Evidence:

```text
results/.../shell5/.role_resolved_tangent_gate_v1_20260801_staging_198516/
summary SHA256 dfc82c0fa5aa37075319eada9a5ce24c0f085e5b3c050e913fe0a4469899ad51
arrays  SHA256 8b48fbe4de2b5d9d29d58a68652dd9b11ae649db58d7b02b94f223334927e570
```

## 4. Contract correction

No numerical kernel, A/B entry, q/-q pair, or eigensystem was modified. The finite-q APIs were narrowed to their demonstrated authority:

```text
track_p_role_resolved_dense_column_action_v2
track_p_role_resolved_projected_ab_column_action_v2
track_p_role_resolved_projected_signed_q_regulator_v2
```

New provenance states:

```text
role_dependent_kernel = true
cross_role_dense_recomposition_authorized = false
global_dense_scalar_extension = "not_established"
pairing_adjointness_scope = "assembled_signed_ab_pair"
```

The q=0 stored-density functional remains separately scoped and is not downgraded by this finite-q result.

Validation on `test001`:

```text
77 focused tests passed
```

A broader 93-test RLG/hBN set had only two known, unrelated fixed-quotient-anchor failures (`fixed_touched=6.401 meV`, fixed Wilson identity defect `1.397`); there were no new response/provenance failures.

## 5. Consequences

1. Shell-4/5/6 parent-cutoff results remain valid as **projected signed-q regulator A/B calculations**, not as arbitrary dense finite-q scalar Hessians.
2. The unchanged 35/144 classification pattern and nonmonotonic shell energy drift do not become exact C3 evidence.
3. The failed all-ph five-point trajectories cannot repair or invalidate production A/B spectra; they tested a different, unauthorized dense extension.
4. Track C remains fail-closed. A true global extension requires an explicitly defined scalar functional, role-independent derivative or derived dual pairing, and fresh q/-q/self-adjointness/A/B validation.
5. The following numerical “repairs” remain forbidden: real-part projection, q/-q averaging, Hermitization, A/B symmetrization, factor adjustment, residual subtraction, C3 orbit copying, or post-fit shifts/scales.

## Final status

```text
Production projected A/B implementation: no bug found in the tested four-component/all-row gate.
Current role-dependent global dense scalar extension: not established; tested constructions fail.
Parent-cutoff restoration of exact C3: not observed.
Track C provider: blocked/fail-closed pending a new UV/scalar convention.
```
