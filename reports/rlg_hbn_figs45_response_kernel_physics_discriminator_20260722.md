# RLG/hBN Fig. S45 response-kernel physics discriminator (2026-07-22)

## Question

The accepted typed-quotient calculation is C3-covariant but its intervalley and interspin energies are systematically above the published Fig. S45 values. The central question is whether this comes from the stationary HF Koopman spectrum or from the finite-q residual interaction/Hessian.

All comparisons below preserve the same accepted stationary HF source `185536`. A conventional single-representative exchange kernel is evaluated only as an isolated diagnostic by explicitly stripping typed provenance in memory. It is not promoted as a consistent physical result.

## Current C3 status

The previously reported visible C3 issue belongs to obsolete chains:

- legacy source block-spectrum defect: up to `8.8759 meV`;
- first wrapped generic-q matrix defect: Liouvillian spectrum `0.172 meV`.

The accepted microscopic typed-quotient chain is at roundoff:

```text
intraflavor generic L C3 <= 1.27e-11 meV
intervalley generic L C3 <= 2.13e-11 meV
interspin generic L C3   <= 1.41e-11 meV
```

The alternative conventional same-source kernel has a small but nonzero generic C3 defect:

```text
full intervalley A-spectrum residual = 0.013011 meV
lowest intervalley branch spread     = 0.000504 meV
```

Thus the user's visual assessment is correct: the conventional result is nearly C3-symmetric at the lowest branch, but not microscopically exact.

## 26-representative intervalley discriminator

Job `192651` evaluated all 26 C3+inversion representatives with the accepted HF orbitals and compared:

1. the accepted microscopic quotient Hessian;
2. the older conventional single-representative exchange Hessian on the same source;
3. the digitized published raster.

No energy shift or scale was fitted.

| quantity | typed quotient | conventional same source |
|---|---:|---:|
| RMSE from paper over 26 representatives | `0.70207 meV` | `0.07769 meV` |
| mean calculation minus paper | `+0.69362 meV` | `+0.04499 meV` |
| conventional minus typed branch shift | - | mean `-0.64863 meV` |
| maximum A-matrix element difference | - | mean `0.89313 meV` |

Representative values:

| q | typed (meV) | conventional (meV) | paper raster (meV) |
|---|---:|---:|---:|
| `(0,0)` | 2.72346 | 2.18661 | 2.22614 |
| `(1,0)` | 3.06268 | 2.29909 | 2.32975 |
| `(-2,-2)` | 3.40595 | 2.66023 | 2.63514 |
| `(-4,-4)` | 5.03601 | 4.33350 | 4.27662 |
| `(-6,-6)` | 7.17627 | 6.62000 | 6.67612 |

This is not a random numerical coincidence confined to Gamma. The conventional kernel follows the paper throughout the sampled mBZ.

## Koopman versus binding decomposition

The intervalley channel has exactly `Y=0`, as stated by the paper. The mode energy can therefore be decomposed without a new eigensolve as

```text
Omega = <X|A0|X> + <X|Ainteraction|X>.
```

Across the 26 representatives:

| mode | mean Koopman A0 (meV) | mean residual binding (meV) |
|---|---:|---:|
| typed quotient | 55.26833 | -49.67346 |
| conventional same source | 55.16707 | -50.22083 |

The observed 2--8 meV collective energy is a cancellation between an approximately 55 meV Koopman scale and approximately -50 meV excitonic binding. A sub-meV change in the exchange Hessian therefore produces the full paper discrepancy.

This revises the earlier source-only interpretation. The several-meV legacy-to-v3 local `h0` spectral change is real, but much of it is common-mode and cancels in neutral particle-hole gaps. The strongest direct evidence for the intervalley error is now the response kernel, not the absolute source `h0` spectrum.

## Fixed-node wavefunction evidence

The paper states that the lowest intervalley q=0 wavefunction is nearly uniform over the mBZ. The diagnostic gives:

```text
q=0 conventional mode:
  IPR * Nk                  = 1.00605  (uniform = 1)
  fixed-endpoint weight     = 0.01163
  minimum k weight          = 0.00561

q=0 typed quotient mode:
  IPR * Nk                  = 1.02106
  fixed-endpoint weight     = 2.8e-29
  minimum k weight          = 1.2e-29
```

The typed quotient mode has exact nodes at the two nonzero C3-fixed torus sectors. At generic `(1,0)`, its fixed-endpoint weight remains only `0.00681`, compared with `0.02348` for the conventional mode.

This gives a physical interpretation, with an important mode-tracking caveat. The branch-quotient response suppresses fixed-endpoint participation and excludes the conventional nearly uniform low-energy bound state. This is **mode reordering**, not a perturbative upward shift of the same eigenvector: at q=0 the conventional-minus-typed lowest-eigenvalue difference is `-0.53684 meV`, whereas the same matrix difference evaluated on the typed lowest vector is only `+0.000605 meV` (`+0.000491 meV` at q=(1,0)). The conventional kernel restores a different, nearly uniform low mode that is both paper-close in energy and consistent with Fig. S44. A future gate must therefore track overlaps/principal angles of the lowest several states, not compare only sorted eigenvalues.

## Why the conventional hybrid is not yet a result

The accepted HF source is stationary for the typed variational quotient, not for the conventional kernel. Job `192659` confirms the mismatch in the exact SU(2) interspin test:

```text
typed quotient Goldstone energy        2.14e-7 meV
conventional same-source interspin gap 1.591e-3 meV
accepted source stationarity residual  3.43e-5 meV
```

The conventional hybrid violates the source/Hessian Ward identity by much more than the accepted SCF residual. Its excellent Fig. S45 intervalley agreement cannot therefore be promoted directly.

## Physical conclusion

The likely conceptual error was treating the finite-cutoff fixed-point periodic-gauge ambiguity as an exact physical quotient and requiring exact microscopic C3 before first establishing that this quotient is the Hamiltonian used in the paper. At finite 19-RLV cutoff, quotient averaging is not a passive gauge relabel: it changes the response subspace and removes physical fixed-node weight.

The evidence now favors a paper implementation closer to a single-representative finite-cutoff projected Hamiltonian, which retains the uniform intervalley mode and accepts a small regulator-level C3 defect (or copies symmetry-related sectors only at the calculation/plot level).

This is not yet uniquely proven because the paper does not specify the fixed-node prescription.

## Next required derivation

Before another full-mesh production run:

1. define a single-representative finite-cutoff HF energy functional, including fixed nodes, without assembled A/B symmetrization;
2. derive both its HF gradient and TDHF Hessian from the same functional;
3. obtain a fresh stationary source;
4. require q=0 interspin Goldstone/Ward consistency;
5. quantify, rather than hide, its residual C3 defect;
6. compare the q=0 Fig. S44 intervalley wavefunction and a small set of Fig. S45 q points;
7. only then run the full 12x12 mesh.

## Matching-functional implementation status

The candidate has now been implemented as an explicitly typed hybrid rather than mislabeled as a wholly single-representative theory:

```text
active interaction: fixed_g_torus_single_representative_v1
physical G policy: fixed_abs_g_shell_v1       (13 vectors)
frozen remote h0: actual_node_ws_c3_fixed_copy_average_v1
```

The SCF energy/ODA and q=0 response use the same active linear map. The finite-q API independently assembles q and -q, saves both pair-label sets, and enforces `B(q)=B(-q)^T` without post-assembly repair. Focused tests include scalar-energy finite differences, pairing self-adjointness, fail-closed remote hash/shell provenance, nonzero-q signed intraflavor B, and public dispatcher routing.

An immutable 135-file source/archive/cache closure was generated by job `192733`:

```text
overall SHA256 3699594b60a3ea7d01c53b3d2c6d5c62e43be7c3cccfd883822be95da1743b92
remote h0 hash  bd0cb9e54f80899ea4e7b74bc3e1e65a782aa9bb0b79f85d60107d859b022dfd
h0 decomposition residual 1.42e-14 meV
```

Exactly one fresh matching HF job, `192734`, converged in 94 iterations. Slurm marked the wrapper `FAILED` only after the archive had been safely written, because the reporting code called `rlg_hbn_gap_estimate` with the obsolete signature. Follow-up job `193021` independently reloaded and certified the saved source:

```text
final SCF residual          9.8384e-8
energy                     -547.0919427055 meV
gap                         14.5200275809 meV
saved-H closure             2.5121e-15 meV
projector commutator        5.7194e-6 meV
projector idempotency       1.3323e-15
```

Job `193030` then passed the matching q=0 gates:

```text
interspin Goldstone         7.8895e-10 meV
spin Ward residual          4.5925e-6 meV
spin-generator overlap      0.9999999999999828
intervalley energy          2.1669489781 meV
intervalley fixed weight    0.0116299120
intervalley IPR*Nk          1.0059993536
intervalley max |Im Ω|      8.41e-14 meV
```

Thus the matching source retains the paper-local nearly uniform Fig. S44 mode and the nonzero fixed-endpoint weight while restoring SU(2) Ward consistency.

The independent generic-q C3 gate did **not** pass. Job `193042` assembled `(1,0)`, `(0,1)`, and `(-1,-1)` independently for all three channels with signed q/-q residuals below `4.72e-12 meV` and no complex modes. The three-anchor paper-raster RMSEs are already small (`0.0572`, `0.0490`, `0.00374 meV` for intraflavor/intervalley/interspin), but the unaveraged finite-cutoff C3 defects are:

```text
source canonical bands max/mean C3   1.1967 / 0.5167 meV
full A-spectrum C3                    1.7123 / 1.3655 / 1.4577 meV
interaction-only A-spectrum C3        0.0937 / 0.0428 / 0.0681 meV
lowest-branch C3 spread               0.0290 / 0.0133 / 0.00275 meV
```

The failure is dominated by the fresh source/Koopman spectrum, not by signed-q assembly. No matrix/orbit averaging was applied. Therefore this source is accepted for the q=0 functional/Ward/wavefunction checkpoint but **fails the required generic C3 production gate**; full-mesh TDHF remains blocked pending the author's symmetry-replication prescription or a separately derived full-space C3-invariant scalar functional that does not reintroduce fixed-node quotient suppression.

## Evidence

```text
results/RnG_hBN/tdhf_m2_pilot/figs45_c3_audit_v2_20260627/
  ACTUAL_INTERVALLEY_RESPONSE_KERNEL_CONVENTION_DISCRIMINATOR_v1_185536_20260722.json
  ACTUAL_CONVENTIONAL_KERNEL_GOLDSTONE_MISMATCH_v1_185536_20260722.json

jobs:
  192651 COMPLETED 00:32:11 MaxRSS 16198564K
  192659 COMPLETED 00:01:09 MaxRSS 5861948K
```
